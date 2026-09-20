"""The published bundle.

The bundle is the thing production actually serves, so the properties under
test are the ones that make it checkable rather than merely present: that the
version is a function of the contents, that a modified bundle refuses to load,
and that the moment a board stops being a forecast is recorded rather than
inferred.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nflproj.serving.bundle import (
    MANIFEST_NAME,
    PROJECTIONS_NAME,
    BundleError,
    BundleMetadata,
    load_bundle,
    write_bundle,
)

KICKOFF = datetime(2026, 9, 24, 0, 15, tzinfo=UTC)


@pytest.fixture
def board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [3, 3, 3],
            "player_id": ["p1", "p2", "p3"],
            "player_display_name": ["A Back", "B Wideout", "C Passer"],
            "position": ["RB", "WR", "QB"],
            "team": ["ATL", "BAL", "CHI"],
            "opponent_team": ["BAL", "ATL", "GB"],
            "prediction": [18.5, 12.25, 21.0],
        }
    )


def _write(root: Path, board: pd.DataFrame, **kwargs: object) -> BundleMetadata:
    defaults: dict[str, object] = {
        "predictor": "season_decayed_hl3_d0.5",
        "season": 2026,
        "week": 3,
        "valid_from": KICKOFF,
        "raw_assets": {"player_stats/2026": "a" * 64},
        "metrics": {"mae": 4.398133},
        "baseline": "season_to_date_mean",
    }
    defaults.update(kwargs)
    return write_bundle(root, board, **defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ versioning


def test_the_same_contents_produce_the_same_version(tmp_path, board):
    """Otherwise "is production serving what I think it is" is unanswerable."""
    a = _write(tmp_path / "a", board)
    b = _write(tmp_path / "b", board.copy())
    assert a.version == b.version


def test_a_rebuild_at_a_different_time_is_the_same_version(tmp_path, board):
    """`created_at` is excluded from the hash: a rebuild is not a new model."""
    a = _write(tmp_path / "a", board, now=datetime(2026, 9, 22, 12, tzinfo=UTC))
    b = _write(tmp_path / "b", board, now=datetime(2026, 9, 23, 6, tzinfo=UTC))
    assert a.created_at != b.created_at
    assert a.version == b.version


def test_changing_a_single_projection_changes_the_version(tmp_path, board):
    moved = board.copy()
    moved.loc[0, "prediction"] = 18.51
    assert _write(tmp_path / "a", board).version != _write(tmp_path / "b", moved).version


def test_row_order_does_not_change_the_version(tmp_path, board):
    """A sort is not a new model. The hash canonicalises order first."""
    shuffled = board.iloc[::-1].reset_index(drop=True)
    assert _write(tmp_path / "a", board).version == _write(tmp_path / "b", shuffled).version


def test_changing_the_predictor_changes_the_version(tmp_path, board):
    """The version commits to the claims, not only to the numbers."""
    a = _write(tmp_path / "a", board)
    b = _write(tmp_path / "b", board, predictor="ewma_hl3")
    assert a.version != b.version


def test_changing_an_upstream_hash_changes_the_version(tmp_path, board):
    """Same numbers from different source files is a different artifact."""
    a = _write(tmp_path / "a", board)
    b = _write(tmp_path / "b", board, raw_assets={"player_stats/2026": "b" * 64})
    assert a.version != b.version


# ---------------------------------------------------------------- write guards


def test_an_empty_board_is_refused(tmp_path, board):
    with pytest.raises(BundleError, match="empty board"):
        _write(tmp_path / "a", board.iloc[:0])


def test_missing_columns_fail_at_publish_not_in_production(tmp_path, board):
    with pytest.raises(BundleError, match="missing required columns"):
        _write(tmp_path / "a", board.drop(columns=["position"]))


def test_a_naive_valid_from_is_refused(tmp_path, board):
    with pytest.raises(BundleError, match="timezone-aware"):
        _write(tmp_path / "a", board, valid_from=datetime(2026, 9, 24, 0, 15))  # noqa: DTZ001


# ----------------------------------------------------------------- round trip


def test_a_bundle_round_trips(tmp_path, board):
    written = _write(tmp_path / "b", board)
    loaded = load_bundle(tmp_path / "b")
    assert loaded.metadata == written
    assert len(loaded.projections) == len(board)
    assert loaded.metadata.metrics["mae"] == pytest.approx(4.398133)


def test_an_incomplete_bundle_says_which_file_is_missing(tmp_path, board):
    _write(tmp_path / "b", board)
    (tmp_path / "b" / PROJECTIONS_NAME).unlink()
    with pytest.raises(BundleError, match=PROJECTIONS_NAME):
        load_bundle(tmp_path / "b")


def test_a_missing_bundle_directory_is_an_error(tmp_path):
    with pytest.raises(BundleError, match="incomplete"):
        load_bundle(tmp_path / "nothing-here")


def test_an_unreadable_manifest_is_an_error(tmp_path, board):
    _write(tmp_path / "b", board)
    (tmp_path / "b" / MANIFEST_NAME).write_text("{ truncated")
    with pytest.raises(BundleError, match="not readable"):
        load_bundle(tmp_path / "b")


# ------------------------------------------------------------------- integrity


def test_a_tampered_board_refuses_to_load(tmp_path, board):
    """The point of the hash. Edited numbers must not be servable.

    Without this check the service would answer with projections nobody
    generated, and every provenance field in the response would be a lie that
    looks exactly like the truth.
    """
    _write(tmp_path / "b", board)
    edited = board.copy()
    edited.loc[0, "prediction"] = 99.0
    edited.to_parquet(tmp_path / "b" / PROJECTIONS_NAME, index=False)

    with pytest.raises(BundleError, match="integrity check failed"):
        load_bundle(tmp_path / "b")


def test_a_tampered_manifest_refuses_to_load(tmp_path, board):
    """Relabelling which week a board covers is caught the same way."""
    _write(tmp_path / "b", board)
    path = tmp_path / "b" / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["week"] = 4
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleError, match="integrity check failed"):
        load_bundle(tmp_path / "b")


def test_a_truncated_board_refuses_to_load(tmp_path, board):
    """Dropping rows is the failure mode of a half-copied volume."""
    _write(tmp_path / "b", board)
    board.iloc[:1].to_parquet(tmp_path / "b" / PROJECTIONS_NAME, index=False)
    with pytest.raises(BundleError, match="integrity check failed"):
        load_bundle(tmp_path / "b")


def test_the_error_names_both_hashes(tmp_path, board):
    """An operator should not need the source to tell what went wrong."""
    written = _write(tmp_path / "b", board)
    edited = board.copy()
    edited.loc[0, "prediction"] = 99.0
    edited.to_parquet(tmp_path / "b" / PROJECTIONS_NAME, index=False)

    with pytest.raises(BundleError) as exc:
        load_bundle(tmp_path / "b")
    assert written.version in str(exc.value)


# ------------------------------------------------------------------ supersession


def test_a_board_is_not_superseded_before_kickoff(tmp_path, board):
    meta = _write(tmp_path / "b", board)
    assert meta.superseded(now=KICKOFF - timedelta(minutes=1)) is False


def test_a_board_is_superseded_from_kickoff_onward(tmp_path, board):
    """At kickoff, not after it: the first snap is when a forecast becomes a record."""
    meta = _write(tmp_path / "b", board)
    assert meta.superseded(now=KICKOFF) is True
    assert meta.superseded(now=KICKOFF + timedelta(hours=3)) is True


def test_an_unknown_kickoff_is_never_claimed_to_be_superseded(tmp_path, board):
    """No kickoff time means we do not know, and inventing one would be worse."""
    meta = _write(tmp_path / "b", board, valid_from=None)
    assert meta.valid_from_dt is None
    assert meta.superseded(now=datetime(2030, 1, 1, tzinfo=UTC)) is False
