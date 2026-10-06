"""Availability: sourcing discipline and the adjustment itself.

The defect these exist for, measured on the 2026 week-5 board: 19 Out-or-
Doubtful players ranked, five inside the top 100, Justin Jefferson at 58 on a
12.3-point projection while listed Out.

The trap next to the fix is worse than the defect. Upstream rewrites the injury
file in place as a week progresses, so its week-W rows read after week W
already encode who played. Using them to "project" week W would be near-perfect
leakage, because Out does not merely correlate with scoring zero - it partly is
the outcome. Most of what follows tests that the wrong source cannot be reached
rather than merely that it is not reached.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nflproj.config import Settings
from nflproj.features.injuries import (
    AVAILABILITY_FACTORS,
    PRACTICE_COLUMN,
    STATUS_COLUMN,
    InjurySource,
    attach_status,
    availability_factor,
    availability_factors,
    load_status_before,
    status_summary,
)
from nflproj.ingest.archive import INJURIES, Archive
from nflproj.ingest.nflverse import IngestError

#: Week 4 opens Thursday 2026-09-24 20:15 ET -> 2026-09-25 00:15Z.
WEEK4_KICKOFF = datetime(2026, 9, 25, 0, 15, tzinfo=UTC)
BEFORE = WEEK4_KICKOFF - timedelta(hours=6)
AFTER = WEEK4_KICKOFF + timedelta(days=3)


@pytest.fixture
def calendar() -> pd.DataFrame:
    rows = [
        (2026, 3, "2026-09-17 20:15"),
        (2026, 4, "2026-09-24 20:15"),
        (2026, 4, "2026-09-27 13:00"),
        (2026, 5, "2026-10-01 20:15"),
    ]
    frame = pd.DataFrame(rows, columns=["season", "week", "gametime"])
    frame["kickoff_et"] = pd.to_datetime(frame["gametime"]).dt.tz_localize("America/New_York")
    return frame.drop(columns=["gametime"])


def _report(week: int, rows: list[tuple[str, str | None, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": week,
                "gsis_id": pid,
                "report_status": status,
                "practice_status": practice,
            }
            for pid, status, practice in rows
        ]
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        archive_dir=tmp_path / "archive",
    )


@pytest.fixture
def live_feed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Stands in for the upstream file, and records whether it was consulted."""
    state: dict[str, object] = {"calls": 0}

    def _install(frame: pd.DataFrame) -> None:
        path = tmp_path / "live_injuries.parquet"
        frame.to_parquet(path)

        def _fetch(*_a: object, **_k: object) -> Path:
            state["calls"] = int(state["calls"]) + 1  # type: ignore[call-overload]
            return path

        monkeypatch.setattr("nflproj.features.injuries.fetch_asset", _fetch)

    state["install"] = _install
    return state


# --------------------------------------------------------------- factors


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Out", 0.0),
        ("out", 0.0),
        ("  OUT  ", 0.0),
        ("IR", 0.0),
        ("Doubtful", 0.2),
        ("Questionable", 0.9),
        (None, 1.0),
        ("", 1.0),
        (float("nan"), 1.0),
    ],
)
def test_the_factor_for_each_designation(status, expected):
    assert availability_factor(status) == pytest.approx(expected)


def test_an_unrecognised_designation_leaves_the_projection_alone():
    """Upstream can invent a status. Zeroing a player over an unknown string
    would be a worse failure than ignoring it."""
    assert availability_factor("Probably Fine Honestly") == 1.0


def test_out_is_exactly_zero_not_merely_small():
    """A non-zero projection for a player the league says will not play is a
    number we know to be wrong."""
    assert AVAILABILITY_FACTORS["out"] == 0.0


def test_factors_vectorise_and_keep_the_index():
    s = pd.Series(["Out", None, "Questionable"], index=[7, 8, 9])
    out = availability_factors(s)
    assert out.index.tolist() == [7, 8, 9]
    assert out.tolist() == [0.0, 1.0, 0.9]


# ------------------------------------------------------- source discipline


def test_a_week_not_yet_kicked_off_uses_the_live_feed(settings, live_feed):
    """Legitimate: this is what a publisher knows at publication time."""
    live_feed["install"](_report(4, [("p1", "Out", "Did Not Participate")]))
    status, source = load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=BEFORE)
    assert source is InjurySource.LIVE
    assert status.set_index("player_id").loc["p1", STATUS_COLUMN] == "Out"


def test_a_week_already_kicked_off_never_touches_the_live_feed(settings, live_feed):
    """The whole leakage argument, enforced rather than remembered.

    The live file has by now been overwritten with post-game state, where Out
    partly IS the outcome. The archive is empty here, so the honest answer is
    no designations - and crucially the live feed must not be consulted at all.
    """
    live_feed["install"](_report(4, [("p1", "Out", None)]))
    status, source = load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=AFTER)
    assert source is InjurySource.NONE
    assert status.empty
    assert live_feed["calls"] == 0, "the live feed was read for a week already played"


def test_allow_live_false_forces_archive_sourcing(settings, live_feed):
    """What a reproducible backtest asks for, even for a future week."""
    live_feed["install"](_report(4, [("p1", "Out", None)]))
    _, source = load_status_before(
        2026, 4, WEEK4_KICKOFF, settings=settings, allow_live=False, now=BEFORE
    )
    assert source is InjurySource.NONE
    assert live_feed["calls"] == 0


def test_a_week_with_no_report_published_yet_reports_none(settings, live_feed):
    """The normal Tuesday case: the week exists, its report does not.

    This must read as "no designations", not as "everybody healthy" - the
    distinction is what stops a Tuesday board silently claiming a clean slate.
    """
    live_feed["install"](_report(3, [("p1", "Out", None)]))  # week 3, not 4
    status, source = load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=BEFORE)
    assert source is InjurySource.NONE
    assert status.empty


def test_designations_are_read_from_the_archive_for_a_played_week(settings, live_feed, tmp_path):
    archive = Archive(settings.archive_dir, INJURIES)
    blob = tmp_path / "snap.parquet"
    _report(4, [("p1", "Doubtful", "Limited Participation")]).to_parquet(blob)
    body = blob.read_bytes()

    digest = hashlib.sha256(body).hexdigest()
    archive.blob_dir.mkdir(parents=True, exist_ok=True)
    archive.blob_path(digest).write_bytes(body)
    archive.root.mkdir(parents=True, exist_ok=True)
    with archive.manifest_path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "asset": "injuries",
                    "season": 2026,
                    "fetched_at": BEFORE.isoformat(),
                    "sha256": digest,
                    "rows": 1,
                    "source_url": "https://example.invalid/i.parquet",
                    "novel": True,
                }
            )
            + "\n"
        )

    live_feed["install"](_report(4, [("p1", "Out", None)]))
    status, source = load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=AFTER)

    assert source is InjurySource.ARCHIVE
    # Doubtful from the pre-kickoff snapshot, NOT Out from the overwritten file.
    assert status.set_index("player_id").loc["p1", STATUS_COLUMN] == "Doubtful"
    assert live_feed["calls"] == 0


def test_a_missing_season_upstream_is_not_fatal(settings, monkeypatch):
    def _boom(*_a: object, **_k: object) -> None:
        raise IngestError("404")

    monkeypatch.setattr("nflproj.features.injuries.fetch_asset", _boom)
    status, source = load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=BEFORE)
    assert source is InjurySource.NONE
    assert status.empty


def test_a_schema_change_upstream_is_loud(settings, monkeypatch, tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"season": [2026], "week": [4], "whomst": ["p1"]}).to_parquet(path)
    monkeypatch.setattr("nflproj.features.injuries.fetch_asset", lambda *_a, **_k: path)
    with pytest.raises(KeyError, match="gsis_id"):
        load_status_before(2026, 4, WEEK4_KICKOFF, settings=settings, now=BEFORE)


# ------------------------------------------------------------ attach_status


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [4, 4, 5, 5],
            "player_id": ["p1", "p2", "p1", "p2"],
            "fantasy_points": [0.0, 12.0, 0.0, 0.0],
            "kickoff_et": pd.to_datetime(
                [
                    "2026-09-27 13:00",
                    "2026-09-27 13:00",
                    "2026-10-04 13:00",
                    "2026-10-04 13:00",
                ]
            ).tz_localize("America/New_York"),
        }
    )


def test_attach_adds_both_columns_and_leaves_unknowns_null(panel, calendar, settings, live_feed):
    live_feed["install"](_report(4, [("p1", "Out", "Did Not Participate")]))
    out = attach_status(panel, calendar=calendar, settings=settings, now=BEFORE)

    assert STATUS_COLUMN in out.columns
    assert PRACTICE_COLUMN in out.columns
    wk4 = out.loc[out["week"] == 4].set_index("player_id")[STATUS_COLUMN]
    assert wk4["p1"] == "Out"
    assert bool(wk4.isna()["p2"]), "no designation must stay null, not 'healthy'"
    # Week 5 has no report of its own.
    assert out[out["week"] == 5][STATUS_COLUMN].isna().all()


def test_attach_does_not_change_the_row_count(panel, calendar, settings, live_feed):
    """A duplicated designation upstream would fan the panel out on merge."""
    live_feed["install"](
        pd.concat([_report(4, [("p1", "Out", None)]), _report(4, [("p1", "Questionable", None)])])
    )
    out = attach_status(panel, calendar=calendar, settings=settings, now=BEFORE)
    assert len(out) == len(panel)


def test_attach_on_an_empty_panel_still_adds_the_columns(calendar, settings):
    empty = pd.DataFrame(columns=["season", "week", "player_id"])
    out = attach_status(empty, calendar=calendar, settings=settings, now=BEFORE)
    assert STATUS_COLUMN in out.columns
    assert out.empty


def test_status_summary_counts_designations(panel, calendar, settings, live_feed):
    live_feed["install"](_report(4, [("p1", "Out", None), ("p2", "Questionable", None)]))
    out = attach_status(panel, calendar=calendar, settings=settings, now=BEFORE)
    assert status_summary(out) == {"out": 1, "questionable": 1}


def test_status_summary_on_a_panel_without_the_column_is_empty(panel):
    assert status_summary(panel) == {}
