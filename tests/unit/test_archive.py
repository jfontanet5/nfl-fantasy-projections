"""The point-in-time archive.

These tests are about a property that is impossible to verify after the fact:
that a snapshot the archive hands back was genuinely taken before the moment
you asked about. There is no upstream to check that against later - if the
read side is wrong, the resulting model looks fine and is quietly cheating.

So the network is faked and the clock is injected, and the assertions are on
the boundary conditions rather than the happy path.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from nflproj.config import Settings
from nflproj.ingest.archive import (
    INJURIES,
    Archive,
    depth_chart_as_of,
    first_kickoff,
    injuries_before_kickoff,
    snapshot_season,
)
from nflproj.ingest.nflverse import IngestError


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.status_code = 200
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


def _injury_bytes(tmp_path: Path, statuses: list[str]) -> bytes:
    """A stand-in injury report. Changing a status changes the bytes."""
    path = tmp_path / f"injuries_{'_'.join(statuses)}.parquet"
    pd.DataFrame(
        {
            "season": [2026] * len(statuses),
            "week": [3] * len(statuses),
            "gsis_id": [f"p{i}" for i in range(len(statuses))],
            "report_status": statuses,
        }
    ).to_parquet(path)
    return bytes(path.read_bytes())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        archive_dir=tmp_path / "archive",
    )


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A mutable fake upstream: set ``state['body']`` to change what it serves."""
    state: dict[str, Any] = {"body": _injury_bytes(tmp_path, ["Questionable"]), "calls": 0}

    def _get(_url: str, **_kwargs: Any) -> _FakeResponse:
        state["calls"] += 1
        return _FakeResponse(state["body"])

    monkeypatch.setattr(requests, "get", _get)
    return state


@pytest.fixture
def archive(settings: Settings) -> Archive:
    return Archive(settings.archive_dir, INJURIES)


def _at(hour: int, *, day: int = 11) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=UTC)


# ------------------------------------------------------------------ write side


@pytest.mark.usefixtures("upstream")
def test_identical_bytes_are_stored_once_but_recorded_twice(archive, settings):
    """Dedup is what makes a frequent cadence affordable.

    The second capture is not wasted work: it is the evidence that the report
    did not change between two times, which is itself a fact about the week.
    """
    first = archive.capture(2026, settings=settings, now=_at(12))
    second = archive.capture(2026, settings=settings, now=_at(18))

    assert first.novel is True
    assert second.novel is False
    assert first.sha256 == second.sha256
    assert len(list(archive.blob_dir.glob("*.parquet"))) == 1
    assert len(archive.observations()) == 2


def test_a_changed_report_becomes_a_second_blob(archive, settings, upstream, tmp_path):
    archive.capture(2026, settings=settings, now=_at(12))
    upstream["body"] = _injury_bytes(tmp_path, ["Out"])
    second = archive.capture(2026, settings=settings, now=_at(18))

    assert second.novel is True
    assert len(list(archive.blob_dir.glob("*.parquet"))) == 2


def test_capture_always_refetches(archive, settings, upstream):
    """A cached copy would archive a lie about when the bytes were observed."""
    archive.capture(2026, settings=settings, now=_at(12))
    archive.capture(2026, settings=settings, now=_at(13))
    assert upstream["calls"] == 2


def test_the_manifest_is_append_only(archive, settings, upstream, tmp_path):
    archive.capture(2026, settings=settings, now=_at(12))
    before = archive.manifest_path.read_text()
    upstream["body"] = _injury_bytes(tmp_path, ["Out"])
    archive.capture(2026, settings=settings, now=_at(18))
    after = archive.manifest_path.read_text()
    assert after.startswith(before), "an existing manifest line was rewritten"


@pytest.mark.usefixtures("upstream")
def test_a_season_filter_separates_the_record(archive, settings):
    archive.capture(2026, settings=settings, now=_at(12))
    archive.capture(2025, settings=settings, now=_at(13))
    assert len(archive.observations(season=2026)) == 1
    assert len(archive.observations()) == 2


# ------------------------------------------------------------------- read side


@pytest.mark.usefixtures("upstream")
def test_as_of_is_strictly_before_the_cutoff(archive, settings):
    """At-or-after would admit a snapshot taken once inactives were published.

    This is the whole reason the archive exists, so it is asserted on the exact
    boundary rather than somewhere safely inside it.
    """
    taken = _at(12)
    archive.capture(2026, settings=settings, now=taken)

    assert archive.as_of(taken) is None
    assert archive.as_of(taken - timedelta(microseconds=1)) is None
    assert archive.as_of(taken + timedelta(microseconds=1)) is not None


@pytest.mark.usefixtures("upstream")
def test_as_of_returns_none_before_the_archive_begins(archive, settings):
    """Every week before this archive started. Absence, not "nobody was hurt"."""
    archive.capture(2026, settings=settings, now=_at(12))
    assert archive.as_of(_at(12, day=1)) is None


def test_as_of_picks_the_latest_snapshot_before_the_cutoff(archive, settings, upstream, tmp_path):
    archive.capture(2026, settings=settings, now=_at(9))
    upstream["body"] = _injury_bytes(tmp_path, ["Doubtful"])
    archive.capture(2026, settings=settings, now=_at(12))
    upstream["body"] = _injury_bytes(tmp_path, ["Out"])
    archive.capture(2026, settings=settings, now=_at(20))

    frame = archive.as_of(_at(17))
    assert frame is not None
    assert frame["report_status"].tolist() == ["Doubtful"]


@pytest.mark.usefixtures("upstream")
def test_as_of_rejects_a_naive_datetime(archive, settings):
    """A naive cutoff silently means "before or after, depending on the server"."""
    archive.capture(2026, settings=settings, now=_at(12))
    with pytest.raises(ValueError, match="timezone-aware"):
        archive.as_of(datetime(2026, 9, 11, 12))  # noqa: DTZ001


def test_as_of_on_an_empty_archive_is_none(archive):
    assert archive.as_of(_at(12)) is None
    assert archive.observations() == []


@pytest.mark.usefixtures("upstream")
def test_a_malformed_manifest_line_is_skipped_not_fatal(archive, settings):
    """One bad append must not make the rest of the record unreadable."""
    archive.capture(2026, settings=settings, now=_at(12))
    with archive.manifest_path.open("a") as fh:
        fh.write("{not json at all\n")
        fh.write('{"asset": "injuries", "unexpected_field": 1}\n')
        fh.write("\n")

    assert len(archive.observations()) == 1


@pytest.mark.usefixtures("upstream")
def test_a_line_with_an_unparseable_timestamp_is_skipped(archive, settings):
    """An observation that cannot be placed in time is not an observation."""
    archive.capture(2026, settings=settings, now=_at(12))
    good = archive.observations()[0]
    broken = {**asdict(good), "fetched_at": "sometime on Friday"}
    naive = {**asdict(good), "fetched_at": "2026-09-11T12:00:00"}
    with archive.manifest_path.open("a") as fh:
        fh.write(json.dumps(broken) + "\n")
        fh.write(json.dumps(naive) + "\n")

    assert len(archive.observations()) == 1


@pytest.mark.usefixtures("upstream")
def test_ordering_survives_mixed_utc_offsets(archive, settings):
    """Sorting by the raw string would put 08:00-04:00 (=12:00Z) before 10:00Z.

    Nothing writes offsets other than +00:00 today. The archive is meant to
    outlive the code that writes it, so it does not rely on that.
    """
    archive.capture(2026, settings=settings, now=_at(10))
    seen = archive.observations()[0]
    later = {**asdict(seen), "fetched_at": "2026-09-11T08:00:00-04:00", "novel": False}
    with archive.manifest_path.open("a") as fh:
        fh.write(json.dumps(later) + "\n")

    stamps = [o.fetched_at_dt for o in archive.observations()]
    assert stamps == sorted(stamps)
    assert stamps[-1].hour == 8  # the -04:00 line, which is 12:00Z


@pytest.mark.usefixtures("upstream")
def test_a_missing_blob_is_an_error_not_a_silent_none(archive, settings):
    """Returning None would look exactly like "before the archive began"."""
    observation = archive.capture(2026, settings=settings, now=_at(12))
    archive.blob_path(observation.sha256).unlink()
    with pytest.raises(IngestError, match="missing"):
        archive.as_of(_at(18))


# -------------------------------------------------------------------- calendar


@pytest.fixture
def calendar() -> pd.DataFrame:
    """Two weeks: a Thursday opener then a Sunday slate, in Eastern time."""
    rows = [
        (2026, 3, "2026-09-17 20:15"),  # TNF
        (2026, 3, "2026-09-20 13:00"),  # early Sunday
        (2026, 3, "2026-09-20 16:25"),  # late Sunday
        (2026, 4, "2026-09-27 13:00"),
    ]
    frame = pd.DataFrame(rows, columns=["season", "week", "gametime"])
    frame["kickoff_et"] = pd.to_datetime(frame["gametime"]).dt.tz_localize("America/New_York")
    return frame.drop(columns=["gametime"])


def test_first_kickoff_is_the_earliest_game_of_the_week(calendar):
    """A week is published once, so its cutoff is its first game, not each game's.

    Using per-game kickoffs would let a Sunday projection read Thursday's
    result - a leak that is invisible unless you look for it.
    """
    cutoff = first_kickoff(calendar, 2026, 3)
    assert cutoff == datetime(2026, 9, 18, 0, 15, tzinfo=UTC)  # 20:15 ET


def test_first_kickoff_of_an_unscheduled_week_is_none(calendar):
    assert first_kickoff(calendar, 2026, 19) is None


@pytest.mark.usefixtures("upstream")
def test_injuries_before_kickoff_is_none_for_a_week_the_archive_predates(calendar, settings):
    Archive(settings.archive_dir, INJURIES).capture(2026, settings=settings, now=_at(12, day=19))
    # Week 3 kicked off on the 18th; the only snapshot was taken on the 19th.
    assert injuries_before_kickoff(settings.archive_dir, calendar, season=2026, week=3) is None
    assert injuries_before_kickoff(settings.archive_dir, calendar, season=2026, week=4) is not None


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 9, 16, 22, tzinfo=UTC), 2026),  # day before the opener
        (datetime(2026, 9, 18, 12, tzinfo=UTC), 2026),  # between TNF and Sunday
        (datetime(2026, 6, 1, 12, tzinfo=UTC), None),  # offseason
        (datetime(2027, 3, 1, 12, tzinfo=UTC), None),  # after the last game
    ],
)
def test_snapshot_season_answers_is_a_game_imminent(calendar, now, expected):
    assert snapshot_season(calendar, now=now) == expected


def test_snapshot_season_does_not_file_under_last_year_before_an_opener():
    """`current_season` would say 2025 here, and the snapshot would be useless."""
    calendar = pd.DataFrame(
        {
            "season": [2025, 2026],
            "week": [18, 1],
            "kickoff_et": pd.to_datetime(["2026-01-04 13:00", "2026-09-10 20:15"]).tz_localize(
                "America/New_York"
            ),
        }
    )
    now = datetime(2026, 9, 9, 22, tzinfo=UTC)
    assert snapshot_season(calendar, now=now) == 2026


def test_snapshot_season_on_an_empty_calendar_is_none():
    assert snapshot_season(pd.DataFrame(), now=_at(12)) is None


# ----------------------------------------------------------------- depth charts


@pytest.fixture
def depth_charts() -> pd.DataFrame:
    """Three scrapes of one team's chart, the last one after kickoff."""
    return pd.DataFrame(
        {
            "dt": [
                "2026-09-16T14:02:00Z",
                "2026-09-16T14:02:00Z",
                "2026-09-17T14:31:00Z",
                "2026-09-17T14:31:00Z",
                "2026-09-18T02:00:00Z",
            ],
            "player": ["a", "b", "a", "b", "a"],
            "depth": [1, 2, 2, 1, 1],
        }
    )


def test_depth_chart_as_of_takes_one_whole_scrape(depth_charts):
    """Not the latest row per player: a chart is a snapshot of a whole team."""
    out = depth_chart_as_of(depth_charts, datetime(2026, 9, 17, 20, tzinfo=UTC))
    assert out["dt"].unique().tolist() == ["2026-09-17T14:31:00Z"]
    assert out.set_index("player")["depth"].to_dict() == {"a": 2, "b": 1}


def test_depth_chart_as_of_excludes_the_post_kickoff_scrape(depth_charts):
    out = depth_chart_as_of(depth_charts, datetime(2026, 9, 17, 14, 31, tzinfo=UTC))
    assert out["dt"].unique().tolist() == ["2026-09-16T14:02:00Z"]


def test_depth_chart_as_of_before_any_observation_is_empty(depth_charts):
    out = depth_chart_as_of(depth_charts, datetime(2026, 9, 1, tzinfo=UTC))
    assert out.empty
    assert list(out.columns) == list(depth_charts.columns)


def test_depth_chart_as_of_rejects_a_naive_datetime(depth_charts):
    with pytest.raises(ValueError, match="timezone-aware"):
        depth_chart_as_of(depth_charts, datetime(2026, 9, 17, 20))  # noqa: DTZ001


def test_depth_chart_as_of_rejects_a_frame_without_dt(depth_charts):
    """If upstream drops the column, the point-in-time claim is void - say so."""
    with pytest.raises(KeyError, match="dt"):
        depth_chart_as_of(depth_charts.drop(columns=["dt"]), _at(12))
