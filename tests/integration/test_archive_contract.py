"""The upstream facts the archive design rests on.

Two tables, two opposite answers, and the whole snapshotter exists because of
the difference. If either answer changes, the design is wrong in a way no unit
test can see, so both are pinned here against live upstream.

Marked ``network``; CI runs these as their own job so an nflverse outage reads
as an upstream failure rather than a code regression.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.config import Settings
from nflproj.features.calendar import build_team_calendar
from nflproj.ingest import nflverse as nv
from nflproj.ingest.archive import DEPTH_CHARTS, INJURIES, Archive, depth_chart_as_of, first_kickoff

pytestmark = pytest.mark.network

CURRENT_SEASON = 2026

#: Any column whose name suggests an observation time. The claim being tested is
#: that the injury report has none of them, in any spelling.
TIMESTAMP_ISH = ("dt", "timestamp", "scraped", "updated", "observed", "as_of", "last_modified")


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("nflproj-archive")
    return Settings(data_dir=root, reports_dir=root / "reports", archive_dir=root / "archive")


def test_the_injury_report_still_carries_no_observation_time(settings):
    """The premise of the entire snapshotter.

    If nflverse ever adds a timestamp to this table, the archive stops being
    necessary and this test is how we find out rather than discovering it years
    later. A failure here is good news, not a regression.
    """
    path = nv.fetch_asset(INJURIES, CURRENT_SEASON, settings=settings)
    frame = pd.read_parquet(path)
    assert not frame.empty

    suspicious = [c for c in frame.columns if any(hint in c.lower() for hint in TIMESTAMP_ISH)]
    assert suspicious == [], (
        f"injuries now carries {suspicious}; if that is an observation time, "
        "the archive may no longer be needed - check before deleting it"
    )
    # Keyed on the week, not the moment: two Fridays in the same week are
    # indistinguishable in this file, which is precisely the problem.
    assert {"season", "week"} <= set(frame.columns)


def test_depth_charts_still_carry_their_own_observation_time(settings):
    """The mirror image: this table needs no archive, and this is why.

    A failure here means depth charts became overwrite-in-place too, and would
    have to be snapshotted like injuries.
    """
    path = nv.fetch_asset(DEPTH_CHARTS, CURRENT_SEASON, settings=settings)
    frame = pd.read_parquet(path, columns=["dt"])
    observed = pd.to_datetime(frame["dt"], format="ISO8601", utc=True)

    # Cumulative, not overwritten: many distinct scrapes in one file.
    assert observed.nunique() > 20, "depth charts look like a single snapshot, not a log"
    assert observed.min() < observed.max()


def test_a_depth_chart_as_of_view_predates_the_week_it_describes(settings):
    """End to end on real data: the as-of view must be strictly pre-kickoff."""
    calendar = build_team_calendar([CURRENT_SEASON], settings=settings)
    week = int(calendar.loc[calendar["result"].notna(), "week"].max())
    cutoff = first_kickoff(calendar, CURRENT_SEASON, week)
    assert cutoff is not None

    path = nv.fetch_asset(DEPTH_CHARTS, CURRENT_SEASON, settings=settings)
    chart = depth_chart_as_of(pd.read_parquet(path), cutoff)
    assert not chart.empty

    observed = pd.to_datetime(chart["dt"], format="ISO8601", utc=True)
    assert observed.max() < pd.Timestamp(cutoff)
    assert observed.min() == observed.max(), "an as-of view spliced together two scrapes"


def test_a_capture_round_trips_through_the_archive(settings):
    """One real capture, read back through the real read path."""
    archive = Archive(settings.archive_dir, INJURIES)
    observation = archive.capture(CURRENT_SEASON, settings=settings)
    assert observation.rows > 0

    taken = observation.fetched_at_dt
    assert archive.as_of(taken) is None, "as_of must exclude a snapshot taken at the cutoff"

    frame = archive.as_of(taken + pd.Timedelta(seconds=1).to_pytimedelta())
    assert frame is not None
    assert len(frame) == observation.rows
