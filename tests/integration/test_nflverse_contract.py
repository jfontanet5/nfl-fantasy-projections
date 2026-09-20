"""Tests that pin our assumptions about live upstream data.

Marked ``network``. They are the early-warning system for the class of failure
that already bit this project once: nflverse migrated weekly player stats from
the ``player_stats`` release tag to ``stats_player``, and the old tag kept
serving a frozen 2024 snapshot rather than 404ing. A pipeline pointed at the
stale tag would have looked healthy while silently having no current season.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.config import Settings
from nflproj.features.calendar import build_team_calendar, latest_completed_week
from nflproj.features.panel import build_panel, load_weekly_stats
from nflproj.ingest import nflverse as nv
from nflproj.ingest.manifest import Manifest
from nflproj.scoring import REQUIRED_STAT_COLUMNS, compute_fantasy_points

pytestmark = pytest.mark.network

CURRENT_SEASON = 2026


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("nflproj-data")
    return Settings(data_dir=root, reports_dir=root / "reports")


def test_release_tag_serves_the_current_season(settings):
    """The migration guard. A tag that has stopped being updated fails here."""
    path = nv.fetch_asset(nv.PLAYER_STATS, CURRENT_SEASON, settings=settings)
    frame = pd.read_parquet(path, columns=["season", "week"])
    assert not frame.empty
    assert int(frame["season"].max()) == CURRENT_SEASON


def test_legacy_release_tag_is_stale_not_missing(settings):
    """Documents *why* the guard above exists, and detects if upstream revives it."""
    legacy = nv.Asset(name="legacy_stats", release="player_stats", stem="stats_player_week")
    with pytest.raises(nv.IngestError, match="404"):
        nv.fetch_asset(legacy, CURRENT_SEASON, settings=settings)


def test_required_scoring_columns_all_exist(settings):
    path = nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    columns = set(pd.read_parquet(path).columns)
    missing = sorted(set(REQUIRED_STAT_COLUMNS) - columns)
    assert not missing, f"upstream dropped scoring columns: {missing}"


@pytest.mark.parametrize("season", [2015, 2020, 2024])
def test_our_scoring_matches_nflverse_exactly(settings, season):
    """If this drifts, either upstream changed a rule or we broke ours."""
    path = nv.fetch_asset(nv.PLAYER_STATS, season, settings=settings)
    frame = pd.read_parquet(path)
    skill = frame[frame["position"].isin(["QB", "RB", "WR", "TE"])]

    ours = compute_fantasy_points(skill)
    delta = (ours - skill["fantasy_points_ppr"]).abs()
    assert delta.max() < 1e-6, (
        f"{int((delta > 1e-6).sum())} of {len(skill)} {season} rows disagree with nflverse"
    )


def test_schedule_covers_the_current_season(settings):
    calendar = build_team_calendar(range(2015, CURRENT_SEASON + 1), settings=settings)
    assert CURRENT_SEASON in set(calendar["season"])
    # Every scheduled team-week must resolve to exactly one opponent.
    assert not calendar.duplicated(subset=["season", "week", "team"]).any()


def test_spread_sign_matches_realised_margin(settings):
    """A sign flip here would invert every betting-line feature we ever add."""
    calendar = build_team_calendar([2024], settings=settings)
    played = calendar[calendar["result"].notna() & calendar["team_spread_line"].notna()]
    home = played[played["is_home"]]
    # `result` is home margin; the home-perspective spread must correlate positively.
    corr = home["team_spread_line"].corr(home["result"])
    assert corr > 0.3, f"home spread correlates {corr:.3f} with home margin; sign is likely flipped"


def test_manifest_records_every_fetched_asset(settings):
    nv.fetch_asset(nv.PLAYER_STATS, 2023, settings=settings)
    manifest = Manifest(settings.raw_dir)
    entry = manifest.get("player_stats/2023")
    assert entry is not None
    assert entry.size_bytes > 0
    assert len(entry.sha256) == 64
    assert "stats_player" in entry.url


def test_cache_is_reused_within_the_freshness_window(settings):
    first = nv.fetch_asset(nv.PLAYER_STATS, 2022, settings=settings)
    mtime = first.stat().st_mtime
    second = nv.fetch_asset(nv.PLAYER_STATS, 2022, settings=settings)
    assert second == first
    assert second.stat().st_mtime == mtime, "a fresh cached asset was re-downloaded"


def test_missing_season_is_skipped_not_fatal(settings):
    paths = nv.fetch_seasons(nv.PLAYER_STATS, [2024, 2199], settings=settings)
    assert 2024 in paths
    assert 2199 not in paths


def test_panel_shape_is_plausible(settings):
    """Guards against a join that silently fans out or drops most rows."""
    panel = build_panel([2023, 2024], settings=settings)
    projectable = panel[panel["projectable"]]

    per_week = projectable.groupby(["season", "week"]).size()
    assert 250 < per_week.mean() < 600, f"implausible universe size: {per_week.mean():.0f}/week"

    assert not panel.duplicated(subset=["season", "week", "player_id"]).any()
    assert panel["fantasy_points"].notna().all()
    assert panel["game_id"].notna().all()
    assert set(panel["position"]) == {"QB", "RB", "WR", "TE"}

    played_rate = float(projectable["played"].mean())
    assert 0.6 < played_rate < 0.9, f"played rate {played_rate:.3f} is outside a sane band"


def test_no_postseason_rows_leak_into_the_panel(settings):
    stats = load_weekly_stats([2024], settings=settings)
    assert int(stats["week"].max()) <= 18


def test_current_season_progress_is_reported(settings):
    calendar = build_team_calendar([CURRENT_SEASON], settings=settings)
    week = latest_completed_week(calendar, CURRENT_SEASON)
    assert week is None or 1 <= week <= 18
