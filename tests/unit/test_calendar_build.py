"""Building the team calendar from a schedule, with upstream faked out."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nflproj.features import calendar as cal
from nflproj.ingest import nflverse as nv


def _schedule() -> pd.DataFrame:
    """Three REG games plus one playoff game. AAA has a bye in week 3."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2024, 2024, 2024, 2024],
            "game_type": ["REG", "REG", "REG", "WC"],
            "week": [1, 2, 4, 19],
            "gameday": ["2024-09-08", "2024-09-15", "2024-09-29", "2025-01-11"],
            "gametime": ["13:00", "16:25", "20:20", "16:30"],
            "home_team": ["AAA", "BBB", "AAA", "AAA"],
            "away_team": ["BBB", "AAA", "CCC", "CCC"],
            "home_rest": [7, 7, 14, 7],
            "away_rest": [7, 7, 7, 7],
            "spread_line": [3.0, -6.5, 10.0, 1.0],
            "total_line": [44.0, 41.0, 50.0, 45.0],
            "div_game": [1, 1, 0, 0],
            "roof": ["outdoors", "dome", "outdoors", "outdoors"],
            "surface": ["grass", "turf", "grass", "grass"],
            "result": [3.0, -7.0, None, None],
        }
    )


@pytest.fixture(autouse=True)
def fake_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read_asset(_asset: Any, **kwargs: Any) -> pd.DataFrame:
        frame = _schedule()
        columns = kwargs.get("columns")
        return frame[columns] if columns else frame

    monkeypatch.setattr(nv, "read_asset", _read_asset)


def test_each_game_yields_one_row_per_team():
    out = cal.build_team_calendar([2024])
    # Three REG games, two teams each.
    assert len(out) == 6
    assert not out.duplicated(subset=["season", "week", "team"]).any()


def test_playoff_games_are_excluded_by_default():
    out = cal.build_team_calendar([2024])
    assert out["week"].max() == 4


def test_postseason_can_be_opted_back_in():
    out = cal.build_team_calendar([2024], regular_season_only=False)
    assert 19 in set(out["week"])


def test_spread_is_negated_for_the_away_team():
    out = cal.build_team_calendar([2024])
    g1 = out[out["game_id"] == "g1"].set_index("team")
    assert g1.loc["AAA", "is_home"]
    assert g1.loc["AAA", "team_spread_line"] == pytest.approx(3.0)
    assert g1.loc["BBB", "team_spread_line"] == pytest.approx(-3.0)


def test_implied_totals_reconstruct_the_line():
    out = cal.build_team_calendar([2024])
    g1 = out[out["game_id"] == "g1"]
    assert g1["implied_team_total"].sum() == pytest.approx(44.0)
    home = g1[g1["is_home"]]["implied_team_total"].iloc[0]
    away = g1[~g1["is_home"]]["implied_team_total"].iloc[0]
    assert home - away == pytest.approx(3.0)


def test_opponent_and_rest_follow_the_perspective():
    out = cal.build_team_calendar([2024]).set_index(["game_id", "team"])
    assert out.loc[("g3", "AAA"), "opponent_team"] == "CCC"
    assert out.loc[("g3", "AAA"), "rest_days"] == 14
    assert out.loc[("g3", "CCC"), "rest_days"] == 7


def test_team_game_index_is_bye_aware():
    """AAA plays weeks 1, 2 and 4; its third game is index 2, not week 3."""
    aaa = cal.build_team_calendar([2024])
    aaa = aaa[aaa["team"] == "AAA"].sort_values("week")
    assert aaa["week"].tolist() == [1, 2, 4]
    assert aaa["team_game_idx"].tolist() == [0, 1, 2]


def test_kickoff_is_localised_to_eastern():
    out = cal.build_team_calendar([2024])
    kickoff = out[out["game_id"] == "g1"]["kickoff_et"].iloc[0]
    assert str(kickoff.tz) == "America/New_York"
    assert kickoff.hour == 13


def test_absent_season_raises():
    with pytest.raises(ValueError, match="no rows for seasons"):
        cal.build_team_calendar([1999])


def test_latest_completed_week_stops_at_the_first_incomplete_week():
    out = cal.build_team_calendar([2024])
    # Weeks 1 and 2 have results; week 4 does not.
    assert cal.latest_completed_week(out, 2024) == 2


def test_completed_games_filters_on_result():
    out = cal.build_team_calendar([2024])
    assert len(cal.completed_games(out)) == 4
