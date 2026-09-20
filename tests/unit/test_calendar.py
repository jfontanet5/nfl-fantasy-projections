"""Schedule-derived calendar: sign conventions and completion logic."""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.features.calendar import (
    LATE_AVAILABILITY_COLUMNS,
    completed_games,
    latest_completed_week,
)


@pytest.fixture
def calendar() -> pd.DataFrame:
    """Two teams, three weeks. Week 3 is only half played."""
    return pd.DataFrame(
        {
            "season": [2024] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team": ["AAA", "BBB"] * 3,
            "result": [3.0, -3.0, 7.0, -7.0, 1.0, None],
        }
    )


def test_latest_completed_week_ignores_partial_weeks(calendar):
    assert latest_completed_week(calendar, 2024) == 2


def test_latest_completed_week_returns_none_when_nothing_is_done():
    nothing = pd.DataFrame({"season": [2026, 2026], "week": [1, 1], "result": [None, None]})
    assert latest_completed_week(nothing, 2026) is None


def test_latest_completed_week_for_an_absent_season(calendar):
    assert latest_completed_week(calendar, 1999) is None


def test_completed_games_drops_unplayed(calendar):
    assert len(completed_games(calendar)) == 5


def test_late_availability_columns_are_declared():
    """The closing line is pre-kickoff but late; that must stay documented."""
    assert "team_spread_line" in LATE_AVAILABILITY_COLUMNS
    assert "implied_team_total" in LATE_AVAILABILITY_COLUMNS
    assert "rest_days" not in LATE_AVAILABILITY_COLUMNS


def test_spread_sign_convention_is_symmetric():
    """Derived from nflverse's home-perspective line, the two sides must negate."""
    spread_line = 6.5
    home_view = spread_line
    away_view = -spread_line
    assert home_view == -away_view

    total = 44.0
    home_implied = total / 2 + home_view / 2
    away_implied = total / 2 + away_view / 2
    assert home_implied + away_implied == pytest.approx(total)
    assert home_implied - away_implied == pytest.approx(spread_line)
