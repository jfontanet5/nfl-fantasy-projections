"""Universe construction and panel assembly."""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.features.panel import (
    FIRST_PROJECTABLE_WEEK,
    PANEL_KEY,
    UniversePolicy,
    apply_universe,
    build_universe,
)


@pytest.fixture
def stats() -> pd.DataFrame:
    """One team, one player, appearing in weeks 1 and 2 only."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "player_id": ["p1", "p1"],
            "player_display_name": ["Player One", "Player One"],
            "position": ["WR", "WR"],
            "team": ["AAA", "AAA"],
            "fantasy_points": [10.0, 12.0],
        }
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    """Weeks 1-6 with a bye in week 4, so game index 3 is week 5."""
    weeks = [1, 2, 3, 5, 6]
    return pd.DataFrame(
        {
            "season": 2024,
            "week": weeks,
            "team": "AAA",
            "team_game_idx": range(len(weeks)),
        }
    )


def test_activation_window_spans_the_next_n_games(stats, calendar):
    universe = build_universe(stats, calendar, lookback_games=2)
    projected = sorted(universe.loc[universe["week"] >= FIRST_PROJECTABLE_WEEK, "week"].tolist())
    # Week 1 activates weeks 2 and 3; week 2 activates weeks 3 and 5.
    assert projected == [2, 3, 5]


def test_activation_is_bye_aware(stats, calendar):
    """A bye must not consume part of the lookback window."""
    universe = build_universe(stats, calendar, lookback_games=3)
    projected = sorted(universe.loc[universe["week"] >= FIRST_PROJECTABLE_WEEK, "week"].tolist())
    # Week 2's appearance reaches game indices 2, 3, 4 -> weeks 3, 5, 6.
    # Week 4 is a bye and is correctly absent rather than eating a slot.
    assert projected == [2, 3, 5, 6]
    assert 4 not in projected


def test_player_drops_out_after_the_window_expires(stats, calendar):
    universe = build_universe(stats, calendar, lookback_games=1)
    # Last appearance is week 2, which activates only the next game, week 3.
    assert sorted(universe["week"].tolist()) == [1, 2, 3]
    assert 5 not in universe["week"].tolist()


def test_week_one_rows_are_present_as_history(stats, calendar):
    universe = build_universe(stats, calendar, lookback_games=2)
    assert 1 in universe["week"].tolist(), "week 1 must be retained to seed history"


def test_universe_key_is_unique(stats, calendar):
    universe = build_universe(stats, calendar, lookback_games=3)
    assert not universe.duplicated(subset=PANEL_KEY).any()


def test_team_and_position_come_from_the_latest_appearance(calendar):
    """A player who changes team mid-season carries the newer team forward."""
    traded = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "player_id": ["p1", "p1"],
            "player_display_name": ["Player One", "Player One"],
            "position": ["WR", "WR"],
            "team": ["AAA", "AAA"],
            "fantasy_points": [10.0, 12.0],
        }
    )
    universe = build_universe(traded, calendar, lookback_games=3)
    week3 = universe[universe["week"] == 3].iloc[0]
    assert week3["last_appearance_week"] == 2


def test_lookback_must_be_positive(stats, calendar):
    with pytest.raises(ValueError, match="lookback_games must be >= 1"):
        build_universe(stats, calendar, lookback_games=0)


def test_apply_universe_active_recent_keeps_dnps(synthetic_panel):
    active = apply_universe(synthetic_panel, UniversePolicy.ACTIVE_RECENT)
    assert active["projectable"].all()
    assert not active["played"].all(), "ACTIVE_RECENT must retain players who did not play"
    assert (active.loc[~active["played"], "fantasy_points"] == 0.0).all()


def test_apply_universe_played_is_a_strict_subset(synthetic_panel):
    active = apply_universe(synthetic_panel, UniversePolicy.ACTIVE_RECENT)
    played = apply_universe(synthetic_panel, UniversePolicy.PLAYED)
    assert len(played) < len(active)
    assert played["played"].all()


def test_week_one_is_never_projectable(synthetic_panel):
    assert not synthetic_panel.loc[synthetic_panel["week"] == 1, "projectable"].any()
