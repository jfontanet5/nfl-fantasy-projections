"""Loading weekly stats and assembling the panel, with upstream faked out."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nflproj.features import panel as pnl
from nflproj.ingest import nflverse as nv
from nflproj.scoring import REQUIRED_STAT_COLUMNS, ScoringRules

TEAMS = ["AAA", "BBB"]
WEEKS = [1, 2, 3]


def _stats_frame() -> pd.DataFrame:
    """Two teams, three weeks, four players each, plus rows we expect filtered out."""
    rows: list[dict[str, Any]] = []
    for week in WEEKS:
        for team in TEAMS:
            opponent = TEAMS[1] if team == TEAMS[0] else TEAMS[0]
            for slot, position in enumerate(["QB", "RB", "WR", "TE"]):
                rows.append(
                    {
                        "season": 2024,
                        "week": week,
                        "season_type": "REG",
                        "game_id": f"2024_{week:02d}_{opponent}_{team}",
                        "player_id": f"{team}-{slot}",
                        "player_display_name": f"{team} {position}",
                        "position": position,
                        "team": team,
                        "targets": 5.0,
                        "carries": 2.0,
                        "attempts": 0.0,
                        "receiving_yards": 50.0 + slot,
                        "receptions": 4.0,
                    }
                )
    # A kicker: correct position filter should drop it.
    rows.append({**rows[0], "player_id": "AAA-K", "position": "K"})
    # A playoff row: season_type filter should drop it.
    rows.append({**rows[0], "player_id": "AAA-0", "week": 19, "season_type": "POST"})

    frame = pd.DataFrame(rows)
    for column in REQUIRED_STAT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _calendar() -> pd.DataFrame:
    rows = []
    for team in TEAMS:
        opponent = TEAMS[1] if team == TEAMS[0] else TEAMS[0]
        for idx, week in enumerate(WEEKS):
            rows.append(
                {
                    "season": np.int16(2024),
                    "week": np.int16(week),
                    "team": team,
                    "opponent_team": opponent,
                    "game_id": f"2024_{week:02d}_{opponent}_{team}",
                    "team_game_idx": idx,
                    "is_home": True,
                    "rest_days": 7,
                    "team_spread_line": 1.0,
                    "total_line": 44.0,
                    "implied_team_total": 22.5,
                    "div_game": 0,
                    "roof": "outdoors",
                    "surface": "grass",
                    "kickoff_et": pd.Timestamp("2024-09-08 13:00", tz="America/New_York"),
                    "result": 3.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def fake_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read_seasons(_asset: Any, _seasons: Any, **kwargs: Any) -> pd.DataFrame:
        frame = _stats_frame()
        columns = kwargs.get("columns")
        return frame[columns] if columns else frame

    monkeypatch.setattr(nv, "read_seasons", _read_seasons)
    monkeypatch.setattr(pnl, "build_team_calendar", lambda *_a, **_k: _calendar())


# ---------------------------------------------------------------- load


def test_only_fantasy_positions_survive():
    stats = pnl.load_weekly_stats([2024])
    assert set(stats["position"]) == {"QB", "RB", "WR", "TE"}


def test_postseason_rows_are_dropped():
    stats = pnl.load_weekly_stats([2024])
    assert stats["week"].max() == 3


def test_target_is_computed_not_inherited():
    stats = pnl.load_weekly_stats([2024])
    # 50 receiving yards (5.0) + 4 receptions (4.0) for the QB slot.
    qb = stats[(stats["player_id"] == "AAA-0") & (stats["week"] == 1)].iloc[0]
    assert qb["fantasy_points"] == pytest.approx(9.0)


def test_duplicate_panel_keys_are_rejected(monkeypatch):
    """A mid-week trade or an upstream key change must fail loudly, not fan out."""
    doubled = pd.concat([_stats_frame(), _stats_frame()], ignore_index=True)
    monkeypatch.setattr(nv, "read_seasons", lambda *_a, **_k: doubled)
    with pytest.raises(ValueError, match="duplicate"):
        pnl.load_weekly_stats([2024])


# ---------------------------------------------------------------- panel


def test_panel_key_is_unique():
    p = pnl.build_panel([2024])
    assert not p.duplicated(subset=pnl.PANEL_KEY).any()


def test_week_one_present_as_history_but_not_projectable():
    p = pnl.build_panel([2024])
    week1 = p[p["week"] == 1]
    assert len(week1) > 0, "week 1 must seed history"
    assert not week1["projectable"].any()
    assert p[p["week"] >= 2]["projectable"].all()


def test_game_context_is_joined_onto_every_row():
    p = pnl.build_panel([2024])
    assert p["game_id"].notna().all()
    assert p["opponent_team"].notna().all()
    assert p["total_line"].notna().all()


def test_non_players_are_scored_zero_not_dropped(monkeypatch):
    """A player active by recency who does not appear must score 0.0."""
    trimmed = _stats_frame()
    # AAA-1 plays weeks 1-2 then disappears; recency keeps them in week 3.
    trimmed = trimmed[~((trimmed["player_id"] == "AAA-1") & (trimmed["week"] == 3))]
    monkeypatch.setattr(nv, "read_seasons", lambda *_a, **_k: trimmed)

    p = pnl.build_panel([2024])
    row = p[(p["player_id"] == "AAA-1") & (p["week"] == 3)]
    assert len(row) == 1, "an absent player must stay in the universe"
    assert not bool(row["played"].iloc[0])
    assert row["fantasy_points"].iloc[0] == pytest.approx(0.0)


def test_scoring_rules_flow_through_to_the_target():
    ppr = pnl.build_panel([2024])
    standard = pnl.build_panel([2024], scoring=ScoringRules.standard())
    assert standard["fantasy_points"].sum() < ppr["fantasy_points"].sum()


def test_lookback_controls_universe_size():
    narrow = pnl.build_panel([2024], lookback_games=1)
    wide = pnl.build_panel([2024], lookback_games=3)
    assert len(wide) >= len(narrow)


def test_apply_universe_played_drops_only_non_players(monkeypatch):
    trimmed = _stats_frame()
    trimmed = trimmed[~((trimmed["player_id"] == "AAA-1") & (trimmed["week"] == 3))]
    monkeypatch.setattr(nv, "read_seasons", lambda *_a, **_k: trimmed)

    p = pnl.build_panel([2024])
    active = pnl.apply_universe(p, pnl.UniversePolicy.ACTIVE_RECENT)
    played = pnl.apply_universe(p, pnl.UniversePolicy.PLAYED)
    assert len(played) == len(active) - 1


# ------------------------------------------------- weeks that have not happened


def _calendar_with_unplayed_weeks() -> pd.DataFrame:
    """Weeks 1-2 finished, week 3 scheduled but not played."""
    cal = _calendar()
    cal.loc[cal["week"] == 3, "result"] = None
    return cal


def test_unplayed_weeks_are_projectable_but_not_scorable(monkeypatch):
    """The bug this guards: a week nobody has played was being scored as if
    every player had scored zero, which silently corrupts every aggregate."""
    monkeypatch.setattr(
        pnl, "build_team_calendar", lambda *_a, **_k: _calendar_with_unplayed_weeks()
    )
    p = pnl.build_panel([2024])

    week3 = p[p["week"] == 3]
    assert week3["projectable"].all(), "we still want to publish a projection for it"
    assert not week3["scorable"].any(), "but its outcome is not known yet"

    week2 = p[p["week"] == 2]
    assert week2["scorable"].all()


def test_a_partially_played_week_is_not_scorable(monkeypatch):
    """Half a slate biases every metric toward the early kickoffs."""
    cal = _calendar()
    # One team's week-3 game is done, the other's is not.
    cal.loc[(cal["week"] == 3) & (cal["team"] == "BBB"), "result"] = None
    monkeypatch.setattr(pnl, "build_team_calendar", lambda *_a, **_k: cal)

    p = pnl.build_panel([2024])
    assert not p.loc[p["week"] == 3, "scorable"].any()


def test_week_one_is_never_scorable_even_when_complete(monkeypatch):
    monkeypatch.setattr(pnl, "build_team_calendar", lambda *_a, **_k: _calendar())
    p = pnl.build_panel([2024])
    assert not p.loc[p["week"] == 1, "scorable"].any()
