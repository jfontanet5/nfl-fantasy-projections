"""Shared fixtures.

Unit tests run against a synthetic panel so the suite is fast, deterministic and
works with no network. Tests that genuinely need upstream data are marked
``network`` and live under ``tests/integration``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260919)


@pytest.fixture
def synthetic_panel(rng: np.random.Generator) -> pd.DataFrame:
    """A small but structurally faithful panel.

    Two seasons, 8 teams, 6 players per team across the four positions, weeks
    1-8. Player skill is drawn once and persists, so a per-player average is a
    genuinely useful predictor and the baselines have something to find. Roughly
    one in six player-weeks is a DNP scoring 0.0, matching the real panel's
    shape.
    """
    teams = [f"T{i}" for i in range(8)]
    positions = ["QB", "RB", "RB", "WR", "WR", "TE"]
    rows = []

    skill = {}
    for team in teams:
        for slot, pos in enumerate(positions):
            pid = f"{team}-{slot}"
            skill[pid] = {"QB": 17.0, "RB": 10.0, "WR": 9.0, "TE": 6.0}[pos] + rng.normal(0, 3)

    for season in (2020, 2021):
        for week in range(1, 9):
            for t_idx, team in enumerate(teams):
                opponent = teams[(t_idx + week) % len(teams)]
                if opponent == team:
                    continue
                for slot, pos in enumerate(positions):
                    pid = f"{team}-{slot}"
                    played = bool(rng.random() > 1 / 6)
                    points = max(0.0, rng.normal(skill[pid], 5.0)) if played else 0.0
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "player_id": pid,
                            "player_display_name": f"Player {pid}",
                            "position": pos,
                            "team": team,
                            "opponent_team": opponent,
                            "game_id": f"{season}_{week:02d}_{team}",
                            "last_appearance_week": max(1, week - 1),
                            "fantasy_points": round(points, 2),
                            "played": played,
                            "targets": 0.0,
                            "carries": 0.0,
                            "attempts": 0.0,
                            "is_home": t_idx % 2 == 0,
                            "rest_days": 7,
                            "team_spread_line": float(rng.integers(-10, 11)),
                            "total_line": 45.0,
                            "implied_team_total": 22.5,
                            "div_game": 0,
                            "roof": "outdoors",
                            "surface": "grass",
                            "projectable": week >= 2,
                            "week_complete": True,
                            "scorable": week >= 2,
                        }
                    )

    panel = pd.DataFrame(rows)
    return panel.sort_values(["season", "week", "player_id"], kind="mergesort").reset_index(
        drop=True
    )
