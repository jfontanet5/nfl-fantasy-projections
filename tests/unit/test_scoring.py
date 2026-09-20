"""Scoring rules, checked against numbers worked out by hand."""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.scoring import (
    PPR,
    REQUIRED_STAT_COLUMNS,
    ScoringRules,
    compute_fantasy_points,
)


def _row(**overrides: float) -> pd.DataFrame:
    base = dict.fromkeys(REQUIRED_STAT_COLUMNS, 0.0)
    base.update(overrides)
    return pd.DataFrame([base])


def test_passing_line():
    # 300 yards (12.0) + 2 TD (8.0) - 1 INT (2.0) = 18.0
    df = _row(passing_yards=300, passing_tds=2, passing_interceptions=1)
    assert compute_fantasy_points(df).iloc[0] == pytest.approx(18.0)


def test_ppr_reception_bonus():
    df = _row(receiving_yards=80, receptions=8, receiving_tds=1)
    assert compute_fantasy_points(df, ScoringRules.ppr()).iloc[0] == pytest.approx(22.0)
    assert compute_fantasy_points(df, ScoringRules.half_ppr()).iloc[0] == pytest.approx(18.0)
    assert compute_fantasy_points(df, ScoringRules.standard()).iloc[0] == pytest.approx(14.0)


def test_fumbles_from_all_three_sources_are_summed():
    df = _row(rushing_fumbles_lost=1, receiving_fumbles_lost=1, sack_fumbles_lost=1)
    assert compute_fantasy_points(df).iloc[0] == pytest.approx(-6.0)


def test_two_point_conversions_from_all_sources():
    df = _row(passing_2pt_conversions=1, rushing_2pt_conversions=1, receiving_2pt_conversions=1)
    assert compute_fantasy_points(df).iloc[0] == pytest.approx(6.0)


def test_return_touchdowns_count():
    """The one place our scoring used to disagree with nflverse."""
    assert compute_fantasy_points(_row(special_teams_tds=1)).iloc[0] == pytest.approx(6.0)


def test_nulls_are_treated_as_zero():
    df = _row(rushing_yards=50)
    df.loc[0, "receiving_yards"] = None
    assert compute_fantasy_points(df).iloc[0] == pytest.approx(5.0)


def test_missing_column_raises_rather_than_scoring_zero():
    df = _row(rushing_yards=50).drop(columns=["receptions"])
    with pytest.raises(KeyError, match="receptions"):
        compute_fantasy_points(df)


def test_result_is_aligned_to_input_index():
    df = pd.concat([_row(rushing_yards=10), _row(rushing_yards=20)], ignore_index=True)
    df.index = pd.Index([7, 9])
    out = compute_fantasy_points(df)
    assert out.index.tolist() == [7, 9]
    assert out.tolist() == pytest.approx([1.0, 2.0])


def test_scoring_rules_are_immutable():
    with pytest.raises((AttributeError, TypeError)):
        PPR.receptions = 0.5  # type: ignore[misc]
