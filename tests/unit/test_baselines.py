"""Baseline predictors."""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.predictors.baselines import (
    ExponentialMean,
    LastGame,
    PositionMean,
    RollingMean,
    SeasonToDateMean,
    default_baselines,
)


@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2024, 2024],
            "week": [17, 1, 2, 3, 3],
            "player_id": ["p1", "p1", "p1", "p1", "p2"],
            "position": ["WR", "WR", "WR", "WR", "WR"],
            "fantasy_points": [100.0, 6.0, 0.0, 12.0, 8.0],
            "played": [True, True, False, True, True],
        }
    )


@pytest.fixture
def targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [4, 4, 4],
            "player_id": ["p1", "p2", "brand_new"],
            "position": ["WR", "WR", "TE"],
        }
    )


def test_season_to_date_ignores_prior_seasons(history, targets):
    """The 100-point 2023 game must not touch a 2024 season-to-date average."""
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(targets)
    assert out.iloc[0] == pytest.approx((6.0 + 0.0 + 12.0) / 3)


def test_include_dnp_changes_the_estimate(history, targets):
    with_dnp = SeasonToDateMean(include_dnp=True)
    without = SeasonToDateMean(include_dnp=False)
    with_dnp.fit(history)
    without.fit(history)
    assert with_dnp.predict(targets).iloc[0] == pytest.approx(6.0)
    assert without.predict(targets).iloc[0] == pytest.approx(9.0)


def test_unknown_player_falls_back_to_the_position_prior(history, targets):
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(targets)
    # No TE history at all in the target season, so the global prior applies.
    assert out.iloc[2] > 0.0
    assert out.notna().all()


def test_shrinkage_pulls_a_thin_sample_toward_the_position_mean(history, targets):
    unshrunk = SeasonToDateMean()
    shrunk = SeasonToDateMean(shrinkage_games=4.0)
    unshrunk.fit(history)
    shrunk.fit(history)

    # p2 has a single 8.0 game; the WR prior this season is lower.
    position_prior = history[history["season"] == 2024]["fantasy_points"].mean()
    raw = unshrunk.predict(targets).iloc[1]
    pulled = shrunk.predict(targets).iloc[1]
    assert abs(pulled - position_prior) < abs(raw - position_prior)


def test_last_game_is_the_most_recent_result(history, targets):
    pred = LastGame()
    pred.fit(history)
    assert pred.predict(targets).iloc[0] == pytest.approx(12.0)


def test_rolling_mean_uses_only_the_window(history, targets):
    pred = RollingMean(window=2)
    pred.fit(history)
    # p1's last two games: 0.0 and 12.0.
    assert pred.predict(targets).iloc[0] == pytest.approx(6.0)


def test_ewma_weights_recent_games_more(history, targets):
    ewma = ExponentialMean(halflife=1.0)
    flat = RollingMean(window=3)
    ewma.fit(history)
    flat.fit(history)
    # p1's sequence is 6, 0, 12; a recency-weighted mean must exceed the flat one.
    assert ewma.predict(targets).iloc[0] > flat.predict(targets).iloc[0]


def test_position_mean_is_constant_within_a_position(history, targets):
    pred = PositionMean()
    pred.fit(history)
    out = pred.predict(targets)
    assert out.iloc[0] == pytest.approx(out.iloc[1])


def test_predictions_align_to_target_index(history, targets):
    shuffled = targets.sample(frac=1.0, random_state=1)
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(shuffled)
    assert out.index.equals(shuffled.index)


def test_empty_history_still_produces_numbers(targets):
    empty = pd.DataFrame(
        columns=["season", "week", "player_id", "position", "fantasy_points", "played"]
    )
    for predictor in default_baselines():
        predictor.fit(empty)
        out = predictor.predict(targets)
        assert out.notna().all(), f"{predictor.name} emitted nulls on empty history"


def test_default_baseline_names_are_unique():
    names = [p.name for p in default_baselines()]
    assert len(names) == len(set(names))
