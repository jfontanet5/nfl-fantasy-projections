"""Metrics, checked against values computed by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflproj.evaluation import metrics as m


@pytest.fixture
def toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": 2024,
            "week": 3,
            "position": "WR",
            "fantasy_points": [10.0, 20.0, 5.0, 15.0, 0.0, 30.0],
            "prediction": [12.0, 18.0, 6.0, 14.0, 4.0, 25.0],
        }
    )


def test_mae_and_rmse(toy):
    errors = np.array([2.0, 2.0, 1.0, 1.0, 4.0, 5.0])
    assert m.mae(toy["fantasy_points"], toy["prediction"]) == pytest.approx(errors.mean())
    assert m.rmse(toy["fantasy_points"], toy["prediction"]) == pytest.approx(
        np.sqrt((errors**2).mean())
    )


def test_bias_sign_convention(toy):
    """Positive bias means the predictor runs low."""
    under = toy.assign(prediction=toy["prediction"] - 5.0)
    assert m.bias(under["fantasy_points"], under["prediction"]) == pytest.approx(
        m.bias(toy["fantasy_points"], toy["prediction"]) + 5.0
    )


def test_calibration_slope_of_a_perfect_predictor_is_one(toy):
    assert m.calibration_slope(toy["fantasy_points"], toy["fantasy_points"]) == pytest.approx(1.0)


def test_calibration_slope_detects_over_dispersion(toy):
    """Predictions spread twice as wide as reality give a slope of 0.5."""
    centred = toy["fantasy_points"] - toy["fantasy_points"].mean()
    wide = centred * 2 + toy["fantasy_points"].mean()
    assert m.calibration_slope(toy["fantasy_points"], wide) == pytest.approx(0.5)


def test_calibration_slope_is_nan_for_a_constant_predictor(toy):
    assert np.isnan(m.calibration_slope(toy["fantasy_points"], pd.Series(7.0, index=toy.index)))


def test_grouped_spearman_perfect_ordering(toy):
    assert m.grouped_spearman(
        toy.assign(prediction=toy["fantasy_points"]),
        actual_col="fantasy_points",
        pred_col="prediction",
    ) == pytest.approx(1.0)


def test_grouped_spearman_reversed_ordering(toy):
    assert m.grouped_spearman(
        toy.assign(prediction=-toy["fantasy_points"]),
        actual_col="fantasy_points",
        pred_col="prediction",
    ) == pytest.approx(-1.0)


def test_grouped_spearman_skips_tiny_groups(toy):
    tiny = toy.head(3)
    assert np.isnan(m.grouped_spearman(tiny, actual_col="fantasy_points", pred_col="prediction"))


def test_top_n_hit_rate_perfect_and_worst():
    frame = pd.DataFrame(
        {
            "season": 2024,
            "week": 1,
            "position": "QB",
            "fantasy_points": np.arange(24, dtype="float64"),
            "prediction": np.arange(24, dtype="float64"),
        }
    )
    assert m.top_n_hit_rate(
        frame, actual_col="fantasy_points", pred_col="prediction", top_n={"QB": 12}
    ) == pytest.approx(1.0)

    reversed_frame = frame.assign(prediction=-frame["fantasy_points"])
    assert m.top_n_hit_rate(
        reversed_frame, actual_col="fantasy_points", pred_col="prediction", top_n={"QB": 12}
    ) == pytest.approx(0.0)


def test_evaluate_rejects_null_predictions(toy):
    broken = toy.copy()
    broken.loc[0, "prediction"] = np.nan
    with pytest.raises(ValueError, match="null predictions"):
        m.evaluate(broken, predictor="broken")


def test_evaluate_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty frame"):
        m.evaluate(pd.DataFrame(columns=["season", "week", "position"]), predictor="x")


def test_skill_score_directions():
    # Lower-is-better: 4 against a baseline of 5 is a 20% improvement.
    assert m.skill_score(4.0, 5.0) == pytest.approx(0.2)
    # Higher-is-better: 0.6 against 0.5 is likewise 20%.
    assert m.skill_score(0.6, 0.5, lower_is_better=False) == pytest.approx(0.2)
    # A predictor compared against itself has no skill, by construction.
    assert m.skill_score(4.0, 4.0) == pytest.approx(0.0)
    assert np.isnan(m.skill_score(4.0, 0.0))
