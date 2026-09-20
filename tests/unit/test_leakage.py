"""Leakage tests.

These are the tests this project exists to be able to point at. Everything else
verifies that the code does what it says; these verify that what it says is
worth anything.

Three independent arguments, weakest to strongest:

1. The target frame handed to a predictor has no outcome columns (structural).
2. A predictor that tries to read one fails loudly (structural, adversarial).
3. Corrupting every row at or after the target week leaves the predictions for
   that week bit-identical (empirical, and the only one that would catch a
   leak smuggled in through a column we forgot to list).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflproj.evaluation.backtest import BacktestConfig, run_backtest
from nflproj.predictors.base import OUTCOME_COLUMNS, strip_outcomes
from nflproj.predictors.baselines import ExponentialMean, RollingMean, SeasonToDateMean


class _PeekingPredictor:
    """A predictor that tries to cheat. It must not be able to."""

    @property
    def name(self) -> str:
        return "peeker"

    def fit(self, history: pd.DataFrame) -> None:
        self.saw_weeks = sorted(history["week"].unique().tolist())
        self.saw_seasons = sorted(history["season"].unique().tolist())

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        # The answer is simply not present. This raises KeyError.
        return targets["fantasy_points"]


class _HistoryRecorder:
    """Records the maximum (season, week) it was ever shown as history."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int, int, int]] = []

    @property
    def name(self) -> str:
        return "recorder"

    def fit(self, history: pd.DataFrame) -> None:
        self._history = history

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        target_season = int(targets["season"].iloc[0])
        target_week = int(targets["week"].iloc[0])
        if not self._history.empty:
            latest = self._history.sort_values(["season", "week"]).iloc[-1]
            self.seen.append(
                (target_season, target_week, int(latest["season"]), int(latest["week"]))
            )
        return pd.Series(0.0, index=targets.index)


def test_strip_outcomes_removes_every_outcome_column(synthetic_panel):
    blinded = strip_outcomes(synthetic_panel)
    assert not (OUTCOME_COLUMNS & set(blinded.columns))
    # And leaves everything else alone.
    assert set(blinded.columns) == set(synthetic_panel.columns) - OUTCOME_COLUMNS


def test_predictor_cannot_read_the_target(synthetic_panel):
    with pytest.raises(KeyError, match="fantasy_points"):
        run_backtest(
            synthetic_panel,
            [_PeekingPredictor()],
            config=BacktestConfig(seasons=(2021,)),
        )


def test_history_never_reaches_the_target_week(synthetic_panel):
    recorder = _HistoryRecorder()
    run_backtest(synthetic_panel, [recorder], config=BacktestConfig(seasons=(2020, 2021)))

    assert recorder.seen, "recorder was never called"
    for target_season, target_week, hist_season, hist_week in recorder.seen:
        assert (hist_season, hist_week) < (target_season, target_week), (
            f"history reached {hist_season} week {hist_week} while projecting "
            f"{target_season} week {target_week}"
        )


@pytest.mark.parametrize(
    "predictor_factory",
    [SeasonToDateMean, lambda: RollingMean(window=4), lambda: ExponentialMean(halflife=3.0)],
    ids=["season_to_date", "rolling_4", "ewma"],
)
def test_future_perturbation_does_not_change_past_predictions(synthetic_panel, predictor_factory):
    """The empirical leakage test.

    Run the backtest, then replace every outcome at or after a cut week with
    garbage and run it again. Predictions for weeks strictly before the cut must
    be identical to the last bit. If any predictor reads forward - through a
    column we forgot to blind, a groupby that spans the cut, a rolling window
    with the wrong closed side - the two runs diverge here.
    """
    cut_season, cut_week = 2021, 5

    clean = run_backtest(synthetic_panel, [predictor_factory()], config=BacktestConfig())

    corrupted = synthetic_panel.copy()
    future = (corrupted["season"] > cut_season) | (
        (corrupted["season"] == cut_season) & (corrupted["week"] >= cut_week)
    )
    rng = np.random.default_rng(0)
    corrupted.loc[future, "fantasy_points"] = rng.normal(500.0, 50.0, size=int(future.sum()))
    corrupted.loc[future, "played"] = ~corrupted.loc[future, "played"]

    dirty = run_backtest(corrupted, [predictor_factory()], config=BacktestConfig())

    def before_cut(frame: pd.DataFrame) -> pd.DataFrame:
        mask = (frame["season"] < cut_season) | (
            (frame["season"] == cut_season) & (frame["week"] < cut_week)
        )
        return (
            frame[mask]
            .sort_values(["season", "week", "player_id"], kind="mergesort")
            .reset_index(drop=True)
        )

    a, b = before_cut(clean), before_cut(dirty)
    assert len(a) > 0
    pd.testing.assert_series_equal(a["prediction"], b["prediction"], check_exact=True)


def test_perturbation_test_can_actually_fail(synthetic_panel):
    """A control for the test above.

    A deliberately leaky predictor - one that reads the target week's outcomes
    out of a frame it closed over - must make the perturbation test fail.
    Without this, a perturbation test that passes proves nothing, because it
    would also pass if the comparison were vacuous.
    """

    class Leaky:
        def __init__(self, truth: pd.DataFrame) -> None:
            self._truth = truth.set_index(["season", "week", "player_id"])["fantasy_points"]

        @property
        def name(self) -> str:
            return "leaky"

        def fit(self, history: pd.DataFrame) -> None:
            pass

        def predict(self, targets: pd.DataFrame) -> pd.Series:
            idx = pd.MultiIndex.from_frame(targets[["season", "week", "player_id"]])
            return pd.Series(
                self._truth.reindex(idx).to_numpy(dtype="float64"), index=targets.index
            ).fillna(0.0)

    cut_season, cut_week = 2021, 5
    corrupted = synthetic_panel.copy()
    future = (corrupted["season"] == cut_season) & (corrupted["week"] >= cut_week)
    corrupted.loc[future, "fantasy_points"] = 999.0

    clean = run_backtest(synthetic_panel, [Leaky(synthetic_panel)], config=BacktestConfig())
    dirty = run_backtest(corrupted, [Leaky(corrupted)], config=BacktestConfig())

    def after_cut(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["season"] == cut_season) & (frame["week"] >= cut_week)]

    assert not np.allclose(
        after_cut(clean)["prediction"].to_numpy(), after_cut(dirty)["prediction"].to_numpy()
    ), "the leaky control did not diverge; the perturbation test is not sensitive"
