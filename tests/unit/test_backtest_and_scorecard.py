"""Harness contracts and scorecard assembly."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nflproj.evaluation.backtest import BacktestConfig, project_week, run_backtest
from nflproj.evaluation.scorecard import SCORECARD_SCHEMA_VERSION, build_scorecard
from nflproj.features.panel import UniversePolicy
from nflproj.predictors.baselines import (
    HEADLINE_BASELINE_NAME,
    PositionMean,
    SeasonToDateMean,
    default_baselines,
)


class _Constant:
    def __init__(self, value: float, name: str = "constant") -> None:
        self.value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=targets.index, dtype="float64")


class _Misaligned(_Constant):
    def predict(self, targets: pd.DataFrame) -> pd.Series:  # noqa: ARG002
        return pd.Series([1.0, 2.0], index=[0, 1], dtype="float64")


def test_backtest_rejects_duplicate_predictor_names(synthetic_panel):
    with pytest.raises(ValueError, match="unique"):
        run_backtest(synthetic_panel, [_Constant(1.0), _Constant(2.0)])


def test_backtest_rejects_misaligned_predictions(synthetic_panel):
    with pytest.raises(ValueError, match="index does not match"):
        run_backtest(synthetic_panel, [_Misaligned(1.0)], config=BacktestConfig(seasons=(2021,)))


def test_backtest_never_scores_week_one(synthetic_panel):
    out = run_backtest(synthetic_panel, [_Constant(5.0)])
    assert out["week"].min() >= 2


def test_backtest_covers_every_projectable_row(synthetic_panel):
    out = run_backtest(synthetic_panel, [_Constant(5.0)])
    assert len(out) == int(synthetic_panel["projectable"].sum())


def test_backtest_honours_season_bounds(synthetic_panel):
    out = run_backtest(synthetic_panel, [_Constant(5.0)], config=BacktestConfig(seasons=(2021,)))
    assert out["season"].unique().tolist() == [2021]


def test_backtest_honours_min_week(synthetic_panel):
    out = run_backtest(synthetic_panel, [_Constant(5.0)], config=BacktestConfig(min_week=5))
    assert out["week"].min() == 5


def test_backtest_raises_when_bounds_select_nothing(synthetic_panel):
    with pytest.raises(ValueError, match="no predictions"):
        run_backtest(synthetic_panel, [_Constant(5.0)], config=BacktestConfig(seasons=(1999,)))


def test_project_week_matches_the_backtest_for_the_same_week(synthetic_panel):
    """The live path and the measured path must agree, or the scorecard lies."""
    predictor = SeasonToDateMean()
    live = project_week(synthetic_panel, predictor, season=2021, week=6)

    back = run_backtest(
        synthetic_panel, [SeasonToDateMean()], config=BacktestConfig(seasons=(2021,))
    )
    back = back[back["week"] == 6]

    merged = live.merge(back, on=["season", "week", "player_id"], suffixes=("_live", "_back"))
    assert len(merged) == len(live)
    pd.testing.assert_series_equal(
        merged["prediction_live"], merged["prediction_back"], check_names=False
    )


def test_project_week_rejects_a_week_with_no_universe(synthetic_panel):
    with pytest.raises(ValueError, match="no projectable universe rows"):
        project_week(synthetic_panel, SeasonToDateMean(), season=2021, week=99)


def test_scorecard_baseline_has_zero_skill_against_itself(synthetic_panel):
    preds = run_backtest(synthetic_panel, default_baselines())
    card = build_scorecard(preds)
    overall = card.slices["overall"]
    row = overall[overall["predictor"] == HEADLINE_BASELINE_NAME].iloc[0]
    assert row["mae_skill"] == pytest.approx(0.0)
    assert row["rmse_skill"] == pytest.approx(0.0)


def test_scorecard_requires_its_baseline_to_be_present(synthetic_panel):
    preds = run_backtest(synthetic_panel, [PositionMean()])
    with pytest.raises(ValueError, match="not among predictors"):
        build_scorecard(preds)


def test_scorecard_slices_are_all_populated(synthetic_panel):
    preds = run_backtest(synthetic_panel, default_baselines())
    card = build_scorecard(preds)
    assert set(card.slices) == {"overall", "by_position", "by_season_phase", "by_week"}
    for name, frame in card.slices.items():
        assert not frame.empty, f"{name} slice is empty"
    assert set(card.slices["by_position"]["position"]) == {"QB", "RB", "WR", "TE"}
    assert set(card.slices["by_season_phase"]["season_phase"]) == {"weeks_2_4", "weeks_5_plus"}


def test_played_universe_scores_fewer_rows(synthetic_panel):
    preds = run_backtest(synthetic_panel, default_baselines())
    active = build_scorecard(preds, universe=UniversePolicy.ACTIVE_RECENT)
    played = build_scorecard(preds, universe=UniversePolicy.PLAYED)
    assert played.n_predictions < active.n_predictions


def test_scorecard_roundtrips_to_json(synthetic_panel, tmp_path):
    preds = run_backtest(synthetic_panel, default_baselines())
    card = build_scorecard(preds, provenance={"sha": "abc123"})
    path = card.write(tmp_path)

    payload = json.loads(path.read_text())
    assert payload["baseline"] == HEADLINE_BASELINE_NAME
    assert payload["provenance"]["sha"] == "abc123"
    assert payload["schema_version"] == SCORECARD_SCHEMA_VERSION
    assert len(payload["slices"]["overall"]) == len(default_baselines())
    assert (tmp_path / "scorecard.md").exists()


def test_weekly_slice_covers_every_scored_week(synthetic_panel):
    """The record is only a record if no week is missing from it."""
    preds = run_backtest(synthetic_panel, default_baselines())
    card = build_scorecard(preds)
    by_week = card.slices["by_week"]

    scored = preds.groupby(["season", "week"], observed=True).ngroups
    assert by_week.groupby(["season", "week"], observed=True).ngroups == scored
    # Per-week rows describe one week each, so the column would be constant.
    assert "n_weeks" not in by_week.columns


def test_weekly_slice_is_append_ordered(synthetic_panel):
    """Chronological order keeps the weekly commit a small diff, not a rewrite."""
    preds = run_backtest(synthetic_panel, default_baselines())
    by_week = build_scorecard(preds).slices["by_week"]
    keys = list(zip(by_week["season"], by_week["week"], strict=True))
    assert keys == sorted(keys)


def test_metrics_are_rounded_before_serialising(synthetic_panel):
    """Full float repr roughly doubles a file that is committed every week."""
    preds = run_backtest(synthetic_panel, default_baselines())
    payload = json.loads(build_scorecard(preds).to_json())
    for row in payload["slices"]["by_week"]:
        assert len(str(row["mae"]).split(".")[-1]) <= 6


def test_markdown_summarises_the_weekly_slice_instead_of_tabulating_it(synthetic_panel):
    """The markdown scorecard is for reading; the full series lives in the JSON."""
    preds = run_backtest(synthetic_panel, default_baselines())
    card = build_scorecard(preds)
    md = card.to_markdown()

    assert "## by_week" in md
    assert "see `scorecard.json`" in md
    # The other slices are still real tables.
    assert "## overall" in md
    assert "|" in md.split("## overall")[1][:400]
    # And the summary is far shorter than the table would have been.
    assert len(md) < len(card.to_json())
