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


class _HistoryProbe:
    """Records the latest (season, week) it was ever shown as history."""

    def __init__(self) -> None:
        self.max_seen: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return "probe"

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            return
        latest = history.sort_values(["season", "week"]).iloc[-1]
        seen = (int(latest["season"]), int(latest["week"]))
        self.max_seen = seen if self.max_seen is None else max(self.max_seen, seen)

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=targets.index)


class _HistoryWeeks:
    """Records every (season, week) the predictor was shown as history.

    Season and week together: a bare week number pools 2020's settled week 7
    with 2021's unplayed one, which makes the assertion meaningless.
    """

    def __init__(self) -> None:
        self.weeks_seen: set[tuple[int, int]] = set()

    @property
    def name(self) -> str:
        return "history-weeks"

    def fit(self, history: pd.DataFrame) -> None:
        self.weeks_seen |= {
            (int(s), int(w)) for s, w in history[["season", "week"]].drop_duplicates().to_numpy()
        }

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=targets.index)


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


# ------------------------------------------------- weeks that have not happened


def _panel_with_unplayed_tail(panel: pd.DataFrame) -> pd.DataFrame:
    """Mark 2021 weeks 7-8 as scheduled but not yet played."""
    out = panel.copy()
    unplayed = (out["season"] == 2021) & (out["week"] >= 7)
    out["week_complete"] = ~unplayed
    out["scorable"] = out["projectable"] & out["week_complete"]
    # Their outcomes are placeholders, exactly as build_panel would leave them.
    out.loc[unplayed, "fantasy_points"] = 0.0
    out.loc[unplayed, "played"] = False
    return out


def test_backtest_never_scores_an_unplayed_week(synthetic_panel):
    panel = _panel_with_unplayed_tail(synthetic_panel)
    out = run_backtest(panel, [_Constant(5.0)])
    scored = set(zip(out["season"], out["week"], strict=True))
    assert not [w for w in scored if w[0] == 2021 and w[1] >= 7]
    assert (2021, 6) in scored


def test_unplayed_weeks_do_not_leak_into_history(synthetic_panel):
    """Placeholder zeros would drag every historical average toward zero."""
    panel = _panel_with_unplayed_tail(synthetic_panel)
    recorder = _HistoryProbe()
    run_backtest(panel, [recorder], config=BacktestConfig(seasons=(2021,)))
    assert recorder.max_seen is not None
    assert recorder.max_seen < (2021, 7)


def test_an_in_progress_season_does_not_change_settled_results(synthetic_panel):
    """Adding unplayed weeks must not move a single previously scored number."""
    settled = run_backtest(synthetic_panel, [_Constant(5.0)])
    with_tail = run_backtest(_panel_with_unplayed_tail(synthetic_panel), [_Constant(5.0)])

    key = ["season", "week", "player_id"]
    common = settled.merge(with_tail, on=key, suffixes=("_a", "_b"))
    assert len(common) == len(with_tail)
    pd.testing.assert_series_equal(
        common["fantasy_points_a"], common["fantasy_points_b"], check_names=False
    )


def test_project_week_still_projects_an_unplayed_week(synthetic_panel):
    """Not scorable must not mean not projectable - that is the whole point."""
    panel = _panel_with_unplayed_tail(synthetic_panel)
    out = project_week(panel, SeasonToDateMean(), season=2021, week=7)
    assert len(out) > 0
    assert out["prediction"].notna().all()


def test_week_one_stays_in_history(synthetic_panel):
    """Week 1 is never scored but always learned from.

    Filtering history by `scorable` instead of `week_complete` silently drops
    it, which measurably degrades every week-2 projection. An earlier version
    of the unplayed-week fix did exactly that, and it only surfaced because a
    previously computed aggregate moved.
    """
    recorder = _HistoryWeeks()
    run_backtest(synthetic_panel, [recorder], config=BacktestConfig(seasons=(2021,)))
    assert (2021, 1) in recorder.weeks_seen, "week 1 results must reach the predictor"


def test_unplayed_weeks_stay_out_of_history_but_week_one_does_not(synthetic_panel):
    panel = _panel_with_unplayed_tail(synthetic_panel)
    recorder = _HistoryWeeks()
    run_backtest(panel, [recorder], config=BacktestConfig(seasons=(2021,)))
    assert (2021, 1) in recorder.weeks_seen
    assert (2021, 7) not in recorder.weeks_seen
    assert (2021, 8) not in recorder.weeks_seen
    # The prior season's weeks 7-8 are settled and must still be learned from.
    assert (2020, 7) in recorder.weeks_seen
