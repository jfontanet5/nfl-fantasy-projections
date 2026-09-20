"""Walk-forward backtest.

One loop, strictly ordered by (season, week). For each target week the harness
slices history itself, strips every outcome column from the target frame, and
only then calls the predictor. A predictor cannot see the future because it is
never handed the future - see :mod:`nflproj.predictors.base`.

The loop is deliberately not parallel. Weeks are cheap, and a sequential loop
makes the temporal ordering obvious to anyone auditing the code, which is worth
more here than wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nflproj.features.panel import FIRST_PROJECTABLE_WEEK, PANEL_KEY
from nflproj.logging import get_logger
from nflproj.predictors.base import Predictor, strip_outcomes

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Bounds of a walk-forward run."""

    #: Weeks before this many prior games exist in the target season are still
    #: projected - they are the hardest and most honest part of the problem -
    #: but the scorecard reports them separately so early-season noise does not
    #: hide in a season average.
    min_week: int = FIRST_PROJECTABLE_WEEK
    seasons: tuple[int, ...] | None = None


def _target_weeks(panel: pd.DataFrame, config: BacktestConfig) -> list[tuple[int, int]]:
    scorable = panel[panel["projectable"]]
    if config.seasons is not None:
        scorable = scorable[scorable["season"].isin(config.seasons)]
    scorable = scorable[scorable["week"] >= config.min_week]
    weeks = scorable[["season", "week"]].drop_duplicates()
    weeks = weeks.sort_values(["season", "week"], kind="mergesort")
    return [(int(s), int(w)) for s, w in weeks.itertuples(index=False, name=None)]


def run_backtest(
    panel: pd.DataFrame,
    predictors: Sequence[Predictor],
    *,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Produce one prediction per (predictor, projectable player-week).

    Args:
        panel: Output of :func:`nflproj.features.panel.build_panel`.
        predictors: Predictors to run. Names must be unique.
        config: Run bounds.

    Returns:
        Long-format frame: the panel key, the realised target, the predictor
        name and its projection. Long rather than wide so that adding a
        predictor never changes the schema of the scorecard.

    Raises:
        ValueError: On duplicate predictor names, or if a predictor returns a
            series that does not align with the targets it was given.
    """
    config = config or BacktestConfig()
    names = [p.name for p in predictors]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        msg = f"predictor names must be unique; duplicated: {dupes}"
        raise ValueError(msg)

    panel = panel.sort_values(["season", "week"], kind="mergesort")
    targets_by_week = dict(tuple(panel.groupby(["season", "week"], observed=True)))

    rows: list[pd.DataFrame] = []
    for season, week in _target_weeks(panel, config):
        # Everything that had already happened when this week kicked off.
        history = panel[
            (panel["season"] < season) | ((panel["season"] == season) & (panel["week"] < week))
        ]
        target_rows = targets_by_week[season, week]
        target_rows = target_rows[target_rows["projectable"]]
        if target_rows.empty:
            continue

        blinded = strip_outcomes(target_rows)

        for predictor in predictors:
            predictor.fit(history)
            preds = predictor.predict(blinded)
            if not preds.index.equals(blinded.index):
                msg = (
                    f"{predictor.name!r} returned predictions whose index does not match "
                    f"the {len(blinded)} targets it was given for {season} week {week}"
                )
                raise ValueError(msg)
            rows.append(
                pd.DataFrame(
                    {
                        "season": target_rows["season"].to_numpy(),
                        "week": target_rows["week"].to_numpy(),
                        "player_id": target_rows["player_id"].to_numpy(),
                        "position": target_rows["position"].to_numpy(),
                        "played": target_rows["played"].to_numpy(),
                        "fantasy_points": target_rows["fantasy_points"].to_numpy(),
                        "predictor": predictor.name,
                        "prediction": preds.to_numpy(dtype="float64"),
                    }
                )
            )

        log.debug("backtest.week_done", season=season, week=week, n_targets=len(target_rows))

    if not rows:
        msg = "backtest produced no predictions; check the season and week bounds"
        raise ValueError(msg)

    out = pd.concat(rows, ignore_index=True)
    log.info(
        "backtest.complete",
        predictions=len(out),
        predictors=len(predictors),
        weeks=int(out.groupby(["season", "week"], observed=True).ngroups),
    )
    return out


def project_week(
    panel: pd.DataFrame,
    predictor: Predictor,
    *,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Project a single upcoming week, using only data before it.

    The live-publication path. Identical slicing to :func:`run_backtest`, which
    is the point: the weekly job cannot drift away from what the backtest
    measured, because it is the same history rule applied by the same code.
    """
    history = panel[
        (panel["season"] < season) | ((panel["season"] == season) & (panel["week"] < week))
    ]
    targets = panel[(panel["season"] == season) & (panel["week"] == week) & panel["projectable"]]
    if targets.empty:
        msg = f"no projectable universe rows for {season} week {week}"
        raise ValueError(msg)

    predictor.fit(history)
    preds = predictor.predict(strip_outcomes(targets))

    out = targets[[*PANEL_KEY, "player_display_name", "position", "team", "opponent_team"]].copy()
    out["predictor"] = predictor.name
    out["prediction"] = preds.to_numpy(dtype="float64")
    return out.sort_values("prediction", ascending=False).reset_index(drop=True)


def available_seasons(panel: pd.DataFrame) -> list[int]:
    return sorted(int(s) for s in panel["season"].unique())


def iter_weeks(panel: pd.DataFrame, seasons: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """Convenience wrapper over the harness's own week enumeration."""
    cfg = BacktestConfig(seasons=tuple(seasons) if seasons is not None else None)
    return _target_weeks(panel, cfg)
