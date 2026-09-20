"""The scorecard: the thing this project actually ships.

A scorecard is a signed statement of how every predictor did, against a named
baseline, over a declared universe, on a declared data snapshot. It is written
to ``reports/`` and committed, so the track record accumulates in public and
cannot be quietly revised.

Three slices are always reported, because each one can flatter or damn a
predictor on its own:

overall
    Every projectable player-week under the active universe policy.
by_position
    QB/RB/WR/TE separately. A predictor can win overall purely by being good at
    the position with the most rows.
by_season_phase
    Weeks 2-4 against weeks 5+. Early-season projections have almost no
    within-season history and are where naive baselines are weakest; averaging
    them into a season number hides the hardest part of the problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from nflproj.evaluation.metrics import MetricResult, evaluate, skill_score
from nflproj.features.panel import UniversePolicy
from nflproj.logging import get_logger
from nflproj.predictors.baselines import HEADLINE_BASELINE_NAME

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

#: Week at which "early season" ends. Before this, a season-to-date average is
#: built on three games or fewer.
EARLY_SEASON_THROUGH_WEEK: Final = 4

SCORECARD_SCHEMA_VERSION: Final = "1"


@dataclass(frozen=True, slots=True)
class Scorecard:
    """A complete evaluation run."""

    generated_at: str
    schema_version: str
    universe: str
    baseline: str
    seasons: list[int]
    n_predictions: int
    slices: dict[str, pd.DataFrame]
    provenance: dict[str, Any]

    def to_json(self) -> str:
        payload = {
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "universe": self.universe,
            "baseline": self.baseline,
            "seasons": self.seasons,
            "n_predictions": self.n_predictions,
            "provenance": self.provenance,
            "slices": {
                name: frame.to_dict(orient="records") for name, frame in self.slices.items()
            },
        }
        return json.dumps(payload, indent=2, default=str) + "\n"

    def write(self, directory: Path, *, stem: str = "scorecard") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{stem}.json"
        json_path.write_text(self.to_json())
        (directory / f"{stem}.md").write_text(self.to_markdown())
        log.info("scorecard.written", path=str(json_path), slices=sorted(self.slices))
        return json_path

    def to_markdown(self) -> str:
        lines = [
            "# Projection scorecard",
            "",
            f"- Generated: `{self.generated_at}`",
            f"- Universe: `{self.universe}`",
            f"- Baseline: `{self.baseline}`",
            f"- Seasons: {self.seasons[0]}-{self.seasons[-1]}" if self.seasons else "- Seasons: -",
            f"- Predictions scored: {self.n_predictions:,}",
            "",
            "`mae_skill` is the fractional MAE improvement over the baseline; "
            "positive beats it. `spearman` is the mean within position-week rank "
            "correlation. `top_n_hit_rate` is the share of the predicted "
            "starter-tier that finished in the actual starter-tier.",
            "",
        ]
        for name, frame in self.slices.items():
            lines += [f"## {name}", "", frame.to_markdown(index=False, floatfmt=".4f"), ""]
        return "\n".join(lines)


def _metrics_frame(
    predictions: pd.DataFrame,
    *,
    baseline: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate every predictor, optionally within groups, and add skill columns."""
    group_cols = group_cols or []
    results: list[dict[str, Any]] = []

    chunks: list[tuple[tuple[Any, ...], pd.DataFrame]]
    if group_cols:
        chunks = [
            (key if isinstance(key, tuple) else (key,), chunk)
            for key, chunk in predictions.groupby(group_cols, observed=True)
        ]
    else:
        chunks = [((), predictions)]

    for key_tuple, chunk in chunks:
        per_predictor: dict[str, MetricResult] = {}
        for predictor, sub in chunk.groupby("predictor", observed=True):
            per_predictor[str(predictor)] = evaluate(sub, predictor=str(predictor))

        base = per_predictor.get(baseline)
        for _predictor, res in sorted(per_predictor.items()):
            row: dict[str, Any] = dict(zip(group_cols, key_tuple, strict=True))
            row.update(res.as_dict())
            if base is not None:
                row["mae_skill"] = skill_score(res.mae, base.mae)
                row["rmse_skill"] = skill_score(res.rmse, base.rmse)
                row["spearman_skill"] = skill_score(
                    res.spearman, base.spearman, lower_is_better=False
                )
            results.append(row)

    frame = pd.DataFrame(results)
    sort_cols = [*group_cols, "mae"]
    return frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_scorecard(
    predictions: pd.DataFrame,
    *,
    universe: UniversePolicy = UniversePolicy.ACTIVE_RECENT,
    baseline: str = HEADLINE_BASELINE_NAME,
    provenance: dict[str, Any] | None = None,
) -> Scorecard:
    """Aggregate backtest predictions into a scorecard.

    Args:
        predictions: Long-format output of :func:`nflproj.evaluation.backtest.run_backtest`.
        universe: Which population to score. ``PLAYED`` drops players who did
            not appear, which conditions on the outcome; it is reported for
            comparability and labelled as such.
        baseline: Predictor name that skill scores are computed against.
        provenance: Data-snapshot details recorded verbatim in the output.

    Raises:
        ValueError: If the baseline predictor is absent from ``predictions``.
    """
    available = set(predictions["predictor"].unique())
    if baseline not in available:
        msg = f"baseline {baseline!r} not among predictors {sorted(available)}"
        raise ValueError(msg)

    scored = (
        predictions
        if universe is UniversePolicy.ACTIVE_RECENT
        else predictions[predictions["played"]]
    )
    scored = scored.copy()
    scored["season_phase"] = (
        scored["week"].le(EARLY_SEASON_THROUGH_WEEK).map({True: "weeks_2_4", False: "weeks_5_plus"})
    )

    slices = {
        "overall": _metrics_frame(scored, baseline=baseline),
        "by_position": _metrics_frame(scored, baseline=baseline, group_cols=["position"]),
        "by_season_phase": _metrics_frame(scored, baseline=baseline, group_cols=["season_phase"]),
    }

    return Scorecard(
        generated_at=datetime.now(UTC).isoformat(),
        schema_version=SCORECARD_SCHEMA_VERSION,
        universe=str(universe),
        baseline=baseline,
        seasons=sorted(int(s) for s in predictions["season"].unique()),
        n_predictions=len(scored),
        slices=slices,
        provenance=provenance or {},
    )
