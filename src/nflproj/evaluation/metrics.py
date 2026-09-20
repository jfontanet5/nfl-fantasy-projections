"""Evaluation metrics.

Point-error metrics (MAE, RMSE) answer "how close is the number". Fantasy
decisions are mostly ordering decisions - who do I start - so ranking metrics
are reported alongside and are arguably the ones that matter. A projection can
have a worse MAE and still be strictly more useful if it orders players better.

Every ranking metric is computed *within* a (season, week, position) group and
then averaged over groups. Pooling across positions would mostly measure the
fact that QBs outscore TEs, which no one needs a model for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

GROUP_COLUMNS: Final[list[str]] = ["season", "week", "position"]

#: Starter-sized cutoffs per position for top-N hit rate: roughly the number of
#: players at each position started in a 12-team league each week.
DEFAULT_TOP_N: Final[dict[str, int]] = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

#: A slope or rank correlation needs at least two distinct observations.
_MIN_OBSERVATIONS: Final = 2


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Metrics for one predictor over one slice of the panel."""

    predictor: str
    n_rows: int
    n_weeks: int
    mae: float
    rmse: float
    bias: float
    spearman: float
    top_n_hit_rate: float
    calibration_slope: float
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str | int]:
        base: dict[str, float | str | int] = {
            "predictor": self.predictor,
            "n_rows": self.n_rows,
            "n_weeks": self.n_weeks,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "spearman": self.spearman,
            "top_n_hit_rate": self.top_n_hit_rate,
            "calibration_slope": self.calibration_slope,
        }
        base.update(self.extras)
        return base


def mae(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - pred.to_numpy())))


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - pred.to_numpy()) ** 2)))


def bias(actual: pd.Series, pred: pd.Series) -> float:
    """Mean signed error. Positive means the predictor runs low."""
    return float(np.mean(actual.to_numpy() - pred.to_numpy()))


def calibration_slope(actual: pd.Series, pred: pd.Series) -> float:
    """OLS slope of actual on predicted.

    1.0 means a one-point rise in the projection buys one point of realised
    scoring on average. Below 1.0 means the projections are over-dispersed -
    the classic failure of a model that chases weekly spikes.
    """
    x = pred.to_numpy(dtype="float64")
    y = actual.to_numpy(dtype="float64")
    var = float(np.var(x))
    if var == 0.0 or len(x) < _MIN_OBSERVATIONS:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / var)


def _spearman(actual: np.ndarray, pred: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    if len(actual) < _MIN_OBSERVATIONS:
        return float("nan")
    ar = pd.Series(actual).rank().to_numpy()
    pr = pd.Series(pred).rank().to_numpy()
    if np.std(ar) == 0 or np.std(pr) == 0:
        return float("nan")
    return float(np.corrcoef(ar, pr)[0, 1])


def grouped_spearman(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    group_cols: Sequence[str] = tuple(GROUP_COLUMNS),
    min_group_size: int = 5,
) -> float:
    """Mean within-group Spearman correlation.

    Groups smaller than ``min_group_size`` are dropped: a rank correlation over
    three players is noise, and averaging it in would make the metric jumpy for
    reasons that have nothing to do with the predictor.
    """
    scores: list[float] = []
    for _, grp in frame.groupby(list(group_cols), observed=True):
        if len(grp) < min_group_size:
            continue
        rho = _spearman(grp[actual_col].to_numpy(), grp[pred_col].to_numpy())
        if not np.isnan(rho):
            scores.append(rho)
    return float(np.mean(scores)) if scores else float("nan")


def top_n_hit_rate(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    top_n: dict[str, int] | None = None,
) -> float:
    """Fraction of the predicted top-N at a position that landed in the actual top-N.

    This is the closest single number to "did the projection help me set a
    lineup". Ties in the actual scores are broken arbitrarily but consistently,
    which is acceptable because ties at the cutoff are rare.
    """
    top_n = top_n or DEFAULT_TOP_N
    hits: list[float] = []
    for (_, _, position), grp in frame.groupby(GROUP_COLUMNS, observed=True):
        n = top_n.get(str(position))
        if n is None or len(grp) < n:
            continue
        predicted = set(grp.nlargest(n, pred_col).index)
        realised = set(grp.nlargest(n, actual_col).index)
        hits.append(len(predicted & realised) / n)
    return float(np.mean(hits)) if hits else float("nan")


def evaluate(
    frame: pd.DataFrame,
    *,
    predictor: str,
    actual_col: str = "fantasy_points",
    pred_col: str = "prediction",
    top_n: dict[str, int] | None = None,
) -> MetricResult:
    """Compute the full metric set for one predictor over ``frame``."""
    if frame.empty:
        msg = f"cannot evaluate {predictor!r} over an empty frame"
        raise ValueError(msg)
    if frame[pred_col].isna().any():
        n_null = int(frame[pred_col].isna().sum())
        msg = f"{predictor!r} produced {n_null} null predictions; a predictor must commit"
        raise ValueError(msg)

    actual = frame[actual_col]
    pred = frame[pred_col]
    return MetricResult(
        predictor=predictor,
        n_rows=len(frame),
        n_weeks=int(frame.groupby(["season", "week"], observed=True).ngroups),
        mae=mae(actual, pred),
        rmse=rmse(actual, pred),
        bias=bias(actual, pred),
        spearman=grouped_spearman(frame, actual_col=actual_col, pred_col=pred_col),
        top_n_hit_rate=top_n_hit_rate(frame, actual_col=actual_col, pred_col=pred_col, top_n=top_n),
        calibration_slope=calibration_slope(actual, pred),
    )


def skill_score(value: float, baseline: float, *, lower_is_better: bool = True) -> float:
    """Fractional improvement over a baseline.

    Positive is better for both metric directions, so a skill column can be read
    the same way whether it came from MAE or Spearman.
    """
    if baseline == 0 or np.isnan(baseline) or np.isnan(value):
        return float("nan")
    if lower_is_better:
        return float((baseline - value) / baseline)
    return float((value - baseline) / abs(baseline))
