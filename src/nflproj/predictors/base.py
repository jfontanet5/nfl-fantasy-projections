"""The predictor contract.

Leakage is prevented structurally rather than by convention. A predictor never
receives the week it is projecting: the backtest hands it ``history`` (panel
rows strictly before the target week) and ``targets`` (the target week's
universe with every outcome column removed). There is no code path by which a
predictor can read the answer, so "did we leak?" stops being a question about
discipline and becomes a question about this one function - which is tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

#: Columns that only exist after kickoff. Stripped from every target frame
#: before a predictor sees it.
OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "fantasy_points",
        "played",
        "targets",
        "carries",
        "attempts",
        "result",
    }
)


@runtime_checkable
class Predictor(Protocol):
    """Anything that can turn a history plus a target week into projections.

    Implementations must be deterministic given the same inputs, because the
    scorecard is a reproducibility claim as much as an accuracy one.
    """

    @property
    def name(self) -> str:
        """Stable identifier used as a column name and scorecard row label."""
        ...

    def fit(self, history: pd.DataFrame) -> None:
        """Absorb everything known before the target week.

        Called once per target week with an expanding history. Implementations
        should not retain references to ``history`` beyond what they need.
        """
        ...

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        """Project fantasy points for each row of ``targets``.

        Args:
            targets: Target-week universe rows with :data:`OUTCOME_COLUMNS`
                removed. Index is meaningful and must be preserved.

        Returns:
            Float projections aligned to ``targets.index``. Nulls are not
            permitted; a predictor with no opinion must say so numerically.
        """
        ...


def strip_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove post-kickoff columns from a frame handed to a predictor."""
    drop = [c for c in frame.columns if c in OUTCOME_COLUMNS]
    return frame.drop(columns=drop)
