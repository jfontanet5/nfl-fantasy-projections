"""Fantasy scoring rules.

We compute the target ourselves from box-score components rather than trusting a
precomputed ``fantasy_points_ppr`` column. Two reasons:

1. The scoring rule becomes an explicit, testable object instead of an upstream
   assumption, so half-PPR or a custom league is a parameter, not a rewrite.
2. Recomputing gives us a free consistency check against upstream (see
   ``tests/integration/test_scoring_matches_nflverse.py``). If our number and
   theirs diverge, one of us changed and we find out immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class ScoringRules:
    """Points awarded per box-score event.

    Defaults are full-PPR, the most common public format and the one nflverse's
    own ``fantasy_points_ppr`` uses, which keeps the cross-check meaningful.
    """

    passing_yards: float = 0.04
    passing_tds: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    receptions: float = 1.0
    fumbles_lost: float = -2.0
    two_point_conversions: float = 2.0
    #: Kick and punt return touchdowns. Rare (13 of 6,141 skill-position lines in
    #: 2024) but real points, and omitting them is the single discrepancy that
    #: shows up against nflverse's own PPR column.
    special_teams_tds: float = 6.0

    @classmethod
    def ppr(cls) -> ScoringRules:
        return cls(receptions=1.0)

    @classmethod
    def half_ppr(cls) -> ScoringRules:
        return cls(receptions=0.5)

    @classmethod
    def standard(cls) -> ScoringRules:
        return cls(receptions=0.0)


PPR: Final = ScoringRules.ppr()

#: Columns consumed by :func:`compute_fantasy_points`. Missing columns are an
#: error, not a silent zero - a renamed upstream column must fail loudly.
REQUIRED_STAT_COLUMNS: Final[tuple[str, ...]] = (
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "special_teams_tds",
)


def compute_fantasy_points(stats: pd.DataFrame, rules: ScoringRules = PPR) -> pd.Series:
    """Score a frame of weekly box-score rows.

    Args:
        stats: One row per player-week, carrying :data:`REQUIRED_STAT_COLUMNS`.
            Nulls are treated as zero, which is correct here: a null receiving
            yard count means the player had no receiving involvement.
        rules: Scoring configuration.

    Returns:
        A float Series of fantasy points aligned to ``stats.index``.

    Raises:
        KeyError: If any required column is absent.
    """
    missing = [c for c in REQUIRED_STAT_COLUMNS if c not in stats.columns]
    if missing:
        msg = f"stats frame is missing required scoring columns: {missing}"
        raise KeyError(msg)

    s = stats[list(REQUIRED_STAT_COLUMNS)].fillna(0.0).astype("float64")

    fumbles_lost = s["rushing_fumbles_lost"] + s["receiving_fumbles_lost"] + s["sack_fumbles_lost"]
    two_point = (
        s["passing_2pt_conversions"] + s["rushing_2pt_conversions"] + s["receiving_2pt_conversions"]
    )

    points = (
        s["passing_yards"] * rules.passing_yards
        + s["passing_tds"] * rules.passing_tds
        + s["passing_interceptions"] * rules.interceptions
        + s["rushing_yards"] * rules.rushing_yards
        + s["rushing_tds"] * rules.rushing_tds
        + s["receiving_yards"] * rules.receiving_yards
        + s["receiving_tds"] * rules.receiving_tds
        + s["receptions"] * rules.receptions
        + fumbles_lost * rules.fumbles_lost
        + two_point * rules.two_point_conversions
        + s["special_teams_tds"] * rules.special_teams_tds
    )

    return points.rename("fantasy_points").astype("float64")
