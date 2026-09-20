"""Turn scorecard metrics into plain language and a recommended action.

The translations are computed from the numbers, not written next to them. If a
future model changes calibration from 0.83 to 1.01, the page stops saying
"shade the extremes toward the middle" on its own. Hardcoded prose beside a
live metric is a lie waiting to happen.

Every band below is a judgement call, so each one is a named constant with its
reasoning attached rather than a magic number buried in an ``if``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


class Confidence(StrEnum):
    """How much weight a reader should put on this part of the system."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass(frozen=True, slots=True)
class Reading:
    """One metric, translated.

    Attributes:
        metric: The scorecard column this came from.
        value: The raw value, kept so the page can show both.
        label: Human name for the metric.
        plain: What the number means, in words a non-specialist can act on.
        action: What to actually do about it.
        confidence: Band the value falls in.
    """

    metric: str
    value: float
    label: str
    plain: str
    action: str
    confidence: Confidence


# ---------------------------------------------------------------- bands

#: Share of the predicted starter tier that finished in the real starter tier.
#: A constant-per-position predictor scores about 0.25 on our panel, which is
#: the effective random floor; 0.55+ is close to the practical ceiling for
#: weekly fantasy, where single-game variance dominates.
TOP_N_STRONG: Final = 0.55
TOP_N_MODERATE: Final = 0.45

#: Mean within position-week rank correlation. This is the metric that maps
#: onto the actual decision - the order you start players in.
SPEARMAN_STRONG: Final = 0.60
SPEARMAN_MODERATE: Final = 0.45

#: Distance from a calibration slope of 1.0 that we treat as well-behaved.
CALIBRATION_TIGHT: Final = 0.10
CALIBRATION_LOOSE: Final = 0.25

#: Mean signed error, in points, beyond which a systematic lean is worth
#: mentioning to a reader rather than filing as noise.
BIAS_NOTABLE: Final = 0.5

#: Gap in rank correlation between early and later weeks below which the two
#: are treated as equivalent rather than worth warning a reader about.
PHASE_GAP_NOTABLE: Final = 0.02


def _band(value: float, strong: float, moderate: float) -> Confidence:
    if value >= strong:
        return Confidence.STRONG
    if value >= moderate:
        return Confidence.MODERATE
    return Confidence.WEAK


# ---------------------------------------------------------------- readings


def read_top_n(value: float) -> Reading:
    """Translate the starter-tier hit rate."""
    pct = round(value * 100)
    confidence = _band(value, TOP_N_STRONG, TOP_N_MODERATE)

    plain = (
        f"Of the players this ranked as startable, about {pct}% actually "
        f"finished startable that week."
    )
    if confidence is Confidence.WEAK:
        action = (
            "That is close to guessing. Treat this ranking as a starting point "
            "and lean on matchup and your own read."
        )
    else:
        action = (
            "Roughly half your borderline calls will be wrong no matter what, so "
            "when two players project within a point of each other, pick the one "
            "with more upside and stop deliberating."
        )
    return Reading("top_n_hit_rate", value, "Start/sit accuracy", plain, action, confidence)


def read_spearman(value: float) -> Reading:
    """Translate the within position-week rank correlation."""
    confidence = _band(value, SPEARMAN_STRONG, SPEARMAN_MODERATE)
    plain = (
        "Measures whether players at a position are put in the right order, "
        "which is the decision you are actually making. 1.0 is perfect, 0 is a "
        f"coin flip. This system scores {value:.2f}."
    )
    action = {
        Confidence.STRONG: (
            "The ordering is reliable. Follow it for anything but the closest calls."
        ),
        Confidence.MODERATE: (
            "The broad tiers are trustworthy; the ordering within a tier is not. "
            "Use it to pick your tier, not your exact starter."
        ),
        Confidence.WEAK: (
            "The ordering carries little information here. Use it as one input among several."
        ),
    }[confidence]
    return Reading("spearman", value, "Ranking quality", plain, action, confidence)


def read_calibration(value: float) -> Reading:
    """Translate the calibration slope.

    Below 1.0 the projections are over-dispersed: they spread wider than reality,
    so the biggest numbers are optimistic and the smallest are pessimistic. This
    is the single most actionable metric on the page, because the correction is
    something a reader can apply by eye.
    """
    distance = abs(value - 1.0)
    if distance <= CALIBRATION_TIGHT:
        confidence = Confidence.STRONG
    elif distance <= CALIBRATION_LOOSE:
        confidence = Confidence.MODERATE
    else:
        confidence = Confidence.WEAK

    if value < 1.0 - CALIBRATION_TIGHT:
        shortfall = round((1.0 - value) * 100)
        plain = (
            f"A one-point rise in the projection is worth about {value:.2f} points "
            f"in reality, so the spread between players is overstated by roughly {shortfall}%."
        )
        action = (
            "Shade the extremes toward the middle. The highest projections are "
            "a little optimistic and the lowest are a little pessimistic; the gap "
            "between your best and worst option is smaller than it looks."
        )
    elif value > 1.0 + CALIBRATION_TIGHT:
        plain = (
            f"A one-point rise in the projection is worth about {value:.2f} points "
            "in reality, so the spread between players is understated."
        )
        action = (
            "Differences between players are larger than the projections suggest. "
            "Favour the higher projection more strongly than the gap implies."
        )
    else:
        plain = (
            "A one-point rise in the projection is worth about one point in "
            "reality. The scale is honest."
        )
        action = "Take the numbers at face value; no mental correction needed."

    return Reading("calibration_slope", value, "Scale honesty", plain, action, confidence)


def read_bias(value: float) -> Reading:
    """Translate the mean signed error.

    Positive means the system runs low, since bias is ``mean(actual - predicted)``.
    """
    magnitude = abs(value)
    confidence = Confidence.STRONG if magnitude < BIAS_NOTABLE else Confidence.MODERATE

    if magnitude < BIAS_NOTABLE:
        plain = (
            f"Across every projection, the average miss nets out to {value:+.2f} points. "
            "There is no meaningful systematic lean."
        )
        action = "Nothing to correct for."
    elif value < 0:
        plain = f"Projections run about {magnitude:.1f} points high per player on average."
        action = (
            f"Mentally subtract roughly {magnitude:.1f} points before comparing to a target score."
        )
    else:
        plain = f"Projections run about {magnitude:.1f} points low per player on average."
        action = f"Mentally add roughly {magnitude:.1f} points before comparing to a target score."

    return Reading("bias", value, "Systematic lean", plain, action, confidence)


def read_mae(value: float, *, baseline: float | None = None) -> Reading:
    """Translate mean absolute error, in terms a reader can feel."""
    plain = (
        f"A typical projection misses by about {value:.1f} fantasy points. "
        "Weekly football is genuinely this noisy - a single touchdown is six points "
        "and nobody predicts those."
    )
    if baseline is not None and baseline > 0:
        skill = (baseline - value) / baseline
        pct = round(abs(skill) * 100, 1)
        direction = "better than" if skill > 0 else "worse than"
        plain += f" That is {pct}% {direction} the named baseline."

    action = (
        "Do not read a projection as a prediction of the actual score. Read it as "
        "the middle of a wide range."
    )
    # MAE alone is not a quality verdict: it depends entirely on which players
    # were scored, so it is reported without a confidence band.
    return Reading("mae", value, "Typical miss", plain, action, Confidence.MODERATE)


# ---------------------------------------------------------------- slices


@dataclass(frozen=True, slots=True)
class GroupReading:
    """A per-position or per-phase comparison."""

    key: str
    spearman: float
    top_n_hit_rate: float
    confidence: Confidence


def read_positions(rows: Sequence[Mapping[str, Any]]) -> list[GroupReading]:
    """Rank positions by how trustworthy the ordering is.

    Cross-position MAE comparison is meaningless - quarterbacks score more, so
    they miss by more - which is why this uses rank quality instead.
    """
    out = [
        GroupReading(
            key=str(row["position"]),
            spearman=float(row["spearman"]),
            top_n_hit_rate=float(row["top_n_hit_rate"]),
            confidence=_band(float(row["spearman"]), SPEARMAN_STRONG, SPEARMAN_MODERATE),
        )
        for row in rows
    ]
    return sorted(out, key=lambda g: g.spearman, reverse=True)


def position_guidance(groups: Sequence[GroupReading]) -> str:
    """One sentence naming where to trust the rankings and where not to."""
    if not groups:
        return "Not enough data to compare positions."
    best, worst = groups[0], groups[-1]
    if best.key == worst.key:
        return f"Only {best.key} has been scored so far."
    return (
        f"The {best.key} rankings are the most reliable ({best.spearman:.2f}) and "
        f"{worst.key} the least ({worst.spearman:.2f}). If you are streaming any "
        f"position on matchup rather than projection, make it {worst.key}."
    )


def phase_guidance(early_spearman: float, late_spearman: float) -> str:
    """One sentence on how much to discount early-season projections."""
    gap = late_spearman - early_spearman
    if gap <= PHASE_GAP_NOTABLE:
        return (
            "Early-season projections hold up about as well as later ones, which is "
            "unusual and worth a second look."
        )
    return (
        f"Weeks 2-4 are measurably weaker ({early_spearman:.2f} versus "
        f"{late_spearman:.2f} from week 5 on), because there is barely any "
        "current-season form to go on yet. Lean more on your preseason read in September."
    )


def availability_note(played_rate: float) -> str:
    """Explain that the projection already prices in the chance of not playing."""
    miss = round((1 - played_rate) * 100)
    return (
        f"About {miss}% of the players projected here do not end up playing in a given "
        "week - injuries, benchings, late scratches. Those weeks are scored as zero "
        "rather than quietly dropped, so a projection is the expected value including "
        "the chance he sits, not what he scores if he suits up."
    )
