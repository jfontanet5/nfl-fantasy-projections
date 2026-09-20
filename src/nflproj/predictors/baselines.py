"""Named baselines.

A projection system with no baseline is a number with no meaning. These are the
things a competent human does in their head, and beating them is the minimum
bar the model has to clear before anything else about it is interesting.

The designated headline baseline is :class:`SeasonToDateMean` with
``include_dnp=True`` - "what is he averaging this season", counting weeks he
missed as zero. Every scorecard reports skill relative to it.

The ``include_dnp`` flag is the interesting knob. Averaging only games a player
actually played answers "how good is he when he plays", which is a different and
systematically higher number than the panel's target, since the panel scores
inactive players as zero. Both are implemented so the backtest can settle which
is the better estimator instead of us asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from nflproj.predictors.base import Predictor

#: Fallback used before any history exists at all (season opener weeks for a
#: position that somehow has no prior rows). Deliberately not zero: a predictor
#: that returns all-zeros would score suspiciously well on a panel that is 36%
#: zeros, and we do not want that to look like skill.
_GLOBAL_PRIOR = 6.0


def _history_for_averaging(history: pd.DataFrame, *, include_dnp: bool) -> pd.DataFrame:
    return history if include_dnp else history[history["played"]]


def _position_means(history: pd.DataFrame, *, include_dnp: bool) -> pd.Series:
    frame = _history_for_averaging(history, include_dnp=include_dnp)
    if frame.empty:
        return pd.Series(dtype="float64")
    return frame.groupby("position", observed=True)["fantasy_points"].mean()


def _apply_shrinkage(
    player_mean: pd.Series,
    player_count: pd.Series,
    prior: pd.Series,
    shrinkage_games: float,
) -> pd.Series:
    """James-Stein style shrinkage toward a positional prior.

    With ``k`` prior games of pseudo-weight, a player with ``n`` observed games
    gets ``n/(n+k)`` of their own average and the rest of the position's. This
    matters most in weeks 2-4, where an unshrunk season average is one game of
    noise.
    """
    if shrinkage_games <= 0:
        return player_mean
    weight = player_count / (player_count + shrinkage_games)
    return weight * player_mean + (1.0 - weight) * prior


@dataclass
class _PlayerAverageBase:
    """Shared machinery for per-player historical averages."""

    #: When true, ``fit`` discards history from earlier seasons. The harness
    #: passes every prior week it has, across seasons; a season-to-date average
    #: must reset in September, a rolling average should not.
    season_scoped: ClassVar[bool] = False

    include_dnp: bool = True
    shrinkage_games: float = 0.0
    _player_stat: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"), init=False)
    _player_count: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"), init=False)
    _position_prior: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64"), init=False
    )

    def _summarise(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        raise NotImplementedError

    def fit(self, history: pd.DataFrame) -> None:
        if self.season_scoped and not history.empty:
            # The target season is the latest one present: the harness only
            # projects weeks >= 2, so the target season always contributes at
            # least its week-1 rows to the history it hands us.
            history = history[history["season"] == history["season"].max()]
        self._position_prior = _position_means(history, include_dnp=self.include_dnp)
        if history.empty:
            self._player_stat = pd.Series(dtype="float64")
            self._player_count = pd.Series(dtype="float64")
            return
        self._player_stat, self._player_count = self._summarise(
            _history_for_averaging(history, include_dnp=self.include_dnp)
        )

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        prior = (
            targets["position"].map(self._position_prior).astype("float64")
            if not self._position_prior.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        prior = prior.fillna(_GLOBAL_PRIOR)

        own = targets["player_id"].map(self._player_stat).astype("float64")
        count = targets["player_id"].map(self._player_count).astype("float64").fillna(0.0)

        shrunk = _apply_shrinkage(own.fillna(prior), count, prior, self.shrinkage_games)
        # Players with no history at all fall back to the positional prior.
        out = shrunk.where(own.notna(), prior)
        return out.astype("float64").rename("prediction")


@dataclass
class SeasonToDateMean(_PlayerAverageBase):
    """Mean fantasy points across the player's prior games this season.

    The headline baseline. Resets every season, which is the right behaviour:
    it is what a person means by "he's averaging 14 a game".
    """

    season_scoped: ClassVar[bool] = True

    @property
    def name(self) -> str:
        parts = ["season_to_date_mean"]
        if not self.include_dnp:
            parts.append("played_only")
        if self.shrinkage_games:
            parts.append(f"shrunk{self.shrinkage_games:g}")
        return "_".join(parts)

    def _summarise(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        # ``season_scoped`` has already trimmed history to the target season.
        grouped = history.groupby("player_id", observed=True)["fantasy_points"]
        return grouped.mean(), grouped.size().astype("float64")


@dataclass
class RollingMean(_PlayerAverageBase):
    """Mean over the player's most recent ``window`` games."""

    window: int = 4

    @property
    def name(self) -> str:
        parts = [f"rolling_mean_{self.window}"]
        if not self.include_dnp:
            parts.append("played_only")
        if self.shrinkage_games:
            parts.append(f"shrunk{self.shrinkage_games:g}")
        return "_".join(parts)

    def _summarise(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        recent = history.sort_values(["season", "week"], kind="mergesort").groupby(
            "player_id", observed=True
        )
        stat = recent["fantasy_points"].apply(lambda s: s.tail(self.window).mean())
        count = recent["fantasy_points"].apply(lambda s: float(len(s.tail(self.window))))
        return stat, count


@dataclass
class ExponentialMean(_PlayerAverageBase):
    """Exponentially weighted mean, recent games weighted more heavily."""

    halflife: float = 3.0

    @property
    def name(self) -> str:
        parts = [f"ewma_hl{self.halflife:g}"]
        if not self.include_dnp:
            parts.append("played_only")
        if self.shrinkage_games:
            parts.append(f"shrunk{self.shrinkage_games:g}")
        return "_".join(parts)

    def _summarise(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        ordered = history.sort_values(["season", "week"], kind="mergesort")
        grouped = ordered.groupby("player_id", observed=True)["fantasy_points"]
        stat = grouped.apply(lambda s: s.ewm(halflife=self.halflife).mean().iloc[-1])
        return stat, grouped.size().astype("float64")


@dataclass
class LastGame(_PlayerAverageBase):
    """The player's most recent result. The naive persistence forecast."""

    @property
    def name(self) -> str:
        return "last_game" if self.include_dnp else "last_game_played_only"

    def _summarise(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        ordered = history.sort_values(["season", "week"], kind="mergesort")
        grouped = ordered.groupby("player_id", observed=True)["fantasy_points"]
        return grouped.last(), grouped.size().astype("float64")


@dataclass
class PositionMean:
    """Every player at a position gets that position's average.

    The floor. Any predictor that cannot beat this has learned nothing about
    individual players.
    """

    include_dnp: bool = True
    _position_prior: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64"), init=False
    )

    @property
    def name(self) -> str:
        return "position_mean" if self.include_dnp else "position_mean_played_only"

    def fit(self, history: pd.DataFrame) -> None:
        self._position_prior = _position_means(history, include_dnp=self.include_dnp)

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        out = targets["position"].map(self._position_prior).astype("float64")
        return out.fillna(_GLOBAL_PRIOR).rename("prediction")


#: The baseline every scorecard is scored against.
HEADLINE_BASELINE_NAME = "season_to_date_mean"

#: The predictor whose numbers the public page shows. Distinct from the
#: baseline on purpose: we publish our best estimator and report its skill
#: against the baseline, rather than publishing the thing we are measured by.
PUBLISHED_PREDICTOR_NAME = "ewma_hl3"


def default_baselines() -> list[Predictor]:
    """The baseline slate carried on every scorecard."""
    return [
        PositionMean(),
        LastGame(),
        SeasonToDateMean(),
        SeasonToDateMean(include_dnp=False),
        SeasonToDateMean(shrinkage_games=2.0),
        RollingMean(window=4),
        ExponentialMean(halflife=3.0),
    ]
