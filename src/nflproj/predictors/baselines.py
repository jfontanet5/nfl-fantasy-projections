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
#: Promoted from `ewma_hl3` once the season discount beat it on both MAE and
#: ranking in 10 of 11 seasons.
PUBLISHED_PREDICTOR_NAME = "season_decayed_hl3_d0.5"


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
        SeasonDecayedMean(),
        AvailabilityWeighted(),
    ]


@dataclass
class AvailabilityWeighted:
    """Project ``P(plays) x E[points | plays]`` instead of one blended average.

    **This loses to `ewma_hl3` and is kept as a published negative result.**

    The motivation was a real defect. A blended average folds absence into the
    scoring estimate, so a resolved injury suppresses a player for months: a
    back who missed three games last November still carries those zeros as
    recent news, and ranks below a teammate he has clearly overtaken. Splitting
    the two terms should let each use the evidence that bears on it - scoring
    from games he played, availability from how often he suits up.

    It does not work. Swept over windows of 4, 8 and 16 games and shrinkage
    priors of 1, 2 and 5, every configuration is worse than the blend:

    ========================  ======  ========  ===========
    predictor                    MAE  Spearman  calibration
    ========================  ======  ========  ===========
    ewma_hl3                  4.4166    0.5959       0.8464
    this, best (w4, p1)       4.4602    0.5855       0.8743
    this, worst (w16, p5)     4.5402    0.5665       0.8994
    ========================  ======  ========  ===========

    The sweep is monotone: shorter windows and weaker priors always do better,
    so the measured optimum is *less* availability adjustment. Extrapolating
    that trend arrives back at the blend.

    Why: a binary played/did-not-play indicator throws away information the
    blend keeps. In a continuous series a DNP is a 0, which carries both "was
    absent" and "contributed nothing", weighted by recency alongside every
    other week. Binarising it into a rate discards the magnitudes and adds
    estimator noise, and the noise costs more than the separation gains.

    It does improve calibration (0.847 -> 0.874, and up to 0.955 with a heavy
    prior) and halves the bias. That is the same trade shrinkage makes: a
    better-scaled number that orders players less well.

    The real conclusion is about the input, not the shape. Availability
    inferred from past play is backward-looking by construction - it cannot
    know the injury has resolved, which was the whole complaint. This
    decomposition is the right structure for injury data and the wrong thing
    to ship without it: when a point-in-time injury feed exists, only the
    availability term changes, and this becomes worth re-testing.

    Defaults are the best configuration found, so the published number is the
    idea's strongest form rather than an arbitrary one.
    """

    halflife: float = 3.0
    availability_window: int = 4
    #: Pseudo-observations of the league rate mixed into every player's
    #: availability. The sweep found less is better at every window, so this is
    #: one game: enough to keep a run of absences from projecting a literal
    #: zero, which no honest estimate of a rostered player should be.
    availability_prior_games: float = 1.0

    _scoring: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"), init=False)
    _availability: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"), init=False)
    _position_scoring: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64"), init=False
    )
    _league_availability: float = field(default=1.0, init=False)

    @property
    def name(self) -> str:
        return (
            f"availability_weighted_hl{self.halflife:g}"
            f"_w{self.availability_window}"
            f"_p{self.availability_prior_games:g}"
        )

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            self._scoring = pd.Series(dtype="float64")
            self._availability = pd.Series(dtype="float64")
            self._position_scoring = pd.Series(dtype="float64")
            self._league_availability = 1.0
            return

        ordered = history.sort_values(["season", "week"], kind="mergesort")

        # How good he is when he plays - played games only.
        played = ordered[ordered["played"]]
        if played.empty:
            self._scoring = pd.Series(dtype="float64")
            self._position_scoring = pd.Series(dtype="float64")
        else:
            grouped = played.groupby("player_id", observed=True)["fantasy_points"]
            self._scoring = grouped.apply(lambda s: s.ewm(halflife=self.halflife).mean().iloc[-1])
            self._position_scoring = played.groupby("position", observed=True)[
                "fantasy_points"
            ].mean()

        # How often he actually suits up, over recent universe weeks.
        self._league_availability = float(ordered["played"].mean())
        recent = ordered.groupby("player_id", observed=True)["played"]
        appeared = recent.apply(lambda s: float(s.tail(self.availability_window).sum()))
        opportunities = recent.apply(lambda s: float(len(s.tail(self.availability_window))))

        prior = self.availability_prior_games
        self._availability = (appeared + prior * self._league_availability) / (
            opportunities + prior
        )

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        position_fallback = (
            targets["position"].map(self._position_scoring).astype("float64")
            if not self._position_scoring.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        scoring = (
            targets["player_id"].map(self._scoring).astype("float64")
            if not self._scoring.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        scoring = scoring.fillna(position_fallback).fillna(_GLOBAL_PRIOR)

        availability = (
            targets["player_id"].map(self._availability).astype("float64")
            if not self._availability.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        availability = availability.fillna(self._league_availability)

        return (scoring * availability).astype("float64").rename("prediction")


@dataclass
class SeasonDecayedMean:
    """Recency-weighted mean that also discounts games from earlier seasons.

    :class:`ExponentialMean` walks a player's games in sequence and weights by
    position alone, so a week-18 game from last season carries 79% of the weight
    of this season's opener. Nothing in that model knows an offseason happened -
    but a great deal does: new team, new scheme, new role, a year older, a
    rookie ahead of him on the depth chart.

    So the weight on a game is two factors, not one::

        weight = 0.5 ** (games_ago / halflife) * season_decay ** (seasons_ago)

    ``season_decay=1.0`` applies no discount and reproduces
    :class:`ExponentialMean` exactly, which is asserted in the tests - it makes
    this a strict generalisation rather than a different estimator that happens
    to look similar. ``season_decay=0.0`` uses the current season only.

    The tension is real in both directions. Week 2 gives a player exactly one
    game of current-season evidence, and trusting it alone is trusting a single
    noisy observation; trusting last season equally is trusting a role he may no
    longer have. The right discount is an empirical question, which is what the
    harness is for - and the answer is an interior optimum:

    ==========  ======  ========  ===========
    decay          MAE  Spearman  calibration
    ==========  ======  ========  ===========
    1.0 (none)  4.4166    0.5959       0.8464
    0.8         4.4081    0.5979       0.8423
    **0.5**     4.3981    0.6000       0.8320
    0.3         4.3972    0.6006       0.8194
    0.15        4.4065    0.5996       0.8027
    0.0         4.5055    0.5866       0.7663
    ==========  ======  ========  ===========

    Both ends lose. No discount trusts a role the player may not have; a full
    discount throws away everything but one noisy game. 0.3 and 0.5 are
    indistinguishable overall, so 0.5 is published: it does better in weeks 2-4
    where the evidence is thinnest (MAE 4.8101 against 4.8174) and is the more
    conservative of the two.

    The gain is larger early in the season, which is what the reasoning
    predicts - +0.80% skill in weeks 2-4 against +0.42% overall.

    Robustness, because a parameter chosen on the same data it is scored on is
    a fair thing to distrust: 0.5 beats the undiscounted mean in 10 of 11
    seasons. The exception is 2015, the first season in the panel, where there
    is no prior year to discount and the two are identical by construction.

    The cost is calibration, which drifts from 0.846 to 0.832 - the same trade
    every sharpening move in this project makes.
    """

    halflife: float = 3.0
    #: Weight multiplier per season back. 0.5 was chosen by sweep - see the
    #: class docstring for the measured result and the robustness check.
    season_decay: float = 0.5
    include_dnp: bool = True

    _player_stat: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"), init=False)
    _position_prior: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64"), init=False
    )

    @property
    def name(self) -> str:
        parts = [f"season_decayed_hl{self.halflife:g}_d{self.season_decay:g}"]
        if not self.include_dnp:
            parts.append("played_only")
        return "_".join(parts)

    def fit(self, history: pd.DataFrame) -> None:
        self._position_prior = _position_means(history, include_dnp=self.include_dnp)
        frame = _history_for_averaging(history, include_dnp=self.include_dnp)
        if frame.empty:
            self._player_stat = pd.Series(dtype="float64")
            return

        ordered = frame.sort_values(["season", "week"], kind="mergesort")
        # Seasons back from the most recent one present. The harness only ever
        # projects week >= 2, so the target season is always represented.
        latest_season = int(ordered["season"].max())
        seasons_ago = (latest_season - ordered["season"]).to_numpy(dtype="float64")
        season_weight = np.power(self.season_decay, seasons_ago)

        work = pd.DataFrame(
            {
                "player_id": ordered["player_id"].to_numpy(),
                "points": ordered["fantasy_points"].to_numpy(dtype="float64"),
                "season_weight": season_weight,
            }
        )

        stats: dict[str, float] = {}
        for player_id, group in work.groupby("player_id", observed=True):
            n = len(group)
            games_ago = np.arange(n - 1, -1, -1, dtype="float64")
            recency = np.power(0.5, games_ago / self.halflife)
            weights = recency * group["season_weight"].to_numpy()
            total = float(weights.sum())
            if total <= 0.0:
                # Everything this player has is discounted to nothing, which
                # happens at season_decay=0 for someone yet to play this year.
                # The positional prior handles him.
                continue
            stats[str(player_id)] = float(np.dot(weights, group["points"].to_numpy()) / total)

        self._player_stat = pd.Series(stats, dtype="float64")

    def predict(self, targets: pd.DataFrame) -> pd.Series:
        prior = (
            targets["position"].map(self._position_prior).astype("float64")
            if not self._position_prior.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        prior = prior.fillna(_GLOBAL_PRIOR)

        own = (
            targets["player_id"].map(self._player_stat).astype("float64")
            if not self._player_stat.empty
            else pd.Series(np.nan, index=targets.index, dtype="float64")
        )
        return own.fillna(prior).astype("float64").rename("prediction")
