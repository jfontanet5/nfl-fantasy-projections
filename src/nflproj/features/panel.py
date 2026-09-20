"""The player-week panel: who we are asked to project, and what they scored.

Defining the prediction universe is the highest-leverage honesty decision in
this project, so it is made explicitly here rather than falling out of a join.

The trap is to evaluate on the set of players who *did* play. That set is only
knowable after kickoff, so conditioning on it quietly deletes every
inactive-and-scored-zero case - the exact cases a projection needs to get right -
and flatters every model that touches the data. See ``docs/leakage.md``.

Two universes are therefore supported and both are reported:

``UniversePolicy.ACTIVE_RECENT``
    The default and the only one we treat as decision-grade. A player is in
    week ``W``'s universe if their team has a game that week and they recorded
    a box-score line in at least one of their team's previous
    ``lookback_games`` games. Players who then sit out score exactly 0.0 and
    stay in the evaluation. Constructed from weeks strictly before ``W``.

``UniversePolicy.PLAYED``
    Only players with a week-``W`` box-score line. Reported for comparability
    with public numbers, always labelled as outcome-conditioned.

Known limitation: week 1 of each season is out of scope. Establishing who is on
which roster in week 1 needs offseason transaction data we do not yet have a
clean as-of source for, and guessing from the prior season's team is wrong for
every player who changed teams. Projecting week 1 by carrying stale teams
forward would be a leakage-adjacent fudge, so v1 declines to project it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final, assert_never

import pandas as pd

from nflproj.config import FANTASY_POSITIONS, Settings, get_settings
from nflproj.features.calendar import build_team_calendar
from nflproj.ingest import nflverse as nv
from nflproj.logging import get_logger
from nflproj.scoring import PPR, REQUIRED_STAT_COLUMNS, ScoringRules, compute_fantasy_points

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)

#: Weeks 1 of each season are excluded; see the module docstring.
FIRST_PROJECTABLE_WEEK: Final = 2

#: How many of a team's *previous games* (bye-aware, not calendar weeks) an
#: appearance keeps a player in the universe for.
DEFAULT_LOOKBACK_GAMES: Final = 3

#: Identity and usage columns carried from the weekly stats table alongside the
#: scoring inputs. ``targets``/``carries`` drive the volume tiers used to
#: stratify the scorecard.
_STAT_ID_COLUMNS: Final[list[str]] = [
    "season",
    "week",
    "season_type",
    "game_id",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "targets",
    "carries",
    "attempts",
]

PANEL_KEY: Final[list[str]] = ["season", "week", "player_id"]


class UniversePolicy(StrEnum):
    """Which players a week's metrics are computed over."""

    ACTIVE_RECENT = "active_recent"
    PLAYED = "played"


def load_weekly_stats(
    seasons: Iterable[int],
    *,
    scoring: ScoringRules = PPR,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load scored regular-season box-score lines for fantasy positions.

    Returns one row per player-week that actually happened, with the target
    ``fantasy_points`` computed by :mod:`nflproj.scoring` rather than taken from
    upstream.
    """
    settings = settings or get_settings()
    columns = sorted(set(_STAT_ID_COLUMNS) | set(REQUIRED_STAT_COLUMNS))
    raw = nv.read_seasons(nv.PLAYER_STATS, sorted(set(seasons)), columns=columns, settings=settings)

    stats = raw[(raw["season_type"] == "REG") & (raw["position"].isin(FANTASY_POSITIONS))].copy()

    stats["fantasy_points"] = compute_fantasy_points(stats, scoring)

    out = stats[
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "player_display_name",
            "position",
            "team",
            "targets",
            "carries",
            "attempts",
            "fantasy_points",
        ]
    ].copy()
    out["season"] = out["season"].astype("int16")
    out["week"] = out["week"].astype("int16")

    dupes = out.duplicated(subset=PANEL_KEY).sum()
    if dupes:
        # A player traded mid-week, or an upstream key change. Either way the
        # panel key is no longer unique and downstream joins would fan out.
        msg = f"weekly stats contain {dupes} duplicate (season, week, player_id) rows"
        raise ValueError(msg)

    log.info("panel.stats_loaded", rows=len(out), seasons=sorted(out["season"].unique().tolist()))
    return out


def build_universe(
    stats: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> pd.DataFrame:
    """Return the set of player-weeks we commit to projecting.

    An appearance in a team's game activates that player for the team's next
    ``lookback_games`` *games*. Walking the team's game index rather than the
    calendar week means a bye does not silently drop a player from the universe.

    Args:
        stats: Output of :func:`load_weekly_stats`.
        calendar: Output of :func:`nflproj.features.calendar.build_team_calendar`.
        lookback_games: Activation window length.

    Returns:
        A frame keyed on (season, week, player_id) with the player's team and
        position carried forward from their most recent prior appearance.

    Raises:
        ValueError: If ``lookback_games`` is not positive.
    """
    if lookback_games < 1:
        msg = f"lookback_games must be >= 1, got {lookback_games}"
        raise ValueError(msg)

    cal_keys = calendar[["season", "week", "team", "team_game_idx"]]

    # Where each appearance sits in its team's season.
    appearances = stats.merge(cal_keys, on=["season", "week", "team"], how="inner")
    appearances = appearances[
        ["season", "player_id", "player_display_name", "position", "team", "team_game_idx", "week"]
    ].rename(columns={"week": "source_week"})

    # Project each appearance forward onto the team's next N games.
    activations = []
    for step in range(1, lookback_games + 1):
        shifted = appearances.assign(team_game_idx=appearances["team_game_idx"] + step)
        activations.append(shifted)
    activated = pd.concat(activations, ignore_index=True)

    universe = activated.merge(
        cal_keys,
        on=["season", "team", "team_game_idx"],
        how="inner",
        suffixes=("", "_target"),
    )

    # Several appearances can activate the same target week; keep the most
    # recent one so team and position reflect the latest known state.
    universe = universe.sort_values(
        ["season", "week", "player_id", "source_week"], kind="mergesort"
    )
    universe = universe.drop_duplicates(subset=PANEL_KEY, keep="last")

    cols = ["season", "week", "player_id", "player_display_name", "position", "team", "source_week"]
    out = universe[cols].rename(columns={"source_week": "last_appearance_week"})

    # Week 1 cannot be activated by recency, but its results are still the
    # history every week-2 projection depends on. Add the players who actually
    # appeared in week 1 so the history is complete. These rows are never
    # projected - `build_panel` marks them ``projectable=False`` - so the fact
    # that the set is outcome-conditioned cannot flatter any metric.
    opener = stats[stats["week"] < FIRST_PROJECTABLE_WEEK].copy()
    opener["last_appearance_week"] = opener["week"]
    opener = opener[[*cols[:-1], "last_appearance_week"]]

    combined = pd.concat([out, opener], ignore_index=True)
    combined = combined.drop_duplicates(subset=PANEL_KEY, keep="first")
    return combined.sort_values(PANEL_KEY, kind="mergesort").reset_index(drop=True)


def build_panel(
    seasons: Iterable[int],
    *,
    scoring: ScoringRules = PPR,
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Build the full player-week panel for ``seasons``.

    Returns:
        One row per player-week in the ``ACTIVE_RECENT`` universe, carrying the
        realised target (``fantasy_points``, 0.0 when the player did not play),
        a ``played`` flag, and the pre-kickoff game context from the schedule.
    """
    seasons = sorted(set(seasons))
    settings = settings or get_settings()

    stats = load_weekly_stats(seasons, scoring=scoring, settings=settings)
    calendar = build_team_calendar(seasons, settings=settings)
    universe = build_universe(stats, calendar, lookback_games=lookback_games)

    actuals = stats[[*PANEL_KEY, "fantasy_points", "targets", "carries", "attempts", "game_id"]]
    panel = universe.merge(actuals, on=PANEL_KEY, how="left", validate="one_to_one")

    panel["played"] = panel["fantasy_points"].notna()
    panel["fantasy_points"] = panel["fantasy_points"].fillna(0.0).astype("float64")
    #: Week 1 rows exist only to seed history; we do not claim to project them.
    panel["projectable"] = panel["week"] >= FIRST_PROJECTABLE_WEEK
    for col in ("targets", "carries", "attempts"):
        panel[col] = panel[col].fillna(0.0).astype("float64")

    panel = panel.merge(
        calendar.drop(columns=["team_game_idx", "result"]),
        on=["season", "week", "team"],
        how="left",
        suffixes=("", "_cal"),
    )
    # The stats table's game_id is null for players who did not play; the
    # calendar always has one, so prefer it.
    panel["game_id"] = panel["game_id_cal"].fillna(panel["game_id"])
    panel = panel.drop(columns=["game_id_cal"])

    panel = panel.sort_values(PANEL_KEY, kind="mergesort").reset_index(drop=True)

    log.info(
        "panel.built",
        rows=len(panel),
        projectable_rows=int(panel["projectable"].sum()),
        seasons=f"{seasons[0]}-{seasons[-1]}",
        played_rate=round(float(panel.loc[panel["projectable"], "played"].mean()), 4),
        lookback_games=lookback_games,
    )
    return panel


def apply_universe(panel: pd.DataFrame, policy: UniversePolicy) -> pd.DataFrame:
    """Restrict a panel to the rows a given universe policy scores."""
    scorable = panel[panel["projectable"]]
    if policy is UniversePolicy.ACTIVE_RECENT:
        return scorable.reset_index(drop=True)
    if policy is UniversePolicy.PLAYED:
        return scorable[scorable["played"]].reset_index(drop=True)
    # Exhaustiveness is checked statically: adding a policy without handling it
    # here becomes a type error rather than a silent fall-through.
    assert_never(policy)
