"""Team-week calendar derived from the schedule.

The schedule is the one upstream table a week-``W`` prediction is allowed to read
week-``W`` rows from: matchups, kickoff times, rest days and closing betting
lines are all published before the game. Everything else in the pipeline is
restricted to weeks strictly before the target.

One caveat is recorded in :data:`LATE_AVAILABILITY_COLUMNS`: nflverse stores the
*closing* line, which is only final minutes before kickoff. A projection
published on Wednesday could not have used it. Those columns are therefore
tagged rather than silently mixed in with the genuinely early ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from nflproj.ingest import nflverse as nv

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nflproj.config import Settings

#: Pre-kickoff, but only hours before it. Any predictor consuming these is
#: implicitly claiming a kickoff-time publication slot, which the scorecard
#: must disclose. See ``docs/leakage.md``.
LATE_AVAILABILITY_COLUMNS: Final[frozenset[str]] = frozenset(
    {"team_spread_line", "total_line", "implied_team_total"}
)

_SCHEDULE_COLUMNS: Final[list[str]] = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_rest",
    "away_rest",
    "spread_line",
    "total_line",
    "div_game",
    "roof",
    "surface",
    "result",
]


def build_team_calendar(
    seasons: Iterable[int],
    *,
    settings: Settings | None = None,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """Return one row per (season, week, team) for every scheduled game.

    Args:
        seasons: Seasons to include.
        settings: Overrides process settings (tests).
        regular_season_only: Drop playoff games. Fantasy seasons end in week 17
            or 18, and postseason usage patterns differ enough that mixing them
            into a weekly-projection backtest is misleading.

    Returns:
        A frame keyed on (season, week, team) carrying opponent, home flag,
        rest days, venue and the closing betting line expressed from the
        perspective of ``team``. ``team_game_idx`` is the 0-based index of this
        game among the team's scheduled games that season, which makes "the
        previous N games for this team" a bye-aware operation.
    """
    sched = nv.read_asset(nv.SCHEDULES, columns=_SCHEDULE_COLUMNS, settings=settings)
    wanted = sorted(set(seasons))
    sched = sched[sched["season"].isin(wanted)]
    if regular_season_only:
        sched = sched[sched["game_type"] == "REG"]

    if sched.empty:
        msg = f"schedule contains no rows for seasons {wanted}"
        raise ValueError(msg)

    home = sched.assign(
        team=sched["home_team"],
        opponent_team=sched["away_team"],
        is_home=True,
        rest_days=sched["home_rest"],
        # nflverse stores the line from the home team's perspective, where a
        # positive value means the home team is favoured by that many points.
        team_spread_line=sched["spread_line"],
    )
    away = sched.assign(
        team=sched["away_team"],
        opponent_team=sched["home_team"],
        is_home=False,
        rest_days=sched["away_rest"],
        team_spread_line=-sched["spread_line"],
    )

    cal = pd.concat([home, away], ignore_index=True)

    # Points this team is expected to score, the standard decomposition of a
    # total and a spread. Left null when either leg is missing.
    cal["implied_team_total"] = cal["total_line"] / 2.0 + cal["team_spread_line"] / 2.0

    cal["kickoff_et"] = pd.to_datetime(
        cal["gameday"].astype(str) + " " + cal["gametime"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    ).dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")

    cal = cal.sort_values(["season", "team", "week"], kind="mergesort").reset_index(drop=True)
    cal["team_game_idx"] = cal.groupby(["season", "team"], observed=True).cumcount()

    keep = [
        "season",
        "week",
        "team",
        "opponent_team",
        "game_id",
        "team_game_idx",
        "is_home",
        "rest_days",
        "team_spread_line",
        "total_line",
        "implied_team_total",
        "div_game",
        "roof",
        "surface",
        "kickoff_et",
        "result",
    ]
    out = cal[keep].copy()
    out["week"] = out["week"].astype("int16")
    out["season"] = out["season"].astype("int16")
    return out


def completed_games(calendar: pd.DataFrame) -> pd.DataFrame:
    """Rows whose game has a final score.

    Used to decide how far a live backtest can run: a week is only scorable
    once every game in it has finished.
    """
    return calendar[calendar["result"].notna()]


def latest_completed_week(calendar: pd.DataFrame, season: int) -> int | None:
    """Highest week of ``season`` in which every scheduled game has a result.

    Returns ``None`` if no week is fully complete. A partially played week is
    deliberately not counted: scoring half a slate would bias any metric toward
    whichever games happened to be early in the week.
    """
    season_cal = calendar[calendar["season"] == season]
    if season_cal.empty:
        return None
    done = season_cal.groupby("week", observed=True)["result"].apply(lambda s: s.notna().all())
    complete = done[done].index
    if len(complete) == 0:
        return None
    return int(max(complete))


def next_projectable_week(calendar: pd.DataFrame, season: int) -> int | None:
    """The earliest week of ``season`` that is still to be played.

    This is what a weekly job should publish: the week after the last one that
    finished. Deriving it from the data rather than from the calendar date means
    the schedule is the single source of truth, and a postponed game shifts the
    published week automatically.

    Returns ``None`` when the season has not started, has finished, or the next
    week would be week 1, which is out of scope.
    """
    last_done = latest_completed_week(calendar, season)
    if last_done is None:
        return None
    candidate = last_done + 1
    season_weeks = calendar.loc[calendar["season"] == season, "week"]
    if season_weeks.empty or candidate > int(season_weeks.max()):
        return None
    return candidate


def current_season(calendar: pd.DataFrame) -> int | None:
    """The most recent season that has at least one completed game."""
    played = calendar[calendar["result"].notna()]
    if played.empty:
        return None
    return int(played["season"].max())
