"""Availability: what the injury report said before kickoff.

Why this module exists
----------------------

The published predictor is a weighted mean of past fantasy points. It is
structurally blind to injuries, so a player hurt on Sunday carries his healthy
average into the next board at full strength. Measured on the 2026 week-5
board: 19 Out-or-Doubtful players ranked, five of them inside the top 100, with
Justin Jefferson at 58 on a 12.3-point projection while listed Out.

This is also the slice-2 negative result coming due. ``AvailabilityWeighted``
tried to infer availability from past play and lost every configuration in a
nine-point sweep; the diagnosis recorded at the time was that availability
inferred from past play is backward-looking by construction, and the right
input is a point-in-time injury feed. The feed exists now.

The leakage trap, which is severe
---------------------------------

Upstream rewrites the injury file in place as a week progresses - that is the
entire reason :mod:`nflproj.ingest.archive` exists. So the file's week-*W* rows,
read after week *W*, already encode who actually played. Using them to
"project" week *W* would be near-perfect leakage, because *Out* does not merely
correlate with scoring zero, it partly **is** the outcome.

Why the cutoff is per *game*, not per week
------------------------------------------

The rest of this project uses one cutoff per week - its first kickoff - because
a board is published once, before any of it has been played. For availability
that rule is not conservative, it is useless, and the archive proves it: the
last snapshot before 2026 week 4's Thursday kickoff contains ten week-4 rows
and **not one game-status designation**. Teams file practice participation on
Wednesday and designations on *Friday*, and Friday is after Thursday night.

So availability is evaluated against each player's **own** kickoff. Friday's
report is legitimately pre-kickoff information for a Sunday game, and using it
leaks nothing about that player's outcome, which is still unknown. History
stays per-week and therefore stays strictly conservative about *outcomes*;
only this one pre-kickoff column reads per game. They are different questions:
history is about what has happened, availability is about what was known.

Hence two sources, and the distinction is enforced here rather than remembered
by callers:

:data:`InjurySource.ARCHIVE`
    ``Archive.as_of(first_kickoff)`` - the only source a backtest may use. It
    reaches back only to the day the archive started, and returns nothing for
    earlier weeks. That absence is the honest answer.

:data:`InjurySource.LIVE`
    The current upstream file. Legitimate **only** for a player whose own game
    has not kicked off, where it is information available at publication time
    rather than hindsight.

On the factors
--------------

:data:`AVAILABILITY_FACTORS` is a **stated prior, not a measurement**, and the
distinction matters because the obvious measurement is contaminated. Computing
``P(play | Questionable)`` from the upstream file uses rows that were relabelled
after the game: players who were Questionable and then sat are now recorded as
Out, so the surviving Questionable rows skew toward those who played and any
factor measured that way is an optimistic upper bound. The archive can measure
this honestly, but only over the weeks it covers. So these are priors, the
scorecard says so, and the archive refines them as it accumulates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import pandas as pd

from nflproj.config import Settings, get_settings
from nflproj.ingest.archive import INJURIES, Archive, first_kickoff
from nflproj.ingest.nflverse import IngestError, fetch_asset
from nflproj.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

#: Panel column carrying the game-status designation, or null when none was
#: published. Null means *unknown*, never *healthy*.
STATUS_COLUMN: Final = "report_status"

#: Panel column carrying practice participation. Not used by the availability
#: factor yet; carried because it is the richer signal a model will want.
PRACTICE_COLUMN: Final = "practice_status"

#: Multipliers applied to a projection, by designation. A stated prior - see
#: the module docstring on why the obvious measurement is contaminated.
#:
#: Out is 0.0 rather than merely small: publishing a non-zero projection for a
#: player the league has declared will not play is publishing a number we know
#: to be wrong.
AVAILABILITY_FACTORS: Final[dict[str, float]] = {
    "out": 0.0,
    "ir": 0.0,
    "pup": 0.0,
    "suspended": 0.0,
    "doubtful": 0.2,
    "questionable": 0.9,
}

#: Designations meaning the player will not take the field.
UNAVAILABLE: Final[frozenset[str]] = frozenset(
    {status for status, factor in AVAILABILITY_FACTORS.items() if factor == 0.0}
)

_STATUS_COLUMNS: Final = (STATUS_COLUMN, PRACTICE_COLUMN)


class InjurySource(StrEnum):
    """Where a week's designations came from."""

    ARCHIVE = "archive"
    LIVE = "live"
    #: No designations were available. Distinct from "nobody was hurt".
    NONE = "none"


class InjurySourceError(RuntimeError):
    """Raised when the live feed is requested for a week already played."""


def availability_factor(status: object) -> float:
    """Multiplier for one designation. Unknown or unrecognised means 1.0.

    An unrecognised designation leaves the projection alone rather than
    guessing. Upstream can add a status we have never seen, and silently
    zeroing a player because of a string we do not recognise would be a worse
    failure than ignoring it.
    """
    if status is None or not isinstance(status, str) or not status.strip():
        return 1.0
    return AVAILABILITY_FACTORS.get(status.strip().lower(), 1.0)


def availability_factors(statuses: pd.Series) -> pd.Series:
    """Vectorised :func:`availability_factor`, aligned to ``statuses.index``."""
    return statuses.map(availability_factor).astype("float64")


def _empty_status() -> pd.DataFrame:
    return pd.DataFrame({"player_id": pd.Series(dtype="object")}).assign(
        **{c: pd.Series(dtype="object") for c in _STATUS_COLUMNS}
    )


def _normalise(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Reduce an upstream injury frame to one row per player for one week."""
    if frame.empty:
        return _empty_status()

    missing = [c for c in ("season", "week", "gsis_id") if c not in frame.columns]
    if missing:
        msg = f"injury frame is missing {missing}; upstream schema changed"
        raise KeyError(msg)

    rows = frame[(frame["season"] == season) & (frame["week"] == week)].copy()
    if rows.empty:
        return _empty_status()

    rows = rows.rename(columns={"gsis_id": "player_id"})
    for col in _STATUS_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    # One designation per player. Upstream is already unique on this key, but a
    # duplicate would silently fan out the panel on merge.
    return (
        rows[["player_id", *_STATUS_COLUMNS]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"], keep="last")
        .reset_index(drop=True)
    )


def load_status_before(
    season: int,
    week: int,
    cutoff: datetime | None,
    *,
    settings: Settings | None = None,
    allow_live: bool = True,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, InjurySource]:
    """The designations for one week as they stood before ``cutoff``.

    ``cutoff`` is a kickoff - a player's own, normally. Before it, the live
    upstream file is the right source: it is what a publisher knows at
    publication time. At or after it, only the archive will do, because the
    live file has since been overwritten with post-game state.

    Returns an empty frame and :data:`InjurySource.NONE` when no designations
    exist - the common case on a Tuesday before the week's reports are
    published, and for every week predating the archive.
    """
    settings = settings or get_settings()
    now = now or datetime.now(UTC)
    started = cutoff is not None and now >= cutoff

    if not started and allow_live:
        try:
            path = fetch_asset(INJURIES, season, settings=settings)
        except IngestError as exc:
            log.warning("injuries.live_unavailable", season=season, week=week, error=str(exc))
            return _empty_status(), InjurySource.NONE
        status = _normalise(pd.read_parquet(path), season, week)
        if status.empty:
            # Normal before midweek: the week exists but its report does not.
            log.info("injuries.none_published_yet", season=season, week=week)
            return status, InjurySource.NONE
        return status, InjurySource.LIVE

    if cutoff is None:
        return _empty_status(), InjurySource.NONE

    archived = Archive(settings.archive_dir, INJURIES).as_of(cutoff, season=season)
    if archived is None:
        return _empty_status(), InjurySource.NONE
    status = _normalise(archived, season, week)
    return status, (InjurySource.ARCHIVE if not status.empty else InjurySource.NONE)


def attach_status(
    panel: pd.DataFrame,
    *,
    calendar: pd.DataFrame,
    settings: Settings | None = None,
    allow_live: bool = True,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Add :data:`STATUS_COLUMN` and :data:`PRACTICE_COLUMN` to a panel.

    Per *game*, not per week. Each player's designation is taken as it stood
    before his own kickoff, which is why this can see Friday's report at all -
    see the module docstring. A row with no kickoff falls back to the week's
    first, which is the conservative direction.

    Weeks with nothing available keep nulls, and a null must be read as "no
    designation was published", never as "healthy".
    """
    out = panel.copy()
    for col in _STATUS_COLUMNS:
        out[col] = pd.Series([None] * len(out), index=out.index, dtype="object")

    if out.empty:
        return out

    kickoffs = (
        pd.to_datetime(out["kickoff_et"], utc=True)
        if "kickoff_et" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )

    # Memoised per (season, week, cutoff): a week has a handful of distinct
    # kickoff times, and without this a full backtest re-reads the manifest
    # and a blob once per row.
    cache: dict[tuple[int, int, datetime | None], tuple[pd.DataFrame, InjurySource]] = {}
    sources: list[InjurySource] = []

    pairs = out[["season", "week"]].drop_duplicates().to_numpy()
    for raw_season, raw_week in pairs:
        season, week = int(raw_season), int(raw_week)
        in_week = (out["season"] == season) & (out["week"] == week)
        fallback = first_kickoff(calendar, season, week)

        for cutoff in sorted({*kickoffs[in_week].dropna().tolist()}) or [fallback]:
            cutoff_dt = cutoff.to_pydatetime() if isinstance(cutoff, pd.Timestamp) else cutoff
            key = (season, week, cutoff_dt)
            if key not in cache:
                cache[key] = load_status_before(
                    season,
                    week,
                    cutoff_dt,
                    settings=settings,
                    allow_live=allow_live,
                    now=now,
                )
            status, source = cache[key]
            sources.append(source)
            if status.empty:
                continue

            mask = in_week & (kickoffs.eq(cutoff) if cutoff is not None else kickoffs.isna())
            if not mask.any():
                continue
            lookup = status.set_index("player_id")
            for col in _STATUS_COLUMNS:
                out.loc[mask, col] = out.loc[mask, "player_id"].map(lookup[col])

    log.info(
        "injuries.attached",
        cutoffs=len(sources),
        cutoffs_with_designations=sum(1 for s in sources if s is not InjurySource.NONE),
        designations=int(out[STATUS_COLUMN].notna().sum()),
    )
    return out


def status_summary(panel: pd.DataFrame) -> dict[str, int]:
    """Counts by designation, for a bundle manifest or a scorecard note."""
    if STATUS_COLUMN not in panel.columns:
        return {}
    counts = panel[STATUS_COLUMN].dropna().astype(str).str.lower().value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def archive_root(settings: Settings | None = None) -> Path:
    """Where the injury archive lives, for callers that need the path."""
    return (settings or get_settings()).archive_dir
