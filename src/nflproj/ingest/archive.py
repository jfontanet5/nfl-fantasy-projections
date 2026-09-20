"""A point-in-time archive of data that upstream serves only in the present tense.

Some tables nflverse publishes are observation logs: every row carries the
moment it was seen, so a historical as-of view can be reconstructed at any time.
Depth charts are one of these - they carry a ``dt`` column, and filtering
``dt < kickoff`` gives a correct point-in-time snapshot years later.

Injury reports are not. The file is keyed on (season, week) and rewritten in
place as the week progresses: Wednesday's limited participant becomes Friday's
questionable becomes Sunday's inactive, and each overwrite destroys the last.
By the time a week is over, what the report said on Friday is gone. No amount
of care at model-training time can recover it.

So the only way to have that history is to record it as it happens. This module
does exactly that, and nothing else: it fetches the current file, stores it if
the bytes are new, and appends one line to a manifest saying what was observed
and when.

Two properties make the archive trustworthy:

Content-addressed
    A snapshot is stored under the hash of its bytes. An unchanged report costs
    one manifest line, not another copy of the file, so a daily cadence stays
    cheap over a season.

Append-only
    Nothing is ever rewritten. The manifest is JSONL so a new observation is one
    added line in a diff, and a rewritten history would be obvious in review.

The read side is :func:`as_of`, which is deliberately strict: it returns the
latest snapshot taken *before* a cutoff, never one taken at or after it. That is
the whole point of keeping the archive, so it is not a parameter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import pandas as pd

from nflproj.ingest.manifest import sha256_file
from nflproj.ingest.nflverse import Asset, IngestError, fetch_asset
from nflproj.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from nflproj.config import Settings

log = get_logger(__name__)

#: Weekly injury report: practice participation and game-status designations.
#: Carries no timestamp of its own, which is why it is archived here.
INJURIES = Asset(name="injuries", release="injuries", stem="injuries")

#: Depth charts are deliberately NOT archived. Every row carries a ``dt``
#: observation timestamp and the file is cumulative, so upstream already is a
#: point-in-time log. Snapshotting it would duplicate megabytes weekly to
#: reconstruct information we can already read directly. See `depth_chart_as_of`.
DEPTH_CHARTS = Asset(name="depth_charts", release="depth_charts", stem="depth_charts")

MANIFEST_NAME: Final = "manifest.jsonl"
BLOB_DIR: Final = "blobs"

#: How close a kickoff has to be for a snapshot to be worth taking. Injury
#: reports only move in the days around a game; capturing in June would append
#: manifest lines forever to record that nothing changed. Three days keeps every
#: in-season capture (the longest mid-season gap between a Monday night game and
#: the following Thursday is under 3 days) and stops in the offseason.
SNAPSHOT_LEAD: Final = timedelta(days=3)


@dataclass(frozen=True, slots=True)
class Observation:
    """One fetch of an upstream file, and what it contained."""

    asset: str
    season: int
    fetched_at: str
    sha256: str
    rows: int
    source_url: str
    #: True when these bytes were seen for the first time. A False here is not a
    #: wasted run: it is evidence the report did not change between two times.
    novel: bool

    @property
    def fetched_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.fetched_at)


class Archive:
    """Content-addressed, append-only store for one asset."""

    def __init__(self, root: Path, asset: Asset) -> None:
        self.asset = asset
        self.root = root / asset.name
        self.manifest_path = self.root / MANIFEST_NAME
        self.blob_dir = self.root / BLOB_DIR

    # ------------------------------------------------------------ read

    def observations(self, season: int | None = None) -> list[Observation]:
        """Every recorded observation, oldest first.

        A malformed line is skipped rather than fatal: the archive is the record
        of what we saw, and one bad append should not make the rest unreadable.
        A line whose timestamp will not parse counts as malformed - an
        observation we cannot place in time is not an observation.

        Sorting is by parsed instant, not by the raw string. Those agree only as
        long as every line carries the same UTC offset, which is a property of
        how we happen to write them today and not something a file that is meant
        to outlive this code should depend on.
        """
        if not self.manifest_path.exists():
            return []
        out: list[Observation] = []
        for line in self.manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                observation = Observation(**payload)
                if observation.fetched_at_dt.tzinfo is None:
                    msg = f"naive timestamp {observation.fetched_at!r} has no defined instant"
                    raise ValueError(msg)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.warning("archive.bad_manifest_line", error=str(exc))
                continue
            if season is None or observation.season == season:
                out.append(observation)
        out.sort(key=lambda o: o.fetched_at_dt)
        return out

    def blob_path(self, sha256: str) -> Path:
        return self.blob_dir / f"{sha256}.parquet"

    def as_of(self, cutoff: datetime, *, season: int | None = None) -> pd.DataFrame | None:
        """The most recent snapshot taken strictly before ``cutoff``.

        Strictly before, never at or after: a snapshot taken at kickoff may
        already reflect the inactive list, which is exactly the information a
        pre-kickoff projection must not have.

        Returns ``None`` when nothing was recorded before that moment - which is
        the honest answer for any week predating the archive, and must not be
        confused with "no injuries".
        """
        if cutoff.tzinfo is None:
            msg = "cutoff must be timezone-aware; a naive datetime has no defined instant"
            raise ValueError(msg)

        candidates = [o for o in self.observations(season) if o.fetched_at_dt < cutoff]
        if not candidates:
            return None

        chosen = candidates[-1]
        path = self.blob_path(chosen.sha256)
        if not path.exists():
            msg = (
                f"manifest references blob {chosen.sha256[:12]} for "
                f"{chosen.fetched_at}, but the file is missing from {self.blob_dir}"
            )
            raise IngestError(msg)
        return pd.read_parquet(path)

    # ------------------------------------------------------------ write

    def capture(
        self,
        season: int,
        *,
        settings: Settings | None = None,
        now: datetime | None = None,
    ) -> Observation:
        """Fetch the asset as it stands right now and record what was seen.

        Always forces a fresh download: the point of a snapshot is the current
        state, so serving it from a cache would archive a lie about when it was
        observed.
        """
        now = now or datetime.now(UTC)
        source = fetch_asset(self.asset, season, settings=settings, force=True)
        digest = sha256_file(source)

        self.blob_dir.mkdir(parents=True, exist_ok=True)
        destination = self.blob_path(digest)
        novel = not destination.exists()
        if novel:
            destination.write_bytes(source.read_bytes())

        observation = Observation(
            asset=self.asset.name,
            season=season,
            fetched_at=now.isoformat(),
            sha256=digest,
            rows=len(pd.read_parquet(destination)),
            source_url=self.asset.url(season),
            novel=novel,
        )
        self._append(observation)
        log.info(
            "archive.captured",
            asset=self.asset.name,
            season=season,
            rows=observation.rows,
            novel=novel,
            sha256=digest[:12],
        )
        return observation

    def _append(self, observation: Observation) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a") as fh:
            fh.write(json.dumps(asdict(observation), sort_keys=True) + "\n")


# ---------------------------------------------------------------- helpers


def _kickoffs_utc(calendar: pd.DataFrame) -> pd.Series:
    """The calendar's kickoff column as a tz-aware UTC series, nulls dropped."""
    if "kickoff_et" not in calendar.columns:
        msg = "calendar has no kickoff_et column; build it with build_team_calendar"
        raise KeyError(msg)
    return pd.to_datetime(calendar["kickoff_et"], utc=True).dropna()


def snapshot_season(
    calendar: pd.DataFrame,
    *,
    now: datetime | None = None,
    lead: timedelta = SNAPSHOT_LEAD,
) -> int | None:
    """Which season a snapshot taken *now* belongs to, or ``None`` for neither.

    Deliberately not :func:`nflproj.features.calendar.current_season`, which
    answers "which season has games in the books". In the week before an opener
    that still points at last year, and a snapshot of this week's injuries filed
    under last season is a snapshot of nothing.

    The question a snapshotter actually asks is "is a game imminent, and if so
    whose". So: the season of the next kickoff, provided that kickoff is within
    ``lead``. ``None`` in the offseason, which the caller should treat as
    "nothing to do" rather than as a failure.
    """
    now = now or datetime.now(UTC)
    if calendar.empty:
        return None

    kickoffs = _kickoffs_utc(calendar)
    if kickoffs.empty:
        return None

    horizon = pd.Timestamp(now + lead)
    imminent = calendar.loc[kickoffs.index][(kickoffs >= pd.Timestamp(now)) & (kickoffs <= horizon)]
    if imminent.empty:
        return None
    return int(imminent["season"].min())


def first_kickoff(calendar: pd.DataFrame, season: int, week: int) -> datetime | None:
    """When the earliest game of a week starts.

    The cutoff for a whole week is its *first* kickoff, not each game's own. A
    projection published for the week is published once, before any of it has
    been played.
    """
    rows = calendar[(calendar["season"] == season) & (calendar["week"] == week)]
    kickoffs = rows["kickoff_et"].dropna()
    if kickoffs.empty:
        return None
    earliest = pd.Timestamp(kickoffs.min())
    return earliest.to_pydatetime().astimezone(UTC)


def injuries_before_kickoff(
    archive_root: Path,
    calendar: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> pd.DataFrame | None:
    """The injury report as it stood before a week's first game.

    Returns ``None`` for any week the archive does not reach back to, which for
    now is every week before this archive started. That absence is the honest
    answer and callers must treat it as "unknown", never as "nobody was hurt".
    """
    cutoff = first_kickoff(calendar, season, week)
    if cutoff is None:
        return None
    return Archive(archive_root, INJURIES).as_of(cutoff, season=season)


def depth_chart_as_of(depth_charts: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Point-in-time view of the depth chart, read straight from upstream.

    No archive needed: every row carries its own observation time in ``dt``, and
    the upstream file is cumulative, so the as-of view is a filter rather than a
    recording problem.
    """
    if cutoff.tzinfo is None:
        msg = "cutoff must be timezone-aware; a naive datetime has no defined instant"
        raise ValueError(msg)
    if "dt" not in depth_charts.columns:
        msg = "depth charts have no dt column; upstream schema changed"
        raise KeyError(msg)

    observed = pd.to_datetime(depth_charts["dt"], format="ISO8601", utc=True)
    earlier = observed < pd.Timestamp(cutoff)
    if not earlier.any():
        return depth_charts.iloc[:0]

    # The latest *scrape*, not the latest row per team: a depth chart is a
    # snapshot of a whole team at a moment, and mixing rows from two scrapes
    # would produce a chart that never existed.
    latest = observed[earlier].max()
    return depth_charts[earlier & (observed == latest)]
