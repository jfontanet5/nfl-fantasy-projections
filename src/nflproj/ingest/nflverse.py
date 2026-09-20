"""Download raw assets from the nflverse-data GitHub releases.

nflverse publishes one parquet per (dataset, season) under a stable release tag,
plus a few whole-history files. We mirror those into ``data/raw`` and record
provenance in the manifest. Nothing in this module interprets the data; parsing
and any column semantics live in :mod:`nflproj.features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import requests

from nflproj.config import Settings, get_settings
from nflproj.ingest.manifest import Manifest
from nflproj.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)

RELEASE_BASE: Final = "https://github.com/nflverse/nflverse-data/releases/download"
_DOWNLOAD_CHUNK_BYTES = 1 << 16


@dataclass(frozen=True, slots=True)
class Asset:
    """A logical nflverse dataset.

    Attributes:
        name: Our short name, used as the raw-zone subdirectory.
        release: The nflverse release tag the files live under.
        stem: Filename stem upstream uses. Per-season assets append ``_{season}``.
        per_season: Whether upstream splits this dataset by season.
    """

    name: str
    release: str
    stem: str
    per_season: bool = True

    def filename(self, season: int | None) -> str:
        if not self.per_season:
            return f"{self.stem}.parquet"
        if season is None:
            msg = f"asset {self.name!r} is per-season; a season is required"
            raise ValueError(msg)
        return f"{self.stem}_{season}.parquet"

    def url(self, season: int | None) -> str:
        return f"{RELEASE_BASE}/{self.release}/{self.filename(season)}"

    def key(self, season: int | None) -> str:
        return self.name if season is None else f"{self.name}/{season}"


#: Weekly box-score lines. This is the source of the prediction target.
#:
#: Note the release tag is ``stats_player``, not the older ``player_stats``.
#: nflverse migrated in 2025: the legacy tag froze at season 2024, while
#: ``stats_player`` carries 1999-present and is a strict column superset (it
#: additionally exposes ``game_id``, which we rely on to join schedules).
#: Pointing at the legacy tag silently yields no current-season rows.
PLAYER_STATS = Asset(name="player_stats", release="stats_player", stem="stats_player_week")

#: Whole-history schedule. Known well before kickoff, so it is the one table a
#: week-W feature may read week-W rows from.
SCHEDULES = Asset(name="schedules", release="schedules", stem="games", per_season=False)

#: Week-resolution roster snapshots: team, position, status.
WEEKLY_ROSTERS = Asset(name="weekly_rosters", release="weekly_rosters", stem="roster_weekly")

#: Offensive/defensive snap participation per player-game.
SNAP_COUNTS = Asset(name="snap_counts", release="snap_counts", stem="snap_counts")

DEFAULT_ASSETS: Final[tuple[Asset, ...]] = (
    PLAYER_STATS,
    SCHEDULES,
    WEEKLY_ROSTERS,
    SNAP_COUNTS,
)


class IngestError(RuntimeError):
    """Raised when an upstream asset cannot be retrieved."""


def _destination(asset: Asset, season: int | None, settings: Settings) -> Path:
    return settings.raw_dir / asset.name / asset.filename(season)


def fetch_asset(
    asset: Asset,
    season: int | None = None,
    *,
    settings: Settings | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
) -> Path:
    """Download one asset into the raw zone, reusing a fresh cached copy.

    "Fresh" is defined by ``settings.raw_max_age_hours``. Completed seasons never
    change in practice, but the current season is revised for days after each
    game, so a blanket cache would quietly serve stale numbers during the exact
    window we care most about.

    Args:
        asset: Dataset descriptor.
        season: Season to fetch, or ``None`` for whole-history assets.
        settings: Overrides the process settings (tests).
        manifest: Overrides the manifest instance (tests).
        force: Re-download even if the cached copy is fresh.

    Returns:
        Path to the local parquet file.

    Raises:
        IngestError: On any non-200 response or network failure.
    """
    settings = settings or get_settings()
    manifest = manifest or Manifest(settings.raw_dir)

    key = asset.key(season)
    dest = _destination(asset, season, settings)
    entry = manifest.get(key)

    if not force and dest.exists() and entry is not None:
        age = entry.age_hours()
        if age < settings.raw_max_age_hours:
            log.debug("raw.cache_hit", key=key, age_hours=round(age, 2))
            return dest

    url = asset.url(season)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")

    log.info("raw.fetch", key=key, url=url)
    try:
        with requests.get(url, stream=True, timeout=settings.http_timeout) as resp:
            if resp.status_code != requests.codes.ok:
                msg = f"{url} returned HTTP {resp.status_code}"
                raise IngestError(msg)
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        tmp.unlink(missing_ok=True)
        msg = f"failed to fetch {url}: {exc}"
        raise IngestError(msg) from exc

    tmp.replace(dest)
    recorded = manifest.record(key=key, url=url, path=dest)
    log.info("raw.stored", key=key, size_bytes=recorded.size_bytes, sha256=recorded.sha256[:12])
    return dest


def fetch_seasons(
    asset: Asset,
    seasons: Iterable[int],
    *,
    settings: Settings | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
) -> dict[int, Path]:
    """Fetch a per-season asset for several seasons.

    A season that does not exist upstream yet (a future season, or one nflverse
    has not published) is skipped with a warning rather than failing the run, so
    an in-season pipeline does not break the moment it reaches for next year.
    """
    settings = settings or get_settings()
    manifest = manifest or Manifest(settings.raw_dir)
    out: dict[int, Path] = {}
    for season in seasons:
        try:
            out[season] = fetch_asset(
                asset, season, settings=settings, manifest=manifest, force=force
            )
        except IngestError as exc:
            log.warning("raw.season_unavailable", asset=asset.name, season=season, error=str(exc))
    return out


def read_seasons(
    asset: Asset,
    seasons: Iterable[int],
    *,
    columns: list[str] | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch and concatenate a per-season asset into one frame."""
    paths = fetch_seasons(asset, seasons, settings=settings, force=force)
    if not paths:
        msg = f"no seasons available for asset {asset.name!r}"
        raise IngestError(msg)
    frames = [pd.read_parquet(p, columns=columns) for _, p in sorted(paths.items())]
    return pd.concat(frames, ignore_index=True)


def read_asset(
    asset: Asset,
    *,
    columns: list[str] | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch and read a whole-history asset."""
    path = fetch_asset(asset, None, settings=settings, force=force)
    return pd.read_parquet(path, columns=columns)
