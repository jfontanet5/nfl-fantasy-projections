"""Runtime configuration.

Every path the pipeline touches is resolved here so that tests can redirect the
whole data tree at a temporary directory with one environment variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: First season in the nflverse era with complete weekly player stats and
#: play-by-play participation. Earlier seasons exist but have thinner coverage
#: for snap counts and depth charts, so we do not claim support for them.
FIRST_SUPPORTED_SEASON = 2015

#: Positions we project. Kickers and team defenses have different data
#: generating processes and are deliberately out of scope for v1.
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


class Settings(BaseSettings):
    """Process-wide settings, overridable via ``NFLPROJ_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="NFLPROJ_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    reports_dir: Path = Field(default=Path("reports"))

    #: Seasons pulled by default. The upper bound is intentionally open; the
    #: ingest layer clamps to seasons that actually exist upstream.
    first_season: int = Field(default=FIRST_SUPPORTED_SEASON)

    #: Seconds to wait on any single nflverse HTTP request.
    http_timeout: float = Field(default=60.0)

    #: Re-download a raw asset if the cached copy is older than this many hours.
    #: In-season data is revised for several days after each game.
    raw_max_age_hours: float = Field(default=12.0)

    @field_validator("first_season")
    @classmethod
    def _season_supported(cls, v: int) -> int:
        if v < FIRST_SUPPORTED_SEASON:
            msg = (
                f"first_season={v} predates supported coverage; minimum is {FIRST_SUPPORTED_SEASON}"
            )
            raise ValueError(msg)
        return v

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings.

    Cached so that repeated calls are free; tests clear the cache via
    :func:`reset_settings` after mutating the environment.
    """
    return Settings()


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment."""
    get_settings.cache_clear()
