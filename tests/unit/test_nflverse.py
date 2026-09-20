"""Ingest layer, with the network faked out.

The live-upstream behaviour is covered by ``tests/integration``. What matters
here is the logic around the download - URL construction, cache freshness,
error handling - which does not need a network to be wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from nflproj.config import Settings
from nflproj.ingest import nflverse as nv
from nflproj.ingest.manifest import Manifest


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")


@pytest.fixture
def parquet_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "seed.parquet"
    pd.DataFrame({"season": [2024, 2024], "week": [1, 2], "x": [1.0, 2.0]}).to_parquet(path)
    return bytes(path.read_bytes())


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch, parquet_bytes: bytes) -> dict[str, Any]:
    """Patch requests.get and record every URL it was asked for."""
    state: dict[str, Any] = {"calls": [], "status": 200}

    def _get(url: str, **_kwargs: Any) -> _FakeResponse:
        state["calls"].append(url)
        return _FakeResponse(state["status"], parquet_bytes)

    monkeypatch.setattr(requests, "get", _get)
    return state


# ---------------------------------------------------------------- Asset


def test_per_season_filename_and_url():
    asset = nv.Asset(name="x", release="rel", stem="stem")
    assert asset.filename(2024) == "stem_2024.parquet"
    assert asset.url(2024).endswith("/rel/stem_2024.parquet")
    assert asset.key(2024) == "x/2024"


def test_whole_history_filename_and_url():
    asset = nv.Asset(name="x", release="rel", stem="stem", per_season=False)
    assert asset.filename(None) == "stem.parquet"
    assert asset.key(None) == "x"


def test_per_season_asset_requires_a_season():
    with pytest.raises(ValueError, match="is per-season"):
        nv.PLAYER_STATS.filename(None)


def test_player_stats_points_at_the_migrated_release_tag():
    """The legacy `player_stats` tag froze at 2024 without 404ing."""
    assert nv.PLAYER_STATS.release == "stats_player"


# ---------------------------------------------------------------- fetch


def test_fetch_writes_the_file_and_records_provenance(settings, fake_get):
    path = nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    assert path.exists()
    assert len(fake_get["calls"]) == 1

    entry = Manifest(settings.raw_dir).get("player_stats/2024")
    assert entry is not None
    assert len(entry.sha256) == 64
    assert entry.size_bytes == path.stat().st_size


def test_fresh_cache_is_reused_without_a_request(settings, fake_get):
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    assert len(fake_get["calls"]) == 1, "a fresh cached asset was re-downloaded"


def test_force_bypasses_a_fresh_cache(settings, fake_get):
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings, force=True)
    assert len(fake_get["calls"]) == 2


def test_stale_cache_is_refetched(tmp_path, fake_get):
    """In-season data is revised for days; the cache must expire."""
    settings = Settings(data_dir=tmp_path / "d", reports_dir=tmp_path / "r", raw_max_age_hours=0.0)
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)
    assert len(fake_get["calls"]) == 2


def test_non_200_raises_and_leaves_no_partial_file(settings, fake_get):
    fake_get["status"] = 404
    with pytest.raises(nv.IngestError, match="404"):
        nv.fetch_asset(nv.PLAYER_STATS, 2099, settings=settings)

    stray = list(settings.raw_dir.rglob("*.part"))
    assert not stray, f"a partial download was left behind: {stray}"
    assert Manifest(settings.raw_dir).get("player_stats/2099") is None


def test_network_error_is_wrapped(settings, monkeypatch):
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise requests.RequestException("connection reset")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(nv.IngestError, match="failed to fetch"):
        nv.fetch_asset(nv.PLAYER_STATS, 2024, settings=settings)


# ---------------------------------------------------------------- multi-season


def test_fetch_seasons_skips_an_unavailable_season(settings, monkeypatch, parquet_bytes):
    def _get(url: str, **_kwargs: Any) -> _FakeResponse:
        status = 404 if "2099" in url else 200
        return _FakeResponse(status, parquet_bytes)

    monkeypatch.setattr(requests, "get", _get)
    paths = nv.fetch_seasons(nv.PLAYER_STATS, [2024, 2099], settings=settings)
    assert 2024 in paths
    assert 2099 not in paths, "a missing season must be skipped, not fatal"


@pytest.mark.usefixtures("fake_get")
def test_read_seasons_concatenates_in_season_order(settings):
    frame = nv.read_seasons(nv.PLAYER_STATS, [2023, 2024], settings=settings)
    assert len(frame) == 4  # two rows per season fixture
    assert list(frame.columns) == ["season", "week", "x"]


def test_read_seasons_raises_when_nothing_is_available(settings, monkeypatch, parquet_bytes):
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeResponse(404, parquet_bytes))
    with pytest.raises(nv.IngestError, match="no seasons available"):
        nv.read_seasons(nv.PLAYER_STATS, [2098, 2099], settings=settings)


def test_read_asset_handles_whole_history_assets(settings, fake_get):
    frame = nv.read_asset(nv.SCHEDULES, settings=settings)
    assert len(frame) == 2
    assert fake_get["calls"][0].endswith("/schedules/games.parquet")


@pytest.mark.usefixtures("fake_get")
def test_column_projection_is_passed_through(settings):
    frame = nv.read_seasons(nv.PLAYER_STATS, [2024], columns=["season"], settings=settings)
    assert list(frame.columns) == ["season"]
