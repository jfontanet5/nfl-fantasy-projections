"""Provenance manifest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nflproj.ingest.manifest import Manifest, ManifestEntry, sha256_file


def _write(root: Path, name: str, body: bytes = b"hello") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_record_and_reload_roundtrip(tmp_path):
    path = _write(tmp_path, "player_stats/x.parquet")
    manifest = Manifest(tmp_path)
    entry = manifest.record(key="player_stats/2024", url="https://example/x", path=path)

    assert entry.sha256 == sha256_file(path)
    assert entry.size_bytes == 5
    assert entry.path == "player_stats/x.parquet"

    reloaded = Manifest(tmp_path)
    assert "player_stats/2024" in reloaded
    assert reloaded.get("player_stats/2024") == entry


def test_hash_changes_when_content_changes(tmp_path):
    path = _write(tmp_path, "a.parquet", b"one")
    manifest = Manifest(tmp_path)
    first = manifest.record(key="a", url="u", path=path)

    path.write_bytes(b"two")
    second = manifest.record(key="a", url="u", path=path)
    assert first.sha256 != second.sha256


def test_corrupt_manifest_does_not_wedge_the_pipeline(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "_manifest.json").write_text("{not json")
    manifest = Manifest(tmp_path)
    assert len(manifest) == 0

    path = _write(tmp_path, "b.parquet")
    manifest.record(key="b", url="u", path=path)
    assert len(Manifest(tmp_path)) == 1


def test_age_hours():
    entry = ManifestEntry(
        key="k",
        url="u",
        path="p",
        sha256="x",
        size_bytes=1,
        fetched_at=(datetime.now(UTC) - timedelta(hours=5)).isoformat(),
    )
    assert 4.9 < entry.age_hours() < 5.1


def test_keys_are_sorted(tmp_path):
    manifest = Manifest(tmp_path)
    for name in ("c", "a", "b"):
        manifest.record(key=name, url="u", path=_write(tmp_path, f"{name}.parquet"))
    assert manifest.keys() == ["a", "b", "c"]
