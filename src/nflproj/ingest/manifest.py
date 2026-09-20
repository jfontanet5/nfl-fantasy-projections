"""Provenance manifest for the raw data zone.

Every file we pull from upstream gets an entry recording where it came from,
when we fetched it, and what it hashed to. This is what lets a scorecard
published in week 5 be re-derived in week 15 - and what lets us prove a metric
moved because the model changed, not because upstream silently revised a
2019 box score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "_manifest.json"
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One raw asset as we received it."""

    key: str
    url: str
    path: str
    sha256: str
    size_bytes: int
    fetched_at: str

    @property
    def fetched_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.fetched_at)

    def age_hours(self, *, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return (now - self.fetched_at_dt).total_seconds() / 3600.0


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class Manifest:
    """A JSON-backed mapping of asset key -> :class:`ManifestEntry`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / MANIFEST_FILENAME
        self._entries: dict[str, ManifestEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            # A truncated manifest (killed mid-write) must not wedge the
            # pipeline; the raw files are re-fetchable, so start clean.
            return
        for key, payload in raw.items():
            self._entries[key] = ManifestEntry(**payload)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        body = {k: asdict(v) for k, v in sorted(self._entries.items())}
        tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)  # atomic within a filesystem

    def get(self, key: str) -> ManifestEntry | None:
        return self._entries.get(key)

    def record(self, *, key: str, url: str, path: Path) -> ManifestEntry:
        entry = ManifestEntry(
            key=key,
            url=url,
            path=str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            fetched_at=datetime.now(UTC).isoformat(),
        )
        self._entries[key] = entry
        self.save()
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> dict[str, ManifestEntry]:
        """A copy of every recorded entry, keyed by asset key."""
        return dict(self._entries)
