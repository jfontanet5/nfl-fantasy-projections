"""A published projection bundle: the unit the serving layer loads.

A fantasy projection system is batch-scored. The board for a week is computed
once, before that week kicks off, and then it is a fact about that week - it
does not change as Sunday progresses, and if it did, the whole evaluation
argument in :doc:`../../docs/leakage` would collapse. So the thing the API
serves is an *artifact*, not a live computation, and this module defines it.

The bundle is a directory with two files:

``projections.parquet``
    The board. One row per projected player.

``bundle.json``
    Everything needed to argue that the board is what it claims to be: which
    predictor produced it, for which week, from which upstream files (by
    SHA-256), and what that predictor's measured track record was at the time.

Three properties are worth stating explicitly, because each one exists to stop
a specific way of being wrong.

**The version is a content hash, not a timestamp or a counter.** Rebuilding a
bundle from the same data with the same predictor produces the same version
string, so "is production serving what I think it is" is answerable by
comparison rather than by trust. ``created_at`` is deliberately excluded from
the hash; otherwise every rebuild would look like a new model.

**Loading verifies the hash.** A truncated download, a half-written volume or
an edited parquet makes :func:`load_bundle` raise rather than quietly serving
numbers nobody generated. The API turns that into a failed readiness probe,
which is the correct outcome: a pod holding a corrupt bundle should never
receive traffic.

**The bundle records when it stops being a prediction.** ``valid_from`` is the
week's first kickoff. After that moment the board is a historical record of
what was claimed, not advice, and a caller that cannot tell the difference will
eventually present a settled week as a forecast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from nflproj.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

MANIFEST_NAME: Final = "bundle.json"
PROJECTIONS_NAME: Final = "projections.parquet"

#: Length of the version string. 16 hex characters is 64 bits - collision-proof
#: for anything this will ever hold, and short enough to be a container tag.
VERSION_CHARS: Final = 16

#: Columns the API depends on. Asserted at publish time so a schema change is a
#: publish failure rather than a 500 in production three days later.
REQUIRED_COLUMNS: Final = (
    "season",
    "week",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "prediction",
)


class BundleError(RuntimeError):
    """Raised when a bundle is missing, malformed, or fails its integrity check."""


@dataclass(frozen=True, slots=True)
class BundleMetadata:
    """What a bundle claims about itself."""

    version: str
    predictor: str
    season: int
    week: int
    #: The week's first kickoff, UTC ISO-8601. ``None`` when the schedule had no
    #: kickoff time for the week, which is honest rather than defaulted - a
    #: guessed cutoff is worse than an absent one.
    valid_from: str | None
    players: int
    #: SHA-256 of every upstream file the board was built from.
    raw_assets: dict[str, str] = field(default_factory=dict)
    #: Headline metrics for this predictor, as measured by the scorecard.
    metrics: dict[str, float] = field(default_factory=dict)
    baseline: str | None = None
    #: Excluded from the version hash on purpose: a rebuild is the same model.
    created_at: str = ""

    @property
    def valid_from_dt(self) -> datetime | None:
        if self.valid_from is None:
            return None
        return datetime.fromisoformat(self.valid_from)

    def superseded(self, *, now: datetime | None = None) -> bool:
        """Whether the week this board covers has already started.

        Not an error and not a reason to refuse service - the board is still
        exactly what was published, and the track record depends on it staying
        readable. It is a flag so a caller cannot mistake a settled week for a
        forecast.
        """
        cutoff = self.valid_from_dt
        if cutoff is None:
            return False
        return (now or datetime.now(UTC)) >= cutoff


@dataclass(frozen=True, slots=True)
class Bundle:
    """A loaded, verified bundle."""

    metadata: BundleMetadata
    projections: pd.DataFrame


def _hashable_payload(meta: dict[str, Any]) -> dict[str, Any]:
    """The subset of the metadata the version commits to."""
    return {k: v for k, v in meta.items() if k not in {"version", "created_at", "players"}}


def compute_version(projections: pd.DataFrame, meta: dict[str, Any]) -> str:
    """Content hash over the board and the claims made about it.

    Hashes the frame's canonical CSV rather than its parquet bytes: parquet
    encodes compression settings, writer version and row-group layout, none of
    which are facts about the projections. Two runs that produce identical
    numbers must produce an identical version, or the version means nothing.
    """
    digest = hashlib.sha256()
    ordered = projections.sort_values(["season", "week", "player_id"], kind="mergesort")
    digest.update(ordered[list(REQUIRED_COLUMNS)].to_csv(index=False).encode())
    digest.update(
        json.dumps(_hashable_payload(meta), sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()[:VERSION_CHARS]


def write_bundle(
    root: Path,
    projections: pd.DataFrame,
    *,
    predictor: str,
    season: int,
    week: int,
    valid_from: datetime | None = None,
    raw_assets: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
    baseline: str | None = None,
    now: datetime | None = None,
) -> BundleMetadata:
    """Write a bundle, deriving its version from its contents."""
    missing = [c for c in REQUIRED_COLUMNS if c not in projections.columns]
    if missing:
        msg = f"projections are missing required columns: {missing}"
        raise BundleError(msg)
    if projections.empty:
        msg = "refusing to publish an empty board"
        raise BundleError(msg)
    if valid_from is not None and valid_from.tzinfo is None:
        msg = "valid_from must be timezone-aware; a naive datetime has no defined instant"
        raise BundleError(msg)

    claims: dict[str, Any] = {
        "predictor": predictor,
        "season": int(season),
        "week": int(week),
        "valid_from": None if valid_from is None else valid_from.isoformat(),
        "raw_assets": dict(sorted((raw_assets or {}).items())),
        "metrics": {k: round(float(v), 6) for k, v in sorted((metrics or {}).items())},
        "baseline": baseline,
    }
    metadata = BundleMetadata(
        version=compute_version(projections, claims),
        players=len(projections),
        created_at=(now or datetime.now(UTC)).isoformat(),
        **claims,
    )

    root.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(root / PROJECTIONS_NAME, index=False)
    (root / MANIFEST_NAME).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")

    log.info(
        "bundle.written",
        path=str(root),
        version=metadata.version,
        predictor=predictor,
        season=season,
        week=week,
        players=metadata.players,
    )
    return metadata


def load_bundle(root: Path) -> Bundle:
    """Load a bundle and verify it is what its manifest says it is.

    Raises rather than degrading. A bundle that fails here must not be served:
    the caller turns this into a red readiness probe, so the pod holding it
    receives no traffic instead of answering with numbers of unknown origin.
    """
    manifest_path = root / MANIFEST_NAME
    projections_path = root / PROJECTIONS_NAME
    for path in (manifest_path, projections_path):
        if not path.exists():
            msg = f"bundle at {root} is incomplete: {path.name} is missing"
            raise BundleError(msg)

    try:
        payload = json.loads(manifest_path.read_text())
        metadata = BundleMetadata(**payload)
    except (json.JSONDecodeError, TypeError) as exc:
        msg = f"bundle manifest at {manifest_path} is not readable: {exc}"
        raise BundleError(msg) from exc

    try:
        projections = pd.read_parquet(projections_path)
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt-file path
        msg = f"bundle projections at {projections_path} are not readable: {exc}"
        raise BundleError(msg) from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in projections.columns]
    if missing:
        msg = f"bundle projections are missing required columns: {missing}"
        raise BundleError(msg)

    recomputed = compute_version(projections, _hashable_payload(asdict(metadata)))
    if recomputed != metadata.version:
        msg = (
            f"bundle integrity check failed: manifest claims {metadata.version}, "
            f"contents hash to {recomputed}. The bundle was modified or truncated "
            "after it was published; refusing to serve it."
        )
        raise BundleError(msg)

    log.info(
        "bundle.loaded",
        version=metadata.version,
        predictor=metadata.predictor,
        season=metadata.season,
        week=metadata.week,
        players=metadata.players,
    )
    return Bundle(metadata=metadata, projections=projections)
