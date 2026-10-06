"""Is the deployed page actually being deployed?

The weekly workflow can report success while publishing nothing. It did: the
`deploy` job's condition referenced a `workflow_dispatch` input that does not
exist on a schedule trigger, so it skipped on every scheduled run for 16 days.
The scorecard was committed on time, the run was green, and the public page sat
at a single manual deployment from 2026-09-20.

Nothing could have caught that by looking at the run, because the run was fine.
The only way to know is to read the artifact. So this fetches the live page and
asks how old the scorecard behind it is.

The same lesson as :func:`nflproj.ingest.archive.coverage`: a job's exit status
describes the job, not the thing it was supposed to produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import requests

from nflproj.report.html import GENERATED_AT_META

#: How stale the deployed page may be before it counts as broken. The page is
#: rebuilt weekly, so ten days tolerates one late run and catches a cadence
#: that has actually stopped.
DEFAULT_MAX_AGE: Final = timedelta(days=10)

_META = re.compile(
    rf'<meta\s+name="{re.escape(GENERATED_AT_META)}"\s+content="([^"]+)"',
    re.IGNORECASE,
)


class PageHealthError(RuntimeError):
    """Raised when the deployed page cannot be fetched or read."""


@dataclass(frozen=True, slots=True)
class PageHealth:
    """What the live page says about itself."""

    url: str
    generated_at: datetime
    fetched_at: datetime

    @property
    def age(self) -> timedelta:
        return self.fetched_at - self.generated_at

    @property
    def age_days(self) -> float:
        return self.age.total_seconds() / 86400.0

    def stale(self, *, max_age: timedelta = DEFAULT_MAX_AGE) -> bool:
        return self.age > max_age


def fetch_page_health(
    url: str,
    *,
    timeout: float = 30.0,
    now: datetime | None = None,
) -> PageHealth:
    """Read the deployed page's build timestamp.

    Raises rather than returning a default. A page that cannot be fetched, or
    that carries no timestamp, is not evidence of freshness - and treating it
    as "probably fine" is how the original failure stayed invisible.
    """
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        msg = f"could not fetch {url}: {exc}"
        raise PageHealthError(msg) from exc

    if response.status_code != requests.codes.ok:
        msg = f"{url} returned HTTP {response.status_code}"
        raise PageHealthError(msg)

    match = _META.search(response.text)
    if match is None:
        msg = (
            f"{url} carries no {GENERATED_AT_META!r} meta tag. Either the "
            "deployed page predates this check, or it was not built by "
            "`nflproj report`."
        )
        raise PageHealthError(msg)

    try:
        generated = datetime.fromisoformat(match.group(1))
    except ValueError as exc:
        msg = f"{url} has an unparseable build timestamp {match.group(1)!r}"
        raise PageHealthError(msg) from exc

    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)

    return PageHealth(url=url, generated_at=generated, fetched_at=now or datetime.now(UTC))
