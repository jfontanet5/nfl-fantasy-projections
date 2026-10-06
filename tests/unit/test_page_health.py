"""Freshness of the deployed page.

The bug these exist for: the weekly workflow's deploy job referenced
`inputs.deploy`, which does not exist on a schedule trigger, so it skipped on
every scheduled run for 16 days. The runs were green, the scorecard was
committed on time, and the public page sat at one manual deployment. Nothing
noticed because nothing read the page.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests

from nflproj.report.health import PageHealthError, fetch_page_health
from nflproj.report.html import GENERATED_AT_META

BUILT = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
URL = "https://example.invalid/page/"


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


def _page(generated_at: str) -> str:
    return (
        "<!doctype html><html><head>"
        f'<meta name="{GENERATED_AT_META}" content="{generated_at}">'
        "<title>nflproj</title></head><body></body></html>"
    )


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(status_code: int = 200, text: str = "") -> None:
        monkeypatch.setattr(requests, "get", lambda *_a, **_k: _Response(status_code, text))

    return _install


def test_a_fresh_page_is_ok(serve):
    serve(text=_page(BUILT.isoformat()))
    health = fetch_page_health(URL, now=BUILT + timedelta(days=2))
    assert health.age_days == pytest.approx(2.0)
    assert health.stale() is False


def test_the_sixteen_day_freeze_is_caught(serve):
    """The actual outage, to the day it was found."""
    serve(text=_page(datetime(2026, 9, 20, 15, 36, tzinfo=UTC).isoformat()))
    health = fetch_page_health(URL, now=datetime(2026, 10, 6, 18, 12, tzinfo=UTC))
    assert health.age_days == pytest.approx(16.1, abs=0.1)
    assert health.stale() is True


def test_the_bound_is_adjustable(serve):
    serve(text=_page(BUILT.isoformat()))
    health = fetch_page_health(URL, now=BUILT + timedelta(days=9))
    assert health.stale(max_age=timedelta(days=10)) is False
    assert health.stale(max_age=timedelta(days=7)) is True


def test_a_page_with_no_marker_is_an_error_not_a_pass(serve):
    """Absence of evidence must not read as evidence of freshness.

    A page deployed before this check existed has no marker. Treating that as
    healthy is precisely how the original failure stayed invisible, so it
    raises and the operator decides.
    """
    serve(text="<!doctype html><html><head><title>nflproj</title></head></html>")
    with pytest.raises(PageHealthError, match=r"no .*meta tag"):
        fetch_page_health(URL)


def test_a_404_is_an_error(serve):
    serve(status_code=404, text="not found")
    with pytest.raises(PageHealthError, match="404"):
        fetch_page_health(URL)


def test_an_unreachable_page_is_an_error(monkeypatch):
    def _boom(*_a: object, **_k: object) -> None:
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(PageHealthError, match="could not fetch"):
        fetch_page_health(URL)


def test_an_unparseable_timestamp_is_an_error(serve):
    serve(text=_page("last Tuesday"))
    with pytest.raises(PageHealthError, match="unparseable"):
        fetch_page_health(URL)


def test_a_naive_timestamp_is_read_as_utc(serve):
    """Older pages may carry a naive stamp; assume UTC rather than crashing."""
    serve(text=_page("2026-10-06T12:00:00"))
    health = fetch_page_health(URL, now=BUILT + timedelta(days=1))
    assert health.age_days == pytest.approx(1.0)
