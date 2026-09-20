"""The report page.

This page is the public face of the project, so the tests focus on the ways a
generated page embarrasses you: unescaped names, a board longer than the
decision it serves, a page that breaks when there is nothing to project, and a
document that claims a metric the scorecard does not contain.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import pytest

from nflproj.report import html as rh


def _scorecard_row(predictor: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "predictor": predictor,
        "n_rows": 1000,
        "n_weeks": 20,
        "mae": 4.417,
        "rmse": 6.331,
        "bias": -0.272,
        "spearman": 0.596,
        "top_n_hit_rate": 0.522,
        "calibration_slope": 0.846,
    }
    row.update(overrides)
    return row


@pytest.fixture
def scorecard() -> dict[str, Any]:
    return {
        "generated_at": "2026-09-20T12:00:00+00:00",
        "schema_version": "1",
        "universe": "active_recent",
        "baseline": "season_to_date_mean",
        "seasons": [2015, 2025],
        "n_predictions": 492485,
        "provenance": {
            "raw_assets": {"player_stats/2024": "a" * 64},
            "played_rate": 0.746,
        },
        "slices": {
            "overall": [
                _scorecard_row("ewma_hl3"),
                _scorecard_row("season_to_date_mean", mae=4.591),
            ],
            "by_position": [
                _scorecard_row("ewma_hl3", position=p, spearman=s)
                for p, s in (("QB", 0.616), ("RB", 0.613), ("WR", 0.606), ("TE", 0.548))
            ],
            "by_season_phase": [
                _scorecard_row("ewma_hl3", season_phase="weeks_2_4", spearman=0.540),
                _scorecard_row("ewma_hl3", season_phase="weeks_5_plus", spearman=0.608),
            ],
            "by_week": [
                _scorecard_row("ewma_hl3", season=2024, week=w, mae_skill=skill, mae=4.4 + w / 100)
                for w, skill in ((2, 0.08), (3, -0.05), (4, 0.12), (5, 0.03))
            ],
        },
    }


@pytest.fixture
def projections() -> pd.DataFrame:
    rows = []
    for position, count in (("QB", 15), ("RB", 30), ("WR", 40), ("TE", 14)):
        for i in range(count):
            rows.append(
                {
                    "season": 2026,
                    "week": 3,
                    "player_id": f"{position}-{i}",
                    "player_display_name": f"{position} Player {i}",
                    "position": position,
                    "team": "BUF",
                    "opponent_team": "LAC",
                    "predictor": "ewma_hl3",
                    "prediction": float(count - i),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def data(scorecard, projections, tmp_path) -> rh.ReportData:
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    return rh.build_report_data(
        scorecard_path=path,
        projections=projections,
        season=2026,
        week=3,
        predictor="ewma_hl3",
    )


# ---------------------------------------------------------------- structure


def test_document_is_a_complete_html_page(data):
    doc = rh.render_document(data)
    assert doc.startswith("<!doctype html>")
    assert '<html lang="en">' in doc
    assert doc.rstrip().endswith("</html>")
    assert "viewport-fit=cover" in doc


def test_body_omits_the_document_skeleton(data):
    """The Artifact pipeline supplies its own skeleton; duplicating it breaks the page."""
    body = rh.render_body(data)
    assert "<!doctype" not in body.lower()
    assert "<html" not in body.lower()
    assert "<body" not in body.lower()
    assert "<title>" in body
    assert "<style>" in body


def test_both_renderings_share_the_same_content(data):
    content = rh.render_content(data)
    assert content in rh.render_body(data)
    assert content in rh.render_document(data)


def test_page_declares_both_themes(data):
    doc = rh.render_document(data)
    assert "prefers-color-scheme: dark" in doc
    assert '[data-theme="dark"]' in doc
    assert ':root:not([data-theme="light"])' in doc


def test_every_fact_survives_without_javascript(data):
    """JS may add the chart and the controls; it must never be the only source
    of a number the page states."""
    content = rh.render_content(data)
    stripped = re.sub(r"<script.*?</script>", "", content, flags=re.S | re.I)
    # The board, the readings and the weekly series all remain.
    assert "QB Player 0" in stripped
    assert "Start/sit accuracy" in stripped
    assert "Week-by-week numbers" in stripped
    assert "borderline calls" in stripped


def test_controls_are_hidden_until_script_runs(data):
    """A control that does nothing without JS must not be shown."""
    content = rh.render_content(data)
    assert 'id="board-controls" hidden' in content


def test_chart_library_is_pinned(data):
    doc = rh.render_document(data)
    assert f"Chart.js/{rh.CHART_LIB_VERSION}/" in doc
    assert "@latest" not in doc


def test_no_unverified_integrity_hash_is_emitted(data):
    """A wrong SRI hash blocks the script silently; absent is safer than guessed."""
    doc = rh.render_document(data)
    if rh.CHART_LIB_SRI is None:
        assert "integrity=" not in doc
    else:
        assert f'integrity="{rh.CHART_LIB_SRI}"' in doc


def test_script_has_no_unsubstituted_tokens(data):
    doc = rh.render_document(data)
    assert "__POSITIVE" not in doc
    assert "__NEGATIVE" not in doc


# ---------------------------------------------------------------- board


def test_board_is_capped_at_the_starter_tier(data):
    content = rh.render_content(data)
    # QB tier is 12; the fixture supplies 15 players.
    assert "QB Player 0" in content
    assert "QB Player 11" in content
    assert "QB Player 12" not in content


def test_board_is_ordered_by_projection(data):
    content = rh.render_content(data)
    first = content.index("QB Player 0")
    second = content.index("QB Player 1<")
    assert first < second


def test_every_position_appears(data):
    content = rh.render_content(data)
    for position in ("QB", "RB", "WR", "TE"):
        assert f"{position} Player 0" in content


def test_missing_projections_do_not_break_the_page(scorecard, tmp_path):
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    data = rh.build_report_data(
        scorecard_path=path, projections=None, season=2026, week=3, predictor="ewma_hl3"
    )
    content = rh.render_content(data)
    assert "No projections available" in content
    # The track record is still worth publishing without a board.
    assert "How much to trust it" in content


def test_over_dispersion_warning_appears_when_calibration_is_low(data):
    assert "spread wider than reality" in rh.render_content(data)


def test_no_over_dispersion_warning_when_well_calibrated(scorecard, projections, tmp_path):
    for row in scorecard["slices"]["overall"]:
        row["calibration_slope"] = 1.01
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    data = rh.build_report_data(
        scorecard_path=path,
        projections=projections,
        season=2026,
        week=3,
        predictor="ewma_hl3",
    )
    assert "spread wider than reality" not in rh.render_content(data)


# ---------------------------------------------------------------- safety


def test_player_names_are_escaped(scorecard, tmp_path):
    hostile = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "player_id": "x",
                "player_display_name": '<script>alert("xss")</script>',
                "position": "QB",
                "team": "BUF",
                "opponent_team": "LAC",
                "predictor": "ewma_hl3",
                "prediction": 20.0,
            }
        ]
    )
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    data = rh.build_report_data(
        scorecard_path=path, projections=hostile, season=2026, week=3, predictor="ewma_hl3"
    )
    content = rh.render_content(data)
    # The name survives only in escaped form...
    assert "&lt;script&gt;" in content
    # ...and adds no live <script> of its own. The page emits exactly two: the
    # JSON island and the behaviour script.
    assert content.count("<script") == 2


def test_unknown_predictor_fails_loudly(scorecard, projections, tmp_path):
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    data = rh.build_report_data(
        scorecard_path=path,
        projections=projections,
        season=2026,
        week=3,
        predictor="does_not_exist",
    )
    with pytest.raises(KeyError, match="does_not_exist"):
        rh.render_content(data)


# ---------------------------------------------------------------- content


def test_the_verdict_leads_with_plain_language(data):
    content = rh.render_content(data)
    lede = content[: content.index("This week&rsquo;s board")]
    assert "borderline calls" in lede
    assert "stop deliberating" in lede


def test_weakest_position_is_named_for_the_reader(data):
    assert "make it TE" in rh.render_content(data)


def test_methodology_is_present_but_collapsed(data):
    content = rh.render_content(data)
    assert "<details" in content
    assert "unknowable in advance" in content


def test_provenance_hashes_are_published(data):
    assert "player_stats/2024" in rh.render_content(data)


def test_skill_against_the_baseline_is_stated(data):
    """4.417 against a 4.591 baseline is a 3.8% improvement."""
    assert "3.8% better than" in rh.render_content(data)


# ---------------------------------------------------------------- trend


def test_trend_reports_the_win_rate(data):
    """Three of the four fixture weeks beat the baseline."""
    content = rh.render_content(data)
    assert '3<span class="stat-of">/4</span>' in content
    assert "75%" in content


def test_trend_names_the_worst_week(data):
    content = rh.render_content(data)
    assert "-5.0%" in content
    assert "2024 W3" in content


def test_trend_series_is_chronological(data):
    series = rh._weekly_series(data.scorecard, "ewma_hl3")
    weeks = [int(r["week"]) for r in series]
    assert weeks == sorted(weeks)


def test_trend_losses_are_stated_not_hidden(data):
    """The point of publishing a track record is the weeks it lost."""
    assert "1 of them" in rh.render_content(data) or "there are 1" in rh.render_content(data)


def test_page_without_a_by_week_slice_still_renders(scorecard, projections, tmp_path):
    """An older scorecard must not break the page."""
    del scorecard["slices"]["by_week"]
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    data = rh.build_report_data(
        scorecard_path=path,
        projections=projections,
        season=2026,
        week=3,
        predictor="ewma_hl3",
    )
    content = rh.render_content(data)
    assert "Week by week" not in content
    assert "QB Player 0" in content


def test_json_island_cannot_be_broken_out_of():
    """`</script>` inside a JSON island closes it, whatever the JSON grammar says."""
    out = rh._json_island({"x": "</script><script>alert(1)</script>"})
    assert "</script>" not in out
    assert json.loads(out)["x"] == "</script><script>alert(1)</script>"


# ---------------------------------------------------------------- SRI


def test_a_well_formed_sri_is_emitted(monkeypatch):
    monkeypatch.setattr(rh, "CHART_LIB_SRI", "sha512-" + "A" * 86 + "==")
    assert 'integrity="sha512-' in rh._chart_lib_tag()


@pytest.mark.parametrize(
    "bad",
    [
        "A" * 86 + "==",  # the bare openssl output - the easy mistake
        "sha512 " + "A" * 86,  # space instead of a hyphen
        "md5-abc",  # algorithm SRI does not allow
        "sha512-not valid base64!",
        "",
    ],
)
def test_a_malformed_sri_fails_the_build(monkeypatch, bad):
    """A malformed integrity value blocks the script as silently as a wrong one.

    Anything set but not well-formed raises, empty string included: the way to
    mean "no hash" is None, and a blank is far more likely to be a mistake.
    """
    monkeypatch.setattr(rh, "CHART_LIB_SRI", bad)
    with pytest.raises(ValueError, match="not a valid Subresource Integrity"):
        rh._chart_lib_tag()


def test_the_error_names_the_likely_mistake(monkeypatch):
    monkeypatch.setattr(rh, "CHART_LIB_SRI", "A" * 86 + "==")
    with pytest.raises(ValueError, match="missing the 'sha512-' prefix"):
        rh._chart_lib_tag()
