"""The plain-language layer.

These tests exist because the copy is a claim about the numbers. If the
translation says "shade the extremes toward the middle" while calibration is
1.0, the page is lying to a reader who cannot check it.
"""

from __future__ import annotations

import pytest

from nflproj.report import interpret as itp
from nflproj.report.interpret import Confidence

# ---------------------------------------------------------------- top N


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.70, Confidence.STRONG),
        (0.55, Confidence.STRONG),
        (0.50, Confidence.MODERATE),
        (0.45, Confidence.MODERATE),
        (0.30, Confidence.WEAK),
    ],
)
def test_top_n_bands(value, expected):
    assert itp.read_top_n(value).confidence is expected


def test_top_n_reports_the_percentage_a_reader_can_check():
    reading = itp.read_top_n(0.522)
    assert "52%" in reading.plain


def test_weak_top_n_does_not_tell_people_to_trust_it():
    action = itp.read_top_n(0.28).action.lower()
    assert "guessing" in action


# ---------------------------------------------------------------- spearman


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.75, Confidence.STRONG), (0.50, Confidence.MODERATE), (0.20, Confidence.WEAK)],
)
def test_spearman_bands(value, expected):
    assert itp.read_spearman(value).confidence is expected


def test_spearman_action_changes_with_the_band():
    strong = itp.read_spearman(0.75).action
    weak = itp.read_spearman(0.20).action
    assert strong != weak


# ---------------------------------------------------------------- calibration


def test_over_dispersed_tells_you_to_shade_toward_the_middle():
    reading = itp.read_calibration(0.85)
    assert "toward the middle" in reading.action
    assert "overstated" in reading.plain


def test_under_dispersed_says_the_opposite():
    reading = itp.read_calibration(1.30)
    assert "understated" in reading.plain
    assert "toward the middle" not in reading.action


def test_well_calibrated_asks_for_no_correction():
    reading = itp.read_calibration(1.02)
    assert reading.confidence is Confidence.STRONG
    assert "face value" in reading.action


def test_calibration_bands_are_symmetric_around_one():
    assert itp.read_calibration(0.92).confidence is itp.read_calibration(1.08).confidence
    assert itp.read_calibration(0.80).confidence is itp.read_calibration(1.20).confidence


def test_the_stated_shortfall_matches_the_slope():
    """A reader can check this arithmetic, so it has to be right."""
    assert "17%" in itp.read_calibration(0.83).plain


# ---------------------------------------------------------------- bias


def test_negative_bias_means_running_high():
    """bias = mean(actual - predicted), so negative means over-predicting."""
    reading = itp.read_bias(-1.47)
    assert "high" in reading.plain
    assert "subtract" in reading.action


def test_positive_bias_means_running_low():
    reading = itp.read_bias(1.47)
    assert "low" in reading.plain
    assert "add" in reading.action


def test_small_bias_is_reported_as_nothing_to_do():
    reading = itp.read_bias(-0.27)
    assert reading.confidence is Confidence.STRONG
    assert reading.action == "Nothing to correct for."


# ---------------------------------------------------------------- mae


def test_mae_against_a_baseline_states_the_direction():
    better = itp.read_mae(4.417, baseline=4.591)
    assert "better than" in better.plain
    worse = itp.read_mae(5.140, baseline=4.591)
    assert "worse than" in worse.plain


def test_mae_without_a_baseline_makes_no_comparison():
    assert "baseline" not in itp.read_mae(4.417).plain


def test_mae_survives_a_zero_baseline():
    """A degenerate baseline must not divide by zero on a public page."""
    assert itp.read_mae(4.0, baseline=0.0).plain


# ---------------------------------------------------------------- slices


@pytest.fixture
def position_rows() -> list[dict[str, object]]:
    return [
        {"position": "QB", "spearman": 0.616, "top_n_hit_rate": 0.527},
        {"position": "TE", "spearman": 0.548, "top_n_hit_rate": 0.451},
        {"position": "WR", "spearman": 0.606, "top_n_hit_rate": 0.549},
        {"position": "RB", "spearman": 0.613, "top_n_hit_rate": 0.562},
    ]


def test_positions_are_ranked_by_ordering_quality(position_rows):
    groups = itp.read_positions(position_rows)
    assert [g.key for g in groups] == ["QB", "RB", "WR", "TE"]


def test_position_guidance_names_the_weakest_position(position_rows):
    guidance = itp.position_guidance(itp.read_positions(position_rows))
    assert "TE" in guidance
    assert "stream" in guidance.lower()


def test_position_guidance_handles_a_single_position():
    groups = itp.read_positions([{"position": "QB", "spearman": 0.6, "top_n_hit_rate": 0.5}])
    assert "QB" in itp.position_guidance(groups)


def test_position_guidance_handles_no_data():
    assert itp.position_guidance([]).startswith("Not enough data")


def test_phase_guidance_flags_a_weaker_september():
    guidance = itp.phase_guidance(0.540, 0.608)
    assert "0.54" in guidance
    assert "September" in guidance


def test_phase_guidance_when_early_season_holds_up():
    assert "unusual" in itp.phase_guidance(0.600, 0.605)


def test_availability_note_states_the_miss_rate():
    note = itp.availability_note(0.746)
    assert "25%" in note
    assert "zero" in note
