"""Baseline predictors."""

from __future__ import annotations

import pandas as pd
import pytest

from nflproj.predictors.baselines import (
    PUBLISHED_PREDICTOR_NAME,
    UNADJUSTED_PREDICTOR_NAME,
    AvailabilityAdjusted,
    AvailabilityWeighted,
    ExponentialMean,
    LastGame,
    PositionMean,
    RollingMean,
    SeasonDecayedMean,
    SeasonToDateMean,
    default_baselines,
)


@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2024, 2024],
            "week": [17, 1, 2, 3, 3],
            "player_id": ["p1", "p1", "p1", "p1", "p2"],
            "position": ["WR", "WR", "WR", "WR", "WR"],
            "fantasy_points": [100.0, 6.0, 0.0, 12.0, 8.0],
            "played": [True, True, False, True, True],
        }
    )


@pytest.fixture
def targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [4, 4, 4],
            "player_id": ["p1", "p2", "brand_new"],
            "position": ["WR", "WR", "TE"],
        }
    )


def test_season_to_date_ignores_prior_seasons(history, targets):
    """The 100-point 2023 game must not touch a 2024 season-to-date average."""
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(targets)
    assert out.iloc[0] == pytest.approx((6.0 + 0.0 + 12.0) / 3)


def test_include_dnp_changes_the_estimate(history, targets):
    with_dnp = SeasonToDateMean(include_dnp=True)
    without = SeasonToDateMean(include_dnp=False)
    with_dnp.fit(history)
    without.fit(history)
    assert with_dnp.predict(targets).iloc[0] == pytest.approx(6.0)
    assert without.predict(targets).iloc[0] == pytest.approx(9.0)


def test_unknown_player_falls_back_to_the_position_prior(history, targets):
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(targets)
    # No TE history at all in the target season, so the global prior applies.
    assert out.iloc[2] > 0.0
    assert out.notna().all()


def test_shrinkage_pulls_a_thin_sample_toward_the_position_mean(history, targets):
    unshrunk = SeasonToDateMean()
    shrunk = SeasonToDateMean(shrinkage_games=4.0)
    unshrunk.fit(history)
    shrunk.fit(history)

    # p2 has a single 8.0 game; the WR prior this season is lower.
    position_prior = history[history["season"] == 2024]["fantasy_points"].mean()
    raw = unshrunk.predict(targets).iloc[1]
    pulled = shrunk.predict(targets).iloc[1]
    assert abs(pulled - position_prior) < abs(raw - position_prior)


def test_last_game_is_the_most_recent_result(history, targets):
    pred = LastGame()
    pred.fit(history)
    assert pred.predict(targets).iloc[0] == pytest.approx(12.0)


def test_rolling_mean_uses_only_the_window(history, targets):
    pred = RollingMean(window=2)
    pred.fit(history)
    # p1's last two games: 0.0 and 12.0.
    assert pred.predict(targets).iloc[0] == pytest.approx(6.0)


def test_ewma_weights_recent_games_more(history, targets):
    ewma = ExponentialMean(halflife=1.0)
    flat = RollingMean(window=3)
    ewma.fit(history)
    flat.fit(history)
    # p1's sequence is 6, 0, 12; a recency-weighted mean must exceed the flat one.
    assert ewma.predict(targets).iloc[0] > flat.predict(targets).iloc[0]


def test_position_mean_is_constant_within_a_position(history, targets):
    pred = PositionMean()
    pred.fit(history)
    out = pred.predict(targets)
    assert out.iloc[0] == pytest.approx(out.iloc[1])


def test_predictions_align_to_target_index(history, targets):
    shuffled = targets.sample(frac=1.0, random_state=1)
    pred = SeasonToDateMean()
    pred.fit(history)
    out = pred.predict(shuffled)
    assert out.index.equals(shuffled.index)


def test_empty_history_still_produces_numbers(targets):
    empty = pd.DataFrame(
        columns=["season", "week", "player_id", "position", "fantasy_points", "played"]
    )
    for predictor in default_baselines():
        predictor.fit(empty)
        out = predictor.predict(targets)
        assert out.notna().all(), f"{predictor.name} emitted nulls on empty history"


def test_default_baseline_names_are_unique():
    names = [p.name for p in default_baselines()]
    assert len(names) == len(set(names))


# ------------------------------------------------- availability decomposition


@pytest.fixture
def availability_history() -> pd.DataFrame:
    """Two players with identical scoring when they play, different availability."""
    rows = []
    for week in range(1, 9):
        # Ever-present: plays every week, scores 10.
        rows.append(
            {
                "season": 2024,
                "week": week,
                "player_id": "iron",
                "position": "RB",
                "fantasy_points": 10.0,
                "played": True,
            }
        )
        # Fragile: same 10 when he plays, but misses half his weeks.
        played = week % 2 == 0
        rows.append(
            {
                "season": 2024,
                "week": week,
                "player_id": "fragile",
                "position": "RB",
                "fantasy_points": 10.0 if played else 0.0,
                "played": played,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def availability_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [9, 9, 9],
            "player_id": ["iron", "fragile", "unknown"],
            "position": ["RB", "RB", "RB"],
        }
    )


def test_availability_discounts_the_player_who_misses_time(
    availability_history, availability_targets
):
    """Identical scoring when they play; the difference must be availability."""
    pred = AvailabilityWeighted()
    pred.fit(availability_history)
    out = pred.predict(availability_targets)
    assert out.iloc[0] > out.iloc[1]


def test_scoring_term_ignores_missed_games(availability_history, availability_targets):
    """The fragile player's zeros must not drag his points-when-playing estimate.

    This is the defect the decomposition exists to address: a blended average
    would halve his number, when he in fact scores exactly what the other does.
    """
    pred = AvailabilityWeighted(availability_prior_games=0.0, availability_window=99)
    pred.fit(availability_history)
    out = pred.predict(availability_targets)
    # 10.0 scoring x 0.5 availability - the zeros priced once, not twice.
    assert out.iloc[1] == pytest.approx(5.0, abs=0.2)


def test_a_run_of_absences_never_projects_exactly_zero():
    """No rostered player is a certainty to sit; shrinkage must prevent 0.0."""
    never = pd.DataFrame(
        {
            "season": 2024,
            "week": range(1, 9),
            "player_id": "absent",
            "position": "RB",
            "fantasy_points": 0.0,
            "played": False,
        }
    )
    present = pd.DataFrame(
        {
            "season": 2024,
            "week": range(1, 9),
            "player_id": "other",
            "position": "RB",
            "fantasy_points": 12.0,
            "played": True,
        }
    )
    pred = AvailabilityWeighted()
    pred.fit(pd.concat([never, present], ignore_index=True))
    out = pred.predict(
        pd.DataFrame({"season": [2024], "week": [9], "player_id": ["absent"], "position": ["RB"]})
    )
    assert out.iloc[0] > 0.0


def test_unknown_player_falls_back_without_nulls(availability_history, availability_targets):
    pred = AvailabilityWeighted()
    pred.fit(availability_history)
    out = pred.predict(availability_targets)
    assert out.notna().all()
    assert out.iloc[2] > 0.0


def test_empty_history_is_survivable(availability_targets):
    pred = AvailabilityWeighted()
    pred.fit(
        pd.DataFrame(
            columns=["season", "week", "player_id", "position", "fantasy_points", "played"]
        )
    )
    assert pred.predict(availability_targets).notna().all()


def test_parameters_appear_in_the_name():
    """Variants must be distinguishable, or the harness rejects the sweep."""
    a = AvailabilityWeighted(availability_window=4, availability_prior_games=1.0)
    b = AvailabilityWeighted(availability_window=8, availability_prior_games=1.0)
    c = AvailabilityWeighted(availability_window=4, availability_prior_games=5.0)
    assert len({a.name, b.name, c.name}) == 3


# ------------------------------------------------- season decay


@pytest.fixture
def two_season_history() -> pd.DataFrame:
    """A player who was excellent last season and ordinary this one."""
    rows = []
    for week in range(14, 19):
        rows.append(
            {
                "season": 2024,
                "week": week,
                "player_id": "p1",
                "position": "RB",
                "fantasy_points": 20.0,
                "played": True,
            }
        )
    rows.append(
        {
            "season": 2025,
            "week": 1,
            "player_id": "p1",
            "position": "RB",
            "fantasy_points": 5.0,
            "played": True,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def decay_targets() -> pd.DataFrame:
    return pd.DataFrame({"season": [2025], "week": [2], "player_id": ["p1"], "position": ["RB"]})


def test_no_decay_reproduces_the_plain_ewma(two_season_history, decay_targets):
    """`season_decay=1.0` must be exactly ExponentialMean, not merely close.

    That equivalence is what makes this a strict generalisation: the published
    gain is attributable to the discount alone, not to an incidental difference
    in how the weighted mean is computed.
    """
    plain = ExponentialMean(halflife=3.0)
    decayed = SeasonDecayedMean(halflife=3.0, season_decay=1.0)
    plain.fit(two_season_history)
    decayed.fit(two_season_history)
    assert decayed.predict(decay_targets).iloc[0] == pytest.approx(
        plain.predict(decay_targets).iloc[0], abs=1e-9
    )


def test_decay_pulls_toward_the_current_season(two_season_history, decay_targets):
    """Last season was 20s, this season a 5. More discount must mean lower."""
    undiscounted = SeasonDecayedMean(halflife=3.0, season_decay=1.0)
    discounted = SeasonDecayedMean(halflife=3.0, season_decay=0.3)
    undiscounted.fit(two_season_history)
    discounted.fit(two_season_history)
    assert discounted.predict(decay_targets).iloc[0] < undiscounted.predict(decay_targets).iloc[0]


def test_full_discount_uses_only_the_current_season(two_season_history, decay_targets):
    pred = SeasonDecayedMean(halflife=3.0, season_decay=0.0)
    pred.fit(two_season_history)
    assert pred.predict(decay_targets).iloc[0] == pytest.approx(5.0)


def test_a_player_with_no_current_season_games_falls_back(decay_targets):
    """At full discount his whole history weighs nothing; he must not be null."""
    stale = pd.DataFrame(
        {
            "season": 2024,
            "week": range(1, 6),
            "player_id": "p1",
            "position": "RB",
            "fantasy_points": 20.0,
            "played": True,
        }
    )
    other = pd.DataFrame(
        {
            "season": 2025,
            "week": [1],
            "player_id": ["p2"],
            "position": ["RB"],
            "fantasy_points": [8.0],
            "played": [True],
        }
    )
    pred = SeasonDecayedMean(halflife=3.0, season_decay=0.0)
    pred.fit(pd.concat([stale, other], ignore_index=True))
    out = pred.predict(decay_targets)
    assert out.notna().all()
    assert out.iloc[0] > 0.0


def test_decay_appears_in_the_name():
    a = SeasonDecayedMean(season_decay=0.5)
    b = SeasonDecayedMean(season_decay=0.3)
    assert a.name != b.name
    assert "0.5" in a.name


def test_the_published_predictor_is_in_the_slate():
    """The page publishes a predictor the scorecard also scores."""
    assert PUBLISHED_PREDICTOR_NAME in {p.name for p in default_baselines()}


# ------------------------------------------------- availability adjustment


@pytest.fixture
def adjust_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * 4,
            "week": [4] * 4,
            "player_id": ["healthy", "out", "doubtful", "questionable"],
            "position": ["WR"] * 4,
            "report_status": [None, "Out", "Doubtful", "Questionable"],
        }
    )


@pytest.fixture
def adjust_history() -> pd.DataFrame:
    rows = []
    for pid in ("healthy", "out", "doubtful", "questionable"):
        for week in (1, 2, 3):
            rows.append(
                {
                    "season": 2026,
                    "week": week,
                    "player_id": pid,
                    "position": "WR",
                    "fantasy_points": 10.0,
                    "played": True,
                }
            )
    return pd.DataFrame(rows)


def test_an_out_player_projects_exactly_zero(adjust_history, adjust_targets):
    """The defect this exists for: Justin Jefferson ranked 58 at 12.3 while Out."""
    pred = AvailabilityAdjusted(SeasonDecayedMean())
    pred.fit(adjust_history)
    out = pred.predict(adjust_targets)
    assert out.iloc[1] == 0.0


def test_the_adjustment_is_monotone_in_severity(adjust_history, adjust_targets):
    pred = AvailabilityAdjusted(SeasonDecayedMean())
    pred.fit(adjust_history)
    out = pred.predict(adjust_targets)
    healthy, gone, doubtful, questionable = out.tolist()
    assert gone < doubtful < questionable < healthy


def test_an_undesignated_player_is_untouched(adjust_history, adjust_targets):
    """Wrapping must be free where there is no evidence to act on."""
    inner = SeasonDecayedMean()
    wrapped = AvailabilityAdjusted(SeasonDecayedMean())
    inner.fit(adjust_history)
    wrapped.fit(adjust_history)
    assert wrapped.predict(adjust_targets).iloc[0] == pytest.approx(
        inner.predict(adjust_targets).iloc[0]
    )


def test_the_wrapper_is_a_no_op_without_the_status_column(adjust_history, adjust_targets):
    """Over 2015-2025 the archive reaches nothing, so the published scorecard
    must be bit-identical to the unadjusted one. The adjustment can only ever
    act where there is point-in-time evidence that it should.
    """
    bare = adjust_targets.drop(columns=["report_status"])
    inner = SeasonDecayedMean()
    wrapped = AvailabilityAdjusted(SeasonDecayedMean())
    inner.fit(adjust_history)
    wrapped.fit(adjust_history)
    pd.testing.assert_series_equal(wrapped.predict(bare), inner.predict(bare), check_names=False)


def test_all_null_designations_are_also_a_no_op(adjust_history, adjust_targets):
    """The Tuesday case: the column exists, no report has been published."""
    blank = adjust_targets.assign(report_status=None)
    inner = SeasonDecayedMean()
    wrapped = AvailabilityAdjusted(SeasonDecayedMean())
    inner.fit(adjust_history)
    wrapped.fit(adjust_history)
    pd.testing.assert_series_equal(wrapped.predict(blank), inner.predict(blank), check_names=False)


def test_the_name_marks_the_adjustment():
    pred = AvailabilityAdjusted(SeasonDecayedMean())
    assert pred.name == "season_decayed_hl3_d0.5+avail"
    assert pred.name == PUBLISHED_PREDICTOR_NAME


def test_the_published_predictor_is_the_adjusted_one():
    """The page and the API publish availability-adjusted numbers."""
    names = {p.name for p in default_baselines()}
    assert PUBLISHED_PREDICTOR_NAME in names
    assert UNADJUSTED_PREDICTOR_NAME in names, "keep both, so the delta is attributable"


def test_the_index_is_preserved(adjust_history, adjust_targets):
    shuffled = adjust_targets.sample(frac=1.0, random_state=3)
    pred = AvailabilityAdjusted(SeasonDecayedMean())
    pred.fit(adjust_history)
    assert pred.predict(shuffled).index.equals(shuffled.index)


def test_custom_factors_override_the_prior(adjust_history, adjust_targets):
    """The factors are a stated prior; a sweep must be able to replace them."""
    pred = AvailabilityAdjusted(SeasonDecayedMean(), factors={"questionable": 0.5})
    pred.fit(adjust_history)
    out = pred.predict(adjust_targets)
    # Only questionable is remapped; Out falls back to 1.0 under these factors.
    assert out.iloc[3] == pytest.approx(out.iloc[0] * 0.5)


def test_the_wrapper_composes_with_any_predictor(adjust_history, adjust_targets):
    """It will wrap XGBoost unchanged; that is the point of a wrapper."""
    for inner in (SeasonToDateMean(), ExponentialMean(halflife=3.0), PositionMean()):
        pred = AvailabilityAdjusted(inner)
        pred.fit(adjust_history)
        out = pred.predict(adjust_targets)
        assert out.iloc[1] == 0.0, f"{inner.name} not zeroed when Out"
        assert out.notna().all()
