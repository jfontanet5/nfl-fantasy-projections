# Leakage and evaluation honesty

This document is the argument that the numbers in `reports/` mean something. It
records every decision where an easier choice would have produced better-looking
metrics, and why that choice was refused.

## 1. The predictor is never handed the future

Leakage here is prevented structurally, not by discipline. The backtest slices
history itself and strips every post-kickoff column before calling a predictor:

```python
history = panel[(panel.season < season) | ((panel.season == season) & (panel.week < week))]
blinded = strip_outcomes(targets)  # drops fantasy_points, played, targets, carries, attempts
predictor.fit(history)
predictor.predict(blinded)
```

A predictor cannot read the answer because the answer is not in the frame. The
columns removed are listed in `OUTCOME_COLUMNS` (`nflproj/predictors/base.py`).

Five tests defend this, in `tests/unit/test_leakage.py`:

| Test | What it proves |
|---|---|
| `test_strip_outcomes_removes_every_outcome_column` | The blinding is complete. |
| `test_predictor_cannot_read_the_target` | A predictor that tries to read `fantasy_points` raises `KeyError`. |
| `test_history_never_reaches_the_target_week` | The maximum `(season, week)` in history is always strictly less than the target. |
| `test_future_perturbation_does_not_change_past_predictions` | Replacing every outcome at or after week *W* with garbage leaves predictions before *W* bit-identical. |
| `test_perturbation_test_can_actually_fail` | A deliberately leaky control predictor **does** diverge - so a pass above is evidence, not a vacuous assertion. |

That last pair matters. A perturbation test that can never fail proves nothing,
so the suite includes a known-leaky predictor and asserts that it is caught.

## 2. The prediction universe is defined before kickoff

The most common way a fantasy projection evaluation flatters itself is to score
only the players who actually played. That set is unknowable before kickoff.
Conditioning on it deletes every inactive-and-scored-zero case, which are
precisely the cases a projection needs to get right.

Two universes are implemented and both are published:

**`active_recent`** (default, decision-grade). A player is projected in week *W*
if their team has a game that week and they recorded a box-score line in at
least one of their team's previous 3 games. Built entirely from weeks before
*W*. Players who then sit out score `0.0` and remain in the evaluation. About
26% of the universe does not play in a given week.

**`played`** (reported, not decision-grade). Only players with a week-*W* box
score. This is the number most public comparisons implicitly use, so it is
published for comparability and labelled as outcome-conditioned everywhere it
appears.

Activation walks the team's *game* index rather than calendar weeks, so a bye
does not silently consume part of the lookback window.

## 3. Week 1 is out of scope

Knowing who is on which roster in week 1 requires offseason transaction data we
do not yet have a clean point-in-time source for. Carrying each player's prior
season team forward is wrong for everyone who changed teams, and would be a
leakage-adjacent fudge dressed up as a feature.

So week 1 is not projected. Week-1 results are still loaded, because they are
the history every week-2 projection depends on, but they are marked
`projectable=False` and never scored. This costs about 6% of available rows and
removes an entire class of argument about what we knew when.

Revisiting this needs a preseason roster or depth-chart snapshot; it is the
first item in the roadmap.

## 4. Data availability is tiered, not assumed

The schedule is the one upstream table a week-*W* prediction may read week-*W*
rows from - matchups, kickoff times and rest days are published days ahead.

But not everything in it is equally early. nflverse stores the **closing**
betting line, which is only final minutes before kickoff. A projection published
on Wednesday could not have used it. Those columns are tagged in
`LATE_AVAILABILITY_COLUMNS` rather than quietly mixed in:

```python
LATE_AVAILABILITY_COLUMNS = {"team_spread_line", "total_line", "implied_team_total"}
```

Any predictor that consumes them is implicitly claiming a kickoff-time
publication slot, and the scorecard must say so. No baseline currently uses
them.

Injury reports and depth charts are **excluded from v1** for the same reason in
a harsher form: nflverse serves the current state of those tables, not their
state as of any past Friday, so a historical backtest using them would be
reading revisions that did not exist at projection time. Building a true
point-in-time archive requires snapshotting them ourselves, weekly, going
forward - which is why live publication starts now rather than after the
model is finished.

## 5. The target is computed, not inherited

Fantasy points are recomputed from box-score components by `nflproj.scoring`
rather than taken from upstream's `fantasy_points_ppr`. This makes the scoring
rule an explicit, testable object, and gives a free consistency check: an
integration test asserts exact agreement with nflverse across 2015, 2020 and
2024 (17,417 player-weeks, max absolute difference `0.0`).

That check has already paid for itself. It surfaced that our first
implementation omitted kick and punt return touchdowns - 13 rows in 2024, worth
6 points each.

## 6. Upstream drift is a tested failure, not a silent one

While building this, nflverse migrated weekly player stats from the
`player_stats` release tag to `stats_player`. The old tag did not 404 - it kept
serving a frozen 2024 snapshot. A pipeline pointed at it would have looked
completely healthy while having no current-season data at all.

`tests/integration/test_nflverse_contract.py` now asserts that the tag we use
serves the current season, and separately documents that the legacy tag is
stale. The ingest layer records a SHA-256 of every file it downloads, and every
scorecard embeds those hashes in its `provenance` block, so any published number
can be traced to the exact bytes it came from.

## 7. The live path is the backtested path

`project_week` and `run_backtest` apply the same history rule through the same
code. A unit test asserts that projecting week *W* live produces predictions
identical to that week's row in the backtest. If they could drift apart, the
scorecard would be measuring something other than what gets published.

## Known limitations

These are real and currently unaddressed. They are listed here rather than
discovered later by a reader.

- **Week 1 is not projected** (section 3).
- **No injury or depth-chart features** (section 4). This is the largest
  accuracy gap in the current system and is deliberate.
- **Rookies enter the universe only after their first appearance.** Until then
  they are unprojectable, which understates coverage in September.
- **Snap counts are not used.** nflverse keys them on Pro-Football-Reference
  player ids, which need a crosswalk to the GSIS ids everything else uses.
- **Kickers and team defenses are out of scope.** Different data generating
  process; including them would add rows without adding insight.
- **Only full-PPR is scored by default.** Half-PPR and standard are supported by
  `ScoringRules` but not published.
