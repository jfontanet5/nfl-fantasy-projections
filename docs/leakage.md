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
reading revisions that did not exist at projection time. Section 4a is what we
are doing about it.

## 4a. Some tables can only be known by having written them down

A table that is served "as it stands" is not automatically unusable. The
question is whether upstream keeps its own history. We looked at both tables we
want, and they answered differently — which changed what had to be built.

**Depth charts already are a point-in-time log.** Every row carries a `dt`
observation timestamp, the file is cumulative rather than overwritten, and there
are 188 distinct scrape times between March and September in the 2026 file
alone — roughly twice a day. So the as-of view is a filter, not a recording
problem, and `depth_chart_as_of` is eleven lines that select the latest *whole
scrape* before a cutoff. No archive is needed and building one would duplicate
megabytes a week to reconstruct what upstream already tells us. The one subtlety
is *whole scrape*: taking the latest row per player would splice two scrapes
together and produce a depth chart that never existed.

**Injury reports are not.** The file is keyed on `(season, week)` with no
timestamp anywhere in it, and it is rewritten in place as the week progresses.
Wednesday's *limited participant* becomes Friday's *questionable* becomes
Sunday's *inactive*, and each overwrite destroys its predecessor. By Tuesday the
only surviving version is the post-game one — which is the single state a
projection may never see, because it encodes who actually played.

This is the asymmetric case in the whole project. Everything else here can be
rebuilt from upstream at any time; the backtest, the panel and the page are all
pure functions of data that is still out there. Friday's injury report is not
recoverable at any price. **If the snapshot job does not run on Friday, no
future version of this code can ever know what Friday said.** That is why
`nflproj snapshot` and its workflow exist now, before the model, and why live
publication started this season rather than after the interesting part was
finished.

The archive (`nflproj.ingest.archive`) has three properties that matter:

- **Content-addressed.** A snapshot is stored under the SHA-256 of its bytes, so
  an unchanged report costs one manifest line rather than another copy. That is
  what makes seven captures a week affordable. A no-change capture is not a
  wasted run either: it is positive evidence that the report did not move
  between two known times.
- **Append-only.** The manifest is JSONL, so a new observation is a one-line
  diff and a rewritten history would be conspicuous in review. The archive is
  the record of what we claimed to have seen; it should be as hard to edit
  quietly as the scorecard is.
- **Strictly-before reads.** `as_of(cutoff)` returns the latest snapshot taken
  *before* the cutoff, never at or after it, and that is not a parameter. A
  snapshot taken at kickoff may already reflect the inactive list. The cutoff
  for a week is its **first** kickoff, not each game's own — a week's
  projections are published once, so a Sunday player must not be priced with
  knowledge of Thursday's result.

Two absences are deliberately distinguishable. `as_of` returns `None` when the
archive does not reach back to that moment, which is true of every week before
the archive began; it never returns an empty frame there. An empty frame would
read as "nobody was hurt", and the difference between *unknown* and *nobody*
is exactly the difference a feature built on this will get wrong if we blur it.
For the same reason a manifest reference to a missing blob raises rather than
returning `None`, because a silent `None` would be indistinguishable from
"before the archive began".

The archive is therefore **committed to git**, unlike everything under `data/`,
which is gitignored precisely because it is reproducible.

### The cadence is itself a tested property

Running this unattended for two weeks produced fifteen successful captures and
*zero usable ones*. GitHub ran every scheduled job two to three and a half
hours late, consistently, so the Sunday captures aimed at two hours before
kickoff landed ninety minutes after it, and the Thursday capture aimed at
pre-TNF landed after the game began. Every week ended up covered only by the
previous Wednesday's report - a 23-hour-old view of a document that changes
daily.

Nothing failed, and that is the point. `as_of` is strict, so a late snapshot is
never selected; it is simply inert. The workflow was green, the manifest grew,
and the archive was accumulating evidence of nothing.

The schedule now carries a four-hour delay budget and runs at :07 rather than
:00. But a budget is an assumption, so `nflproj archive-health` checks it
against reality: for each week that has kicked off and that the archive was
running for, it reports the gap between the week's first kickoff and the latest
snapshot preceding it, and exits non-zero past a 12-hour bound. The weekly
workflow runs it as a separate job, so a cadence problem turns the run red
without withholding the scorecard - trading one silent failure for another
would be no improvement.

Weeks predating the archive are reported and **never** failed. That absence is
honest, and a check that cries wolf about it is a check nobody reads.

The same reasoning narrows the gate itself. A week whose captures were missed
cannot be repaired - the snapshots do not exist and cannot be recreated - so
judging all history would leave the weekly run red for the rest of the season
over something nobody can fix. That is the cries-wolf failure wearing a
different hat. The report lists every week and marks the stale ones; only the
**most recent judged week** decides the exit code, because "is the cadence
broken *now*" is the only version of the question you can act on.
`--all-weeks` judges everything, for an on-demand audit of a whole season.

One limitation this exposed, and does not fix: because a week's cutoff is its
*first* kickoff, and most weeks open on Thursday night, Friday's final injury
designations can never be used for that week - even though they precede the
Sunday games that 95% of players are in. That is conservative rather than
wrong, and the fix is per-game rather than per-week publication, which is a
real change to what "the board" means. Captures cannot be backfilled but the
cutoff rule can be changed later, so the cadence captures Friday, Saturday and
Sunday morning regardless.

## 4b. Availability is read per game, not per week

Everything else here uses one cutoff per week - its first kickoff - because a
board is published once, before any of it has been played. For injuries that
rule is not conservative, it is useless, and the archive proves it with a
number: the last snapshot before 2026 week 4's Thursday kickoff contains ten
week-4 rows and **not one game-status designation**. Teams file practice
participation on Wednesday and designations on *Friday*, and Friday is after
Thursday night.

Measured against each player's **own** kickoff instead:

| cutoff | designations available |
|---|---|
| Thursday 00:15Z (the week's first kickoff) | **0** |
| Sunday 13:30Z | 70 Out, 58 Questionable, 2 Doubtful |
| Sunday 20:05Z | 73 Out, 63 Questionable, 2 Doubtful |

Friday's report is legitimately pre-kickoff information for a Sunday game, and
using it leaks nothing about that player's outcome, which is still unknown. The
counts rising across Sunday's windows is the archive behaving correctly: a later
game gets a later snapshot.

So the two questions are separated. **History stays per week** and therefore
stays strictly conservative about *outcomes* - a Sunday projection still cannot
see Thursday's results. **Availability reads per game**, because it is
pre-kickoff information rather than an outcome. History is about what has
happened; availability is about what was known.

The source rule is enforced in code rather than remembered. For a player whose
game has not started, the live upstream file is correct - it is what a publisher
knows at publication time. At or after his kickoff, only the archive will do,
because the live file has by then been overwritten with post-game state. That
distinction is not cosmetic: *Out* does not merely correlate with scoring zero,
it partly **is** the outcome, so reading the current file to "project" a played
week would be near-perfect leakage. There is a test asserting the live feed is
not even consulted for a week already played.

A null designation means *no designation was published*, never *healthy*. The
bundle records the counts it found, so a board built before midweek is
distinguishable from one where nobody was hurt - and the page says which it is
rather than leaving a reader to infer it from an absence of zeros.

The factors themselves are a **stated prior, not a measurement**, because the
obvious measurement is contaminated: computing `P(play | Questionable)` from the
upstream file uses rows relabelled after the game, so players who were
Questionable and then sat now read as Out, the surviving Questionable rows skew
toward those who played, and any factor derived that way is an optimistic upper
bound. The archive can measure this honestly over the weeks it covers, and will.

Over 2015-2025 the adjustment is a **strict no-op**, because the archive does
not reach back that far and an absent designation is a factor of 1.0. Verified
on real data rather than asserted: across 2022-2024 the adjusted and unadjusted
predictors agree on every metric to four decimal places. The adjustment can only
act where there is point-in-time evidence that it should.

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
- **No injury or depth-chart features yet** (sections 4, 4a). Still the largest
  accuracy gap. The archive is now recording, but it only reaches back to the
  day it started, so injury features cannot be backtested over 2015-2025 and
  will not be until the archive has a season of its own behind it. Any future
  claim about them will have to be scored on that window alone, and said so.
- **Rookies enter the universe only after their first appearance.** Until then
  they are unprojectable, which understates coverage in September.
- **Snap counts are not used.** nflverse keys them on Pro-Football-Reference
  player ids, which need a crosswalk to the GSIS ids everything else uses.
- **Kickers and team defenses are out of scope.** Different data generating
  process; including them would add rows without adding insight.
- **Only full-PPR is scored by default.** Half-PPR and standard are supported by
  `ScoringRules` but not published.
