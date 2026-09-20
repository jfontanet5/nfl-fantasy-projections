# nfl-fantasy-projections

Weekly NFL fantasy point projections that publish their own scorecard.

The differentiator is **evaluation rigor, not model novelty**. Anyone can fit a
gradient booster to weekly stats. The hard part — and the part this repo is
actually about — is building an evaluation you can hand to a skeptic: a
prediction universe defined before kickoff, a named baseline you are measured
against, leakage prevented structurally rather than by discipline, and a track
record that accumulates in git where it cannot be quietly revised.

**There is no model yet, on purpose.** The harness, the baselines and the
leakage tests came first, so that when a model does land there is already
something honest to measure it against.

---

## Current scorecard

Walk-forward over **2015–2025**, 181 weeks, 492,485 scored projections. Universe
is `active_recent`; the named baseline is `season_to_date_mean`.

| predictor | MAE | RMSE | Spearman | top-N hit | calib. slope | MAE skill |
|---|---:|---:|---:|---:|---:|---:|
| `ewma_hl3` | **4.417** | **6.331** | **0.596** | **0.522** | 0.846 | **+3.8%** |
| `rolling_mean_4` | 4.541 | 6.621 | 0.572 | 0.503 | 0.744 | +1.1% |
| `season_to_date_mean` *(baseline)* | 4.591 | 6.609 | 0.567 | 0.507 | 0.760 | — |
| `season_to_date_mean_shrunk2` | 4.867 | 6.486 | 0.557 | 0.510 | **1.002** | −6.0% |
| `last_game` | 4.935 | 7.659 | 0.562 | 0.471 | 0.535 | −7.5% |
| `season_to_date_mean_played_only` | 5.140 | 7.052 | 0.493 | 0.500 | 0.706 | −11.9% |
| `position_mean` *(floor)* | 6.129 | 7.742 | n/a | 0.250 | 0.919 | −33.5% |

Three things worth reading off that table:

**Recency beats accumulation.** A half-life-3 exponential mean beats the
season-to-date average on every metric. Fantasy scoring is non-stationary
within a season — roles change — and an average that weights week 2 as heavily
as week 11 is fighting that.

**Averaging only games a player *played* is actively harmful.** It scores 11.9%
worse than the same estimator that counts missed weeks as zero, with a bias of
−1.47 points. It answers "how good is he when he plays", which is not the
question. This is the `include_dnp` flag, and the backtest settled it rather
than us asserting it.

**MAE and calibration disagree, and that is informative.** Shrinking toward a
positional prior makes MAE *worse* (−6.0%) while making calibration nearly
perfect (slope 1.002 vs 0.760) and RMSE better (+1.9%). Un-shrunk averages are
over-dispersed: a one-point rise in the projection buys only 0.76 points of
realised scoring. Which you prefer depends on whether you are picking a lineup
or pricing a contract — so both are published.

`position_mean` has no Spearman because it assigns every player at a position
the same number, so within-position rank correlation is undefined. That is the
floor behaving correctly.

Full output, including per-position and early-season slices:
[`reports/scorecard.md`](reports/scorecard.md) ·
[`reports/scorecard.json`](reports/scorecard.json)

---

## Why you should believe the numbers

The full argument is in **[`docs/leakage.md`](docs/leakage.md)**. The short
version:

**A predictor is never handed the future.** The harness slices history and
strips every post-kickoff column before calling a predictor. Leakage is not
prevented by remembering to be careful; it is prevented because the answer is
not in the frame.

**The perturbation test.** Run the backtest, then replace every outcome at or
after week *W* with garbage and run it again. Predictions before *W* must be
bit-identical. And because a test that cannot fail proves nothing, the suite
includes a deliberately leaky control predictor and asserts that it **does**
diverge.

**The universe is defined before kickoff.** Scoring only players who actually
played conditions on the outcome and deletes every inactive-scored-zero case —
26% of the universe. The default universe includes them at 0.0. The
outcome-conditioned view is published too, labelled as such.

**Week 1 is refused.** Knowing week-1 rosters needs offseason data we have no
clean point-in-time source for. Rather than carry stale teams forward, we do
not project it. Costs ~6% of rows; removes an entire class of argument.

**The target is computed, not inherited.** Fantasy points are recomputed from
box-score components and cross-checked against nflverse — exact agreement across
17,417 player-weeks. That check already caught a real bug (missing return TDs).

**Upstream drift is a tested failure.** Mid-build, nflverse migrated weekly
stats to a new release tag. The old tag did not 404 — it kept serving a frozen
2024 snapshot. A pipeline pointed at it would have looked healthy with no
current-season data. There is now a test for exactly that, and every scorecard
embeds SHA-256 hashes of the files its numbers came from.

---

## Quickstart

```bash
uv sync --all-extras

uv run nflproj status                              # how far has the season got
uv run nflproj ingest --seasons 2015-2026          # mirror upstream, with provenance
uv run nflproj backtest --seasons 2015-2025        # walk-forward + scorecard
uv run nflproj project 2026 --week 3               # project an upcoming week
```

Or in a container:

```bash
docker build -t nflproj .
docker run --rm -v "$PWD/data:/data" -v "$PWD/reports:/reports" nflproj \
  backtest --seasons 2015-2025
```

---

## Architecture

```
ingest/      nflverse release mirror + SHA-256 provenance manifest
features/    team calendar (schedule-derived) -> player-week panel + universe
predictors/  Predictor protocol; baselines implementing it
evaluation/  walk-forward harness -> metrics -> versioned scorecard
```

Data flows one way. Each stage is independently testable and none of them can
see the future, because the harness that drives them is what decides what they
see.

**Design decisions worth arguing with:**

- *DuckDB/parquet over a warehouse.* Deliberately outside Snowflake. The whole
  11-season panel is 74k rows; a warehouse would be ceremony.
- *Long-format predictions.* Adding a predictor never changes the scorecard
  schema.
- *The backtest loop is sequential, not parallel.* Weeks are cheap, and a
  sequential loop makes the temporal ordering obvious to anyone auditing it.
  That is worth more here than wall-clock time.
- *Predictors are a `Protocol`, not a base class.* An XGBoost model will satisfy
  the same three methods the baselines do, so the harness needs no changes to
  score it.

---

## Quality gates

| Gate | Status |
|---|---|
| `pytest` | 139 tests (123 hermetic unit, 16 live-upstream) |
| `mypy --strict` | clean on `src`, no `type: ignore` |
| `ruff` | clean, ~20 rule families |
| Coverage | 93% from unit tests alone, CI floor 85% |
| Container | multi-stage, non-root; CI builds it and asserts it runs unprivileged |

> The container cannot be built in the environment this was developed in
> (Docker Hub is blocked), so CI is its only verification. The first CI run
> caught a real bug this way: `uv sync` honours `UV_PROJECT_ENVIRONMENT`, not
> `VIRTUAL_ENV`, so the image built cleanly with an empty virtualenv and died
> on first run. The builder now asserts the entrypoint runs before shipping the
> layer, turning that class of failure into a build error.

CI separates **unit** tests (hermetic, no network, gate every PR) from
**contract** tests (hit live nflverse), so an upstream outage reads as an
upstream failure rather than a code regression.

```bash
make check    # lint + format + types + tests
```

---

## Roadmap

In order, and the order is the point — each step is only meaningful because the
scorecard already exists to measure it.

1. **Point-in-time injury and depth-chart archive.** The largest accuracy gap.
   nflverse serves current state, not historical as-of state, so this requires
   snapshotting weekly going forward. Started now rather than after the model,
   which is why live publication begins this season.
2. **XGBoost projector.** Usage-based features (targets, carries, snap share)
   with a Tweedie objective for the zero-inflated target. Measured against the
   same baseline, on the same universe, by the same harness.
3. **PFR↔GSIS player crosswalk**, unlocking snap counts.
4. **Model serving.** FastAPI + container, deployed outside Snowflake.
5. **Public consensus baseline.** Compare against published projections, not
   only naive ones.
6. **Week 1**, once a preseason roster source exists.

## Licence

MIT.
