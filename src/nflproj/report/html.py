"""Render the public report page.

The page has two readers and one of them is in a hurry. A fantasy player wants
the board and a straight answer on how much to trust it; an engineer wants the
universe definition and the provenance hashes. So the board and the plain-language
readings lead, and the methodology sits underneath in a disclosure - present, not
in the way.

Progressive enhancement, strictly. Every number the page states - the board, the
readings, the whole week-by-week series - is in the HTML. JavaScript only adds
the chart, the player filter and column sorting, and the controls that depend on
it stay hidden until it runs. With scripting off the page loses a picture and
keeps every fact, which is also why it still reads correctly as a link preview.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from nflproj.report import interpret as itp
from nflproj.report.interpret import Confidence, Reading

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

#: Starter-tier sizes in a 12-team league - the number of each position actually
#: started in a week. The board shows exactly this many, because a list longer
#: than the decision is noise.
STARTER_TIERS: Final[dict[str, int]] = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

POSITION_ORDER: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE")

#: Calibration slope below which the board warns that its own spread is
#: overstated. Matches the band `interpret.read_calibration` calls notable.
OVER_DISPERSION_THRESHOLD: Final = 1.0 - itp.CALIBRATION_TIGHT

_CONFIDENCE_LABEL: Final[dict[Confidence, str]] = {
    Confidence.STRONG: "Reliable",
    Confidence.MODERATE: "Use with judgement",
    Confidence.WEAK: "Barely better than guessing",
}


@dataclass(frozen=True, slots=True)
class ReportData:
    """Everything the page renders."""

    season: int
    week: int
    generated_at: str
    projections: pd.DataFrame | None
    scorecard: dict[str, Any]
    predictor: str


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _row_for(scorecard: dict[str, Any], slice_name: str, predictor: str) -> dict[str, Any]:
    rows = scorecard["slices"][slice_name]
    for row in rows:
        if row["predictor"] == predictor:
            return dict(row)
    msg = f"predictor {predictor!r} not present in the {slice_name!r} slice"
    raise KeyError(msg)


def _rows_for(scorecard: dict[str, Any], slice_name: str, predictor: str) -> list[dict[str, Any]]:
    return [dict(r) for r in scorecard["slices"][slice_name] if r["predictor"] == predictor]


# ---------------------------------------------------------------- fragments


def _reading_card(reading: Reading) -> str:
    return f"""
      <article class="reading" data-confidence="{_esc(reading.confidence)}">
        <header>
          <h3>{_esc(reading.label)}</h3>
          <p class="value">{reading.value:.2f}</p>
        </header>
        <p class="plain">{_esc(reading.plain)}</p>
        <p class="action"><span class="action-key">Do this</span>{_esc(reading.action)}</p>
      </article>"""


def _meter(group: itp.GroupReading) -> str:
    """A single-series meter, not a chart: one value per position on a 0-1 scale.

    The confidence band is carried by a text label as well as colour, so the
    reading never depends on hue alone.
    """
    pct = max(0.0, min(1.0, group.spearman)) * 100
    return f"""
        <div class="meter-row">
          <span class="pos-badge">{_esc(group.key)}</span>
          <div class="meter" role="img"
               aria-label="{_esc(group.key)} ranking quality {group.spearman:.2f} out of 1">
            <span class="meter-fill" data-confidence="{_esc(group.confidence)}"
                  style="width:{pct:.1f}%"></span>
          </div>
          <span class="meter-value">{group.spearman:.2f}</span>
          <span class="chip" data-confidence="{_esc(group.confidence)}"
            >{_esc(_CONFIDENCE_LABEL[group.confidence])}</span>
        </div>"""


def _board(projections: pd.DataFrame | None, calibration: float) -> str:
    if projections is None or projections.empty:
        return """
      <p class="empty">No projections available for this week yet. The board fills
      once the schedule and the previous week's results are both published.</p>"""

    blocks: list[str] = []
    for position in POSITION_ORDER:
        subset = projections[projections["position"] == position]
        if subset.empty:
            continue
        limit = STARTER_TIERS.get(position, 12)
        subset = subset.nlargest(limit, "prediction")

        rows = "\n".join(
            f"""            <tr>
              <td class="rank">{rank}</td>
              <td class="player">{_esc(r.player_display_name)}</td>
              <td class="team">{_esc(r.team)}</td>
              <td class="opp">{_esc(r.opponent_team)}</td>
              <td class="pts">{r.prediction:.1f}</td>
            </tr>"""
            for rank, r in enumerate(subset.itertuples(index=False), start=1)
        )

        blocks.append(f"""
      <section class="board-block">
        <h3><span class="pos-badge">{_esc(position)}</span>Top {limit}</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th data-sort="text">Player</th>
                <th data-sort="text">Team</th>
                <th data-sort="text">Opp</th>
                <th data-sort="number" aria-sort="descending">Proj</th>
              </tr>
            </thead>
            <tbody>
{rows}
            </tbody>
          </table>
        </div>
      </section>""")

    spread_note = ""
    if calibration < OVER_DISPERSION_THRESHOLD:
        spread_note = (
            '<p class="board-note">These numbers are spread wider than reality '
            f"(scale factor {calibration:.2f}). The gap between the top of a list and "
            "the bottom is smaller than it looks.</p>"
        )

    # Hidden until the script enables them, so a reader without JavaScript is
    # never shown a control that does nothing.
    controls = """
      <div class="board-controls" id="board-controls" hidden>
        <input type="search" id="player-search" placeholder="Find a player"
               aria-label="Filter the board by player name">
        <span class="board-hint">Click a column heading to re-sort.</span>
      </div>"""

    return controls + spread_note + "\n".join(blocks)


@dataclass(frozen=True, slots=True)
class _TrendPoint:
    """One scored week, ready to plot."""

    label: str
    skill_pct: float


def _weekly_series(scorecard: dict[str, Any], predictor: str) -> list[dict[str, Any]]:
    """The published predictor's week-by-week record, in chronological order."""
    rows = _rows_for(scorecard, "by_week", predictor)
    rows.sort(key=lambda r: (int(r["season"]), int(r["week"])))
    return rows


def _trend_section(scorecard: dict[str, Any], predictor: str) -> str:
    """The time dimension: how the system did each week against the baseline.

    Plotted as skill rather than raw error. Two overlapping error lines would be
    a tangle at 181 points, and the question a reader actually has is not "how
    many points did it miss by" - that is on the cards above - but "does it
    beat the thing I would have done myself, and how often does it fail to".
    Zero is the baseline; below zero is a week this system lost.
    """
    if "by_week" not in scorecard.get("slices", {}):
        return ""
    rows = _weekly_series(scorecard, predictor)
    if not rows:
        return ""

    points = [
        _TrendPoint(
            label=f"{int(r['season'])} W{int(r['week'])}",
            skill_pct=round(float(r["mae_skill"]) * 100, 2),
        )
        for r in rows
        if r.get("mae_skill") is not None
    ]
    if not points:
        return ""

    island = _json_island([{"x": p.label, "y": p.skill_pct} for p in points])
    wins = sum(1 for p in points if p.skill_pct > 0)
    total = len(points)
    pct = round(wins / total * 100)
    worst = min(points, key=lambda p: p.skill_pct)

    return f"""
  <h2>Week by week</h2>
  <p class="section-lede">Each point is one scored week: how much better or worse
  than the baseline this system did that week. Zero is the baseline. Below zero
  is a week it lost &mdash; and there are {total - wins} of them, which is the
  point of publishing this at all.</p>

  <div class="trend-stats">
    <div class="stat">
      <span class="stat-value">{wins}<span class="stat-of">/{total}</span></span>
      <span class="stat-label">weeks beaten ({pct}%)</span>
    </div>
    <div class="stat">
      <span class="stat-value">{worst.skill_pct:.1f}%</span>
      <span class="stat-label">worst week ({_esc(worst.label)})</span>
    </div>
  </div>

  <figure class="chart-figure">
    <div class="chart-box"><canvas id="trend" aria-label="Weekly skill against the baseline"
      role="img"></canvas></div>
    <figcaption>Higher is better. Rendered from
    <code>reports/scorecard.json</code>; the table below carries the same
    numbers for anyone without JavaScript.</figcaption>
  </figure>

  <details class="weekly-table">
    <summary>Week-by-week numbers</summary>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Season</th><th>Week</th><th>MAE</th><th>vs baseline</th></tr></thead>
        <tbody>
{_weekly_rows(rows)}
        </tbody>
      </table>
    </div>
  </details>

  <script id="trend-data" type="application/json">{island}</script>"""


def _json_island(payload: object) -> str:
    """Serialise data for an inline ``application/json`` block.

    The browser ends the block at the first literal ``</script>`` in the text,
    regardless of JSON syntax, so that sequence is escaped. ``\u003c`` is still
    valid JSON and parses back to the same string.
    """
    return json.dumps(payload).replace("<", "\\u003c")


def _weekly_rows(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"          <tr><td>{int(r['season'])}</td><td>{int(r['week'])}</td>"
        f"<td class='pts'>{float(r['mae']):.2f}</td>"
        f"<td class='pts'>{float(r['mae_skill']) * 100:+.1f}%</td></tr>"
        for r in rows
        if r.get("mae_skill") is not None
    )


def _methodology(scorecard: dict[str, Any], predictor: str) -> str:
    provenance = scorecard.get("provenance", {})
    assets = provenance.get("raw_assets", {})
    asset_rows = "\n".join(
        f"            <tr><td>{_esc(k)}</td><td class='hash'>{_esc(v[:16])}…</td></tr>"
        for k, v in sorted(assets.items())
    )
    seasons = scorecard.get("seasons", [])
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "&mdash;"

    return f"""
      <details class="methodology">
        <summary>Methodology, and why these numbers are trustworthy</summary>
        <div class="methodology-body">
          <p>Every figure above comes from a walk-forward backtest over
          {span}: {scorecard.get("n_predictions", 0):,} projections, each made using
          only data that existed before that week kicked off.</p>

          <h4>The population is defined before kickoff</h4>
          <p>The obvious way to score a projection system is to check it against the
          players who actually played. That set is unknowable in advance, and
          excluding the players who were inactive deletes exactly the cases a
          projection needs to get right. So a player enters the week's list if
          their team plays and they appeared in one of their team's previous three
          games &mdash; and if they then sit out, they score zero and stay in the
          evaluation. The universe here is
          <code>{_esc(scorecard.get("universe", "?"))}</code>.</p>

          <h4>Measured against a named baseline</h4>
          <p>Accuracy with nothing to compare it to is a number without a meaning.
          Everything is scored against <code>{_esc(scorecard.get("baseline", "?"))}</code>
          &mdash; what a person means by &ldquo;he&rsquo;s averaging 14 a game&rdquo;.
          The predictor shown on this page is <code>{_esc(predictor)}</code>.</p>

          <h4>Leakage is prevented structurally</h4>
          <p>The backtest slices history itself and removes every post-kickoff column
          before a predictor sees the week. A predictor cannot read the answer because
          the answer is not in the data it is given. A test corrupts every result from
          a chosen week onward and asserts that earlier predictions are bit-identical
          &mdash; and a deliberately leaky control predictor proves that test can fail.</p>

          <h4>Week 1 is not projected</h4>
          <p>Knowing who is on which roster in week 1 needs offseason data with no
          clean point-in-time source. Rather than carry stale teams forward, the
          system declines to project it.</p>

          <h4>Source data</h4>
          <p>Fetched from nflverse and pinned by content hash, so any number here can
          be traced to the exact bytes it came from.</p>
          <div class="table-wrap">
            <table class="provenance">
              <thead><tr><th>Asset</th><th>SHA-256</th></tr></thead>
              <tbody>
{asset_rows}
              </tbody>
            </table>
          </div>
        </div>
      </details>"""


# ---------------------------------------------------------------- styles

_STYLE: Final = """
  :root {
    --bg: #eef1f3;
    --surface: #ffffff;
    --surface-2: #f6f8f9;
    --ink: #12191c;
    --ink-2: #44535a;
    --ink-3: #74838b;
    --line: #d8e0e3;
    --line-strong: #bcc8cd;
    --accent: #14655a;
    --accent-soft: #d6e8e4;
    --good: #1f7a4d;
    --warn: #9a6207;
    --poor: #99392f;
    --shadow: 0 1px 2px rgba(18, 25, 28, .06), 0 8px 24px -16px rgba(18, 25, 28, .35);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0e1416;
      --surface: #161f22;
      --surface-2: #1b262a;
      --ink: #eaf0f1;
      --ink-2: #a8b7bd;
      --ink-3: #7d8d94;
      --line: #263338;
      --line-strong: #35464c;
      --accent: #58c0ae;
      --accent-soft: #14312e;
      --good: #4cc185;
      --warn: #d9a441;
      --poor: #e0776a;
      --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
    }
  }
  :root[data-theme="dark"] {
    --bg: #0e1416;
    --surface: #161f22;
    --surface-2: #1b262a;
    --ink: #eaf0f1;
    --ink-2: #a8b7bd;
    --ink-3: #7d8d94;
    --line: #263338;
    --line-strong: #35464c;
    --accent: #58c0ae;
    --accent-soft: #14312e;
    --good: #4cc185;
    --warn: #d9a441;
    --poor: #e0776a;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Newsreader, Georgia, "Times New Roman", serif;
    font-size: 17px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  .wrap {
    max-width: 62rem;
    margin: 0 auto;
    padding-inline: 20px;
    padding-block: 0 72px;
  }

  h1, h2, h3, h4, .pos-badge, .eyebrow, th, .chip, .action-key {
    font-family: Oswald, "Helvetica Neue", Arial, sans-serif;
    font-weight: 600;
  }
  .num, .value, .pts, .meter-value, td.rank, .hash {
    font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }

  /* ---------- masthead ---------- */
  .masthead {
    border-bottom: 2px solid var(--ink);
    padding-block: 40px 20px;
    margin-bottom: 28px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: .14em;
    font-size: 12px;
    color: var(--accent);
    margin: 0 0 6px;
  }
  h1 {
    font-size: clamp(2.1rem, 6vw, 3.4rem);
    line-height: 1.02;
    margin: 0 0 12px;
    text-wrap: balance;
    letter-spacing: -.01em;
  }
  .standfirst {
    margin: 0;
    max-width: 54ch;
    color: var(--ink-2);
    font-size: 1.05rem;
  }
  .stamp {
    margin-top: 16px;
    font-size: 13px;
    color: var(--ink-3);
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
  }

  h2 {
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--ink-2);
    margin: 48px 0 4px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line-strong);
  }
  .section-lede {
    margin: 0 0 20px;
    color: var(--ink-2);
    max-width: 62ch;
  }

  /* ---------- verdict ---------- */
  .verdict {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding: 18px 22px;
    margin-bottom: 8px;
  }
  .verdict p { margin: 0; max-width: 62ch; }
  .verdict p + p { margin-top: 10px; }

  /* ---------- board ---------- */
  .board-note {
    color: var(--ink-2);
    font-size: .95rem;
    margin: 0 0 20px;
    max-width: 62ch;
  }
  .board-block { margin-bottom: 26px; }
  .board-block h3 {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: .95rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--ink-3);
    margin: 0 0 8px;
  }
  .pos-badge {
    display: inline-block;
    background: var(--ink);
    color: var(--bg);
    font-size: 12px;
    letter-spacing: .06em;
    padding: 3px 8px;
    min-width: 34px;
    text-align: center;
  }
  .table-wrap { overflow-x: auto; }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    box-shadow: var(--shadow);
  }
  /* Five short columns stretched across a wide viewport leaves the projection
     stranded from the name it belongs to. Cap the board so the row reads as
     one unit; the wrapper still scrolls on a narrow screen. */
  .board-block table { max-width: 44rem; }
  th {
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--ink-3);
    padding: 10px 12px;
    border-bottom: 1px solid var(--line-strong);
    white-space: nowrap;
  }
  td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--line);
    font-size: .95rem;
    white-space: nowrap;
  }
  tbody tr:last-child td { border-bottom: none; }
  td.rank { color: var(--ink-3); width: 2.5rem; font-size: .85rem; }
  td.player { font-weight: 500; white-space: normal; }
  td.team, td.opp { color: var(--ink-2); font-size: .85rem; }
  th:last-child, td.pts { text-align: right; }
  td.pts { font-weight: 600; }

  /* ---------- readings ---------- */
  .readings {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
    gap: 16px;
  }
  .reading {
    background: var(--surface);
    border-top: 3px solid var(--line-strong);
    padding: 18px 20px 20px;
    box-shadow: var(--shadow);
  }
  .reading[data-confidence="strong"] { border-top-color: var(--good); }
  .reading[data-confidence="moderate"] { border-top-color: var(--warn); }
  .reading[data-confidence="weak"] { border-top-color: var(--poor); }
  .reading header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }
  .reading h3 {
    margin: 0;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--ink-3);
  }
  .reading .value { margin: 0; font-size: 1.5rem; font-weight: 600; }
  .reading .plain { margin: 0 0 12px; font-size: .95rem; }
  .action {
    margin: 0;
    font-size: .92rem;
    color: var(--ink-2);
    border-top: 1px solid var(--line);
    padding-top: 12px;
  }
  .action-key {
    display: block;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--accent);
    margin-bottom: 3px;
  }

  /* ---------- meters ---------- */
  .meters { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
  .meter-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .meter {
    flex: 1 1 10rem;
    height: 10px;
    background: var(--surface-2);
    border: 1px solid var(--line);
    position: relative;
  }
  .meter-fill { display: block; height: 100%; background: var(--ink-3); }
  .meter-fill[data-confidence="strong"] { background: var(--good); }
  .meter-fill[data-confidence="moderate"] { background: var(--warn); }
  .meter-fill[data-confidence="weak"] { background: var(--poor); }
  .meter-value { font-size: .9rem; min-width: 3ch; }
  .chip {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .09em;
    padding: 3px 8px;
    border: 1px solid var(--line-strong);
    color: var(--ink-2);
    white-space: nowrap;
  }
  .chip[data-confidence="strong"] { color: var(--good); border-color: var(--good); }
  .chip[data-confidence="moderate"] { color: var(--warn); border-color: var(--warn); }
  .chip[data-confidence="weak"] { color: var(--poor); border-color: var(--poor); }

  .note { color: var(--ink-2); max-width: 62ch; margin: 0 0 14px; }

  /* ---------- methodology ---------- */
  .methodology {
    margin-top: 48px;
    background: var(--surface);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
  }
  .methodology summary {
    cursor: pointer;
    padding: 16px 20px;
    font-family: Oswald, Arial, sans-serif;
    font-size: .85rem;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: var(--ink-2);
  }
  .methodology summary:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .methodology-body { padding: 0 20px 20px; border-top: 1px solid var(--line); }
  .methodology-body h4 {
    margin: 22px 0 6px;
    font-size: .95rem;
    letter-spacing: .01em;
  }
  .methodology-body p { margin: 0; max-width: 66ch; color: var(--ink-2); font-size: .95rem; }
  code {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: .85em;
    background: var(--surface-2);
    padding: 1px 5px;
    border: 1px solid var(--line);
  }
  .provenance td { font-size: .8rem; }
  .hash { color: var(--ink-3); }
  .empty { color: var(--ink-3); font-style: italic; }

  footer {
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid var(--line);
    font-size: 13px;
    color: var(--ink-3);
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
  }
  a { color: var(--accent); }

  /* ---------- trend ---------- */
  .trend-stats { display: flex; flex-wrap: wrap; gap: 28px; margin-bottom: 20px; }
  .stat { display: flex; flex-direction: column; gap: 2px; }
  .stat-value {
    font-family: Oswald, Arial, sans-serif;
    font-size: 2rem;
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }
  .stat-of { color: var(--ink-3); font-size: 1.1rem; }
  .stat-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--ink-3);
  }
  .chart-figure { margin: 0 0 8px; }
  .chart-box {
    position: relative;
    height: 320px;
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 12px;
    box-shadow: var(--shadow);
  }
  .chart-figure figcaption {
    font-size: 12px;
    color: var(--ink-3);
    margin-top: 8px;
    max-width: 62ch;
  }
  .weekly-table { margin-bottom: 8px; }
  .weekly-table summary {
    cursor: pointer;
    font-family: Oswald, Arial, sans-serif;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: var(--ink-2);
    padding: 8px 0;
  }
  .weekly-table summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .weekly-table table { max-width: 34rem; }

  /* ---------- board controls (revealed by script) ---------- */
  .board-controls { display: flex; gap: 12px; align-items: center; margin-bottom: 18px; }
  .board-controls input {
    font: inherit;
    font-size: .95rem;
    padding: 8px 12px;
    border: 1px solid var(--line-strong);
    background: var(--surface);
    color: var(--ink);
    min-width: 0;
    flex: 1 1 14rem;
    max-width: 22rem;
  }
  .board-controls input:focus-visible { outline: 2px solid var(--accent); outline-offset: -1px; }
  .board-hint { font-size: 12px; color: var(--ink-3); }
  th[data-sort] { cursor: pointer; user-select: none; }
  th[data-sort]:hover { color: var(--ink); }
  th[aria-sort="ascending"]::after { content: " \\2191"; }
  th[aria-sort="descending"]::after { content: " \\2193"; }
  tr.filtered { display: none; }
  .no-match { color: var(--ink-3); font-style: italic; padding: 10px 12px; }

  @media (max-width: 30rem) {
    body { font-size: 16px; }
    .reading { padding: 16px; }
  }
"""


# ---------------------------------------------------------------- page


PAGE_TITLE: Final = "The Sunday Board"

#: Pinned, loaded from cdnjs, and the page's only runtime dependency. Everything
#: it draws is also present as a table in the HTML, so a reader who never runs
#: it loses the picture and nothing else.
#:
#: No Subresource Integrity hash yet. The environment this was built in cannot
#: reach cdnjs, so the hash could not be computed, and an unverified one is
#: worse than none - a wrong hash blocks the script silently. To add it, run
#: this on a machine that can reach cdnjs and paste the whole line, prefix
#: included, as CHART_LIB_SRI:
#:
#:     V=4.4.4
#:     curl -sL "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/$V/chart.umd.min.js" \
#:       | openssl dgst -sha512 -binary | openssl base64 -A \
#:       | sed 's/^/sha512-/'
#:
#: The `sha512-` prefix is part of the attribute value, not decoration: without
#: it the browser rejects the value as malformed and blocks the script, which is
#: the same silent failure an outright wrong hash causes. `_chart_lib_tag`
#: rejects a value that does not look like an SRI hash rather than emit it.
CHART_LIB_VERSION: Final = "4.4.4"
CHART_LIB_SRI: Final[str | None] = None

#: `<algorithm>-<base64 digest>`, per the Subresource Integrity spec.
_SRI_PATTERN: Final = re.compile(r"^sha(256|384|512)-[A-Za-z0-9+/]+={0,2}$")


def _chart_lib_tag() -> str:
    """Build the script tag, refusing to emit a malformed integrity value.

    Raises:
        ValueError: If ``CHART_LIB_SRI`` is set but is not a well-formed SRI
            hash. Failing the build is the point: a malformed attribute would
            silently stop the chart from loading on the published page.
    """
    src = f"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/{CHART_LIB_VERSION}/chart.umd.min.js"
    if CHART_LIB_SRI is not None and not _SRI_PATTERN.match(CHART_LIB_SRI):
        msg = (
            f"CHART_LIB_SRI={CHART_LIB_SRI!r} is not a valid Subresource Integrity "
            "hash. It must look like 'sha512-<base64>' - the bare output of "
            "`openssl base64` is missing the 'sha512-' prefix."
        )
        raise ValueError(msg)
    integrity = f' integrity="{CHART_LIB_SRI}"' if CHART_LIB_SRI else ""
    return (
        f'<script src="{src}"{integrity} crossorigin="anonymous" '
        'referrerpolicy="no-referrer" defer></script>'
    )


#: Series colours, validated for both themes with the dataviz palette checker:
#: lightness band, chroma floor, CVD separation, normal-vision floor and
#: contrast against each theme's chart surface all pass.
_CHART_INK: Final = {
    "light": {"positive": "#008573", "negative": "#b35309"},
    "dark": {"positive": "#1ea48f", "negative": "#bf7f28"},
}

_SCRIPT: Final = """
(function () {
  "use strict";

  // ---- board: search and sort -------------------------------------------
  // Progressive enhancement. The board is complete in the HTML; this only
  // filters and reorders what is already there.
  var controls = document.getElementById("board-controls");
  var search = document.getElementById("player-search");
  var tables = Array.prototype.slice.call(
    document.querySelectorAll(".board-block table")
  );

  if (controls && search && tables.length) {
    controls.hidden = false;

    search.addEventListener("input", function () {
      var needle = search.value.trim().toLowerCase();
      tables.forEach(function (table) {
        var shown = 0;
        var rows = table.tBodies[0].rows;
        for (var i = 0; i < rows.length; i++) {
          var name = rows[i].cells[1].textContent.toLowerCase();
          var hit = !needle || name.indexOf(needle) !== -1;
          rows[i].classList.toggle("filtered", !hit);
          if (hit) shown++;
        }
        var block = table.closest(".board-block");
        if (block) block.hidden = shown === 0;
      });
    });

    tables.forEach(function (table) {
      var headers = table.tHead.rows[0].cells;
      Array.prototype.forEach.call(headers, function (th, index) {
        var kind = th.getAttribute("data-sort");
        if (!kind) return;
        th.tabIndex = 0;
        var activate = function () {
          var ascending = th.getAttribute("aria-sort") !== "ascending";
          Array.prototype.forEach.call(headers, function (other) {
            other.removeAttribute("aria-sort");
          });
          th.setAttribute("aria-sort", ascending ? "ascending" : "descending");

          var body = table.tBodies[0];
          var rows = Array.prototype.slice.call(body.rows);
          rows.sort(function (a, b) {
            var x = a.cells[index].textContent.trim();
            var y = b.cells[index].textContent.trim();
            var result =
              kind === "number"
                ? parseFloat(x) - parseFloat(y)
                : x.localeCompare(y);
            return ascending ? result : -result;
          });
          rows.forEach(function (row, position) {
            row.cells[0].textContent = String(position + 1);
            body.appendChild(row);
          });
        };
        th.addEventListener("click", activate);
        th.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        });
      });
    });
  }

  // ---- weekly trend ------------------------------------------------------
  var holder = document.getElementById("trend-data");
  var canvas = document.getElementById("trend");
  if (!holder || !canvas || typeof Chart === "undefined") return;

  var points;
  try {
    points = JSON.parse(holder.textContent);
  } catch (err) {
    return;
  }
  if (!points || !points.length) return;

  var chart = null;

  function tokens() {
    var css = getComputedStyle(document.documentElement);
    var dark =
      document.documentElement.getAttribute("data-theme") === "dark" ||
      (document.documentElement.getAttribute("data-theme") !== "light" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    return {
      ink: css.getPropertyValue("--ink-2").trim() || "#44535a",
      muted: css.getPropertyValue("--ink-3").trim() || "#74838b",
      line: css.getPropertyValue("--line").trim() || "#d8e0e3",
      positive: dark ? "__POSITIVE_DARK__" : "__POSITIVE_LIGHT__",
      negative: dark ? "__NEGATIVE_DARK__" : "__NEGATIVE_LIGHT__"
    };
  }

  function draw() {
    var t = tokens();
    if (chart) chart.destroy();
    chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: points.map(function (p) { return p.x; }),
        datasets: [
          {
            data: points.map(function (p) { return p.y; }),
            borderColor: t.positive,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: t.positive,
            tension: 0.15,
            fill: {
              target: { value: 0 },
              above: t.positive + "22",
              below: t.negative + "33"
            }
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (item) {
                var v = item.parsed.y;
                return (
                  (v >= 0 ? "Beat" : "Lost to") +
                  " the baseline by " +
                  Math.abs(v).toFixed(1) +
                  "%"
                );
              }
            }
          }
        },
        scales: {
          x: {
            ticks: { color: t.muted, maxTicksLimit: 10, autoSkip: true },
            grid: { display: false },
            border: { color: t.line }
          },
          y: {
            title: {
              display: true,
              text: "Better than baseline (%)",
              color: t.muted
            },
            ticks: {
              color: t.muted,
              callback: function (value) { return value + "%"; }
            },
            // Zero is the baseline itself, so it is drawn as a real reference
            // line rather than one gridline among several.
            grid: {
              color: function (ctx) {
                return ctx.tick && ctx.tick.value === 0 ? t.ink : t.line;
              },
              lineWidth: function (ctx) {
                return ctx.tick && ctx.tick.value === 0 ? 1.5 : 1;
              }
            },
            border: { color: t.line }
          }
        }
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", draw);
  } else {
    draw();
  }
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  if (media.addEventListener) media.addEventListener("change", draw);
})();
"""

_FONT_LINKS: Final = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Oswald:wght@500;600;700&"
    "family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&"
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)


def _script() -> str:
    """The page's behaviour, with the validated series colours substituted in."""
    out = _SCRIPT
    for token, colour in (
        ("__POSITIVE_LIGHT__", _CHART_INK["light"]["positive"]),
        ("__NEGATIVE_LIGHT__", _CHART_INK["light"]["negative"]),
        ("__POSITIVE_DARK__", _CHART_INK["dark"]["positive"]),
        ("__NEGATIVE_DARK__", _CHART_INK["dark"]["negative"]),
    ):
        out = out.replace(token, colour)
    return out


def render_content(data: ReportData) -> str:
    """Render the page markup alone - no title, links or style block."""
    card = data.scorecard
    overall = _row_for(card, "overall", data.predictor)
    baseline_row = _row_for(card, "overall", card["baseline"])

    top_n = itp.read_top_n(float(overall["top_n_hit_rate"]))
    spearman = itp.read_spearman(float(overall["spearman"]))
    calibration = itp.read_calibration(float(overall["calibration_slope"]))
    bias = itp.read_bias(float(overall["bias"]))
    mae = itp.read_mae(float(overall["mae"]), baseline=float(baseline_row["mae"]))

    positions = itp.read_positions(_rows_for(card, "by_position", data.predictor))
    phases = {
        str(r.get("season_phase")): r for r in _rows_for(card, "by_season_phase", data.predictor)
    }
    early = float(phases.get("weeks_2_4", {}).get("spearman", overall["spearman"]))
    late = float(phases.get("weeks_5_plus", {}).get("spearman", overall["spearman"]))

    played_rate = float(card.get("provenance", {}).get("played_rate", 0.746))

    readings = "\n".join(_reading_card(r) for r in (top_n, spearman, calibration, bias, mae))
    meters = "\n".join(_meter(g) for g in positions)

    generated = data.generated_at.replace("T", " ")[:16]

    return f"""<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Week {data.week} &middot; {data.season} season</p>
    <h1>The Sunday Board</h1>
    <p class="standfirst">Weekly fantasy projections that publish their own report
    card &mdash; including the weeks they get it wrong.</p>
    <p class="stamp">
      <span>Generated {_esc(generated)} UTC</span>
      <span>Predictor: {_esc(data.predictor)}</span>
      <span>Scoring: full PPR</span>
    </p>
  </header>

  <div class="verdict">
    <p><strong>The short version.</strong> {_esc(top_n.plain)} {_esc(top_n.action)}</p>
    <p>{_esc(itp.position_guidance(positions))}</p>
  </div>

  <h2>This week&rsquo;s board</h2>
  <p class="section-lede">Projected points for the players you would actually
  consider starting. Ranked within each position, because that is the decision
  you are making.</p>
{_board(data.projections, float(overall["calibration_slope"]))}

  <h2>How much to trust it</h2>
  <p class="section-lede">Each number below is measured over
  {card.get("n_predictions", 0):,} past projections, every one of them made before
  the week it covers had kicked off.</p>
  <div class="readings">
{readings}
  </div>

{_trend_section(card, data.predictor)}

  <h2>Where it is strong, and where it is not</h2>
  <p class="section-lede">Ranking quality by position. Higher means the order
  within that position is more reliable &mdash; comparing points missed across
  positions would be meaningless, since quarterbacks simply score more.</p>
  <div class="meters">
{meters}
  </div>
  <p class="note">{_esc(itp.phase_guidance(early, late))}</p>
  <p class="note">{_esc(itp.availability_note(played_rate))}</p>

{_methodology(card, data.predictor)}

  <footer>
    <span>Built with nflverse data.</span>
    <span>Not affiliated with the NFL.</span>
    <span>Projections are estimates, not advice.</span>
  </footer>
</div>

<script>{_script()}</script>"""


def render_body(data: ReportData) -> str:
    """Title, font links, style and content - the form the Artifact tool publishes.

    The publish pipeline supplies its own ``<!doctype>``/``<head>``/``<body>``
    skeleton, so this deliberately omits them.
    """
    return (
        f"<title>{PAGE_TITLE}</title>\n"
        f"{_FONT_LINKS}\n"
        f"{_chart_lib_tag()}\n"
        f"<style>{_STYLE}</style>\n\n"
        f"{render_content(data)}"
    )


def render_document(data: ReportData) -> str:
    """A complete standalone HTML document, for static hosting on GitHub Pages."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, '
        'viewport-fit=cover">\n'
        '<meta name="description" content="Weekly NFL fantasy projections with a '
        'published accuracy track record.">\n'
        f"<title>{PAGE_TITLE}</title>\n"
        f"{_FONT_LINKS}\n"
        f"{_chart_lib_tag()}\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{render_content(data)}\n"
        "</body>\n"
        "</html>\n"
    )


def build_report_data(
    *,
    scorecard_path: Path,
    projections: pd.DataFrame | None,
    season: int,
    week: int,
    predictor: str,
) -> ReportData:
    """Assemble the inputs the page needs."""
    scorecard = json.loads(scorecard_path.read_text())
    return ReportData(
        season=season,
        week=week,
        generated_at=datetime.now(UTC).isoformat(),
        projections=projections,
        scorecard=scorecard,
        predictor=predictor,
    )
