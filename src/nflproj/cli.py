"""Command line entry points.

Thin: every command is a few lines of wiring over the library. Anything worth
testing lives in a module, not here.
"""

from __future__ import annotations

from typing import Annotated

import typer

from nflproj.config import get_settings
from nflproj.evaluation.backtest import BacktestConfig, project_week, run_backtest
from nflproj.evaluation.scorecard import build_scorecard
from nflproj.features.calendar import (
    build_team_calendar,
    current_season,
    latest_completed_week,
    next_projectable_week,
)
from nflproj.features.panel import FIRST_PROJECTABLE_WEEK, UniversePolicy, build_panel
from nflproj.ingest import nflverse as nv
from nflproj.ingest.archive import INJURIES, Archive, snapshot_season
from nflproj.ingest.manifest import Manifest
from nflproj.logging import configure_logging, get_logger
from nflproj.predictors.baselines import (
    HEADLINE_BASELINE_NAME,
    PUBLISHED_PREDICTOR_NAME,
    default_baselines,
)
from nflproj.report.html import build_report_data, render_document

app = typer.Typer(
    add_completion=False,
    help="Weekly NFL fantasy point projections, evaluated against named baselines.",
)
log = get_logger(__name__)

SeasonRange = Annotated[str, typer.Option(help="Season range, e.g. '2015-2024' or '2024'.")]


def _parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def ingest(
    seasons: SeasonRange = "2015-2026",
    force: Annotated[bool, typer.Option(help="Re-download even if cached.")] = False,
) -> None:
    """Mirror the upstream nflverse assets we depend on into the raw zone."""
    settings = get_settings()
    settings.ensure_dirs()
    manifest = Manifest(settings.raw_dir)
    years = _parse_seasons(seasons)

    nv.fetch_asset(nv.SCHEDULES, None, settings=settings, manifest=manifest, force=force)
    nv.fetch_seasons(nv.PLAYER_STATS, years, settings=settings, manifest=manifest, force=force)
    log.info("ingest.done", assets=len(manifest), raw_dir=str(settings.raw_dir))


@app.command()
def backtest(
    seasons: SeasonRange = "2015-2025",
    universe: Annotated[
        UniversePolicy, typer.Option(help="Population the metrics are computed over.")
    ] = UniversePolicy.ACTIVE_RECENT,
    stem: Annotated[str, typer.Option(help="Output filename stem under reports/.")] = "scorecard",
) -> None:
    """Run the walk-forward backtest and write a scorecard."""
    settings = get_settings()
    settings.ensure_dirs()
    years = _parse_seasons(seasons)

    panel = build_panel(years, settings=settings)
    predictors = default_baselines()
    predictions = run_backtest(panel, predictors, config=BacktestConfig(seasons=tuple(years)))

    manifest = Manifest(settings.raw_dir)
    card = build_scorecard(
        predictions,
        universe=universe,
        baseline=HEADLINE_BASELINE_NAME,
        provenance={
            "raw_assets": {k: e.sha256 for k, e in sorted(manifest.entries().items())},
            "panel_rows": len(panel),
            "seasons_requested": years,
            "played_rate": round(float(panel.loc[panel["projectable"], "played"].mean()), 4),
        },
    )
    path = card.write(settings.reports_dir, stem=stem)
    typer.echo(card.slices["overall"].to_markdown(index=False, floatfmt=".4f"))
    typer.echo(f"\nwrote {path}")


@app.command()
def project(
    season: int,
    week: Annotated[int, typer.Option(help="Target week. Must be >= 2.")],
    predictor: Annotated[str, typer.Option(help="Predictor name.")] = HEADLINE_BASELINE_NAME,
    top: Annotated[int, typer.Option(help="Rows to print.")] = 25,
) -> None:
    """Project a single upcoming week from data available before it."""
    # Validate arguments before touching the network: a typo should cost a
    # second, not two seasons of downloads.
    chosen = next((p for p in default_baselines() if p.name == predictor), None)
    if chosen is None:
        names = sorted(p.name for p in default_baselines())
        msg = f"unknown predictor {predictor!r}; available: {names}"
        raise typer.BadParameter(msg)
    if week < FIRST_PROJECTABLE_WEEK:
        msg = (
            f"week must be >= {FIRST_PROJECTABLE_WEEK}; "
            "week 1 is out of scope (see docs/leakage.md)"
        )
        raise typer.BadParameter(msg)

    settings = get_settings()
    panel = build_panel(range(season - 1, season + 1), settings=settings)
    out = project_week(panel, chosen, season=season, week=week)
    typer.echo(out.head(top).to_markdown(index=False, floatfmt=".2f"))


@app.command()
def report(
    season: Annotated[
        int | None, typer.Argument(help="Season. Defaults to the one in progress.")
    ] = None,
    week: Annotated[
        int | None,
        typer.Option(help="Week the board covers. Defaults to the next unplayed week."),
    ] = None,
    predictor: Annotated[
        str, typer.Option(help="Predictor whose projections the page publishes.")
    ] = PUBLISHED_PREDICTOR_NAME,
    scorecard: Annotated[
        str, typer.Option(help="Scorecard stem under reports/ to read metrics from.")
    ] = "scorecard",
    out: Annotated[str, typer.Option(help="Output filename under reports/.")] = "index.html",
) -> None:
    """Render the public report page: this week's board plus the track record."""
    chosen = next((p for p in default_baselines() if p.name == predictor), None)
    if chosen is None:
        names = sorted(p.name for p in default_baselines())
        msg = f"unknown predictor {predictor!r}; available: {names}"
        raise typer.BadParameter(msg)
    settings = get_settings()
    settings.ensure_dirs()

    if season is None or week is None:
        calendar = build_team_calendar(range(settings.first_season, 2030), settings=settings)
        season = season if season is not None else current_season(calendar)
        if season is None:
            msg = "no season has started yet; pass one explicitly"
            raise typer.BadParameter(msg)
        week = week if week is not None else next_projectable_week(calendar, season)
        if week is None:
            msg = f"no upcoming week in {season}; pass --week explicitly"
            raise typer.BadParameter(msg)
        log.info("report.derived_target", season=season, week=week)

    if week < FIRST_PROJECTABLE_WEEK:
        msg = f"week must be >= {FIRST_PROJECTABLE_WEEK}; week 1 is out of scope"
        raise typer.BadParameter(msg)

    scorecard_path = settings.reports_dir / f"{scorecard}.json"
    if not scorecard_path.exists():
        msg = f"no scorecard at {scorecard_path}; run `nflproj backtest` first"
        raise typer.BadParameter(msg)

    # A missing week is not fatal: the track record is still worth publishing
    # when the upcoming slate cannot be projected yet.
    projections = None
    try:
        panel = build_panel(range(season - 1, season + 1), settings=settings)
        projections = project_week(panel, chosen, season=season, week=week)
    except (ValueError, KeyError) as exc:
        log.warning("report.no_projections", season=season, week=week, error=str(exc))

    data = build_report_data(
        scorecard_path=scorecard_path,
        projections=projections,
        season=season,
        week=week,
        predictor=predictor,
    )
    destination = settings.reports_dir / out
    destination.write_text(render_document(data))
    log.info(
        "report.written",
        path=str(destination),
        bytes=destination.stat().st_size,
        projected_players=0 if projections is None else len(projections),
    )
    typer.echo(f"wrote {destination}")


@app.command()
def snapshot(
    season: Annotated[
        int | None, typer.Argument(help="Season to file under. Defaults to the one in progress.")
    ] = None,
) -> None:
    """Record the injury report as it stands right now.

    The injury file upstream is keyed on (season, week) and rewritten in place
    as the week progresses, so Friday's report is destroyed by Sunday's. The
    only way to ever have it is to have written it down on Friday. Run often;
    identical bytes cost one manifest line, not another copy.
    """
    settings = get_settings()
    settings.ensure_dirs()

    if season is None:
        calendar = build_team_calendar(range(settings.first_season, 2030), settings=settings)
        season = snapshot_season(calendar)
        if season is None:
            # Not an error. Injury reports only move around games; in the
            # offseason there is genuinely nothing to record.
            typer.echo("no kickoff within the snapshot window; nothing to record")
            return

    observation = Archive(settings.archive_dir, INJURIES).capture(season, settings=settings)
    state = "new" if observation.novel else "unchanged since the last capture"
    typer.echo(
        f"{observation.asset} {observation.season}: {observation.rows} rows, "
        f"sha256 {observation.sha256[:12]} ({state})"
    )


@app.command()
def status() -> None:
    """Report how far the current season has progressed."""
    settings = get_settings()
    calendar = build_team_calendar(range(settings.first_season, 2027), settings=settings)
    for season in sorted(calendar["season"].unique())[-2:]:
        done = latest_completed_week(calendar, int(season))
        typer.echo(f"season {season}: last fully completed week = {done}")


if __name__ == "__main__":  # pragma: no cover
    app()
