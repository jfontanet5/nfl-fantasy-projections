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
from nflproj.features.calendar import build_team_calendar, latest_completed_week
from nflproj.features.panel import FIRST_PROJECTABLE_WEEK, UniversePolicy, build_panel
from nflproj.ingest import nflverse as nv
from nflproj.ingest.manifest import Manifest
from nflproj.logging import configure_logging, get_logger
from nflproj.predictors.baselines import HEADLINE_BASELINE_NAME, default_baselines

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
def status() -> None:
    """Report how far the current season has progressed."""
    settings = get_settings()
    calendar = build_team_calendar(range(settings.first_season, 2027), settings=settings)
    for season in sorted(calendar["season"].unique())[-2:]:
        done = latest_completed_week(calendar, int(season))
        typer.echo(f"season {season}: last fully completed week = {done}")


if __name__ == "__main__":  # pragma: no cover
    app()
