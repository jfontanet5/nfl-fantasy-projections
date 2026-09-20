"""The projection API.

What this serves, and why it is shaped this way
-----------------------------------------------

A fantasy projection is a weekly batch product. Nobody needs sub-100ms
inference for a number that changes once every seven days, and building this as
a live-scoring service would be architecture theatre. So the primary endpoints
read a **bundle** - a versioned, content-hashed artifact produced before the
week kicked off (see :mod:`nflproj.serving.bundle`). The image is the model
version: the bundle is baked in, so a rollback is redeploying last week's tag
and there is no runtime dependency on a model store.

``POST /predict`` exists alongside it and does run the predictor live, on
history the caller supplies. It is for what-ifs, and it takes history rather
than a player id on purpose - a request naming only a player would either be a
lookup pretending to be a prediction, or the server quietly reading data the
caller never saw.

Liveness and readiness are genuinely different here
---------------------------------------------------

``/healthz`` answers "is this process alive" and is true as soon as the
interpreter is running. ``/readyz`` answers "should this pod receive traffic",
which is false until a bundle has loaded and verified. Wiring both probes to
the same endpoint - the usual shortcut - means Kubernetes either restarts pods
that are merely still loading, or routes traffic to pods holding a corrupt
bundle. Those are different failures and they need different answers.

A bundle that fails its integrity check leaves the service permanently
not-ready rather than falling back to something. Serving numbers of unknown
origin would be worse than serving none, in a repo whose entire claim is that
you can check where the numbers came from.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from nflproj.logging import configure_logging, get_logger
from nflproj.predictors.baselines import PUBLISHED_PREDICTOR_NAME, default_baselines
from nflproj.serving.bundle import Bundle, BundleError, load_bundle
from nflproj.serving.schemas import (
    Board,
    ErrorResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    Projection,
    Provenance,
    ReadyResponse,
    VersionResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

log = get_logger(__name__)

SERVICE_NAME: Final = "nflproj-api"
DEFAULT_BUNDLE_DIR: Final = Path("bundle")
MAX_PAGE_SIZE: Final = 500

REQUESTS = Counter(
    "nflproj_http_requests_total",
    "HTTP requests handled.",
    labelnames=("method", "route", "status"),
)
LATENCY = Histogram(
    "nflproj_http_request_seconds",
    "Request latency.",
    labelnames=("method", "route"),
    # Tuned for a read-a-dataframe service, not a database. If p99 lands in the
    # top bucket the shape of the problem has changed and the buckets are wrong.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PROJECTIONS_SERVED = Counter(
    "nflproj_projections_served_total",
    "Individual player projections returned.",
    labelnames=("endpoint",),
)
BUNDLE_LOADED = Gauge(
    "nflproj_bundle_loaded",
    "1 when a verified bundle is in memory, 0 otherwise.",
)
BUNDLE_WEEK = Gauge(
    "nflproj_bundle_week",
    "Week the loaded bundle covers.",
    labelnames=("season",),
)


def _app_version() -> str:
    try:
        return package_version("nflproj")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "unknown"


def bundle_dir() -> Path:
    return Path(os.environ.get("NFLPROJ_BUNDLE_DIR", str(DEFAULT_BUNDLE_DIR)))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the bundle once, at startup.

    Loading per request would re-read and re-verify a parquet on every call for
    data that cannot change while the process lives. A failure here is recorded
    and left to the readiness probe rather than raised: a pod that crash-loops
    tells an operator less than one that stays up and reports itself not-ready
    with a reason.
    """
    configure_logging()
    root = bundle_dir()
    app.state.bundle = None
    app.state.bundle_error = None
    try:
        app.state.bundle = load_bundle(root)
        BUNDLE_LOADED.set(1)
        meta = app.state.bundle.metadata
        BUNDLE_WEEK.labels(season=str(meta.season)).set(meta.week)
        log.info("api.ready", bundle=meta.version, season=meta.season, week=meta.week)
    except BundleError as exc:
        app.state.bundle_error = str(exc)
        BUNDLE_LOADED.set(0)
        log.error("api.bundle_unavailable", path=str(root), error=str(exc))
    yield
    log.info("api.shutdown")


def get_bundle(request: Request) -> Bundle:
    """The loaded bundle, or a 503 that says why there isn't one."""
    loaded: Bundle | None = request.app.state.bundle
    if loaded is None:
        detail = request.app.state.bundle_error or "no bundle is loaded"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return loaded


BundleDep = Annotated[Bundle, Depends(get_bundle)]


def _provenance(bundle: Bundle) -> Provenance:
    meta = bundle.metadata
    return Provenance(
        bundle_version=meta.version,
        predictor=meta.predictor,
        baseline=meta.baseline,
        generated_at=meta.created_at,
        valid_from=meta.valid_from,
        superseded=meta.superseded(),
        raw_asset_hashes=meta.raw_assets,
        metrics=meta.metrics,
    )


def _to_projections(frame: pd.DataFrame) -> Iterator[Projection]:
    for row in frame.to_dict(orient="records"):
        opponent = row.get("opponent_team")
        yield Projection(
            player_id=str(row["player_id"]),
            player_name=str(row["player_display_name"]),
            position=str(row["position"]),  # type: ignore[arg-type]
            team=str(row["team"]),
            opponent=None if opponent is None or pd.isna(opponent) else str(opponent),
            season=int(row["season"]),
            week=int(row["week"]),
            projected_points=round(float(row["prediction"]), 2),
            rank=int(row["rank"]),
            position_rank=int(row["position_rank"]),
        )


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    """Board-wide ranks, assigned before any filtering.

    Computed on the whole board on purpose: a receiver's position rank is a
    fact about the week, not about the query. Ranking after a filter would make
    ``?position=WR`` report every receiver as WR1 through WR60 of a different
    universe than the unfiltered call, and the two answers would disagree.
    """
    out = frame.sort_values("prediction", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["position_rank"] = out.groupby("position", observed=True)["prediction"].rank(
        method="first", ascending=False
    )
    return out


def create_app() -> FastAPI:
    app = FastAPI(
        title="nflproj",
        summary="Weekly NFL fantasy point projections, with the provenance to check them.",
        version=_app_version(),
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
    )
    _register_observability(app)
    _register_ops_routes(app)
    _register_projection_routes(app)
    return app


def _register_observability(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Request id, structured access log, and the metrics both feed."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started

        # The *route template*, never the raw path: labelling by path would mint
        # a new time series per player id and melt the metrics store.
        route = request.scope.get("route")
        label = getattr(route, "path", "unmatched")

        REQUESTS.labels(request.method, label, str(response.status_code)).inc()
        LATENCY.labels(request.method, label).observe(elapsed)
        response.headers["x-request-id"] = request_id
        log.info(
            "api.request",
            request_id=request_id,
            method=request.method,
            route=label,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response


def _register_ops_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz() -> HealthResponse:
        """Liveness. True whenever the process can answer - restart me if not."""
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    def readyz(request: Request) -> Response:
        """Readiness. False until a bundle has loaded *and* verified."""
        loaded: Bundle | None = request.app.state.bundle
        if loaded is None:
            body = ReadyResponse(
                status="not_ready",
                detail=request.app.state.bundle_error or "no bundle is loaded",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=body.model_dump(),
            )
        return JSONResponse(
            content=ReadyResponse(
                status="ready", bundle_version=loaded.metadata.version
            ).model_dump()
        )

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics() -> Response:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/version", response_model=VersionResponse, tags=["ops"])
    def version(bundle: BundleDep) -> VersionResponse:
        meta = bundle.metadata
        return VersionResponse(
            service=SERVICE_NAME,
            app_version=_app_version(),
            bundle_version=meta.version,
            predictor=meta.predictor,
            season=meta.season,
            week=meta.week,
            git_sha=os.environ.get("NFLPROJ_GIT_SHA"),
        )


def _register_projection_routes(app: FastAPI) -> None:
    @app.get(
        "/projections/{season}/{week}",
        response_model=Board,
        tags=["projections"],
        responses={404: {"model": ErrorResponse}},
    )
    def board(
        season: int,
        week: int,
        bundle: BundleDep,
        *,
        position: Annotated[str | None, Query(description="Filter to one position.")] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Board:
        """The published board for a week."""
        meta = bundle.metadata
        if (season, week) != (meta.season, meta.week):
            # Say what *is* held. "Not found" alone sends an operator to the
            # logs to answer a question the response could have answered.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"this instance serves {meta.season} week {meta.week} "
                    f"(bundle {meta.version}); {season} week {week} is not in it"
                ),
            )

        frame = _ranked(bundle.projections)
        if position is not None:
            frame = frame[frame["position"].str.upper() == position.upper()]
        total = len(frame)
        page = frame.iloc[offset : offset + limit]
        PROJECTIONS_SERVED.labels("board").inc(len(page))

        return Board(
            season=meta.season,
            week=meta.week,
            count=len(page),
            total=total,
            projections=list(_to_projections(page)),
            provenance=_provenance(bundle),
        )

    @app.get(
        "/projections/{season}/{week}/{player_id}",
        response_model=Projection,
        tags=["projections"],
        responses={404: {"model": ErrorResponse}},
    )
    def player(season: int, week: int, player_id: str, bundle: BundleDep) -> Projection:
        meta = bundle.metadata
        if (season, week) != (meta.season, meta.week):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"this instance serves {meta.season} week {meta.week} "
                    f"(bundle {meta.version}); {season} week {week} is not in it"
                ),
            )
        frame = _ranked(bundle.projections)
        match = frame[frame["player_id"] == player_id]
        if match.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"player {player_id!r} is not in the {meta.season} week {meta.week} "
                    "universe. Players enter it only after an appearance in one of "
                    "their team's previous three games (see docs/leakage.md)."
                ),
            )
        PROJECTIONS_SERVED.labels("player").inc(1)
        return next(_to_projections(match))

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["projections"],
        responses={422: {"model": ErrorResponse}},
    )
    def predict(payload: PredictRequest, bundle: BundleDep) -> PredictResponse:
        """Score supplied history with the real predictor.

        The same object the backtest scored, fitted on exactly the history in
        the request. Week 1 is refused here as it is everywhere else - the
        schema enforces it, so the rule is one number in one place rather than
        a check each caller has to remember.
        """
        name = payload.predictor or bundle.metadata.predictor or PUBLISHED_PREDICTOR_NAME
        chosen = next((p for p in default_baselines() if p.name == name), None)
        if chosen is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"unknown predictor {name!r}; available: "
                    f"{sorted(p.name for p in default_baselines())}"
                ),
            )

        history = pd.DataFrame(
            [
                {
                    "season": g.season,
                    "week": g.week,
                    "player_id": p.player_id,
                    "position": p.position,
                    "fantasy_points": g.fantasy_points,
                    "played": g.played,
                }
                for p in payload.players
                for g in p.games
            ]
        )
        targets = pd.DataFrame(
            [
                {
                    "season": payload.season,
                    "week": payload.week,
                    "player_id": p.player_id,
                    "position": p.position,
                }
                for p in payload.players
            ]
        )

        chosen.fit(history)
        scored = chosen.predict(targets)
        PROJECTIONS_SERVED.labels("predict").inc(len(targets))

        return PredictResponse(
            season=payload.season,
            week=payload.week,
            predictor=chosen.name,
            predictions=[
                Prediction(
                    player_id=str(row.player_id),
                    position=str(row.position),  # type: ignore[arg-type]
                    projected_points=round(float(value), 2),
                )
                for row, value in zip(targets.itertuples(index=False), scored, strict=True)
            ],
            provenance=_provenance(bundle),
        )


app = create_app()
