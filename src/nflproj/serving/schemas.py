"""Request and response models.

Every response carries provenance. That is the point of this whole repo, and an
API that returns a bare number is a worse artifact than a markdown file that
says where the number came from. A caller should be able to take any response,
read off the bundle version and the upstream file hashes, and reproduce it.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from nflproj.config import FANTASY_POSITIONS
from nflproj.features.panel import FIRST_PROJECTABLE_WEEK

Position = Literal["QB", "RB", "WR", "TE"]

# Kept in step with the config tuple so adding a position cannot silently skip
# the API layer. A raise rather than an assert: `python -O` strips asserts, and
# this check is worth more in a production image than in a test run.
if set(FANTASY_POSITIONS) != set(get_args(Position)):
    _msg = (
        f"FANTASY_POSITIONS {FANTASY_POSITIONS} and the API's Position type "
        f"{get_args(Position)} disagree; the serving layer would silently reject "
        "a position the rest of the pipeline projects"
    )
    raise RuntimeError(_msg)


class Provenance(BaseModel):
    """Where a response's numbers came from."""

    model_config = ConfigDict(frozen=True)

    bundle_version: str = Field(description="Content hash of the bundle that produced this.")
    predictor: str = Field(description="Predictor that generated the projections.")
    baseline: str | None = Field(
        default=None, description="Named baseline the predictor is measured against."
    )
    generated_at: str = Field(description="When the bundle was built, UTC ISO-8601.")
    valid_from: str | None = Field(
        default=None, description="First kickoff of the covered week, UTC ISO-8601."
    )
    superseded: bool = Field(
        description=(
            "True once the covered week has kicked off. The board is still exactly "
            "what was published; it is a record rather than a forecast."
        )
    )
    raw_asset_hashes: dict[str, str] = Field(
        default_factory=dict, description="SHA-256 of every upstream file behind the board."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The predictor's measured track record when the bundle was built.",
    )


class Projection(BaseModel):
    """One player's projected points for one week."""

    model_config = ConfigDict(frozen=True)

    player_id: str
    player_name: str
    position: Position
    team: str
    opponent: str | None = None
    season: int
    week: int
    projected_points: float = Field(description="Full-PPR fantasy points.")
    rank: int = Field(description="Rank within this response, 1 = highest projection.")
    position_rank: int = Field(description="Rank within the player's position, board-wide.")


class Board(BaseModel):
    """A week's projections."""

    season: int
    week: int
    count: int = Field(description="Rows returned.")
    total: int = Field(description="Rows matching the filter, before limit/offset.")
    projections: list[Projection]
    provenance: Provenance


class PlayerGame(BaseModel):
    """One historical game, as supplied by a caller to /predict."""

    model_config = ConfigDict(frozen=True)

    season: int = Field(ge=1999, le=2100)
    week: int = Field(ge=1, le=22)
    fantasy_points: float = Field(ge=0.0, le=100.0)
    played: bool = True


class PlayerHistory(BaseModel):
    """A player and the games behind him."""

    model_config = ConfigDict(frozen=True)

    player_id: str = Field(min_length=1, max_length=64)
    position: Position
    games: Annotated[list[PlayerGame], Field(min_length=1, max_length=100)]


class PredictRequest(BaseModel):
    """Ad-hoc scoring against supplied history.

    Deliberately takes history rather than a player id. The server holds one
    week's board, not a live panel, so a request naming only a player would
    either be a lookup wearing a prediction's clothes or an invitation to
    quietly read data the caller never saw.
    """

    model_config = ConfigDict(frozen=True)

    season: int = Field(ge=1999, le=2100)
    week: int = Field(
        ge=FIRST_PROJECTABLE_WEEK,
        le=22,
        description=(
            f"Target week. Must be >= {FIRST_PROJECTABLE_WEEK}: week 1 needs offseason "
            "roster data with no clean point-in-time source, so it is out of scope."
        ),
    )
    predictor: str | None = Field(
        default=None, description="Defaults to the predictor the bundle published."
    )
    players: Annotated[list[PlayerHistory], Field(min_length=1, max_length=500)]


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    player_id: str
    position: Position
    projected_points: float


class PredictResponse(BaseModel):
    season: int
    week: int
    predictor: str
    predictions: list[Prediction]
    provenance: Provenance


class VersionResponse(BaseModel):
    service: str
    app_version: str
    bundle_version: str
    predictor: str
    season: int
    week: int
    git_sha: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    bundle_version: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
