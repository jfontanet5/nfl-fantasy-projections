"""The projection API.

Hermetic: a synthetic bundle on disk, no network, no panel build. What is being
tested is the contract a Kubernetes cluster and a caller both depend on -
particularly that liveness and readiness mean different things, because wiring
both probes to one endpoint is the standard mistake and it fails in two
opposite directions at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nflproj.serving.app import create_app
from nflproj.serving.bundle import PROJECTIONS_NAME, write_bundle

KICKOFF = datetime(2026, 9, 24, 0, 15, tzinfo=UTC)


@pytest.fixture
def board() -> pd.DataFrame:
    rows = [
        ("p1", "A Back", "RB", "ATL", "BAL", 18.5),
        ("p2", "B Wideout", "WR", "BAL", "ATL", 12.25),
        ("p3", "C Passer", "QB", "CHI", "GB", 21.0),
        ("p4", "D Back", "RB", "GB", "CHI", 9.75),
        ("p5", "E Wideout", "WR", "ATL", "BAL", 14.0),
    ]
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "player_id": pid,
                "player_display_name": name,
                "position": pos,
                "team": team,
                "opponent_team": opp,
                "prediction": pred,
            }
            for pid, name, pos, team, opp, pred in rows
        ]
    )


@pytest.fixture
def bundle_root(tmp_path: Path, board: pd.DataFrame) -> Path:
    root = tmp_path / "bundle"
    write_bundle(
        root,
        board,
        predictor="season_decayed_hl3_d0.5",
        season=2026,
        week=3,
        valid_from=KICKOFF,
        raw_assets={"player_stats/2026": "a" * 64},
        metrics={"mae": 4.398133, "spearman": 0.600003},
        baseline="season_to_date_mean",
    )
    return root


@pytest.fixture
def client(bundle_root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", str(bundle_root))
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def clientless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A service that came up with no bundle to serve."""
    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", str(tmp_path / "absent"))
    with TestClient(create_app()) as c:
        yield c


# ----------------------------------------------------------------- probes


def test_liveness_is_true_even_with_no_bundle(clientless):
    """Liveness asks "is this process alive", and it is.

    Answering 503 here would make Kubernetes restart a pod whose problem a
    restart cannot fix - a missing bundle is not a hung process.
    """
    response = clientless.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_is_false_with_no_bundle_and_says_why(clientless):
    response = clientless.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "incomplete" in body["detail"]


def test_readiness_is_true_once_a_bundle_is_loaded(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert len(response.json()["bundle_version"]) == 16


def test_a_tampered_bundle_leaves_the_service_permanently_not_ready(
    bundle_root, board, monkeypatch
):
    """The integrity check has to reach the probe, or it protects nothing.

    A pod holding numbers nobody generated must receive no traffic. Serving
    them while reporting healthy is the exact failure this repo exists to argue
    against.
    """
    edited = board.copy()
    edited.loc[0, "prediction"] = 99.0
    edited.to_parquet(bundle_root / PROJECTIONS_NAME, index=False)

    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", str(bundle_root))
    with TestClient(create_app()) as c:
        assert c.get("/healthz").status_code == 200
        ready = c.get("/readyz")
        assert ready.status_code == 503
        assert "integrity check failed" in ready.json()["detail"]
        assert c.get("/projections/2026/3").status_code == 503


# ----------------------------------------------------------------- the board


def test_the_board_ranks_and_paginates(client):
    body = client.get("/projections/2026/3", params={"limit": 2}).json()
    assert body["count"] == 2
    assert body["total"] == 5
    assert [p["rank"] for p in body["projections"]] == [1, 2]
    assert body["projections"][0]["player_name"] == "C Passer"


def test_offset_walks_the_board_without_regrading(client):
    first = client.get("/projections/2026/3", params={"limit": 2}).json()
    second = client.get("/projections/2026/3", params={"limit": 2, "offset": 2}).json()
    assert [p["rank"] for p in second["projections"]] == [3, 4]
    assert not {p["player_id"] for p in first["projections"]} & {
        p["player_id"] for p in second["projections"]
    }


def test_position_rank_is_board_wide_not_query_wide(client):
    """A filtered call and an unfiltered one must agree about who the RB1 is.

    Ranking after filtering would make `?position=RB` renumber from 1 against a
    different universe, so two responses from the same bundle would disagree.
    """
    full = client.get("/projections/2026/3", params={"limit": 500}).json()
    backs = client.get("/projections/2026/3", params={"position": "RB"}).json()

    expected = {
        p["player_id"]: p["position_rank"] for p in full["projections"] if p["position"] == "RB"
    }
    assert {p["player_id"]: p["position_rank"] for p in backs["projections"]} == expected
    assert backs["total"] == 2


def test_the_position_filter_is_case_insensitive(client):
    assert client.get("/projections/2026/3", params={"position": "rb"}).json()["total"] == 2


def test_a_week_this_instance_does_not_hold_says_what_it_does_hold(client):
    """ "Not found" alone sends an operator to the logs to answer this."""
    response = client.get("/projections/2026/9")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "serves 2026 week 3" in detail
    assert "2026 week 9 is not in it" in detail


def test_an_unknown_player_explains_the_universe_rather_than_404ing_bare(client):
    response = client.get("/projections/2026/3/nobody")
    assert response.status_code == 404
    assert "universe" in response.json()["detail"]


def test_a_single_player_matches_his_row_on_the_board(client):
    board = client.get("/projections/2026/3", params={"limit": 500}).json()
    top = board["projections"][0]
    single = client.get(f"/projections/2026/3/{top['player_id']}").json()
    assert single == top


def test_limits_are_bounded(client):
    """An unbounded limit is a denial-of-service parameter with a friendly name."""
    assert client.get("/projections/2026/3", params={"limit": 10_000}).status_code == 422
    assert client.get("/projections/2026/3", params={"limit": 0}).status_code == 422


# -------------------------------------------------------------- provenance


def test_every_board_response_carries_provenance(client):
    prov = client.get("/projections/2026/3").json()["provenance"]
    assert len(prov["bundle_version"]) == 16
    assert prov["predictor"] == "season_decayed_hl3_d0.5"
    assert prov["baseline"] == "season_to_date_mean"
    assert prov["raw_asset_hashes"] == {"player_stats/2026": "a" * 64}
    assert prov["metrics"]["mae"] == pytest.approx(4.398133)


@pytest.mark.parametrize(
    ("kickoff", "expected"),
    [
        (datetime(2020, 9, 24, 0, 15, tzinfo=UTC), True),
        (datetime(2099, 9, 24, 0, 15, tzinfo=UTC), False),
    ],
)
def test_the_superseded_flag_reaches_the_response(tmp_path, board, monkeypatch, kickoff, expected):
    """The board stays readable either way; the caller is told which it is.

    Both cases use unambiguous dates rather than one near today: a test that
    flips from failing to passing as the calendar advances is not a test. The
    boundary itself is pinned with an injected clock in test_bundle.py.
    """
    root = tmp_path / "bundle"
    write_bundle(
        root,
        board,
        predictor="season_decayed_hl3_d0.5",
        season=2026,
        week=3,
        valid_from=kickoff,
    )
    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", str(root))
    with TestClient(create_app()) as c:
        prov = c.get("/projections/2026/3").json()["provenance"]
    assert prov["valid_from"] == kickoff.isoformat()
    assert prov["superseded"] is expected


def test_version_reports_what_is_deployed(client):
    body = client.get("/version").json()
    assert body["service"] == "nflproj-api"
    assert body["season"] == 2026
    assert body["week"] == 3
    assert body["predictor"] == "season_decayed_hl3_d0.5"


# ------------------------------------------------------------------ predict


def test_predict_runs_the_real_predictor_on_supplied_history(client):
    response = client.post(
        "/predict",
        json={
            "season": 2026,
            "week": 4,
            "players": [
                {
                    "player_id": "x",
                    "position": "RB",
                    "games": [
                        {"season": 2026, "week": 2, "fantasy_points": 18.0},
                        {"season": 2026, "week": 3, "fantasy_points": 4.0},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predictor"] == "season_decayed_hl3_d0.5"
    # Recency-weighted, so strictly between the two games and below their mean.
    points = body["predictions"][0]["projected_points"]
    assert 4.0 < points < 11.0


def test_predict_refuses_week_one(client):
    """The leakage rule holds at the API boundary, not only in the CLI.

    Enforced by the schema, so it is one number in one place rather than a
    check each caller is trusted to remember.
    """
    response = client.post(
        "/predict",
        json={
            "season": 2026,
            "week": 1,
            "players": [
                {
                    "player_id": "x",
                    "position": "RB",
                    "games": [{"season": 2025, "week": 17, "fantasy_points": 12.0}],
                }
            ],
        },
    )
    assert response.status_code == 422


def test_predict_rejects_an_unknown_predictor_by_name(client):
    response = client.post(
        "/predict",
        json={
            "season": 2026,
            "week": 4,
            "predictor": "definitely_not_real",
            "players": [
                {
                    "player_id": "x",
                    "position": "RB",
                    "games": [{"season": 2026, "week": 2, "fantasy_points": 9.0}],
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "unknown predictor" in response.json()["detail"]


def test_predict_rejects_an_unprojected_position(client):
    """Kickers and defenses are out of scope, and the type says so."""
    response = client.post(
        "/predict",
        json={
            "season": 2026,
            "week": 4,
            "players": [
                {
                    "player_id": "k",
                    "position": "K",
                    "games": [{"season": 2026, "week": 2, "fantasy_points": 9.0}],
                }
            ],
        },
    )
    assert response.status_code == 422


def test_predict_rejects_an_empty_player_list(client):
    response = client.post("/predict", json={"season": 2026, "week": 4, "players": []})
    assert response.status_code == 422


def test_predict_answers_for_every_player_asked_about(client):
    players = [
        {
            "player_id": f"p{i}",
            "position": "WR",
            "games": [{"season": 2026, "week": 2, "fantasy_points": float(i)}],
        }
        for i in range(1, 6)
    ]
    body = client.post("/predict", json={"season": 2026, "week": 4, "players": players}).json()
    assert [p["player_id"] for p in body["predictions"]] == [p["player_id"] for p in players]


# ------------------------------------------------------------- observability


def test_metrics_are_labelled_by_route_template_not_by_path(client):
    """Labelling by raw path mints a time series per player id.

    That is how a metrics store gets taken down by a service that looks healthy,
    so it is worth a test rather than a comment.
    """
    client.get("/projections/2026/3/p1")
    client.get("/projections/2026/3/p2")
    body = client.get("/metrics").text

    assert 'route="/projections/{season}/{week}/{player_id}"' in body
    assert 'route="/projections/2026/3/p1"' not in body


def test_projections_served_is_counted(client):
    before = client.get("/metrics").text
    client.get("/projections/2026/3", params={"limit": 3})
    after = client.get("/metrics").text

    def served(text: str) -> float:
        for line in text.splitlines():
            if line.startswith('nflproj_projections_served_total{endpoint="board"}'):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    assert served(after) - served(before) == 3.0


def test_a_request_id_is_echoed_back(client):
    response = client.get("/healthz", headers={"x-request-id": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"


def test_a_request_id_is_generated_when_absent(client):
    assert client.get("/healthz").headers["x-request-id"]


def test_the_openapi_schema_is_served(client):
    """A typed API that does not publish its schema is just a typed API."""
    schema = client.get("/openapi.json").json()
    assert "/projections/{season}/{week}" in schema["paths"]
    assert "/predict" in schema["paths"]
