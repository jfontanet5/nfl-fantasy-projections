"""CLI surface: argument parsing and the validation that runs before any I/O."""

from __future__ import annotations

import json
import os

import pytest
import uvicorn
from typer.testing import CliRunner

from nflproj import cli
from nflproj.ingest import archive as archive_module
from nflproj.ingest.archive import Observation
from nflproj.predictors.baselines import HEADLINE_BASELINE_NAME, default_baselines

runner = CliRunner()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2024", [2024]),
        ("2023-2024", [2023, 2024]),
        ("2015-2018", [2015, 2016, 2017, 2018]),
    ],
)
def test_season_range_parsing(spec, expected):
    assert cli._parse_seasons(spec) == expected


def test_help_lists_every_command():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "ingest",
        "backtest",
        "project",
        "report",
        "snapshot",
        "publish",
        "serve",
        "status",
    ):
        assert command in result.output


@pytest.mark.parametrize(
    "command",
    ["ingest", "backtest", "project", "report", "snapshot", "publish", "serve", "status"],
)
def test_each_command_has_help(command):
    result = runner.invoke(cli.app, [command, "--help"])
    assert result.exit_code == 0


def test_unknown_predictor_fails_before_any_download(monkeypatch, tmp_path):
    """Validation must precede ingestion, or a typo costs two seasons of downloads."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("build_panel must not run before arguments are validated")

    monkeypatch.setattr(cli, "build_panel", _explode)

    result = runner.invoke(cli.app, ["project", "2024", "--week", "5", "--predictor", "nope"])
    assert result.exit_code != 0
    assert "unknown predictor" in result.output.lower()


def test_week_one_is_refused_before_any_download(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("build_panel must not run for an out-of-scope week")

    monkeypatch.setattr(cli, "build_panel", _explode)

    result = runner.invoke(cli.app, ["project", "2024", "--week", "1"])
    assert result.exit_code != 0
    assert "out of scope" in result.output.lower()


def test_the_default_predictor_exists():
    """The CLI default must name a real predictor."""
    assert HEADLINE_BASELINE_NAME in {p.name for p in default_baselines()}


def test_error_message_lists_the_available_predictors(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "build_panel", lambda *_a, **_k: None)
    result = runner.invoke(cli.app, ["project", "2024", "--week", "5", "--predictor", "nope"])
    assert "ewma_hl3" in result.output


def test_snapshot_in_the_offseason_is_a_no_op_not_a_failure(monkeypatch, tmp_path):
    """A red job every night from February to August would train us to ignore it."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "build_team_calendar", lambda *_a, **_k: "calendar")
    monkeypatch.setattr(cli, "snapshot_season", lambda *_a, **_k: None)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nothing should be fetched when no game is imminent")

    monkeypatch.setattr(archive_module.Archive, "capture", _explode)

    result = runner.invoke(cli.app, ["snapshot"])
    assert result.exit_code == 0
    assert "nothing to record" in result.output


def test_snapshot_reports_whether_the_report_changed(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    captured: dict[str, object] = {}

    def _capture(_self: object, season: int, **_kwargs: object) -> Observation:
        captured["season"] = season
        return Observation(
            asset="injuries",
            season=season,
            fetched_at="2026-09-17T22:00:00+00:00",
            sha256="a" * 64,
            rows=433,
            source_url="https://example.invalid/injuries_2026.parquet",
            novel=False,
        )

    monkeypatch.setattr(archive_module.Archive, "capture", _capture)

    result = runner.invoke(cli.app, ["snapshot", "2026"])
    assert result.exit_code == 0
    assert captured["season"] == 2026
    assert "433 rows" in result.output
    assert "unchanged" in result.output


def test_publish_rejects_an_unknown_predictor_before_any_download(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the calendar must not be built before validation")

    monkeypatch.setattr(cli, "build_team_calendar", _explode)

    result = runner.invoke(cli.app, ["publish", "2026", "--predictor", "nope"])
    assert result.exit_code != 0
    assert "unknown predictor" in result.output.lower()


def test_publish_refuses_week_one(monkeypatch, tmp_path):
    """The leakage rule reaches the artifact, not just the report."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "build_team_calendar", lambda *_a, **_k: "calendar")
    monkeypatch.setattr(cli, "current_season", lambda *_a, **_k: 2026)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no panel should be built for an out-of-scope week")

    monkeypatch.setattr(cli, "build_panel", _explode)

    result = runner.invoke(cli.app, ["publish", "2026", "--week", "1"])
    assert result.exit_code != 0
    assert "out of scope" in result.output.lower()


def test_headline_metrics_are_absent_rather_than_zero_without_a_scorecard(tmp_path):
    """A bundle claiming no metrics is honest; one claiming zeros is not."""
    assert cli._headline_metrics(tmp_path / "nothing.json", "any") == {}


def test_headline_metrics_are_absent_when_the_predictor_was_not_scored(tmp_path):
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps({"slices": {"overall": [{"predictor": "other", "mae": 1.0}]}}))
    assert cli._headline_metrics(path, "season_decayed_hl3_d0.5") == {}


def test_headline_metrics_read_the_predictors_own_row(tmp_path):
    path = tmp_path / "scorecard.json"
    path.write_text(
        json.dumps(
            {
                "slices": {
                    "overall": [
                        {"predictor": "other", "mae": 9.9, "spearman": 0.1},
                        {"predictor": "mine", "mae": 4.4, "spearman": 0.6, "mae_skill": 0.04},
                    ]
                }
            }
        )
    )
    metrics = cli._headline_metrics(path, "mine")
    assert metrics["mae"] == pytest.approx(4.4)
    assert metrics["mae_skill"] == pytest.approx(0.04)
    assert "rmse" not in metrics


def test_serve_does_not_clobber_a_configured_bundle_dir(monkeypatch):
    """The image sets NFLPROJ_BUNDLE_DIR=/bundle; the CLI default must not win.

    It did, once. `--bundle` defaulted to the relative string "bundle" and the
    command wrote it into the environment unconditionally, so the absolute path
    baked into the image was replaced by one resolved against the container's
    working directory. The pod started, reported itself not-ready forever, and
    the Kubernetes rollout timed out. Nothing but a real cluster could catch it,
    because the bug lives exactly where an env var and a CLI default meet.
    """
    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", "/bundle")
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: None)

    result = runner.invoke(cli.app, ["serve"])
    assert result.exit_code == 0
    assert os.environ["NFLPROJ_BUNDLE_DIR"] == "/bundle"


def test_serve_flag_overrides_the_environment(monkeypatch):
    """Still an override when asked for explicitly - just not by default."""
    monkeypatch.setenv("NFLPROJ_BUNDLE_DIR", "/bundle")
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: None)

    result = runner.invoke(cli.app, ["serve", "--bundle", "/elsewhere"])
    assert result.exit_code == 0
    assert os.environ["NFLPROJ_BUNDLE_DIR"] == "/elsewhere"
