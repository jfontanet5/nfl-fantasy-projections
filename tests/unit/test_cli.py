"""CLI surface: argument parsing and the validation that runs before any I/O."""

from __future__ import annotations

import pytest
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
    for command in ("ingest", "backtest", "project", "report", "snapshot", "status"):
        assert command in result.output


@pytest.mark.parametrize(
    "command", ["ingest", "backtest", "project", "report", "snapshot", "status"]
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
