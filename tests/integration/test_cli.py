"""CLI smoke tests.

The commands are thin, so these check wiring rather than logic: that each one
is reachable, that the end-to-end path produces a scorecard on disk, and that a
bad predictor name fails with a useful message instead of a traceback.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from nflproj.cli import app
from nflproj.config import reset_settings

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "backtest", "project", "status"):
        assert command in result.stdout


@pytest.mark.parametrize("command", ["ingest", "backtest", "project", "status"])
def test_each_command_has_help(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_unknown_predictor_fails_before_any_download(monkeypatch, tmp_path):
    """Argument validation must precede ingestion, or a typo costs two seasons."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["project", "2024", "--week", "5", "--predictor", "nope"])
    assert result.exit_code != 0
    assert "unknown predictor" in result.output.lower()
    assert not (tmp_path / "raw").exists(), "the command downloaded data before validating"


def test_week_one_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["project", "2024", "--week", "1"])
    assert result.exit_code != 0
    assert "out of scope" in result.output.lower()
    assert not (tmp_path / "raw").exists()


@pytest.mark.network
@pytest.mark.slow
def test_backtest_writes_a_scorecard(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NFLPROJ_REPORTS_DIR", str(tmp_path / "reports"))
    reset_settings()

    result = runner.invoke(app, ["backtest", "--seasons", "2023-2024", "--stem", "test_card"])
    assert result.exit_code == 0, result.stdout

    card = tmp_path / "reports" / "test_card.json"
    assert card.exists()
    payload = json.loads(card.read_text())
    assert payload["seasons"] == [2023, 2024]
    assert payload["slices"]["overall"]
    assert (tmp_path / "reports" / "test_card.md").exists()
    # Provenance must pin the exact upstream bytes the numbers came from.
    assert payload["provenance"]["raw_assets"]
    reset_settings()


@pytest.mark.network
def test_status_reports_current_season(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path / "data"))
    reset_settings()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "last fully completed week" in result.stdout
    reset_settings()
