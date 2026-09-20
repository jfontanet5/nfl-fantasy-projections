"""End-to-end CLI runs against live upstream data.

Argument parsing and pre-flight validation are unit-tested in
``tests/unit/test_cli.py``; what is left here is the part that genuinely needs
the network - that a real backtest produces a scorecard on disk with provenance.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from nflproj.cli import app
from nflproj.config import reset_settings

runner = CliRunner()


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
