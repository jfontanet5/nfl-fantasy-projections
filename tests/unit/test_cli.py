"""CLI surface: argument parsing and the validation that runs before any I/O."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest
import uvicorn
from typer.testing import CliRunner

from nflproj import cli
from nflproj.ingest import archive as archive_module
from nflproj.ingest.archive import Observation
from nflproj.predictors.baselines import HEADLINE_BASELINE_NAME, default_baselines
from nflproj.report.health import PageHealth, PageHealthError

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


# ------------------------------------------------- the off-season is not a failure


def _frozen_calendar(monkeypatch, *, season: int | None, upcoming: int | None) -> None:
    """A schedule whose season has ended (or never started)."""
    monkeypatch.setattr(cli, "build_team_calendar", lambda *_a, **_k: "calendar")
    monkeypatch.setattr(cli, "current_season", lambda *_a, **_k: season)
    monkeypatch.setattr(cli, "next_projectable_week", lambda *_a, **_k: upcoming)


@pytest.mark.parametrize("command", ["report", "publish"])
def test_no_upcoming_week_fails_loudly_by_default(monkeypatch, tmp_path, command):
    """Interactively, "there is no week" is a question, not a no-op."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    _frozen_calendar(monkeypatch, season=2026, upcoming=None)
    monkeypatch.setattr(cli, "build_panel", lambda *_a, **_k: None)

    result = runner.invoke(cli.app, [command])
    assert result.exit_code != 0
    assert "no upcoming week" in result.output


@pytest.mark.parametrize("command", ["report", "publish"])
def test_allow_no_week_makes_the_offseason_a_clean_no_op(monkeypatch, tmp_path, command):
    """Seven months of red runs would teach us to stop reading the signal.

    `report` and `publish` derive their week from the schedule, so between the
    last game of a season and the following September there is no week to
    publish. Without this flag the weekly and midweek workflows fail on every
    run for seven months - about 120 red runs that all mean "it is February".
    """
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    _frozen_calendar(monkeypatch, season=2026, upcoming=None)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nothing should be built when there is no week to publish")

    monkeypatch.setattr(cli, "build_panel", _explode)

    result = runner.invoke(cli.app, [command, "--allow-no-week"])
    assert result.exit_code == 0
    assert "no upcoming week in 2026" in result.output


@pytest.mark.parametrize("command", ["report", "publish"])
def test_allow_no_week_does_not_excuse_a_season_that_never_started(monkeypatch, tmp_path, command):
    """Scoped to the off-season, not to every way the schedule can be unusable.

    An empty schedule means the ingest failed or the data is wrong, which is
    actionable and looks nothing like February.
    """
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    _frozen_calendar(monkeypatch, season=None, upcoming=None)
    monkeypatch.setattr(cli, "build_panel", lambda *_a, **_k: None)

    result = runner.invoke(cli.app, [command, "--allow-no-week"])
    assert result.exit_code != 0
    assert "no season has started" in result.output


def test_allow_no_week_still_refuses_an_out_of_scope_week(monkeypatch, tmp_path):
    """The flag covers a missing week, never an invalid one."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    _frozen_calendar(monkeypatch, season=2026, upcoming=1)
    monkeypatch.setattr(cli, "build_panel", lambda *_a, **_k: None)

    result = runner.invoke(cli.app, ["report", "--allow-no-week"])
    assert result.exit_code != 0
    assert "out of scope" in result.output


def test_an_explicit_season_and_week_need_no_upcoming_week(monkeypatch, tmp_path):
    """Rebuilding a past week must not depend on there being a future one."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    _frozen_calendar(monkeypatch, season=2026, upcoming=None)

    target = cli._resolve_target(2026, 5, allow_no_week=False, command="report")
    assert target is not None
    assert (target.season, target.week) == (2026, 5)


def _stale_page(days: float) -> PageHealth:
    now = datetime(2027, 3, 2, tzinfo=UTC)
    return PageHealth(
        url="https://example.invalid/board/",
        generated_at=now - timedelta(days=days),
        fetched_at=now,
    )


def test_page_health_is_red_when_the_page_goes_stale_in_season(monkeypatch, tmp_path):
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "fetch_page_health", lambda *_a, **_k: _stale_page(30))
    _frozen_calendar(monkeypatch, season=2026, upcoming=7)

    result = runner.invoke(cli.app, ["page-health", "--url", "https://example.invalid/board/"])
    assert result.exit_code == 1
    assert "STALE" in result.output


def test_page_health_does_not_call_a_frozen_offseason_page_stale(monkeypatch, tmp_path):
    """Off-season the page is correctly frozen at the last week that was played.

    The age bound measures whether the page is still being deployed, and that
    question only has an answer while there are weeks to deploy. Judging it in
    February measures the calendar.
    """
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "fetch_page_health", lambda *_a, **_k: _stale_page(30))
    _frozen_calendar(monkeypatch, season=2026, upcoming=None)

    result = runner.invoke(
        cli.app,
        ["page-health", "--url", "https://example.invalid/board/", "--allow-no-week"],
    )
    assert result.exit_code == 0
    assert "off-season" in result.output


def test_an_unreachable_page_is_red_even_in_the_offseason(monkeypatch, tmp_path):
    """A 404 is a real failure in any month, and the flag must not hide one."""
    monkeypatch.setenv("NFLPROJ_DATA_DIR", str(tmp_path))

    def _unreachable(*_args: object, **_kwargs: object) -> PageHealth:
        raise PageHealthError("404 Not Found")

    monkeypatch.setattr(cli, "fetch_page_health", _unreachable)
    _frozen_calendar(monkeypatch, season=2026, upcoming=None)

    result = runner.invoke(
        cli.app,
        ["page-health", "--url", "https://example.invalid/board/", "--allow-no-week"],
    )
    assert result.exit_code == 1
    assert "page health unknown" in result.output
