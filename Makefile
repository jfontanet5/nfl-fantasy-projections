.DEFAULT_GOAL := help
.PHONY: help install lint format types test test-all cov check backtest project docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment
	uv sync --all-extras

lint: ## Lint
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix and format
	uv run ruff check --fix .
	uv run ruff format .

types: ## Type check (strict)
	uv run mypy

test: ## Unit tests only - hermetic, no network
	uv run pytest tests/unit -m "not network"

test-all: ## Every test, including live upstream contract tests
	uv run pytest

cov: ## Unit tests with coverage, enforcing the CI floor
	uv run pytest tests/unit -m "not network" \
		--cov=nflproj --cov-report=term-missing --cov-fail-under=85

check: lint types cov ## Everything CI runs on a PR

backtest: ## Full walk-forward backtest and scorecard
	uv run nflproj backtest --seasons 2015-2025

project: ## Project the upcoming week (override: make project SEASON=2026 WEEK=3)
	uv run nflproj project $(or $(SEASON),2026) --week $(or $(WEEK),3)

docker: ## Build the container and verify it runs unprivileged
	docker build -t nflproj:local .
	docker run --rm nflproj:local --help
	@test "$$(docker run --rm --entrypoint id nflproj:local -u)" != "0" \
		&& echo "OK: container runs unprivileged"

clean: ## Remove caches and the reproducible data zone
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -rf data/raw data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
