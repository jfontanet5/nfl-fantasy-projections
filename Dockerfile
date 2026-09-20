# syntax=docker/dockerfile:1.7
#
# Multi-stage build. The builder resolves and installs dependencies into a
# virtualenv; the runtime stage copies that venv and nothing else, so uv, build
# tooling and the lockfile never ship.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

WORKDIR /build

# Dependencies resolve from the lockfile alone, so this layer is cached until
# the lockfile itself changes - source edits do not trigger a reinstall.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="nflproj" \
      org.opencontainers.image.description="Weekly NFL fantasy point projections with a published scorecard" \
      org.opencontainers.image.source="https://github.com/jfontanet5/nfl-fantasy-projections" \
      org.opencontainers.image.licenses="MIT"

# Run unprivileged. The data and reports directories are the only paths the
# process writes to, so they are the only ones it owns.
RUN groupadd --system --gid 1001 nflproj && \
    useradd --system --uid 1001 --gid nflproj --create-home nflproj

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NFLPROJ_LOG_JSON=1 \
    NFLPROJ_DATA_DIR=/data \
    NFLPROJ_REPORTS_DIR=/reports

COPY --from=builder --chown=nflproj:nflproj /opt/venv /opt/venv

RUN mkdir -p /data /reports && chown -R nflproj:nflproj /data /reports
VOLUME ["/data", "/reports"]

USER nflproj
WORKDIR /home/nflproj

# Fails if the package or any dependency did not make it into the image.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["python", "-c", "import nflproj.cli; nflproj.cli.app"]

ENTRYPOINT ["nflproj"]
CMD ["--help"]
