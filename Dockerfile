# syntax=docker/dockerfile:1.7
#
# Multi-stage build. The builder resolves and installs dependencies into a
# virtualenv; the runtime stage copies that venv and nothing else, so uv, build
# tooling and the lockfile never ship.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

# UV_PROJECT_ENVIRONMENT, not VIRTUAL_ENV, is what `uv sync` honours. Setting
# VIRTUAL_ENV alone silently installs into ./.venv instead, leaving /opt/venv
# empty - the image then builds cleanly and fails on first run.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies resolve from the lockfile alone, so this layer is cached until
# the lockfile itself changes - source edits do not trigger a reinstall.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Prove the entrypoint exists and imports before shipping the layer. A broken
# environment is then a build failure with a clear cause, rather than an image
# that passes `docker build` and dies on `docker run`.
RUN /opt/venv/bin/nflproj --help > /dev/null && \
    /opt/venv/bin/python -c "import nflproj.cli, pandas, pyarrow, duckdb"

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

# Stamped into /version so a running pod can name the commit it was built from.
ARG GIT_SHA=unknown

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
    NFLPROJ_REPORTS_DIR=/reports \
    NFLPROJ_BUNDLE_DIR=/bundle \
    NFLPROJ_GIT_SHA=${GIT_SHA}

COPY --from=builder --chown=nflproj:nflproj /opt/venv /opt/venv

RUN mkdir -p /data /reports && chown -R nflproj:nflproj /data /reports
VOLUME ["/data", "/reports"]

# The model artifact travels inside the image, so the image tag *is* the model
# version and a rollback is redeploying last week's tag. A weekly batch cadence
# makes that the boring choice: there is no model store to be unavailable, no
# runtime fetch to fail, and nothing to drift between what was tested and what
# is serving.
#
# `bundle/` always exists in the build context (it is kept with a .gitkeep), so
# a build without `nflproj publish` still succeeds - the resulting image simply
# comes up not-ready and says why, which is the correct behaviour for a serving
# container that has nothing to serve.
COPY --chown=nflproj:nflproj bundle/ /bundle/

USER nflproj
WORKDIR /home/nflproj
EXPOSE 8000

# Fails if the package or any dependency did not make it into the image.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["python", "-c", "import nflproj.cli; nflproj.cli.app"]

ENTRYPOINT ["nflproj"]
CMD ["--help"]
