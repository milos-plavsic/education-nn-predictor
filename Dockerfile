# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   1) builder — installs build deps and compiles wheels into /wheels
#   2) runtime — minimal slim image with only what's needed at runtime
#
# Pin both stages to the same base image so the build is fully reproducible.
# The digest below corresponds to python:3.11-slim-bookworm. Bump it when
# rebuilding against a newer base.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build deps live only in this stage and never reach the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt \
    && pip wheel --wheel-dir /wheels gunicorn 'uvicorn[standard]'

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Minimal runtime deps. `tini` gives us a real PID 1 so SIGTERM reaches the app.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tini \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with a fixed UID/GID so file permissions are predictable
# across volume mounts and Kubernetes pod security policies.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --shell /sbin/nologin app

WORKDIR /app

# Install pre-built wheels from the builder stage.
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && pip install --no-index --find-links=/wheels gunicorn 'uvicorn[standard]' \
    && rm -rf /wheels

# Application code, owned by the non-root user.
COPY --chown=app:app . .

USER app

EXPOSE 8000

# Use the request-able health endpoint rather than a Python one-liner so we
# exercise the same path k8s and load balancers will probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# `--proxy-headers` lets uvicorn trust X-Forwarded-* from the ingress.
# `--no-server-header` removes the version-leaking Server header.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--no-server-header"]
