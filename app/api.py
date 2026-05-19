"""FastAPI application for the Education NN Predictor.

Endpoints
---------
POST /v1/fit_metrics         — run the full LangGraph predictor pipeline
POST /v1/finetune_pipeline   — run two-phase NN fine-tune directly
GET  /v1/predictor/status    — most-recent run status
GET  /health                 — liveness probe
GET  /metrics                — Prometheus metrics
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from ml_core import (
    APIKeyMiddleware,
    RateLimiter,
    RateLimitExceeded,
    configure_logging,
    install_middleware,
)
from ml_core.observability import metrics_router, observe_request
from pydantic import BaseModel, Field

from app.datasets import DATA_SOURCE
from app.langgraph_predictor import run_agentic_predictor
from finetune.nn_finetune import run_two_phase_finetune

logger = configure_logging("api")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Education NN Predictor",
    version="1.0.0",
    description=(
        "LangGraph-orchestrated two-phase neural-network predictor "
        "for UCI Student Performance (Math) grade prediction."
    ),
)

# Middleware: request IDs, security headers, CORS
install_middleware(app, cors_allow_origins=("*",))

# API-key auth (no-op when API_KEY env var is unset — dev mode)
app.add_middleware(APIKeyMiddleware)

# Prometheus metrics endpoint
app.include_router(metrics_router)

# Per-IP rate limiter: 20 req/s, burst 40
_limiter = RateLimiter(rate=20.0, burst=40.0)

# ---------------------------------------------------------------------------
# In-memory run-status store
# ---------------------------------------------------------------------------

_status_lock = threading.Lock()
_run_status: dict[str, Any] = {
    "run_id": None,
    "status": "idle",  # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "error": None,
    "result": None,
}


def _get_client_key(request: Request) -> str:
    """Return a stable per-client key for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_dep(request: Request) -> None:
    """FastAPI dependency that enforces per-client rate limiting."""
    key = _get_client_key(request)
    try:
        _limiter.acquire(key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class FitRequest(BaseModel):
    """Request body for POST /v1/fit_metrics."""

    confidence_threshold: float = Field(
        0.66,
        ge=0.0,
        le=1.0,
        description="Confidence target used by the LangGraph retry loop.",
    )
    max_iterations: int = Field(
        3,
        ge=1,
        le=8,
        description="Max refinement iterations before returning best result.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post(
    "/v1/fit_metrics",
    tags=["predictor"],
    dependencies=[Depends(rate_limit_dep)],
)
async def fit_metrics(request: Request, body: FitRequest) -> dict:
    """Run the full LangGraph predictor pipeline and return val metrics + report.

    Uses a confidence-based retry loop: if the model does not reach
    `confidence_threshold` it will re-train (up to `max_iterations` times).
    Divergence in loss history triggers a normalization step automatically.
    """
    run_id = uuid.uuid4().hex
    os.environ.setdefault("NN_EPOCHS", os.getenv("NN_EPOCHS", "30"))

    with _status_lock:
        _run_status.update(
            {
                "run_id": run_id,
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "result": None,
            }
        )

    try:
        out = run_agentic_predictor(
            confidence_threshold=body.confidence_threshold,
            max_iterations=body.max_iterations,
        )
        with _status_lock:
            _run_status.update(
                {
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": out,
                }
            )
        logger.info(f"Predictor run {run_id} completed: confidence={out.get('confidence_score')}")
        return {"run_id": run_id, "data_source": DATA_SOURCE, **out}

    except Exception as exc:
        logger.error(f"Predictor run {run_id} failed: {exc}")
        with _status_lock:
            _run_status.update(
                {
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(exc),
                }
            )
        raise HTTPException(status_code=500, detail=f"Predictor failed: {exc}") from exc


@app.post(
    "/v1/finetune_pipeline",
    tags=["finetune"],
    dependencies=[Depends(rate_limit_dep)],
)
async def finetune_pipeline() -> dict:
    """Run the two-phase NN fine-tune directly and return metrics."""
    return {**run_two_phase_finetune(), "data_source": DATA_SOURCE}


@app.get(
    "/v1/predictor/status",
    tags=["predictor"],
    dependencies=[Depends(rate_limit_dep)],
)
async def predictor_status() -> dict[str, Any]:
    """Return the most recent predictor run status."""
    with _status_lock:
        snapshot = dict(_run_status)

    result = snapshot.pop("result", None)
    if result:
        snapshot["summary"] = {
            "val_mae": result.get("val_mae"),
            "val_r2": result.get("val_r2"),
            "confidence_score": result.get("confidence_score"),
            "iterations": result.get("iterations"),
            "selected_model": result.get("selected_model"),
            "normalize_count": result.get("normalize_count", 0),
        }
    return snapshot


# ---------------------------------------------------------------------------
# Prometheus request-observation middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _observe(request: Request, call_next):
    return await observe_request(request, call_next)
