import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.datasets import DATA_SOURCE
from app.langgraph_predictor import run_agentic_predictor
from finetune.nn_finetune import run_two_phase_finetune

app = FastAPI(title="Education NN Predictor", version="0.2.0")


class FitRequest(BaseModel):
    confidence_threshold: float = Field(0.66, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/fit_metrics")
def fit_metrics(body: FitRequest) -> dict:
    os.environ.setdefault("NN_EPOCHS", os.getenv("NN_EPOCHS", "60"))
    out = run_agentic_predictor(
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )
    return {**out, "data_source": DATA_SOURCE}


@app.post("/v1/finetune_pipeline")
def finetune_pipeline() -> dict:
    return {**run_two_phase_finetune(), "data_source": DATA_SOURCE}
