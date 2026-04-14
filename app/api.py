import os

from fastapi import FastAPI

from app.datasets import DATA_SOURCE
from app.nn_train import train_and_evaluate

app = FastAPI(title="Education NN Predictor", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/fit_metrics")
def fit_metrics() -> dict:
    os.environ.setdefault("NN_EPOCHS", os.getenv("NN_EPOCHS", "60"))
    metrics = train_and_evaluate()
    return {**metrics, "data_source": DATA_SOURCE}
