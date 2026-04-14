import os

os.environ["NN_EPOCHS"] = "5"

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_fit_metrics() -> None:
    r = client.post("/v1/fit_metrics")
    assert r.status_code == 200
    data = r.json()
    assert "val_mae" in data
    assert "data_source" in data
