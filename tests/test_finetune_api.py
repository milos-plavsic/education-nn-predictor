import pytest


def test_finetune_pipeline_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_PHASE1_EPOCHS", "2")
    monkeypatch.setenv("FT_PHASE2_EPOCHS", "2")
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    r = client.post("/v1/finetune_pipeline")
    assert r.status_code == 200
    data = r.json()
    assert "val_mae_after_head_finetune" in data
    assert "data_source" in data
