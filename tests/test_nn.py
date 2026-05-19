import os

os.environ["NN_EPOCHS"] = "5"


def test_train_and_evaluate_smoke() -> None:
    """Execute the test train and evaluate smoke routine."""
    from app.nn_train import train_and_evaluate

    out = train_and_evaluate()
    assert out["val_mae"] >= 0
    assert out["val_r2"] <= 1.0
