import os

from app.datasets import DATA_SOURCE
from app.nn_train import train_and_evaluate


def main() -> None:
    os.environ.setdefault("NN_EPOCHS", os.getenv("NN_EPOCHS", "60"))
    metrics = train_and_evaluate()
    print("Education NN Predictor (PyTorch MLP on UCI student math)")
    print(DATA_SOURCE)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
