"""Neural network training with safety checks."""

from __future__ import annotations

import numpy as np
from ml_core import configure_logging
from ml_core.exceptions import ApplicationError
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = configure_logging("nn_train")


class ModelError(ApplicationError):
    """Raised when model training/inference fails."""


class _LegacyValidator:
    """Adapter so existing code using validator.validate_arrays() still works."""

    @staticmethod
    def validate_arrays(
        X,
        y=None,
        *,
        allow_inf: bool = False,
    ) -> None:
        import numpy as np

        if X is None or len(X) == 0:
            raise ModelError("X array is None or empty")
        if not allow_inf and not np.isfinite(X).all():
            raise ModelError("X contains NaN or inf values")
        if y is not None:
            if len(y) != len(X):
                raise ModelError(f"X and y length mismatch: {len(X)} vs {len(y)}")
            if not allow_inf and not np.isfinite(y).all():
                raise ModelError("y contains NaN or inf values")


validator = _LegacyValidator()


class NNTrainer:
    """Neural network trainer with validation."""

    # Safe ranges for hyperparameters
    HIDDEN_LAYER_SIZE_RANGE = (10, 1000)
    LEARNING_RATE_RANGE = (1e-5, 1e-1)
    ALPHA_RANGE = (1e-6, 1e-1)
    MAX_ITERATIONS = 5000

    @classmethod
    def validate_hyperparams(
        cls,
        hidden_layer_sizes: tuple[int, ...],
        learning_rate: float,
        alpha: float,
        max_iter: int,
    ) -> None:
        """Validate neural network hyperparameters.

        Raises:
            ModelError: If parameters are invalid
        """
        # Validate hidden layers
        for i, size in enumerate(hidden_layer_sizes):
            if not isinstance(size, int):
                raise ModelError(f"hidden_layer_sizes[{i}] must be int, got {type(size)}")

            min_size, max_size = cls.HIDDEN_LAYER_SIZE_RANGE
            if not (min_size <= size <= max_size):
                raise ModelError(
                    f"hidden_layer_sizes[{i}] must be in [{min_size}, {max_size}], " f"got {size}"
                )

        # Validate learning rate
        min_lr, max_lr = cls.LEARNING_RATE_RANGE
        if not (min_lr <= learning_rate <= max_lr):
            raise ModelError(f"learning_rate must be in [{min_lr}, {max_lr}], got {learning_rate}")

        # Validate alpha
        min_alpha, max_alpha = cls.ALPHA_RANGE
        if not (min_alpha <= alpha <= max_alpha):
            raise ModelError(f"alpha must be in [{min_alpha}, {max_alpha}], got {alpha}")

        # Validate max iterations
        if not (1 <= max_iter <= cls.MAX_ITERATIONS):
            raise ModelError(f"max_iter must be in [1, {cls.MAX_ITERATIONS}], got {max_iter}")

        logger.info(
            f"Hyperparameters valid: hidden={hidden_layer_sizes}, "
            f"lr={learning_rate}, alpha={alpha}"
        )

    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        learning_rate: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> tuple[MLPRegressor, StandardScaler]:
        """Train neural network with validation and divergence detection.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix (optional)
            y_test: Test target vector (optional)
            hidden_layer_sizes: Hidden layer sizes
            learning_rate: Learning rate
            alpha: L2 regularization
            max_iter: Maximum iterations
            random_state: Random seed

        Returns:
            Tuple of (trained model, scaler)

        Raises:
            ModelError: If training fails or diverges
        """
        # Validate inputs
        validator.validate_arrays(X_train, y_train, allow_inf=False)
        if X_test is not None:
            validator.validate_arrays(X_test, y_test, allow_inf=False)

        # Validate hyperparameters
        cls.validate_hyperparams(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate=learning_rate,
            alpha=alpha,
            max_iter=max_iter,
        )

        logger.info(f"Training NN on {len(X_train)} samples, {X_train.shape[1]} features")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)

        # Train with monitoring
        try:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,  # Stop if validation score doesn't improve
                validation_fraction=0.1,
                batch_size=32,
                warm_start=False,
                verbose=0,
            )

            # Fit with test data if available for validation
            if X_test is not None:
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)

                # Check for divergence
                if not np.isfinite(y_pred).all():
                    raise ModelError("Model diverged: predictions contain NaN/Inf")

                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)

                if np.isnan(rmse) or np.isinf(rmse):
                    raise ModelError("Model diverged: RMSE is NaN/Inf")

                logger.info(f"Test RMSE: {rmse:.4f}")
            else:
                model.fit(X_train_scaled, y_train)

                # Check for divergence on training set
                y_pred = model.predict(X_train_scaled)
                if not np.isfinite(y_pred).all():
                    raise ModelError("Model diverged: predictions contain NaN/Inf")

            logger.info(f"Training complete. Loss: {model.loss_:.6f}")

        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Neural network training failed: {e}") from e

        return model, scaler


# ---------------------------------------------------------------------------
# Backward-compatible module-level helper used by app.langgraph_predictor
# ---------------------------------------------------------------------------


def train_and_evaluate(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int | None = None,
) -> dict[str, float]:
    """Load the standard dataset, train an NN, and return val metrics.

    A thin wrapper around :class:`NNTrainer` that the LangGraph workflow uses
    as its baseline node. ``epochs`` overrides ``max_iter``; if ``None``, the
    value is read from the ``NN_EPOCHS`` environment variable so the orchestration
    layer can re-use the same knob, defaulting to 30.
    """
    import os

    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    from app.datasets import load_xy_for_grade_prediction

    X, y, _features = load_xy_for_grade_prediction()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if epochs is None:
        try:
            epochs = int(os.environ.get("NN_EPOCHS", "30"))
        except ValueError:
            epochs = 30

    model, scaler = NNTrainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_iter=max(5, epochs),
        random_state=random_state,
    )
    y_pred = model.predict(scaler.transform(X_test))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"mae": mae, "r2": r2, "val_mae": mae, "val_r2": r2}


def baseline_val_predictions(
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return validation targets, predictions, and metrics for report generation."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    from app.datasets import load_xy_for_grade_prediction

    X, y, _features = load_xy_for_grade_prediction()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model, scaler = NNTrainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_iter=5,
        random_state=random_state,
    )
    pred = model.predict(scaler.transform(X_test))
    metrics = {
        "val_mae": float(mean_absolute_error(y_test, pred)),
        "val_r2": float(r2_score(y_test, pred)),
    }
    return np.asarray(y_test, dtype=float), np.asarray(pred, dtype=float), metrics


# ---------------------------------------------------------------------------
# PyTorch MLP — used by the two-phase fine-tuning pipeline in
# finetune/nn_finetune.py. Kept here to preserve the historical import path.
# ---------------------------------------------------------------------------

try:
    import torch
    from torch import nn as _nn
except ImportError:  # pragma: no cover - torch is an optional heavy dep
    torch = None
    _nn = None


if _nn is not None:

    class MLP(_nn.Module):
        """Two-layer MLP with a separable backbone for transfer-learning.

        Phase 1 (``set_finetune_mode(False)``) trains the whole network end to
        end. Phase 2 (``set_finetune_mode(True)``) freezes the hidden layers
        and only updates the final regression head with a lower learning rate.
        """

        def __init__(self, n_features: int, hidden: int = 64) -> None:
            """Initialize a new MLP instance."""
            super().__init__()
            self.backbone = _nn.Sequential(
                _nn.Linear(n_features, hidden),
                _nn.ReLU(),
                _nn.Linear(hidden, hidden // 2),
                _nn.ReLU(),
            )
            self.head = _nn.Linear(hidden // 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Perform the forward operation on this mlp."""
            return self.head(self.backbone(x))

        def set_finetune_mode(self, finetune: bool) -> None:
            """Toggle whether the backbone parameters require gradients."""
            for param in self.backbone.parameters():
                param.requires_grad = not finetune
            for param in self.head.parameters():
                param.requires_grad = True
