from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI — Student Performance (Math). https://archive.ics.uci.edu/dataset/320/student+performance"
)


class _ValidatorShim:
    """Shim so existing callers using validator.validate_dataframe() keep working.

    The original code used app.core_validation.DataValidator; this shim replaces
    that dependency with a lightweight check while application-level code migrates
    to ml_core.validation where stricter validation is needed.
    """

    @staticmethod
    def validate_dataframe(df, **kwargs) -> None:
        """Raise ValueError if the DataFrame is None or empty."""
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError("DataFrame is None or empty")


validator = _ValidatorShim()


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _ensure_student_mat_csv(path: Path) -> None:
    """Download student-mat.csv from UCI if it does not already exist locally."""
    if path.exists():
        return
    try:
        fetch_uci_student_csv("student-mat.csv", path)
    except Exception as e:
        raise RuntimeError("Could not obtain student-mat.csv from UCI") from e


def load_xy_for_grade_prediction() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the UCI student-math dataset and return (X, y, feature_names)."""
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    validator.validate_dataframe(df)
    y = df["G3"].to_numpy(dtype=np.float32)
    Xdf = df.drop(columns=["G3"])
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    return Xdf.to_numpy(dtype=np.float32), y, list(Xdf.columns)
