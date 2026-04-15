from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI — Student Performance (Math). https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_student_mat_csv(path: Path) -> None:
    if path.exists():
        return
    try:
        fetch_uci_student_csv("student-mat.csv", path)
    except Exception as e:
        raise RuntimeError("Could not obtain student-mat.csv from UCI") from e


def load_xy_for_grade_prediction() -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    y = df["G3"].to_numpy(dtype=np.float32)
    Xdf = df.drop(columns=["G3"])
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    return Xdf.to_numpy(dtype=np.float32), y, list(Xdf.columns)
