from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Math). https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_xy_for_grade_prediction() -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = project_root() / "data" / "student-mat.csv"
    df = pd.read_csv(path, sep=";")
    y = df["G3"].to_numpy(dtype=np.float32)
    Xdf = df.drop(columns=["G3"])
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    return Xdf.to_numpy(dtype=np.float32), y, list(Xdf.columns)
