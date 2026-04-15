from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Math). https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


_UCI_STUDENT_MAT = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
)
_UCI_STUDENT_MAT_HTTP = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
)


def _download_first_working(urls: tuple[str, ...], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; portfolio-report/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as e:
            last_err = e
    raise RuntimeError("Could not download student-mat.csv from UCI") from last_err


def _ensure_student_mat_csv(path: Path) -> None:
    if path.exists():
        return
    _download_first_working((_UCI_STUDENT_MAT, _UCI_STUDENT_MAT_HTTP), path)


def load_xy_for_grade_prediction() -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    y = df["G3"].to_numpy(dtype=np.float32)
    Xdf = df.drop(columns=["G3"])
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    return Xdf.to_numpy(dtype=np.float32), y, list(Xdf.columns)
