"""Fetch UCI Student Performance CSVs from the official archive (dataset 320).

The legacy direct ``/ml/machine-learning-databases/00320/*.csv`` URLs return 404.
This module downloads the published ``student%2Bperformance.zip``, then reads the
nested ``student.zip`` and extracts ``student-mat.csv`` / ``student-por.csv``.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

UCI_320_ZIP = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"
USER_AGENT = "Mozilla/5.0 (compatible; portfolio-report/1.0)"

_ALLOWED = frozenset({"student-mat.csv", "student-por.csv"})


def fetch_uci_student_csv(filename: str, dest: Path) -> None:
    if filename not in _ALLOWED:
        raise ValueError(f"unsupported UCI student file: {filename!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(UCI_320_ZIP, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        outer = resp.read()
    with zipfile.ZipFile(io.BytesIO(outer)) as z_outer:
        inner_zip = z_outer.read("student.zip")
    with zipfile.ZipFile(io.BytesIO(inner_zip)) as z_inner:
        dest.write_bytes(z_inner.read(filename))
