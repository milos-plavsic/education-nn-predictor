from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert values for RFC 8259 JSON (no NaN/Inf; NumPy scalars → Python)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_json_serializable(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj
    raise TypeError(f"Not JSON-serializable: {type(obj)!r}")


def dumps_pretty(obj: Any, *, indent: int = 2) -> str:
    return json.dumps(to_json_serializable(obj), indent=indent, allow_nan=False)
