from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict


class LoopDecision(TypedDict):
    """Implementation of the loop decision."""

    continue_loop: bool
    stop_reason: str


def clip01(x: float) -> float:
    """Execute the clip01 routine."""
    return float(max(0.0, min(1.0, x)))


def normalize_threshold(threshold: float) -> float:
    """Execute the normalize threshold routine."""
    return clip01(float(threshold))


def confidence_label(score: float) -> str:
    """Execute the confidence label routine."""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def normalized_mae_quality(mae: float, *, scale: float = 20.0) -> float:
    """Higher is better: 1 means very low MAE, 0 means high MAE."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return clip01(1.0 - clip01(float(mae) / scale))


def normalized_r2_quality(r2: float) -> float:
    """Maps R2 from [-1, 1] to [0, 1], clipped outside range."""
    return clip01((float(r2) + 1.0) / 2.0)


def normalized_stability(std_value: float, *, cap: float = 1.0) -> float:
    """Higher is better: lower metric std => higher stability."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    return clip01(1.0 - min(float(std_value), cap) / cap)


def weighted_confidence(
    components: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute bounded confidence score from named components.

    Contract default components and weights:
    - primary_quality: 0.55
    - secondary_quality: 0.30
    - stability: 0.15
    """
    default_weights: Mapping[str, float] = {
        "primary_quality": 0.55,
        "secondary_quality": 0.30,
        "stability": 0.15,
    }
    ws = weights or default_weights
    score = 0.0
    weight_sum = 0.0
    for k, w in ws.items():
        if w < 0:
            raise ValueError(f"negative weight for {k!r}: {w}")
        score += float(w) * clip01(float(components.get(k, 0.0)))
        weight_sum += float(w)
    if weight_sum == 0:
        raise ValueError("weights sum to zero")
    return clip01(score / weight_sum)


def decide_loop(
    *,
    confidence_score: float,
    confidence_threshold: float,
    iteration: int,
    max_iterations: int,
) -> LoopDecision:
    """Execute the decide loop routine."""
    threshold = normalize_threshold(confidence_threshold)
    reached_conf = confidence_score >= threshold
    reached_limit = int(iteration) >= int(max_iterations)
    continue_loop = not reached_conf and not reached_limit

    if reached_conf:
        reason = "confidence_threshold_reached"
    elif reached_limit:
        reason = "max_iterations_reached"
    else:
        reason = "retry_with_additional_information"

    return {"continue_loop": continue_loop, "stop_reason": reason}


def safe_metric(x: float) -> float | None:
    """Normalize NaN/Inf metrics to None for strict JSON outputs when needed."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return x
