from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from ml_core import configure_logging

from app.datasets import DATA_SOURCE
from app.nn_train import train_and_evaluate
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalize_threshold,
    normalized_mae_quality,
    normalized_r2_quality,
    weighted_confidence,
)
from finetune.nn_finetune import run_two_phase_finetune

logger = configure_logging("langgraph_predictor")


# ---------------------------------------------------------------------------
# Divergence detection (CRITICAL)
# ---------------------------------------------------------------------------


def detect_divergence(loss_history: list[float], window: int = 3) -> bool:
    """Return True if the last `window` losses are strictly increasing (diverging)."""
    if len(loss_history) < window + 1:
        return False
    recent = loss_history[-window:]
    return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class IterMetrics(TypedDict):
    """Per-iteration metrics stored in history."""

    iteration: int
    nn_epochs: int
    phase1_epochs: int
    phase2_epochs: int
    baseline_mae: float
    baseline_r2: float
    finetune_mae: float
    finetune_r2: float
    selected_model: str
    selected_mae: float
    selected_r2: float
    confidence_score: float


class PredictorState(TypedDict, total=False):
    """LangGraph state for the education NN predictor pipeline."""

    confidence_threshold: float
    max_iterations: int

    iteration: int
    nn_epochs: int
    phase1_epochs: int
    phase2_epochs: int

    # Scale factor applied by normalize_node when divergence is detected.
    # Starts at 1.0 and is doubled each time divergence is detected.
    normalize_scale: float
    # Running loss history used for divergence detection
    loss_history: list[float]
    # Number of times normalize_node has been triggered in this run
    normalize_count: int

    baseline_metrics: dict[str, float]
    finetune_metrics: dict[str, float]

    selected_model: str
    selected_mae: float
    selected_r2: float

    confidence_score: float
    confidence_label: str
    continue_loop: bool
    stop_reason: str

    history: list[IterMetrics]
    decision_log: list[str]

    error: str | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _validate(state: PredictorState) -> PredictorState:
    """Initialise run-level bookkeeping fields from caller-supplied values."""
    return {
        "confidence_threshold": normalize_threshold(state.get("confidence_threshold", 0.66)),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "iteration": 0,
        "normalize_scale": 1.0,
        "normalize_count": 0,
        "loss_history": [],
        "history": [],
        "decision_log": [],
        "error": None,
    }


def normalize_node(state: PredictorState) -> PredictorState:
    """Apply stronger feature scaling when training diverges.

    Each call doubles `normalize_scale`. The scale value is picked up by
    _run_baseline as an env-var hint for future iterations so the NN uses
    a lower effective learning rate.
    """
    current_scale = float(state.get("normalize_scale", 1.0))
    new_scale = current_scale * 2.0
    count = int(state.get("normalize_count", 0)) + 1
    decision = (
        f"normalize_node triggered (count={count}): scale {current_scale:.1f} → {new_scale:.1f}"
    )
    logger.warning(decision)
    os.environ["FT_LR_PRETRAIN"] = str(max(1e-4, 1e-2 / new_scale))
    os.environ["FT_LR_FINETUNE"] = str(max(1e-5, 5e-4 / new_scale))
    return {
        "normalize_scale": new_scale,
        "normalize_count": count,
        "decision_log": [*state.get("decision_log", []), decision],
    }


def _plan_iteration(state: PredictorState) -> PredictorState:
    """Increment the iteration counter and decide epoch counts for this step."""
    it = int(state.get("iteration", 0)) + 1
    nn_epochs = 5 if it == 1 else 8
    p1 = 3 if it == 1 else 5
    p2 = 2 if it == 1 else 3
    decision = (
        f"iteration={it}: NN_EPOCHS={nn_epochs}, " f"FT_PHASE1_EPOCHS={p1}, FT_PHASE2_EPOCHS={p2}"
    )
    return {
        "iteration": it,
        "nn_epochs": nn_epochs,
        "phase1_epochs": p1,
        "phase2_epochs": p2,
        "decision_log": [*state.get("decision_log", []), decision],
    }


def _run_baseline(state: PredictorState) -> PredictorState:
    """Train the baseline MLP and record its loss for divergence detection."""
    os.environ["NN_EPOCHS"] = str(state.get("nn_epochs", 5))
    try:
        metrics = train_and_evaluate()
    except Exception as exc:  # pragma: no cover
        logger.error(f"Baseline training failed: {exc}")
        return {"error": str(exc), "baseline_metrics": {"val_mae": 99.0, "val_r2": -1.0}}

    # Accumulate loss history (use val_mae as proxy for loss)
    history = [*state.get("loss_history", []), float(metrics.get("val_mae", 0.0))]
    return {
        "baseline_metrics": metrics,
        "loss_history": history,
        "error": None,
    }


def _run_finetune(state: PredictorState) -> PredictorState:
    """Run the two-phase fine-tune on top of the baseline."""
    os.environ["FT_PHASE1_EPOCHS"] = str(state.get("phase1_epochs", 3))
    os.environ["FT_PHASE2_EPOCHS"] = str(state.get("phase2_epochs", 2))
    try:
        metrics = run_two_phase_finetune()
    except Exception as exc:  # pragma: no cover
        logger.error(f"Finetune failed: {exc}")
        baseline_mae = float(state.get("baseline_metrics", {}).get("val_mae", 99.0))
        return {
            "finetune_metrics": {
                "val_mae_after_head_finetune": baseline_mae,
                "val_r2_final": -1.0,
            },
            "error": str(exc),
        }
    return {"finetune_metrics": metrics, "error": None}


def _assess(state: PredictorState) -> PredictorState:
    """Compare baseline vs finetune, compute confidence, decide loop continuation."""
    b = state.get("baseline_metrics", {})
    f = state.get("finetune_metrics", {})
    baseline_mae = float(b.get("val_mae", 99.0))
    baseline_r2 = float(b.get("val_r2", -1.0))
    finetune_mae = float(f.get("val_mae_after_head_finetune", 99.0))
    finetune_r2 = float(f.get("val_r2_final", -1.0))

    if finetune_mae < baseline_mae:
        selected_model = "two_phase_finetune"
        selected_mae = finetune_mae
        selected_r2 = finetune_r2
    else:
        selected_model = "baseline_mlp"
        selected_mae = baseline_mae
        selected_r2 = baseline_r2

    components = {
        "primary_quality": normalized_mae_quality(selected_mae),
        "secondary_quality": normalized_r2_quality(selected_r2),
        "stability": 1.0,
    }
    score = weighted_confidence(components)
    conf_label = confidence_label(score)

    loop = decide_loop(
        confidence_score=score,
        confidence_threshold=float(state.get("confidence_threshold", 0.66)),
        iteration=int(state.get("iteration", 1)),
        max_iterations=int(state.get("max_iterations", 3)),
    )

    h: IterMetrics = {
        "iteration": int(state.get("iteration", 1)),
        "nn_epochs": int(state.get("nn_epochs", 5)),
        "phase1_epochs": int(state.get("phase1_epochs", 3)),
        "phase2_epochs": int(state.get("phase2_epochs", 2)),
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
        "finetune_mae": finetune_mae,
        "finetune_r2": finetune_r2,
        "selected_model": selected_model,
        "selected_mae": selected_mae,
        "selected_r2": selected_r2,
        "confidence_score": score,
    }

    return {
        "selected_model": selected_model,
        "selected_mae": selected_mae,
        "selected_r2": selected_r2,
        "confidence_score": score,
        "confidence_label": conf_label,
        "continue_loop": loop["continue_loop"],
        "stop_reason": loop["stop_reason"],
        "history": [*state.get("history", []), h],
    }


def _finalize(state: PredictorState) -> PredictorState:
    """No-op terminal node that allows the graph to reach END."""
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_baseline(
    state: PredictorState,
) -> Literal["normalize", "finetune"]:
    """After running baseline, check for divergence.

    If the last 3 MAE values are strictly increasing → divert to normalize_node.
    Otherwise continue to fine-tune.
    """
    if detect_divergence(state.get("loss_history", []), window=3):
        logger.warning("Divergence detected — routing to normalize_node")
        return "normalize"
    return "finetune"


def _route_after_assess(
    state: PredictorState,
) -> Literal["plan", "finalize"]:
    """Route back to plan for another iteration, or finalize."""
    return "plan" if state.get("continue_loop", False) else "finalize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_predictor_graph() -> Any:
    """Build and compile the LangGraph predictor with divergence-detection loop."""
    g = StateGraph(PredictorState)

    g.add_node("validate", _validate)
    g.add_node("plan", _plan_iteration)
    g.add_node("baseline", _run_baseline)
    g.add_node("normalize", normalize_node)
    g.add_node("finetune", _run_finetune)
    g.add_node("assess", _assess)
    g.add_node("finalize", _finalize)

    g.set_entry_point("validate")
    g.add_edge("validate", "plan")
    g.add_edge("plan", "baseline")

    # Divergence conditional: after baseline, either normalize or proceed
    g.add_conditional_edges(
        "baseline",
        _route_after_baseline,
        {
            "normalize": "normalize",
            "finetune": "finetune",
        },
    )

    # After normalization, go back to plan (which re-runs baseline with new scale)
    g.add_edge("normalize", "plan")

    g.add_edge("finetune", "assess")

    g.add_conditional_edges(
        "assess",
        _route_after_assess,
        {"plan": "plan", "finalize": "finalize"},
    )

    g.add_edge("finalize", END)
    return g.compile()


_PRED_GRAPH = build_predictor_graph()


def run_agentic_predictor(
    *,
    confidence_threshold: float = 0.66,
    max_iterations: int = 3,
) -> dict:
    """Invoke the compiled LangGraph predictor and return a serialisable result."""
    out = _PRED_GRAPH.invoke(
        {
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
        }
    )
    return {
        "selected_model": out.get("selected_model", "baseline_mlp"),
        "val_mae": out.get("selected_mae", 0.0),
        "val_r2": out.get("selected_r2", 0.0),
        "baseline_metrics": out.get("baseline_metrics", {}),
        "finetune_metrics": out.get("finetune_metrics", {}),
        "confidence_score": float(out.get("confidence_score", 0.0)),
        "confidence_label": out.get("confidence_label", "low"),
        "confidence_threshold": float(out.get("confidence_threshold", confidence_threshold)),
        "iterations": int(out.get("iteration", 1)),
        "loop_terminated_reason": out.get("stop_reason", "max_iterations_reached"),
        "iteration_history": out.get("history", []),
        "decision_log": out.get("decision_log", []),
        "normalize_count": int(out.get("normalize_count", 0)),
        "data_source": DATA_SOURCE,
    }
