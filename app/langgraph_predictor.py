from __future__ import annotations

import os
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.datasets import DATA_SOURCE
from app.nn_train import train_and_evaluate
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalized_mae_quality,
    normalized_r2_quality,
    normalize_threshold,
    weighted_confidence,
)
from finetune.nn_finetune import run_two_phase_finetune


class IterMetrics(TypedDict):
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
    confidence_threshold: float
    max_iterations: int

    iteration: int
    nn_epochs: int
    phase1_epochs: int
    phase2_epochs: int

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


def _validate(state: PredictorState) -> PredictorState:
    return {
        "confidence_threshold": normalize_threshold(state.get("confidence_threshold", 0.66)),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "iteration": 0,
        "history": [],
        "decision_log": [],
    }


def _plan_iteration(state: PredictorState) -> PredictorState:
    it = int(state["iteration"]) + 1
    nn_epochs = 30 if it == 1 else 60
    p1 = 20 if it == 1 else 40
    p2 = 15 if it == 1 else 30
    decision = f"iteration={it}: NN_EPOCHS={nn_epochs}, FT_PHASE1_EPOCHS={p1}, FT_PHASE2_EPOCHS={p2}"
    return {
        "iteration": it,
        "nn_epochs": nn_epochs,
        "phase1_epochs": p1,
        "phase2_epochs": p2,
        "decision_log": [*state["decision_log"], decision],
    }


def _run_baseline(state: PredictorState) -> PredictorState:
    os.environ["NN_EPOCHS"] = str(state["nn_epochs"])
    return {"baseline_metrics": train_and_evaluate()}


def _run_finetune(state: PredictorState) -> PredictorState:
    os.environ["FT_PHASE1_EPOCHS"] = str(state["phase1_epochs"])
    os.environ["FT_PHASE2_EPOCHS"] = str(state["phase2_epochs"])
    return {"finetune_metrics": run_two_phase_finetune()}


def _assess(state: PredictorState) -> PredictorState:
    b = state["baseline_metrics"]
    f = state["finetune_metrics"]
    baseline_mae = float(b["val_mae"])
    baseline_r2 = float(b["val_r2"])
    finetune_mae = float(f["val_mae_after_head_finetune"])
    finetune_r2 = float(f["val_r2_final"])

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
        confidence_threshold=state["confidence_threshold"],
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )

    h: IterMetrics = {
        "iteration": state["iteration"],
        "nn_epochs": state["nn_epochs"],
        "phase1_epochs": state["phase1_epochs"],
        "phase2_epochs": state["phase2_epochs"],
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
        "history": [*state["history"], h],
    }


def _route(state: PredictorState) -> Literal["plan", "finalize"]:
    return "plan" if state["continue_loop"] else "finalize"


def _finalize(state: PredictorState) -> PredictorState:
    return {}


def build_predictor_graph():
    g = StateGraph(PredictorState)
    g.add_node("validate", _validate)
    g.add_node("plan", _plan_iteration)
    g.add_node("baseline", _run_baseline)
    g.add_node("finetune", _run_finetune)
    g.add_node("assess", _assess)
    g.add_node("finalize", _finalize)

    g.set_entry_point("validate")
    g.add_edge("validate", "plan")
    g.add_edge("plan", "baseline")
    g.add_edge("baseline", "finetune")
    g.add_edge("finetune", "assess")
    g.add_conditional_edges("assess", _route, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_PRED_GRAPH = build_predictor_graph()


def run_agentic_predictor(
    *,
    confidence_threshold: float = 0.66,
    max_iterations: int = 3,
) -> dict:
    out = _PRED_GRAPH.invoke(
        {
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
        }
    )
    return {
        "selected_model": out["selected_model"],
        "val_mae": out["selected_mae"],
        "val_r2": out["selected_r2"],
        "baseline_metrics": out["baseline_metrics"],
        "finetune_metrics": out["finetune_metrics"],
        "confidence_score": out["confidence_score"],
        "confidence_label": out["confidence_label"],
        "confidence_threshold": out["confidence_threshold"],
        "iterations": out["iteration"],
        "loop_terminated_reason": out["stop_reason"],
        "iteration_history": out["history"],
        "decision_log": out["decision_log"],
        "data_source": DATA_SOURCE,
    }
