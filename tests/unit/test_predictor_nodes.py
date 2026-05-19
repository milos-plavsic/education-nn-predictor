"""Unit tests for individual LangGraph predictor nodes and divergence detection.

Each test exercises a node function (or helper) in isolation using a minimal
synthetic state dict — no heavy graph compilation or network access needed.
Tests cover:
  - detect_divergence helper
  - _validate node
  - normalize_node
  - _plan_iteration node
  - _run_baseline node
  - _run_finetune node
  - _assess node
  - _finalize node
  - _route_after_baseline router
  - _route_after_assess router
  - Full compiled graph (end-to-end)
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Helpers: build minimal states
# ---------------------------------------------------------------------------

_BASE_STATE: dict = {
    "confidence_threshold": 0.66,
    "max_iterations": 3,
}


def _make_assessed_state(
    *,
    baseline_mae: float = 2.5,
    baseline_r2: float = 0.6,
    finetune_mae: float = 2.0,
    finetune_r2: float = 0.65,
    iteration: int = 1,
    max_iterations: int = 3,
    confidence_threshold: float = 0.66,
) -> dict:
    """Return a minimal state ready for _assess."""
    return {
        "confidence_threshold": confidence_threshold,
        "max_iterations": max_iterations,
        "iteration": iteration,
        "nn_epochs": 5,
        "phase1_epochs": 3,
        "phase2_epochs": 2,
        "normalize_scale": 1.0,
        "normalize_count": 0,
        "loss_history": [baseline_mae],
        "history": [],
        "decision_log": [],
        "baseline_metrics": {"val_mae": baseline_mae, "val_r2": baseline_r2},
        "finetune_metrics": {
            "val_mae_after_head_finetune": finetune_mae,
            "val_r2_final": finetune_r2,
        },
    }


# ---------------------------------------------------------------------------
# Tests: detect_divergence
# ---------------------------------------------------------------------------


class TestDetectDivergence:
    """Tests for the detect_divergence helper."""

    def test_short_history_returns_false(self):
        """Fewer than window+1 values must return False."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([1.0, 2.0, 3.0], window=3) is False

    def test_strictly_increasing_detects_divergence(self):
        """Four strictly-increasing values with window=3 must return True."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([1.0, 2.0, 3.0, 4.0], window=3) is True

    def test_flat_history_no_divergence(self):
        """Equal values are not strictly increasing — must return False."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([2.0, 2.0, 2.0, 2.0], window=3) is False

    def test_decreasing_no_divergence(self):
        """Strictly decreasing values must not be detected as divergence."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([4.0, 3.0, 2.0, 1.0], window=3) is False

    def test_partial_increase_no_divergence(self):
        """Only last 2 of 3 values increasing is not strictly all-increasing."""
        from app.langgraph_predictor import detect_divergence

        # recent window = [3.0, 2.0, 4.0]: 2.0 < 3.0, so not all increasing
        assert detect_divergence([1.0, 3.0, 2.0, 4.0], window=3) is False

    def test_window_1_single_increase(self):
        """Window=1 means we check the last 2 elements; strictly increasing → True."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([1.0, 2.0], window=1) is True

    def test_empty_list_returns_false(self):
        """Empty loss history must return False (no divergence possible)."""
        from app.langgraph_predictor import detect_divergence

        assert detect_divergence([], window=3) is False


# ---------------------------------------------------------------------------
# Tests: _validate node
# ---------------------------------------------------------------------------


class TestValidateNode:
    """Tests for the _validate initialisation node."""

    def test_validate_sets_iteration_to_zero(self):
        from app.langgraph_predictor import _validate

        result = _validate(dict(_BASE_STATE))
        assert result["iteration"] == 0

    def test_validate_sets_empty_history(self):
        from app.langgraph_predictor import _validate

        result = _validate(dict(_BASE_STATE))
        assert result["history"] == []
        assert result["decision_log"] == []

    def test_validate_sets_normalize_scale_to_one(self):
        from app.langgraph_predictor import _validate

        result = _validate(dict(_BASE_STATE))
        assert result["normalize_scale"] == 1.0

    def test_validate_clips_confidence_threshold(self):
        from app.langgraph_predictor import _validate

        result = _validate({**_BASE_STATE, "confidence_threshold": 1.5})
        assert 0.0 <= result["confidence_threshold"] <= 1.0

    def test_validate_enforces_min_max_iterations(self):
        from app.langgraph_predictor import _validate

        result = _validate({**_BASE_STATE, "max_iterations": 0})
        assert result["max_iterations"] >= 1


# ---------------------------------------------------------------------------
# Tests: normalize_node
# ---------------------------------------------------------------------------


class TestNormalizeNode:
    """Tests for the normalize_node (divergence recovery)."""

    def test_normalize_doubles_scale(self):
        from app.langgraph_predictor import normalize_node

        state = {
            "normalize_scale": 1.0,
            "normalize_count": 0,
            "decision_log": [],
        }
        result = normalize_node(state)
        assert result["normalize_scale"] == 2.0

    def test_normalize_increments_count(self):
        from app.langgraph_predictor import normalize_node

        state = {
            "normalize_scale": 2.0,
            "normalize_count": 1,
            "decision_log": [],
        }
        result = normalize_node(state)
        assert result["normalize_count"] == 2

    def test_normalize_appends_to_decision_log(self):
        from app.langgraph_predictor import normalize_node

        state = {
            "normalize_scale": 1.0,
            "normalize_count": 0,
            "decision_log": ["prior entry"],
        }
        result = normalize_node(state)
        assert len(result["decision_log"]) == 2

    def test_normalize_sets_lr_env_vars(self):
        from app.langgraph_predictor import normalize_node

        state = {
            "normalize_scale": 1.0,
            "normalize_count": 0,
            "decision_log": [],
        }
        normalize_node(state)
        assert "FT_LR_PRETRAIN" in os.environ
        assert "FT_LR_FINETUNE" in os.environ

    def test_normalize_lr_decreases_with_repeated_calls(self):
        from app.langgraph_predictor import normalize_node

        s1 = {"normalize_scale": 1.0, "normalize_count": 0, "decision_log": []}
        r1 = normalize_node(s1)
        lr1 = float(os.environ.get("FT_LR_PRETRAIN", "0.01"))

        s2 = {
            **s1,
            "normalize_scale": r1["normalize_scale"],
            "normalize_count": r1["normalize_count"],
        }
        normalize_node(s2)
        lr2 = float(os.environ.get("FT_LR_PRETRAIN", "0.01"))

        assert lr2 <= lr1


# ---------------------------------------------------------------------------
# Tests: _plan_iteration node
# ---------------------------------------------------------------------------


class TestPlanIterationNode:
    """Tests for the _plan_iteration node."""

    def test_plan_increments_iteration(self):
        from app.langgraph_predictor import _plan_iteration

        state = {"iteration": 0, "decision_log": []}
        result = _plan_iteration(state)
        assert result["iteration"] == 1

    def test_plan_sets_epochs_for_first_iteration(self):
        from app.langgraph_predictor import _plan_iteration

        state = {"iteration": 0, "decision_log": []}
        result = _plan_iteration(state)
        assert result["nn_epochs"] == 5
        assert result["phase1_epochs"] == 3
        assert result["phase2_epochs"] == 2

    def test_plan_sets_epochs_for_later_iterations(self):
        from app.langgraph_predictor import _plan_iteration

        state = {"iteration": 1, "decision_log": []}
        result = _plan_iteration(state)
        assert result["nn_epochs"] == 8

    def test_plan_appends_decision_log(self):
        from app.langgraph_predictor import _plan_iteration

        state = {"iteration": 0, "decision_log": []}
        result = _plan_iteration(state)
        assert len(result["decision_log"]) == 1


# ---------------------------------------------------------------------------
# Tests: _assess node
# ---------------------------------------------------------------------------


class TestAssessNode:
    """Tests for the _assess node."""

    def test_assess_picks_finetune_when_lower_mae(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state(finetune_mae=1.5, baseline_mae=2.5)
        result = _assess(state)
        assert result["selected_model"] == "two_phase_finetune"

    def test_assess_picks_baseline_when_finetune_worse(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state(finetune_mae=3.5, baseline_mae=2.5)
        result = _assess(state)
        assert result["selected_model"] == "baseline_mlp"

    def test_assess_confidence_score_in_bounds(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state()
        result = _assess(state)
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_assess_appends_history(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state()
        state["history"] = []
        result = _assess(state)
        assert len(result["history"]) == 1

    def test_assess_sets_continue_loop_false_at_max_iterations(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state(iteration=3, max_iterations=3, confidence_threshold=0.999)
        result = _assess(state)
        assert result["continue_loop"] is False

    def test_assess_sets_continue_loop_true_when_below_threshold(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state(
            iteration=1,
            max_iterations=5,
            confidence_threshold=0.9999,
        )
        result = _assess(state)
        assert result["continue_loop"] is True

    def test_assess_confidence_label_valid(self):
        from app.langgraph_predictor import _assess

        state = _make_assessed_state()
        result = _assess(state)
        assert result["confidence_label"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Tests: routing functions
# ---------------------------------------------------------------------------


class TestRoutingFunctions:
    """Tests for conditional edge routing functions."""

    def test_route_after_baseline_no_divergence_goes_to_finetune(self):
        from app.langgraph_predictor import _route_after_baseline

        # Only 2 values — cannot have 3-window divergence
        state = {"loss_history": [3.0, 2.0]}
        assert _route_after_baseline(state) == "finetune"

    def test_route_after_baseline_divergence_goes_to_normalize(self):
        from app.langgraph_predictor import _route_after_baseline

        # 4 strictly-increasing values → divergence detected
        state = {"loss_history": [1.0, 2.0, 3.0, 4.0]}
        assert _route_after_baseline(state) == "normalize"

    def test_route_after_assess_continues_when_loop_true(self):
        from app.langgraph_predictor import _route_after_assess

        state = {"continue_loop": True}
        assert _route_after_assess(state) == "plan"

    def test_route_after_assess_finalizes_when_loop_false(self):
        from app.langgraph_predictor import _route_after_assess

        state = {"continue_loop": False}
        assert _route_after_assess(state) == "finalize"


# ---------------------------------------------------------------------------
# Tests: full compiled graph (end-to-end)
# ---------------------------------------------------------------------------


class TestCompiledPredictorGraph:
    """Tests for the full compiled LangGraph predictor pipeline."""

    def test_full_pipeline_returns_dict(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.1, max_iterations=1)
        assert isinstance(result, dict)
        assert result

    def test_full_pipeline_selected_model_is_valid(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.1, max_iterations=1)
        assert result["selected_model"] in {"baseline_mlp", "two_phase_finetune"}

    def test_full_pipeline_confidence_within_bounds(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.5, max_iterations=1)
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_full_pipeline_has_required_keys(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.5, max_iterations=1)
        required = {
            "selected_model",
            "val_mae",
            "val_r2",
            "confidence_score",
            "confidence_label",
            "iterations",
            "loop_terminated_reason",
            "iteration_history",
            "decision_log",
            "data_source",
        }
        missing = required - set(result.keys())
        assert not missing, f"Missing result keys: {missing}"

    def test_full_pipeline_max_iterations_respected(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.9999, max_iterations=2)
        assert result["iterations"] == 2
        assert result["loop_terminated_reason"] == "max_iterations_reached"

    def test_full_pipeline_early_stop_when_threshold_low(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.0, max_iterations=5)
        assert result["iterations"] == 1
        assert result["loop_terminated_reason"] == "confidence_threshold_reached"

    def test_full_pipeline_history_length_matches_iterations(self):
        from app.langgraph_predictor import run_agentic_predictor

        result = run_agentic_predictor(confidence_threshold=0.9999, max_iterations=2)
        assert len(result["iteration_history"]) == result["iterations"]
