from app.langgraph_predictor import run_agentic_predictor


def test_predictor_loop_hits_max_iterations_with_strict_threshold() -> None:
    """Test predictor loop hits max iterations with strict threshold."""
    out = run_agentic_predictor(confidence_threshold=0.99, max_iterations=2)
    assert out["iterations"] == 2
    assert out["loop_terminated_reason"] == "max_iterations_reached"
    assert len(out["iteration_history"]) == 2


def test_predictor_returns_selected_model_fields() -> None:
    """Test predictor returns selected model fields."""
    out = run_agentic_predictor(confidence_threshold=0.1, max_iterations=3)
    assert out["selected_model"] in {"baseline_mlp", "two_phase_finetune"}
    assert "baseline_metrics" in out
    assert "finetune_metrics" in out
