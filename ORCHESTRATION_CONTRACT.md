# Orchestration Contract (LangGraph)

This project follows a shared graph-orchestration contract used across portfolio repos.

## Required Runtime Inputs
- `confidence_threshold` in `[0, 1]`
- `max_iterations >= 1`
- domain-specific request parameters (for example `dataset_name`, `goal`, `cv_splits`)

## Required Output Fields
Every graph-driven API/CLI result must include:
- `confidence_score` in `[0, 1]`
- `confidence_label` in `{low, medium, high}`
- `confidence_threshold`
- `iterations`
- `loop_terminated_reason` in:
  - `confidence_threshold_reached`
  - `max_iterations_reached`
  - `retry_with_additional_information`
- `iteration_history` (structured per-iteration metrics)
- `decision_log` (human-readable iteration decisions)

## Shared Confidence Policy
All graphs use `app/orchestration_policy.py`:
- component normalization helpers:
  - `normalized_mae_quality(...)`
  - `normalized_r2_quality(...)`
  - `normalized_stability(...)`
- weighted confidence combiner: `weighted_confidence(...)`
- loop decision helper: `decide_loop(...)`

Default weights:
- `primary_quality`: `0.55`
- `secondary_quality`: `0.30`
- `stability`: `0.15`

## Iterative Retry Pattern
Each graph should follow this pattern:
1. Validate request and initialize state.
2. Plan iteration strategy.
3. Execute training/evaluation step.
4. Compute confidence.
5. If below threshold and under budget, retry with additional information.
6. Finalize and return best known result plus trace.

## Best-Practice Notes
- Keep graph state typed (`TypedDict` / pydantic schema).
- Store all iteration decisions and metrics for observability.
- Keep confidence and loop logic centralized in `orchestration_policy.py`.
- Ensure deterministic defaults (`random_state`) for reproducibility.
