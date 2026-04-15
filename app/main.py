import os

from app.langgraph_predictor import run_agentic_predictor


def main() -> None:
    confidence_threshold = float(os.getenv("PIPELINE_CONFIDENCE_THRESHOLD", "0.66"))
    max_iterations = int(os.getenv("PIPELINE_MAX_ITERATIONS", "3"))
    result = run_agentic_predictor(
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
    )
    print("Education NN Predictor (LangGraph confidence loop)")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
