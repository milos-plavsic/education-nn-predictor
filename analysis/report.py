from __future__ import annotations

import math
from pathlib import Path

from analysis.json_util import dumps_pretty
from analysis.plotting import mae_comparison_bar, scatter_actual_vs_predicted
from analysis.stats_utils import regression_summary
from app.datasets import DATA_SOURCE
from app.nn_train import baseline_val_predictions
from finetune.nn_finetune import finetune_val_predictions


def generate_report(out_dir: Path | None = None, random_state: int = 42) -> dict:
    out = Path(out_dir or "reports")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    y_b, p_b, base_m = baseline_val_predictions(random_state=random_state)
    yv, p1, p2, ft_m = finetune_val_predictions(random_state=random_state)

    stats_baseline = regression_summary(y_b, p_b)
    stats_pre = regression_summary(yv, p1)
    stats_ft = regression_summary(yv, p2)

    summary = {
        "data_source": DATA_SOURCE,
        "target": "G3",
        "random_state": random_state,
        "baseline_val_metrics": {**base_m, "regression": stats_baseline},
        "finetune_val_metrics": {
            **ft_m,
            "after_pretrain": stats_pre,
            "after_head_finetune": stats_ft,
        },
    }
    (out / "summary.json").write_text(dumps_pretty(summary), encoding="utf-8")

    scatter_actual_vs_predicted(
        y_b, p_b, fig_dir / "baseline_val_actual_vs_pred.png", title="Baseline MLP — validation"
    )
    scatter_actual_vs_predicted(
        yv, p1, fig_dir / "finetune_after_pretrain_val.png", title="Two-phase NN — after pretrain (val)"
    )
    scatter_actual_vs_predicted(
        yv, p2, fig_dir / "finetune_after_head_val.png", title="Two-phase NN — after head fine-tune (val)"
    )
    mae_comparison_bar(
        ["Baseline MLP", "After pretrain", "After head FT"],
        [
            stats_baseline["mae"],
            stats_pre["mae"],
            stats_ft["mae"],
        ],
        fig_dir / "mae_val_comparison.png",
        title="Validation MAE — baseline vs two-phase training",
    )

    def _fmt(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        return f"{x:.3f}"

    md = "\n".join(
        [
            "# Neural grade predictor — validation statistics",
            "",
            f"**Data:** {DATA_SOURCE}",
            "",
            "## Baseline MLP (single phase)",
            "",
            f"- val MAE: **{_fmt(stats_baseline['mae'])}**, R²: **{_fmt(stats_baseline['r2'])}**",
            "",
            "## Two-phase fine-tune (same validation split)",
            "",
            f"- After pretrain — MAE: **{_fmt(stats_pre['mae'])}**, R²: **{_fmt(stats_pre['r2'])}**",
            f"- After head fine-tune — MAE: **{_fmt(stats_ft['mae'])}**, R²: **{_fmt(stats_ft['r2'])}**",
            "",
            "## Figures",
            "",
            "- `figures/baseline_val_actual_vs_pred.png`",
            "- `figures/finetune_after_pretrain_val.png`",
            "- `figures/finetune_after_head_val.png`",
            "- `figures/mae_val_comparison.png`",
        ]
    )
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return {"output_dir": str(out.resolve())}


def main() -> None:
    print(dumps_pretty(generate_report()))


if __name__ == "__main__":
    main()
