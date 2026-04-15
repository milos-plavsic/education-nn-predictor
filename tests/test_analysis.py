from pathlib import Path

from analysis.report import generate_report


def test_generate_report_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NN_EPOCHS", "3")
    monkeypatch.setenv("FT_PHASE1_EPOCHS", "2")
    monkeypatch.setenv("FT_PHASE2_EPOCHS", "2")
    out = generate_report(tmp_path, random_state=0)
    assert "output_dir" in out
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "figures" / "baseline_val_actual_vs_pred.png").is_file()
    assert (tmp_path / "figures" / "mae_val_comparison.png").is_file()
