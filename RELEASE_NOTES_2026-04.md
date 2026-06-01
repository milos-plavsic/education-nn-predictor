# Release Notes (2026-04)

## Scope
This release adds NN reporting (statistics + plots), improves data fetch resilience, and stabilizes CI including CLI smoke execution.

## Data Source
- UCI Student Performance dataset (ID 320): `student-mat.csv`

## Reporting Added
- New `analysis/` package with:
  - `report.py`, `plotting.py`, `stats_utils.py`, `json_util.py`, module entrypoint
- Generated outputs:
  - `reports/summary.json`
  - `reports/REPORT.md`
  - `reports/figures/baseline_val_actual_vs_pred.png`
  - `reports/figures/finetune_after_pretrain_val.png`
  - `reports/figures/finetune_after_head_val.png`
  - `reports/figures/mae_val_comparison.png`

## Latest Report Snapshot (validation)
- Baseline MLP: MAE `8.8556`, R^2 `-3.5793`
- Two-phase after pretrain: MAE `9.9055`, R^2 `-4.6540`
- Two-phase after head fine-tune: MAE `9.8939`, R^2 `-4.6412`

## Reliability and CI
- Added ZIP-based UCI fetch fallback via `app/uci_fetch.py`.
- Ensured local/offline stability with vendored `data/student-mat.csv`.
- CI runs:
  - CPU torch install
  - tests
  - `python -m analysis` smoke
  - CLI smoke via `python -m app.main` (fixed module import failure from `python app/main.py`).
- Upgraded actions to:
  - `actions/checkout@v6`
  - `actions/setup-python@v6`

## Latest CI Status
- Latest successful run: https://github.com/milos-plavsic/education-nn-predictor/actions/runs/24447655447

## Dependency Notes
- CPU-only PyTorch install in CI: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
