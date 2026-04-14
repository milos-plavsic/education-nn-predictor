# Education NN Predictor (PyTorch)

[![CI](https://github.com/milos-plavsic/education-nn-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/milos-plavsic/education-nn-predictor/actions/workflows/ci.yml)
[![Python3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

Feedforward **neural network** (PyTorch MLP) trained on **real secondary-school student records** to predict the **final math grade (G3)** from earlier performance and survey attributes.

## Dataset (real world, education sector)

- **UCI Student Performance — Mathematics (`student-mat.csv`)**  
  Source: [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)  
  Citation: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008.

The CSV is vendored under `data/` for reproducible CI and offline demos.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make run
make test
make api
```

`make install` installs **CPU-only PyTorch** first (smaller wheels), then Python deps.

## Architecture

```mermaid
flowchart LR
  T[Tabular features] --> S[StandardScaler]
  S --> M[MLP torch.nn]
  M --> G3[predicted G3]
```

## API

- `GET /health`
- `POST /v1/fit_metrics` — trains on a train/val split and returns validation MAE / R² (set `NN_EPOCHS` low for quick demos).

## License

MIT
