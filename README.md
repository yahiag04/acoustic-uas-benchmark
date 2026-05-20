# Acoustic Counter-UAS Detection Benchmark

Reproducible AI benchmark for acoustic drone detection. The project trains a binary `drone` / `no_drone` classifier, evaluates threshold-aware detection metrics, measures false alarm behavior, checks inference latency, and produces report artifacts suitable for a defense-oriented ML portfolio.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Smoke Run

```bash
counter-uas prepare-synthetic --output-dir data/synthetic --samples-per-class 12
counter-uas train --manifest data/synthetic/manifest.csv --root-dir data/synthetic --artifact-dir artifacts/baseline_cnn
counter-uas evaluate --checkpoint artifacts/baseline_cnn/best_model.pt --manifest data/synthetic/manifest.csv --root-dir data/synthetic --output-dir artifacts/baseline_cnn/eval
counter-uas robustness --checkpoint artifacts/baseline_cnn/best_model.pt --manifest data/synthetic/manifest.csv --root-dir data/synthetic --output-dir artifacts/baseline_cnn/robustness
counter-uas report --manifest data/synthetic/manifest.csv --eval-dir artifacts/baseline_cnn/eval --robustness-dir artifacts/baseline_cnn/robustness --report-path reports/final_report.md
```

## DADS Data Preparation

```bash
counter-uas prepare-dads --config configs/baseline_cnn.yaml --output-dir data/processed
```

Use `--max-rows` for a smaller development subset.

## Artifacts

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `prediction_samples.csv`
- `robustness_report.csv`
- `robustness_summary.md`
- `reports/final_report.md`
