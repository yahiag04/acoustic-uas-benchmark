# Acoustic Counter-UAS Detection Benchmark
acoustic drone detection. The project trains a binary `drone` / `no_drone` classifier, evaluates threshold-aware detection metrics, measures false alarm behavior, checks inference latency, and produces reports

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
- `clip_predictions.csv`
- `error_analysis.csv`
- `robustness_report.csv`
- `robustness_summary.md`
- `reports/final_report.md`

## Current Modeling Notes

The default report now separates clip-level metrics from window-level metrics. Clip-level metrics are the primary result because repeated windows from one audio clip are not independent samples.

Training supports deterministic noise, gain, and time-mask augmentation through `configs/baseline_cnn.yaml`. Evaluation uses a target-recall/min-FPR threshold objective by default so reports can trade a little recall for fewer false alarms. For stronger experiments, change `model.name` from `baseline_cnn` to `enhanced_cnn`; the enhanced model keeps the same log-Mel input pipeline but adds residual convolution blocks and dropout.
