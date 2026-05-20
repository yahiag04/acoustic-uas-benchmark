# Benchmark Results Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the benchmark so weak noisy-audio results are measured honestly, reported clearly, and addressed with training-time robustness tools plus a stronger CNN option.

**Architecture:** Evaluation will produce both window-level and clip-level artifacts, with clip-level metrics as the primary report surface. Training will accept optional audio augmentations and balanced sampling without changing the default smoke path. Model creation will move behind a small factory so the baseline remains available and an enhanced residual CNN can be selected from config.

**Tech Stack:** Python, PyTorch, torchaudio, pandas, scikit-learn, matplotlib, pytest.

---

### Task 1: Evaluation Correctness

**Files:**
- Modify: `src/counter_uas/evaluation/evaluate.py`
- Modify: `src/counter_uas/evaluation/metrics.py`
- Modify: `src/counter_uas/config.py`
- Modify: `configs/baseline_cnn.yaml`
- Test: `tests/test_metrics.py`
- Test: `tests/test_evaluate.py`

- [x] Write failing tests for clip aggregation, clip prediction artifacts, error analysis, and noninteractive plotting.
- [x] Implement clip aggregation helpers, select thresholds on validation clips, write `clip_predictions.csv`, and write `error_analysis.csv`.
- [x] Prefix secondary window metrics with `window_` while keeping primary clip metrics unprefixed.
- [x] Configure matplotlib with the `Agg` backend before importing `pyplot`.
- [x] Run the focused evaluation and metrics tests.

### Task 2: Training Robustness

**Files:**
- Modify: `src/counter_uas/config.py`
- Modify: `configs/baseline_cnn.yaml`
- Modify: `src/counter_uas/training/dataset.py`
- Modify: `src/counter_uas/training/train.py`
- Test: `tests/test_config.py`
- Test: `tests/test_features_dataset.py`
- Test: `tests/test_training.py`

- [x] Write failing tests for augmentation config, deterministic per-window perturbation, and best validation metric persistence.
- [x] Add optional additive-noise, gain, and time-mask training perturbations.
- [x] Save `best_validation_metrics.json` and make `validation_metrics.json` reflect the best checkpoint rather than the final epoch.
- [x] Run focused dataset and training tests.

### Task 3: Stronger Model Option

**Files:**
- Create: `src/counter_uas/models/enhanced_cnn.py`
- Create: `src/counter_uas/models/factory.py`
- Modify: `src/counter_uas/models/__init__.py`
- Modify: `src/counter_uas/training/train.py`
- Modify: `src/counter_uas/evaluation/evaluate.py`
- Test: `tests/test_training.py`

- [x] Write failing tests for creating and checkpointing an `enhanced_cnn` model.
- [x] Add a compact residual CNN with dropout and adaptive pooling.
- [x] Use the model factory in training and evaluation checkpoints.
- [x] Run focused model/training tests.

### Task 4: Reporting And Portfolio Output

**Files:**
- Modify: `src/counter_uas/reporting/report.py`
- Test: `tests/test_robustness_report.py`

- [x] Write failing tests for manifest metadata, clip/window metric sections, robustness gap, and known measurement limitations.
- [x] Add dataset metadata, artifact manifest reference, clip/window metric tables, robustness gap summary, and error artifact references.
- [x] Run report tests.

### Task 5: Verification

**Files:**
- All changed files.

- [x] Run `python3 -m pytest -q` without requiring `MPLBACKEND=Agg`.
- [x] Run a smoke train/evaluate/report path on synthetic data.
- [x] Review `git diff` for unrelated changes before final response.
