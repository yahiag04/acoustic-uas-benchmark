# Acoustic Counter-UAS Detection Benchmark Design

## Goal

Build a reproducible machine learning benchmark for acoustic Counter-UAS detection: classify short audio windows as `drone` or `no_drone` and evaluate the model with metrics that matter for both defense-oriented AI and general ML engineering.

The project is not an operational countermeasure system. It detects likely drone acoustic signatures from public audio data and reports model quality, robustness, and inference cost. It does not include tracking, jamming, targeting, interception, or autonomous response.

## Positioning

The portfolio message is:

> End-to-end robust AI benchmark for acoustic drone detection, covering dataset ingestion, audio preprocessing, supervised training, operational metrics, robustness testing, and reproducible reports.

This is defense-relevant because Counter-UAS systems care about detection rate, false alarm rate, robustness under noise, and deployment constraints. It is also general AI-relevant because the same pipeline patterns apply to audio event detection, anomaly detection, safety monitoring, and edge ML.

## Dataset Scope

Primary dataset: Drone Audio Detection Samples (DADS) on Hugging Face.

Useful public references:

- DADS: https://huggingface.co/datasets/halitergezer/drone-audio-detection-samples
- Original DroneAudioDataset: https://github.com/saraalemadi/DroneAudioDataset
- NASA Small UAS Flyover Acoustics, reserved for later external validation: https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data

DADS is the first implementation target because it is already binary-labeled and standardized as WAV PCM mono audio at 16 kHz. Labels are mapped to:

- `0`: `no_drone`
- `1`: `drone`

The initial benchmark uses DADS only. NASA external validation is explicitly out of scope for the first implementation, but the repository layout should leave room for adding it later.

## Success Criteria

The project is successful when it can:

1. Download or load the DADS dataset through a documented command.
2. Create deterministic train, validation, and test splits.
3. Convert audio into fixed-length model inputs.
4. Train at least one baseline model.
5. Evaluate on the held-out test split without threshold tuning on test data.
6. Save metrics, plots, model checkpoints, and a final Markdown report.
7. Run a small smoke-test path without requiring the full dataset.

## Architecture

The project will be a Python/PyTorch repository with a CLI-first workflow. The top-level structure should be:

```text
project_drone/
  README.md
  pyproject.toml
  configs/
    baseline_cnn.yaml
  data/
    raw/
    processed/
    splits/
  reports/
    figures/
  src/
    counter_uas/
      data/
      features/
      models/
      training/
      evaluation/
      robustness/
      reporting/
      cli.py
  tests/
```

The code should be modular enough that dataset loading, feature extraction, model definition, training, evaluation, and reporting can be tested independently.

## Components

### Dataset Ingestion

Responsibilities:

- Load DADS from Hugging Face or from a local cached path.
- Normalize labels into `drone` and `no_drone`.
- Resample audio to 16 kHz when needed.
- Convert stereo or multi-channel audio to mono if encountered.
- Write a dataset manifest with file identifiers, label, duration, sample rate, and split.

Split policy:

- Use a deterministic 70/15/15 train/validation/test split.
- Prefer source-aware grouping if path or metadata exposes a source/original-recording identifier.
- If source-aware grouping is not possible, use stratified deterministic splitting and write this limitation into the split report.

### Feature Extraction

Primary feature: log-Mel spectrogram.

Default settings:

- Sample rate: 16 kHz.
- Window length: 2 seconds.
- Hop between windows: configurable, default 1 second for overlapping windows.
- Mel bins: 64.
- FFT size: 1024.
- Hop length: 160 samples.
- Window length: 400 samples.

Clips shorter than the target window are zero-padded. Longer clips are segmented into fixed windows. Aggregation from window predictions to clip prediction uses mean probability by default.

### Models

Version 1 includes one required model:

- `BaselineCNN`: compact convolutional classifier over log-Mel spectrograms.

The design should leave a clean interface for a later stronger model such as EfficientNet, PANNs, or Audio Spectrogram Transformer. The first implementation should not block on large pretrained downloads.

### Training

Training uses:

- Binary cross-entropy or cross-entropy loss.
- Class weighting or weighted sampling if class imbalance is material.
- Validation ROC-AUC and PR-AUC for checkpoint selection.
- Early stopping on validation PR-AUC.
- Deterministic seed control where supported.

The training command writes:

- Best model checkpoint.
- Final config snapshot.
- Training history CSV.
- Validation metrics JSON.

### Evaluation

Evaluation must be threshold-aware:

- The decision threshold is selected on validation data.
- The test set is evaluated once with the selected threshold.
- The default threshold objective is maximum F1 on validation.
- A defense-oriented operating point is also reported: lowest false positive rate that still reaches at least 95% validation recall, if achievable.

Test outputs:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `prediction_samples.csv`

Required metrics:

- Accuracy.
- Precision.
- Recall / detection rate.
- F1-score.
- ROC-AUC.
- PR-AUC.
- False positive rate.
- False negative rate.
- FPR at target recall.
- Inference latency per audio window.

### Robustness

Robustness tests run on the test split by applying controlled audio degradations:

- Additive white noise at configurable SNR levels.
- Gain changes.
- Optional time masking or frequency masking if already available through the audio stack.

The first implementation should include additive noise and gain changes. Results are saved as:

- `robustness_report.csv`
- `robustness_summary.md`

### Reporting

A report generator creates `reports/final_report.md` with:

- Dataset summary.
- Split summary.
- Model/config summary.
- Main test metrics.
- Defense-style operating point.
- Robustness table.
- Latency summary.
- Known limitations.

The README should explain the project in portfolio terms and link to the report artifacts.

## Data Flow

1. `prepare-data` loads or discovers DADS and writes manifest plus split files.
2. `train` reads a config, builds datasets, extracts features on the fly or from cache, trains the model, and writes a checkpoint.
3. `evaluate` loads the checkpoint, selects threshold from validation predictions, evaluates test predictions, and writes metrics and figures.
4. `robustness` re-evaluates the checkpoint under controlled perturbations.
5. `report` collects artifacts into a final Markdown report.

## Error Handling

The CLI should fail with actionable messages when:

- The dataset cannot be downloaded or found locally.
- A split file is missing or inconsistent with the manifest.
- Audio files cannot be decoded.
- A checkpoint does not match the requested model architecture.
- Metrics cannot be computed because a split contains only one class.

Decoding failures should be recorded in a skipped-files report rather than silently ignored.

## Testing Strategy

Tests should cover:

- Label normalization.
- Deterministic splitting.
- Audio windowing and padding.
- Log-Mel output shape.
- Baseline model forward pass.
- Metric computation on controlled predictions.
- Threshold selection behavior.
- CLI smoke test using synthetic audio fixtures.

Full training on DADS is not required in automated tests. The repository should provide a small synthetic or fixture-based path that verifies the pipeline without large downloads.

## Out of Scope for Version 1

- Visual detection.
- Radar or RF sensing.
- Drone localization.
- Drone identification by model/type.
- Real-time microphone UI.
- Active countermeasure logic.
- External NASA validation.
- Large pretrained model training as a required path.

## Implementation Notes

Use conservative defaults and keep the project runnable on a standard laptop. The baseline must work without a GPU, though GPU acceleration should be supported automatically when available.

Recommended dependencies:

- PyTorch.
- torchaudio.
- Hugging Face `datasets`.
- scikit-learn.
- pandas.
- numpy.
- matplotlib.
- PyYAML or OmegaConf.
- pytest.

The implementation should optimize for reproducibility and clear artifacts rather than maximum leaderboard accuracy.
