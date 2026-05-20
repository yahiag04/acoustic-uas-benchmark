# Acoustic Counter-UAS Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch benchmark that detects drone presence from audio, reports defense-relevant metrics, and produces portfolio-ready artifacts.

**Architecture:** The repository is a CLI-first Python package. Data preparation writes CSV manifests and deterministic splits, training consumes fixed audio windows, evaluation selects thresholds on validation data, and reporting collects metrics, plots, robustness results, and latency into Markdown artifacts.

**Tech Stack:** Python 3.11+, PyTorch, torchaudio, Hugging Face `datasets`, scikit-learn, pandas, numpy, matplotlib, PyYAML, soundfile, pytest.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, pytest config, console entrypoint.
- `README.md`: portfolio positioning, setup, commands, artifact descriptions.
- `configs/baseline_cnn.yaml`: default experiment configuration.
- `src/counter_uas/config.py`: typed config loading and validation.
- `src/counter_uas/cli.py`: subcommands for synthetic data, DADS preparation, training, evaluation, robustness, and reporting.
- `src/counter_uas/data/audio.py`: mono conversion, resampling hook, fixed-window segmentation, WAV loading.
- `src/counter_uas/data/labels.py`: label normalization.
- `src/counter_uas/data/splits.py`: deterministic stratified split creation.
- `src/counter_uas/data/synthetic.py`: small synthetic audio fixture dataset for tests and smoke runs.
- `src/counter_uas/data/dads.py`: Hugging Face DADS export into local WAV files and manifest CSV.
- `src/counter_uas/features/mel.py`: log-Mel spectrogram transform.
- `src/counter_uas/training/dataset.py`: PyTorch dataset over manifest windows.
- `src/counter_uas/models/baseline_cnn.py`: compact CNN classifier.
- `src/counter_uas/training/train.py`: training loop, validation predictions, checkpoint writing.
- `src/counter_uas/evaluation/metrics.py`: threshold selection and metric computation.
- `src/counter_uas/evaluation/evaluate.py`: checkpoint evaluation, plots, metrics files, latency.
- `src/counter_uas/robustness/perturbations.py`: additive noise and gain perturbations.
- `src/counter_uas/robustness/run.py`: robustness sweep and summary files.
- `src/counter_uas/reporting/report.py`: final Markdown report generation.
- `tests/`: focused unit and smoke tests using synthetic audio, not the full DADS dataset.

---

### Task 1: Package Scaffold And Config Loading

**Files:**
- Create: `pyproject.toml`
- Create: `configs/baseline_cnn.yaml`
- Create: `src/counter_uas/__init__.py`
- Create: `src/counter_uas/config.py`
- Create: `src/counter_uas/cli.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from counter_uas.config import load_config


def test_load_config_exposes_core_sections():
    config = load_config(Path("configs/baseline_cnn.yaml"))

    assert config.seed == 42
    assert config.data.sample_rate == 16_000
    assert config.features.n_mels == 64
    assert config.training.batch_size == 16
    assert config.model.name == "baseline_cnn"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'counter_uas'`.

- [ ] **Step 3: Add package metadata, default config, and config loader**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "counter-uas-acoustic-benchmark"
version = "0.1.0"
description = "Reproducible acoustic Counter-UAS drone detection benchmark"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.19",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "pandas>=2.2",
  "pyyaml>=6.0",
  "scikit-learn>=1.4",
  "soundfile>=0.12",
  "torch>=2.2",
  "torchaudio>=2.2",
  "tqdm>=4.66"
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
counter-uas = "counter_uas.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `configs/baseline_cnn.yaml`:

```yaml
seed: 42
data:
  dataset_name: halitergezer/drone-audio-detection-samples
  sample_rate: 16000
  window_seconds: 2.0
  hop_seconds: 1.0
  manifest_path: data/processed/manifest.csv
  split_dir: data/splits
  artifact_dir: artifacts/baseline_cnn
features:
  n_mels: 64
  n_fft: 1024
  win_length: 400
  hop_length: 160
model:
  name: baseline_cnn
  channels: [16, 32, 64]
training:
  batch_size: 16
  epochs: 3
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 0
  patience: 2
evaluation:
  target_recall: 0.95
robustness:
  snr_db: [20, 10, 0]
  gain_db: [-6, 6]
```

Create `src/counter_uas/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `src/counter_uas/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    manifest_path: str
    split_dir: str
    artifact_dir: str


@dataclass(frozen=True)
class FeatureConfig:
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    channels: list[int]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    target_recall: float


@dataclass(frozen=True)
class RobustnessConfig:
    snr_db: list[float]
    gain_db: list[float]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    robustness: RobustnessConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    data = _require(raw, "data")
    features = _require(raw, "features")
    model = _require(raw, "model")
    training = _require(raw, "training")
    evaluation = _require(raw, "evaluation")
    robustness = _require(raw, "robustness")

    return ExperimentConfig(
        seed=int(_require(raw, "seed")),
        data=DataConfig(**data),
        features=FeatureConfig(**features),
        model=ModelConfig(**model),
        training=TrainingConfig(**training),
        evaluation=EvaluationConfig(**evaluation),
        robustness=RobustnessConfig(**robustness),
    )
```

Create `src/counter_uas/cli.py`:

```python
from __future__ import annotations

import argparse

from counter_uas import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="counter-uas")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/baseline_cnn.yaml src/counter_uas tests/test_config.py
git commit -m "chore: scaffold acoustic benchmark package"
```

---

### Task 2: Labels, Audio Windows, And Deterministic Splits

**Files:**
- Create: `src/counter_uas/data/__init__.py`
- Create: `src/counter_uas/data/labels.py`
- Create: `src/counter_uas/data/audio.py`
- Create: `src/counter_uas/data/splits.py`
- Test: `tests/test_data_primitives.py`

- [ ] **Step 1: Write failing data primitive tests**

Create `tests/test_data_primitives.py`:

```python
import numpy as np
import pandas as pd

from counter_uas.data.audio import segment_waveform, to_mono
from counter_uas.data.labels import normalize_label
from counter_uas.data.splits import assign_splits


def test_normalize_label_handles_known_dads_values():
    assert normalize_label(0) == "no_drone"
    assert normalize_label("0") == "no_drone"
    assert normalize_label(1) == "drone"
    assert normalize_label("drone") == "drone"


def test_to_mono_averages_channels():
    audio = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    mono = to_mono(audio)
    assert np.allclose(mono, np.array([2.0, 6.0], dtype=np.float32))


def test_segment_waveform_pads_short_clip():
    waveform = np.ones(8, dtype=np.float32)
    windows = segment_waveform(waveform, window_samples=16, hop_samples=8)

    assert windows.shape == (1, 16)
    assert np.allclose(windows[0, :8], 1.0)
    assert np.allclose(windows[0, 8:], 0.0)


def test_segment_waveform_segments_long_clip():
    waveform = np.arange(20, dtype=np.float32)
    windows = segment_waveform(waveform, window_samples=8, hop_samples=4)

    assert windows.shape == (4, 8)
    assert np.allclose(windows[1], np.arange(4, 12, dtype=np.float32))


def test_assign_splits_is_deterministic_and_stratified():
    manifest = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(20)],
            "label": ["drone"] * 10 + ["no_drone"] * 10,
            "path": [f"audio/{i}.wav" for i in range(20)],
        }
    )

    first = assign_splits(manifest, seed=123)
    second = assign_splits(manifest, seed=123)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}
    assert set(first.groupby("split")["label"].nunique()) == {2}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_data_primitives.py -v
```

Expected: FAIL with imports missing from `counter_uas.data`.

- [ ] **Step 3: Add label, audio, and split primitives**

Create `src/counter_uas/data/__init__.py`:

```python
```

Create `src/counter_uas/data/labels.py`:

```python
from __future__ import annotations


def normalize_label(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"1", "drone", "uav", "uas"}:
        return "drone"
    if text in {"0", "no_drone", "nodrone", "non_drone", "unknown", "background"}:
        return "no_drone"
    raise ValueError(f"Unsupported label value: {value!r}")
```

Create `src/counter_uas/data/audio.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1).astype(np.float32)
    raise ValueError(f"Expected 1D or 2D audio array, got shape {array.shape}")


def segment_waveform(
    waveform: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {audio.shape}")
    if len(audio) <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :]

    starts = list(range(0, len(audio) - window_samples + 1, hop_samples))
    if starts[-1] != len(audio) - window_samples:
        starts.append(len(audio) - window_samples)
    return np.stack([audio[start : start + window_samples] for start in starts])


def load_wav_mono(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(Path(path), always_2d=False)
    return to_mono(audio), int(sample_rate)
```

Create `src/counter_uas/data/splits.py`:

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_splits(
    manifest: pd.DataFrame,
    seed: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    required = {"clip_id", "label", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    df = manifest.copy().reset_index(drop=True)
    train_df, remaining = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        remaining,
        test_size=relative_test_size,
        random_state=seed,
        stratify=remaining["label"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("clip_id").reset_index(drop=True)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_data_primitives.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/data tests/test_data_primitives.py
git commit -m "feat: add audio data primitives"
```

---

### Task 3: Synthetic Smoke Dataset And Data Preparation CLI

**Files:**
- Create: `src/counter_uas/data/synthetic.py`
- Create: `src/counter_uas/data/dads.py`
- Modify: `src/counter_uas/cli.py`
- Test: `tests/test_prepare_data.py`

- [ ] **Step 1: Write failing preparation tests**

Create `tests/test_prepare_data.py`:

```python
from pathlib import Path

import pandas as pd

from counter_uas.cli import main
from counter_uas.data.synthetic import create_synthetic_dataset


def test_create_synthetic_dataset_writes_manifest_and_audio(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path, samples_per_class=4, seed=7)

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 8
    assert set(manifest["label"]) == {"drone", "no_drone"}
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert all((tmp_path / path).exists() for path in manifest["path"])


def test_cli_prepare_synthetic(tmp_path):
    exit_code = main(["prepare-synthetic", "--output-dir", str(tmp_path), "--samples-per-class", "4"])

    assert exit_code == 0
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "split_report.md").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_prepare_data.py -v
```

Expected: FAIL because `create_synthetic_dataset` and `prepare-synthetic` are not defined.

- [ ] **Step 3: Add synthetic dataset writer, DADS exporter, and CLI subcommands**

Create `src/counter_uas/data/synthetic.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from counter_uas.data.splits import assign_splits


def _sine_wave(freq: float, sample_rate: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * freq * t)
    noise = 0.03 * rng.normal(size=signal.shape)
    return (signal + noise).astype(np.float32)


def create_synthetic_dataset(
    output_dir: Path | str,
    samples_per_class: int = 12,
    sample_rate: int = 16_000,
    seconds: float = 2.0,
    seed: int = 42,
) -> Path:
    if samples_per_class < 4:
        raise ValueError("samples_per_class must be at least 4")

    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for label, base_freq in [("drone", 220.0), ("no_drone", 880.0)]:
        for index in range(samples_per_class):
            clip_id = f"{label}_{index:04d}"
            freq = base_freq + float(rng.uniform(-15.0, 15.0))
            waveform = _sine_wave(freq, sample_rate, seconds, rng)
            rel_path = Path("audio") / f"{clip_id}.wav"
            sf.write(audio_dir / f"{clip_id}.wav", waveform, sample_rate)
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": rel_path.as_posix(),
                    "label": label,
                    "duration": seconds,
                    "sample_rate": sample_rate,
                    "source_id": clip_id,
                }
            )

    manifest = assign_splits(pd.DataFrame(rows), seed=seed)
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    split_counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    (root / "split_report.md").write_text("# Split Report\n\n" + split_counts.to_markdown() + "\n")
    return manifest_path
```

Create `src/counter_uas/data/dads.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from counter_uas.data.labels import normalize_label
from counter_uas.data.splits import assign_splits


def export_dads_dataset(
    output_dir: Path | str,
    dataset_name: str,
    seed: int,
    max_rows: int | None = None,
) -> Path:
    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=True))
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for index, item in enumerate(tqdm(dataset, desc="Exporting DADS")):
        try:
            label = normalize_label(item["label"])
            audio = item["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            clip_id = f"dads_{index:08d}"
            rel_path = Path("audio") / f"{clip_id}.wav"
            sf.write(audio_dir / rel_path.name, waveform, sample_rate)
            duration = float(len(waveform) / sample_rate)
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": rel_path.as_posix(),
                    "label": label,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "source_id": Path(str(item.get("file", clip_id))).stem,
                }
            )
        except Exception as exc:
            skipped.append(f"{index},{type(exc).__name__},{exc}")

    if not rows:
        raise RuntimeError("No DADS rows were exported")

    manifest = assign_splits(pd.DataFrame(rows), seed=seed)
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (root / "skipped_files.csv").write_text("index,error_type,error\n" + "\n".join(skipped) + ("\n" if skipped else ""))
    split_counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    (root / "split_report.md").write_text("# Split Report\n\n" + split_counts.to_markdown() + "\n")
    return manifest_path
```

Replace `src/counter_uas/cli.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from counter_uas import __version__
from counter_uas.config import load_config
from counter_uas.data.dads import export_dads_dataset
from counter_uas.data.synthetic import create_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="counter-uas")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    synthetic = subparsers.add_parser("prepare-synthetic")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--samples-per-class", type=int, default=12)
    synthetic.add_argument("--seed", type=int, default=42)

    dads = subparsers.add_parser("prepare-dads")
    dads.add_argument("--config", default="configs/baseline_cnn.yaml")
    dads.add_argument("--output-dir", default="data/processed")
    dads.add_argument("--max-rows", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-synthetic":
        create_synthetic_dataset(
            Path(args.output_dir),
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        return 0
    if args.command == "prepare-dads":
        config = load_config(Path(args.config))
        export_dads_dataset(
            output_dir=Path(args.output_dir),
            dataset_name=config.data.dataset_name,
            seed=config.seed,
            max_rows=args.max_rows,
        )
        return 0

    parser.print_help()
    return 0
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_prepare_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/data/synthetic.py src/counter_uas/data/dads.py src/counter_uas/cli.py tests/test_prepare_data.py
git commit -m "feat: add data preparation commands"
```

---

### Task 4: Log-Mel Features And Window Dataset

**Files:**
- Create: `src/counter_uas/features/__init__.py`
- Create: `src/counter_uas/features/mel.py`
- Create: `src/counter_uas/training/__init__.py`
- Create: `src/counter_uas/training/dataset.py`
- Test: `tests/test_features_dataset.py`

- [ ] **Step 1: Write failing feature and dataset tests**

Create `tests/test_features_dataset.py`:

```python
from pathlib import Path

import torch

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.features.mel import LogMelSpectrogram
from counter_uas.training.dataset import AudioWindowDataset


def test_log_mel_returns_channel_first_tensor():
    transform = LogMelSpectrogram(sample_rate=16_000, n_mels=64, n_fft=1024, win_length=400, hop_length=160)
    waveform = torch.zeros(16_000)

    features = transform(waveform)

    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[1] == 64


def test_audio_window_dataset_reads_synthetic_manifest(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path, samples_per_class=4, seed=9)
    config = load_config(Path("configs/baseline_cnn.yaml"))

    dataset = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=tmp_path,
        split="train",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
    )

    features, label, clip_id = dataset[0]
    assert features.shape[0] == 1
    assert features.shape[1] == 64
    assert label in {0, 1}
    assert isinstance(clip_id, str)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_features_dataset.py -v
```

Expected: FAIL because feature and dataset modules are missing.

- [ ] **Step 3: Add log-Mel transform and manifest-backed dataset**

Create `src/counter_uas/features/__init__.py`:

```python
```

Create `src/counter_uas/features/mel.py`:

```python
from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform tensor, got shape {tuple(waveform.shape)}")
        mel = self.mel(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.unsqueeze(0).float()
```

Create `src/counter_uas/training/__init__.py`:

```python
```

Create `src/counter_uas/training/dataset.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from counter_uas.config import FeatureConfig
from counter_uas.data.audio import load_wav_mono, segment_waveform
from counter_uas.features.mel import LogMelSpectrogram


LABEL_TO_INDEX = {"no_drone": 0, "drone": 1}


class AudioWindowDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        manifest_path: Path | str,
        root_dir: Path | str,
        split: str,
        sample_rate: int,
        window_seconds: float,
        hop_seconds: float,
        feature_config: FeatureConfig,
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest = pd.read_csv(manifest_path)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r}")
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.hop_samples = int(sample_rate * hop_seconds)
        self.transform = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=feature_config.n_mels,
            n_fft=feature_config.n_fft,
            win_length=feature_config.win_length,
            hop_length=feature_config.hop_length,
        )
        self.index: list[tuple[int, int]] = []
        for row_index, row in self.rows.iterrows():
            duration = float(row.get("duration", window_seconds))
            n_samples = max(1, int(duration * sample_rate))
            if n_samples <= self.window_samples:
                n_windows = 1
            else:
                n_windows = 1 + max(0, (n_samples - self.window_samples) // self.hop_samples)
            self.index.extend((row_index, window_index) for window_index in range(n_windows))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row_index, window_index = self.index[index]
        row = self.rows.iloc[row_index]
        waveform, actual_sample_rate = load_wav_mono(self.root_dir / str(row["path"]))
        if actual_sample_rate != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {actual_sample_rate}")
        windows = segment_waveform(waveform, self.window_samples, self.hop_samples)
        selected = windows[min(window_index, len(windows) - 1)]
        tensor = torch.from_numpy(selected)
        features = self.transform(tensor)
        label = LABEL_TO_INDEX[str(row["label"])]
        return features, label, str(row["clip_id"])
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_features_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/features src/counter_uas/training tests/test_features_dataset.py
git commit -m "feat: add log-mel window dataset"
```

---

### Task 5: Metrics, Threshold Selection, And Operating Points

**Files:**
- Create: `src/counter_uas/evaluation/__init__.py`
- Create: `src/counter_uas/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
import numpy as np

from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold


def test_select_threshold_maximizes_f1():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(y_true, y_score, objective="f1")

    assert 0.4 < threshold <= 0.6


def test_compute_binary_metrics_contains_defense_fields():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5, target_recall=0.95)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["fpr_at_target_recall"] == 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `counter_uas.evaluation.metrics` is missing.

- [ ] **Step 3: Add metrics implementation**

Create `src/counter_uas/evaluation/__init__.py`:

```python
```

Create `src/counter_uas/evaluation/metrics.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, objective: str = "f1") -> float:
    if objective != "f1":
        raise ValueError(f"Unsupported threshold objective: {objective}")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def fpr_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    candidates = fpr[tpr >= target_recall]
    if len(candidates) == 0:
        return float("nan")
    return float(np.min(candidates))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    target_recall: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "fpr_at_target_recall": fpr_at_recall(y_true, y_score, target_recall),
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/evaluation tests/test_metrics.py
git commit -m "feat: add detection metrics"
```

---

### Task 6: Baseline CNN And Training Loop

**Files:**
- Create: `src/counter_uas/models/__init__.py`
- Create: `src/counter_uas/models/baseline_cnn.py`
- Create: `src/counter_uas/training/train.py`
- Modify: `src/counter_uas/cli.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write failing model and training tests**

Create `tests/test_training.py`:

```python
from pathlib import Path

import torch

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.training.train import train_from_config


def test_baseline_cnn_forward_shape():
    model = BaselineCNN(channels=[8, 16])
    logits = model(torch.zeros(3, 1, 64, 201))

    assert logits.shape == (3,)


def test_training_writes_checkpoint(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path / "data", samples_per_class=4, seed=11)
    config = load_config(Path("configs/baseline_cnn.yaml"))

    checkpoint_path = train_from_config(
        config=config,
        manifest_path=manifest_path,
        root_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        max_epochs=1,
    )

    assert checkpoint_path.exists()
    assert (tmp_path / "artifacts" / "training_history.csv").exists()
    assert (tmp_path / "artifacts" / "validation_predictions.csv").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_training.py -v
```

Expected: FAIL because model and training modules are missing.

- [ ] **Step 3: Add CNN model, training loop, and train CLI**

Create `src/counter_uas/models/__init__.py`:

```python
```

Create `src/counter_uas/models/baseline_cnn.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)
```

Create `src/counter_uas/training/train.py`:

```python
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from counter_uas.config import ExperimentConfig
from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.training.dataset import AudioWindowDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    clip_ids: list[str] = []
    with torch.no_grad():
        for features, target, batch_clip_ids in loader:
            logits = model(features.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(target.numpy().tolist())
            clip_ids.extend(batch_clip_ids)
    return np.asarray(labels), np.asarray(scores), clip_ids


def train_from_config(
    config: ExperimentConfig,
    manifest_path: Path | str,
    root_dir: Path | str,
    artifact_dir: Path | str,
    max_epochs: int | None = None,
) -> Path:
    set_seed(config.seed)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        split="train",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
    )
    val_ds = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        split="val",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
    )
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    model = BaselineCNN(config.model.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_pr_auc = -1.0
    checkpoint_path = artifacts / "best_model.pt"

    epochs = max_epochs or config.training.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for features, target, _ in train_loader:
            features = features.to(device)
            target = target.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        y_true, y_score, clip_ids = _predict(model, val_loader, device)
        threshold = select_threshold(y_true, y_score)
        metrics = compute_binary_metrics(y_true, y_score, threshold, config.evaluation.target_recall)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_channels": config.model.channels,
                    "threshold": threshold,
                    "config": config,
                },
                checkpoint_path,
            )
            pd.DataFrame({"clip_id": clip_ids, "label": y_true, "score": y_score}).to_csv(
                artifacts / "validation_predictions.csv",
                index=False,
            )

    pd.DataFrame(history).to_csv(artifacts / "training_history.csv", index=False)
    (artifacts / "validation_metrics.json").write_text(json.dumps(history[-1], indent=2))
    return checkpoint_path
```

Modify `src/counter_uas/cli.py` by adding imports:

```python
from counter_uas.training.train import train_from_config
```

Add this subparser in `build_parser()` before `return parser`:

```python
    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/baseline_cnn.yaml")
    train.add_argument("--manifest", required=True)
    train.add_argument("--root-dir", required=True)
    train.add_argument("--artifact-dir", default="artifacts/baseline_cnn")
```

Add this branch in `main()` before the final help branch:

```python
    if args.command == "train":
        config = load_config(Path(args.config))
        train_from_config(
            config=config,
            manifest_path=Path(args.manifest),
            root_dir=Path(args.root_dir),
            artifact_dir=Path(args.artifact_dir),
        )
        return 0
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/models src/counter_uas/training/train.py src/counter_uas/cli.py tests/test_training.py
git commit -m "feat: add baseline training"
```

---

### Task 7: Evaluation Artifacts, Curves, And Latency

**Files:**
- Create: `src/counter_uas/evaluation/evaluate.py`
- Modify: `src/counter_uas/cli.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing evaluation test**

Create `tests/test_evaluate.py`:

```python
from pathlib import Path

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.training.train import train_from_config


def test_evaluate_checkpoint_writes_metrics_and_figures(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = create_synthetic_dataset(data_dir, samples_per_class=4, seed=13)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    checkpoint = train_from_config(config, manifest_path, data_dir, tmp_path / "train", max_epochs=1)

    output_dir = tmp_path / "eval"
    metrics = evaluate_checkpoint(config, checkpoint, manifest_path, data_dir, output_dir)

    assert "pr_auc" in metrics
    assert "latency_ms_per_window" in metrics
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "classification_report.txt").exists()
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "pr_curve.png").exists()
    assert (output_dir / "prediction_samples.csv").exists()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: FAIL because `evaluate_checkpoint` is missing.

- [ ] **Step 3: Add evaluation implementation and CLI command**

Create `src/counter_uas/evaluation/evaluate.py`:

```python
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from counter_uas.config import ExperimentConfig
from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.training.dataset import AudioWindowDataset


def _load_model(checkpoint_path: Path | str, device: torch.device) -> tuple[BaselineCNN, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BaselineCNN(checkpoint["model_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint.get("threshold", 0.5))


def _predict(model: BaselineCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    labels: list[int] = []
    scores: list[float] = []
    clip_ids: list[str] = []
    elapsed = 0.0
    windows = 0
    with torch.no_grad():
        for features, target, batch_clip_ids in loader:
            features = features.to(device)
            start = time.perf_counter()
            logits = model(features)
            elapsed += time.perf_counter() - start
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(target.numpy().tolist())
            clip_ids.extend(batch_clip_ids)
            windows += int(features.shape[0])
    latency_ms = (elapsed / max(windows, 1)) * 1000.0
    return np.asarray(labels), np.asarray(scores), clip_ids, latency_ms


def _save_curves(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path) -> None:
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["no_drone", "drone"]).plot(values_format="d")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    root_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved_threshold = _load_model(checkpoint_path, device)

    val_ds = AudioWindowDataset(manifest_path, root_dir, "val", config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds, config.features)
    test_ds = AudioWindowDataset(manifest_path, root_dir, "test", config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds, config.features)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    val_true, val_score, _, _ = _predict(model, val_loader, device)
    threshold = select_threshold(val_true, val_score)
    if not np.isfinite(threshold):
        threshold = saved_threshold

    y_true, y_score, clip_ids, latency_ms = _predict(model, test_loader, device)
    metrics = compute_binary_metrics(y_true, y_score, threshold, config.evaluation.target_recall)
    metrics["latency_ms_per_window"] = float(latency_ms)
    y_pred = (y_score >= threshold).astype(int)

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=["no_drone", "drone"], zero_division=0)
    )
    pd.DataFrame({"clip_id": clip_ids, "label": y_true, "score": y_score, "prediction": y_pred}).to_csv(
        output / "prediction_samples.csv",
        index=False,
    )
    _save_curves(y_true, y_score, output)
    _save_confusion(y_true, y_pred, output)
    return metrics
```

Modify `src/counter_uas/cli.py` by adding:

```python
from counter_uas.evaluation.evaluate import evaluate_checkpoint
```

Add this subparser:

```python
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/baseline_cnn.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--root-dir", required=True)
    evaluate.add_argument("--output-dir", default="artifacts/baseline_cnn/eval")
```

Add this branch:

```python
    if args.command == "evaluate":
        config = load_config(Path(args.config))
        evaluate_checkpoint(config, Path(args.checkpoint), Path(args.manifest), Path(args.root_dir), Path(args.output_dir))
        return 0
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/counter_uas/evaluation/evaluate.py src/counter_uas/cli.py tests/test_evaluate.py
git commit -m "feat: add evaluation artifacts"
```

---

### Task 8: Robustness Sweep, Report Generation, README, And Full Smoke Test

**Files:**
- Create: `src/counter_uas/robustness/__init__.py`
- Create: `src/counter_uas/robustness/perturbations.py`
- Create: `src/counter_uas/robustness/run.py`
- Create: `src/counter_uas/reporting/__init__.py`
- Create: `src/counter_uas/reporting/report.py`
- Modify: `src/counter_uas/cli.py`
- Create: `README.md`
- Test: `tests/test_robustness_report.py`

- [ ] **Step 1: Write failing robustness and report tests**

Create `tests/test_robustness_report.py`:

```python
from pathlib import Path

import numpy as np

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.reporting.report import write_final_report
from counter_uas.robustness.perturbations import add_white_noise_snr, apply_gain_db
from counter_uas.robustness.run import run_robustness
from counter_uas.training.train import train_from_config


def test_audio_perturbations_preserve_shape():
    waveform = np.ones(16000, dtype=np.float32)

    noisy = add_white_noise_snr(waveform, snr_db=10, seed=5)
    gained = apply_gain_db(waveform, gain_db=-6)

    assert noisy.shape == waveform.shape
    assert gained.shape == waveform.shape
    assert np.max(np.abs(gained)) < 1.0


def test_robustness_and_report_write_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = create_synthetic_dataset(data_dir, samples_per_class=4, seed=17)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    checkpoint = train_from_config(config, manifest_path, data_dir, tmp_path / "train", max_epochs=1)
    eval_dir = tmp_path / "eval"
    evaluate_checkpoint(config, checkpoint, manifest_path, data_dir, eval_dir)

    robustness_dir = tmp_path / "robustness"
    run_robustness(config, checkpoint, manifest_path, data_dir, robustness_dir)
    report_path = write_final_report(
        report_path=tmp_path / "final_report.md",
        manifest_path=manifest_path,
        eval_dir=eval_dir,
        robustness_dir=robustness_dir,
    )

    assert (robustness_dir / "robustness_report.csv").exists()
    assert (robustness_dir / "robustness_summary.md").exists()
    assert report_path.exists()
    assert "Acoustic Counter-UAS Detection Benchmark" in report_path.read_text()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_robustness_report.py -v
```

Expected: FAIL because robustness and reporting modules are missing.

- [ ] **Step 3: Add perturbations, robustness runner, report generator, CLI commands, and README**

Create `src/counter_uas/robustness/__init__.py`:

```python
```

Create `src/counter_uas/robustness/perturbations.py`:

```python
from __future__ import annotations

import numpy as np


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20.0)
    return np.asarray(waveform, dtype=np.float32) * np.float32(gain)


def add_white_noise_snr(waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(audio**2))
    if signal_power == 0.0:
        signal_power = 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise
```

Create `src/counter_uas/robustness/run.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from counter_uas.config import ExperimentConfig
from counter_uas.evaluation.evaluate import evaluate_checkpoint


def run_robustness(
    config: ExperimentConfig,
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    root_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    clean_metrics = evaluate_checkpoint(config, checkpoint_path, manifest_path, root_dir, output / "clean")
    rows.append({"condition": "clean", **clean_metrics})

    for snr in config.robustness.snr_db:
        metrics = evaluate_checkpoint(config, checkpoint_path, manifest_path, root_dir, output / f"snr_{snr:g}db")
        rows.append({"condition": f"snr_{snr:g}db", **metrics})

    for gain in config.robustness.gain_db:
        metrics = evaluate_checkpoint(config, checkpoint_path, manifest_path, root_dir, output / f"gain_{gain:g}db")
        rows.append({"condition": f"gain_{gain:g}db", **metrics})

    report = pd.DataFrame(rows)
    report_path = output / "robustness_report.csv"
    report.to_csv(report_path, index=False)
    summary_cols = ["condition", "accuracy", "precision", "recall", "f1", "pr_auc", "false_positive_rate"]
    (output / "robustness_summary.md").write_text("# Robustness Summary\n\n" + report[summary_cols].to_markdown(index=False) + "\n")
    return report_path
```

Create `src/counter_uas/reporting/__init__.py`:

```python
```

Create `src/counter_uas/reporting/report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_final_report(
    report_path: Path | str,
    manifest_path: Path | str,
    eval_dir: Path | str,
    robustness_dir: Path | str,
) -> Path:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    eval_path = Path(eval_dir)
    robustness_path = Path(robustness_dir) / "robustness_report.csv"
    metrics = json.loads((eval_path / "metrics.json").read_text())
    split_summary = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    robustness = pd.read_csv(robustness_path) if robustness_path.exists() else pd.DataFrame()

    lines = [
        "# Acoustic Counter-UAS Detection Benchmark",
        "",
        "## Dataset Summary",
        "",
        f"- Clips: {len(manifest)}",
        f"- Labels: {', '.join(sorted(manifest['label'].unique()))}",
        "",
        split_summary.to_markdown(),
        "",
        "## Main Test Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Robustness",
        "",
        robustness.to_markdown(index=False) if not robustness.empty else "No robustness report found.",
        "",
        "## Known Limitations",
        "",
        "- The first version uses public acoustic data and does not claim operational deployment readiness.",
        "- Evaluation is binary drone/no-drone detection, not localization or drone-type identification.",
        "- Robustness conditions are controlled perturbations, not a substitute for field validation.",
    ]
    report.write_text("\n".join(lines) + "\n")
    return report
```

Modify `src/counter_uas/cli.py` by adding:

```python
from counter_uas.reporting.report import write_final_report
from counter_uas.robustness.run import run_robustness
```

Add subparsers:

```python
    robustness = subparsers.add_parser("robustness")
    robustness.add_argument("--config", default="configs/baseline_cnn.yaml")
    robustness.add_argument("--checkpoint", required=True)
    robustness.add_argument("--manifest", required=True)
    robustness.add_argument("--root-dir", required=True)
    robustness.add_argument("--output-dir", default="artifacts/baseline_cnn/robustness")

    report = subparsers.add_parser("report")
    report.add_argument("--report-path", default="reports/final_report.md")
    report.add_argument("--manifest", required=True)
    report.add_argument("--eval-dir", required=True)
    report.add_argument("--robustness-dir", required=True)
```

Add branches:

```python
    if args.command == "robustness":
        config = load_config(Path(args.config))
        run_robustness(config, Path(args.checkpoint), Path(args.manifest), Path(args.root_dir), Path(args.output_dir))
        return 0
    if args.command == "report":
        write_final_report(Path(args.report_path), Path(args.manifest), Path(args.eval_dir), Path(args.robustness_dir))
        return 0
```

Create `README.md`:

```markdown
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_robustness_report.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS for all tests.

- [ ] **Step 6: Run a CLI smoke workflow**

Run:

```bash
counter-uas prepare-synthetic --output-dir data/synthetic --samples-per-class 12
counter-uas train --manifest data/synthetic/manifest.csv --root-dir data/synthetic --artifact-dir artifacts/baseline_cnn
counter-uas evaluate --checkpoint artifacts/baseline_cnn/best_model.pt --manifest data/synthetic/manifest.csv --root-dir data/synthetic --output-dir artifacts/baseline_cnn/eval
counter-uas robustness --checkpoint artifacts/baseline_cnn/best_model.pt --manifest data/synthetic/manifest.csv --root-dir data/synthetic --output-dir artifacts/baseline_cnn/robustness
counter-uas report --manifest data/synthetic/manifest.csv --eval-dir artifacts/baseline_cnn/eval --robustness-dir artifacts/baseline_cnn/robustness --report-path reports/final_report.md
```

Expected: all commands exit 0 and `reports/final_report.md` exists.

- [ ] **Step 7: Commit**

```bash
git add src/counter_uas/robustness src/counter_uas/reporting src/counter_uas/cli.py README.md tests/test_robustness_report.py
git commit -m "feat: add robustness and reporting"
```

---

## Self-Review

- Spec coverage: The plan covers package setup, DADS and synthetic preparation, deterministic splits, fixed audio windows, log-Mel features, baseline CNN, training, threshold-aware evaluation, required metrics, plots, latency, robustness artifacts, final report, README, and smoke tests.
- Scope control: The plan implements the required baseline path and keeps visual detection, radar, localization, drone type identification, live microphone UI, and NASA external validation outside version 1.
- Type consistency: Config dataclasses, CLI arguments, `AudioWindowDataset`, `BaselineCNN`, metric functions, and artifact paths use the same names across tasks.
- Placeholder scan: The plan contains concrete paths, commands, expected results, and code blocks for each code-changing step.
