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
