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
    dropout: float = 0.0


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    noise_snr_db: list[float]
    gain_db: list[float]
    time_mask_fraction: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    patience: int
    balanced_sampling: bool
    augmentation: AugmentationConfig


@dataclass(frozen=True)
class EvaluationConfig:
    target_recall: float
    clip_aggregation: str
    threshold_objective: str


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

    augmentation = training.get("augmentation", {})

    return ExperimentConfig(
        seed=int(_require(raw, "seed")),
        data=DataConfig(**data),
        features=FeatureConfig(**features),
        model=ModelConfig(
            name=str(_require(model, "name")),
            channels=list(_require(model, "channels")),
            dropout=float(model.get("dropout", 0.0)),
        ),
        training=TrainingConfig(
            batch_size=int(_require(training, "batch_size")),
            epochs=int(_require(training, "epochs")),
            learning_rate=float(_require(training, "learning_rate")),
            weight_decay=float(_require(training, "weight_decay")),
            num_workers=int(_require(training, "num_workers")),
            patience=int(_require(training, "patience")),
            balanced_sampling=bool(training.get("balanced_sampling", False)),
            augmentation=AugmentationConfig(
                enabled=bool(augmentation.get("enabled", False)),
                noise_snr_db=[
                    float(value) for value in augmentation.get("noise_snr_db", [])
                ],
                gain_db=[float(value) for value in augmentation.get("gain_db", [])],
                time_mask_fraction=float(
                    augmentation.get("time_mask_fraction", 0.0)
                ),
            ),
        ),
        evaluation=EvaluationConfig(
            target_recall=float(_require(evaluation, "target_recall")),
            clip_aggregation=str(evaluation.get("clip_aggregation", "mean")),
            threshold_objective=str(evaluation.get("threshold_objective", "f1")),
        ),
        robustness=RobustnessConfig(**robustness),
    )
