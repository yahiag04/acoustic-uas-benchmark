import json
from dataclasses import replace
from pathlib import Path

import torch

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.models.factory import create_model
from counter_uas.training.train import train_from_config


def test_baseline_cnn_forward_shape():
    model = BaselineCNN(channels=[8, 16])
    logits = model(torch.zeros(3, 1, 64, 201))

    assert logits.shape == (3,)


def test_model_factory_builds_enhanced_cnn_forward_shape():
    config = load_config(Path("configs/baseline_cnn.yaml"))
    model_config = replace(config.model, name="enhanced_cnn", channels=[8, 16])
    model = create_model(model_config)

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
    assert (tmp_path / "artifacts" / "best_validation_metrics.json").exists()
    assert (tmp_path / "artifacts" / "validation_metrics.json").exists()

    best_metrics = json.loads(
        (tmp_path / "artifacts" / "best_validation_metrics.json").read_text()
    )
    validation_metrics = json.loads(
        (tmp_path / "artifacts" / "validation_metrics.json").read_text()
    )

    assert validation_metrics == best_metrics
    assert validation_metrics["checkpoint_metric"] == "pr_auc"
    assert validation_metrics["model_name"] == config.model.name


def test_training_checkpoint_preserves_enhanced_model_name(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path / "data", samples_per_class=4, seed=23)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    config = replace(config, model=replace(config.model, name="enhanced_cnn", channels=[8, 16]))

    checkpoint_path = train_from_config(
        config=config,
        manifest_path=manifest_path,
        root_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        max_epochs=1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["model_name"] == "enhanced_cnn"
