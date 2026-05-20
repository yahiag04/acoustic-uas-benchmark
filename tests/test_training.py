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
