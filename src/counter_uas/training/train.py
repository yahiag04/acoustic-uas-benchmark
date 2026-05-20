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
from counter_uas.training.early_stopping import EarlyStopper


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
    early_stopper = EarlyStopper(patience=config.training.patience, mode="max")

    epochs = max_epochs or config.training.epochs
    stopped_early = False
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

        if early_stopper.update(metrics["pr_auc"]):
            stopped_early = True
            break

    pd.DataFrame(history).to_csv(artifacts / "training_history.csv", index=False)
    last_metrics = dict(history[-1])
    last_metrics["stopped_early"] = stopped_early
    last_metrics["epochs_run"] = len(history)
    (artifacts / "validation_metrics.json").write_text(json.dumps(last_metrics, indent=2))
    return checkpoint_path
