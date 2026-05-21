from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from counter_uas.config import AugmentationConfig, ExperimentConfig
from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold
from counter_uas.models.factory import create_model, needs_raw_waveform
from counter_uas.training.dataset import AudioWindowDataset, LABEL_TO_INDEX, Perturbation
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


def _stable_window_seed(seed: int, clip_id: str, start_sample: int) -> int:
    key = f"{seed}:{clip_id}:{start_sample}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=False)


def build_training_perturbation(
    augmentation: AugmentationConfig,
    seed: int,
) -> Perturbation | None:
    if not augmentation.enabled:
        return None

    def perturb(
        waveform: np.ndarray,
        sample_rate: int,
        clip_id: str,
        start_sample: int,
    ) -> np.ndarray:
        augmented = np.asarray(waveform, dtype=np.float32).copy()
        rng = np.random.default_rng(_stable_window_seed(seed, clip_id, start_sample))

        if augmentation.gain_db:
            gain_db = float(rng.choice(augmentation.gain_db))
            augmented *= np.float32(10 ** (gain_db / 20.0))

        if augmentation.noise_snr_db:
            snr_db = float(rng.choice(augmentation.noise_snr_db))
            signal_power = float(np.mean(augmented**2))
            if signal_power == 0.0:
                signal_power = 1e-12
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise = rng.normal(
                0.0, np.sqrt(noise_power), size=augmented.shape
            ).astype(np.float32)
            augmented = augmented + noise

        if augmentation.time_mask_fraction > 0.0:
            mask_len = int(len(augmented) * augmentation.time_mask_fraction)
            if mask_len > 0 and mask_len < len(augmented):
                start = int(rng.integers(0, len(augmented) - mask_len + 1))
                augmented[start : start + mask_len] = 0.0

        return augmented.astype(np.float32, copy=False)

    return perturb


def _build_balanced_sampler(dataset: AudioWindowDataset) -> WeightedRandomSampler | None:
    labels = [
        LABEL_TO_INDEX[str(dataset.rows.iloc[row_index]["label"])]
        for row_index, _ in dataset.index
    ]
    counts = pd.Series(labels).value_counts().to_dict()
    if len(counts) < 2:
        return None
    weights = torch.as_tensor(
        [1.0 / counts[label] for label in labels],
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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

    raw = needs_raw_waveform(config.model)
    train_ds = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        split="train",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
        perturbation=build_training_perturbation(
            config.training.augmentation,
            seed=config.seed,
        ),
        return_raw_waveform=raw,
    )
    val_ds = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        split="val",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
        return_raw_waveform=raw,
    )
    sampler = (
        _build_balanced_sampler(train_ds)
        if config.training.balanced_sampling
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    model = create_model(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_pr_auc = -1.0
    best_metrics: dict[str, float | bool | str] | None = None
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
        threshold = select_threshold(
            y_true,
            y_score,
            objective=config.evaluation.threshold_objective,
            target_recall=config.evaluation.target_recall,
        )
        metrics = compute_binary_metrics(y_true, y_score, threshold, config.evaluation.target_recall)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_metrics = {
                **metrics,
                "checkpoint_metric": "pr_auc",
                "model_name": config.model.name,
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": config.model.name,
                    "model_config": config.model,
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
    if best_metrics is None:
        raise RuntimeError("Training finished without validation metrics")
    best_metrics["stopped_early"] = stopped_early
    best_metrics["epochs_run"] = len(history)
    metrics_text = json.dumps(best_metrics, indent=2)
    (artifacts / "best_validation_metrics.json").write_text(metrics_text)
    (artifacts / "validation_metrics.json").write_text(metrics_text)
    return checkpoint_path
