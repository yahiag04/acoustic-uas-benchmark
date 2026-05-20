from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from counter_uas.config import ExperimentConfig
from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.training.dataset import AudioWindowDataset, Perturbation


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
    test_perturbation: Perturbation | None = None,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved_threshold = _load_model(checkpoint_path, device)

    val_ds = AudioWindowDataset(
        manifest_path, root_dir, "val",
        config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds,
        config.features,
    )
    test_ds = AudioWindowDataset(
        manifest_path, root_dir, "test",
        config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds,
        config.features,
        perturbation=test_perturbation,
    )
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
