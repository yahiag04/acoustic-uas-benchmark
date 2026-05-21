from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

_mpl_config_dir = Path(tempfile.gettempdir()) / "counter_uas_matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
import matplotlib
matplotlib.use("Agg", force=True)
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

from counter_uas.config import ExperimentConfig, ModelConfig
from counter_uas.evaluation.metrics import (
    aggregate_clip_predictions,
    compute_binary_metrics,
    select_threshold,
)
from counter_uas.models.factory import create_model, needs_raw_waveform
from counter_uas.training.dataset import AudioWindowDataset, Perturbation


def _load_model(checkpoint_path: Path | str, device: torch.device) -> tuple[torch.nn.Module, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = ModelConfig(
            name=str(checkpoint.get("model_name", "baseline_cnn")),
            channels=list(checkpoint["model_channels"]),
        )
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint.get("threshold", 0.5))


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], float]:
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


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved_threshold = _load_model(checkpoint_path, device)

    raw = needs_raw_waveform(config.model)
    val_ds = AudioWindowDataset(
        manifest_path, root_dir, "val",
        config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds,
        config.features,
        return_raw_waveform=raw,
    )
    test_ds = AudioWindowDataset(
        manifest_path, root_dir, "test",
        config.data.sample_rate, config.data.window_seconds, config.data.hop_seconds,
        config.features,
        perturbation=test_perturbation,
        return_raw_waveform=raw,
    )
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    val_true, val_score, val_clip_ids, _ = _predict(model, val_loader, device)
    val_clips = aggregate_clip_predictions(
        val_true,
        val_score,
        val_clip_ids,
        method=config.evaluation.clip_aggregation,
    )
    threshold = select_threshold(
        val_clips["label"].to_numpy(),
        val_clips["score"].to_numpy(),
        objective=config.evaluation.threshold_objective,
        target_recall=config.evaluation.target_recall,
    )
    if not np.isfinite(threshold):
        threshold = saved_threshold

    y_true, y_score, clip_ids, latency_ms = _predict(model, test_loader, device)
    clip_predictions = aggregate_clip_predictions(
        y_true,
        y_score,
        clip_ids,
        method=config.evaluation.clip_aggregation,
    )
    clip_true = clip_predictions["label"].to_numpy()
    clip_score = clip_predictions["score"].to_numpy()
    metrics: dict[str, object] = compute_binary_metrics(
        clip_true,
        clip_score,
        threshold,
        config.evaluation.target_recall,
    )
    window_metrics = compute_binary_metrics(
        y_true,
        y_score,
        threshold,
        config.evaluation.target_recall,
    )
    metrics.update(_prefix_metrics(window_metrics, "window"))
    metrics["evaluation_level"] = "clip"
    metrics["clip_aggregation"] = config.evaluation.clip_aggregation
    metrics["threshold_objective"] = config.evaluation.threshold_objective
    metrics["clip_count"] = int(len(clip_predictions))
    metrics["window_count"] = int(len(y_true))
    metrics["latency_ms_per_window"] = float(latency_ms)
    y_pred = (y_score >= threshold).astype(int)
    clip_predictions["prediction"] = (clip_predictions["score"] >= threshold).astype(int)
    error_analysis = clip_predictions[
        clip_predictions["label"] != clip_predictions["prediction"]
    ].copy()
    error_analysis["error_type"] = np.where(
        error_analysis["prediction"] == 1,
        "false_positive",
        "false_negative",
    )

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "classification_report.txt").write_text(
        classification_report(
            clip_true,
            clip_predictions["prediction"].to_numpy(),
            target_names=["no_drone", "drone"],
            zero_division=0,
        )
    )
    pd.DataFrame({"clip_id": clip_ids, "label": y_true, "score": y_score, "prediction": y_pred}).to_csv(
        output / "prediction_samples.csv",
        index=False,
    )
    clip_predictions.to_csv(output / "clip_predictions.csv", index=False)
    error_analysis.to_csv(output / "error_analysis.csv", index=False)
    _save_curves(clip_true, clip_score, output)
    _save_confusion(clip_true, clip_predictions["prediction"].to_numpy(), output)
    return metrics
