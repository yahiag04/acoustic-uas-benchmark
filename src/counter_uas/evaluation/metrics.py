from __future__ import annotations

import numpy as np
import pandas as pd
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


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective: str = "f1",
    target_recall: float = 0.95,
) -> float:
    if objective not in {"f1", "target_recall_min_fpr"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    if objective == "target_recall_min_fpr":
        thresholds = np.unique(y_score)
        if len(thresholds) == 0:
            return 0.5
        candidates: list[tuple[float, float, float]] = []
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            recall = tp / max(tp + fn, 1)
            if recall < target_recall:
                continue
            false_positive_rate = fp / max(fp + tn, 1)
            candidates.append(
                (float(false_positive_rate), float(-threshold), float(threshold))
            )
        if not candidates:
            return select_threshold(y_true, y_score, objective="f1")
        return min(candidates)[2]

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


def aggregate_clip_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    clip_ids: list[str],
    method: str = "mean",
) -> pd.DataFrame:
    if method not in {"mean", "max"}:
        raise ValueError(f"Unsupported clip aggregation method: {method}")
    if len(y_true) != len(y_score) or len(y_true) != len(clip_ids):
        raise ValueError("labels, scores, and clip_ids must have matching lengths")

    frame = pd.DataFrame(
        {
            "clip_id": clip_ids,
            "label": np.asarray(y_true, dtype=int),
            "score": np.asarray(y_score, dtype=float),
        }
    )
    label_counts = frame.groupby("clip_id", sort=False)["label"].nunique()
    mixed = label_counts[label_counts > 1]
    if not mixed.empty:
        raise ValueError(
            f"Found mixed labels for clip_ids: {', '.join(mixed.index.astype(str))}"
        )

    score_agg = "mean" if method == "mean" else "max"
    return (
        frame.groupby("clip_id", sort=False)
        .agg(label=("label", "first"), score=("score", score_agg), windows=("score", "size"))
        .reset_index()
    )
