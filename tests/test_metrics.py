import numpy as np

from counter_uas.evaluation.metrics import (
    aggregate_clip_predictions,
    compute_binary_metrics,
    select_threshold,
)


def test_select_threshold_maximizes_f1():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(y_true, y_score, objective="f1")

    assert 0.4 < threshold <= 0.6


def test_select_threshold_can_minimize_fpr_at_target_recall():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.6, 0.5, 0.9])

    threshold = select_threshold(
        y_true,
        y_score,
        objective="target_recall_min_fpr",
        target_recall=0.5,
    )

    assert threshold == 0.9


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


def test_aggregate_clip_predictions_supports_mean_and_max_scores():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.7, 0.4, 0.8, 0.9])
    clip_ids = ["bg", "bg", "drone", "drone", "single"]

    mean_predictions = aggregate_clip_predictions(
        y_true, y_score, clip_ids, method="mean"
    )
    max_predictions = aggregate_clip_predictions(
        y_true, y_score, clip_ids, method="max"
    )

    assert list(mean_predictions.columns) == ["clip_id", "label", "score", "windows"]
    assert mean_predictions[["clip_id", "label", "windows"]].to_dict("records") == [
        {"clip_id": "bg", "label": 0, "windows": 2},
        {"clip_id": "drone", "label": 1, "windows": 2},
        {"clip_id": "single", "label": 1, "windows": 1},
    ]
    np.testing.assert_allclose(mean_predictions["score"], [0.4, 0.6, 0.9])
    assert max_predictions.loc[max_predictions["clip_id"] == "bg", "score"].item() == 0.7


def test_aggregate_clip_predictions_rejects_mixed_labels_for_same_clip():
    with np.testing.assert_raises_regex(ValueError, "mixed labels"):
        aggregate_clip_predictions(
            np.array([0, 1]),
            np.array([0.1, 0.2]),
            ["same", "same"],
            method="mean",
        )
