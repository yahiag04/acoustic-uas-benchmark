import numpy as np

from counter_uas.evaluation.metrics import compute_binary_metrics, select_threshold


def test_select_threshold_maximizes_f1():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(y_true, y_score, objective="f1")

    assert 0.4 < threshold <= 0.6


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
