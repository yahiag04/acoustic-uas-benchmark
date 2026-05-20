import json
import os
from pathlib import Path

import pandas as pd

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.training.train import train_from_config
import matplotlib


def test_evaluation_uses_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"
    assert Path(os.environ["MPLCONFIGDIR"]).exists()


def test_evaluate_checkpoint_writes_metrics_and_figures(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = create_synthetic_dataset(data_dir, samples_per_class=4, seed=13)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    checkpoint = train_from_config(config, manifest_path, data_dir, tmp_path / "train", max_epochs=1)

    output_dir = tmp_path / "eval"
    metrics = evaluate_checkpoint(config, checkpoint, manifest_path, data_dir, output_dir)

    assert "pr_auc" in metrics
    assert "window_pr_auc" in metrics
    assert "latency_ms_per_window" in metrics
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "classification_report.txt").exists()
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "pr_curve.png").exists()
    assert (output_dir / "prediction_samples.csv").exists()
    assert (output_dir / "clip_predictions.csv").exists()
    assert (output_dir / "error_analysis.csv").exists()

    saved_metrics = json.loads((output_dir / "metrics.json").read_text())
    clip_predictions = pd.read_csv(output_dir / "clip_predictions.csv")
    error_analysis = pd.read_csv(output_dir / "error_analysis.csv")

    assert saved_metrics["evaluation_level"] == "clip"
    assert saved_metrics["clip_aggregation"] == config.evaluation.clip_aggregation
    assert len(clip_predictions) <= len(pd.read_csv(output_dir / "prediction_samples.csv"))
    assert {"clip_id", "label", "score", "prediction", "windows"}.issubset(
        clip_predictions.columns
    )
    assert {"clip_id", "label", "score", "prediction", "error_type"}.issubset(
        error_analysis.columns
    )
