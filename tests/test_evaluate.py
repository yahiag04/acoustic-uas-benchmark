from pathlib import Path

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.training.train import train_from_config


def test_evaluate_checkpoint_writes_metrics_and_figures(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = create_synthetic_dataset(data_dir, samples_per_class=4, seed=13)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    checkpoint = train_from_config(config, manifest_path, data_dir, tmp_path / "train", max_epochs=1)

    output_dir = tmp_path / "eval"
    metrics = evaluate_checkpoint(config, checkpoint, manifest_path, data_dir, output_dir)

    assert "pr_auc" in metrics
    assert "latency_ms_per_window" in metrics
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "classification_report.txt").exists()
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "pr_curve.png").exists()
    assert (output_dir / "prediction_samples.csv").exists()
