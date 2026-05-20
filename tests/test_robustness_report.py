from pathlib import Path

import numpy as np

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.reporting.report import write_final_report
from counter_uas.robustness.perturbations import add_white_noise_snr, apply_gain_db
from counter_uas.robustness.run import run_robustness
from counter_uas.training.train import train_from_config


def test_audio_perturbations_preserve_shape():
    waveform = np.ones(16000, dtype=np.float32)

    noisy = add_white_noise_snr(waveform, snr_db=10, seed=5)
    gained = apply_gain_db(waveform, gain_db=-6)

    assert noisy.shape == waveform.shape
    assert gained.shape == waveform.shape
    assert np.max(np.abs(gained)) < 1.0


def test_robustness_and_report_write_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    manifest_path = create_synthetic_dataset(data_dir, samples_per_class=4, seed=17)
    config = load_config(Path("configs/baseline_cnn.yaml"))
    checkpoint = train_from_config(config, manifest_path, data_dir, tmp_path / "train", max_epochs=1)
    eval_dir = tmp_path / "eval"
    evaluate_checkpoint(config, checkpoint, manifest_path, data_dir, eval_dir)

    robustness_dir = tmp_path / "robustness"
    run_robustness(config, checkpoint, manifest_path, data_dir, robustness_dir)
    report_path = write_final_report(
        report_path=tmp_path / "final_report.md",
        manifest_path=manifest_path,
        eval_dir=eval_dir,
        robustness_dir=robustness_dir,
    )

    assert (robustness_dir / "robustness_report.csv").exists()
    assert (robustness_dir / "robustness_summary.md").exists()
    assert report_path.exists()
    assert "Acoustic Counter-UAS Detection Benchmark" in report_path.read_text()
