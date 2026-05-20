from pathlib import Path

from counter_uas.config import load_config


def test_load_config_exposes_core_sections():
    config = load_config(Path("configs/baseline_cnn.yaml"))

    assert config.seed == 42
    assert config.data.sample_rate == 16_000
    assert config.features.n_mels == 64
    assert config.training.batch_size == 16
    assert config.model.name == "baseline_cnn"
    assert config.model.dropout == 0.2
    assert config.training.augmentation.enabled is True
    assert config.training.augmentation.noise_snr_db == [30.0, 20.0, 10.0]
    assert config.training.augmentation.gain_db == [-6.0, 0.0, 6.0]
    assert config.training.augmentation.time_mask_fraction == 0.1
    assert config.training.balanced_sampling is True
    assert config.evaluation.clip_aggregation == "mean"
    assert config.evaluation.target_recall == 0.985
    assert config.evaluation.threshold_objective == "target_recall_min_fpr"
