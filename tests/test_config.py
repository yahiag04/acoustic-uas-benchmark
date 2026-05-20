from pathlib import Path

from counter_uas.config import load_config


def test_load_config_exposes_core_sections():
    config = load_config(Path("configs/baseline_cnn.yaml"))

    assert config.seed == 42
    assert config.data.sample_rate == 16_000
    assert config.features.n_mels == 64
    assert config.training.batch_size == 16
    assert config.model.name == "baseline_cnn"
