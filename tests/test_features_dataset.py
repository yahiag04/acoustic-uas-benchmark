from pathlib import Path

import torch

from counter_uas.config import load_config
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.features.mel import LogMelSpectrogram
from counter_uas.training.dataset import AudioWindowDataset


def test_log_mel_returns_channel_first_tensor():
    transform = LogMelSpectrogram(
        sample_rate=16_000, n_mels=64, n_fft=1024, win_length=400, hop_length=160
    )
    waveform = torch.zeros(16_000)

    features = transform(waveform)

    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[1] == 64


def test_audio_window_dataset_reads_synthetic_manifest(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path, samples_per_class=4, seed=9)
    config = load_config(Path("configs/baseline_cnn.yaml"))

    dataset = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=tmp_path,
        split="train",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
    )

    features, label, clip_id = dataset[0]
    assert features.shape[0] == 1
    assert features.shape[1] == 64
    assert label in {0, 1}
    assert isinstance(clip_id, str)
