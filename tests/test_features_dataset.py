from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from counter_uas.config import load_config
from counter_uas.data.audio import segment_waveform
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


def test_audio_window_dataset_counts_final_end_aligned_window(tmp_path):
    config = load_config(Path("configs/baseline_cnn.yaml"))
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "tail_window.wav"
    waveform = np.zeros(
        int(config.data.sample_rate * (config.data.window_seconds + 0.5)),
        dtype=np.float32,
    )
    sf.write(audio_path, waveform, config.data.sample_rate)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "clip_id": "tail_window",
                "path": "audio/tail_window.wav",
                "label": "drone",
                "duration": len(waveform) / config.data.sample_rate,
                "sample_rate": config.data.sample_rate,
                "source_id": "tail_window",
                "split": "train",
            }
        ]
    ).to_csv(manifest_path, index=False)

    dataset = AudioWindowDataset(
        manifest_path=manifest_path,
        root_dir=tmp_path,
        split="train",
        sample_rate=config.data.sample_rate,
        window_seconds=config.data.window_seconds,
        hop_seconds=config.data.hop_seconds,
        feature_config=config.features,
    )

    expected_windows = segment_waveform(
        waveform,
        int(config.data.sample_rate * config.data.window_seconds),
        int(config.data.sample_rate * config.data.hop_seconds),
    )
    assert len(dataset) == len(expected_windows)
