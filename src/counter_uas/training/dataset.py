from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from counter_uas.config import FeatureConfig
from counter_uas.data.audio import load_wav_mono, segment_waveform
from counter_uas.features.mel import LogMelSpectrogram


LABEL_TO_INDEX = {"no_drone": 0, "drone": 1}


class AudioWindowDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        manifest_path: Path | str,
        root_dir: Path | str,
        split: str,
        sample_rate: int,
        window_seconds: float,
        hop_seconds: float,
        feature_config: FeatureConfig,
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest = pd.read_csv(manifest_path)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r}")
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.hop_samples = int(sample_rate * hop_seconds)
        self.transform = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=feature_config.n_mels,
            n_fft=feature_config.n_fft,
            win_length=feature_config.win_length,
            hop_length=feature_config.hop_length,
        )
        self.index: list[tuple[int, int]] = []
        for row_index, row in self.rows.iterrows():
            duration = float(row.get("duration", window_seconds))
            n_samples = max(1, int(duration * sample_rate))
            if n_samples <= self.window_samples:
                n_windows = 1
            else:
                n_windows = 1 + max(
                    0, (n_samples - self.window_samples) // self.hop_samples
                )
            self.index.extend(
                (row_index, window_index) for window_index in range(n_windows)
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row_index, window_index = self.index[index]
        row = self.rows.iloc[row_index]
        waveform, actual_sample_rate = load_wav_mono(self.root_dir / str(row["path"]))
        if actual_sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.sample_rate}, got {actual_sample_rate}"
            )
        windows = segment_waveform(waveform, self.window_samples, self.hop_samples)
        selected = windows[min(window_index, len(windows) - 1)]
        tensor = torch.from_numpy(selected)
        features = self.transform(tensor)
        label = LABEL_TO_INDEX[str(row["label"])]
        return features, label, str(row["clip_id"])
