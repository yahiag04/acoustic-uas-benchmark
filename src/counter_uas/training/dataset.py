from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from counter_uas.config import FeatureConfig
from counter_uas.data.audio import load_wav_mono_window
from counter_uas.features.mel import LogMelSpectrogram


LABEL_TO_INDEX = {"no_drone": 0, "drone": 1}

Perturbation = Callable[[np.ndarray, int], np.ndarray]


def _window_starts(n_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    if n_samples <= window_samples:
        return [0]
    starts = list(range(0, n_samples - window_samples + 1, hop_samples))
    if starts[-1] != n_samples - window_samples:
        starts.append(n_samples - window_samples)
    return starts


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
        perturbation: Perturbation | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest = pd.read_csv(manifest_path)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r}")
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.hop_samples = int(sample_rate * hop_seconds)
        if self.window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if self.hop_samples <= 0:
            raise ValueError("hop_samples must be positive")
        self.perturbation = perturbation
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
            self.index.extend(
                (row_index, start_sample)
                for start_sample in _window_starts(
                    n_samples, self.window_samples, self.hop_samples
                )
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row_index, start_sample = self.index[index]
        row = self.rows.iloc[row_index]
        selected, actual_sample_rate = load_wav_mono_window(
            self.root_dir / str(row["path"]),
            start_sample=start_sample,
            window_samples=self.window_samples,
        )
        if actual_sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.sample_rate}, got {actual_sample_rate}"
            )
        if self.perturbation is not None:
            selected = np.asarray(
                self.perturbation(selected, self.sample_rate), dtype=np.float32
            )
        tensor = torch.from_numpy(selected)
        features = self.transform(tensor)
        label = LABEL_TO_INDEX[str(row["label"])]
        return features, label, str(row["clip_id"])
