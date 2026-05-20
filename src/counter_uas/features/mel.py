from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected mono waveform tensor, got shape {tuple(waveform.shape)}"
            )
        mel = self.mel(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.unsqueeze(0).float()
