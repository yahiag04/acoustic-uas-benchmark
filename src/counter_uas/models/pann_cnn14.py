from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torchaudio


PANN_CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
PANN_SAMPLE_RATE = 32_000


def _load_panns_cnn14() -> nn.Module:
    """Load a fresh Cnn14 instance with AudioSet-pretrained weights."""
    from panns_inference.models import Cnn14  # heavy import deferred

    model = Cnn14(
        sample_rate=PANN_SAMPLE_RATE,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    if not PANN_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"PANNs CNN14 checkpoint not found at {PANN_CHECKPOINT_PATH}. "
            "Download with: curl -L -o ~/panns_data/Cnn14_mAP=0.431.pth "
            "'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'"
        )
    state = torch.load(PANN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model


class PannCnn14Classifier(nn.Module):
    """PANNs CNN14 (AudioSet-pretrained) + small binary classification head.

    Input: raw mono waveform at 16 kHz (any length).
    Internally upsamples 16 kHz → 32 kHz (PANN native rate) and runs Cnn14 to
    obtain a 2048-d embedding, which is fed into a 2-layer MLP.
    """

    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = _load_panns_cnn14()
        # Resample 16 kHz → 32 kHz on the fly (kaiser_window default lowpass).
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=16_000, new_freq=PANN_SAMPLE_RATE
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(
                f"Expected (batch, samples) waveform, got shape {tuple(waveform.shape)}"
            )
        x = self.resampler(waveform)
        out = self.backbone(x, None)
        embedding = out["embedding"]
        return self.head(embedding).squeeze(-1)
