from __future__ import annotations

from torch import nn

from counter_uas.config import ModelConfig
from counter_uas.models.baseline_cnn import BaselineCNN
from counter_uas.models.enhanced_cnn import EnhancedCNN


def create_model(config: ModelConfig) -> nn.Module:
    if config.name == "baseline_cnn":
        return BaselineCNN(config.channels)
    if config.name == "enhanced_cnn":
        return EnhancedCNN(config.channels, dropout=config.dropout)
    raise ValueError(f"Unsupported model name: {config.name}")
