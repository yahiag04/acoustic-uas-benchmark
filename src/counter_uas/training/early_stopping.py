from __future__ import annotations

import math
from typing import Literal


class EarlyStopper:
    def __init__(self, patience: int, mode: Literal["max", "min"] = "max") -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.mode = mode
        self._best = -math.inf if mode == "max" else math.inf
        self._epochs_since_improvement = 0

    def update(self, metric: float) -> bool:
        improved = metric > self._best if self.mode == "max" else metric < self._best
        if improved:
            self._best = float(metric)
            self._epochs_since_improvement = 0
            return False
        self._epochs_since_improvement += 1
        return self._epochs_since_improvement > self.patience

    @property
    def best(self) -> float:
        return self._best
