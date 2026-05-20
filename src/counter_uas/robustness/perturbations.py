from __future__ import annotations

import numpy as np


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20.0)
    return np.asarray(waveform, dtype=np.float32) * np.float32(gain)


def add_white_noise_snr(waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(audio**2))
    if signal_power == 0.0:
        signal_power = 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise
