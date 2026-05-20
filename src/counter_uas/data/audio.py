from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1).astype(np.float32)
    raise ValueError(f"Expected 1D or 2D audio array, got shape {array.shape}")


def segment_waveform(
    waveform: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {audio.shape}")
    if len(audio) <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :]

    starts = list(range(0, len(audio) - window_samples + 1, hop_samples))
    if starts[-1] != len(audio) - window_samples:
        starts.append(len(audio) - window_samples)
    return np.stack([audio[start : start + window_samples] for start in starts])


def load_wav_mono(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(Path(path), always_2d=False)
    return to_mono(audio), int(sample_rate)


def load_wav_mono_window(
    path: Path | str,
    start_sample: int,
    window_samples: int,
) -> tuple[np.ndarray, int]:
    if start_sample < 0:
        raise ValueError("start_sample must be non-negative")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    with sf.SoundFile(Path(path)) as audio_file:
        audio_file.seek(start_sample)
        audio = audio_file.read(window_samples, always_2d=False, dtype="float32")
        sample_rate = int(audio_file.samplerate)

    waveform = to_mono(audio)
    if len(waveform) < window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: len(waveform)] = waveform
        waveform = padded
    return waveform, sample_rate
