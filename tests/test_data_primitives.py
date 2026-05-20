import numpy as np
import pandas as pd

from counter_uas.data.audio import segment_waveform, to_mono
from counter_uas.data.labels import normalize_label
from counter_uas.data.splits import assign_splits


def test_normalize_label_handles_known_dads_values():
    assert normalize_label(0) == "no_drone"
    assert normalize_label("0") == "no_drone"
    assert normalize_label(1) == "drone"
    assert normalize_label("drone") == "drone"


def test_to_mono_averages_channels():
    audio = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    mono = to_mono(audio)
    assert np.allclose(mono, np.array([2.0, 6.0], dtype=np.float32))


def test_segment_waveform_pads_short_clip():
    waveform = np.ones(8, dtype=np.float32)
    windows = segment_waveform(waveform, window_samples=16, hop_samples=8)

    assert windows.shape == (1, 16)
    assert np.allclose(windows[0, :8], 1.0)
    assert np.allclose(windows[0, 8:], 0.0)


def test_segment_waveform_segments_long_clip():
    waveform = np.arange(20, dtype=np.float32)
    windows = segment_waveform(waveform, window_samples=8, hop_samples=4)

    assert windows.shape == (4, 8)
    assert np.allclose(windows[1], np.arange(4, 12, dtype=np.float32))


def test_assign_splits_is_deterministic_and_stratified():
    manifest = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(20)],
            "label": ["drone"] * 10 + ["no_drone"] * 10,
            "path": [f"audio/{i}.wav" for i in range(20)],
        }
    )

    first = assign_splits(manifest, seed=123)
    second = assign_splits(manifest, seed=123)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}
    assert set(first.groupby("split")["label"].nunique()) == {2}
