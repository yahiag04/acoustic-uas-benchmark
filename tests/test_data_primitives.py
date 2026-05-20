import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from counter_uas.data.audio import load_wav_mono, segment_waveform, to_mono
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


def test_assign_splits_supports_small_balanced_manifests():
    for samples_per_class in (4, 5):
        manifest = pd.DataFrame(
            {
                "clip_id": [f"drone_{i}" for i in range(samples_per_class)]
                + [f"no_drone_{i}" for i in range(samples_per_class)],
                "label": ["drone"] * samples_per_class
                + ["no_drone"] * samples_per_class,
                "path": [f"audio/{i}.wav" for i in range(samples_per_class * 2)],
            }
        )

        assigned = assign_splits(manifest, seed=123)

        assert len(assigned) == len(manifest)
        assert set(assigned["clip_id"]) == set(manifest["clip_id"])
        assert set(assigned["split"]) == {"train", "val", "test"}
        assert set(assigned.groupby("split")["label"].nunique()) == {2}
        assert (
            assigned["split"].tolist()
            == assign_splits(manifest, seed=123)["split"].tolist()
        )


def test_assign_splits_rejects_null_required_fields():
    manifest = pd.DataFrame(
        {
            "clip_id": ["c0", None, "c2"],
            "label": ["drone", None, "no_drone"],
            "path": ["audio/0.wav", "audio/1.wav", None],
        }
    )

    with pytest.raises(
        ValueError,
        match="Manifest required columns contain nulls: \\['clip_id', 'label', 'path'\\]",
    ):
        assign_splits(manifest, seed=123)


def test_load_wav_mono_round_trips_temp_wav(tmp_path):
    path = tmp_path / "stereo.wav"
    audio = np.array([[0.25, 0.75], [0.5, -0.5]], dtype=np.float32)
    sf.write(path, audio, 16_000, subtype="FLOAT")

    waveform, sample_rate = load_wav_mono(path)

    assert sample_rate == 16_000
    assert np.allclose(waveform, np.array([0.5, 0.0], dtype=np.float32))
