import numpy as np
import pandas as pd
import soundfile as sf

from counter_uas.cli import main
from counter_uas.data.dads import export_dads_dataset
from counter_uas.data.synthetic import create_synthetic_dataset


class _FakeDadsDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def cast_column(self, name, feature):
        assert name == "audio"
        return self

    def select(self, indices):
        return _FakeDadsDataset([self._rows[index] for index in indices])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _ExplodingLabel:
    def __str__(self):
        raise ValueError("bad label, with comma\nand newline")


def test_create_synthetic_dataset_writes_manifest_and_audio(tmp_path):
    manifest_path = create_synthetic_dataset(tmp_path, samples_per_class=4, seed=7)

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 8
    assert set(manifest["label"]) == {"drone", "no_drone"}
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert all((tmp_path / path).exists() for path in manifest["path"])


def test_cli_prepare_synthetic(tmp_path):
    exit_code = main(["prepare-synthetic", "--output-dir", str(tmp_path), "--samples-per-class", "4"])

    assert exit_code == 0
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "split_report.md").exists()


def test_export_dads_dataset_writes_manifest_audio_and_safe_skipped_csv(tmp_path, monkeypatch):
    waveform = np.linspace(-0.25, 0.25, 160, dtype=np.float32)
    rows = []
    for index in range(4):
        rows.append(
            {
                "label": 1,
                "audio": {"array": waveform, "sampling_rate": 16_000},
                "file": f"drone/source_{index}.wav",
            }
        )
        rows.append(
            {
                "label": "background",
                "audio": {"array": waveform, "sampling_rate": 16_000},
                "file": f"background/source_{index}.wav",
            }
        )
    rows.append(
        {
            "label": _ExplodingLabel(),
            "audio": {"array": waveform, "sampling_rate": 16_000},
            "file": "bad/source.wav",
        }
    )

    def fake_load_dataset(dataset_name, split):
        assert dataset_name == "fake/dads"
        assert split == "train"
        return _FakeDadsDataset(rows)

    monkeypatch.setattr("counter_uas.data.dads.load_dataset", fake_load_dataset)

    manifest_path = export_dads_dataset(tmp_path, "fake/dads", seed=11)

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 8
    assert set(manifest["label"]) == {"drone", "no_drone"}
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert {
        "clip_id",
        "path",
        "label",
        "duration",
        "sample_rate",
        "source_id",
        "split",
    }.issubset(manifest.columns)
    assert all((tmp_path / path).exists() for path in manifest["path"])

    first_audio = tmp_path / manifest.iloc[0]["path"]
    audio_info = sf.info(first_audio)
    assert audio_info.samplerate == 16_000
    assert audio_info.frames == len(waveform)
    assert (tmp_path / "split_report.md").exists()

    skipped = pd.read_csv(tmp_path / "skipped_files.csv")
    assert skipped.to_dict("records") == [
        {
            "index": 8,
            "error_type": "ValueError",
            "error": "bad label, with comma\nand newline",
        }
    ]
