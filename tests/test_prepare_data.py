from pathlib import Path

import pandas as pd

from counter_uas.cli import main
from counter_uas.data.synthetic import create_synthetic_dataset


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
