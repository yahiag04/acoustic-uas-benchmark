from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from counter_uas.data.splits import assign_splits


def _split_counts_markdown(manifest: pd.DataFrame) -> str:
    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    table = counts.reset_index()
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _sine_wave(
    freq: float,
    sample_rate: int,
    seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * freq * t)
    noise = 0.03 * rng.normal(size=signal.shape)
    return (signal + noise).astype(np.float32)


def create_synthetic_dataset(
    output_dir: Path | str,
    samples_per_class: int = 12,
    sample_rate: int = 16_000,
    seconds: float = 2.0,
    seed: int = 42,
) -> Path:
    if samples_per_class < 4:
        raise ValueError("samples_per_class must be at least 4")

    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for label, base_freq in [("drone", 220.0), ("no_drone", 880.0)]:
        for index in range(samples_per_class):
            clip_id = f"{label}_{index:04d}"
            freq = base_freq + float(rng.uniform(-15.0, 15.0))
            waveform = _sine_wave(freq, sample_rate, seconds, rng)
            rel_path = Path("audio") / f"{clip_id}.wav"
            sf.write(audio_dir / f"{clip_id}.wav", waveform, sample_rate)
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": rel_path.as_posix(),
                    "label": label,
                    "duration": seconds,
                    "sample_rate": sample_rate,
                    "source_id": clip_id,
                }
            )

    manifest = assign_splits(pd.DataFrame(rows), seed=seed)
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (root / "split_report.md").write_text(
        "# Split Report\n\n" + _split_counts_markdown(manifest) + "\n"
    )
    return manifest_path
