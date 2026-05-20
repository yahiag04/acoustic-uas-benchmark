from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from counter_uas.data.labels import normalize_label
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


def export_dads_dataset(
    output_dir: Path | str,
    dataset_name: str,
    seed: int,
    max_rows: int | None = None,
) -> Path:
    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=True))
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for index, item in enumerate(tqdm(dataset, desc="Exporting DADS")):
        try:
            label = normalize_label(item["label"])
            audio = item["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            clip_id = f"dads_{index:08d}"
            rel_path = Path("audio") / f"{clip_id}.wav"
            sf.write(audio_dir / rel_path.name, waveform, sample_rate)
            duration = float(len(waveform) / sample_rate)
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": rel_path.as_posix(),
                    "label": label,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "source_id": Path(str(item.get("file", clip_id))).stem,
                }
            )
        except Exception as exc:
            skipped.append(f"{index},{type(exc).__name__},{exc}")

    if not rows:
        raise RuntimeError("No DADS rows were exported")

    manifest = assign_splits(pd.DataFrame(rows), seed=seed)
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (root / "skipped_files.csv").write_text(
        "index,error_type,error\n"
        + "\n".join(skipped)
        + ("\n" if skipped else "")
    )
    (root / "split_report.md").write_text(
        "# Split Report\n\n" + _split_counts_markdown(manifest) + "\n"
    )
    return manifest_path
