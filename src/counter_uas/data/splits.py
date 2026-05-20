from __future__ import annotations

import math

import numpy as np
import pandas as pd


_SPLITS = ("train", "val", "test")


def _allocate_counts(
    sample_count: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> dict[str, int]:
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    if any(size < 0 for size in sizes.values()):
        raise ValueError("split sizes must be non-negative")

    positive_splits = [split for split in _SPLITS if sizes[split] > 0]
    if not positive_splits:
        raise ValueError("at least one split size must be positive")
    if sample_count <= 0:
        return dict.fromkeys(_SPLITS, 0)
    if sample_count < len(positive_splits):
        counts = dict.fromkeys(_SPLITS, 0)
        ordered = sorted(
            positive_splits,
            key=lambda split: (-sizes[split], _SPLITS.index(split)),
        )
        for split in ordered[:sample_count]:
            counts[split] = 1
        return counts

    raw_counts = {split: sample_count * sizes[split] for split in _SPLITS}
    counts = {split: math.floor(raw_counts[split]) for split in _SPLITS}
    remaining = sample_count - sum(counts.values())
    ordered = sorted(
        _SPLITS,
        key=lambda split: (-(raw_counts[split] - counts[split]), _SPLITS.index(split)),
    )
    for split in ordered[:remaining]:
        counts[split] += 1

    for split in positive_splits:
        if counts[split] > 0:
            continue
        donor = max(
            (candidate for candidate in positive_splits if counts[candidate] > 1),
            key=lambda candidate: (counts[candidate], sizes[candidate]),
        )
        counts[donor] -= 1
        counts[split] = 1
    return counts


def assign_splits(
    manifest: pd.DataFrame,
    seed: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    required = {"clip_id", "label", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    null_columns = sorted(
        column for column in required if manifest[column].isna().any()
    )
    if null_columns:
        raise ValueError(
            f"Manifest required columns contain nulls: {null_columns}"
        )

    df = manifest.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    split_frames: dict[str, list[pd.DataFrame]] = {split: [] for split in _SPLITS}

    for _, label_df in df.groupby("label", sort=True):
        label_df = label_df.sort_values("clip_id")
        shuffled_positions = rng.permutation(len(label_df))
        shuffled = label_df.iloc[shuffled_positions].reset_index(drop=True)
        counts = _allocate_counts(len(shuffled), train_size, val_size, test_size)

        start = 0
        for split in _SPLITS:
            count = counts[split]
            if count:
                split_frames[split].append(
                    shuffled.iloc[start : start + count].assign(split=split)
                )
            start += count

    frames = [
        frame
        for split in _SPLITS
        for frame in split_frames[split]
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("clip_id")
        .reset_index(drop=True)
    )
