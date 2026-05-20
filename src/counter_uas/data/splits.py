from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


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

    df = manifest.copy().reset_index(drop=True)
    train_df, remaining = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        remaining,
        test_size=relative_test_size,
        random_state=seed,
        stratify=remaining["label"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values("clip_id")
        .reset_index(drop=True)
    )
