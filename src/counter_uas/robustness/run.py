from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from counter_uas.config import ExperimentConfig
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.reporting.markdown import dataframe_to_markdown
from counter_uas.robustness.perturbations import add_white_noise_snr, apply_gain_db


def _noise_perturbation(snr_db: float, seed: int):
    counter = [0]

    def perturb(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        counter[0] += 1
        return add_white_noise_snr(waveform, snr_db=snr_db, seed=seed + counter[0])

    return perturb


def _gain_perturbation(gain_db: float):
    def perturb(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return apply_gain_db(waveform, gain_db=gain_db)

    return perturb


def run_robustness(
    config: ExperimentConfig,
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    root_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    clean_metrics = evaluate_checkpoint(
        config, checkpoint_path, manifest_path, root_dir, output / "clean"
    )
    rows.append({"condition": "clean", **clean_metrics})

    for snr in config.robustness.snr_db:
        metrics = evaluate_checkpoint(
            config,
            checkpoint_path,
            manifest_path,
            root_dir,
            output / f"snr_{snr:g}db",
            test_perturbation=_noise_perturbation(snr_db=float(snr), seed=config.seed),
        )
        rows.append({"condition": f"snr_{snr:g}db", **metrics})

    for gain in config.robustness.gain_db:
        metrics = evaluate_checkpoint(
            config,
            checkpoint_path,
            manifest_path,
            root_dir,
            output / f"gain_{gain:g}db",
            test_perturbation=_gain_perturbation(gain_db=float(gain)),
        )
        rows.append({"condition": f"gain_{gain:g}db", **metrics})

    report = pd.DataFrame(rows)
    report_path = output / "robustness_report.csv"
    report.to_csv(report_path, index=False)
    summary_cols = [
        "condition",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
    ]
    (output / "robustness_summary.md").write_text(
        "# Robustness Summary\n\n"
        + dataframe_to_markdown(report[summary_cols])
        + "\n"
    )
    return report_path
