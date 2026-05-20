from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from counter_uas.reporting.markdown import dataframe_to_markdown


def write_final_report(
    report_path: Path | str,
    manifest_path: Path | str,
    eval_dir: Path | str,
    robustness_dir: Path | str,
) -> Path:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    eval_path = Path(eval_dir)
    robustness_path = Path(robustness_dir) / "robustness_report.csv"
    metrics = json.loads((eval_path / "metrics.json").read_text())
    split_summary = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    robustness = pd.read_csv(robustness_path) if robustness_path.exists() else pd.DataFrame()

    lines = [
        "# Acoustic Counter-UAS Detection Benchmark",
        "",
        "## Dataset Summary",
        "",
        f"- Clips: {len(manifest)}",
        f"- Labels: {', '.join(sorted(manifest['label'].unique()))}",
        "",
        dataframe_to_markdown(split_summary, include_index=True),
        "",
        "## Main Test Metrics",
        "",
        dataframe_to_markdown(pd.DataFrame([metrics])),
        "",
        "## Robustness",
        "",
        dataframe_to_markdown(robustness) if not robustness.empty else "No robustness report found.",
        "",
        "## Known Limitations",
        "",
        "- The first version uses public acoustic data and does not claim operational deployment readiness.",
        "- Evaluation is binary drone/no-drone detection, not localization or drone-type identification.",
        "- Robustness conditions are controlled perturbations, not a substitute for field validation.",
    ]
    report.write_text("\n".join(lines) + "\n")
    return report
