from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from counter_uas.reporting.markdown import dataframe_to_markdown


_METADATA_KEYS = {
    "evaluation_level",
    "clip_aggregation",
    "threshold_objective",
    "clip_count",
    "window_count",
}


def _clip_metric_table(metrics: dict[str, object]) -> pd.DataFrame:
    values = {
        key: value
        for key, value in metrics.items()
        if not key.startswith("window_")
        and key not in _METADATA_KEYS
        and key != "latency_ms_per_window"
    }
    return pd.DataFrame([values])


def _window_metric_table(metrics: dict[str, object]) -> pd.DataFrame:
    values = {
        key.removeprefix("window_"): value
        for key, value in metrics.items()
        if key.startswith("window_")
    }
    values["latency_ms_per_window"] = metrics.get("latency_ms_per_window", "nan")
    return pd.DataFrame([values])


def _robustness_gap_table(robustness: pd.DataFrame) -> pd.DataFrame:
    if robustness.empty or "condition" not in robustness:
        return pd.DataFrame()
    clean = robustness[robustness["condition"] == "clean"]
    if clean.empty:
        return pd.DataFrame()
    clean_row = clean.iloc[0]
    rows = []
    for row in robustness[robustness["condition"] != "clean"].to_dict("records"):
        rows.append(
            {
                "condition": row["condition"],
                "recall_delta": float(row.get("recall", 0.0))
                - float(clean_row.get("recall", 0.0)),
                "f1_delta": float(row.get("f1", 0.0))
                - float(clean_row.get("f1", 0.0)),
                "false_positive_rate_delta": float(
                    row.get("false_positive_rate", 0.0)
                )
                - float(clean_row.get("false_positive_rate", 0.0)),
            }
        )
    return pd.DataFrame(rows)


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
    robustness_gap = _robustness_gap_table(robustness)

    lines = [
        "# Acoustic Counter-UAS Detection Benchmark",
        "",
        "## Artifact Metadata",
        "",
        f"- Manifest: `{Path(manifest_path)}`",
        f"- Evaluation directory: `{eval_path}`",
        f"- Robustness directory: `{Path(robustness_dir)}`",
        f"- Evaluation level: {metrics.get('evaluation_level', 'window')}",
        f"- Clip aggregation: {metrics.get('clip_aggregation', 'not recorded')}",
        f"- Threshold objective: {metrics.get('threshold_objective', 'not recorded')}",
        f"- Clip predictions: `{eval_path / 'clip_predictions.csv'}`",
        f"- Window predictions: `{eval_path / 'prediction_samples.csv'}`",
        f"- Error analysis: `{eval_path / 'error_analysis.csv'}`",
        "",
        "## Dataset Summary",
        "",
        f"- Clips: {len(manifest)}",
        f"- Labels: {', '.join(sorted(manifest['label'].unique()))}",
        "",
        dataframe_to_markdown(split_summary, include_index=True),
        "",
        "## Clip-Level Test Metrics",
        "",
        dataframe_to_markdown(_clip_metric_table(metrics)),
        "",
        "## Window-Level Test Metrics",
        "",
        dataframe_to_markdown(_window_metric_table(metrics)),
        "",
        "## Robustness",
        "",
        dataframe_to_markdown(robustness) if not robustness.empty else "No robustness report found.",
        "",
        "## Robustness Gap",
        "",
        dataframe_to_markdown(robustness_gap)
        if not robustness_gap.empty
        else "No robustness gap could be computed.",
        "",
        "## Known Limitations",
        "",
        "- The first version uses public acoustic data and does not claim operational deployment readiness.",
        "- Clean split results may be optimistic when source metadata does not expose true recording groups.",
        "- Reports include both clip-level and window-level metrics because repeated windows from one clip are not independent samples.",
        "- Evaluation is binary drone/no-drone detection, not localization or drone-type identification.",
        "- Robustness conditions are controlled perturbations, not a substitute for field validation.",
    ]
    report.write_text("\n".join(lines) + "\n")
    return report
