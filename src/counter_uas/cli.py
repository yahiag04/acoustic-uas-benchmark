from __future__ import annotations

import argparse
from pathlib import Path

from counter_uas import __version__
from counter_uas.config import load_config
from counter_uas.data.dads import export_dads_dataset
from counter_uas.data.synthetic import create_synthetic_dataset
from counter_uas.evaluation.evaluate import evaluate_checkpoint
from counter_uas.reporting.report import write_final_report
from counter_uas.robustness.run import run_robustness
from counter_uas.training.train import train_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="counter-uas")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    synthetic = subparsers.add_parser("prepare-synthetic")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--samples-per-class", type=int, default=12)
    synthetic.add_argument("--seed", type=int, default=42)

    dads = subparsers.add_parser("prepare-dads")
    dads.add_argument("--config", default="configs/baseline_cnn.yaml")
    dads.add_argument("--output-dir", default="data/processed")
    dads.add_argument("--max-rows", type=int)

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/baseline_cnn.yaml")
    train.add_argument("--manifest", required=True)
    train.add_argument("--root-dir", required=True)
    train.add_argument("--artifact-dir", default="artifacts/baseline_cnn")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/baseline_cnn.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--root-dir", required=True)
    evaluate.add_argument("--output-dir", default="artifacts/baseline_cnn/eval")

    robustness = subparsers.add_parser("robustness")
    robustness.add_argument("--config", default="configs/baseline_cnn.yaml")
    robustness.add_argument("--checkpoint", required=True)
    robustness.add_argument("--manifest", required=True)
    robustness.add_argument("--root-dir", required=True)
    robustness.add_argument("--output-dir", default="artifacts/baseline_cnn/robustness")

    report = subparsers.add_parser("report")
    report.add_argument("--report-path", default="reports/final_report.md")
    report.add_argument("--manifest", required=True)
    report.add_argument("--eval-dir", required=True)
    report.add_argument("--robustness-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-synthetic":
        create_synthetic_dataset(
            Path(args.output_dir),
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        return 0
    if args.command == "prepare-dads":
        config = load_config(Path(args.config))
        export_dads_dataset(
            output_dir=Path(args.output_dir),
            dataset_name=config.data.dataset_name,
            seed=config.seed,
            max_rows=args.max_rows,
        )
        return 0
    if args.command == "train":
        config = load_config(Path(args.config))
        train_from_config(
            config=config,
            manifest_path=Path(args.manifest),
            root_dir=Path(args.root_dir),
            artifact_dir=Path(args.artifact_dir),
        )
        return 0
    if args.command == "evaluate":
        config = load_config(Path(args.config))
        evaluate_checkpoint(
            config,
            Path(args.checkpoint),
            Path(args.manifest),
            Path(args.root_dir),
            Path(args.output_dir),
        )
        return 0
    if args.command == "robustness":
        config = load_config(Path(args.config))
        run_robustness(
            config,
            Path(args.checkpoint),
            Path(args.manifest),
            Path(args.root_dir),
            Path(args.output_dir),
        )
        return 0
    if args.command == "report":
        write_final_report(
            Path(args.report_path),
            Path(args.manifest),
            Path(args.eval_dir),
            Path(args.robustness_dir),
        )
        return 0

    parser.print_help()
    return 0
