from __future__ import annotations

import argparse
from pathlib import Path

from counter_uas import __version__
from counter_uas.config import load_config
from counter_uas.data.dads import export_dads_dataset
from counter_uas.data.synthetic import create_synthetic_dataset
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

    parser.print_help()
    return 0
