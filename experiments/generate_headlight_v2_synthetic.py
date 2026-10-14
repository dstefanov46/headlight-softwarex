#!/usr/bin/env python3
"""Generate the large deterministic Headlight synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.synthetic_v2 import SyntheticConfig, generate_dataset, validate_dataset_files  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic-headlight-v2" / "large-v1",
    )
    parser.add_argument(
        "--paper-summary",
        type=Path,
        default=REPO_ROOT / "results" / "socc26" / "matrix-v2" / "paper-artifacts-43-of-45" / "aggregate_summary.csv",
    )
    parser.add_argument("--fitting-runs", type=int, default=128)
    parser.add_argument("--calibration-runs", type=int, default=32)
    parser.add_argument("--holdout-runs", type=int, default=32)
    parser.add_argument("--steps-per-run", type=int, default=360)
    parser.add_argument("--candidates-per-step", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.replace:
        raise SystemExit(f"output already exists: {manifest_path}; pass --replace to regenerate")
    config = SyntheticConfig(
        fitting_runs=args.fitting_runs,
        calibration_runs=args.calibration_runs,
        holdout_runs=args.holdout_runs,
        steps_per_run=args.steps_per_run,
        candidates_per_step=args.candidates_per_step,
        base_seed=args.base_seed,
    )
    manifest = generate_dataset(
        args.output_dir.resolve(),
        config=config,
        paper_summary=args.paper_summary.resolve(),
    )
    accounting = validate_dataset_files(args.output_dir / "manifest.json")
    print(json.dumps({"dataset_hash": manifest["dataset_hash"], **accounting}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
