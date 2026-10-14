#!/usr/bin/env python3
"""Plan or execute the research-v3 offline ablation campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT
from experiments.lib.protocol import evaluation_seeds, protocol_reference


ABLATION_VARIANTS = ("full", "static_adjacency", "no_attention", "no_graph")
DEFAULT_ABLATION_SEEDS = (21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
SCHEMA = "headlight.research-v3.offline-ablation/v1"


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not 5 <= len(seeds) <= 10:
        raise ValueError("offline ablations require 5-10 paired training seeds")
    leaked = sorted(set(seeds) & set(evaluation_seeds()))
    if leaked:
        raise ValueError(f"offline ablation seeds must be distinct from evaluation seeds: {leaked}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("offline ablation seeds must be unique")
    return seeds


def plan_payload(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    cells = []
    ordinal = 1
    for seed in seeds:
        for variant in ABLATION_VARIANTS:
            cells.append(
                {
                    "ordinal": ordinal,
                    "campaign": "offline-ablation",
                    "variant": variant,
                    "seed": seed,
                    "dataset": str(args.dataset),
                    "service_profile": str(args.service_profile),
                    "adjacency": str(args.adjacency),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lookback": args.lookback,
                    "horizon": args.horizon,
                    "learning_rate": args.learning_rate,
                    "hidden_width": args.hidden_width,
                    "graph_blocks": args.graph_blocks,
                    "residual_bound": args.residual_bound,
                    "folds": "shared chronological 70/15/15 split with identical purge/preprocessing for every variant",
                }
            )
            ordinal += 1
    per_cell_seconds = int(args.expected_seconds_per_training)
    return {
        "schema_version": SCHEMA,
        "protocol": protocol_reference(),
        "campaign": "offline-ablation",
        "separate_from_live_matrix": True,
        "statistical_unit": "paired training seed",
        "variants": list(ABLATION_VARIANTS),
        "seeds": list(seeds),
        "cell_count": len(cells),
        "expected_runtime_seconds": len(cells) * per_cell_seconds,
        "expected_runtime_hours": round(len(cells) * per_cell_seconds / 3600.0, 3),
        "paired_design": {
            "baseline_variant": "full",
            "paired_by": "seed",
            "replicate_definition": "one training seed; services and samples within a seed are not independent replicates",
            "uncertainty_interval": "mean paired difference with small-n t interval; reported as descriptive, not a strong significance claim",
        },
        "cells": cells,
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _train_command(args: argparse.Namespace, output_dir: Path, seed: int, variant: str) -> list[str]:
    root = output_dir / f"seed-{seed}" / variant
    return [
        sys.executable,
        str(REPO_ROOT / "ai-module" / "ai-module" / "train_model.py"),
        "--once",
        "--controller",
        "headlight",
        "--dataset",
        str(_resolve(args.dataset)),
        "--service-profile",
        str(_resolve(args.service_profile)),
        "--adjacency",
        str(_resolve(args.adjacency)),
        "--bundle-output",
        str(root / "bundle"),
        "--metrics-output",
        str(root / "training-metrics.json"),
        "--seed",
        str(seed),
        "--variant",
        variant,
        "--lookback",
        str(args.lookback),
        "--horizon",
        str(args.horizon),
        "--graph-blocks",
        str(args.graph_blocks),
        "--hidden-width",
        str(args.hidden_width),
        "--learning-rate",
        str(args.learning_rate),
        "--residual-bound",
        str(args.residual_bound),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--overwrite",
    ]


def execute(args: argparse.Namespace, seeds: tuple[int, ...], output_dir: Path) -> int:
    for required in [args.dataset, args.service_profile, args.adjacency]:
        if not _resolve(required).exists():
            raise FileNotFoundError(f"offline ablation input is missing: {required}")
    for seed in seeds:
        for variant in ABLATION_VARIANTS:
            root = output_dir / f"seed-{seed}" / variant
            root.mkdir(parents=True, exist_ok=True)
            command = _train_command(args, output_dir, seed, variant)
            started = time.time()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT / "ai-module" / "ai-module",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            (root / "training.log").write_text(completed.stdout, encoding="utf-8")
            status = {
                "variant": variant,
                "seed": seed,
                "returncode": completed.returncode,
                "started_at_unix": started,
                "finished_at_unix": time.time(),
            }
            (root / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if completed.returncode != 0:
                return completed.returncode
    write_summary(output_dir)
    return 0


def _metric(path: Path, dotted: str) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cursor: Any = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    try:
        return float(cursor)
    except (TypeError, ValueError):
        return None


def _t_critical_95(df: int) -> float:
    return {
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }.get(df, 1.96)


def paired_interval(differences: list[float]) -> tuple[float, float, float]:
    n = len(differences)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(differences) / n
    if n == 1:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
    margin = _t_critical_95(n - 1) * math.sqrt(variance / n)
    return mean, mean - margin, mean + margin


def write_summary(output_dir: Path) -> None:
    plan = json.loads((output_dir / "offline-ablation-plan.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in plan["seeds"]]
    metrics = ["holdout.loss", "best_validation_loss", "inference.p95_ms"]
    rows: list[dict[str, str]] = []
    for variant in ABLATION_VARIANTS:
        if variant == "full":
            continue
        for metric in metrics:
            differences: list[float] = []
            for seed in seeds:
                full = _metric(output_dir / f"seed-{seed}" / "full" / "training-metrics.json", metric)
                observed = _metric(output_dir / f"seed-{seed}" / variant / "training-metrics.json", metric)
                if full is not None and observed is not None:
                    differences.append(observed - full)
            mean, low, high = paired_interval(differences)
            rows.append(
                {
                    "campaign": "offline-ablation",
                    "variant": variant,
                    "baseline_variant": "full",
                    "metric": metric,
                    "paired_seed_count": str(len(differences)),
                    "mean_paired_difference": f"{mean:.8g}" if differences else "",
                    "ci95_low": f"{low:.8g}" if differences else "",
                    "ci95_high": f"{high:.8g}" if differences else "",
                    "claim": "descriptive-only; no strong significance claim from this sample size",
                    "status": "measured" if differences else "pending_measurements",
                }
            )
    with (output_dir / "offline-ablation-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_placeholder_summary(output_dir: Path) -> None:
    rows = []
    for variant in ABLATION_VARIANTS:
        if variant == "full":
            continue
        for metric in ["holdout.loss", "best_validation_loss", "inference.p95_ms"]:
            rows.append(
                {
                    "campaign": "offline-ablation",
                    "variant": variant,
                    "baseline_variant": "full",
                    "metric": metric,
                    "paired_seed_count": "0",
                    "mean_paired_difference": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "claim": "pending measurements; no result wording is preregistered",
                    "status": "pending_measurements",
                }
            )
    with (output_dir / "offline-ablation-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("results/calibration/train-ticket-calibration.csv"))
    parser.add_argument("--service-profile", type=Path, default=Path("ai-module/configs/service_profiles/advanced-ticket.json"))
    parser.add_argument("--adjacency", type=Path, default=Path("results/calibration/static-adjacency.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/offline-ablations/research-v3"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_ABLATION_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--graph-blocks", type=int, default=2)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--residual-bound", type=float, default=0.3)
    parser.add_argument("--expected-seconds-per-training", type=int, default=900)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_payload(args, seeds)
    (output_dir / "offline-ablation-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(f"offline ablation cells: {plan['cell_count']}")
        print(f"expected sequential runtime: {plan['expected_runtime_seconds']}s ({plan['expected_runtime_hours']}h)")
        print("ordinal\tseed\tvariant\tcampaign")
        for cell in plan["cells"]:
            print(f"{cell['ordinal']}\t{cell['seed']}\t{cell['variant']}\t{cell['campaign']}")
    if not args.execute:
        write_placeholder_summary(output_dir)
        print("dry run only; pass --execute to train offline ablations", file=sys.stderr if args.json else sys.stdout)
        return 0
    return execute(args, seeds, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
