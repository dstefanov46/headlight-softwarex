#!/usr/bin/env python3
"""Gate paper seeds 102-105 on the predefined paired seed-101 result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT, resolve_run
from experiments.lib.paths import resolve_output_path
from experiments.write_mixed_score_artifacts import DEFAULT_WEIGHTS, min_max_normalize


CONTROLLERS = ("headlight", "lstm", "keda", "hpa")


def _read_run_row(run_path: str, seed: int) -> dict[str, float | str] | None:
    manifest = resolve_run(Path(run_path))
    controller = manifest["spec"]["controller"]["controller"]
    if controller not in CONTROLLERS:
        return None
    summary = REPO_ROOT / manifest["spec"]["run"]["output_dir"] / f"seed-{seed}" / "summaries" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"promotion gate is missing {summary}")
    with summary.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "controller": controller,
        "excess_response_time_seconds": float(row["excess_response_time_seconds"]),
        "sla_violation_rate": float(row["sla_violation_rate"]),
        "pod_seconds": float(row.get("pod_seconds") or row["mean_pod_count"]),
        "p95_ms": float(row["p95_latency_ms"]),
        "action_churn": float(row.get("action_churn") or 0),
    }


def _aggregate(rows: list[dict[str, float | str]], seed: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for controller in CONTROLLERS:
        controller_rows = [row for row in rows if row["controller"] == controller]
        if not controller_rows:
            raise SystemExit(f"promotion gate has no seed-{seed} row for {controller}")
        grouped[controller] = {
            field: sum(float(row[field]) for row in controller_rows) / len(controller_rows)
            for field in DEFAULT_WEIGHTS
        }
        grouped[controller]["workloads"] = float(len(controller_rows))
    return grouped


def _scores(grouped: dict[str, dict[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = dict.fromkeys(CONTROLLERS, 0.0)
    for field, weight in DEFAULT_WEIGHTS.items():
        values = [grouped[controller][field] for controller in CONTROLLERS]
        for controller in CONTROLLERS:
            scores[controller] += weight * min_max_normalize(grouped[controller][field], values)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [row for run in args.run if (row := _read_run_row(run, args.seed)) is not None]
    grouped = _aggregate(rows, args.seed)
    if any(float(row["p95_ms"]) > 1000 for row in rows if row["controller"] == "headlight"):
        raise SystemExit("Headlight seed-101 failed the p95 <= 1000ms promotion gate")
    scores = _scores(grouped)
    ranking = sorted(CONTROLLERS, key=lambda controller: (scores[controller], controller))
    result = {
        "seed": args.seed,
        "weights": DEFAULT_WEIGHTS,
        "aggregates": grouped,
        "scores": scores,
        "ranking": ranking,
        "promoted": ranking[0] == "headlight",
    }
    output = resolve_output_path(args.output or REPO_ROOT / "results" / "socc26" / f"seed-{args.seed}-promotion.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["promoted"]:
        raise SystemExit(
            f"Headlight is not the best mixed-score controller on seed {args.seed}: {ranking}"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
