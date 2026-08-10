#!/usr/bin/env python3
"""Run the best three frozen candidates for 20 minutes on seeds 41 and 43."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT, write_yaml
from experiments.lib.paths import resolve_output_path, resolve_repo_path
from experiments.run_capacity_qualification import qualification_manifest
from experiments.run_experiment import environment, run_one_seed
from experiments.write_mixed_score_artifacts import DEFAULT_WEIGHTS, min_max_normalize


TUNING_SEEDS = (41, 43)


def _stable_order_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": candidate["candidate"],
        "step_num_in": candidate["lookback"],
        "num_blocks": candidate["graph_blocks"],
        "hidden_width": candidate["hidden_width"],
        "learning_rate": candidate["learning_rate"],
        "residual_bound": candidate["residual_bound"],
    }


def _summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "excess_response_time_seconds": float(row["excess_response_time_seconds"]),
        "sla_violation_rate": float(row["sla_violation_rate"]),
        "pod_seconds": float(row.get("pod_seconds") or row["mean_pod_count"]),
        "p95_ms": float(row["p95_latency_ms"]),
        "action_churn": float(row.get("action_churn") or 0),
    }


def _run_candidate(candidate: dict[str, Any], seed: int, args: argparse.Namespace) -> dict[str, Any]:
    metadata = _candidate_metadata(candidate)
    os.environ["HEADLIGHT_MODEL_BUNDLE_DIR"] = str(resolve_repo_path(candidate["bundle"], must_exist=True))
    output = args.output_dir / candidate["candidate"] / f"seed-{seed}-run"
    manifest = qualification_manifest(
        "headlight",
        args.selected_capacity,
        1200,
        output,
        "advanced-ticket",
        "capacity-vm",
        seed,
        metadata,
    )
    manifest["metadata"]["name"] = f"tuning-{candidate['candidate']}-seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    write_yaml(output / "manifest.yaml", manifest)
    (output / "environment.json").write_text(
        json.dumps(environment(args.dry_run), indent=2) + "\n",
        encoding="utf-8",
    )
    run_one_seed(manifest, output, seed, args.dry_run)
    summary = (
        dict.fromkeys(DEFAULT_WEIGHTS, 0.0)
        if args.dry_run
        else _summary(output / f"seed-{seed}" / "summaries" / "summary.csv")
    )
    return {"candidate": candidate["candidate"], "seed": seed, **summary}


def _run_all_candidates(
    candidates: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    run_order: dict[str, list[str]] = {}
    completed_runs = 0
    total_runs = len(TUNING_SEEDS) * len(candidates)
    for seed in TUNING_SEEDS:
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate, seed=seed: _stable_order_key(seed, candidate["candidate"]),
        )
        run_order[str(seed)] = [candidate["candidate"] for candidate in ordered_candidates]
        for candidate in ordered_candidates:
            rows.append(_run_candidate(candidate, seed, args))
            completed_runs += 1
            if not args.dry_run and args.cooldown_seconds > 0 and completed_runs < total_runs:
                time.sleep(args.cooldown_seconds)
    return rows, run_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-three",
        type=Path,
        default=REPO_ROOT / "results" / "headlight-grid" / "best-three.json",
    )
    parser.add_argument("--selected-capacity", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "headlight-tuning")
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.best_three = resolve_repo_path(args.best_three, must_exist=True)
    args.output_dir = resolve_output_path(args.output_dir)
    payload = json.loads(args.best_three.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if len(candidates) != 3:
        raise SystemExit("--best-three must contain exactly three candidates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_environment = os.environ.copy()
    try:
        os.environ.update(
            {
                "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
                "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
            }
        )
        rows, run_order = _run_all_candidates(candidates, args)
    finally:
        os.environ.clear()
        os.environ.update(old_environment)

    aggregates: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        name = candidate["candidate"]
        selected = [row for row in rows if row["candidate"] == name]
        aggregates[name] = {
            field: sum(float(row[field]) for row in selected) / len(selected)
            for field in DEFAULT_WEIGHTS
        }
        if not args.dry_run and any(float(row["p95_ms"]) > 1000 for row in selected):
            aggregates[name]["quality_rejected"] = 1.0
    scores = {candidate["candidate"]: 0.0 for candidate in candidates}
    for field, weight in DEFAULT_WEIGHTS.items():
        values = [aggregates[candidate["candidate"]][field] for candidate in candidates]
        for candidate in candidates:
            name = candidate["candidate"]
            scores[name] += weight * min_max_normalize(aggregates[name][field], values)
    eligible = [
        candidate
        for candidate in candidates
        if not aggregates[candidate["candidate"]].get("quality_rejected")
    ]
    if not eligible:
        raise SystemExit("all tuning candidates failed the p95 quality gate")
    winner = min(eligible, key=lambda candidate: (scores[candidate["candidate"]], candidate["candidate"]))
    result = {
        "frozen_before_paper_seeds": True,
        "tuning_seeds": list(TUNING_SEEDS),
        "duration_seconds": 1200,
        "cooldown_seconds": args.cooldown_seconds,
        "randomized_run_order": run_order,
        "weights": DEFAULT_WEIGHTS,
        "runs": rows,
        "aggregates": aggregates,
        "scores": scores,
        "winner": winner,
    }
    (args.output_dir / "winner.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "winner.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
