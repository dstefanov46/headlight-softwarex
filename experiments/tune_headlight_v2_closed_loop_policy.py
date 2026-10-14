#!/usr/bin/env python3
"""Select Headlight policy weights on development-only closed-loop seeds."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.closed_loop_v2 import CASE_NAMES, ClosedLoopConfig, ClosedLoopEnvironment, run_episode  # noqa: E402
from run_headlight_v2_closed_loop import NeuralPolicyWeights, _build_neural_controllers  # noqa: E402


DEVELOPMENT_SEEDS = (711, 712, 713)


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("development seeds must not be empty")
    if set(result) & {701, 702, 703, 704, 705}:
        raise argparse.ArgumentTypeError("evaluation seeds 701-705 cannot be used for tuning")
    return result


def _grid() -> tuple[NeuralPolicyWeights, ...]:
    return tuple(
        NeuralPolicyWeights(resource, 1.0, migration, reliability)
        for resource, migration, reliability in itertools.product(
            (0.25, 0.50), (0.10, 1.00), (0.25, 0.50)
        )
    )


def _mean(rows: list[dict[str, object]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEVELOPMENT_SEEDS)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    config = ClosedLoopConfig(steps=args.steps)
    candidates: list[dict[str, object]] = []
    for weights in _grid():
        factory = _build_neural_controllers(
            args.checkpoint_report.resolve(), device, weights
        )["headlight-v2"]
        episodes: list[dict[str, object]] = []
        for case in CASE_NAMES:
            for seed in args.seeds:
                result = run_episode(
                    ClosedLoopEnvironment(case=case, seed=seed, config=config), factory()
                )
                episodes.append(result)
        candidates.append(
            {
                "weights": asdict(weights),
                "episodes": len(episodes),
                "success_rate": _mean(episodes, "success_rate"),
                "request_sla_violation_rate": _mean(
                    episodes, "request_sla_violation_rate"
                ),
                "mean_pod_count": _mean(episodes, "mean_pod_count"),
                "migration_proxy": _mean(episodes, "migration_proxy"),
                "guard_modified_decision_rate": _mean(
                    episodes, "guard_modified_decision_rate"
                ),
                "model_graph_free_disagreement_rate": _mean(
                    episodes, "model_graph_free_disagreement_rate"
                ),
                "graph_edge_sensitivity": _mean(episodes, "graph_edge_sensitivity"),
            }
        )
        print(json.dumps({"event": "candidate-complete", **candidates[-1]}, sort_keys=True), flush=True)

    best_success = max(float(row["success_rate"]) for row in candidates)
    best_sla = min(float(row["request_sla_violation_rate"]) for row in candidates)
    eligible = [
        row
        for row in candidates
        if float(row["success_rate"]) >= best_success - 0.002
        and float(row["request_sla_violation_rate"]) <= best_sla + 0.01
    ]
    pod_min = min(float(row["mean_pod_count"]) for row in eligible)
    pod_span = max(float(row["mean_pod_count"]) for row in eligible) - pod_min
    migration_min = min(float(row["migration_proxy"]) for row in eligible)
    migration_span = max(float(row["migration_proxy"]) for row in eligible) - migration_min
    for row in eligible:
        row["resource_migration_score"] = (
            (float(row["mean_pod_count"]) - pod_min) / max(pod_span, 1e-9)
            + (float(row["migration_proxy"]) - migration_min) / max(migration_span, 1e-9)
        )
    selected = min(
        eligible,
        key=lambda row: (
            float(row["resource_migration_score"]),
            float(row["request_sla_violation_rate"]),
            -float(row["success_rate"]),
            json.dumps(row["weights"], sort_keys=True),
        ),
    )
    result = {
        "schema_version": "headlight.closed-loop-policy-selection/v1",
        "purpose": "development-only policy selection; evaluation seeds excluded",
        "development_seeds": list(args.seeds),
        "evaluation_seeds_excluded": [701, 702, 703, 704, 705],
        "cases": list(CASE_NAMES),
        "steps": args.steps,
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "selection_rule": (
            "within 0.002 of best success and 0.01 of best request-SLA rate; "
            "then minimize equally normalized mean pod count and migration proxy"
        ),
        "weights": selected["weights"],
        "selected_development_metrics": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "selection-complete", "output": str(args.output), "weights": result["weights"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
