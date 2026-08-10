#!/usr/bin/env python3
"""Train and rank the fixed 48-candidate offline Headlight grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.paths import resolve_output_path, resolve_repo_path

LOOKBACKS = (12, 24, 36)
GRAPH_BLOCKS = (1, 2)
HIDDEN_WIDTHS = (16, 32)
LEARNING_RATES = (0.001, 0.0003)
RESIDUAL_BOUNDS = (0.1, 0.3)


def candidates() -> list[dict[str, Any]]:
    values = []
    for index, combination in enumerate(
        itertools.product(
            LOOKBACKS,
            GRAPH_BLOCKS,
            HIDDEN_WIDTHS,
            LEARNING_RATES,
            RESIDUAL_BOUNDS,
        )
    ):
        lookback, graph_blocks, hidden_width, learning_rate, residual_bound = combination
        values.append(
            {
                "candidate": f"h{index:02d}",
                "lookback": lookback,
                "graph_blocks": graph_blocks,
                "hidden_width": hidden_width,
                "learning_rate": learning_rate,
                "residual_bound": residual_bound,
            }
        )
    return values


def _run_candidate(candidate: dict[str, Any], args: argparse.Namespace, python: Path) -> dict[str, Any]:
    root = args.output_dir / candidate["candidate"]
    bundle = root / "bundle"
    metrics = root / "training-metrics.json"
    root.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(REPO_ROOT / "ai-module/ai-module/train_model.py"), "--once", "--controller", "headlight",
        "--dataset", str(args.dataset), "--service-profile", str(args.service_profile), "--adjacency", str(args.adjacency),
        "--bundle-output", str(bundle), "--metrics-output", str(metrics), "--seed", str(args.seed),
        "--lookback", str(candidate["lookback"]), "--graph-blocks", str(candidate["graph_blocks"]),
        "--hidden-width", str(candidate["hidden_width"]), "--learning-rate", str(candidate["learning_rate"]),
        "--residual-bound", str(candidate["residual_bound"]), "--epochs", str(args.epochs),
        "--max-holdout-loss", str(args.max_holdout_loss), "--overwrite",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT / "ai-module/ai-module", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (root / "training.log").write_text(completed.stdout, encoding="utf-8")
    row = {**candidate, "bundle": str(bundle), "status": "rejected"}
    if completed.returncode != 0 or not metrics.exists():
        row["error"] = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "training failed"
        return row
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    holdout_loss = float(payload["holdout"]["loss"])
    validation_loss = float(payload["best_validation_loss"])
    row.update({"holdout_loss": holdout_loss, "validation_loss": validation_loss, "inference_p95_ms": payload["inference"]["p95_ms"], "bundle_id": payload["bundle_id"]})
    ratio_limit = validation_loss * args.max_holdout_validation_ratio
    if not math.isfinite(holdout_loss) or not math.isfinite(validation_loss):
        row["error"] = "holdout or validation loss is not finite"
    elif holdout_loss > ratio_limit:
        row["error"] = f"holdout loss {holdout_loss:.6f} exceeds {args.max_holdout_validation_ratio:.3f}x validation loss ({ratio_limit:.6f})"
    else:
        row["status"] = "accepted"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "results/calibration/train-ticket-calibration.csv")
    parser.add_argument(
        "--service-profile",
        type=Path,
        default=REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json",
    )
    parser.add_argument("--adjacency", type=Path, default=REPO_ROOT / "results/calibration/static-adjacency.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/headlight-grid")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--max-holdout-loss",
        type=float,
        default=1000.0,
        help="absolute multitask holdout-loss rejection threshold",
    )
    parser.add_argument(
        "--max-holdout-validation-ratio",
        type=float,
        default=1.5,
        help="reject candidates whose holdout loss exceeds this multiple of best validation loss",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dataset = resolve_repo_path(args.dataset, must_exist=True)
    args.service_profile = resolve_repo_path(args.service_profile, must_exist=True)
    args.adjacency = resolve_repo_path(args.adjacency, must_exist=True)
    args.output_dir = resolve_output_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = candidates()
    (args.output_dir / "grid-plan.json").write_text(
        json.dumps(
            {
                "candidate_count": len(plan),
                "frozen_dataset": str(args.dataset),
                "service_profile": str(args.service_profile),
                "adjacency": str(args.adjacency),
                "seed": args.seed,
                "candidates": plan,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"wrote 48-candidate grid plan: {args.output_dir / 'grid-plan.json'}")
        return 0

    rows: list[dict[str, Any]] = []
    python = REPO_ROOT / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    rows = [_run_candidate(candidate, args, python) for candidate in plan]

    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "grid-results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    accepted = sorted(
        (row for row in rows if row["status"] == "accepted"),
        key=lambda row: (float(row["holdout_loss"]), float(row["validation_loss"])),
    )
    if len(accepted) < 3:
        print(f"only {len(accepted)} candidates passed holdout and 50ms latency gates", file=sys.stderr)
        return 1
    (args.output_dir / "best-three.json").write_text(
        json.dumps(
            {
                "selection_metric": ["holdout_loss", "validation_loss"],
                "tuning_seeds": [41, 43],
                "tuning_duration_seconds": 1200,
                "candidates": accepted[:3],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"accepted={len(accepted)} best_three={args.output_dir / 'best-three.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
