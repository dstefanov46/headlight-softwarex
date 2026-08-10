#!/usr/bin/env python3
"""Summarize reduced-slice capacity qualification results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT


COMPARISON_FIELDS = [
    "candidate",
    "headlight_step_num_in",
    "headlight_num_blocks",
    "headlight_hidden_width",
    "headlight_learning_rate",
    "headlight_residual_bound",
    "controller",
    "p95_ms",
    "sla_violation_rate",
    "excess_response_time_seconds",
    "cpu_seconds",
    "mean_pod_count",
    "pod_seconds",
    "action_churn",
    "result_root",
]
BASELINE_CONTROLLERS = ("hpa", "keda", "lstm")


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def measured_bundle(bundle_root: Path, include_dry_run: bool) -> bool:
    environment_path = bundle_root / "environment.json"
    if include_dry_run or not environment_path.exists():
        return True
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return not bool(environment.get("dry_run"))


def complete_measured_seed(summary_path: Path, include_dry_run: bool) -> bool:
    if include_dry_run:
        return True
    seed_dir = summary_path.parents[1]
    return (seed_dir / "raw" / "measurement_window.json").exists()


def successful_campaign(summary_path: Path, include_dry_run: bool) -> bool:
    if include_dry_run:
        return True
    for parent in summary_path.parents:
        status_path = parent / "campaign_status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return status.get("status") == "success"
    return True


def bundle_root_for_summary(summary_path: Path) -> Path:
    for parent in summary_path.parents:
        if (parent / "manifest.yaml").exists():
            return parent
    return summary_path.parents[2]


def manifest_candidate(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return (
        manifest.get("spec", {})
        .get("run", {})
        .get("qualification_overrides", {})
        .get("headlight_candidate", {})
        or {}
    )


def _summary_row(row: dict[str, str], candidate: dict[str, Any], bundle_root: Path) -> dict[str, str]:
    return {
        "candidate": row.get("candidate") or str(candidate.get("candidate", "")),
        "headlight_step_num_in": row.get("headlight_step_num_in") or str(candidate.get("step_num_in", "")),
        "headlight_num_blocks": row.get("headlight_num_blocks") or str(candidate.get("num_blocks", "")),
        "headlight_hidden_width": row.get("headlight_hidden_width") or str(candidate.get("hidden_width", "")),
        "headlight_learning_rate": row.get("headlight_learning_rate") or str(candidate.get("learning_rate", "")),
        "headlight_residual_bound": row.get("headlight_residual_bound") or str(candidate.get("residual_bound", "")),
        "controller": row.get("controller", ""),
        "p95_ms": row.get("p95_latency_ms", ""),
        "sla_violation_rate": row.get("sla_violation_rate", ""),
        "excess_response_time_seconds": row.get("excess_response_time_seconds", ""),
        "cpu_seconds": row.get("cpu_seconds", ""),
        "mean_pod_count": row.get("mean_pod_count", ""),
        "pod_seconds": row.get("pod_seconds") or row.get("mean_pod_count", ""),
        "action_churn": row.get("action_churn", "0"),
        "result_root": display_path(bundle_root),
    }


def _eligible_summary(summary_path: Path, include_dry_run: bool, seen: set[Path]) -> tuple[Path, dict[str, Any]] | None:
    summary_path = summary_path.resolve()
    if summary_path in seen:
        return None
    seen.add(summary_path)
    bundle_root = bundle_root_for_summary(summary_path)
    if not measured_bundle(bundle_root, include_dry_run):
        return None
    if not complete_measured_seed(summary_path, include_dry_run):
        return None
    if not successful_campaign(summary_path, include_dry_run):
        return None
    return bundle_root, manifest_candidate(bundle_root)


def collect_rows(result_roots: list[Path], include_dry_run: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in result_roots:
        root = (REPO_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
        for summary_path in sorted(root.glob("**/seed-*/summaries/summary.csv")):
            eligible = _eligible_summary(summary_path, include_dry_run, seen)
            if eligible is None:
                continue
            bundle_root, candidate = eligible
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(_summary_row(row, candidate, bundle_root))
    return rows


def numeric(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def score_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        numeric(row, "excess_response_time_seconds"),
        numeric(row, "sla_violation_rate"),
        numeric(row, "cpu_seconds"),
        numeric(row, "mean_pod_count"),
    )


def best_headlight(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [row for row in rows if row.get("controller") == "headlight"]
    if not candidates:
        raise RuntimeError("no Headlight rows found")
    return min(candidates, key=score_key)


def latest_baseline_rows(rows: list[dict[str, str]], controllers: tuple[str, ...]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for controller in controllers:
        matches = [row for row in rows if row.get("controller") == controller]
        if not matches:
            continue
        selected.append(max(matches, key=lambda row: row.get("result_root", "")))
    return selected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def emit_csv(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=COMPARISON_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize capacity qualification result rows")
    parser.add_argument("result_roots", nargs="*", type=Path, default=[Path("results/capacity-qualification")])
    parser.add_argument("--candidate-output", type=Path, help="write all collected rows to this CSV")
    parser.add_argument(
        "--final-output",
        type=Path,
        default=Path("results/single_seed_reduced_slice_summary.csv"),
        help="path for the final best-Headlight plus baseline CSV",
    )
    parser.add_argument("--write-final", action="store_true", help="write the final reduced-slice summary CSV")
    parser.add_argument("--include-dry-run", action="store_true", help="include dry-run bundles")
    args = parser.parse_args()

    rows = collect_rows(args.result_roots, include_dry_run=args.include_dry_run)
    rows.sort(key=lambda row: (row["controller"], score_key(row), row["result_root"]))
    if args.candidate_output:
        write_csv(args.candidate_output, rows)
    if args.write_final:
        final_rows = [best_headlight(rows), *latest_baseline_rows(rows, BASELINE_CONTROLLERS)]
        write_csv(args.final_output, final_rows)
    if not args.candidate_output and not args.write_final:
        emit_csv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
