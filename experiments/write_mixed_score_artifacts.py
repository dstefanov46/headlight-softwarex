#!/usr/bin/env python3
"""Write mixed-score ranking artifacts for capacity qualification CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT
from experiments.summarize_capacity_results import COMPARISON_FIELDS


DEFAULT_WEIGHTS = {
    "excess_response_time_seconds": 0.45,
    "sla_violation_rate": 0.25,
    "pod_seconds": 0.15,
    "p95_ms": 0.10,
    "action_churn": 0.05,
}

SCORE_FIELDS = [
    *COMPARISON_FIELDS,
    *(f"normalized_{field}" for field in DEFAULT_WEIGHTS),
    "mixed_score",
]


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_weights(value: str) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_WEIGHTS)
    aliases = {
        "excess": "excess_response_time_seconds",
        "sla": "sla_violation_rate",
        "p95": "p95_ms",
        "pods": "pod_seconds",
        "churn": "action_churn",
    }
    weights = dict(DEFAULT_WEIGHTS)
    for item in value.split(","):
        name, separator, raw_weight = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"invalid weight {item!r}; expected name=value")
        field = aliases.get(name.strip(), name.strip())
        if field not in DEFAULT_WEIGHTS:
            raise argparse.ArgumentTypeError(f"unknown weight field {name!r}")
        try:
            weights[field] = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid weight {raw_weight!r}") from exc
    total = sum(weights.values())
    if total <= 0:
        raise argparse.ArgumentTypeError("weight sum must be positive")
    return {field: weight / total for field, weight in weights.items()}


def numeric(row: dict[str, str], field: str) -> float:
    raw_value = row.get(field, "")
    if field == "pod_seconds" and raw_value in (None, ""):
        raw_value = row.get("mean_pod_count", "")
    if field == "action_churn" and raw_value in (None, ""):
        raw_value = "0"
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with resolve_path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                normalized = {field: row.get(field, "") for field in COMPARISON_FIELDS}
                key = (
                    normalized.get("controller", ""),
                    normalized.get("candidate", ""),
                    normalized.get("result_root", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(normalized)
    return rows


def filter_rows(rows: list[dict[str, str]], controller: str) -> list[dict[str, str]]:
    if controller == "all":
        selected = rows
    else:
        selected = [row for row in rows if row.get("controller") == controller]
    return [
        row
        for row in selected
        if numeric(row, "p95_ms") <= 1000
        and all(math.isfinite(numeric(row, field)) for field in DEFAULT_WEIGHTS)
    ]


def min_max_normalize(value: float, values: list[float]) -> float:
    low = min(values)
    high = max(values)
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def score_rows(rows: list[dict[str, str]], weights: dict[str, float]) -> list[dict[str, str]]:
    values_by_field = {field: [numeric(row, field) for row in rows] for field in weights}
    scored: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        score = 0.0
        for field, weight in weights.items():
            normalized = min_max_normalize(numeric(row, field), values_by_field[field])
            output[f"normalized_{field}"] = f"{normalized:.6f}"
            score += weight * normalized
        output["mixed_score"] = f"{score:.6f}"
        scored.append(output)
    scored.sort(
        key=lambda row: (
            float(row["mixed_score"]),
            numeric(row, "excess_response_time_seconds"),
            numeric(row, "sla_violation_rate"),
            numeric(row, "p95_ms"),
            row.get("candidate", ""),
            row.get("result_root", ""),
        )
    )
    return scored


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], weights: dict[str, float], controller: str) -> None:
    lines = [
        "# Mixed-score ranking",
        "",
        f"Controller filter: `{controller}`.",
        "",
        "Lower score is better. Metrics are min-max normalized over the scored rows.",
        "",
        (
            "Score = "
            f"{weights['excess_response_time_seconds']:.2f} * normalized excess response time + "
            f"{weights['sla_violation_rate']:.2f} * normalized SLA violation rate + "
            f"{weights['pod_seconds']:.2f} * normalized pod-seconds + "
            f"{weights['p95_ms']:.2f} * normalized p95 latency + "
            f"{weights['action_churn']:.2f} * normalized action churn."
        ),
        "",
        "| Rank | Controller | Candidate | Mixed score | Excess (s) | SLA violation (%) | p95 (ms) | CPU (s) | Mean pods |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        candidate = row.get("candidate") or "baseline"
        lines.append(
            "| "
            f"{index} | {row.get('controller', '')} | {candidate} | "
            f"{float(row['mixed_score']):.4f} | "
            f"{numeric(row, 'excess_response_time_seconds'):.2f} | "
            f"{numeric(row, 'sla_violation_rate') * 100:.2f} | "
            f"{numeric(row, 'p95_ms'):.2f} | "
            f"{numeric(row, 'cpu_seconds'):.2f} | "
            f"{numeric(row, 'mean_pod_count'):.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank capacity rows by mixed SLA/latency score")
    parser.add_argument("input_csv", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller", default="headlight", help="controller to score, or 'all'")
    parser.add_argument("--weights", type=parse_weights, default=dict(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = filter_rows(read_rows(args.input_csv), args.controller)
    if not rows:
        raise SystemExit(f"no scoreable rows found for controller={args.controller}")

    scored = score_rows(rows, args.weights)
    write_csv(output_dir / "mixed_scores.csv", scored)
    write_markdown(output_dir / "mixed_score_summary.md", scored, args.weights, args.controller)
    (output_dir / "best_mixed_score.json").write_text(
        json.dumps(scored[0], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "mixed_scores.csv")
    print(output_dir / "mixed_score_summary.md")
    print(output_dir / "best_mixed_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
