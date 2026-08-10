#!/usr/bin/env python3
"""Build a frozen, traffic-weighted adjacency from calibration Istio edges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.graph import build_static_adjacency
from headlight.profile import load_service_profile
from experiments.lib.paths import resolve_output_path, resolve_repo_path


def load_edges(path: Path) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row.get("source_service") or row.get("source_workload")
            destination = row.get("destination_service") or row.get("destination_workload")
            if not source or not destination or source == "unknown" or destination == "unknown":
                continue
            raw_weight = (
                row.get("request_rate")
                or row.get("requests")
                or row.get("weight")
                or "1"
            )
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            edges.append(
                {
                    "source": source.removesuffix("-service"),
                    "destination": destination.removesuffix("-service"),
                    "weight": weight,
                }
            )
    return edges


def main() -> int:
    parser = argparse.ArgumentParser(description="Build adjacency from Istio source/destination edges")
    parser.add_argument("--input", required=True, help="CSV with source_service,destination_service columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--service-profile", required=True)
    args = parser.parse_args()
    input_path = resolve_repo_path(args.input, must_exist=True)
    service_profile = resolve_repo_path(args.service_profile, must_exist=True)
    profile = load_service_profile(service_profile)
    serializable = build_static_adjacency(profile.node_order, load_edges(input_path))
    output = resolve_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(serializable['observed_edges'])} weighted edges "
        f"for {len(profile.nodes)} nodes: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
