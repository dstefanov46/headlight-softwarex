#!/usr/bin/env python3
"""Validate the 60 selected evaluation runs and reproduce their aggregates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
CONTROLLERS = ("headlight-v2", "hpa", "keda", "threshold")
WORKLOADS = ("gradual-ramp", "flash-crowd", "diurnal")
SEEDS = tuple(range(101, 106))


def validate_evaluation(index: dict, root: Path = ROOT) -> dict:
    if index.get("software_version", SOFTWARE_VERSION) != SOFTWARE_VERSION:
        raise ValueError("evaluation index software version differs from VERSION")
    runs = index["runs"]
    expected = Counter(itertools.product(CONTROLLERS, WORKLOADS, SEEDS))
    actual = Counter((run["controller"], run["workload"], int(run["seed"])) for run in runs)
    if actual != expected:
        raise ValueError("evaluation matrix must contain exactly one run for each of the 60 cells")
    grouped = defaultdict(list)
    verified = {}
    for run in runs:
        if run.get("technical_validity") != "valid":
            raise ValueError("selected run is not technically valid")
        if run.get("performance_outcome") != "measured":
            raise ValueError("selected run is not measured evidence")
        if "summary" not in run["artifacts"]:
            raise ValueError("selected run has no summary")
        for artifact in run["artifacts"].values():
            path = (root / artifact["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("artifact path is outside the repository")
            if path not in verified:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                verified[path] = (digest.hexdigest(), path.stat().st_size)
            if verified[path] != (artifact["sha256"], artifact["size_bytes"]):
                raise ValueError(f"artifact checksum or size mismatch: {artifact['path']}")
        environment = run["artifacts"].get("environment")
        if environment and json.loads((root / environment["path"]).read_text(encoding="utf-8")).get("dry_run"):
            raise ValueError("dry-run environment is not measured evidence")
        with (root / run["artifacts"]["summary"]["path"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError("each selected summary must contain exactly one run")
        row = rows[0]
        if row.get("technical_validity") != "valid":
            raise ValueError("selected summary does not report valid status")
        if row.get("performance_outcome") != "measured":
            raise ValueError("selected summary is not measured evidence")
        if (row["controller"], row["workload"], int(row["seed"])) != (run["controller"], run["workload"], int(run["seed"])):
            raise ValueError("summary identity differs from the index")
        values = {name: float(row[name]) for name in ("p95_latency_ms", "sla_violation_rate", "pod_seconds")}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("summary metrics must be finite and nonnegative")
        if values["sla_violation_rate"] > 1:
            raise ValueError("SLA violation fraction must be between zero and one")
        grouped[(run["controller"], run["workload"])].append(values)
    aggregates = []
    for controller, workload in itertools.product(CONTROLLERS, WORKLOADS):
        rows = grouped[(controller, workload)]
        aggregates.append({"controller": controller, "workload": workload, "n": len(rows), **{
            name: {"mean": statistics.mean(row[name] for row in rows), "sample_sd": statistics.stdev(row[name] for row in rows)}
            for name in ("p95_latency_ms", "sla_violation_rate", "pod_seconds")
        }})
    return {"schema_version": "headlight.softwarex-evaluation-validation/v1", "software_version": SOFTWARE_VERSION, "validation_status": "validated", "validated_run_count": len(runs), "verified_artifact_count": len(verified), "controllers": list(CONTROLLERS), "workloads": list(WORKLOADS), "seeds": list(SEEDS), "aggregates": aggregates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "results/controller-archive/run-index.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_evaluation(json.loads(args.index.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(f"Evaluation validated: {result['validated_run_count']} runs, {result['verified_artifact_count']} intact artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
