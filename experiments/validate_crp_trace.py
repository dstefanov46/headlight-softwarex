#!/usr/bin/env python3
"""Validate controlled replica propagation traces and maxUnavailable invariants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.paths import resolve_output_path, resolve_repo_path

SCHEMA = "headlight.crp-validation/v1"


class CRPValidationError(RuntimeError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA:
        raise CRPValidationError([f"{config_path}: unsupported CRP validation config"])
    return data


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("events", payload if isinstance(payload, list) else [])
        if not isinstance(rows, list):
            raise CRPValidationError([f"{path}: JSON trace must be a list or contain events[]"])
        return [{key: str(value) for key, value in row.items()} for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "ok"}


def validate_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    required_fields = set(config["record_fields"])
    required_events = set(config.get("required_events", []))
    observed_events = {row.get("event_type", "") for row in rows}
    if not rows:
        issues.append("trace contains no CRP events")
    for event in sorted(required_events - observed_events):
        issues.append(f"missing required CRP event type: {event}")
    for index, row in enumerate(rows, start=1):
        missing = sorted(required_fields - set(row))
        if missing:
            issues.append(f"row {index}: missing fields: {', '.join(missing)}")
            continue
        try:
            desired = int(row["desired_replicas"])
            available = int(row["available_replicas"])
            max_unavailable = int(row.get("max_unavailable", config["max_unavailable"]))
            minimum = int(row.get("minimum_available", desired - max_unavailable))
        except ValueError:
            issues.append(f"row {index}: replica fields must be integers")
            continue
        invariant_status = truthy(row.get("availability_invariant_status"))
        if available < minimum or not invariant_status:
            issues.append(
                f"row {index}: availability violation service={row.get('service')} available={available} minimum={minimum}"
            )
        if max_unavailable != int(config["max_unavailable"]):
            issues.append(f"row {index}: max_unavailable={max_unavailable}, expected {config['max_unavailable']}")
    if issues:
        raise CRPValidationError(issues)
    return {
        "schema_version": "headlight.crp-validation-result/v1",
        "validation_status": "validated",
        "events": len(rows),
        "observed_event_types": sorted(observed_events),
        "max_unavailable": int(config["max_unavailable"]),
    }


def write_dry_run_plan(config: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [*config["record_fields"], "max_unavailable", "validation_status"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "",
                "service": "",
                "desired_replicas": "",
                "available_replicas": "",
                "source_cluster": "",
                "destination_cluster": "",
                "placement_decision": "",
                "event_type": "extension-point",
                "availability_invariant_status": "not-evaluated",
                "max_unavailable": config["max_unavailable"],
                "validation_status": "dry-run-unvalidated",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/config/crp_validation.yaml")
    parser.add_argument("--trace", help="CSV or JSON CRP event trace to validate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="results/softwarex-reproduction/dry-run/crp_validation_plan.csv")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dry_run:
            output = resolve_output_path(args.output)
            write_dry_run_plan(config, output)
            print(output)
            return 0
        if not args.trace:
            raise CRPValidationError(["--trace is required outside dry-run mode"])
        result = validate_rows(load_rows(resolve(args.trace)), config)
    except CRPValidationError as exc:
        for issue in exc.issues:
            print(issue, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
