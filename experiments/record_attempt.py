#!/usr/bin/env python3
"""Validate and append research-v3 attempt-registry records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.attempt_registry import (
    append_record,
    build_record,
    latest_valid_record,
    new_attempt_id,
    read_registry,
    registry_path_from_env,
    retry_decision,
)
from experiments.lib.capacity_contract import contract_path_from_env
from experiments.lib.config import REPO_ROOT, resolve_run
from experiments.lib.quality import performance_outcome
from experiments.lib.quality import validate_bundle


def _registry_arg(value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path
    return registry_path_from_env()


def _summary_row(seed_dir: Path) -> dict[str, str]:
    path = seed_dir / "summaries" / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


def _ceiling(seed_dir: Path) -> dict[str, Any]:
    path = seed_dir / "raw" / "ceiling_binding.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _path_hash_env(name: str) -> Path | None:
    configured = os.environ.get(name)
    if not configured and name == "HEADLIGHT_POLICY_CONFIG":
        configured = str(REPO_ROOT / "ai-module" / "configs" / "policies" / "research-v3-forecast-policy.json")
    if not configured and name == "HEADLIGHT_MODEL_BUNDLE_DIR":
        configured = str(REPO_ROOT / "results" / "calibration" / "headlight-bundle")
    if not configured and name == "HEADLIGHT_LSTM_BUNDLE_DIR":
        configured = str(REPO_ROOT / "results" / "calibration" / "lstm-bundle")
    if not configured:
        return None
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def command_new_id(args: argparse.Namespace) -> int:
    manifest = resolve_run(Path(args.run))
    print(new_attempt_id(manifest["spec"]["controller"]["controller"], manifest["spec"]["workload"]["pattern"], int(args.seed)))
    return 0


def command_status(args: argparse.Namespace) -> int:
    manifest = resolve_run(Path(args.run))
    controller = manifest["spec"]["controller"]["controller"]
    workload = manifest["spec"]["workload"]["pattern"]
    record = latest_valid_record(read_registry(_registry_arg(args.registry)), controller, workload, int(args.seed))
    if record:
        evidence = record.get("evidence_paths") or []
        seed_dir = Path(str(evidence[0])) if evidence else Path()
        try:
            validate_bundle(seed_dir, manifest, dry_run=False)
        except Exception as exc:
            print(json.dumps({"status": "pending", "reason": f"registered evidence is invalid: {exc}"}, sort_keys=True))
            return 0
        print(json.dumps({"status": "complete", "attempt_id": record["attempt_id"], "evidence_paths": evidence}, sort_keys=True))
    else:
        print(json.dumps({"status": "pending"}, sort_keys=True))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    records = read_registry(_registry_arg(args.registry))
    print(json.dumps({"status": "valid", "records": len(records)}, sort_keys=True))
    return 0


def command_append(args: argparse.Namespace) -> int:
    registry = _registry_arg(args.registry)
    prior = read_registry(registry)
    manifest = resolve_run(Path(args.run))
    run = manifest["spec"]["run"]
    controller = manifest["spec"]["controller"]["controller"]
    workload = manifest["spec"]["workload"]["pattern"]
    seed = int(args.seed)
    technical_status = args.technical_status
    failure_category = args.failure_category
    summary = _summary_row(Path(args.seed_dir))
    perf = (
        performance_outcome(summary, run, _ceiling(Path(args.seed_dir)))
        if technical_status == "valid"
        else "not_measured"
    )
    decision = retry_decision(
        records=prior,
        controller=controller,
        workload=workload,
        seed=seed,
        technical_status=technical_status,
        performance_status=perf,
        failure_category=failure_category,
    )
    record = build_record(
        attempt_id=args.attempt_id,
        controller=controller,
        workload=workload,
        seed=seed,
        started_at_unix=int(args.started_at_unix),
        finished_at_unix=int(args.finished_at_unix or time.time()),
        technical_status=technical_status,
        performance_status=perf,
        failure_category=failure_category,
        evidence_paths=[str(Path(args.seed_dir).resolve())],
        capacity_contract_path=contract_path_from_env(),
        model_path=_path_hash_env("HEADLIGHT_LSTM_BUNDLE_DIR" if controller == "lstm" else "HEADLIGHT_MODEL_BUNDLE_DIR"),
        policy_path=_path_hash_env("HEADLIGHT_POLICY_FILE"),
        retry_allowed=bool(decision["retry_allowed"]),
        retry_rule=str(decision["retry_rule"]),
        retry_decision_value=str(decision["retry_decision"]),
        superseding_attempt_id=args.superseding_attempt_id,
    )
    append_record(registry, record)
    print(json.dumps({"attempt_id": args.attempt_id, **decision}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry")
    sub = parser.add_subparsers(dest="command", required=True)
    new_id = sub.add_parser("new-id")
    new_id.add_argument("--run", required=True)
    new_id.add_argument("--seed", required=True)
    new_id.set_defaults(func=command_new_id)
    status = sub.add_parser("status")
    status.add_argument("--run", required=True)
    status.add_argument("--seed", required=True)
    status.set_defaults(func=command_status)
    validate = sub.add_parser("validate")
    validate.set_defaults(func=command_validate)
    append = sub.add_parser("append")
    append.add_argument("--run", required=True)
    append.add_argument("--seed", required=True)
    append.add_argument("--attempt-id", required=True)
    append.add_argument("--seed-dir", required=True)
    append.add_argument("--started-at-unix", required=True)
    append.add_argument("--finished-at-unix")
    append.add_argument("--technical-status", choices=["valid", "invalid"], required=True)
    append.add_argument("--failure-category", default="")
    append.add_argument("--superseding-attempt-id", default="")
    append.set_defaults(func=command_append)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
