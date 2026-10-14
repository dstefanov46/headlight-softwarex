"""Append-only attempt registry and research-v3 retry decisions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from experiments.lib.config import REPO_ROOT
from experiments.lib.protocol import load_protocol, protocol_reference


REGISTRY_SCHEMA = "headlight.attempt-registry/v1"
REQUIRED_FIELDS = {
    "schema_version",
    "attempt_id",
    "controller",
    "workload",
    "seed",
    "started_at_unix",
    "finished_at_unix",
    "commit",
    "protocol_hash",
    "capacity_contract_hash",
    "model_hash",
    "policy_hash",
    "technical_status",
    "performance_status",
    "failure_category",
    "evidence_paths",
    "retry_allowed",
    "retry_rule",
    "retry_decision",
    "superseding_attempt_id",
}


class AttemptRegistryError(RuntimeError):
    """Raised when the append-only registry cannot be trusted."""


def registry_path_from_env(default_root: Path | None = None) -> Path:
    configured = os.environ.get("HEADLIGHT_ATTEMPT_REGISTRY")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / path
    root = default_root or REPO_ROOT / "results" / "socc26"
    return root / "attempt-registry.jsonl"


def file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
        return digest.hexdigest()
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def new_attempt_id(controller: str, workload: str, seed: int) -> str:
    return f"{controller}-{workload}-seed-{seed}-{int(time.time())}-{uuid.uuid4().hex[:10]}"


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AttemptRegistryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise AttemptRegistryError(f"{path}:{line_number}: registry line is not an object")
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            raise AttemptRegistryError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
        attempt_id = str(record.get("attempt_id", ""))
        if attempt_id in seen_ids:
            raise AttemptRegistryError(f"{path}:{line_number}: duplicate attempt_id {attempt_id}")
        seen_ids.add(attempt_id)
        records.append(record)
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    read_registry(path)
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise AttemptRegistryError(f"attempt record is missing fields: {', '.join(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def records_for_cell(
    records: Iterable[dict[str, Any]],
    controller: str,
    workload: str,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("controller") == controller
        and record.get("workload") == workload
        and int(record.get("seed", -1)) == int(seed)
    ]


def effective_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    superseded = {
        str(record["attempt_id"])
        for record in records
        if str(record.get("superseding_attempt_id") or "")
    }
    superseded.update(
        str(record.get("superseded_attempt_id"))
        for record in records
        if str(record.get("superseded_attempt_id") or "")
    )
    return [record for record in records if str(record["attempt_id"]) not in superseded]


def latest_valid_record(
    records: Iterable[dict[str, Any]],
    controller: str,
    workload: str,
    seed: int,
) -> dict[str, Any] | None:
    matches = [
        record
        for record in effective_records(records)
        if record.get("controller") == controller
        and record.get("workload") == workload
        and int(record.get("seed", -1)) == int(seed)
        and record.get("technical_status") == "valid"
    ]
    return sorted(matches, key=lambda record: str(record.get("finished_at_unix") or ""))[-1] if matches else None


def retry_decision(
    *,
    records: Iterable[dict[str, Any]],
    controller: str,
    workload: str,
    seed: int,
    technical_status: str,
    performance_status: str,
    failure_category: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retry_rules = (protocol or load_protocol())["spec"]["retry_rules"]
    max_attempts = int(retry_rules.get("max_attempts_per_seed_run", 1))
    attempts_after_current = len(records_for_cell(records, controller, workload, seed)) + 1
    technical_invalid = technical_status != "valid"
    if not technical_invalid:
        return {"retry_allowed": False, "retry_decision": "do_not_retry", "retry_rule": "performance_never_retried"}
    if not bool(retry_rules.get("retry_only_for_technical_invalidity", True)):
        return {"retry_allowed": False, "retry_decision": "do_not_retry", "retry_rule": "retry_disabled"}
    if attempts_after_current >= max_attempts:
        return {"retry_allowed": False, "retry_decision": "do_not_retry", "retry_rule": "max_attempts_exhausted"}
    if failure_category in {"unplanned_manual_intervention", "contamination"}:
        return {"retry_allowed": False, "retry_decision": "manual_review", "retry_rule": "manual_review_required"}
    return {
        "retry_allowed": True,
        "retry_decision": "retry_same_cell",
        "retry_rule": "technical_invalidity_retry",
        "performance_status_ignored": performance_status,
    }


def build_record(
    *,
    attempt_id: str,
    controller: str,
    workload: str,
    seed: int,
    started_at_unix: int,
    finished_at_unix: int | None,
    technical_status: str,
    performance_status: str,
    failure_category: str = "",
    evidence_paths: list[str] | None = None,
    capacity_contract_path: Path | None = None,
    model_path: Path | None = None,
    policy_path: Path | None = None,
    retry_allowed: bool = False,
    retry_rule: str = "",
    retry_decision_value: str = "do_not_retry",
    superseding_attempt_id: str = "",
) -> dict[str, Any]:
    reference = protocol_reference()
    return {
        "schema_version": REGISTRY_SCHEMA,
        "attempt_id": attempt_id,
        "controller": controller,
        "workload": workload,
        "seed": int(seed),
        "started_at_unix": int(started_at_unix),
        "finished_at_unix": int(finished_at_unix or time.time()),
        "commit": git_commit(),
        "protocol_hash": reference["hash"],
        "capacity_contract_hash": file_sha256(capacity_contract_path),
        "model_hash": file_sha256(model_path),
        "policy_hash": file_sha256(policy_path),
        "technical_status": technical_status,
        "performance_status": performance_status,
        "failure_category": failure_category,
        "evidence_paths": evidence_paths or [],
        "retry_allowed": bool(retry_allowed),
        "retry_rule": retry_rule,
        "retry_decision": retry_decision_value,
        "superseding_attempt_id": superseding_attempt_id,
    }
