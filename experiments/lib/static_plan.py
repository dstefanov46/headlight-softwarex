"""Frozen static-baseline plan generation and validation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from experiments.lib.capacity_contract import load_capacity_contract
from experiments.lib.config import REPO_ROOT
from experiments.lib.policies import static_split_replicas
from experiments.lib.protocol import evaluation_seeds, protocol_reference, protocol_spec


STATIC_PLAN_SCHEMA = "headlight.static-plan/v1"


class StaticPlanError(RuntimeError):
    pass


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _seeds(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "seeds", "capacity_seed", "calibration_seeds", "tuning_seeds"}:
                values = item if isinstance(item, list) else [item]
                for candidate in values:
                    try:
                        found.add(int(candidate))
                    except (TypeError, ValueError):
                        pass
            found.update(_seeds(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_seeds(item))
    return found


def static_plan_path() -> Path:
    configured = protocol_spec()["static_baseline"]["plan_path"]
    return REPO_ROOT / str(configured)


def build_static_plan(
    *,
    service_names: list[str],
    capacity_contract_path: Path,
    calibration_trace_name: str,
    calibration_trace_hash: str,
    calibration_seeds: list[int],
) -> dict[str, Any]:
    contract = load_capacity_contract(capacity_contract_path)
    selected_capacity = int(contract["selected_capacity"])
    split = static_split_replicas(selected_capacity)
    created_at = int(time.time())
    assignments = {
        service: {"cluster2": int(split["cluster2"]), "cluster3": int(split["cluster3"])}
        for service in service_names
    }
    payload: dict[str, Any] = {
        "schema_version": STATIC_PLAN_SCHEMA,
        "immutable": True,
        "created_at_unix": created_at,
        "finalized_at_unix": created_at,
        "protocol": protocol_reference(),
        "capacity_contract_hash": _json_hash(contract),
        "selected_capacity": selected_capacity,
        "calibration_trace": {
            "name": calibration_trace_name,
            "sha256": calibration_trace_hash,
            "seeds": list(calibration_seeds),
        },
        "selection_procedure": "Use calibration-selected C and apply protocol static_split_replicas(C) to every scalable service.",
        "assignments": assignments,
    }
    payload["plan_hash"] = _json_hash({key: value for key, value in payload.items() if key != "plan_hash"})
    return payload


def validate_static_plan(path: Path | None = None) -> dict[str, Any]:
    plan_path = path or static_plan_path()
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StaticPlanError(f"frozen static plan is absent: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise StaticPlanError(f"static plan is not valid JSON: {plan_path}: {exc}") from exc
    if plan.get("schema_version") != STATIC_PLAN_SCHEMA:
        raise StaticPlanError("static plan schema is incompatible")
    if plan.get("immutable") is not True:
        raise StaticPlanError("static plan is not marked immutable")
    if plan.get("protocol") != protocol_reference():
        raise StaticPlanError("static plan protocol reference is incompatible")
    actual = _json_hash({key: value for key, value in plan.items() if key != "plan_hash"})
    if plan.get("plan_hash") != actual:
        raise StaticPlanError("static plan hash mismatch")
    leaked = sorted(set(evaluation_seeds()) & _seeds(plan))
    if leaked:
        raise StaticPlanError(f"static plan references evaluation seeds: {leaked}")
    if not isinstance(plan.get("assignments"), dict) or not plan["assignments"]:
        raise StaticPlanError("static plan has no service assignments")
    return plan
