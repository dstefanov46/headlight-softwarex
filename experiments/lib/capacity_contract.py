"""Immutable capacity contract generation and validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.lib.protocol import (
    REPO_ROOT,
    capacity_contract_path,
    capacity_seeds,
    common_topology,
    evaluation_seeds,
    protocol_reference,
    replica_limits,
)


CAPACITY_CONTRACT_SCHEMA = "headlight.capacity-contract/v1"


class CapacityContractError(RuntimeError):
    """Raised when evaluation cannot rely on the capacity contract."""


def maximum_headroom_safe_capacity(aggregate_max_replicas: int, headroom_fraction: float) -> int:
    if aggregate_max_replicas <= 0:
        raise CapacityContractError("aggregate replica capacity must be positive")
    if not 0.0 <= headroom_fraction < 1.0:
        raise CapacityContractError("required peak headroom fraction must be in [0, 1)")
    return int(aggregate_max_replicas * (1.0 - headroom_fraction))


def validate_selected_capacity(selected_capacity: int, replica_bounds: dict[str, Any]) -> None:
    aggregate = int(replica_bounds.get("aggregate_max_replicas_per_service", 0) or 0)
    headroom = float(replica_bounds.get("required_peak_headroom_fraction", 0.0) or 0.0)
    maximum = maximum_headroom_safe_capacity(aggregate, headroom)
    if int(selected_capacity) > maximum:
        raise CapacityContractError(
            f"selected capacity C={selected_capacity} exceeds the headroom-safe maximum "
            f"C={maximum} for aggregate capacity {aggregate} and required headroom {headroom:.0%}"
        )


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def evaluation_results_exist(results_root: Path | None = None) -> bool:
    root = results_root or REPO_ROOT / "results" / "socc26"
    if not root.exists():
        return False
    for environment_path in root.glob("**/environment.json"):
        try:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return True
        if not environment.get("dry_run", False):
            return True
    return False


def contract_path_from_env() -> Path:
    configured = os.environ.get("HEADLIGHT_CAPACITY_CONTRACT")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / path
    return capacity_contract_path()


def build_capacity_contract(
    *,
    selected_capacity: int,
    seed: int,
    qualification_inputs: dict[str, Any],
    environment: dict[str, Any],
    pilot_results: list[dict[str, Any]],
    workload_manifest_hashes: dict[str, str],
    gateway_protocol: dict[str, str] | None = None,
) -> dict[str, Any]:
    reference = protocol_reference()
    topology = common_topology()
    limits = replica_limits()
    validate_selected_capacity(selected_capacity, limits)
    created_at = int(time.time())
    payload: dict[str, Any] = {
        "schema_version": CAPACITY_CONTRACT_SCHEMA,
        "immutable": True,
        "created_at_unix": created_at,
        "finalized_at_unix": created_at,
        "protocol": reference,
        "git_commit": git_commit(),
        "selected_capacity": int(selected_capacity),
        "capacity_headroom": {
            "required_fraction": float(limits["required_peak_headroom_fraction"]),
            "minimum_reserved_replicas": int(
                limits["aggregate_max_replicas_per_service"] - selected_capacity
            ),
            "maximum_headroom_safe_capacity": maximum_headroom_safe_capacity(
                limits["aggregate_max_replicas_per_service"],
                limits["required_peak_headroom_fraction"],
            ),
        },
        "capacity_seed": int(seed),
        "evaluation_seeds_excluded": list(evaluation_seeds()),
        "registered_capacity_seeds": list(capacity_seeds()),
        "qualification_inputs": qualification_inputs,
        "environment": environment,
        "pilot_results": pilot_results,
        "topology": topology,
        "replica_bounds": limits,
        "workload_manifest_hashes": workload_manifest_hashes,
        "hashes": {
            "qualification_inputs_sha256": _json_hash(qualification_inputs),
            "environment_sha256": _json_hash(environment),
            "pilot_results_sha256": _json_hash(pilot_results),
            "workload_manifest_hashes_sha256": _json_hash(workload_manifest_hashes),
        },
    }
    if gateway_protocol is not None:
        payload["gateway_protocol"] = gateway_protocol
        payload["topology"] = {
            **topology,
            "ingress_cluster": "cluster1",
            "load_generator_cluster": "cluster1",
            "gateway_protocol": gateway_protocol,
        }
    payload["contract_payload_sha256"] = _json_hash(payload)
    return payload


def write_capacity_contract(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Contracts are finalized read-only. Write a replacement beside the
    # existing file so a later resumable campaign can replace an old
    # finalized contract without changing its permissions first.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_capacity_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = path or contract_path_from_env()
    try:
        return json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CapacityContractError(f"capacity contract is absent: {contract_path}") from exc
    except json.JSONDecodeError as exc:
        raise CapacityContractError(f"capacity contract is not valid JSON: {contract_path}: {exc}") from exc


def _is_read_only(path: Path) -> bool:
    return path.exists() and path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0


def validate_capacity_contract(
    path: Path | None = None,
    *,
    expected_trainticket_path: str | None = None,
    expected_resource_profile: str | None = None,
    expected_gateway_protocol: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    contract_path = path or contract_path_from_env()
    payload = load_capacity_contract(contract_path)
    issues: list[str] = []
    if payload.get("schema_version") != CAPACITY_CONTRACT_SCHEMA:
        issues.append("capacity contract schema is incompatible")
    if payload.get("immutable") is not True or not _is_read_only(contract_path):
        issues.append("capacity contract is mutable")
    if payload.get("protocol") != protocol_reference():
        issues.append("capacity contract protocol reference is incompatible")
    if payload.get("gateway_protocol") != expected_gateway_protocol:
        issues.append("capacity contract gateway protocol reference is incompatible")
    if int(payload.get("capacity_seed", -1)) in evaluation_seeds():
        issues.append("capacity contract was generated from an evaluation seed")
    if int(payload.get("capacity_seed", -1)) not in capacity_seeds():
        issues.append("capacity contract was not generated from the registered capacity seed")
    topology = payload.get("topology", {})
    if expected_trainticket_path and topology.get("trainticket_path") != expected_trainticket_path:
        issues.append("capacity contract TrainTicket path is incompatible")
    if expected_resource_profile and topology.get("resource_profile") != expected_resource_profile:
        issues.append("capacity contract resource profile is incompatible")
    if payload.get("replica_bounds") != replica_limits():
        issues.append("capacity contract replica bounds are incompatible with the registered protocol")
    if issues:
        raise CapacityContractError("; ".join(issues))

    warnings: list[str] = []
    aggregate = int(payload.get("replica_bounds", {}).get("aggregate_max_replicas_per_service", 0) or 0)
    selected_capacity = int(payload.get("selected_capacity", 0) or 0)
    validate_selected_capacity(selected_capacity, payload.get("replica_bounds", {}))
    required_headroom = float(
        payload.get("replica_bounds", {}).get("required_peak_headroom_fraction", 0.0) or 0.0
    )
    if aggregate and selected_capacity > aggregate * (1.0 - required_headroom) * 0.9:
        warnings.append(
            f"peak demand C={selected_capacity} approaches the headroom-safe limit for aggregate capacity {aggregate}"
        )
    return payload, warnings
