"""Opt-in topology overlay for the new shared-gateway experiment path."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from experiments.lib.config import REPO_ROOT


ENV_NAME = "HEADLIGHT_GATEWAY_PROTOCOL_PATH"


def gateway_protocol_from_env() -> dict[str, Any] | None:
    raw_path = os.environ.get(ENV_NAME)
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "GatewayExperimentProtocol":
        raise ValueError(f"invalid gateway protocol: {path}")
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"missing gateway protocol spec: {path}")
    expected = {
        "version": "research-v4-gateway-v1",
        "load_generator_cluster": "cluster1",
        "gateway_cluster": "cluster1",
        "gateway_namespace": "trainticket",
        "workload_clusters": ["cluster2", "cluster3"],
        "transaction_mix": "advanced-ticket",
        "entry_services": {
            "auth": "ts-auth-service",
            "route": "ts-route-service",
            "travel": "ts-travel-service",
        },
        "routing_rule": "one-weight-unit-per-ready-pod-endpointslice-address",
        "zero_ready_endpoint_weight": 0,
        "prior_research_v3_cells_directly_comparable": False,
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise ValueError(f"gateway protocol {field} differs from supported value: {path}")
    if int(spec.get("update_interval_seconds", 0)) <= 0:
        raise ValueError(f"gateway protocol update interval must be positive: {path}")
    if not str(spec.get("gateway_url", "")).startswith("http://trainticket-gateway.trainticket.svc.cluster.local:"):
        raise ValueError(f"gateway protocol URL is not the cluster1 Service: {path}")
    base_path = REPO_ROOT / str(spec.get("base_protocol_path", ""))
    base_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if base_sha256 != spec.get("base_protocol_sha256"):
        raise ValueError(f"gateway protocol base hash mismatch: {base_path}")
    payload["reference"] = {
        "version": spec["version"],
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "base_protocol_sha256": base_sha256,
    }
    return payload
