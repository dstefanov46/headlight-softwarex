"""Versioned TrainTicket service-profile loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROFILE_SCHEMA_VERSION = "headlight.service-profile/v1"
DEFAULT_CLUSTERS = ("cluster2", "cluster3")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ServiceNode:
    name: str
    workload: str
    service_name: str
    service_type: str
    scalable: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServiceNode":
        required = {"name", "workload", "service_name", "service_type", "scalable"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"service node is missing fields: {', '.join(missing)}")
        service_type = str(value["service_type"])
        if service_type not in {"application", "database"}:
            raise ValueError(f"{value['name']}: unsupported service_type {service_type!r}")
        scalable = bool(value["scalable"])
        if service_type == "database" and scalable:
            raise ValueError(f"{value['name']}: database nodes must be fixed")
        return cls(
            name=str(value["name"]),
            workload=str(value["workload"]),
            service_name=str(value["service_name"]),
            service_type=service_type,
            scalable=scalable,
        )


@dataclass(frozen=True)
class ServiceProfile:
    schema_version: str
    profile_name: str
    profile_version: int
    namespace: str
    sampling_interval_seconds: int
    cluster_limits: dict[str, dict[str, int]]
    nodes: tuple[ServiceNode, ...]
    source: dict[str, Any]
    profile_hash: str

    @property
    def node_order(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def scalable_nodes(self) -> tuple[ServiceNode, ...]:
        return tuple(node for node in self.nodes if node.scalable)

    @property
    def database_nodes(self) -> tuple[ServiceNode, ...]:
        return tuple(node for node in self.nodes if node.service_type == "database")

    def node(self, name: str) -> ServiceNode:
        try:
            return self.nodes[self.node_order.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "namespace": self.namespace,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "cluster_limits": self.cluster_limits,
            "node_order": list(self.node_order),
            "nodes": [
                {
                    "name": node.name,
                    "workload": node.workload,
                    "service_name": node.service_name,
                    "service_type": node.service_type,
                    "scalable": node.scalable,
                }
                for node in self.nodes
            ],
            "source": self.source,
        }
        if include_hash:
            payload["profile_hash"] = self.profile_hash
        return payload

    def assert_compatible(self, other: "ServiceProfile") -> None:
        if self.profile_hash != other.profile_hash:
            raise ValueError(
                "service profile mismatch: "
                f"bundle={self.profile_name}:{self.profile_hash} "
                f"runtime={other.profile_name}:{other.profile_hash}"
            )
        if self.node_order != other.node_order:
            raise ValueError("service profile node order differs from the model bundle")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServiceProfile":
        schema_version = str(value.get("schema_version", ""))
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported service profile schema {schema_version!r}; expected {PROFILE_SCHEMA_VERSION!r}"
            )
        nodes = tuple(ServiceNode.from_dict(node) for node in value.get("nodes", []))
        if not nodes:
            raise ValueError("service profile has no nodes")
        names = tuple(node.name for node in nodes)
        if len(set(names)) != len(names):
            raise ValueError("service profile contains duplicate node names")
        declared_order = tuple(str(name) for name in value.get("node_order", []))
        if declared_order != names:
            raise ValueError("service profile node_order does not match nodes")
        cluster_limits = _validate_cluster_limits(value.get("cluster_limits", {}))
        interval = int(value.get("sampling_interval_seconds", 0))
        if interval <= 0:
            raise ValueError("sampling_interval_seconds must be positive")
        unhashed = dict(value)
        declared_hash = str(unhashed.pop("profile_hash", ""))
        calculated_hash = content_sha256(unhashed)
        if not declared_hash:
            raise ValueError("service profile is missing profile_hash")
        if declared_hash != calculated_hash:
            raise ValueError(
                f"service profile hash mismatch: declared={declared_hash} calculated={calculated_hash}"
            )
        return cls(
            schema_version=schema_version,
            profile_name=str(value.get("profile_name", "")),
            profile_version=int(value.get("profile_version", 0)),
            namespace=str(value.get("namespace", "trainticket")),
            sampling_interval_seconds=interval,
            cluster_limits=cluster_limits,
            nodes=nodes,
            source=dict(value.get("source", {})),
            profile_hash=declared_hash,
        )


def _validate_cluster_limits(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("service profile cluster_limits must be a non-empty object")
    limits: dict[str, dict[str, int]] = {}
    for cluster, raw_limits in value.items():
        if not isinstance(raw_limits, dict):
            raise ValueError(f"{cluster}: cluster limits must be an object")
        minimum = int(raw_limits.get("min_replicas", 1))
        maximum = int(raw_limits.get("max_replicas", 0))
        initial = int(raw_limits.get("initial_replicas", minimum))
        if minimum < 1 or maximum < minimum or not minimum <= initial <= maximum:
            raise ValueError(f"{cluster}: invalid min/initial/max replica limits")
        limits[str(cluster)] = {
            "min_replicas": minimum,
            "initial_replicas": initial,
            "max_replicas": maximum,
        }
    return limits


def build_profile_payload(
    profile_name: str,
    namespace: str,
    nodes: Iterable[dict[str, Any]],
    source: dict[str, Any],
    cluster_limits: dict[str, dict[str, int]] | None = None,
    sampling_interval_seconds: int = 15,
    profile_version: int = 1,
) -> dict[str, Any]:
    node_list = list(nodes)
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_name": profile_name,
        "profile_version": profile_version,
        "namespace": namespace,
        "sampling_interval_seconds": sampling_interval_seconds,
        "cluster_limits": cluster_limits
        or {
            "cluster2": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 3},
            "cluster3": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 5},
        },
        "node_order": [node["name"] for node in node_list],
        "nodes": node_list,
        "source": source,
    }
    payload["profile_hash"] = content_sha256(payload)
    return payload


def load_service_profile(path: str | Path) -> ServiceProfile:
    profile_path = Path(path)
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"service profile does not exist: {profile_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid service profile JSON {profile_path}: {exc}") from exc
    return ServiceProfile.from_dict(payload)
