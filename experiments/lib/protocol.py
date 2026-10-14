"""Research protocol loading, hashing, and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments" / "configs" / "protocols" / "research-v3.yaml"
PROTOCOL_API_VERSION = "headlight.experiments/v1alpha1"
PROTOCOL_KIND = "ResearchProtocol"


class ProtocolError(ValueError):
    """Raised when the checked-in research protocol is invalid."""


def protocol_hash(path: Path = PROTOCOL_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProtocolError(f"{path}: protocol must be a mapping")
    validate_protocol(data, path)
    data.setdefault("metadata", {})["protocol_hash"] = protocol_hash(path)
    return data


def protocol_spec(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    return (protocol or load_protocol())["spec"]


def _require(path: Path, obj: dict[str, Any], field: str) -> Any:
    cursor: Any = obj
    for part in field.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ProtocolError(f"{path}: missing required field `{field}`")
        cursor = cursor[part]
    return cursor


def _int_list(path: Path, value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ProtocolError(f"{path}: `{field}` must be a non-empty integer list")
    return list(value)


def validate_protocol(data: dict[str, Any], path: Path = PROTOCOL_PATH) -> None:
    if data.get("apiVersion") != PROTOCOL_API_VERSION:
        raise ProtocolError(f"{path}: unsupported apiVersion `{data.get('apiVersion')}`")
    if data.get("kind") != PROTOCOL_KIND:
        raise ProtocolError(f"{path}: expected kind `{PROTOCOL_KIND}`, got `{data.get('kind')}`")
    spec = _require(path, data, "spec")
    if spec.get("version") != "research-v3":
        raise ProtocolError(f"{path}: spec.version must be research-v3")
    if spec.get("status") != "pre-execution":
        raise ProtocolError(f"{path}: spec.status must be pre-execution")
    _require(path, spec, "upstream_base_commit")
    seeds = _require(path, spec, "seeds")
    capacity = _int_list(path, seeds.get("capacity"), "spec.seeds.capacity")
    calibration = _int_list(path, seeds.get("calibration"), "spec.seeds.calibration")
    tuning = _int_list(path, seeds.get("tuning"), "spec.seeds.tuning")
    evaluation = _int_list(path, seeds.get("evaluation"), "spec.seeds.evaluation")
    seed_sets = [set(capacity), set(calibration), set(tuning), set(evaluation)]
    if sum(len(item) for item in seed_sets) != len(set().union(*seed_sets)):
        raise ProtocolError(f"{path}: capacity, calibration, tuning, and evaluation seeds must be disjoint")
    if evaluation != [101, 102, 103, 104, 105]:
        raise ProtocolError(f"{path}: evaluation seeds must be [101, 102, 103, 104, 105]")
    if len(capacity) != 1:
        raise ProtocolError(f"{path}: exactly one registered capacity seed is required")

    controllers = _require(path, spec, "controllers.primary")
    controller_ids = [str(item.get("id")) for item in controllers if isinstance(item, dict)]
    if controller_ids != ["headlight", "lstm", "threshold"]:
        raise ProtocolError(f"{path}: primary controllers must be headlight, lstm, threshold")
    excluded = set(_require(path, spec, "controllers.excluded_from_primary"))
    if not {"keda", "hpa", "static"}.issubset(excluded):
        raise ProtocolError(f"{path}: keda, hpa, and static must be excluded from the primary matrix")
    workloads = list(_require(path, spec, "workloads.primary"))
    if workloads != ["gradual-ramp", "flash-crowd", "diurnal"]:
        raise ProtocolError(f"{path}: primary workloads must be gradual-ramp, flash-crowd, diurnal")
    matrix = _require(path, spec, "primary_matrix")
    if int(matrix.get("expected_cell_count", 0)) != 45:
        raise ProtocolError(f"{path}: primary matrix must register exactly 45 cells")
    if matrix.get("concurrency") != "sequential-only":
        raise ProtocolError(f"{path}: primary matrix concurrency must be sequential-only")
    if not {"keda", "hpa", "static"}.issubset(set(matrix.get("live_matrix_excludes", []))):
        raise ProtocolError(f"{path}: keda, hpa, and static must be excluded from live primary matrix")
    if not bool(matrix.get("no_data_dependent_seed_promotion")):
        raise ProtocolError(f"{path}: seed promotion gates must be disabled")

    timing = _require(path, spec, "timing")
    if int(timing.get("warmup_seconds", 0)) != 480:
        raise ProtocolError(f"{path}: warmup_seconds must be 480")
    if int(timing.get("cooldown_seconds", 0)) != 60:
        raise ProtocolError(f"{path}: cooldown_seconds must be 60")
    measurements = timing.get("measurement_seconds", {})
    if measurements != {"gradual-ramp": 1200, "flash-crowd": 1200, "diurnal": 2400}:
        raise ProtocolError(f"{path}: measurement durations do not match research-v3")
    if int(timing.get("diurnal_cycle_count", 0)) != 2:
        raise ProtocolError(f"{path}: diurnal_cycle_count must be 2")

    bounds = _require(path, spec, "replica_bounds")
    per_cluster = bounds.get("per_cluster_max_replicas", {})
    if per_cluster != {"cluster2": 6, "cluster3": 8}:
        raise ProtocolError(f"{path}: per-cluster replica limits must be cluster2=6, cluster3=8")
    if int(bounds.get("aggregate_max_replicas_per_service", 0)) != sum(per_cluster.values()):
        raise ProtocolError(f"{path}: aggregate replica limit must equal the per-cluster sum")
    if float(bounds.get("required_peak_headroom_fraction", 0)) != 0.20:
        raise ProtocolError(f"{path}: required peak replica headroom must be 20 percent")
    if bool(bounds.get("ceiling_hit_invalidates_run", True)):
        raise ProtocolError(f"{path}: ceiling hits must remain measured outcomes")
    forecast_policy = _require(path, spec, "forecast_policy")
    if forecast_policy.get("shared_by") != ["headlight", "lstm"]:
        raise ProtocolError(f"{path}: forecast policy must be shared by headlight and lstm")
    controllers_by_id = {str(item.get("id")): item for item in controllers if isinstance(item, dict)}
    for controller_id in ("headlight", "lstm", "threshold"):
        controller = controllers_by_id[controller_id]
        if controller.get("controller_pod_cluster") != "cluster3":
            raise ProtocolError(f"{path}: {controller_id} must run on cluster3")
        if controller.get("actuation_path") != "actions-handler":
            raise ProtocolError(f"{path}: {controller_id} must use Actions Handler")
    hpa_fairness = _require(path, spec, "hpa_fairness")
    if hpa_fairness.get("scale_up_behavior") != "kubernetes-default" or not hpa_fairness.get("omit_custom_scale_up_behavior"):
        raise ProtocolError(f"{path}: HPA scale-up behavior must use Kubernetes defaults")
    if int(hpa_fairness.get("expected_sync_period_seconds", 0)) != 15:
        raise ProtocolError(f"{path}: expected HPA sync period must be 15 seconds")
    static_baseline = _require(path, spec, "static_baseline")
    if static_baseline.get("selection_data") != "calibration-only":
        raise ProtocolError(f"{path}: static baseline must be calibration-only")
    offline = _require(path, spec, "offline_ablations")
    if offline.get("variants") != ["full", "static_adjacency", "no_attention", "no_graph"]:
        raise ProtocolError(f"{path}: offline ablation variants must match research-v3")
    offline_seeds = _int_list(path, offline.get("training_seeds"), "spec.offline_ablations.training_seeds")
    if not 5 <= len(offline_seeds) <= 10:
        raise ProtocolError(f"{path}: offline ablations require 5-10 training seeds")
    if set(offline_seeds) & set(evaluation):
        raise ProtocolError(f"{path}: offline ablation seeds must not overlap evaluation seeds")
    if offline.get("separate_from_live_matrix") is not True:
        raise ProtocolError(f"{path}: offline ablations must be separate from the live matrix")
    statistics = _require(path, spec, "statistics")
    if statistics.get("independent_unit") != "seed_run":
        raise ProtocolError(f"{path}: statistics must use seed-runs as independent units")
    if statistics.get("five_run_claim_limit") != "descriptive_only_no_strong_significance_claim":
        raise ProtocolError(f"{path}: statistics must avoid strong significance claims from five runs")
    status_rule = _require(path, spec, "sla.successful_request_status_definition")
    if (
        int(status_rule.get("min_status_inclusive", 0)) != 200
        or int(status_rule.get("max_status_exclusive", 0)) != 400
        or bool(status_rule.get("exception_field_must_be_empty")) is not True
    ):
        raise ProtocolError(f"{path}: successful request definition must be HTTP 2xx/3xx without exception")


def protocol_reference(protocol: dict[str, Any] | None = None) -> dict[str, str]:
    loaded = protocol or load_protocol()
    return {
        "version": str(loaded["spec"]["version"]),
        "hash": str(loaded["metadata"]["protocol_hash"]),
        "path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
    }


def primary_controllers(protocol: dict[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(item["id"] for item in protocol_spec(protocol)["controllers"]["primary"])


def primary_workloads(protocol: dict[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(protocol_spec(protocol)["workloads"]["primary"])


def evaluation_seeds(protocol: dict[str, Any] | None = None) -> tuple[int, ...]:
    return tuple(protocol_spec(protocol)["seeds"]["evaluation"])


def capacity_seeds(protocol: dict[str, Any] | None = None) -> tuple[int, ...]:
    return tuple(protocol_spec(protocol)["seeds"]["capacity"])


def calibration_seeds(protocol: dict[str, Any] | None = None) -> tuple[int, ...]:
    return tuple(protocol_spec(protocol)["seeds"]["calibration"])


def tuning_seeds(protocol: dict[str, Any] | None = None) -> tuple[int, ...]:
    return tuple(protocol_spec(protocol)["seeds"]["tuning"])


def replica_limits(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    bounds = protocol_spec(protocol)["replica_bounds"]
    return {
        "min_replicas_per_cluster": int(bounds["min_replicas_per_cluster"]),
        "per_cluster_max_replicas": dict(bounds["per_cluster_max_replicas"]),
        "aggregate_max_replicas_per_service": int(bounds["aggregate_max_replicas_per_service"]),
        "required_peak_headroom_fraction": float(bounds.get("required_peak_headroom_fraction", 0.0)),
    }


def replica_limit_for_cluster(cluster: str, protocol: dict[str, Any] | None = None) -> int:
    limits = replica_limits(protocol)["per_cluster_max_replicas"]
    return int(limits[cluster])


def common_topology(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(protocol_spec(protocol)["common_topology"])


def protocol_timing(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(protocol_spec(protocol)["timing"])


def capacity_contract_path(protocol: dict[str, Any] | None = None) -> Path:
    configured = protocol_spec(protocol)["capacity_normalization"]["contract_path"]
    return REPO_ROOT / str(configured)
