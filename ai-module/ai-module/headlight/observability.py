"""Strict, versioned Headlight decision and profiling data contracts.

This module is deliberately independent from the v1 runtime and policy.  It
provides append/read/validation helpers for research-v4 instrumentation while
continuing to accept the unversioned action rows emitted by research-v3.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DECISION_RECORD_SCHEMA_VERSION = "headlight.decision-record/v2"
PROFILING_RECORD_SCHEMA_VERSION = "headlight.profiling-record/v2"
PROFILING_DATASET_SCHEMA_VERSION = "headlight.profiling-dataset/v2"
FROZEN_RESEARCH_V3_EVALUATION_SEEDS = frozenset({101, 102, 103, 104, 105})

_OBSERVED_FIELDS = ("demand_rps", "cpu_utilization", "p95_latency_ms", "error_rate")
_REPLICA_FIELDS = ("ready", "desired", "pending")
_OBJECTIVE_FIELDS = ("resource", "sla_risk", "churn")
_OPTIONAL_OBJECTIVE_FIELDS = ("reliability",)
_ACTION_STATUSES = frozenset({"not_requested", "requested", "acknowledged", "ready", "failed"})
_ACTION_ORIGINS = frozenset({"profiling_isolated", "model_predictive", "reactive_emergency", "manual"})


class SchemaValidationError(ValueError):
    """Raised when a v2 contract is malformed or internally inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable JSON representation suitable for content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fail(path: str, message: str) -> None:
    raise SchemaValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _required_fields(value: Mapping[str, Any], required: Iterable[str], path: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        _fail(path, f"is missing required fields: {', '.join(missing)}")


def _only_fields(value: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    unexpected = sorted(set(value) - set(allowed))
    if unexpected:
        _fail(path, f"has unsupported fields: {', '.join(unexpected)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number" if not nullable else "must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        _fail(path, "must be finite")
    if minimum is not None and number < minimum:
        _fail(path, f"must be at least {minimum}")
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(path, "must be a lowercase SHA-256 hex digest")
    return digest


def _normalise_observed_state(value: Any, path: str) -> dict[str, float]:
    state = _mapping(value, path)
    _required_fields(state, _OBSERVED_FIELDS, path)
    _only_fields(state, _OBSERVED_FIELDS, path)
    return {field: float(_number(state[field], f"{path}.{field}", minimum=0.0)) for field in _OBSERVED_FIELDS}


def _normalise_masks(value: Any, path: str) -> dict[str, dict[str, bool]]:
    masks = _mapping(value, path)
    _required_fields(masks, ("missing", "stale"), path)
    _only_fields(masks, ("missing", "stale"), path)
    result: dict[str, dict[str, bool]] = {}
    for kind in ("missing", "stale"):
        field_masks = _mapping(masks[kind], f"{path}.{kind}")
        _required_fields(field_masks, _OBSERVED_FIELDS, f"{path}.{kind}")
        _only_fields(field_masks, _OBSERVED_FIELDS, f"{path}.{kind}")
        result[kind] = {
            field: _boolean(field_masks[field], f"{path}.{kind}.{field}")
            for field in _OBSERVED_FIELDS
        }
    return result


def _normalise_replicas(value: Any, path: str) -> dict[str, dict[str, int]]:
    replicas = _mapping(value, path)
    _required_fields(replicas, _REPLICA_FIELDS, path)
    _only_fields(replicas, _REPLICA_FIELDS, path)
    cluster_sets: list[set[str]] = []
    result: dict[str, dict[str, int]] = {}
    for state in _REPLICA_FIELDS:
        allocation = _mapping(replicas[state], f"{path}.{state}")
        if not allocation:
            _fail(f"{path}.{state}", "must contain at least one cluster")
        parsed = {
            _string(cluster, f"{path}.{state} cluster"): _integer(count, f"{path}.{state}.{cluster}", minimum=0)
            for cluster, count in allocation.items()
        }
        cluster_sets.append(set(parsed))
        result[state] = dict(sorted(parsed.items()))
    if any(clusters != cluster_sets[0] for clusters in cluster_sets[1:]):
        _fail(path, "ready, desired, and pending placements must name the same clusters")
    return result


def _normalise_forecasts(value: Any, path: str) -> dict[str, dict[str, float]]:
    forecasts = _mapping(value, path)
    if not forecasts:
        _fail(path, "must contain at least one forecast horizon")
    result: dict[str, dict[str, float]] = {}
    for horizon, quantiles_value in forecasts.items():
        horizon_key = _string(horizon, f"{path} horizon")
        try:
            horizon_seconds = int(horizon_key)
        except ValueError:
            _fail(f"{path}.{horizon_key}", "horizon key must be a positive integer number of seconds")
        if horizon_seconds <= 0 or str(horizon_seconds) != horizon_key:
            _fail(f"{path}.{horizon_key}", "horizon key must be a canonical positive integer number of seconds")
        quantiles = _mapping(quantiles_value, f"{path}.{horizon_key}")
        _required_fields(quantiles, ("q50", "q90"), f"{path}.{horizon_key}")
        _only_fields(quantiles, ("q50", "q90"), f"{path}.{horizon_key}")
        q50 = float(_number(quantiles["q50"], f"{path}.{horizon_key}.q50", minimum=0.0))
        q90 = float(_number(quantiles["q90"], f"{path}.{horizon_key}.q90", minimum=0.0))
        if q50 > q90:
            _fail(f"{path}.{horizon_key}", "q50 must not exceed q90")
        result[horizon_key] = {"q50": q50, "q90": q90}
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _normalise_candidate(value: Any, path: str) -> dict[str, Any]:
    candidate = _mapping(value, path)
    required = {
        "candidate_id",
        "total_replicas",
        "placement",
        "prediction_available",
        "predicted_p95_latency_ms",
        "predicted_sla_probability",
        "objective_terms",
    }
    optional = {"predicted_success_probability"}
    _required_fields(candidate, required, path)
    _only_fields(candidate, required | optional, path)
    placement_value = _mapping(candidate["placement"], f"{path}.placement")
    if not placement_value:
        _fail(f"{path}.placement", "must contain at least one cluster")
    placement = {
        _string(cluster, f"{path}.placement cluster"): _integer(replicas, f"{path}.placement.{cluster}", minimum=0)
        for cluster, replicas in placement_value.items()
    }
    total = _integer(candidate["total_replicas"], f"{path}.total_replicas", minimum=0)
    if total != sum(placement.values()):
        _fail(f"{path}.total_replicas", "must equal the sum of placement replicas")
    prediction_available = _boolean(candidate["prediction_available"], f"{path}.prediction_available")
    predicted_p95 = _number(candidate["predicted_p95_latency_ms"], f"{path}.predicted_p95_latency_ms", minimum=0.0, nullable=True)
    predicted_sla = _number(candidate["predicted_sla_probability"], f"{path}.predicted_sla_probability", minimum=0.0, nullable=True)
    predicted_success = _number(
        candidate.get("predicted_success_probability"),
        f"{path}.predicted_success_probability",
        minimum=0.0,
        nullable=True,
    )
    if predicted_sla is not None and predicted_sla > 1.0:
        _fail(f"{path}.predicted_sla_probability", "must not exceed 1")
    if predicted_success is not None and predicted_success > 1.0:
        _fail(f"{path}.predicted_success_probability", "must not exceed 1")
    if prediction_available != (predicted_p95 is not None and predicted_sla is not None):
        _fail(path, "prediction_available must match both candidate prediction fields")
    objective = _mapping(candidate["objective_terms"], f"{path}.objective_terms")
    _required_fields(objective, _OBJECTIVE_FIELDS, f"{path}.objective_terms")
    _only_fields(objective, _OBJECTIVE_FIELDS + _OPTIONAL_OBJECTIVE_FIELDS, f"{path}.objective_terms")
    return {
        "candidate_id": _string(candidate["candidate_id"], f"{path}.candidate_id"),
        "total_replicas": total,
        "placement": dict(sorted(placement.items())),
        "prediction_available": prediction_available,
        "predicted_p95_latency_ms": predicted_p95,
        "predicted_sla_probability": predicted_sla,
        "predicted_success_probability": predicted_success,
        "objective_terms": {
            field: float(_number(objective[field], f"{path}.objective_terms.{field}", minimum=0.0))
            for field in _OBJECTIVE_FIELDS + _OPTIONAL_OBJECTIVE_FIELDS
            if field in objective
        },
    }


def _normalise_action_timestamps(value: Any, path: str) -> dict[str, Any]:
    timestamps = _mapping(value, path)
    required = {"status", "requested_at", "acknowledged_at", "ready_at", "readiness_delay_seconds"}
    _required_fields(timestamps, required, path)
    _only_fields(timestamps, required, path)
    status = _string(timestamps["status"], f"{path}.status")
    if status not in _ACTION_STATUSES:
        _fail(f"{path}.status", f"must be one of {', '.join(sorted(_ACTION_STATUSES))}")
    requested = _number(timestamps["requested_at"], f"{path}.requested_at", minimum=0.0, nullable=True)
    acknowledged = _number(timestamps["acknowledged_at"], f"{path}.acknowledged_at", minimum=0.0, nullable=True)
    ready = _number(timestamps["ready_at"], f"{path}.ready_at", minimum=0.0, nullable=True)
    delay = _number(timestamps["readiness_delay_seconds"], f"{path}.readiness_delay_seconds", minimum=0.0, nullable=True)
    if status == "not_requested":
        if any(item is not None for item in (requested, acknowledged, ready, delay)):
            _fail(path, "not_requested actions must have null timestamps and readiness delay")
    else:
        if requested is None:
            _fail(f"{path}.requested_at", "is required when action status is not_requested is false")
        if status == "requested" and any(item is not None for item in (acknowledged, ready, delay)):
            _fail(path, "requested actions must not have acknowledgement, Ready, or readiness-delay values")
        if status == "acknowledged" and acknowledged is None:
            _fail(f"{path}.acknowledged_at", "is required when action status is acknowledged")
        if status == "acknowledged" and any(item is not None for item in (ready, delay)):
            _fail(path, "acknowledged actions must not have Ready or readiness-delay values")
        if status == "failed" and any(item is not None for item in (ready, delay)):
            _fail(path, "failed actions must not have Ready or readiness-delay values")
        if acknowledged is not None and acknowledged < requested:
            _fail(f"{path}.acknowledged_at", "must not precede requested_at")
        if ready is not None and ready < requested:
            _fail(f"{path}.ready_at", "must not precede requested_at")
        if status == "ready":
            if acknowledged is None:
                _fail(f"{path}.acknowledged_at", "is required when action status is ready")
            if ready is None:
                _fail(f"{path}.ready_at", "is required when action status is ready")
            if ready < acknowledged:
                _fail(f"{path}.ready_at", "must not precede acknowledged_at")
        if ready is None and delay is not None:
            _fail(f"{path}.readiness_delay_seconds", "requires ready_at")
        if ready is not None:
            expected = ready - requested
            if delay is None or not math.isclose(delay, expected, rel_tol=0.0, abs_tol=1e-6):
                _fail(f"{path}.readiness_delay_seconds", "must equal ready_at minus requested_at")
    return {
        "status": status,
        "requested_at": requested,
        "acknowledged_at": acknowledged,
        "ready_at": ready,
        "readiness_delay_seconds": delay,
    }


def validate_decision_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a Headlight decision record.

    The function never fills missing values.  Callers must state unavailable
    candidate predictions as null with ``prediction_available=false`` and must
    state absent actions as ``status=not_requested``.
    """

    record = _mapping(value, "decision record")
    required = {
        "schema_version",
        "record_id",
        "run_id",
        "seed",
        "timestamp",
        "service",
        "observed_state",
        "masks",
        "replicas_by_cluster",
        "time_since_last_action_seconds",
        "forecast_quantiles_by_horizon",
        "candidates",
        "selected_candidate_id",
        "selection_reason",
        "selection_rule",
        "action_timestamps",
        "policy_schema_version",
        "policy_version",
        "policy_id",
        "feature_schema_hash",
        "adjacency_hash",
    }
    optional = {"controller", "bundle_id", "action_origin", "out_of_distribution", "metadata"}
    _required_fields(record, required, "decision record")
    _only_fields(record, required | optional, "decision record")
    if record["schema_version"] != DECISION_RECORD_SCHEMA_VERSION:
        _fail("decision record.schema_version", f"must be {DECISION_RECORD_SCHEMA_VERSION!r}")
    action_origin = _string(record.get("action_origin", "model_predictive"), "decision record.action_origin")
    if action_origin not in _ACTION_ORIGINS:
        _fail("decision record.action_origin", f"must be one of {', '.join(sorted(_ACTION_ORIGINS))}")
    out_of_distribution = _boolean(record.get("out_of_distribution", False), "decision record.out_of_distribution")
    if action_origin == "reactive_emergency" and not out_of_distribution:
        _fail("decision record.out_of_distribution", "must be true for reactive_emergency actions")
    replicas_by_cluster = _normalise_replicas(record["replicas_by_cluster"], "decision record.replicas_by_cluster")
    candidates_raw = record["candidates"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        _fail("decision record.candidates", "must be a non-empty array")
    candidates = [_normalise_candidate(candidate, f"decision record.candidates[{index}]") for index, candidate in enumerate(candidates_raw)]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        _fail("decision record.candidates", "contains duplicate candidate_id values")
    replica_clusters = set(replicas_by_cluster["desired"])
    for candidate in candidates:
        if set(candidate["placement"]) != replica_clusters:
            _fail(
                f"decision record.candidates.{candidate['candidate_id']}.placement",
                "must name exactly the observed replica clusters",
            )
    selected_candidate_id = _string(record["selected_candidate_id"], "decision record.selected_candidate_id")
    if selected_candidate_id not in candidate_ids:
        _fail("decision record.selected_candidate_id", "must name a candidate in candidates")
    metadata = record.get("metadata", {})
    _mapping(metadata, "decision record.metadata")
    return {
        "schema_version": DECISION_RECORD_SCHEMA_VERSION,
        "record_id": _string(record["record_id"], "decision record.record_id"),
        "run_id": _string(record["run_id"], "decision record.run_id"),
        "seed": _integer(record["seed"], "decision record.seed", minimum=0),
        "timestamp": float(_number(record["timestamp"], "decision record.timestamp", minimum=0.0)),
        "service": _string(record["service"], "decision record.service"),
        "observed_state": _normalise_observed_state(record["observed_state"], "decision record.observed_state"),
        "masks": _normalise_masks(record["masks"], "decision record.masks"),
        "replicas_by_cluster": replicas_by_cluster,
        "time_since_last_action_seconds": _number(
            record["time_since_last_action_seconds"],
            "decision record.time_since_last_action_seconds",
            minimum=0.0,
            nullable=True,
        ),
        "forecast_quantiles_by_horizon": _normalise_forecasts(
            record["forecast_quantiles_by_horizon"], "decision record.forecast_quantiles_by_horizon"
        ),
        "candidates": candidates,
        "selected_candidate_id": selected_candidate_id,
        "selection_reason": _string(record["selection_reason"], "decision record.selection_reason"),
        "selection_rule": _string(record["selection_rule"], "decision record.selection_rule"),
        "action_timestamps": _normalise_action_timestamps(record["action_timestamps"], "decision record.action_timestamps"),
        "policy_schema_version": _string(record["policy_schema_version"], "decision record.policy_schema_version"),
        "policy_version": _string(record["policy_version"], "decision record.policy_version"),
        "policy_id": _string(record["policy_id"], "decision record.policy_id"),
        "feature_schema_hash": _sha256(record["feature_schema_hash"], "decision record.feature_schema_hash"),
        "adjacency_hash": _sha256(record["adjacency_hash"], "decision record.adjacency_hash"),
        "controller": _string(record["controller"], "decision record.controller") if "controller" in record else None,
        "bundle_id": _string(record["bundle_id"], "decision record.bundle_id") if "bundle_id" in record else None,
        "action_origin": action_origin,
        "out_of_distribution": out_of_distribution,
        "metadata": dict(metadata),
    }


def _normalise_outcomes(value: Any, path: str) -> dict[str, dict[str, dict[str, Any]]]:
    horizons = _mapping(value, path)
    if not horizons:
        _fail(path, "must contain at least one horizon outcome")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for horizon, endpoints_value in horizons.items():
        horizon_key = _string(horizon, f"{path} horizon")
        try:
            if int(horizon_key) <= 0 or str(int(horizon_key)) != horizon_key:
                raise ValueError
        except ValueError:
            _fail(f"{path}.{horizon_key}", "horizon key must be a canonical positive integer number of seconds")
        endpoints = _mapping(endpoints_value, f"{path}.{horizon_key}")
        if not endpoints:
            _fail(f"{path}.{horizon_key}", "must contain at least one endpoint outcome")
        parsed_endpoints: dict[str, dict[str, Any]] = {}
        for endpoint, outcome_value in endpoints.items():
            outcome = _mapping(outcome_value, f"{path}.{horizon_key}.{endpoint}")
            required = {"offered_requests", "completed_requests", "end_to_end_p95_latency_ms", "sla_violations"}
            _required_fields(outcome, required, f"{path}.{horizon_key}.{endpoint}")
            _only_fields(outcome, required, f"{path}.{horizon_key}.{endpoint}")
            offered = _integer(outcome["offered_requests"], f"{path}.{horizon_key}.{endpoint}.offered_requests", minimum=1)
            completed = _integer(outcome["completed_requests"], f"{path}.{horizon_key}.{endpoint}.completed_requests", minimum=0)
            violations = _integer(outcome["sla_violations"], f"{path}.{horizon_key}.{endpoint}.sla_violations", minimum=0)
            if completed > offered:
                _fail(f"{path}.{horizon_key}.{endpoint}.completed_requests", "must not exceed offered_requests")
            if violations > offered:
                _fail(f"{path}.{horizon_key}.{endpoint}.sla_violations", "must not exceed offered_requests")
            parsed_endpoints[_string(endpoint, f"{path}.{horizon_key} endpoint")] = {
                "offered_requests": offered,
                "completed_requests": completed,
                "end_to_end_p95_latency_ms": float(
                    _number(outcome["end_to_end_p95_latency_ms"], f"{path}.{horizon_key}.{endpoint}.end_to_end_p95_latency_ms", minimum=0.0)
                ),
                "sla_violations": violations,
            }
        result[horizon_key] = dict(sorted(parsed_endpoints.items()))
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def validate_profiling_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one profiling record without inferring any action or outcome."""

    record = _mapping(value, "profiling record")
    required = {"schema_version", "record_id", "run_id", "seed", "decision", "outcomes", "clean_action_label", "exclusion_reasons"}
    _required_fields(record, required, "profiling record")
    _only_fields(record, required, "profiling record")
    if record["schema_version"] != PROFILING_RECORD_SCHEMA_VERSION:
        _fail("profiling record.schema_version", f"must be {PROFILING_RECORD_SCHEMA_VERSION!r}")
    decision = validate_decision_record(_mapping(record["decision"], "profiling record.decision"))
    run_id = _string(record["run_id"], "profiling record.run_id")
    seed = _integer(record["seed"], "profiling record.seed", minimum=0)
    if decision["run_id"] != run_id or decision["seed"] != seed:
        _fail("profiling record.decision", "run_id and seed must match the enclosing profiling record")
    reasons = record["exclusion_reasons"]
    if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        _fail("profiling record.exclusion_reasons", "must be an array of non-empty strings")
    clean = _boolean(record["clean_action_label"], "profiling record.clean_action_label")
    if clean and (decision["out_of_distribution"] or decision["action_origin"] == "reactive_emergency"):
        _fail("profiling record.clean_action_label", "cannot be true for an out-of-distribution emergency action")
    if clean and reasons:
        _fail("profiling record.exclusion_reasons", "must be empty for a clean action label")
    if not clean and not reasons:
        _fail("profiling record.exclusion_reasons", "must contain at least one reason for an excluded action label")
    return {
        "schema_version": PROFILING_RECORD_SCHEMA_VERSION,
        "record_id": _string(record["record_id"], "profiling record.record_id"),
        "run_id": run_id,
        "seed": seed,
        "decision": decision,
        "outcomes": _normalise_outcomes(record["outcomes"], "profiling record.outcomes"),
        "clean_action_label": clean,
        "exclusion_reasons": list(reasons),
    }


def _normalise_seed_set(values: Iterable[int], path: str) -> tuple[int, ...]:
    parsed = tuple(sorted({_integer(value, path, minimum=0) for value in values}))
    if not parsed:
        _fail(path, "must not be empty")
    return parsed


def _validate_split_sets(
    fitting_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    holdout_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    splits = {
        "fitting": _normalise_seed_set(fitting_seeds, "fitting_seeds"),
        "calibration": _normalise_seed_set(calibration_seeds, "calibration_seeds"),
        "holdout": _normalise_seed_set(holdout_seeds, "holdout_seeds"),
        "evaluation": _normalise_seed_set(evaluation_seeds, "evaluation_seeds"),
    }
    names = tuple(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(splits[left]) & set(splits[right]))
            if overlap:
                _fail("seed_partitions", f"{left} and {right} seed sets overlap: {overlap}")
    return splits


def build_profiling_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    feature_schema: Mapping[str, Any],
    adjacency: Mapping[str, Any],
    fitting_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    holdout_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
    forbidden_seeds: Iterable[int] = FROZEN_RESEARCH_V3_EVALUATION_SEEDS,
) -> dict[str, Any]:
    """Build a hash-addressed dataset manifest from complete profiling records.

    Records with incomplete actions/outcomes fail validation; unclean records are
    retained in the manifest's excluded list with their explicit reasons rather
    than becoming clean labels by inference.
    """

    schema = _mapping(feature_schema, "feature_schema")
    graph = _mapping(adjacency, "adjacency")
    if not schema:
        _fail("feature_schema", "must not be empty")
    if not graph:
        _fail("adjacency", "must not be empty")
    feature_schema_hash = content_sha256(schema)
    adjacency_hash = content_sha256(graph)
    splits = _validate_split_sets(fitting_seeds, calibration_seeds, holdout_seeds, evaluation_seeds)
    forbidden = set(FROZEN_RESEARCH_V3_EVALUATION_SEEDS) | set(
        _normalise_seed_set(forbidden_seeds, "forbidden_seeds")
    )
    all_training = set(splits["fitting"]) | set(splits["calibration"]) | set(splits["holdout"])
    if overlap := sorted(all_training & set(splits["evaluation"])):
        _fail("seed_partitions", f"training and evaluation seed leakage: {overlap}")
    if overlap := sorted(all_training & forbidden):
        _fail("seed_partitions", f"training partitions include forbidden seeds: {overlap}")
    validated = [validate_profiling_record(record) for record in records]
    if not validated:
        _fail("records", "must contain at least one profiling record")
    record_ids = [record["record_id"] for record in validated]
    if len(record_ids) != len(set(record_ids)):
        _fail("records", "contains duplicate profiling record_id values")
    sample_keys = [(record["run_id"], record["decision"]["record_id"]) for record in validated]
    if len(sample_keys) != len(set(sample_keys)):
        _fail("records", "contains duplicate profiling decision samples")
    fingerprints = [content_sha256(record) for record in validated]
    if len(fingerprints) != len(set(fingerprints)):
        _fail("records", "contains duplicate profiling samples")
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record, fingerprint in zip(validated, fingerprints):
        seed = record["seed"]
        if seed in set(splits["evaluation"]):
            _fail(f"records.{record['record_id']}", f"evaluation seed leakage: {seed}")
        if seed in forbidden:
            _fail(f"records.{record['record_id']}", f"forbidden seed: {seed}")
        if seed not in all_training:
            _fail(f"records.{record['record_id']}", f"seed {seed} is absent from fitting/calibration/holdout partitions")
        if record["decision"]["feature_schema_hash"] != feature_schema_hash:
            _fail(f"records.{record['record_id']}.decision.feature_schema_hash", "does not match the dataset feature schema")
        if record["decision"]["adjacency_hash"] != adjacency_hash:
            _fail(f"records.{record['record_id']}.decision.adjacency_hash", "does not match the dataset adjacency")
        if record["clean_action_label"]:
            accepted.append(record)
        else:
            excluded.append(
                {
                    "record_id": record["record_id"],
                    "source_record_hash": fingerprint,
                    "exclusion_reasons": record["exclusion_reasons"],
                }
            )
    if not accepted:
        _fail("records", "contains no clean profiling labels after explicit exclusions")
    payload: dict[str, Any] = {
        "schema_version": PROFILING_DATASET_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash,
        "adjacency_hash": adjacency_hash,
        "seed_partitions": {name: list(seeds) for name, seeds in splits.items()},
        "forbidden_seeds": sorted(forbidden),
        "samples": accepted,
        "source_record_hashes": [content_sha256(record) for record in accepted],
        "excluded_records": excluded,
        "sample_count": len(accepted),
        "excluded_record_count": len(excluded),
    }
    payload["dataset_hash"] = content_sha256(payload)
    return payload


def validate_profiling_dataset(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized profiling dataset manifest and its self-hash."""

    dataset = _mapping(value, "profiling dataset")
    required = {
        "schema_version", "feature_schema_hash", "adjacency_hash", "seed_partitions", "forbidden_seeds",
        "samples", "source_record_hashes", "excluded_records", "sample_count", "excluded_record_count", "dataset_hash",
    }
    _required_fields(dataset, required, "profiling dataset")
    _only_fields(dataset, required, "profiling dataset")
    if dataset["schema_version"] != PROFILING_DATASET_SCHEMA_VERSION:
        _fail("profiling dataset.schema_version", f"must be {PROFILING_DATASET_SCHEMA_VERSION!r}")
    _sha256(dataset["feature_schema_hash"], "profiling dataset.feature_schema_hash")
    _sha256(dataset["adjacency_hash"], "profiling dataset.adjacency_hash")
    expected = dict(dataset)
    declared_hash = _sha256(expected.pop("dataset_hash"), "profiling dataset.dataset_hash")
    actual_hash = content_sha256(expected)
    if declared_hash != actual_hash:
        _fail("profiling dataset.dataset_hash", f"mismatch: declared={declared_hash} calculated={actual_hash}")
    partitions = _mapping(dataset["seed_partitions"], "profiling dataset.seed_partitions")
    _required_fields(partitions, ("fitting", "calibration", "holdout", "evaluation"), "profiling dataset.seed_partitions")
    _only_fields(partitions, ("fitting", "calibration", "holdout", "evaluation"), "profiling dataset.seed_partitions")
    splits = _validate_split_sets(partitions["fitting"], partitions["calibration"], partitions["holdout"], partitions["evaluation"])
    forbidden = set(_normalise_seed_set(dataset["forbidden_seeds"], "profiling dataset.forbidden_seeds"))
    missing_frozen = sorted(FROZEN_RESEARCH_V3_EVALUATION_SEEDS - forbidden)
    if missing_frozen:
        _fail("profiling dataset.forbidden_seeds", f"must include frozen research-v3 evaluation seeds: {missing_frozen}")
    if not isinstance(dataset["samples"], list) or not dataset["samples"]:
        _fail("profiling dataset.samples", "must contain at least one validated sample")
    validated_samples = [validate_profiling_record(sample) for sample in dataset["samples"]]
    if len(validated_samples) != _integer(dataset["sample_count"], "profiling dataset.sample_count", minimum=0):
        _fail("profiling dataset.sample_count", "does not match samples")
    sample_ids = [sample["record_id"] for sample in validated_samples]
    if len(sample_ids) != len(set(sample_ids)):
        _fail("profiling dataset.samples", "contains duplicate profiling record_id values")
    sample_keys = [(sample["run_id"], sample["decision"]["record_id"]) for sample in validated_samples]
    if len(sample_keys) != len(set(sample_keys)):
        _fail("profiling dataset.samples", "contains duplicate profiling decision samples")
    training_seeds = set(splits["fitting"]) | set(splits["calibration"]) | set(splits["holdout"])
    if overlap := sorted(training_seeds & forbidden):
        _fail("profiling dataset.seed_partitions", f"training partitions include forbidden seeds: {overlap}")
    for sample in validated_samples:
        if not sample["clean_action_label"]:
            _fail(f"profiling dataset.samples.{sample['record_id']}", "must contain only clean action labels")
        if sample["seed"] not in training_seeds or sample["seed"] in forbidden:
            _fail(f"profiling dataset.samples.{sample['record_id']}", "uses a seed outside the permitted training partitions")
        if sample["decision"]["feature_schema_hash"] != dataset["feature_schema_hash"]:
            _fail(f"profiling dataset.samples.{sample['record_id']}.decision.feature_schema_hash", "does not match the manifest feature schema")
        if sample["decision"]["adjacency_hash"] != dataset["adjacency_hash"]:
            _fail(f"profiling dataset.samples.{sample['record_id']}.decision.adjacency_hash", "does not match the manifest adjacency")
    hashes = dataset["source_record_hashes"]
    if not isinstance(hashes, list) or [content_sha256(sample) for sample in validated_samples] != hashes:
        _fail("profiling dataset.source_record_hashes", "does not match canonical samples")
    excluded_records = dataset["excluded_records"]
    if not isinstance(excluded_records, list):
        _fail("profiling dataset.excluded_records", "must be an array")
    if len(excluded_records) != _integer(dataset["excluded_record_count"], "profiling dataset.excluded_record_count", minimum=0):
        _fail("profiling dataset.excluded_record_count", "does not match excluded_records")
    excluded_ids: list[str] = []
    for index, excluded in enumerate(excluded_records):
        path = f"profiling dataset.excluded_records[{index}]"
        entry = _mapping(excluded, path)
        _required_fields(entry, ("record_id", "source_record_hash", "exclusion_reasons"), path)
        _only_fields(entry, ("record_id", "source_record_hash", "exclusion_reasons"), path)
        record_id = _string(entry["record_id"], f"{path}.record_id")
        _sha256(entry["source_record_hash"], f"{path}.source_record_hash")
        reasons = entry["exclusion_reasons"]
        if not isinstance(reasons, list) or not reasons or any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            _fail(f"{path}.exclusion_reasons", "must be a non-empty array of non-empty strings")
        excluded_ids.append(record_id)
    if len(excluded_ids) != len(set(excluded_ids)):
        _fail("profiling dataset.excluded_records", "contains duplicate record_id values")
    if set(excluded_ids) & set(sample_ids):
        _fail("profiling dataset.excluded_records", "contains record_id values already present in samples")
    return dict(dataset)


def write_decision_record(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append one validated v2 decision record as canonical JSONL."""

    validated = validate_decision_record(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(validated).decode("utf-8") + "\n")
    return validated


def write_profiling_dataset(path: str | Path, dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and write a versioned profiling dataset manifest."""

    validated = validate_profiling_dataset(dataset)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(validated) + b"\n")
    return validated


def load_action_records(path: str | Path) -> list[dict[str, Any]]:
    """Load v2 records strictly while preserving unversioned research-v3 rows.

    Legacy rows are returned unchanged.  That is intentional: their historic
    fields do not satisfy the richer v2 contract and must not be silently
    reinterpreted as allocation-sensitive records.
    """

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise SchemaValidationError(f"{source}:{line_number}: action record must be an object")
            if raw.get("schema_version") == DECISION_RECORD_SCHEMA_VERSION:
                try:
                    records.append(validate_decision_record(raw))
                except SchemaValidationError as exc:
                    raise SchemaValidationError(f"{source}:{line_number}: {exc}") from exc
            elif "schema_version" not in raw:
                records.append(raw)
            else:
                raise SchemaValidationError(f"{source}:{line_number}: unsupported action-log schema {raw['schema_version']!r}")
    return records
