"""Deterministic run-level synthetic data for Headlight development.

The simulator is intentionally an engineering benchmark rather than paper
evidence.  Paper aggregates constrain broad operating ranges, while every
trajectory and counterfactual label is generated from an explicit causal model.
Run-level files keep split ownership auditable and avoid copying dense edge
windows tens of thousands of times.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts_v2 import REQUIRED_EDGE_FEATURES, REQUIRED_NODE_FEATURES
from .models_v2 import DERIVED_REQUEST_FEATURES


SCHEMA_VERSION = "headlight.synthetic-dataset/v2"
RUN_SCHEMA_VERSION = "headlight.synthetic-run/v2"
SCENARIOS = (
    "clean",
    "saturation",
    "readiness",
    "placement",
    "telemetry",
    "mixed",
)
WORKLOADS = ("gradual-ramp", "flash-crowd", "diurnal")
ENDPOINTS = ("login", "search", "booking")
HORIZONS_SECONDS = (30, 60, 180)
SAMPLING_INTERVAL_SECONDS = 15
LOOKBACK_STEPS = 24
SLA_LATENCY_MS = 1000.0
CLUSTERS = ("cluster2", "cluster3")

# The paper service order is part of the model and artifact contract.
NODE_ORDER = (
    "ts-auth",
    "ts-auth-mongo",
    "ts-basic",
    "ts-price",
    "ts-price-mongo",
    "ts-route",
    "ts-route-mongo",
    "ts-route-plan",
    "ts-station",
    "ts-station-mongo",
    "ts-ticketinfo",
    "ts-train",
    "ts-train-mongo",
    "ts-travel",
    "ts-travel-mongo",
    "ts-travel-plan",
    "ts-travel2",
    "ts-travel2-mongo",
)
DATABASE_NODES = frozenset(name for name in NODE_ORDER if name.endswith("-mongo"))

# Directed request propagation.  These edges are deliberately richer than the
# three edges observed in the old calibration scrape and match TrainTicket roles.
ROUTING_EDGES = (
    ("ts-basic", "ts-auth", 0.40),
    ("ts-basic", "ts-route", 0.75),
    ("ts-basic", "ts-price", 0.70),
    ("ts-basic", "ts-station", 0.45),
    ("ts-basic", "ts-ticketinfo", 0.85),
    ("ts-basic", "ts-travel", 0.65),
    ("ts-basic", "ts-travel2", 0.55),
    ("ts-auth", "ts-auth-mongo", 0.80),
    ("ts-price", "ts-price-mongo", 0.80),
    ("ts-route", "ts-route-plan", 0.65),
    ("ts-route", "ts-route-mongo", 0.70),
    ("ts-station", "ts-station-mongo", 0.75),
    ("ts-ticketinfo", "ts-train", 0.55),
    ("ts-ticketinfo", "ts-travel", 0.60),
    ("ts-ticketinfo", "ts-travel2", 0.45),
    ("ts-train", "ts-train-mongo", 0.75),
    ("ts-travel", "ts-travel-plan", 0.60),
    ("ts-travel", "ts-travel-mongo", 0.75),
    ("ts-travel2", "ts-travel-plan", 0.55),
    ("ts-travel2", "ts-travel2-mongo", 0.75),
)

ENDPOINT_PATHS = {
    "login": ("ts-basic", "ts-auth", "ts-auth-mongo"),
    "search": (
        "ts-basic", "ts-route", "ts-route-plan", "ts-station",
        "ts-ticketinfo", "ts-train", "ts-travel",
    ),
    "booking": (
        "ts-basic", "ts-auth", "ts-ticketinfo", "ts-travel",
        "ts-travel-plan", "ts-travel2",
    ),
}

NODE_FEATURE_SCALES = {
    "request_rate": 12.0,
    "cpu_usage_cores": 4.0,
    "cpu_utilization": 1.0,
    "mean_latency_ms": 1000.0,
    "p95_latency_ms": 1000.0,
    "error_rate": 1.0,
    "desired_replicas": 14.0,
    "ready_replicas": 14.0,
    "pending_replicas": 14.0,
    "pending_age_seconds": 300.0,
    "time_since_last_action_seconds": 600.0,
    "request_rate_missing": 1.0,
    "cpu_missing": 1.0,
    "latency_missing": 1.0,
    "error_missing": 1.0,
    "request_rate_stale": 1.0,
    "cpu_stale": 1.0,
    "latency_stale": 1.0,
    "error_stale": 1.0,
}
EDGE_FEATURE_SCALES = {
    "call_rate": 12.0,
    "call_ratio": 1.0,
    "latency_ms": 1000.0,
    "error_rate": 1.0,
    "cross_cluster": 1.0,
    "missing": 1.0,
    "stale": 1.0,
}


@dataclass(frozen=True)
class SyntheticConfig:
    fitting_runs: int = 128
    calibration_runs: int = 32
    holdout_runs: int = 32
    steps_per_run: int = 360
    candidates_per_step: int = 8
    base_seed: int = 10_000
    lookback_steps: int = LOOKBACK_STEPS
    sampling_interval_seconds: int = SAMPLING_INTERVAL_SECONDS
    horizons_seconds: tuple[int, ...] = HORIZONS_SECONDS
    sla_latency_ms: float = SLA_LATENCY_MS

    def validate(self) -> None:
        if min(self.fitting_runs, self.calibration_runs, self.holdout_runs) <= 0:
            raise ValueError("every synthetic split must contain at least one run")
        if self.steps_per_run < self.lookback_steps + max(self.horizon_steps) + 8:
            raise ValueError("steps_per_run is too small for lookback and horizons")
        if self.candidates_per_step < 4:
            raise ValueError("at least four counterfactual candidates are required")
        if len(set(self.horizons_seconds)) != len(self.horizons_seconds):
            raise ValueError("synthetic horizons must be unique")
        if any(value <= 0 or value % self.sampling_interval_seconds for value in self.horizons_seconds):
            raise ValueError("synthetic horizons must be positive sampling-interval multiples")

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(value // self.sampling_interval_seconds for value in self.horizons_seconds)

    @property
    def total_runs(self) -> int:
        return self.fitting_runs + self.calibration_runs + self.holdout_runs


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paper_priors(aggregate_summary: Path) -> dict[str, Any]:
    """Reduce paper rows to broad, non-label operating priors."""

    with aggregate_summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("paper aggregate summary contains no rows")

    def values(metric: str) -> list[float]:
        result = [float(row["mean"]) for row in rows if row["metric"] == metric]
        if not result:
            raise ValueError(f"paper aggregate summary has no {metric!r} rows")
        return result

    throughput = values("successful_throughput_rps")
    latency = values("p95_latency_ms")
    cpu = values("cpu_seconds")
    pod = values("pod_seconds")
    return {
        "source": str(aggregate_summary),
        "source_sha256": sha256_file(aggregate_summary),
        "use": "broad operating priors only; paper rows are not synthetic labels",
        "successful_throughput_rps": {"minimum_mean": min(throughput), "maximum_mean": max(throughput)},
        "p95_latency_ms": {"minimum_mean": min(latency), "maximum_mean": max(latency)},
        "cpu_seconds": {"minimum_mean": min(cpu), "maximum_mean": max(cpu)},
        "pod_seconds": {"minimum_mean": min(pod), "maximum_mean": max(pod)},
        "paper_capacity_users": 4,
        "paper_sla_latency_ms": SLA_LATENCY_MS,
    }


def feature_schema() -> dict[str, Any]:
    return {
        "schema_version": "headlight.features/v2",
        "node_feature_names": list(REQUIRED_NODE_FEATURES),
        "edge_feature_names": list(REQUIRED_EDGE_FEATURES),
        "derived_request_features": list(DERIVED_REQUEST_FEATURES),
        "node_feature_scales": [NODE_FEATURE_SCALES[name] for name in REQUIRED_NODE_FEATURES],
        "edge_feature_scales": [EDGE_FEATURE_SCALES[name] for name in REQUIRED_EDGE_FEATURES],
        "missing_values": "zero-with-explicit-mask",
    }


def build_adjacency() -> dict[str, Any]:
    size = len(NODE_ORDER)
    index = {name: offset for offset, name in enumerate(NODE_ORDER)}
    raw = np.eye(size, dtype=np.float64)
    observed = np.eye(size, dtype=np.int64)
    for source, destination, weight in ROUTING_EDGES:
        raw[index[source], index[destination]] += float(weight)
        observed[index[source], index[destination]] = 1
    matrix = raw / raw.sum(axis=1, keepdims=True)
    return {
        "schema_version": "headlight.adjacency/v1",
        "node_order": list(NODE_ORDER),
        "orientation": "source-to-destination",
        "matrix": matrix.astype(np.float32).tolist(),
        "observed_mask": observed.tolist(),
    }


def model_config(config: SyntheticConfig) -> dict[str, Any]:
    service_types = [1 if name in DATABASE_NODES else 0 for name in NODE_ORDER]
    return {
        "schema_version": "headlight.model-config/v2",
        "model_type": "headlight-v2",
        "num_nodes": len(NODE_ORDER),
        "node_order": list(NODE_ORDER),
        "node_feature_names": list(REQUIRED_NODE_FEATURES),
        "edge_feature_names": list(REQUIRED_EDGE_FEATURES),
        "service_type_ids": service_types,
        "num_service_types": 2,
        "horizons_seconds": list(config.horizons_seconds),
        "endpoint_names": list(ENDPOINTS),
        "sampling_interval_seconds": config.sampling_interval_seconds,
        "hidden_width": 48,
        "tcn_dilations": [1, 2, 4],
        "tcn_kernel_size": 3,
        "graph_blocks": 2,
        "service_embedding_dim": 8,
        "service_type_embedding_dim": 4,
        "position_encoding_dim": 8,
        "short_slope_window": 3,
        "long_slope_window": 8,
        "num_clusters": len(CLUSTERS),
        "readiness_feature_size": 2,
        "readiness_feature_names": ["ready_fraction", "pending_age_seconds"],
        "residual_bound": 0.1,
        "dropout": 0.05,
        "demand_output_scale_rps": 6.0,
        "latency_output_scale_ms": 300.0,
    }


def _workload_users(workload: str, steps: int, capacity: float, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(steps, dtype=np.float64)
    low = max(0.5, capacity * rng.uniform(0.15, 0.30))
    if workload == "gradual-ramp":
        ramp = max(8, int(steps * rng.uniform(0.60, 0.82)))
        users = low + (capacity - low) * np.minimum(time / ramp, 1.0)
    elif workload == "flash-crowd":
        users = np.full(steps, low)
        width = max(5, int(steps * rng.uniform(0.08, 0.15)))
        for center in (int(steps * rng.uniform(0.25, 0.38)), int(steps * rng.uniform(0.68, 0.82))):
            distance = np.abs(time - center)
            users += (capacity - low) * np.clip(1.0 - distance / width, 0.0, 1.0)
        users = np.minimum(users, capacity)
    elif workload == "diurnal":
        cycles = rng.uniform(1.5, 2.5)
        phase = rng.uniform(-math.pi, math.pi)
        wave = 0.5 + 0.5 * np.sin((2.0 * math.pi * cycles * time / steps) + phase)
        users = low + (capacity - low) * wave
    else:
        raise ValueError(f"unsupported workload {workload!r}")
    noise = rng.normal(0.0, max(0.02, capacity * 0.015), size=steps)
    return np.clip(users + noise, 0.1, capacity * 1.08)


def _routing_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {name: offset for offset, name in enumerate(NODE_ORDER)}
    sources = np.asarray([index[source] for source, _, _ in ROUTING_EDGES], dtype=np.int64)
    destinations = np.asarray([index[destination] for _, destination, _ in ROUTING_EDGES], dtype=np.int64)
    ratios = np.asarray([ratio for _, _, ratio in ROUTING_EDGES], dtype=np.float32)
    return sources, destinations, ratios


def _propagated_demand(users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    steps = len(users)
    nodes = len(NODE_ORDER)
    source, destination, ratios = _routing_arrays()
    injection_weight = np.zeros(nodes, dtype=np.float64)
    injection_weight[NODE_ORDER.index("ts-basic")] = rng.uniform(0.75, 1.05)
    injection_weight[NODE_ORDER.index("ts-auth")] = rng.uniform(0.08, 0.16)
    injection_weight[NODE_ORDER.index("ts-travel")] = rng.uniform(0.05, 0.12)
    demand = users[:, None] * injection_weight[None, :] * rng.uniform(0.85, 1.20)
    # The graph is acyclic enough for a few deterministic propagation passes.
    for _ in range(4):
        propagated = np.zeros_like(demand)
        for edge, (left, right) in enumerate(zip(source, destination)):
            propagated[:, right] += demand[:, left] * float(ratios[edge])
        demand = users[:, None] * injection_weight[None, :] + propagated
    service_jitter = rng.lognormal(mean=0.0, sigma=0.10, size=nodes)
    demand *= service_jitter[None, :]
    demand += rng.normal(0.0, 0.015, size=demand.shape) * np.maximum(1.0, demand)
    return np.clip(demand, 0.0, None).astype(np.float32)


def _replica_state(
    demand: np.ndarray,
    per_replica_capacity: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps, nodes = demand.shape
    desired = np.ones((steps, nodes), dtype=np.float32)
    ready = np.ones_like(desired)
    pending_age = np.zeros_like(desired)
    action_age = np.full_like(desired, 600.0)
    placement = np.ones((steps, nodes, len(CLUSTERS)), dtype=np.float32)
    placement[..., 1] = 1.0
    last_action = np.full(nodes, -10_000, dtype=np.int64)
    startup = rng.integers(2, 7, size=nodes)
    aggressive = 1.35 if scenario in {"saturation", "mixed"} else 1.05

    for step in range(1, steps):
        desired[step] = desired[step - 1]
        ready[step] = ready[step - 1]
        placement[step] = placement[step - 1]
        target = np.ceil(aggressive * demand[step] / np.maximum(per_replica_capacity, 0.1)).clip(2, 14)
        target[[NODE_ORDER.index(name) for name in DATABASE_NODES]] = 2
        if step % 4 == 0:
            change = np.sign(target - desired[step])
            change = np.clip(change, -1, 2)
            changed = change != 0
            desired[step, changed] = np.clip(desired[step, changed] + change[changed], 2, 14)
            last_action[changed] = step
            for node in np.flatnonzero(changed):
                total = int(desired[step, node])
                skew = 0.65 if scenario == "placement" and (step // 12) % 2 else 0.43
                left = int(np.clip(round(total * skew), 1, min(6, total - 1)))
                placement[step, node] = (left, total - left)
        for node in range(nodes):
            elapsed = step - last_action[node]
            action_age[step, node] = min(600.0, max(0, elapsed) * SAMPLING_INTERVAL_SECONDS)
            if ready[step, node] < desired[step, node]:
                delay = int(startup[node] * (1.8 if scenario in {"readiness", "mixed"} else 1.0))
                if elapsed >= delay:
                    ready[step, node] = min(desired[step, node], ready[step, node] + 1)
                else:
                    pending_age[step, node] = max(0, elapsed) * SAMPLING_INTERVAL_SECONDS
            elif ready[step, node] > desired[step, node]:
                ready[step, node] = desired[step, node]
    pending = np.maximum(desired - ready, 0.0)
    return desired, ready, pending, pending_age, placement


def _service_observations(
    demand: np.ndarray,
    desired: np.ndarray,
    ready: np.ndarray,
    pending: np.ndarray,
    pending_age: np.ndarray,
    per_replica_capacity: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    utilization = demand / np.maximum(ready * per_replica_capacity[None, :], 0.1)
    base = rng.uniform(12.0, 42.0, size=demand.shape[1])
    queue = np.square(np.maximum(utilization, 0.0)) / np.maximum(0.08, 1.15 - utilization)
    p95 = base[None, :] + 55.0 * queue
    if scenario in {"saturation", "mixed"}:
        p95 *= 1.12
    p95 *= rng.lognormal(0.0, 0.035, size=p95.shape)
    p95 = np.clip(p95, 5.0, 5000.0)
    mean_latency = np.clip(p95 * rng.uniform(0.38, 0.58, size=p95.shape), 2.0, None)
    cpu_utilization = np.clip(0.04 + 0.78 * utilization + rng.normal(0, 0.025, utilization.shape), 0.0, 1.5)
    cpu_usage = cpu_utilization * np.maximum(ready, 1.0) * rng.uniform(0.12, 0.30, size=demand.shape[1])
    error = np.clip(0.001 + 0.18 * np.square(np.maximum(utilization - 0.85, 0.0)), 0.0, 1.0)
    return cpu_usage.astype(np.float32), cpu_utilization.astype(np.float32), p95.astype(np.float32), error.astype(np.float32)


def _candidate_arrays(
    desired: np.ndarray,
    current_placement: np.ndarray,
    ready: np.ndarray,
    pending_age: np.ndarray,
    demand: np.ndarray,
    per_replica_capacity: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, nodes = desired.shape
    totals = np.empty((steps, count, nodes), dtype=np.float32)
    placements = np.empty((steps, count, nodes, len(CLUSTERS)), dtype=np.float32)
    readiness = np.empty((steps, count, nodes, 2), dtype=np.float32)
    database_indices = np.asarray([NODE_ORDER.index(name) for name in DATABASE_NODES])
    offsets = np.linspace(-3, 4, count).round().astype(np.int64)
    for step in range(steps):
        oracle = np.ceil(demand[step] / np.maximum(per_replica_capacity, 0.1)).clip(2, 14)
        for candidate, offset in enumerate(offsets):
            if candidate == 0:
                total = desired[step].copy()
            elif candidate == count - 1:
                total = np.full(nodes, 14.0)
            else:
                blend = 0.65 * oracle + 0.35 * desired[step]
                total = np.clip(np.rint(blend + offset), 2, 14)
            total[database_indices] = 2
            totals[step, candidate] = total
            if candidate == 0:
                left = current_placement[step, :, 0]
            else:
                current_total = np.maximum(current_placement[step].sum(axis=-1), 1.0)
                current_share = current_placement[step, :, 0] / current_total
                left = np.clip(np.rint(total * current_share), 1, total - 1)
            placements[step, candidate, :, 0] = left
            placements[step, candidate, :, 1] = total - left
            ready_fraction = np.minimum(ready[step], total) / np.maximum(total, 1.0)
            readiness[step, candidate, :, 0] = ready_fraction
            readiness[step, candidate, :, 1] = np.where(
                ready_fraction < 1.0,
                pending_age[step] + rng.uniform(0.0, 15.0),
                0.0,
            )
    return totals, placements, readiness


def _counterfactual_targets(
    demand: np.ndarray,
    candidates: np.ndarray,
    placement: np.ndarray,
    readiness: np.ndarray,
    per_replica_capacity: np.ndarray,
    horizon_steps: Sequence[int],
    sla_latency_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps, candidate_count, nodes = candidates.shape
    endpoint_count = len(ENDPOINTS)
    horizon_count = len(horizon_steps)
    demand_target = np.full((steps, nodes, horizon_count), np.nan, dtype=np.float32)
    latency = np.full((steps, candidate_count, endpoint_count, horizon_count), np.nan, dtype=np.float32)
    sla = np.full_like(latency, np.nan)
    reliability = np.full_like(latency, np.nan)
    boundary = np.zeros_like(latency, dtype=np.uint8)
    node_index = {name: index for index, name in enumerate(NODE_ORDER)}
    for step in range(steps - max(horizon_steps)):
        for horizon_index, horizon in enumerate(horizon_steps):
            future = demand[step + horizon]
            demand_target[step, :, horizon_index] = future
            for candidate in range(candidate_count):
                effective_ready = np.maximum(
                    1.0,
                    candidates[step, candidate] * np.clip(readiness[step, candidate, :, 0], 0.2, 1.0),
                )
                utilization = future / np.maximum(effective_ready * per_replica_capacity, 0.1)
                service_latency = 18.0 + 62.0 * np.square(utilization) / np.maximum(0.06, 1.12 - utilization)
                cross_cluster = np.minimum(placement[step, candidate, :, 0], placement[step, candidate, :, 1]) > 0
                service_latency += cross_cluster.astype(np.float32) * 5.0
                service_latency += (1.0 - readiness[step, candidate, :, 0]) * 180.0
                service_latency = np.clip(service_latency, 4.0, 5000.0)
                service_error = np.clip(
                    0.001
                    + 0.18 * np.square(np.maximum(utilization - 0.85, 0.0))
                    + 0.12 * (1.0 - readiness[step, candidate, :, 0]),
                    0.0,
                    1.0,
                )
                for endpoint_index, endpoint in enumerate(ENDPOINTS):
                    path = [node_index[name] for name in ENDPOINT_PATHS[endpoint]]
                    value = 35.0 + float(np.sum(service_latency[path]))
                    value *= 1.0 + 0.03 * horizon_index
                    latency[step, candidate, endpoint_index, horizon_index] = value
                    sla[step, candidate, endpoint_index, horizon_index] = float(value > sla_latency_ms)
                    reliability[step, candidate, endpoint_index, horizon_index] = float(
                        np.prod(1.0 - service_error[path])
                    )
                    boundary[step, candidate, endpoint_index, horizon_index] = int(
                        abs(value - sla_latency_ms) <= 200.0
                    )
    return demand_target, latency, sla, reliability, boundary


def _node_features(
    demand: np.ndarray,
    desired: np.ndarray,
    ready: np.ndarray,
    pending: np.ndarray,
    pending_age: np.ndarray,
    cpu_usage: np.ndarray,
    cpu_utilization: np.ndarray,
    p95_latency: np.ndarray,
    error_rate: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    steps, nodes = demand.shape
    values = np.zeros((steps, nodes, len(REQUIRED_NODE_FEATURES)), dtype=np.float32)
    feature = {name: index for index, name in enumerate(REQUIRED_NODE_FEATURES)}
    values[..., feature["request_rate"]] = demand
    values[..., feature["cpu_usage_cores"]] = cpu_usage
    values[..., feature["cpu_utilization"]] = cpu_utilization
    values[..., feature["mean_latency_ms"]] = p95_latency * 0.48
    values[..., feature["p95_latency_ms"]] = p95_latency
    values[..., feature["error_rate"]] = error_rate
    values[..., feature["desired_replicas"]] = desired
    values[..., feature["ready_replicas"]] = ready
    values[..., feature["pending_replicas"]] = pending
    values[..., feature["pending_age_seconds"]] = pending_age
    changed = np.concatenate((np.ones((1, nodes), dtype=bool), desired[1:] != desired[:-1]), axis=0)
    action_age = np.zeros((steps, nodes), dtype=np.float32)
    for node in range(nodes):
        elapsed = 600.0
        for step in range(steps):
            elapsed = 0.0 if changed[step, node] else min(600.0, elapsed + SAMPLING_INTERVAL_SECONDS)
            action_age[step, node] = elapsed
    values[..., feature["time_since_last_action_seconds"]] = action_age

    missing_probability = 0.035 if scenario in {"telemetry", "mixed"} else 0.004
    stale_probability = 0.045 if scenario in {"telemetry", "mixed"} else 0.006
    groups = (
        ("request_rate", "request_rate_missing", "request_rate_stale"),
        ("cpu_usage_cores", "cpu_missing", "cpu_stale"),
        ("p95_latency_ms", "latency_missing", "latency_stale"),
        ("error_rate", "error_missing", "error_stale"),
    )
    for value_name, missing_name, stale_name in groups:
        missing = rng.random((steps, nodes)) < missing_probability
        stale = (~missing) & (rng.random((steps, nodes)) < stale_probability)
        values[..., feature[missing_name]] = missing
        values[..., feature[stale_name]] = stale
        values[..., feature[value_name]][missing] = 0.0
        if value_name == "p95_latency_ms":
            values[..., feature["mean_latency_ms"]][missing] = 0.0
    scales = np.asarray([NODE_FEATURE_SCALES[name] for name in REQUIRED_NODE_FEATURES], dtype=np.float32)
    return values / scales[None, None, :]


def _edge_features(
    demand: np.ndarray,
    p95_latency: np.ndarray,
    error_rate: np.ndarray,
    placement: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, destination, ratios = _routing_arrays()
    steps = demand.shape[0]
    values = np.zeros((steps, len(source), len(REQUIRED_EDGE_FEATURES)), dtype=np.float32)
    feature = {name: index for index, name in enumerate(REQUIRED_EDGE_FEATURES)}
    values[..., feature["call_ratio"]] = ratios[None, :]
    values[..., feature["call_rate"]] = demand[:, source] * ratios[None, :]
    values[..., feature["latency_ms"]] = p95_latency[:, destination]
    values[..., feature["error_rate"]] = error_rate[:, destination]
    source_share = placement[:, source, 0] / np.maximum(placement[:, source].sum(axis=-1), 1.0)
    destination_share = placement[:, destination, 0] / np.maximum(placement[:, destination].sum(axis=-1), 1.0)
    values[..., feature["cross_cluster"]] = np.abs(source_share - destination_share) > 0.20
    missing_probability = 0.03 if scenario in {"telemetry", "mixed"} else 0.003
    stale_probability = 0.04 if scenario in {"telemetry", "mixed"} else 0.005
    missing = rng.random((steps, len(source))) < missing_probability
    stale = (~missing) & (rng.random((steps, len(source))) < stale_probability)
    values[..., feature["missing"]] = missing
    values[..., feature["stale"]] = stale
    values[missing, feature["call_rate"]] = 0.0
    values[missing, feature["call_ratio"]] = 0.0
    scales = np.asarray([EDGE_FEATURE_SCALES[name] for name in REQUIRED_EDGE_FEATURES], dtype=np.float32)
    return source, destination, values / scales[None, None, :]


def simulate_run(
    *,
    seed: int,
    split: str,
    scenario: str,
    workload: str,
    config: SyntheticConfig,
) -> dict[str, np.ndarray | str | int | float]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported synthetic scenario {scenario!r}")
    rng = np.random.default_rng(seed)
    paper_capacity = 4.0
    capacity = paper_capacity * rng.uniform(0.75, 2.75)
    if split == "holdout":
        capacity *= rng.uniform(1.10, 1.35)
    users = _workload_users(workload, config.steps_per_run, capacity, rng)
    demand = _propagated_demand(users, rng)
    per_replica_capacity = rng.uniform(0.55, 1.35, size=len(NODE_ORDER)).astype(np.float32)
    per_replica_capacity[[NODE_ORDER.index(name) for name in DATABASE_NODES]] *= 1.5
    desired, ready, pending, pending_age, placement = _replica_state(
        demand, per_replica_capacity, scenario, rng
    )
    cpu_usage, cpu_utilization, p95_latency, error_rate = _service_observations(
        demand, desired, ready, pending, pending_age, per_replica_capacity, scenario, rng
    )
    candidate, candidate_placement, readiness = _candidate_arrays(
        desired,
        placement,
        ready,
        pending_age,
        demand,
        per_replica_capacity,
        config.candidates_per_step,
        rng,
    )
    demand_target, latency_target, sla_target, reliability_target, boundary = _counterfactual_targets(
        demand,
        candidate,
        candidate_placement,
        readiness,
        per_replica_capacity,
        config.horizon_steps,
        config.sla_latency_ms,
    )
    node = _node_features(
        demand,
        desired,
        ready,
        pending,
        pending_age,
        cpu_usage,
        cpu_utilization,
        p95_latency,
        error_rate,
        scenario,
        rng,
    )
    edge_source, edge_destination, edge = _edge_features(
        demand, p95_latency, error_rate, placement, scenario, rng
    )
    first = config.lookback_steps - 1
    last = config.steps_per_run - max(config.horizon_steps)
    sample_indices = np.arange(first, last, dtype=np.int32)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "seed": seed,
        "split": split,
        "scenario": scenario,
        "workload": workload,
        "capacity_users": float(capacity),
        "node_features": node,
        "edge_source": edge_source,
        "edge_destination": edge_destination,
        "edge_features": edge,
        "candidate_replicas": candidate,
        "candidate_placement": candidate_placement,
        "readiness": readiness,
        "demand_target": demand_target,
        "latency_target": latency_target,
        "sla_target": sla_target,
        "reliability_target": reliability_target,
        "boundary_mask": boundary,
        "sample_indices": sample_indices,
    }


def _write_run(path: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    arrays = {key: value for key, value in run.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in run.items() if not isinstance(value, np.ndarray)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=canonical_json(metadata).decode("utf-8"), **arrays)
    return {
        **metadata,
        "path": str(path),
        "sha256": sha256_file(path),
        "samples": int(len(arrays["sample_indices"])),
        "boundary_labels": int(arrays["boundary_mask"][arrays["sample_indices"]].sum()),
        "sla_positive_labels": int(np.nansum(arrays["sla_target"][arrays["sample_indices"]])),
    }


def generate_dataset(
    output_dir: Path,
    *,
    config: SyntheticConfig,
    paper_summary: Path,
) -> dict[str, Any]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    priors = load_paper_priors(paper_summary)
    graph = build_adjacency()
    schema = feature_schema()
    network = model_config(config)
    for name, value in (("paper-priors.json", priors), ("adjacency.json", graph), ("feature-schema.json", schema), ("model-config.json", network)):
        (output_dir / name).write_bytes(canonical_json(value) + b"\n")

    counts = {
        "fitting": config.fitting_runs,
        "calibration": config.calibration_runs,
        "holdout": config.holdout_runs,
    }
    records: list[dict[str, Any]] = []
    ordinal = 0
    for split, count in counts.items():
        for split_index in range(count):
            seed = config.base_seed + ordinal
            scenario = SCENARIOS[split_index % len(SCENARIOS)]
            workload = WORKLOADS[(split_index // len(SCENARIOS) + split_index) % len(WORKLOADS)]
            run = simulate_run(
                seed=seed,
                split=split,
                scenario=scenario,
                workload=workload,
                config=config,
            )
            relative = Path("runs") / split / f"run-{seed}-{scenario}-{workload}.npz"
            record = _write_run(output_dir / relative, run)
            record["path"] = str(relative)
            records.append(record)
            ordinal += 1

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "engineering recovery benchmark; not research-v3 or research-v4 evaluation evidence",
        "config": {**asdict(config), "horizons_seconds": list(config.horizons_seconds)},
        "node_order": list(NODE_ORDER),
        "clusters": list(CLUSTERS),
        "endpoints": list(ENDPOINTS),
        "scenarios": list(SCENARIOS),
        "workloads": list(WORKLOADS),
        "paper_priors_sha256": sha256_file(output_dir / "paper-priors.json"),
        "feature_schema_sha256": sha256_file(output_dir / "feature-schema.json"),
        "adjacency_sha256": sha256_file(output_dir / "adjacency.json"),
        "model_config_sha256": sha256_file(output_dir / "model-config.json"),
        "runs": records,
        "run_count": len(records),
        "sample_count": sum(int(record["samples"]) for record in records),
        "split_sample_counts": {
            split: sum(int(record["samples"]) for record in records if record["split"] == split)
            for split in counts
        },
        "boundary_label_count": sum(int(record["boundary_labels"]) for record in records),
        "sla_positive_label_count": sum(int(record["sla_positive_labels"]) for record in records),
    }
    manifest_without_hash = dict(manifest)
    manifest["dataset_hash"] = hashlib.sha256(canonical_json(manifest_without_hash)).hexdigest()
    (output_dir / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported synthetic dataset schema")
    expected = dict(manifest)
    declared = expected.pop("dataset_hash", None)
    actual = hashlib.sha256(canonical_json(expected)).hexdigest()
    if declared != actual:
        raise ValueError(f"synthetic dataset hash mismatch: declared={declared} calculated={actual}")
    if int(manifest.get("run_count", -1)) != len(manifest.get("runs", [])):
        raise ValueError("synthetic manifest run_count does not match runs")
    return manifest


def validate_dataset_files(manifest_path: Path, *, verify_hashes: bool = True) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    samples = 0
    boundaries = 0
    positives = 0
    for record in manifest["runs"]:
        path = root / record["path"]
        if not path.is_file():
            raise ValueError(f"missing synthetic run file: {path}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"synthetic run hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as run:
            indices = run["sample_indices"]
            samples += len(indices)
            boundaries += int(run["boundary_mask"][indices].sum())
            positives += int(np.nansum(run["sla_target"][indices]))
            if not np.all(np.isfinite(run["node_features"])) or not np.all(np.isfinite(run["edge_features"])):
                raise ValueError(f"synthetic inputs must be finite: {path}")
            if run["candidate_replicas"].shape[-1] != len(NODE_ORDER):
                raise ValueError(f"synthetic candidate node dimension mismatch: {path}")
    observed = {
        "runs": len(manifest["runs"]),
        "samples": samples,
        "boundary_labels": boundaries,
        "sla_positive_labels": positives,
    }
    expected = {
        "runs": int(manifest["run_count"]),
        "samples": int(manifest["sample_count"]),
        "boundary_labels": int(manifest["boundary_label_count"]),
        "sla_positive_labels": int(manifest["sla_positive_label_count"]),
    }
    if observed != expected:
        raise ValueError(f"synthetic dataset accounting mismatch: expected={expected} observed={observed}")
    if samples < 10_000:
        raise ValueError("synthetic dataset is too small: at least 10,000 run-owned windows are required")
    if boundaries < 1_000 or positives < 1_000:
        raise ValueError("synthetic dataset lacks allocation-boundary or SLA-positive coverage")
    return observed


def split_records(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    records = [dict(record) for record in manifest["runs"] if record["split"] == split]
    if not records:
        raise ValueError(f"synthetic dataset contains no {split!r} runs")
    return records


def expand_sparse_edges(
    edge_features: np.ndarray,
    edge_source: np.ndarray,
    edge_destination: np.ndarray,
    *,
    node_count: int = len(NODE_ORDER),
) -> np.ndarray:
    """Expand `[batch,time,edge,feature]` to the v2 dense edge contract."""

    if edge_features.ndim != 4:
        raise ValueError("sparse edge batch must be [batch,time,edge,feature]")
    batch, steps, edge_count, feature_count = edge_features.shape
    if edge_count != len(edge_source) or edge_count != len(edge_destination):
        raise ValueError("sparse edge indices do not match edge features")
    dense = np.zeros((batch, steps, node_count, node_count, feature_count), dtype=np.float32)
    dense[:, :, edge_source, edge_destination, :] = edge_features
    return dense


def iter_window_batches(
    run: Mapping[str, np.ndarray],
    *,
    lookback_steps: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterable[dict[str, np.ndarray]]:
    indices = np.asarray(run["sample_indices"], dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(indices)
    source = run["edge_source"]
    destination = run["edge_destination"]
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        windows = selected[:, None] - np.arange(lookback_steps - 1, -1, -1)[None, :]
        sparse_edges = run["edge_features"][windows]
        yield {
            "node_features": run["node_features"][windows],
            "edge_features": expand_sparse_edges(sparse_edges, source, destination),
            "candidate_replicas": run["candidate_replicas"][selected],
            "candidate_placement": run["candidate_placement"][selected],
            "readiness": run["readiness"][selected],
            "demand_target": run["demand_target"][selected],
            "latency_target": run["latency_target"][selected],
            "sla_target": run["sla_target"][selected],
            "reliability_target": run["reliability_target"][selected],
            "boundary_mask": run["boundary_mask"][selected],
        }
