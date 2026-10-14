"""Deterministic closed-loop engineering benchmark for Headlight.

This module emulates workload, readiness, capacity, and telemetry dynamics. It
does not emulate Kubernetes, Karmada, HPA, or KEDA internals and its results are
not research-v4 evaluation evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .contracts_v2 import REQUIRED_EDGE_FEATURES, REQUIRED_NODE_FEATURES
from .synthetic_v2 import (
    CLUSTERS,
    DATABASE_NODES,
    EDGE_FEATURE_SCALES,
    ENDPOINTS,
    ENDPOINT_PATHS,
    NODE_FEATURE_SCALES,
    NODE_ORDER,
    ROUTING_EDGES,
    SAMPLING_INTERVAL_SECONDS,
    SLA_LATENCY_MS,
    _propagated_demand,
    _workload_users,
)


CLOSED_LOOP_SCHEMA_VERSION = "headlight.synthetic-closed-loop/v2"
CONTROLLER_IDS = (
    "headlight-v2",
    "headlight-v2-graph-free",
    "hpa-cpu-emulation",
    "keda-request-rate-emulation",
    "threshold-reactive",
    "oracle-bound",
)
PRIMARY_CONTROLLER_IDS = CONTROLLER_IDS[:-1]
CASE_NAMES = (
    "gradual-ramp",
    "flash-crowd",
    "diurnal",
    "capacity-readiness-failure",
)


@dataclass(frozen=True)
class ClosedLoopConfig:
    steps: int = 240
    lookback_steps: int = 24
    sampling_interval_seconds: int = SAMPLING_INTERVAL_SECONDS
    min_replicas: int = 2
    max_replicas: int = 14
    max_scale_out_step: int = 2
    max_scale_in_step: int = 1
    cooldown_steps: int = 4
    candidates_per_step: int = 8
    sla_latency_ms: float = SLA_LATENCY_MS
    normal_cluster_capacity: tuple[int, int] = (120, 120)
    constrained_cluster_capacity: tuple[int, int] = (58, 38)

    def validate(self) -> None:
        if self.steps <= self.lookback_steps + 12:
            raise ValueError("closed-loop episode is too short for the model lookback")
        if self.sampling_interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        if self.min_replicas < len(CLUSTERS) or self.max_replicas < self.min_replicas:
            raise ValueError("replica bounds must permit one pod in each cluster")
        if min(self.normal_cluster_capacity) < len(NODE_ORDER):
            raise ValueError("normal cluster capacity cannot fit minimum placements")
        if min(self.constrained_cluster_capacity) < len(NODE_ORDER):
            raise ValueError("constrained cluster capacity cannot fit minimum placements")
        if self.candidates_per_step != 8:
            raise ValueError("the trained Headlight checkpoint requires eight candidates")


@dataclass(frozen=True)
class ControllerAction:
    desired_replicas: np.ndarray
    placement: np.ndarray
    reason: str

    def validate(self) -> None:
        if self.desired_replicas.shape != (len(NODE_ORDER),):
            raise ValueError("action desired replicas must be [node]")
        if self.placement.shape != (len(NODE_ORDER), len(CLUSTERS)):
            raise ValueError("action placement must be [node,cluster]")
        if not np.all(np.isfinite(self.desired_replicas)) or not np.all(
            np.isfinite(self.placement)
        ):
            raise ValueError("controller actions must be finite")
        if not np.allclose(self.placement.sum(axis=-1), self.desired_replicas):
            raise ValueError("action placements must sum to desired replicas")


@dataclass(frozen=True)
class CandidateActions:
    actions: tuple[ControllerAction, ...]
    replicas: np.ndarray
    placements: np.ndarray
    readiness: np.ndarray


@dataclass(frozen=True)
class ClosedLoopObservation:
    step: int
    offered_rps: float
    successful_throughput_rps: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_count: int
    exception_count: int
    http_status_distribution: Mapping[str, int]
    successful_latency_ms: np.ndarray
    request_rate: np.ndarray
    cpu_usage_cores: np.ndarray
    cpu_utilization: np.ndarray
    service_p95_latency_ms: np.ndarray
    endpoint_p50_latency_ms: np.ndarray
    endpoint_p95_latency_ms: np.ndarray
    endpoint_p99_latency_ms: np.ndarray
    error_rate: np.ndarray
    desired_placement: np.ndarray
    ready_placement: np.ndarray
    pending_age_seconds: np.ndarray
    time_since_last_action_seconds: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray

    @property
    def desired_replicas(self) -> np.ndarray:
        return self.desired_placement.sum(axis=-1)

    @property
    def ready_replicas(self) -> np.ndarray:
        return self.ready_placement.sum(axis=-1)

    @property
    def pending_replicas(self) -> np.ndarray:
        return np.maximum(self.desired_replicas - self.ready_replicas, 0.0)


@dataclass(frozen=True)
class ControllerDecision:
    action: ControllerAction
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


Preview = Callable[[ControllerAction], Mapping[str, float]]


class ClosedLoopController(Protocol):
    controller_id: str

    def reset(self) -> None: ...

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision: ...


@dataclass(frozen=True)
class AppliedAction:
    requested: ControllerAction
    applied: ControllerAction
    replica_changes: int
    placement_changes: int
    migration_proxy: int
    rejected_replicas: int
    cooldown_holds: int


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _balanced_placement(totals: np.ndarray, skew: float = 0.5) -> np.ndarray:
    totals = np.asarray(np.rint(totals), dtype=np.int64)
    left = np.rint(totals * float(skew)).astype(np.int64)
    left = np.clip(left, 1, totals - 1)
    return np.stack((left, totals - left), axis=-1)


def hold_action(observation: ClosedLoopObservation, reason: str = "hold") -> ControllerAction:
    return ControllerAction(
        desired_replicas=observation.desired_replicas.astype(np.int64),
        placement=observation.desired_placement.astype(np.int64),
        reason=reason,
    )


class ClosedLoopEnvironment:
    """Stateful synthetic environment with controller-independent exogenous data."""

    def __init__(
        self,
        *,
        case: str,
        seed: int,
        config: ClosedLoopConfig | None = None,
    ) -> None:
        if case not in CASE_NAMES:
            raise ValueError(f"unsupported closed-loop case {case!r}")
        self.case = case
        self.seed = int(seed)
        self.config = config or ClosedLoopConfig()
        self.config.validate()
        self._build_exogenous_trace()
        self.reset()

    @property
    def constrained(self) -> bool:
        return self.case == "capacity-readiness-failure"

    @property
    def workload(self) -> str:
        return "flash-crowd" if self.constrained else self.case

    @property
    def trace_hash(self) -> str:
        return self._trace_hash

    @property
    def done(self) -> bool:
        return self.step_index >= self.config.steps - 1

    def _build_exogenous_trace(self) -> None:
        rng = np.random.default_rng(self.seed)
        capacity_users = 4.0 * rng.uniform(1.35, 2.75)
        offered = _workload_users(self.workload, self.config.steps, capacity_users, rng)
        demand = _propagated_demand(offered, rng)
        per_replica = rng.uniform(0.55, 1.35, size=len(NODE_ORDER)).astype(np.float32)
        database = [NODE_ORDER.index(name) for name in DATABASE_NODES]
        per_replica[database] *= 1.5
        base_latency = rng.uniform(12.0, 42.0, size=len(NODE_ORDER)).astype(np.float32)
        cpu_per_ready = rng.uniform(0.12, 0.30, size=len(NODE_ORDER)).astype(np.float32)
        latency_noise = rng.lognormal(
            0.0, 0.035, size=(self.config.steps, len(NODE_ORDER))
        ).astype(np.float32)
        cpu_noise = rng.normal(
            0.0, 0.025, size=(self.config.steps, len(NODE_ORDER))
        ).astype(np.float32)
        startup_steps = rng.integers(2, 7, size=(len(NODE_ORDER), len(CLUSTERS)))
        if self.constrained:
            startup_steps[:, 1] = np.ceil(startup_steps[:, 1] * 2.5).astype(np.int64)
        node_missing = rng.random((self.config.steps, len(NODE_ORDER), 4)) < 0.004
        node_stale = (~node_missing) & (
            rng.random((self.config.steps, len(NODE_ORDER), 4)) < 0.006
        )
        edge_missing = rng.random((self.config.steps, len(ROUTING_EDGES))) < 0.003
        edge_stale = (~edge_missing) & (
            rng.random((self.config.steps, len(ROUTING_EDGES))) < 0.005
        )
        self.offered_trace = offered.astype(np.float32)
        self.demand_trace = demand.astype(np.float32)
        self.per_replica_capacity = per_replica
        self.base_latency = base_latency
        self.cpu_per_ready = cpu_per_ready
        self.latency_noise = latency_noise
        self.cpu_noise = cpu_noise
        self.startup_steps = startup_steps.astype(np.int64)
        self.node_missing = node_missing
        self.node_stale = node_stale
        self.edge_missing = edge_missing
        self.edge_stale = edge_stale
        digest = hashlib.sha256()
        for value in (
            self.offered_trace,
            self.demand_trace,
            self.per_replica_capacity,
            self.startup_steps,
            self.node_missing,
            self.node_stale,
            self.edge_missing,
            self.edge_stale,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(_canonical_json({"case": self.case, "seed": self.seed}))
        self._trace_hash = digest.hexdigest()

    def reset(self) -> ClosedLoopObservation:
        self.step_index = 0
        self.desired_placement = np.ones(
            (len(NODE_ORDER), len(CLUSTERS)), dtype=np.int64
        )
        self.ready_placement = self.desired_placement.copy()
        self.pending_age = np.zeros_like(self.desired_placement, dtype=np.float32)
        self.steps_since_action = np.full(len(NODE_ORDER), 10_000, dtype=np.int64)
        self.last_applied: AppliedAction | None = None
        return self.observe()

    def _cluster_capacities(self) -> np.ndarray:
        values = (
            self.config.constrained_cluster_capacity
            if self.constrained
            else self.config.normal_cluster_capacity
        )
        return np.asarray(values, dtype=np.int64)

    def _bounded_action(self, requested: ControllerAction) -> AppliedAction:
        requested.validate()
        current_total = self.desired_placement.sum(axis=-1)
        requested_total = np.rint(requested.desired_replicas).astype(np.int64)
        requested_total = np.clip(
            requested_total, self.config.min_replicas, self.config.max_replicas
        )
        database = np.asarray([name in DATABASE_NODES for name in NODE_ORDER])
        requested_total[database] = self.config.min_replicas
        delta = requested_total - current_total
        delta = np.clip(
            delta, -self.config.max_scale_in_step, self.config.max_scale_out_step
        )
        requested_placement = np.maximum(
            np.rint(requested.placement).astype(np.int64), 0
        )
        placement_change_requested = np.any(
            requested_placement != self.desired_placement, axis=-1
        )
        cooldown = (self.steps_since_action < self.config.cooldown_steps) & (
            (delta != 0) | placement_change_requested
        )
        delta[cooldown] = 0
        totals = current_total + delta

        placement = requested_placement.copy()
        for node in range(len(NODE_ORDER)):
            if totals[node] == current_total[node] and cooldown[node]:
                placement[node] = self.desired_placement[node]
                continue
            raw = placement[node]
            if raw.sum() <= 0:
                raw = self.desired_placement[node]
            share = float(raw[0]) / max(1.0, float(raw.sum()))
            placement[node] = _balanced_placement(
                np.asarray([totals[node]]), share
            )[0]

        rejected = int(np.maximum(requested_total - totals, 0).sum())
        capacities = self._cluster_capacities()
        for cluster in range(len(CLUSTERS)):
            overflow = int(max(0, placement[:, cluster].sum() - capacities[cluster]))
            other = 1 - cluster
            while overflow > 0:
                movable = [
                    node
                    for node in np.argsort(-placement[:, cluster])
                    if placement[node, cluster] > 1
                    and placement[:, other].sum() < capacities[other]
                ]
                if movable:
                    node = int(movable[0])
                    placement[node, cluster] -= 1
                    placement[node, other] += 1
                    overflow -= 1
                    continue
                reducible = [
                    node
                    for node in np.argsort(-placement[:, cluster])
                    if placement[node].sum() > self.config.min_replicas
                    and NODE_ORDER[int(node)] not in DATABASE_NODES
                ]
                if not reducible:
                    break
                node = int(reducible[0])
                placement[node, cluster] -= 1
                totals[node] -= 1
                rejected += 1
                overflow -= 1

        applied = ControllerAction(
            desired_replicas=placement.sum(axis=-1).astype(np.int64),
            placement=placement.astype(np.int64),
            reason=requested.reason,
        )
        placement_l1 = np.abs(placement - self.desired_placement).sum(axis=-1)
        total_l1 = np.abs(applied.desired_replicas - current_total)
        migrations = int(np.maximum(0, (placement_l1 - total_l1) // 2).sum())
        return AppliedAction(
            requested=requested,
            applied=applied,
            replica_changes=int(total_l1.sum()),
            placement_changes=int(placement_l1.sum()),
            migration_proxy=migrations,
            rejected_replicas=rejected,
            cooldown_holds=int(cooldown.sum()),
        )

    def _advance_state(self, applied: AppliedAction) -> None:
        previous = self.desired_placement.copy()
        self.desired_placement = applied.applied.placement.copy()
        changed = np.any(self.desired_placement != previous, axis=-1)
        self.steps_since_action += 1
        self.steps_since_action[changed] = 0
        self.ready_placement = np.minimum(
            self.ready_placement, self.desired_placement
        )
        pending = self.desired_placement - self.ready_placement
        self.pending_age[pending <= 0] = 0.0
        self.pending_age[pending > 0] += self.config.sampling_interval_seconds
        ready_now = (pending > 0) & (
            self.pending_age
            >= self.startup_steps * self.config.sampling_interval_seconds
        )
        self.ready_placement[ready_now] = self.desired_placement[ready_now]
        self.pending_age[ready_now] = 0.0

    def step(self, action: ControllerAction) -> tuple[ClosedLoopObservation, AppliedAction]:
        if self.done:
            raise RuntimeError("cannot step a completed closed-loop episode")
        applied = self._bounded_action(action)
        self._advance_state(applied)
        self.step_index += 1
        self.last_applied = applied
        return self.observe(), applied

    def _raw_metrics(
        self,
        step: int,
        desired: np.ndarray,
        ready: np.ndarray,
        pending_age: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        demand = self.demand_trace[step]
        ready_total = ready.sum(axis=-1).astype(np.float32)
        desired_total = desired.sum(axis=-1).astype(np.float32)
        utilization = demand / np.maximum(
            ready_total * self.per_replica_capacity, 0.1
        )
        queue = np.square(np.maximum(utilization, 0.0)) / np.maximum(
            0.06, 1.12 - utilization
        )
        service_p95 = self.base_latency + 62.0 * queue
        cross_cluster = np.minimum(ready[:, 0], ready[:, 1]) > 0
        service_p95 += cross_cluster.astype(np.float32) * 5.0
        readiness = ready_total / np.maximum(desired_total, 1.0)
        service_p95 += (1.0 - readiness) * 180.0
        service_p95 *= self.latency_noise[step]
        service_p95 = np.clip(service_p95, 4.0, 5000.0).astype(np.float32)
        cpu_utilization = np.clip(
            0.04 + 0.78 * utilization + self.cpu_noise[step], 0.0, 1.5
        ).astype(np.float32)
        cpu_usage = (
            cpu_utilization * np.maximum(ready_total, 1.0) * self.cpu_per_ready
        ).astype(np.float32)
        error = np.clip(
            0.001 + 0.18 * np.square(np.maximum(utilization - 0.85, 0.0)),
            0.0,
            1.0,
        ).astype(np.float32)
        index = {name: offset for offset, name in enumerate(NODE_ORDER)}
        endpoint_p95 = []
        endpoint_error = []
        for endpoint in ENDPOINTS:
            path = [index[name] for name in ENDPOINT_PATHS[endpoint]]
            endpoint_p95.append(35.0 + float(service_p95[path].sum()))
            endpoint_error.append(float(error[path].max()))
        endpoint_p95_values = np.asarray(endpoint_p95, dtype=np.float32)
        endpoint_p50_values = np.maximum(5.0, endpoint_p95_values * 0.48)
        normal = statistics.NormalDist()
        sigma = np.log(endpoint_p95_values / endpoint_p50_values) / normal.inv_cdf(0.95)
        endpoint_p99_values = endpoint_p50_values * np.exp(
            normal.inv_cdf(0.99) * sigma
        )
        request_metrics = self._synthetic_request_metrics(
            offered_rps=float(self.offered_trace[step]),
            endpoint_p50=endpoint_p50_values,
            endpoint_p95=endpoint_p95_values,
            endpoint_error=np.asarray(endpoint_error, dtype=np.float32),
        )
        return {
            "request_rate": demand,
            "cpu_usage": cpu_usage,
            "cpu_utilization": cpu_utilization,
            "service_p95": service_p95,
            "error": error,
            "endpoint_p50": endpoint_p50_values.astype(np.float32),
            "endpoint_p95": endpoint_p95_values,
            "endpoint_p99": endpoint_p99_values.astype(np.float32),
            **request_metrics,
            "pending_age": pending_age.max(axis=-1),
        }

    def _synthetic_request_metrics(
        self,
        *,
        offered_rps: float,
        endpoint_p50: np.ndarray,
        endpoint_p95: np.ndarray,
        endpoint_error: np.ndarray,
    ) -> dict[str, Any]:
        """Materialize deterministic request outcomes from aggregate simulator state.

        These are synthetic request records, not HTTP traffic. Endpoint shares and
        the latency distribution are fixed parts of the v2 measurement contract.
        """

        interval = self.config.sampling_interval_seconds
        total = max(0, int(round(offered_rps * interval)))
        shares = np.asarray((0.20, 0.50, 0.30), dtype=np.float64)
        exact = shares * total
        endpoint_totals = np.floor(exact).astype(np.int64)
        remainder = total - int(endpoint_totals.sum())
        for endpoint in np.argsort(-(exact - endpoint_totals))[:remainder]:
            endpoint_totals[int(endpoint)] += 1

        successful_latencies: list[float] = []
        successful = 0
        timeouts = 0
        exceptions = 0
        normal = statistics.NormalDist()
        for endpoint, endpoint_total in enumerate(endpoint_totals):
            failed = min(
                int(endpoint_total),
                max(0, int(round(float(endpoint_total) * float(endpoint_error[endpoint])))),
            )
            overload = float(
                np.clip(
                    endpoint_p95[endpoint] / self.config.sla_latency_ms - 1.0,
                    0.0,
                    1.0,
                )
            )
            endpoint_timeouts = min(failed, int(round(failed * overload)))
            timeouts += endpoint_timeouts
            exceptions += failed - endpoint_timeouts
            endpoint_successful = int(endpoint_total) - failed
            successful += endpoint_successful
            if endpoint_successful <= 0:
                continue
            sigma = math.log(
                float(endpoint_p95[endpoint]) / float(endpoint_p50[endpoint])
            ) / normal.inv_cdf(0.95)
            for request in range(endpoint_successful):
                quantile = min(0.999, max(0.001, (request + 0.5) / endpoint_successful))
                successful_latencies.append(
                    float(endpoint_p50[endpoint])
                    * math.exp(sigma * normal.inv_cdf(quantile))
                )

        failed = total - successful
        statuses = {
            "200": successful,
            "500": exceptions,
            "504": timeouts,
        }
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "timeout_count": timeouts,
            "exception_count": exceptions,
            "http_status_distribution": statuses,
            "successful_latency_ms": np.asarray(successful_latencies, dtype=np.float32),
            "successful_throughput": successful / interval,
        }

    def _node_features(self, raw: Mapping[str, Any]) -> np.ndarray:
        feature = {name: index for index, name in enumerate(REQUIRED_NODE_FEATURES)}
        values = np.zeros(
            (len(NODE_ORDER), len(REQUIRED_NODE_FEATURES)), dtype=np.float32
        )
        desired = self.desired_placement.sum(axis=-1)
        ready = self.ready_placement.sum(axis=-1)
        pending = np.maximum(desired - ready, 0)
        values[:, feature["request_rate"]] = raw["request_rate"]
        values[:, feature["cpu_usage_cores"]] = raw["cpu_usage"]
        values[:, feature["cpu_utilization"]] = raw["cpu_utilization"]
        values[:, feature["mean_latency_ms"]] = raw["service_p95"] * 0.48
        values[:, feature["p95_latency_ms"]] = raw["service_p95"]
        values[:, feature["error_rate"]] = raw["error"]
        values[:, feature["desired_replicas"]] = desired
        values[:, feature["ready_replicas"]] = ready
        values[:, feature["pending_replicas"]] = pending
        values[:, feature["pending_age_seconds"]] = raw["pending_age"]
        values[:, feature["time_since_last_action_seconds"]] = np.minimum(
            600.0,
            self.steps_since_action * self.config.sampling_interval_seconds,
        )
        groups = (
            ("request_rate", "request_rate_missing", "request_rate_stale"),
            ("cpu_usage_cores", "cpu_missing", "cpu_stale"),
            ("p95_latency_ms", "latency_missing", "latency_stale"),
            ("error_rate", "error_missing", "error_stale"),
        )
        for group, (value_name, missing_name, stale_name) in enumerate(groups):
            missing = self.node_missing[self.step_index, :, group]
            stale = self.node_stale[self.step_index, :, group]
            values[:, feature[missing_name]] = missing
            values[:, feature[stale_name]] = stale
            values[missing, feature[value_name]] = 0.0
            if value_name == "p95_latency_ms":
                values[missing, feature["mean_latency_ms"]] = 0.0
        scales = np.asarray(
            [NODE_FEATURE_SCALES[name] for name in REQUIRED_NODE_FEATURES],
            dtype=np.float32,
        )
        return values / scales[None, :]

    def _edge_features(self, raw: Mapping[str, Any]) -> np.ndarray:
        values = np.zeros(
            (
                len(NODE_ORDER),
                len(NODE_ORDER),
                len(REQUIRED_EDGE_FEATURES),
            ),
            dtype=np.float32,
        )
        feature = {name: index for index, name in enumerate(REQUIRED_EDGE_FEATURES)}
        index = {name: offset for offset, name in enumerate(NODE_ORDER)}
        desired = np.maximum(self.desired_placement.sum(axis=-1), 1)
        shares = self.desired_placement[:, 0] / desired
        for edge, (source_name, destination_name, ratio) in enumerate(ROUTING_EDGES):
            source = index[source_name]
            destination = index[destination_name]
            missing = bool(self.edge_missing[self.step_index, edge])
            stale = bool(self.edge_stale[self.step_index, edge])
            values[source, destination, feature["call_rate"]] = (
                0.0 if missing else float(raw["request_rate"][source]) * ratio
            )
            values[source, destination, feature["call_ratio"]] = 0.0 if missing else ratio
            values[source, destination, feature["latency_ms"]] = raw["service_p95"][destination]
            values[source, destination, feature["error_rate"]] = raw["error"][destination]
            values[source, destination, feature["cross_cluster"]] = (
                abs(float(shares[source] - shares[destination])) > 0.20
            )
            values[source, destination, feature["missing"]] = missing
            values[source, destination, feature["stale"]] = stale
        scales = np.asarray(
            [EDGE_FEATURE_SCALES[name] for name in REQUIRED_EDGE_FEATURES],
            dtype=np.float32,
        )
        return values / scales[None, None, :]

    def observe(self) -> ClosedLoopObservation:
        raw = self._raw_metrics(
            self.step_index,
            self.desired_placement,
            self.ready_placement,
            self.pending_age,
        )
        return ClosedLoopObservation(
            step=self.step_index,
            offered_rps=float(self.offered_trace[self.step_index]),
            successful_throughput_rps=float(raw["successful_throughput"]),
            total_requests=int(raw["total_requests"]),
            successful_requests=int(raw["successful_requests"]),
            failed_requests=int(raw["failed_requests"]),
            timeout_count=int(raw["timeout_count"]),
            exception_count=int(raw["exception_count"]),
            http_status_distribution=dict(raw["http_status_distribution"]),
            successful_latency_ms=np.asarray(
                raw["successful_latency_ms"], dtype=np.float32
            ),
            request_rate=np.asarray(raw["request_rate"], dtype=np.float32),
            cpu_usage_cores=np.asarray(raw["cpu_usage"], dtype=np.float32),
            cpu_utilization=np.asarray(raw["cpu_utilization"], dtype=np.float32),
            service_p95_latency_ms=np.asarray(raw["service_p95"], dtype=np.float32),
            endpoint_p50_latency_ms=np.asarray(raw["endpoint_p50"], dtype=np.float32),
            endpoint_p95_latency_ms=np.asarray(raw["endpoint_p95"], dtype=np.float32),
            endpoint_p99_latency_ms=np.asarray(raw["endpoint_p99"], dtype=np.float32),
            error_rate=np.asarray(raw["error"], dtype=np.float32),
            desired_placement=self.desired_placement.copy(),
            ready_placement=self.ready_placement.copy(),
            pending_age_seconds=np.asarray(raw["pending_age"], dtype=np.float32),
            time_since_last_action_seconds=np.minimum(
                600.0,
                self.steps_since_action * self.config.sampling_interval_seconds,
            ).astype(np.float32),
            node_features=self._node_features(raw),
            edge_features=self._edge_features(raw),
        )

    def candidates(self, observation: ClosedLoopObservation) -> CandidateActions:
        current = observation.desired_replicas.astype(np.float32)
        ready = np.maximum(observation.ready_replicas, 1.0)
        utilization = np.maximum(observation.cpu_utilization - 0.04, 0.05) / 0.78
        capacity_estimate = observation.request_rate / np.maximum(
            ready * utilization, 0.1
        )
        target = np.ceil(
            observation.request_rate / np.maximum(0.70 * capacity_estimate, 0.1)
        )
        target = np.clip(
            target, self.config.min_replicas, self.config.max_replicas
        )
        database = np.asarray([name in DATABASE_NODES for name in NODE_ORDER])
        target[database] = self.config.min_replicas
        offsets = np.arange(-3, 5, dtype=np.int64)
        actions = []
        replicas = np.empty(
            (self.config.candidates_per_step, len(NODE_ORDER)), dtype=np.float32
        )
        placements = np.empty(
            (
                self.config.candidates_per_step,
                len(NODE_ORDER),
                len(CLUSTERS),
            ),
            dtype=np.float32,
        )
        readiness = np.empty(
            (self.config.candidates_per_step, len(NODE_ORDER), 2), dtype=np.float32
        )
        for candidate, offset in enumerate(offsets):
            if candidate == 0:
                total = current.copy()
                placement = observation.desired_placement.copy()
            elif candidate == len(offsets) - 1:
                total = np.full(len(NODE_ORDER), self.config.max_replicas, dtype=np.float32)
                total[database] = self.config.min_replicas
                current_total = np.maximum(observation.desired_placement.sum(axis=-1), 1)
                share = observation.desired_placement[:, 0] / current_total
                left = np.clip(np.rint(total * share), 1, total - 1)
                placement = np.stack((left, total - left), axis=-1)
            else:
                total = np.rint(0.65 * target + 0.35 * current + offset)
                total = np.clip(
                    total, self.config.min_replicas, self.config.max_replicas
                )
                total[database] = self.config.min_replicas
                current_total = np.maximum(observation.desired_placement.sum(axis=-1), 1)
                share = observation.desired_placement[:, 0] / current_total
                left = np.clip(np.rint(total * share), 1, total - 1)
                placement = np.stack((left, total - left), axis=-1)
            replicas[candidate] = total
            placements[candidate] = placement
            ready_fraction = np.minimum(observation.ready_replicas, total) / np.maximum(
                total, 1.0
            )
            readiness[candidate, :, 0] = ready_fraction
            readiness[candidate, :, 1] = np.where(
                ready_fraction < 1.0, observation.pending_age_seconds, 0.0
            )
            actions.append(
                ControllerAction(
                    desired_replicas=total.astype(np.int64),
                    placement=placement.astype(np.int64),
                    reason=f"candidate-{candidate}",
                )
            )
        return CandidateActions(tuple(actions), replicas, placements, readiness)

    def preview(self, action: ControllerAction) -> Mapping[str, float]:
        clone = copy.deepcopy(self)
        observation, applied = clone.step(action)
        cumulative_severity = 0.0
        horizon = min(
            int(np.max(self.startup_steps)) + 1,
            self.config.steps - self.step_index - 1,
        )
        for _ in range(1, horizon):
            cumulative_severity += float(
                np.maximum(
                    observation.endpoint_p95_latency_ms
                    / self.config.sla_latency_ms
                    - 1.0,
                    0.0,
                ).sum()
            )
            observation, _ = clone.step(hold_action(observation, "oracle preview hold"))
        severity = float(
            np.maximum(
                observation.endpoint_p95_latency_ms / self.config.sla_latency_ms - 1.0,
                0.0,
            ).sum()
        )
        return {
            "sla_violations": float(
                (observation.endpoint_p95_latency_ms > self.config.sla_latency_ms).sum()
            ),
            "sla_severity": severity,
            "cumulative_sla_severity": cumulative_severity + severity,
            "replicas": float(observation.desired_replicas.mean()),
            "pending": float(observation.pending_replicas.sum()),
            "rejected_replicas": float(applied.rejected_replicas),
        }


class HpaCpuEmulation:
    controller_id = "hpa-cpu-emulation"

    def __init__(self, target_utilization: float = 0.70) -> None:
        self.target_utilization = float(target_utilization)

    def reset(self) -> None:
        self.recommendations: list[np.ndarray] = []

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision:
        current = observation.desired_replicas
        ratio = observation.cpu_utilization / self.target_utilization
        recommendation = np.ceil(current * ratio)
        recommendation[(ratio >= 0.90) & (ratio <= 1.10)] = current[
            (ratio >= 0.90) & (ratio <= 1.10)
        ]
        recommendation = np.clip(recommendation, 2, 14)
        recommendation[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        self.recommendations.append(recommendation.copy())
        self.recommendations = self.recommendations[-20:]
        down = recommendation < current
        if np.any(down):
            stabilized = np.max(np.stack(self.recommendations), axis=0)
            recommendation[down] = stabilized[down]
        action = ControllerAction(
            recommendation.astype(np.int64),
            _balanced_placement(recommendation, 0.5),
            "HPA CPU formula with 300-second downscale stabilization",
        )
        return ControllerDecision(action)


class KedaRequestRateEmulation:
    controller_id = "keda-request-rate-emulation"

    def __init__(self, target_rps_per_replica: float = 0.80) -> None:
        self.target = float(target_rps_per_replica)

    def reset(self) -> None:
        self.inactive_steps = 0

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision:
        current = observation.desired_replicas
        active = observation.request_rate >= 0.10
        self.inactive_steps = 0 if np.any(active) else self.inactive_steps + 1
        recommendation = np.ceil(observation.request_rate / self.target)
        recommendation = np.clip(recommendation, 2, 14)
        if self.inactive_steps < 20:
            recommendation = np.maximum(recommendation, current - 1)
        recommendation[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        action = ControllerAction(
            recommendation.astype(np.int64),
            _balanced_placement(recommendation, 0.5),
            "KEDA request-rate trigger with 300-second cooldown",
        )
        return ControllerDecision(action)


class ThresholdReactiveController:
    controller_id = "threshold-reactive"

    def reset(self) -> None:
        self.healthy_windows = np.zeros(len(NODE_ORDER), dtype=np.int64)

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision:
        current = observation.desired_replicas
        emergency = (
            (observation.cpu_utilization >= 0.75)
            | (observation.service_p95_latency_ms >= 1000.0)
            | (observation.error_rate >= 0.05)
        )
        healthy = (
            (observation.cpu_utilization <= 0.35)
            & (observation.service_p95_latency_ms <= 800.0)
            & (observation.error_rate <= 0.01)
            & (observation.pending_replicas == 0)
        )
        self.healthy_windows = np.where(healthy, self.healthy_windows + 1, 0)
        target = current.copy()
        target[emergency] += 2
        target[(~emergency) & (self.healthy_windows >= 3)] -= 1
        target = np.clip(target, 2, 14)
        target[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        return ControllerDecision(
            ControllerAction(
                target.astype(np.int64),
                _balanced_placement(target, 0.5),
                "reactive CPU/latency/error thresholds",
            )
        )


class OracleBoundController:
    controller_id = "oracle-bound"

    def reset(self) -> None:
        return None

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision:
        scored = []
        for index, action in enumerate(candidates.actions):
            outcome = preview(action)
            score = (
                outcome["sla_violations"],
                outcome["sla_severity"],
                outcome["cumulative_sla_severity"],
                outcome["pending"],
                outcome["replicas"],
                index,
            )
            scored.append((score, action))
        selected = min(scored, key=lambda item: item[0])
        return ControllerDecision(
            ControllerAction(
                selected[1].desired_replicas,
                selected[1].placement,
                "one-step simulator oracle; sanity bound only",
            ),
            diagnostics={"oracle_score": list(selected[0])},
        )


class EpisodeAccumulator:
    def __init__(
        self,
        *,
        controller_id: str,
        case: str,
        seed: int,
        trace_hash: str,
        config: ClosedLoopConfig,
    ) -> None:
        self.controller_id = controller_id
        self.case = case
        self.seed = seed
        self.trace_hash = trace_hash
        self.config = config
        self.samples = 0
        self.endpoint_samples = 0
        self.sla_violations = 0
        self.sla_severity = 0.0
        self.violation_run = 0
        self.longest_violation_run = 0
        self.p95_values: list[float] = []
        self.successful_latency_values: list[float] = []
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeout_count = 0
        self.exception_count = 0
        self.http_status_distribution: dict[str, int] = {}
        self.successful_throughput = 0.0
        self.offered_throughput = 0.0
        self.cpu_seconds = 0.0
        self.pod_seconds = 0.0
        self.desired_replica_samples_by_cluster = np.zeros(len(CLUSTERS), dtype=np.float64)
        self.ready_replica_samples_by_cluster = np.zeros(len(CLUSTERS), dtype=np.float64)
        self.placement_samples_by_cluster = np.zeros(len(CLUSTERS), dtype=np.float64)
        self.final_desired_placement = np.zeros(
            (len(NODE_ORDER), len(CLUSTERS)), dtype=np.int64
        )
        self.final_ready_placement = self.final_desired_placement.copy()
        self.pending_pod_seconds = 0.0
        self.readiness_disruption_samples = 0
        self.ceiling_samples = 0
        self.replica_changes = 0
        self.action_churn = 0
        self.placement_changes = 0
        self.migration_proxy = 0
        self.rejected_replicas = 0
        self.cooldown_holds = 0
        self.decision_seconds: list[float] = []
        self.forecast_absolute = 0.0
        self.forecast_count = 0
        self.model_decisions = 0
        self.guard_modified_decisions = 0
        self.guard_modified_services = 0
        self.hysteresis_modified_services = 0
        self.shadow_decisions = 0
        self.shadow_disagreements = 0
        self.shadow_probability_delta = 0.0
        self.graph_edge_sensitivity = 0.0
        self.graph_edge_sensitivity_count = 0
        self.selected_candidate_counts: dict[str, int] = {}
        self.objective_component_totals = {
            "resource": 0.0,
            "migration": 0.0,
            "reliability": 0.0,
            "objective": 0.0,
        }

    def update(
        self,
        observation: ClosedLoopObservation,
        applied: AppliedAction,
        decision_seconds: float,
        diagnostics: Mapping[str, Any],
        demand_trace: np.ndarray,
    ) -> None:
        if observation.step < self.config.lookback_steps:
            return
        self.samples += 1
        latency = observation.endpoint_p95_latency_ms
        violated = latency > self.config.sla_latency_ms
        self.endpoint_samples += latency.size
        self.sla_violations += int(violated.sum())
        self.sla_severity += float(
            np.maximum(latency / self.config.sla_latency_ms - 1.0, 0.0).sum()
        )
        if np.any(violated):
            self.violation_run += 1
            self.longest_violation_run = max(
                self.longest_violation_run, self.violation_run
            )
        else:
            self.violation_run = 0
        self.p95_values.extend(float(value) for value in latency)
        self.successful_latency_values.extend(
            float(value) for value in observation.successful_latency_ms
        )
        self.total_requests += observation.total_requests
        self.successful_requests += observation.successful_requests
        self.failed_requests += observation.failed_requests
        self.timeout_count += observation.timeout_count
        self.exception_count += observation.exception_count
        for status, count in observation.http_status_distribution.items():
            self.http_status_distribution[status] = (
                self.http_status_distribution.get(status, 0) + int(count)
            )
        self.successful_throughput += observation.successful_throughput_rps
        self.offered_throughput += observation.offered_rps
        interval = self.config.sampling_interval_seconds
        self.cpu_seconds += float(observation.cpu_usage_cores.sum()) * interval
        self.pod_seconds += float(observation.ready_replicas.sum()) * interval
        self.desired_replica_samples_by_cluster += observation.desired_placement.sum(
            axis=0
        )
        self.ready_replica_samples_by_cluster += observation.ready_placement.sum(axis=0)
        self.placement_samples_by_cluster += (observation.desired_placement > 0).sum(
            axis=0
        )
        self.final_desired_placement = observation.desired_placement.copy()
        self.final_ready_placement = observation.ready_placement.copy()
        self.pending_pod_seconds += float(observation.pending_replicas.sum()) * interval
        self.readiness_disruption_samples += int(np.any(observation.pending_replicas > 0))
        ceiling_fraction = np.mean(
            observation.desired_replicas >= self.config.max_replicas
        )
        self.ceiling_samples += int(ceiling_fraction >= 0.25)
        self.replica_changes += applied.replica_changes
        self.action_churn += int(applied.replica_changes > 0)
        self.placement_changes += applied.placement_changes
        self.migration_proxy += applied.migration_proxy
        self.rejected_replicas += applied.rejected_replicas
        self.cooldown_holds += applied.cooldown_holds
        self.decision_seconds.append(decision_seconds)
        if "selected_candidate" in diagnostics:
            self.model_decisions += 1
            candidate = str(int(diagnostics["selected_candidate"]))
            self.selected_candidate_counts[candidate] = (
                self.selected_candidate_counts.get(candidate, 0) + 1
            )
            guard_services = int(diagnostics.get("reactive_guard_services", 0))
            self.guard_modified_decisions += int(guard_services > 0)
            self.guard_modified_services += guard_services
            self.hysteresis_modified_services += int(
                diagnostics.get("scale_in_hysteresis_services", 0)
            )
            for output_name, diagnostic_name in (
                ("resource", "selected_resource_term"),
                ("migration", "selected_churn_cost"),
                ("reliability", "selected_reliability_penalty"),
                ("objective", "selected_objective"),
            ):
                value = diagnostics.get(diagnostic_name)
                if value is not None:
                    self.objective_component_totals[output_name] += float(value)
            shadow_selected = diagnostics.get("shadow_selected_candidate")
            if shadow_selected is not None:
                self.shadow_decisions += 1
                self.shadow_disagreements += int(bool(diagnostics.get("shadow_disagrees")))
                self.shadow_probability_delta += float(
                    diagnostics.get("shadow_sla_probability_mae") or 0.0
                )
            sensitivity = diagnostics.get("graph_edge_sensitivity")
            if sensitivity is not None:
                self.graph_edge_sensitivity += float(sensitivity)
                self.graph_edge_sensitivity_count += 1
        forecast = diagnostics.get("demand_q50")
        horizons = diagnostics.get("horizon_steps")
        origin = diagnostics.get("forecast_origin_step")
        if forecast is not None and horizons is not None and origin is not None:
            values = np.asarray(forecast, dtype=np.float32)
            for horizon_index, horizon in enumerate(horizons):
                target_step = int(origin) + int(horizon)
                if target_step < len(demand_trace):
                    self.forecast_absolute += float(
                        np.abs(values[:, horizon_index] - demand_trace[target_step]).sum()
                    )
                    self.forecast_count += values.shape[0]

    def result(self) -> dict[str, Any]:
        decisions_ms = np.asarray(self.decision_seconds, dtype=np.float64) * 1000.0
        request_latencies = np.asarray(
            self.successful_latency_values, dtype=np.float64
        )
        measured_seconds = self.samples * self.config.sampling_interval_seconds
        request_sla_violations = int(
            (request_latencies > self.config.sla_latency_ms).sum()
        )
        excess_response_time_seconds = float(
            np.maximum(request_latencies - self.config.sla_latency_ms, 0.0).sum()
            / 1000.0
        )
        final_service_placement = {
            service: {
                cluster: int(self.final_desired_placement[node, cluster_index])
                for cluster_index, cluster in enumerate(CLUSTERS)
            }
            for node, service in enumerate(NODE_ORDER)
        }
        cluster_metrics: dict[str, float | int] = {}
        for cluster_index, cluster in enumerate(CLUSTERS):
            cluster_metrics.update(
                {
                    f"mean_desired_replicas_{cluster}": float(
                        self.desired_replica_samples_by_cluster[cluster_index]
                        / max(1, self.samples)
                    ),
                    f"mean_ready_replicas_{cluster}": float(
                        self.ready_replica_samples_by_cluster[cluster_index]
                        / max(1, self.samples)
                    ),
                    f"mean_placement_count_{cluster}": float(
                        self.placement_samples_by_cluster[cluster_index]
                        / max(1, self.samples)
                    ),
                    f"final_desired_replicas_{cluster}": int(
                        self.final_desired_placement[:, cluster_index].sum()
                    ),
                    f"final_ready_replicas_{cluster}": int(
                        self.final_ready_placement[:, cluster_index].sum()
                    ),
                    f"final_placement_count_{cluster}": int(
                        (self.final_desired_placement[:, cluster_index] > 0).sum()
                    ),
                }
            )
        return {
            "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
            "controller": self.controller_id,
            "case": self.case,
            "seed": self.seed,
            "trace_hash": self.trace_hash,
            "failed": False,
            "samples": self.samples,
            "measurement_seconds": measured_seconds,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "timeout_count": self.timeout_count,
            "exception_count": self.exception_count,
            "http_200_count": self.http_status_distribution.get("200", 0),
            "http_500_count": self.http_status_distribution.get("500", 0),
            "http_504_count": self.http_status_distribution.get("504", 0),
            "http_status_distribution": dict(
                sorted(self.http_status_distribution.items())
            ),
            "request_p50_latency_ms": float(np.quantile(request_latencies, 0.50)),
            "request_p95_latency_ms": float(np.quantile(request_latencies, 0.95)),
            "request_p99_latency_ms": float(np.quantile(request_latencies, 0.99)),
            "request_sla_violation_rate": request_sla_violations
            / max(1, self.successful_requests),
            "excess_response_time_seconds": excess_response_time_seconds,
            "p95_latency_ms": float(np.quantile(self.p95_values, 0.95)),
            "sla_violation_rate": self.sla_violations / max(1, self.endpoint_samples),
            "endpoint_p95_sla_violation_rate": self.sla_violations
            / max(1, self.endpoint_samples),
            "sla_violation_severity": self.sla_severity / max(1, self.endpoint_samples),
            "longest_sla_violation_seconds": self.longest_violation_run
            * self.config.sampling_interval_seconds,
            "mean_successful_throughput_rps": self.successful_throughput
            / max(1, self.samples),
            "mean_offered_throughput_rps": self.offered_throughput
            / max(1, self.samples),
            "successful_throughput_rps": self.successful_requests
            / max(1, measured_seconds),
            "offered_throughput_rps": self.total_requests
            / max(1, measured_seconds),
            "cpu_seconds": self.cpu_seconds,
            "pod_seconds": self.pod_seconds,
            "mean_pod_count": self.pod_seconds / max(1, measured_seconds),
            "replica_changes": self.replica_changes,
            "action_churn": self.action_churn,
            "placement_changes": self.placement_changes,
            "migration_proxy": self.migration_proxy,
            "final_service_placement": final_service_placement,
            **cluster_metrics,
            "ceiling_time_fraction": self.ceiling_samples / max(1, self.samples),
            "pending_pod_seconds": self.pending_pod_seconds,
            "readiness_disruption_rate": self.readiness_disruption_samples
            / max(1, self.samples),
            "rejected_replicas": self.rejected_replicas,
            "cooldown_holds": self.cooldown_holds,
            "decision_mean_ms": float(decisions_ms.mean()) if len(decisions_ms) else 0.0,
            "decision_p95_ms": float(np.quantile(decisions_ms, 0.95))
            if len(decisions_ms)
            else 0.0,
            "inference_mean_ms": float(decisions_ms.mean())
            if self.controller_id.startswith("headlight-v2") and len(decisions_ms)
            else None,
            "forecast_q50_mae_rps": self.forecast_absolute
            / max(1, self.forecast_count)
            if self.forecast_count
            else None,
            "forecast_labels": self.forecast_count,
            "model_decisions": self.model_decisions,
            "guard_modified_decision_rate": self.guard_modified_decisions
            / max(1, self.model_decisions),
            "mean_guard_modified_services": self.guard_modified_services
            / max(1, self.model_decisions),
            "mean_hysteresis_modified_services": self.hysteresis_modified_services
            / max(1, self.model_decisions),
            "selected_candidate_counts": dict(sorted(self.selected_candidate_counts.items())),
            "model_graph_free_disagreement_rate": self.shadow_disagreements
            / max(1, self.shadow_decisions)
            if self.shadow_decisions
            else None,
            "model_graph_free_sla_probability_mae": self.shadow_probability_delta
            / max(1, self.shadow_decisions)
            if self.shadow_decisions
            else None,
            "graph_edge_sensitivity": self.graph_edge_sensitivity
            / max(1, self.graph_edge_sensitivity_count)
            if self.graph_edge_sensitivity_count
            else None,
            "mean_selected_resource_term": self.objective_component_totals["resource"]
            / max(1, self.model_decisions)
            if self.model_decisions
            else None,
            "mean_selected_migration_term": self.objective_component_totals["migration"]
            / max(1, self.model_decisions)
            if self.model_decisions
            else None,
            "mean_selected_reliability_penalty": self.objective_component_totals["reliability"]
            / max(1, self.model_decisions)
            if self.model_decisions
            else None,
            "mean_selected_objective": self.objective_component_totals["objective"]
            / max(1, self.model_decisions)
            if self.model_decisions
            else None,
        }


def run_episode(
    environment: ClosedLoopEnvironment,
    controller: ClosedLoopController,
) -> dict[str, Any]:
    controller.reset()
    observation = environment.reset()
    history = [observation]
    accumulator = EpisodeAccumulator(
        controller_id=controller.controller_id,
        case=environment.case,
        seed=environment.seed,
        trace_hash=environment.trace_hash,
        config=environment.config,
    )
    while not environment.done:
        if observation.step < environment.config.lookback_steps - 1:
            decision = ControllerDecision(hold_action(observation, "shared warm-up"))
            elapsed = 0.0
        else:
            candidates = environment.candidates(observation)
            started = time.perf_counter()
            decision = controller.decide(
                observation,
                history,
                candidates,
                environment.preview,
            )
            elapsed = time.perf_counter() - started
        observation, applied = environment.step(decision.action)
        history.append(observation)
        history = history[-environment.config.lookback_steps :]
        accumulator.update(
            observation,
            applied,
            elapsed,
            decision.diagnostics,
            environment.demand_trace,
        )
    return accumulator.result()
