"""Shared deterministic forecast-to-placement and batched actuation policy."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .profile import ServiceNode, ServiceProfile


@dataclass(frozen=True)
class ServiceForecast:
    request_q50: float
    request_q90: float
    cpu_q90: float
    p95_latency_q90: float
    sla_violation_probability: float

    def to_dict(self) -> dict[str, float]:
        return {
            "request_q50": self.request_q50,
            "request_q90": self.request_q90,
            "cpu_q90": self.cpu_q90,
            "p95_latency_q90": self.p95_latency_q90,
            "sla_violation_probability": self.sla_violation_probability,
        }


@dataclass(frozen=True)
class ServiceRuntimeState:
    placements: dict[str, int]
    ready_replicas: dict[str, int]
    current_p95_latency_ms: float | None
    current_cpu_utilization: float | None
    predictive_fresh: bool
    reactive_fresh: bool
    exists: bool = True
    pending_actuation: bool = False

    @property
    def total_replicas(self) -> int:
        return sum(int(value) for value in self.placements.values())

    @property
    def all_ready(self) -> bool:
        return not self.pending_actuation and all(
            int(self.ready_replicas.get(cluster, 0)) >= int(replicas)
            for cluster, replicas in self.placements.items()
        )


@dataclass(frozen=True)
class PolicyDecision:
    service: str
    prediction: dict[str, float] | None
    reason: str
    previous_placements: dict[str, int]
    desired_placements: dict[str, int]
    previous_total: int
    desired_total: int
    changed: bool
    readiness_delay: bool


@dataclass
class _ServiceMemory:
    healthy_windows: int = 0
    last_scale_down_at: float = float("-inf")


def largest_remainder_split(
    total_replicas: int,
    cluster_limits: dict[str, dict[str, int]],
    weights: dict[str, int] | None = None,
) -> dict[str, int]:
    """Hamilton apportionment with per-cluster minima and maxima."""
    if total_replicas <= 0:
        raise ValueError("total_replicas must be positive")
    clusters = tuple(cluster_limits)
    weights = weights or {"cluster2": 3, "cluster3": 5}
    if set(clusters) != set(weights):
        raise ValueError("placement weights must cover exactly the configured clusters")
    minima = {cluster: int(cluster_limits[cluster]["min_replicas"]) for cluster in clusters}
    maxima = {cluster: int(cluster_limits[cluster]["max_replicas"]) for cluster in clusters}
    minimum_total = sum(minima.values())
    maximum_total = sum(maxima.values())
    if not minimum_total <= total_replicas <= maximum_total:
        raise ValueError(
            f"total replicas {total_replicas} is outside [{minimum_total}, {maximum_total}]"
        )
    weight_total = float(sum(weights.values()))
    quotas = {cluster: total_replicas * weights[cluster] / weight_total for cluster in clusters}
    result = {
        cluster: min(maxima[cluster], max(minima[cluster], int(math.floor(quotas[cluster]))))
        for cluster in clusters
    }
    while sum(result.values()) < total_replicas:
        candidates = [
            cluster for cluster in clusters if result[cluster] < maxima[cluster]
        ]
        if not candidates:
            raise ValueError("cluster maxima cannot hold requested replicas")
        selected = max(
            candidates,
            key=lambda cluster: (
                quotas[cluster] - math.floor(quotas[cluster]),
                weights[cluster],
                cluster,
            ),
        )
        result[selected] += 1
        quotas[selected] = float(result[selected])
    while sum(result.values()) > total_replicas:
        candidates = [
            cluster for cluster in clusters if result[cluster] > minima[cluster]
        ]
        if not candidates:
            raise ValueError("cluster minima exceed requested replicas")
        selected = min(
            candidates,
            key=lambda cluster: (
                quotas[cluster] - math.floor(quotas[cluster]),
                weights[cluster],
                cluster,
            ),
        )
        result[selected] -= 1
        quotas[selected] = float(result[selected])
    return result


class ForecastPlacementPolicy:
    def __init__(
        self,
        profile: ServiceProfile,
        capacities: dict[str, Any],
        sla_latency_ms: float = 1000.0,
        cpu_guard: float = 0.8,
        max_scale_up_step: int = 2,
        scale_down_healthy_windows: int = 12,
        scale_down_cooldown_seconds: int = 180,
        sla_probability_guard: float = 0.5,
    ) -> None:
        self.profile = profile
        self.capacities = capacities
        self.sla_latency_ms = sla_latency_ms
        self.cpu_guard = cpu_guard
        self.max_scale_up_step = max_scale_up_step
        self.scale_down_healthy_windows = scale_down_healthy_windows
        self.scale_down_cooldown_seconds = scale_down_cooldown_seconds
        self.sla_probability_guard = sla_probability_guard
        self.memory = {node.name: _ServiceMemory() for node in profile.nodes}

    def _forecast_total(self, node: ServiceNode, forecast: ServiceForecast) -> int:
        capacity = self.capacities["services"][node.name]
        request_capacity = max(float(capacity["request_rate_per_pod"]), 1e-6)
        cpu_capacity = max(float(capacity["cpu_cores_per_pod"]), 1e-6)
        required = math.ceil(
            max(
                forecast.request_q90 / request_capacity,
                forecast.cpu_q90 / cpu_capacity,
            )
        )
        if (
            forecast.p95_latency_q90 > self.sla_latency_ms
            or forecast.sla_violation_probability >= self.sla_probability_guard
        ):
            required += 1
        minimum = sum(
            limits["min_replicas"] for limits in self.profile.cluster_limits.values()
        )
        maximum = sum(
            limits["max_replicas"] for limits in self.profile.cluster_limits.values()
        )
        return max(minimum, min(maximum, required))

    def _early_decision(
        self,
        node: ServiceNode,
        prediction: dict[str, float] | None,
        state: ServiceRuntimeState,
        current: dict[str, int],
        memory: _ServiceMemory,
    ) -> PolicyDecision | None:
        if not node.scalable:
            return self._decision(node, prediction, "fixed-database", current, current, False)
        if not state.exists:
            return self._decision(node, prediction, "absent-service", current, current, False)
        if not state.all_ready:
            memory.healthy_windows = 0
            return self._decision(node, prediction, "pending-readiness", current, current, True)
        if not state.predictive_fresh and not state.reactive_fresh:
            memory.healthy_windows = 0
            return self._decision(node, prediction, "all-inputs-stale", current, current, False)
        return None

    def _reactive_violation(self, state: ServiceRuntimeState) -> bool:
        return bool(
            state.reactive_fresh
            and (
                state.current_p95_latency_ms is not None and state.current_p95_latency_ms > self.sla_latency_ms
                or state.current_cpu_utilization is not None and state.current_cpu_utilization > self.cpu_guard
            )
        )

    def _target_total(
        self,
        state: ServiceRuntimeState,
        current_total: int,
        forecast_total: int,
        reactive_violation: bool,
    ) -> tuple[int, str]:
        if not state.predictive_fresh:
            return current_total + (self.max_scale_up_step if reactive_violation else 0), (
                "reactive-guard" if reactive_violation else "predictive-input-stale"
            )
        target = max(forecast_total, current_total + self.max_scale_up_step) if reactive_violation else forecast_total
        return target, "forecast-and-reactive-guard" if reactive_violation else "forecast-capacity"

    def _scale_down_or_hold(
        self,
        node: ServiceNode,
        prediction: dict[str, float] | None,
        state: ServiceRuntimeState,
        memory: _ServiceMemory,
        current: dict[str, int],
        current_total: int,
        target_total: int,
        timestamp: float,
    ) -> PolicyDecision:
        healthy = bool(
            state.predictive_fresh and state.reactive_fresh and target_total < current_total
            and state.current_p95_latency_ms is not None and state.current_p95_latency_ms <= self.sla_latency_ms
            and state.current_cpu_utilization is not None and state.current_cpu_utilization <= self.cpu_guard
        )
        memory.healthy_windows = memory.healthy_windows + 1 if healthy else 0
        if target_total >= current_total:
            memory.healthy_windows = 0
            return self._decision(node, prediction, "hold", current, current, False)
        if memory.healthy_windows < self.scale_down_healthy_windows:
            return self._decision(node, prediction, f"scale-down-stability-{memory.healthy_windows}/{self.scale_down_healthy_windows}", current, current, False)
        if timestamp - memory.last_scale_down_at < self.scale_down_cooldown_seconds:
            return self._decision(node, prediction, "scale-down-cooldown", current, current, False)
        minimum = sum(limits["min_replicas"] for limits in self.profile.cluster_limits.values())
        desired = largest_remainder_split(max(minimum, current_total - 1), self.profile.cluster_limits)
        memory.last_scale_down_at = timestamp
        memory.healthy_windows = 0
        return self._decision(node, prediction, "healthy-scale-down", current, desired, False)

    def decide(
        self,
        service_name: str,
        forecast: ServiceForecast | None,
        state: ServiceRuntimeState,
        now: float | None = None,
    ) -> PolicyDecision:
        timestamp = time.time() if now is None else float(now)
        node = self.profile.node(service_name)
        current = {cluster: int(state.placements.get(cluster, 0)) for cluster in self.profile.cluster_limits}
        current_total = sum(current.values())
        memory = self.memory[service_name]
        prediction = forecast.to_dict() if forecast else None
        early = self._early_decision(node, prediction, state, current, memory)
        if early is not None:
            return early

        forecast_total = current_total
        if state.predictive_fresh and forecast is not None:
            forecast_total = self._forecast_total(node, forecast)
        reactive_violation = self._reactive_violation(state)
        target_total, reason = self._target_total(state, current_total, forecast_total, reactive_violation)

        maximum = sum(
            limits["max_replicas"] for limits in self.profile.cluster_limits.values()
        )
        target_total = min(maximum, target_total)
        if target_total > current_total:
            memory.healthy_windows = 0
            desired_total = min(target_total, current_total + self.max_scale_up_step)
            desired = largest_remainder_split(desired_total, self.profile.cluster_limits)
            return self._decision(node, prediction, reason, current, desired, False)

        return self._scale_down_or_hold(node, prediction, state, memory, current, current_total, target_total, timestamp)

    @staticmethod
    def _decision(
        node: ServiceNode,
        prediction: dict[str, float] | None,
        reason: str,
        previous: dict[str, int],
        desired: dict[str, int],
        readiness_delay: bool,
    ) -> PolicyDecision:
        return PolicyDecision(
            service=node.name,
            prediction=prediction,
            reason=reason,
            previous_placements=dict(previous),
            desired_placements=dict(desired),
            previous_total=sum(previous.values()),
            desired_total=sum(desired.values()),
            changed=previous != desired,
            readiness_delay=readiness_delay,
        )


@dataclass(frozen=True)
class BatchResult:
    attempted: bool
    successful: bool
    status_code: int | None
    placements: tuple[dict[str, Any], ...]
    idempotency_key: str | None
    error: str | None = None


class PlacementActuator:
    """Send one delta-only batch and retry failed batches next window."""

    def __init__(
        self,
        action_handler_url: str,
        namespace: str,
        timeout_seconds: float = 5.0,
        shadow: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.action_handler_url = action_handler_url
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        self.shadow = shadow
        self.session = session or requests.Session()
        self.successfully_sent: dict[tuple[str, str], int] = {}
        self.failed_desired: dict[tuple[str, str], int] = {}
        self.pending_readiness: dict[tuple[str, str], int] = {}
        self.sent_at: dict[tuple[str, str], float] = {}

    def _reconcile_acknowledged(self, current: dict[str, dict[str, int]]) -> None:
        for key, replicas in self.successfully_sent.copy().items():
            service, cluster = key
            if int(current.get(service, {}).get(cluster, -1)) == replicas:
                self.successfully_sent.pop(key, None)

    def pending_services(self, current: dict[str, dict[str, int]]) -> set[str]:
        self._reconcile_acknowledged(current)
        return {
            service
            for service, _cluster in set(self.successfully_sent) | set(self.pending_readiness)
        }

    def observe_readiness(
        self,
        states: dict[str, ServiceRuntimeState],
        timestamp: float,
    ) -> dict[str, float]:
        completed: dict[str, float] = {}
        for key, replicas in self.pending_readiness.copy().items():
            service, cluster = key
            state = states.get(service)
            if state is None:
                continue
            if (
                int(state.placements.get(cluster, -1)) == replicas
                and int(state.ready_replicas.get(cluster, -1)) >= replicas
            ):
                sent_timestamp = self.sent_at.pop(key, float(timestamp))
                self.pending_readiness.pop(key, None)
                delay = max(0.0, float(timestamp) - sent_timestamp)
                completed[service] = max(completed.get(service, 0.0), delay)
        return completed

    def _desired_replicas(
        self,
        decisions: list[PolicyDecision],
        current: dict[str, dict[str, int]],
        node_by_name: dict[str, ServiceNode],
    ) -> dict[tuple[str, str], int]:
        desired: dict[tuple[str, str], int] = dict(self.failed_desired)
        for decision in decisions:
            if decision.service not in node_by_name:
                continue
            for cluster, replicas in decision.desired_placements.items():
                key = (decision.service, cluster)
                if int(current.get(decision.service, {}).get(cluster, -1)) == int(replicas):
                    desired.pop(key, None)
                    self.failed_desired.pop(key, None)
                else:
                    desired[key] = int(replicas)
        return desired

    def _placements(
        self,
        desired: dict[tuple[str, str], int],
        node_by_name: dict[str, ServiceNode],
    ) -> list[dict[str, Any]]:
        placements: list[dict[str, Any]] = []
        for (service, cluster), replicas in sorted(desired.items()):
            if self.successfully_sent.get((service, cluster)) == replicas:
                continue
            node = node_by_name.get(service)
            if node is None:
                continue
            placements.append({
                "workload": {"apiVersion": "apps/v1", "kind": "Deployment", "namespace": self.namespace, "name": node.workload},
                "clusterName": cluster,
                "replicas": replicas,
            })
        return placements

    def apply(
        self,
        decisions: list[PolicyDecision],
        current: dict[str, dict[str, int]],
        profile: ServiceProfile,
    ) -> BatchResult:
        self._reconcile_acknowledged(current)
        node_by_name = {node.name: node for node in profile.scalable_nodes}
        desired_by_key = self._desired_replicas(decisions, current, node_by_name)
        placements = self._placements(desired_by_key, node_by_name)
        if not placements:
            return BatchResult(False, True, None, (), None)
        canonical = json.dumps(placements, sort_keys=True, separators=(",", ":"))
        idempotency_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.shadow:
            return BatchResult(True, True, None, tuple(placements), idempotency_key)
        try:
            response = self.session.post(
                self.action_handler_url,
                json=placements,
                headers={"Idempotency-Key": idempotency_key},
                timeout=self.timeout_seconds,
            )
            successful = response.status_code in {200, 202}
            error = None if successful else response.text
            status_code = response.status_code
        except Exception as exc:
            successful = False
            error = str(exc)
            status_code = None
        if successful:
            for key, replicas in desired_by_key.items():
                self.successfully_sent[key] = replicas
                self.pending_readiness[key] = replicas
                self.sent_at[key] = time.time()
                self.failed_desired.pop(key, None)
        else:
            self.failed_desired = dict(desired_by_key)
        return BatchResult(
            attempted=True,
            successful=successful,
            status_code=status_code,
            placements=tuple(placements),
            idempotency_key=idempotency_key,
            error=error,
        )
