"""Reusable baseline policy decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdState:
    replicas: int
    stable_windows: int = 0
    cooldown_remaining_seconds: int = 0


def threshold_decision(
    *,
    p95_latency_ms: float,
    cpu_utilization: float,
    state: ThresholdState,
    min_replicas: int,
    max_replicas: int,
    sla_ms: float = 300.0,
    cooldown_seconds: int = 60,
    sampling_interval_seconds: int = 15,
) -> ThresholdState:
    if state.cooldown_remaining_seconds > 0:
        return ThresholdState(
            replicas=state.replicas,
            stable_windows=state.stable_windows,
            cooldown_remaining_seconds=max(0, state.cooldown_remaining_seconds - sampling_interval_seconds),
        )
    if p95_latency_ms > sla_ms or cpu_utilization > 0.70:
        return ThresholdState(min(max_replicas, state.replicas + 1), 0, cooldown_seconds)
    if p95_latency_ms < sla_ms * 0.60 and cpu_utilization < 0.35:
        stable = state.stable_windows + 1
        replicas = max(min_replicas, state.replicas - 1) if stable >= 2 else state.replicas
        return ThresholdState(replicas, 0 if stable >= 2 else stable, cooldown_seconds if stable >= 2 else 0)
    return ThresholdState(state.replicas, 0, 0)


def static_split_replicas(total_replicas: int, cluster2_weight: int = 3, cluster3_weight: int = 5) -> dict[str, int]:
    total_weight = cluster2_weight + cluster3_weight
    cluster2 = round(total_replicas * cluster2_weight / total_weight)
    cluster3 = total_replicas - cluster2
    return {"cluster2": max(0, cluster2), "cluster3": max(0, cluster3)}

