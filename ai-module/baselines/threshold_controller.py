#!/usr/bin/env python3
"""Reactive threshold controller using the shared CRP placement actuator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/app")

from headlight.features import BASE_FEATURES, PrometheusMetricCollector
from headlight.policy import PlacementActuator, PolicyDecision, largest_remainder_split
from headlight.profile import load_service_profile
from headlight.runtime import KubectlStateReader


@dataclass
class ThresholdMemory:
    stable_windows: int = 0
    cooldown_remaining_seconds: int = 0


class ThresholdController:
    def __init__(self, profile_path: str, prometheus_url: str, action_handler_url: str, namespace: str,
                 action_log: str, interval_seconds: int = 15, shadow: bool = False) -> None:
        self.profile = load_service_profile(profile_path)
        self.interval_seconds = interval_seconds
        self.collector = PrometheusMetricCollector(prometheus_url, self.profile, namespace=namespace)
        self.state_reader = KubectlStateReader(namespace)
        self.actuator = PlacementActuator(action_handler_url, namespace, shadow=shadow)
        self.action_log = Path(action_log)
        self.action_log.parent.mkdir(parents=True, exist_ok=True)
        self.memory = {node.name: ThresholdMemory() for node in self.profile.scalable_nodes}
        self.policy_id = "research-v3-threshold-policy"
        self.policy_hash = hashlib.sha256(
            json.dumps(
                {
                    "scale_up": {"p95_latency_ms_gt": 300, "cpu_utilization_gt": 0.7},
                    "scale_down": {"stable_windows": 2, "p95_latency_ms_lt": 180, "cpu_utilization_lt": 0.35},
                    "cooldown_seconds": 60,
                }, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def _decision(self, node, raw_state, p95: float, cpu: float, timestamp: int, pending: bool) -> PolicyDecision:
        current = {cluster: int(raw_state.placements.get(cluster, 0)) for cluster in self.profile.cluster_limits}
        desired = dict(current)
        memory = self.memory[node.name]
        total = sum(current.values())
        minimum = sum(int(value["min_replicas"]) for value in self.profile.cluster_limits.values())
        maximum = sum(int(value["max_replicas"]) for value in self.profile.cluster_limits.values())
        reason = "hold"
        if not raw_state.exists:
            reason = "absent-service"
        elif pending:
            reason = "pending-actuation"
        elif not math.isfinite(p95) and not math.isfinite(cpu):
            reason = "reactive-input-stale"
            memory.stable_windows = 0
        elif memory.cooldown_remaining_seconds > 0:
            reason = "cooldown"
            memory.cooldown_remaining_seconds = max(0, memory.cooldown_remaining_seconds - self.interval_seconds)
        elif p95 > 300 or cpu > 0.70:
            desired_total = min(maximum, total + 1)
            desired = self._split(desired_total)
            reason = "threshold-scale-up"
            memory.stable_windows = 0
            memory.cooldown_remaining_seconds = 60
        elif p95 < 180 and cpu < 0.35:
            memory.stable_windows += 1
            if memory.stable_windows >= 2:
                desired = self._split(max(minimum, total - 1))
                reason = "threshold-scale-down"
                memory.stable_windows = 0
                memory.cooldown_remaining_seconds = 60
            else:
                reason = "healthy-window"
        else:
            memory.stable_windows = 0

        return PolicyDecision(
            service=node.name,
            prediction=None,
            reason=reason,
            previous_placements=current,
            desired_placements=desired,
            previous_total=total,
            desired_total=sum(desired.values()),
            changed=current != desired,
            readiness_delay=False,
        )

    def _split(self, total: int) -> dict[str, int]:
        # Use the same bounded 6:8 allocation as the predictive controllers.
        weights = {
            cluster: int(value["max_replicas"])
            for cluster, value in self.profile.cluster_limits.items()
        }
        return largest_remainder_split(total, self.profile.cluster_limits, weights)

    def run_once(self, timestamp: int) -> None:
        sample = self.collector.collect(timestamp)
        states = self.state_reader.read(self.profile)
        current = {node.name: dict(states[node.name].placements) for node in self.profile.nodes}
        self.actuator.observe_readiness(states, sample.timestamp)
        pending = self.actuator.pending_services(current)
        p95_index = BASE_FEATURES.index("p95_latency_ms")
        cpu_index = BASE_FEATURES.index("cpu_utilization")
        decisions = []
        for index, node in enumerate(self.profile.scalable_nodes):
            node_index = self.profile.node_order.index(node.name)
            p95 = float(sample.values[node_index, p95_index])
            cpu = float(sample.values[node_index, cpu_index])
            if sample.ages_seconds[node_index, p95_index] > 30:
                p95 = float("nan")
            if sample.ages_seconds[node_index, cpu_index] > 30:
                cpu = float("nan")
            decisions.append(self._decision(node, states[node.name], p95, cpu, timestamp, node.name in pending))
        batch = self.actuator.apply(decisions, current, self.profile)
        attempted = {(item["workload"]["name"], item["clusterName"]) for item in batch.placements}
        with self.action_log.open("a", encoding="utf-8") as handle:
            for decision in decisions:
                applied = decision.desired_placements if batch.successful and any(
                    key[0] == decision.service for key in attempted
                ) and not self.actuator.shadow else decision.previous_placements
                encoded = json.dumps({
                    "timestamp": timestamp,
                    "controller": "threshold",
                    "policy_id": self.policy_id,
                    "policy_configuration_hash": self.policy_hash,
                    "service": decision.service,
                    "prediction": None,
                    "policy_reason": decision.reason,
                    "previous_replicas": decision.previous_placements,
                    "desired_replicas": decision.desired_placements,
                    "applied_replicas": applied,
                    "readiness_delay": decision.readiness_delay,
                    "batch_attempted": batch.attempted,
                    "batch_successful": batch.successful,
                    "batch_status_code": batch.status_code,
                    "idempotency_key": batch.idempotency_key,
                }, sort_keys=True)
                handle.write(encoded + "\n")
                print("HEADLIGHT_ACTION_JSON " + encoded, flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        print("THRESHOLD_INFERENCE_TIMING " + json.dumps({
            "timestamp": timestamp, "controller": "threshold", "applicable": False,
        }, separators=(",", ":")), flush=True)

    def run_forever(self) -> None:
        next_tick = math.ceil(time.time() / self.interval_seconds) * self.interval_seconds
        while True:
            delay = next_tick - time.time()
            if delay > 0:
                time.sleep(delay)
            try:
                self.run_once(int(next_tick))
            except Exception as exc:
                print("THRESHOLD_RUNTIME_ERROR " + json.dumps({"timestamp": time.time(), "error": str(exc)}), flush=True)
            next_tick += self.interval_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the research-v3 threshold controller")
    parser.add_argument("--profile", default=os.environ.get("HEADLIGHT_SERVICE_PROFILE_FILE", "/config/profile/profile.json"))
    parser.add_argument("--prometheus", default=os.environ.get("HEADLIGHT_PROMETHEUS_URL", ""))
    parser.add_argument("--action-handler", default=os.environ.get("HEADLIGHT_ACTION_HANDLER_URL", ""))
    parser.add_argument("--namespace", default=os.environ.get("HEADLIGHT_NAMESPACE", "trainticket"))
    parser.add_argument("--action-log", default=os.environ.get("HEADLIGHT_ACTION_LOG", "/artifacts/actions.jsonl"))
    parser.add_argument("--interval", type=int, default=int(os.environ.get("HEADLIGHT_INTERVAL_SECONDS", "15")))
    parser.add_argument("--shadow", action="store_true", default=os.environ.get("HEADLIGHT_SHADOW_MODE", "0") == "1")
    args = parser.parse_args()
    if not args.prometheus or not args.action_handler:
        parser.error("Prometheus and Actions Handler URLs are required")
    ThresholdController(args.profile, args.prometheus, args.action_handler, args.namespace, args.action_log, args.interval, args.shadow).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
