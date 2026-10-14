"""Frozen-bundle inference loop shared by Headlight and LSTM."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .bundle import ModelBundle, load_checkpoint
from .diagnostics_v1 import V1DiagnosticRecorder
from .features import BASE_FEATURES, MetricSample, PrometheusMetricCollector
from .models import build_model, decode_forecasts
from .policy import (
    BatchResult,
    ForecastPlacementPolicy,
    PlacementActuator,
    PolicyConfig,
    PolicyDecision,
    ServiceForecast,
    ServiceRuntimeState,
)
from .profile import ServiceProfile


class KubectlStateReader:
    """Read current and Ready replicas without adding a Kubernetes SDK dependency."""

    def __init__(
        self,
        namespace: str,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.namespace = namespace
        self.command_runner = command_runner

    def read(self, profile: ServiceProfile) -> dict[str, ServiceRuntimeState]:
        placements = {node.name: dict.fromkeys(profile.cluster_limits, 0) for node in profile.nodes}
        ready = {node.name: dict.fromkeys(profile.cluster_limits, 0) for node in profile.nodes}
        exists = {node.name: False for node in profile.nodes}
        state_complete = True
        for cluster in profile.cluster_limits:
            try:
                result = self.command_runner(
                    [
                        "kubectl",
                        "--context",
                        cluster,
                        "-n",
                        self.namespace,
                        "get",
                        "deployments",
                        "-o",
                        "json",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=float(os.environ.get("HEADLIGHT_STATE_READ_TIMEOUT_SECONDS", "15")),
                )
            except subprocess.TimeoutExpired:
                state_complete = False
                print("HEADLIGHT_STATE_READ_ERROR " + json.dumps({
                    "timestamp": time.time(), "cluster": cluster, "error": "deployment query timed out",
                }), flush=True)
                continue
            if result.returncode != 0:
                state_complete = False
                continue
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                state_complete = False
                continue
            for deployment in payload.get("items", []):
                name = str(deployment.get("metadata", {}).get("name", ""))
                if name not in placements:
                    continue
                exists[name] = True
                placements[name][cluster] = int(deployment.get("spec", {}).get("replicas", 0) or 0)
                ready[name][cluster] = int(deployment.get("status", {}).get("readyReplicas", 0) or 0)
        return {
            node.name: ServiceRuntimeState(
                placements=placements[node.name],
                ready_replicas=ready[node.name],
                current_p95_latency_ms=None,
                current_cpu_utilization=None,
                predictive_fresh=False,
                reactive_fresh=False,
                # Missing member state is unknown, not a zero-replica placement.
                # The existing absent-service guard suppresses actuation.
                exists=exists[node.name] and state_complete,
            )
            for node in profile.nodes
        }


class ControllerRuntime:
    def __init__(
        self,
        bundle: ModelBundle,
        model: torch.nn.Module,
        device: torch.device,
        collector: PrometheusMetricCollector,
        state_reader: KubectlStateReader,
        policy: ForecastPlacementPolicy,
        actuator: PlacementActuator,
        controller_name: str,
        action_log: str | Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.model = model
        self.device = device
        self.collector = collector
        self.state_reader = state_reader
        self.policy = policy
        self.policy_config = policy.policy_config
        self.actuator = actuator
        self.controller_name = controller_name
        self.lookback = int(bundle.model_config["lookback"])
        self.interval = int(bundle.model_config["sampling_interval_seconds"])
        self.buffers = {
            node.name: deque(maxlen=self.lookback) for node in bundle.profile.nodes
        }
        self.diagnostic_timestamps = deque(maxlen=self.lookback)
        self.diagnostics = None
        if os.environ.get("HEADLIGHT_V1_DIAGNOSTIC_LOG"):
            self.diagnostics = V1DiagnosticRecorder(
                os.environ["HEADLIGHT_V1_DIAGNOSTIC_LOG"],
                max_bytes=int(os.environ.get("HEADLIGHT_V1_DIAGNOSTIC_MAX_BYTES", str(128 * 1024 * 1024))),
            )
        self.action_log = Path(action_log) if action_log else None
        if self.action_log:
            self.action_log.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        runtime_profile: ServiceProfile,
        prometheus_url: str,
        action_handler_url: str,
        controller_name: str,
        namespace: str | None = None,
        shadow: bool = False,
        action_log: str | Path | None = None,
        policy_config: str | Path | PolicyConfig | None = None,
        device: torch.device | None = None,
    ) -> "ControllerRuntime":
        selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = ModelBundle.load(
            bundle_path,
            runtime_profile=runtime_profile,
            expected_model_type=controller_name,
        )
        model = build_model(bundle.model_config, bundle.adjacency, selected_device)
        model.load_state_dict(load_checkpoint(bundle, selected_device), strict=True)
        model.eval()
        selected_namespace = namespace or runtime_profile.namespace
        collector = PrometheusMetricCollector(
            prometheus_url,
            runtime_profile,
            namespace=selected_namespace,
            forward_fill_seconds=30,
            capture_observations=bool(os.environ.get("HEADLIGHT_V1_DIAGNOSTIC_LOG")),
            cluster_names=tuple(os.environ["HEADLIGHT_V1_PROMETHEUS_CLUSTERS"].split(","))
                if os.environ.get("HEADLIGHT_V1_DIAGNOSTIC_LOG") and os.environ.get("HEADLIGHT_V1_PROMETHEUS_CLUSTERS") else None,
        )
        state_reader = KubectlStateReader(selected_namespace)
        loaded_policy_config = (
            policy_config
            if isinstance(policy_config, PolicyConfig)
            else PolicyConfig.load(policy_config)
        )
        policy = ForecastPlacementPolicy(runtime_profile, bundle.capacity, loaded_policy_config)
        actuator = PlacementActuator(
            action_handler_url,
            selected_namespace,
            shadow=shadow,
        )
        return cls(
            bundle=bundle,
            model=model,
            device=selected_device,
            collector=collector,
            state_reader=state_reader,
            policy=policy,
            actuator=actuator,
            controller_name=controller_name,
            action_log=action_log,
        )

    def _append_sample(
        self,
        sample: MetricSample,
        cluster_state: dict[str, ServiceRuntimeState],
    ) -> MetricSample:
        values = sample.values.copy()
        missing = sample.missing.copy()
        ages = sample.ages_seconds.copy()
        replica_index = BASE_FEATURES.index("replicas")
        for node_index, node in enumerate(self.bundle.profile.nodes):
            state = cluster_state[node.name]
            if state.exists:
                values[node_index, replica_index] = float(state.total_replicas)
                missing[node_index, replica_index] = False
                ages[node_index, replica_index] = 0
        adjusted = MetricSample(
            timestamp=sample.timestamp,
            values=values,
            missing=missing,
            ages_seconds=ages,
            stale_nodes=sample.stale_nodes,
        )
        model_values = adjusted.model_values(self.bundle.scaler)
        for node_index, node in enumerate(self.bundle.profile.nodes):
            self.buffers[node.name].append(model_values[node_index])
        return adjusted

    def _can_infer(self) -> bool:
        return all(len(buffer) == self.lookback for buffer in self.buffers.values())

    def _input_tensor(self) -> torch.Tensor:
        array = np.zeros(
            (
                1,
                self.lookback,
                len(self.bundle.profile.nodes),
                int(self.bundle.model_config["input_size"]),
            ),
            dtype=np.float32,
        )
        for node_index, node in enumerate(self.bundle.profile.nodes):
            array[0, :, node_index, :] = np.stack(self.buffers[node.name])
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _infer(self) -> tuple[dict[str, ServiceForecast], float]:
        features = self._input_tensor()
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        started = time.perf_counter()
        with torch.no_grad():
            decoded = decode_forecasts(self.model(features))
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        duration_ms = (time.perf_counter() - started) * 1000.0
        output = decoded[0].detach().cpu().numpy()
        forecasts = {
            node.name: ServiceForecast(
                request_q50=float(output[index, 0]),
                request_q90=float(output[index, 1]),
                cpu_q90=float(output[index, 2]),
                p95_latency_q90=float(output[index, 3]),
                sla_violation_probability=float(output[index, 4]),
            )
            for index, node in enumerate(self.bundle.profile.nodes)
        }
        print(
            "HEADLIGHT_INFERENCE_TIMING "
            + json.dumps(
                {
                    "timestamp": time.time(),
                    "controller": self.controller_name,
                    "model_variant": self.bundle.model_config["variant"],
                    "service_count": len(self.bundle.profile.nodes),
                    "duration_ms": duration_ms,
                    "bundle_id": self.bundle.bundle_id,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return forecasts, duration_ms

    def run_once(self, timestamp: int | None = None) -> dict[str, Any]:
        sample = self.collector.collect(timestamp)
        self._diagnostic("observation", sample)
        raw_states = self.state_reader.read(self.bundle.profile)
        sample = self._append_sample(sample, raw_states)
        self.diagnostic_timestamps.append(sample.timestamp)
        forecasts: dict[str, ServiceForecast] = {}
        inference_ms: float | None = None
        if self._can_infer():
            forecasts, inference_ms = self._infer()
        p95_index = BASE_FEATURES.index("p95_latency_ms")
        cpu_index = BASE_FEATURES.index("cpu_utilization")
        decisions: list[PolicyDecision] = []
        current: dict[str, dict[str, int]] = {}
        current = {
            node.name: dict(raw_states[node.name].placements)
            for node in self.bundle.profile.nodes
        }
        readiness_delays = self.actuator.observe_readiness(raw_states, sample.timestamp)
        pending_services = self.actuator.pending_services(current)
        for node_index, node in enumerate(self.bundle.profile.nodes):
            raw_state = raw_states[node.name]
            p95_value = float(sample.values[node_index, p95_index])
            cpu_value = float(sample.values[node_index, cpu_index])
            p95_fresh = (
                math.isfinite(p95_value)
                and sample.ages_seconds[node_index, p95_index] <= 30
            )
            cpu_fresh = (
                math.isfinite(cpu_value)
                and sample.ages_seconds[node_index, cpu_index] <= 30
            )
            reactive_fresh = p95_fresh or cpu_fresh
            state = ServiceRuntimeState(
                placements=raw_state.placements,
                ready_replicas=raw_state.ready_replicas,
                current_p95_latency_ms=p95_value if p95_fresh else None,
                current_cpu_utilization=cpu_value if cpu_fresh else None,
                predictive_fresh=self._can_infer() and not bool(sample.stale_nodes[node_index]),
                reactive_fresh=reactive_fresh,
                exists=raw_state.exists,
                pending_actuation=node.name in pending_services,
            )
            decisions.append(
                self.policy.decide(
                    node.name,
                    forecasts.get(node.name),
                    state,
                    now=sample.timestamp,
                )
            )
        self._record_proposals(sample.timestamp, decisions, raw_states)
        self._diagnostic("decisions", sample, raw_states, forecasts, decisions, inference_ms)
        batch = self.actuator.apply(decisions, current, self.bundle.profile)
        self._record_actions(sample.timestamp, decisions, batch, readiness_delays)
        self._diagnostic("actuation", sample.timestamp, batch, readiness_delays)
        return {
            "timestamp": sample.timestamp,
                "bundle_id": self.bundle.bundle_id,
                "policy_id": self.policy_config.payload["policy_id"],
                "policy_configuration_hash": self.policy_config.configuration_hash,
                "inference_duration_ms": inference_ms,
            "decisions": decisions,
            "batch": batch,
        }

    def _diagnostic(self, method: str, *args: Any) -> None:
        """Optional evidence failures cannot change a policy or actuation result."""
        if self.diagnostics is None or self.diagnostics.disabled:
            return
        try:
            getattr(self.diagnostics, method)(self, *args)
        except Exception as exc:
            self.diagnostics.error(exc)

    def _record_proposals(
        self, timestamp: int, decisions: list[PolicyDecision],
        states: dict[str, ServiceRuntimeState],
    ) -> None:
        """Persist intent before making an actuation request which may fail."""
        if not self.action_log:
            return
        path = self.action_log.with_name("proposals.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            for decision in decisions:
                record = {
                    "timestamp": timestamp, "controller": self.controller_name,
                    "bundle_id": self.bundle.bundle_id,
                    "policy_id": self.policy_config.payload["policy_id"],
                    "policy_configuration_hash": self.policy_config.configuration_hash,
                    "service": decision.service, "prediction": decision.prediction,
                    "policy_reason": decision.reason,
                    "previous_replicas": decision.previous_placements,
                    "desired_replicas": decision.desired_placements,
                    "ready_replicas": states[decision.service].ready_replicas,
                }
                encoded = json.dumps(record, sort_keys=True)
                handle.write(encoded + "\n")
                print("HEADLIGHT_PROPOSAL_JSON " + encoded, flush=True)
            handle.flush()
            os.fsync(handle.fileno())

    def _record_actions(
        self,
        timestamp: int,
        decisions: list[PolicyDecision],
        batch: BatchResult,
        readiness_delays: dict[str, float],
    ) -> None:
        if not self.action_log:
            return
        attempted_keys = {
            (placement["workload"]["name"], placement["clusterName"])
            for placement in batch.placements
        }
        with self.action_log.open("a", encoding="utf-8") as handle:
            for decision in decisions:
                applied = (
                    decision.desired_placements
                    if batch.successful
                    and any(key[0] == decision.service for key in attempted_keys)
                    and not self.actuator.shadow
                    else decision.previous_placements
                )
                record = {
                    "timestamp": timestamp,
                    "controller": self.controller_name,
                    "bundle_id": self.bundle.bundle_id,
                    "policy_id": self.policy_config.payload["policy_id"],
                    "policy_configuration_hash": self.policy_config.configuration_hash,
                    "service": decision.service,
                    "prediction": decision.prediction,
                    "policy_reason": decision.reason,
                    "previous_replicas": decision.previous_placements,
                    "desired_replicas": decision.desired_placements,
                    "applied_replicas": applied,
                    "readiness_delay": decision.readiness_delay,
                    "readiness_delay_seconds": readiness_delays.get(decision.service),
                    "batch_attempted": batch.attempted,
                    "batch_successful": batch.successful,
                    "batch_status_code": batch.status_code,
                    "idempotency_key": batch.idempotency_key,
                }
                encoded = json.dumps(record, sort_keys=True)
                handle.write(encoded + "\n")
                # stdout is followed by the experiment host while the pod is alive.
                # The file remains available for final collection and replay.
                print("HEADLIGHT_ACTION_JSON " + encoded, flush=True)
            handle.flush()
            os.fsync(handle.fileno())

    def run_forever(self) -> None:
        next_tick = math.ceil(time.time() / self.interval) * self.interval
        while True:
            delay = next_tick - time.time()
            if delay > 0:
                time.sleep(delay)
            try:
                self.run_once(int(next_tick))
            except Exception as exc:
                print(
                    "HEADLIGHT_RUNTIME_ERROR "
                    + json.dumps(
                        {
                            "timestamp": time.time(),
                            "controller": self.controller_name,
                            "bundle_id": self.bundle.bundle_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            next_tick += self.interval
