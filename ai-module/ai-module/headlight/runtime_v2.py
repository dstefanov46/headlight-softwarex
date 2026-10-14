"""Shadow-first live runtime for immutable Headlight bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import requests
import torch

from .bundle_v2 import HeadlightV2Bundle, load_v2_checkpoint
from .diagnostics_v2 import DiagnosticStore, action_lifecycle
from .features import BASE_FEATURES, MetricSample, PrometheusMetricCollector
from .models_v2 import build_headlight_v2_model
from .observability import content_sha256
from .policy_v2 import (
    HEADLIGHT_V2_FULL_CONTROLLER,
    CandidatePredictionV2,
    HeadlightPolicyV2,
    PolicyV2Config,
    PolicyV2Decision,
    PolicyV2Executor,
    PolicyV2RecordContext,
    ReplicaPlacementCandidateV2,
    ServiceRuntimeStateV2,
    calibrated_predictions_from_model_output,
    build_policy_v2_decision_record,
)
from .profile import ServiceProfile
from .readiness import PendingAgeTracker
from .runtime import KubectlStateReader


class HeadlightV2Runtime:
    """Collect v1-compatible telemetry, assemble v2 tensors, and run Policy-v2."""

    def __init__(
        self,
        *,
        bundle: HeadlightV2Bundle,
        profile: ServiceProfile,
        model: torch.nn.Module,
        device: torch.device,
        collector: PrometheusMetricCollector,
        state_reader: KubectlStateReader,
        policy: HeadlightPolicyV2,
        action_handler_url: str,
        run_id: str,
        seed: int,
        shadow_mode: bool = True,
        action_log: str | Path | None = None,
        session: requests.Session | None = None,
        diagnostic_dir: str | Path | None = None,
        diagnostic_max_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if tuple(bundle.model_config["node_order"]) != profile.node_order:
            raise ValueError("runtime service-profile node order differs from the v2 bundle")
        if tuple(profile.cluster_limits) != ("cluster2", "cluster3"):
            raise ValueError("Headlight checkpoint expects cluster2,cluster3 placement order")
        if bundle.calibration.get("schema_version") != "headlight.calibration/identity-v1":
            raise ValueError("live v2 runtime supports only the frozen identity calibration contract")
        if bundle.calibration.get("calibrated") is not True:
            raise ValueError("v2 bundle must explicitly mark candidate outputs calibrated")
        self.bundle = bundle
        self.profile = profile
        self.model = model
        self.device = device
        self.collector = collector
        self.state_reader = state_reader
        self.policy = policy
        self.executor = PolicyV2Executor(policy, shadow_mode=shadow_mode)
        self.action_handler_url = action_handler_url
        self.run_id = run_id
        self.seed = int(seed)
        self.shadow_mode = bool(shadow_mode)
        self.session = session or requests.Session()
        self.diagnostics_enabled = diagnostic_dir is not None
        self.diagnostic_store = None
        self.diagnostic_storage_error = None
        self.last_actuator_acceptance = None
        self.last_actuator_outcome = None
        if diagnostic_dir is not None:
            try:
                self.diagnostic_store = DiagnosticStore(diagnostic_dir, max_bytes=diagnostic_max_bytes)
            except Exception as exc:
                self.diagnostic_storage_error = f"{type(exc).__name__}: {exc}"
        self.lookback = int(bundle.model_config.get("lookback_steps", 24))
        self.interval = int(bundle.model_config["sampling_interval_seconds"])
        self.node_frames: deque[np.ndarray] = deque(maxlen=self.lookback)
        self.edge_frames: deque[np.ndarray] = deque(maxlen=self.lookback)
        self.last_action_at: dict[str, float] = {}
        self.pending_age_tracker = PendingAgeTracker()
        self.action_log = Path(action_log) if action_log else None
        if self.action_log:
            self.action_log.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        runtime_profile: ServiceProfile,
        prometheus_url: str,
        action_handler_url: str,
        policy_config: PolicyV2Config,
        run_id: str,
        seed: int,
        namespace: str | None = None,
        shadow_mode: bool = True,
        metric_diagnostics: bool = False,
        prometheus_clusters: tuple[str, ...] | None = None,
        cpu_rate_window: str = "90s",
        action_log: str | Path | None = None,
        device: torch.device | None = None,
        diagnostic_dir: str | Path | None = None,
        diagnostic_max_bytes: int = 128 * 1024 * 1024,
    ) -> "HeadlightV2Runtime":
        if prometheus_clusters is None:
            raise ValueError(
                "Prometheus cluster mapping is required for live v2: set "
                "HEADLIGHT_V2_PROMETHEUS_CLUSTERS and provide one controller-reachable "
                "Prometheus URL per profile cluster"
            )
        collector = PrometheusMetricCollector(
            prometheus_url,
            runtime_profile,
            namespace=namespace or runtime_profile.namespace,
            forward_fill_seconds=30,
            diagnostic_logging=metric_diagnostics,
            cluster_names=prometheus_clusters,
            cpu_rate_window=cpu_rate_window,
            observed_zero_error_rate=True,
        )
        selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = HeadlightV2Bundle.load(bundle_path, expected_model_type="headlight-v2")
        static = torch.tensor(bundle.adjacency["matrix"], dtype=torch.float32, device=selected_device)
        observed = np.asarray(bundle.adjacency["observed_mask"], dtype=bool)
        observed_tensor = torch.tensor(
            observed,
            dtype=torch.float32,
            device=selected_device,
        )
        model = build_headlight_v2_model(bundle.model_config, static, observed_tensor)
        model.load_state_dict(load_v2_checkpoint(bundle, selected_device), strict=True)
        model.eval()
        selected_namespace = namespace or runtime_profile.namespace
        return cls(
            bundle=bundle,
            profile=runtime_profile,
            model=model,
            device=selected_device,
            collector=collector,
            state_reader=KubectlStateReader(selected_namespace),
            policy=HeadlightPolicyV2(
                runtime_profile,
                controller_id=HEADLIGHT_V2_FULL_CONTROLLER,
                config=policy_config,
            ),
            action_handler_url=action_handler_url,
            run_id=run_id,
            seed=seed,
            shadow_mode=shadow_mode,
            action_log=action_log,
            diagnostic_dir=diagnostic_dir,
            diagnostic_max_bytes=diagnostic_max_bytes,
        )

    def _states(
        self,
        sample: MetricSample,
        raw_states: Mapping[str, Any],
    ) -> dict[str, ServiceRuntimeStateV2]:
        result: dict[str, ServiceRuntimeStateV2] = {}
        index = {name: BASE_FEATURES.index(name) for name in BASE_FEATURES}
        now = float(sample.timestamp)
        for node_index, node in enumerate(self.profile.nodes):
            raw = raw_states[node.name]
            desired = {cluster: int(raw.placements[cluster]) for cluster in self.profile.cluster_limits}
            ready = {cluster: int(raw.ready_replicas[cluster]) for cluster in self.profile.cluster_limits}
            pending = {cluster: max(0, desired[cluster] - ready[cluster]) for cluster in desired}
            pending_ages = self.pending_age_tracker.ages(node.name, pending, now)

            def metric(name: str) -> float | None:
                value = float(sample.values[node_index, index[name]])
                return value if math.isfinite(value) else None

            field_map = {
                "demand_rps": "request_rate",
                "cpu_utilization": "cpu_utilization",
                "p95_latency_ms": "p95_latency_ms",
                "error_rate": "error_rate",
            }
            missing = {
                target
                for target, source in field_map.items()
                if bool(sample.missing[node_index, index[source]])
            }
            stale = {
                target
                for target, source in field_map.items()
                if int(sample.ages_seconds[node_index, index[source]]) > 30
            }
            missing.add("queue_depth")
            result[node.name] = ServiceRuntimeStateV2(
                desired_replicas=desired,
                ready_replicas=ready,
                pending_replicas=pending,
                pending_age_seconds=pending_ages,
                demand_rps=metric("request_rate"),
                cpu_utilization=metric("cpu_utilization"),
                queue_depth=None,
                p95_latency_ms=metric("p95_latency_ms"),
                error_rate=metric("error_rate"),
                predictive_fresh=not bool(sample.stale_nodes[node_index]),
                missing_fields=frozenset(missing),
                stale_fields=frozenset(stale),
                time_since_last_action_seconds=(
                    now - self.last_action_at[node.name]
                    if node.name in self.last_action_at
                    else None
                ),
                exists=bool(raw.exists),
                cluster_request_rate=sample.cluster_values.get("request_rate", {}).get(node.name, {}),
                cluster_cpu_utilization=sample.cluster_values.get("cpu_utilization", {}).get(node.name, {}),
            )
        return result

    def _assemble_frames(
        self,
        sample: MetricSample,
        states: Mapping[str, ServiceRuntimeStateV2],
    ) -> tuple[np.ndarray, np.ndarray]:
        names = tuple(self.bundle.model_config["node_feature_names"])
        scales = np.asarray(self.bundle.feature_schema["node_feature_scales"], dtype=np.float32)
        node = np.zeros((len(self.profile.nodes), len(names)), dtype=np.float32)
        base_index = {name: BASE_FEATURES.index(name) for name in BASE_FEATURES}
        for i, service in enumerate(self.profile.node_order):
            state = states[service]
            values = {
                "request_rate": sample.values[i, base_index["request_rate"]],
                "cpu_usage_cores": sample.values[i, base_index["cpu_usage_cores"]],
                "cpu_utilization": sample.values[i, base_index["cpu_utilization"]],
                "mean_latency_ms": sample.values[i, base_index["mean_latency_ms"]],
                "p95_latency_ms": sample.values[i, base_index["p95_latency_ms"]],
                "error_rate": sample.values[i, base_index["error_rate"]],
                "desired_replicas": state.total_desired,
                "ready_replicas": sum(state.ready_replicas.values()),
                "pending_replicas": state.total_pending,
                "pending_age_seconds": max(state.pending_age_seconds.values(), default=0.0),
                "time_since_last_action_seconds": state.time_since_last_action_seconds or 0.0,
                "request_rate_missing": float("demand_rps" in state.missing_fields),
                "cpu_missing": float("cpu_utilization" in state.missing_fields),
                "latency_missing": float("p95_latency_ms" in state.missing_fields),
                "error_missing": float("error_rate" in state.missing_fields),
                "request_rate_stale": float("demand_rps" in state.stale_fields),
                "cpu_stale": float("cpu_utilization" in state.stale_fields),
                "latency_stale": float("p95_latency_ms" in state.stale_fields),
                "error_stale": float("error_rate" in state.stale_fields),
            }
            node[i] = [float(values[name]) if math.isfinite(float(values[name])) else 0.0 for name in names]
        node /= scales

        edge_names = tuple(self.bundle.model_config["edge_feature_names"])
        edge_scales = np.asarray(self.bundle.feature_schema["edge_feature_scales"], dtype=np.float32)
        adjacency = np.asarray(self.bundle.adjacency["matrix"], dtype=np.float32)
        observed = np.asarray(self.bundle.adjacency["observed_mask"], dtype=bool)
        edge = np.zeros((*adjacency.shape, len(edge_names)), dtype=np.float32)
        for source in range(len(self.profile.nodes)):
            for destination in range(len(self.profile.nodes)):
                source_rate = node[source, names.index("request_rate")] * scales[names.index("request_rate")]
                destination_state = states[self.profile.node_order[destination]]
                source_placement = states[self.profile.node_order[source]].desired_replicas
                destination_placement = destination_state.desired_replicas
                dominant_source = max(source_placement, key=source_placement.get)
                dominant_destination = max(destination_placement, key=destination_placement.get)
                values = {
                    "call_rate": source_rate * adjacency[source, destination],
                    "call_ratio": adjacency[source, destination],
                    "latency_ms": destination_state.p95_latency_ms or 0.0,
                    "error_rate": destination_state.error_rate or 0.0,
                    "cross_cluster": float(dominant_source != dominant_destination),
                    "missing": float(not observed[source, destination]),
                    "stale": float(not destination_state.predictive_fresh),
                }
                edge[source, destination] = [values[name] for name in edge_names]
        edge /= edge_scales
        return node, edge

    def append_observation(
        self,
        sample: MetricSample,
        raw_states: Mapping[str, Any],
    ) -> dict[str, ServiceRuntimeStateV2]:
        states = self._states(sample, raw_states)
        node, edge = self._assemble_frames(sample, states)
        self.node_frames.append(node)
        self.edge_frames.append(edge)
        return states

    def prime(self, node_window: np.ndarray, edge_window: np.ndarray) -> None:
        if node_window.shape[0] != self.lookback or edge_window.shape[0] != self.lookback:
            raise ValueError("v2 prime windows must match the bundle lookback")
        self.node_frames.clear()
        self.edge_frames.clear()
        self.node_frames.extend(np.asarray(node_window, dtype=np.float32))
        self.edge_frames.extend(np.asarray(edge_window, dtype=np.float32))

    def _scorer(self, states: Mapping[str, ServiceRuntimeStateV2], trace: dict | None = None) -> Callable[..., Mapping[str, CandidatePredictionV2]]:
        node_window = torch.tensor(np.stack(self.node_frames)[None], dtype=torch.float32, device=self.device)
        edge_window = torch.tensor(np.stack(self.edge_frames)[None], dtype=torch.float32, device=self.device)
        node_index = {name: i for i, name in enumerate(self.profile.node_order)}
        clusters = tuple(self.profile.cluster_limits)

        def score(
            service_name: str,
            candidates: tuple[ReplicaPlacementCandidateV2, ...],
        ) -> Mapping[str, CandidatePredictionV2]:
            count = len(candidates)
            placement = np.zeros((1, count, len(self.profile.nodes), len(clusters)), dtype=np.float32)
            readiness = np.ones((1, count, len(self.profile.nodes), 2), dtype=np.float32)
            for c in range(count):
                for service, state in states.items():
                    i = node_index[service]
                    values = candidates[c].placement_dict() if service == service_name else state.desired_replicas
                    placement[0, c, i] = [values[cluster] for cluster in clusters]
                    total = max(1, sum(values.values()))
                    ready_total = min(total, sum(state.ready_replicas.values()))
                    readiness[0, c, i, 0] = ready_total / total
                    readiness[0, c, i, 1] = max(state.pending_age_seconds.values(), default=0.0)
                target = node_index[service_name]
                readiness[0, c, target, 0] = candidates[c].readiness.ready_fraction
                readiness[0, c, target, 1] = candidates[c].readiness.pending_age_seconds
            replicas = placement.sum(axis=-1)
            started = time.perf_counter()
            with torch.no_grad():
                output = self.model(
                    node_window,
                    edge_window,
                    torch.tensor(replicas, device=self.device),
                    torch.tensor(placement, device=self.device),
                    torch.tensor(readiness, device=self.device),
                )
            performance = dict(output["candidate_performance"])
            performance.update(
                calibrated=True,
                endpoint_names=list(self.bundle.model_config["endpoint_names"]),
                horizons_seconds=list(self.bundle.model_config["horizons_seconds"]),
            )
            predictions = calibrated_predictions_from_model_output(
                self.policy.registration.controller_id,
                candidates,
                performance,
                expected_endpoint_names=tuple(self.bundle.model_config["endpoint_names"]),
                expected_horizons_seconds=tuple(self.bundle.model_config["horizons_seconds"]),
            )
            print(json.dumps({
                "event": "HEADLIGHT_V2_INFERENCE",
                "service": service_name,
                "candidate_count": count,
                "duration_ms": (time.perf_counter() - started) * 1000.0,
                "bundle_id": self.bundle.bundle_id,
            }, separators=(",", ":")), flush=True)
            if trace is not None:
                # Diagnostic conversion/storage errors cannot change predictions
                # already computed for the real policy.
                try:
                    entry = {
                        "forecasts": {
                            str(h): {"q50": float(output["demand_q50"][0, node_index[service_name], j]),
                                     "q90": float(output["demand_q90"][0, node_index[service_name], j])}
                            for j, h in enumerate(self.bundle.model_config["horizons_seconds"])
                        },
                        "duration_ms": (time.perf_counter() - started) * 1000,
                        "candidate_ids": [c.candidate_id for c in candidates],
                    }
                    trace[service_name] = entry
                    trace.setdefault("_forecasts_by_service", {
                        name: {str(h): {"q50": float(output["demand_q50"][0, i, j]),
                                       "q90": float(output["demand_q90"][0, i, j])}
                               for j, h in enumerate(self.bundle.model_config["horizons_seconds"])}
                        for i, name in enumerate(self.profile.node_order)
                    })
                    if self.diagnostic_store is None:
                        raise OSError(self.diagnostic_storage_error or "diagnostic store unavailable")
                    entry["replay"] = {
                        "window": self.diagnostic_store.tensors({
                            "node_window": node_window.detach().cpu().numpy(),
                            "edge_window": edge_window.detach().cpu().numpy(),
                        }),
                        "candidates": self.diagnostic_store.tensors({
                            "replicas": replicas, "placement": placement, "readiness": readiness,
                        }),
                        "outputs": self.diagnostic_store.tensors({
                            "demand_q50": output["demand_q50"].detach().cpu().numpy(),
                            "demand_q90": output["demand_q90"].detach().cpu().numpy(),
                            **{
                            name: value.detach().cpu().numpy()
                            for name, value in output["candidate_performance"].items()
                            if isinstance(value, torch.Tensor)
                            },
                        }),
                    }
                except Exception as exc:
                    trace.setdefault(service_name, {})["capture_error"] = f"{type(exc).__name__}: {exc}"
            return predictions

        return score

    def _actuate(self, decisions: tuple[PolicyV2Decision, ...]) -> None:
        placements = []
        node_by_name = {node.name: node for node in self.profile.scalable_nodes}
        for decision in decisions:
            node = node_by_name.get(decision.service)
            if node is None:
                continue
            previous = decision.previous_placement_dict
            for cluster, replicas in decision.desired_placement.items():
                if replicas == previous[cluster]:
                    continue
                placements.append({
                    "workload": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "namespace": self.profile.namespace,
                        "name": node.workload,
                    },
                    "clusterName": cluster,
                    "replicas": replicas,
                })
        if not placements:
            return
        canonical = json.dumps(placements, sort_keys=True, separators=(",", ":"))
        requested_at = time.time()
        response = None
        try:
            response = self.session.post(
                self.action_handler_url,
                json=placements,
                headers={"Idempotency-Key": hashlib.sha256(canonical.encode()).hexdigest()},
                timeout=5.0,
            )
            response.raise_for_status()
        except Exception as exc:
            self.last_actuator_outcome = {
                "requested_at": requested_at, "acknowledged_at": None,
                "http_status": response.status_code if response is not None else None,
                "acceptance": "rejected" if response is not None else "unknown",
                "error_type": type(exc).__name__, "message": str(exc),
                "idempotency_key": hashlib.sha256(canonical.encode()).hexdigest(),
                "deployment_confirmed": False, "readiness_confirmed": False,
            }
            raise
        now = time.time()
        self.last_actuator_acceptance = {
            "requested_at": requested_at, "acknowledged_at": now,
            "http_status": response.status_code, "idempotency_key": hashlib.sha256(canonical.encode()).hexdigest(),
            "deployment_confirmed": False, "readiness_confirmed": False,
        }
        self.last_actuator_outcome = self.last_actuator_acceptance
        for decision in decisions:
            self.last_action_at[decision.service] = now

    def infer_states(
        self,
        states: Mapping[str, ServiceRuntimeStateV2],
        *,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        if len(self.node_frames) != self.lookback:
            if self.diagnostics_enabled:
                recorded_at = float(timestamp if timestamp is not None else time.time())
                for service, state in states.items():
                    record = {
                        "event": "HEADLIGHT_V2_DIAGNOSTIC", "schema_version": "headlight.diagnostics/v1",
                        "model_status": "lookback-incomplete", "model_observations": len(self.node_frames),
                        "required_observations": self.lookback,
                        "decision": {"schema_version": "headlight.unavailable-model-decision/v1",
                                     "run_id": self.run_id, "seed": self.seed, "timestamp": recorded_at,
                                     "bundle_id": self.bundle.bundle_id, "service": service,
                                     "observed_state": {name: getattr(state, name) for name in
                                                        ("demand_rps", "cpu_utilization", "queue_depth", "p95_latency_ms", "error_rate")},
                                     "masks": {"missing": sorted(state.missing_fields), "stale": sorted(state.stale_fields)},
                                     "replicas_by_cluster": {"desired": dict(state.desired_replicas),
                                                             "ready": dict(state.ready_replicas), "pending": dict(state.pending_replicas)},
                                     "forecast_quantiles_by_horizon": {}, "candidates": []},
                        "effective_policy": asdict(self.policy.config),
                        "model_permitted_action": None, "final_action": None, "replay": None,
                    }
                    try:
                        if self.diagnostic_store is None:
                            raise OSError(self.diagnostic_storage_error or "diagnostic store unavailable")
                        self.diagnostic_store.record(record)
                    except Exception as exc:
                        record["record_storage_error"] = f"{type(exc).__name__}: {exc}"
                    print(json.dumps(record, separators=(",", ":")), flush=True)
            return {"ready": False, "observations": len(self.node_frames), "lookback": self.lookback}
        recorded: list[PolicyV2Decision] = []
        trace = {} if self.diagnostics_enabled else None
        initial_memory = copy.deepcopy(self.policy.memory) if self.diagnostics_enabled else None
        score = self._scorer(states, trace) if self.diagnostics_enabled else self._scorer(states)
        predictions_cache = {}
        def cached_score(service, candidates):
            key = (service, tuple(c.candidate_id for c in candidates))
            if key not in predictions_cache:
                try:
                    predictions_cache[key] = score(service, candidates)
                except Exception as exc:
                    predictions_cache[key] = exc
                    if trace is not None:
                        trace.setdefault(service, {})["inference_error"] = f"{type(exc).__name__}: {exc}"
            value = predictions_cache[key]
            if isinstance(value, Exception):
                raise value
            return value
        self.last_actuator_acceptance = None
        self.last_actuator_outcome = None
        started = time.perf_counter()
        try:
            result = self.executor.execute(
                states,
                cached_score if self.diagnostics_enabled else score,
                recorder=recorded.append,
                actuator=self._actuate,
            )
        except Exception:
            if self.diagnostics_enabled:
                self._emit_diagnostics(recorded, states, cached_score, initial_memory, trace,
                                       float(timestamp if timestamp is not None else time.time()))
            raise
        payload = {
            "event": "HEADLIGHT_V2_DECISION_INTERVAL",
            "timestamp": float(timestamp if timestamp is not None else time.time()),
            "run_id": self.run_id,
            "seed": self.seed,
            "bundle_id": self.bundle.bundle_id,
            "feature_schema_hash": content_sha256(self.bundle.feature_schema),
            "adjacency_hash": content_sha256(self.bundle.adjacency),
            "policy_id": self.policy.config.policy_id,
            "shadow_mode": self.shadow_mode,
            "actuator_called": result.actuator_called,
            "inference_duration_ms": (time.perf_counter() - started) * 1000.0,
            "decisions": [
                {
                    "service": decision.service,
                    "reason": decision.reason,
                    "changed": decision.changed,
                    "selected_candidate_id": decision.selected_candidate_id,
                    "desired_placement": decision.desired_placement,
                    "pending_replicas": dict(states[decision.service].pending_replicas),
                    "pending_age_seconds": dict(states[decision.service].pending_age_seconds),
                }
                for decision in result.decisions
            ],
        }
        if self.action_log:
            with self.action_log.open("a", encoding="utf-8") as handle:
                for decision in result.decisions:
                    handle.write(json.dumps({
                        "schema_version": "headlight.controller-action/v2",
                        "timestamp": payload["timestamp"],
                        "run_id": self.run_id,
                        "seed": self.seed,
                        "bundle_id": self.bundle.bundle_id,
                        "policy_id": self.policy.config.policy_id,
                        "service": decision.service,
                        "previous_replicas": decision.previous_placement_dict,
                        "desired_replicas": decision.desired_placement,
                        "policy_reason": decision.reason,
                        "changed": decision.changed,
                        "batch_successful": bool(
                            not decision.changed
                            or (not self.shadow_mode and result.actuator_called)
                        ),
                        "shadow_mode": self.shadow_mode,
                    }, sort_keys=True) + "\n")
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        if self.diagnostics_enabled:
            self._emit_diagnostics(result.decisions, states, cached_score, initial_memory, trace, payload["timestamp"])
        return {"ready": True, **payload}

    def _emit_diagnostics(self, decisions, states, cached_score, initial_memory, trace, timestamp):
        diagnostic_started = time.perf_counter()
        try:
            proposals = self.policy.diagnose_interval(states, cached_score, initial_memory=initial_memory)
            predictive = [p for p in proposals.values() if p.changed and p.action_origin == "model_predictive"]
            winner = min(predictive, key=lambda p: (-p.objective_improvement, p.service)) if predictive else None
            for final in decisions:
                proposal = proposals[final.service]
                interval_action = proposal if not proposal.changed or proposal is winner else self.policy._replace_with_hold(
                    proposal, f"serialized-behind:{winner.service}")
                self._record_diagnostic(final, proposals[final.service], states[final.service],
                                        trace, timestamp, {
                                            "candidate_id": interval_action.selected_candidate_id,
                                            "placement": interval_action.desired_placement,
                                            "reason": interval_action.reason, "changed": interval_action.changed,
                                        })
            print(json.dumps({"event": "HEADLIGHT_V2_DIAGNOSTIC_TIMING", "timestamp": timestamp,
                              "duration_ms": (time.perf_counter() - diagnostic_started) * 1000}), flush=True)
        except Exception as exc:
            print(json.dumps({"event": "HEADLIGHT_V2_DIAGNOSTIC_ERROR", "timestamp": timestamp,
                              "error_type": type(exc).__name__, "message": str(exc)}), flush=True)

    def _record_diagnostic(self, final, proposal, state, trace, timestamp, model_interval_action=None):
        entry = trace.get(final.service, {})
        context = PolicyV2RecordContext(self.run_id, self.seed, content_sha256(self.bundle.feature_schema),
                                        content_sha256(self.bundle.adjacency), proposal.controller_id, self.bundle.bundle_id)
        # The strict record describes the constrained model proposal; the actual
        # hybrid/ablation action and lifecycle are stated separately.
        forecasts = entry.get("forecasts", {})
        if proposal.reason == "fixed-service" and state.predictive_fresh:
            forecasts = trace.get("_forecasts_by_service", {}).get(final.service, {})
        if forecasts:
            decision = build_policy_v2_decision_record(
                proposal, state, context, timestamp=timestamp,
                forecast_quantiles_by_horizon=forecasts,
                action_timestamps=action_lifecycle(shadow=True, changed=False, accepted=None), shadow_mode=True,
            )
        else:
            # The existing strict schema requires observed forecasts. Do not
            # invent zero forecasts to satisfy it when prediction is unavailable.
            decision = {
                "schema_version": "headlight.unavailable-model-decision/v1",
                "run_id": self.run_id, "seed": self.seed, "timestamp": timestamp,
                "service": final.service, "forecast_quantiles_by_horizon": {},
                "observed_state": {key: getattr(state, key) for key in
                                   ("demand_rps", "cpu_utilization", "queue_depth", "p95_latency_ms", "error_rate")},
                "masks": {"missing": sorted(state.missing_fields), "stale": sorted(state.stale_fields),
                          "predictive_fresh": state.predictive_fresh},
                "replicas_by_cluster": {"desired": dict(state.desired_replicas), "ready": dict(state.ready_replicas),
                                        "pending": dict(state.pending_replicas)},
                "candidates": [c.to_decision_candidate() for c in proposal.candidates],
                "selection_reason": proposal.reason,
                "bundle_id": self.bundle.bundle_id,
            }
        thresholds = {"cpu_utilization": self.policy.config.reactive_cpu_utilization,
                      "queue_depth": self.policy.config.reactive_queue_depth,
                      "p95_latency_ms": self.policy.config.reactive_latency_ms,
                      "error_rate": self.policy.config.reactive_error_rate}
        predicted = [c for c in proposal.candidates if c.prediction_available]
        preferred = min(predicted, key=self.policy._selection_key).candidate.candidate_id if predicted else None
        value = {
            "schema_version": "headlight.diagnostics/v1", "event": "HEADLIGHT_V2_DIAGNOSTIC",
            "decision": decision,
            "model_status": "ok" if predicted and len(predicted) == len(proposal.candidates) else proposal.reason,
            "input_status": "predictive-input-stale" if not state.predictive_fresh else
                            "masked-inputs" if state.missing_fields or state.stale_fields else "fresh",
            "model_preferred_candidate_id": preferred,
            "model_permitted_action": {"candidate_id": proposal.selected_candidate_id, "placement": proposal.desired_placement,
                                       "reason": proposal.reason, "changed": proposal.changed},
            "model_interval_action": model_interval_action,
            "final_action": {"candidate_id": final.selected_candidate_id, "placement": final.desired_placement,
                             "reason": final.reason, "changed": final.changed, "action_origin": final.action_origin},
            "local_override": final.action_origin == "reactive_emergency",
            "interval_override": final.reason == "emergency-priority",
            "reactive_triggers": [{"name": name, "threshold": thresholds[name], "observed": getattr(state, name)}
                                  for name in self.policy._reactive_triggers(state)],
            "cluster_observations": {"request_rate": dict(state.cluster_request_rate), "cpu_utilization": dict(state.cluster_cpu_utilization)},
            "pending_age_seconds": dict(state.pending_age_seconds),
            "forecast_units": "requests_per_second",
            "normalization": {"node_feature_scales": self.bundle.feature_schema.get("node_feature_scales"),
                              "edge_feature_scales": self.bundle.feature_schema.get("edge_feature_scales"),
                              "demand_output_scale_rps": self.bundle.model_config.get("demand_output_scale_rps", 1.0),
                              "forecast_values_already_scaled": True},
            "total_objectives": {c.candidate.candidate_id: c.objective for c in proposal.candidates},
            "effective_policy": asdict(self.policy.config),
            "code_revision": os.environ.get("HEADLIGHT_CODE_REVISION", "unknown"),
            "image_revision": os.environ.get("HEADLIGHT_IMAGE_REVISION", "unknown"),
            "replay": entry.get("replay"), "capture_error": entry.get("capture_error", self.diagnostic_storage_error),
            "inference_error": entry.get("inference_error"),
            "model_inference_duration_ms": entry.get("duration_ms"),
            "candidate_axes": {"candidate_ids": entry.get("candidate_ids"),
                               "endpoint_names": self.bundle.model_config["endpoint_names"],
                               "horizons_seconds": self.bundle.model_config["horizons_seconds"]},
            "actuator": {"lifecycle": action_lifecycle(shadow=self.shadow_mode, changed=final.changed,
                                                        accepted=self.last_actuator_outcome),
                         "acceptance": self.last_actuator_acceptance if final.changed else None,
                         "outcome": self.last_actuator_outcome if final.changed else None,
                         "deployment_confirmed": False, "readiness_confirmed": False},
        }
        try:
            if self.diagnostic_store is None:
                raise OSError(self.diagnostic_storage_error or "diagnostic store unavailable")
            self.diagnostic_store.record(value)
        except Exception as exc:
            value["record_storage_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(value, separators=(",", ":")), flush=True)

    def run_once(self, timestamp: int | None = None) -> dict[str, Any]:
        sample = self.collector.collect(timestamp)
        raw_states = self.state_reader.read(self.profile)
        states = self.append_observation(sample, raw_states)
        return self.infer_states(states, timestamp=sample.timestamp)

    def run_forever(self) -> None:
        next_run = time.monotonic()
        while True:
            self.run_once()
            next_run += self.interval
            time.sleep(max(0.0, next_run - time.monotonic()))
