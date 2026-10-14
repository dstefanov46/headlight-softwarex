"""Opt-in v1 evidence recorder. It observes decisions and never selects actions."""
from __future__ import annotations

import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import BASE_FEATURES, MODEL_FEATURES, prometheus_queries


def json_value(value: Any) -> Any:
    """Represent unavailable numeric observations as null, preserving masks."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class V1DiagnosticRecorder:
    def __init__(self, path: str | Path, *, max_bytes: int = 128 * 1024 * 1024):
        if max_bytes <= 0:
            raise ValueError("v1 diagnostic byte limit must be positive")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.disabled = False
        self.used_bytes = 0
        self.source_hashes = {}
        try:
            directory = Path(__file__).parent
            self.source_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ["diagnostics_v1.py", "runtime.py", "features.py", "models.py", "policy.py"]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.used_bytes = self.path.stat().st_size if self.path.exists() else 0
        except OSError as exc:
            self.error(exc)

    def error(self, exc: Exception) -> None:
        self.disabled = True
        try:
            print("HEADLIGHT_V1_DIAGNOSTIC_ERROR " + json.dumps({
                "timestamp": time.time(), "error": f"{type(exc).__name__}: {exc}",
                "recording_disabled": True,
            }), flush=True)
        except Exception:
            # A broken stdout must not turn an optional evidence failure into
            # a controller failure; the retained file may still be available.
            pass

    def record(self, runtime: Any, phase: str, timestamp: int, payload: dict[str, Any]) -> None:
        if self.disabled:
            return
        try:
            record = json_value({
                "schema_version": "headlight.v1-diagnostics/v1", "phase": phase,
                "timestamp": timestamp, "recorded_at": time.time(),
                "controller": runtime.controller_name, "bundle_id": runtime.bundle.bundle_id,
                "policy_id": runtime.policy_config.payload["policy_id"],
                "policy_configuration_hash": runtime.policy_config.configuration_hash,
                "run_id": os.environ.get("HEADLIGHT_RUN_ID"),
                "seed": os.environ.get("HEADLIGHT_SEED"),
                "code_revision": os.environ.get("HEADLIGHT_CODE_REVISION", "unknown"),
                "image_revision": os.environ.get("HEADLIGHT_IMAGE_REVISION", "unknown"),
                "source_hashes": self.source_hashes,
                "bundle_component_hashes": runtime.bundle.metadata["component_hashes"],
                **payload,
            })
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            size = len(encoded.encode("utf-8")) + 1
            if self.used_bytes + size > self.max_bytes:
                raise OSError("v1 diagnostic byte limit reached")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.used_bytes += size
            print("HEADLIGHT_V1_DIAGNOSTIC_JSON " + encoded, flush=True)
        except Exception as exc:
            self.error(exc)

    def observation(self, runtime: Any, sample: Any) -> None:
        self.record(runtime, "observation", sample.timestamp, {
            "node_order": list(runtime.bundle.profile.node_order),
            "feature_order": list(BASE_FEATURES), "values": sample.values,
            "missing": sample.missing, "ages_seconds": sample.ages_seconds,
            "stale_nodes": sample.stale_nodes, "cluster_values": sample.cluster_values,
            "endpoint_observations": runtime.collector.observation_evidence,
            "expected_clusters": list(runtime.bundle.profile.cluster_limits),
            "replica_source": "collector; decision record includes runtime desired-replica override",
            "queries": prometheus_queries(runtime.collector.namespace,
                cpu_window=runtime.collector.cpu_rate_window,
                observed_zero_error_rate=runtime.collector.observed_zero_error_rate),
            "forward_fill_seconds": runtime.collector.forward_fill_seconds,
        })

    def decisions(self, runtime: Any, sample: Any, states: Any, forecasts: Any,
                  decisions: Any, inference_ms: float | None) -> None:
        minimum = sum(v["min_replicas"] for v in runtime.bundle.profile.cluster_limits.values())
        maximum = sum(v["max_replicas"] for v in runtime.bundle.profile.cluster_limits.values())
        services = {}
        for decision in decisions:
            state = states[decision.service]
            forecast = forecasts.get(decision.service)
            sizing = None
            if forecast is not None and all(math.isfinite(v) for v in forecast.to_dict().values()):
                capacity = runtime.bundle.capacity["services"][decision.service]
                request_required = forecast.request_q90 / max(float(capacity["request_rate_per_pod"]), 1e-6)
                cpu_required = forecast.cpu_q90 / max(float(capacity["cpu_cores_per_pod"]), 1e-6)
                base = math.ceil(max(request_required, cpu_required))
                adjustment = int(forecast.p95_latency_q90 > runtime.policy.sla_latency_ms
                    or forecast.sla_violation_probability >= runtime.policy.sla_probability_guard)
                sizing = {"capacity": capacity, "request_required": request_required,
                    "cpu_required": cpu_required, "base_required": base,
                    "latency_probability_adjustment": adjustment,
                    "unclipped_target": base + adjustment,
                    "clipped_forecast_target": max(minimum, min(maximum, base + adjustment)),
                    "bounds": [minimum, maximum],
                    "diagnostic_only": True}
            services[decision.service] = {
                "prediction": forecast.to_dict() if forecast else None,
                "sizing": sizing, "policy_reason": decision.reason,
                "previous_replicas": decision.previous_placements,
                "proposed_replicas": decision.desired_placements,
                "ready_replicas": state.ready_replicas, "state_complete": state.exists,
                "changed": decision.changed, "readiness_delay": decision.readiness_delay,
                "scalable": runtime.bundle.profile.node(decision.service).scalable,
                "predictive_fresh": bool(runtime._can_infer() and not sample.stale_nodes[
                    runtime.bundle.profile.node_order.index(decision.service)]),
            }
        self.record(runtime, "decision", sample.timestamp, {
            "services": services, "node_order": list(runtime.bundle.profile.node_order),
            "feature_order": list(BASE_FEATURES), "adjusted_values": sample.values,
            "adjusted_missing": sample.missing, "adjusted_ages_seconds": sample.ages_seconds,
            "replica_source": "desired deployment replicas when state is complete; otherwise collector",
            "model_feature_order": list(MODEL_FEATURES),
            "input_timestamps": list(runtime.diagnostic_timestamps),
            "scaled_model_input": runtime._input_tensor().detach().cpu().numpy() if runtime._can_infer() else None,
            "inference_duration_ms": inference_ms,
            "forecast_horizon_seconds": runtime.bundle.model_config["forecast_horizon_seconds"],
            "sla_latency_ms": runtime.policy.sla_latency_ms,
        })

    def actuation(self, runtime: Any, timestamp: int, batch: Any, delays: Any) -> None:
        self.record(runtime, "actuation", timestamp, {
            "shadow": runtime.actuator.shadow, "attempted": batch.attempted,
            "successful": batch.successful, "status_code": batch.status_code,
            "placements": batch.placements, "idempotency_key": batch.idempotency_key,
            "readiness_delays_seconds": delays,
            "confirmation": "request outcome only; subsequent decision states show observed desired/ready replicas",
        })
