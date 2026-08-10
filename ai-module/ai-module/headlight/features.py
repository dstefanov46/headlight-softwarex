"""Aligned Prometheus features, chronological splits, and leak-free scaling."""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import requests

from .profile import ServiceProfile


FEATURE_SCHEMA_VERSION = "headlight.features/v1"
BASE_FEATURES = (
    "replicas",
    "request_rate",
    "cpu_usage_cores",
    "cpu_utilization",
    "mean_latency_ms",
    "p95_latency_ms",
    "error_rate",
)
MASK_FEATURES = tuple(f"{name}_missing" for name in BASE_FEATURES)
MODEL_FEATURES = BASE_FEATURES + MASK_FEATURES
TARGETS = ("request_rate", "cpu_usage_cores", "p95_latency_ms", "sla_violation")


def feature_schema() -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "base_features": list(BASE_FEATURES),
        "mask_features": list(MASK_FEATURES),
        "model_features": list(MODEL_FEATURES),
        "targets": list(TARGETS),
        "missing_mask_value": 1,
        "fresh_mask_value": 0,
        "forward_fill_seconds": 30,
    }


@dataclass(frozen=True)
class MetricSample:
    timestamp: int
    values: np.ndarray
    missing: np.ndarray
    ages_seconds: np.ndarray
    stale_nodes: np.ndarray

    def model_values(self, scaler: "NodeFeatureScaler") -> np.ndarray:
        transformed = scaler.transform(self.values[np.newaxis, ...])[0]
        return np.concatenate((transformed, self.missing.astype(np.float32)), axis=-1)


@dataclass(frozen=True)
class AlignedSeries:
    timestamps: np.ndarray
    values: np.ndarray
    missing: np.ndarray

    def validate(self, profile: ServiceProfile) -> None:
        expected = (len(self.timestamps), len(profile.nodes), len(BASE_FEATURES))
        if self.values.shape != expected:
            raise ValueError(f"feature tensor shape {self.values.shape} does not match {expected}")
        if self.missing.shape != expected:
            raise ValueError(f"missing-mask shape {self.missing.shape} does not match {expected}")
        if len(self.timestamps) < 2 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("feature timestamps must be unique and strictly increasing")


@dataclass(frozen=True)
class SequenceData:
    features: np.ndarray
    targets: np.ndarray
    target_timestamps: np.ndarray
    input_frame_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    holdout: np.ndarray
    first_boundary: int
    second_boundary: int
    purge_seconds: int


class NodeFeatureScaler:
    """Standardize each node/feature pair using training frames only."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, fitted_frame_timestamps: Iterable[int] = ()) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.fitted_frame_timestamps = tuple(int(value) for value in fitted_frame_timestamps)
        if self.mean.shape != self.scale.shape or self.mean.ndim != 2:
            raise ValueError("scaler mean and scale must be [node, feature] arrays with matching shapes")
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("scaler parameters must be finite and scales must be positive")

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        missing: np.ndarray | None = None,
        fitted_frame_timestamps: Iterable[int] = (),
    ) -> "NodeFeatureScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError("scaler input must have shape [time, node, feature]")
        invalid = ~np.isfinite(array)
        if missing is not None:
            mask = np.asarray(missing, dtype=bool)
            if mask.shape != array.shape:
                raise ValueError("scaler missing mask does not match values")
            invalid |= mask
        masked = np.where(invalid, np.nan, array)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(masked, axis=0)
            scale = np.nanstd(masked, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32), fitted_frame_timestamps)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-2:] != self.mean.shape:
            raise ValueError(
                f"scaler expected trailing shape {self.mean.shape}, received {array.shape[-2:]}"
            )
        filled = np.where(np.isfinite(array), array, self.mean)
        return ((filled - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-2:] != self.mean.shape:
            raise ValueError("scaled values have an incompatible node/feature shape")
        return (array * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "headlight.node-feature-scaler/v1",
            "feature_order": list(BASE_FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fitted_frame_timestamps": list(self.fitted_frame_timestamps),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NodeFeatureScaler":
        if value.get("schema_version") != "headlight.node-feature-scaler/v1":
            raise ValueError("unsupported scaler schema")
        if tuple(value.get("feature_order", [])) != BASE_FEATURES:
            raise ValueError("scaler feature order is incompatible with runtime")
        return cls(
            np.asarray(value["mean"], dtype=np.float32),
            np.asarray(value["scale"], dtype=np.float32),
            value.get("fitted_frame_timestamps", ()),
        )


def chronological_split(
    target_timestamps: np.ndarray,
    purge_seconds: int = 180,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> ChronologicalSplit:
    timestamps = np.asarray(target_timestamps, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 3 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("target timestamps must be a strictly increasing vector with at least three samples")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9) or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    first_index = min(len(timestamps) - 2, max(1, int(math.floor(len(timestamps) * ratios[0]))))
    second_index = min(
        len(timestamps) - 1,
        max(first_index + 1, int(math.floor(len(timestamps) * (ratios[0] + ratios[1])))),
    )
    first_boundary = int(timestamps[first_index])
    second_boundary = int(timestamps[second_index])
    train = np.flatnonzero(timestamps < first_boundary - purge_seconds)
    validation = np.flatnonzero(
        (timestamps >= first_boundary) & (timestamps < second_boundary - purge_seconds)
    )
    holdout = np.flatnonzero(timestamps >= second_boundary)
    if not len(train) or not len(validation) or not len(holdout):
        raise ValueError(
            "not enough chronological samples for a 70/15/15 split with "
            f"a {purge_seconds}-second purge"
        )
    return ChronologicalSplit(
        train=train,
        validation=validation,
        holdout=holdout,
        first_boundary=first_boundary,
        second_boundary=second_boundary,
        purge_seconds=purge_seconds,
    )


def build_sequences(
    series: AlignedSeries,
    profile: ServiceProfile,
    lookback: int = 24,
    horizon: int = 12,
    sla_latency_ms: float = 1000.0,
) -> SequenceData:
    series.validate(profile)
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    sample_count = len(series.timestamps) - lookback - horizon + 1
    if sample_count <= 0:
        raise ValueError("not enough aligned samples for the requested lookback and horizon")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_timestamps: list[int] = []
    input_indices: list[np.ndarray] = []
    request_index = BASE_FEATURES.index("request_rate")
    cpu_index = BASE_FEATURES.index("cpu_usage_cores")
    p95_index = BASE_FEATURES.index("p95_latency_ms")
    for start in range(sample_count):
        stop = start + lookback
        target_index = stop + horizon - 1
        raw = series.values[start:stop]
        mask = series.missing[start:stop].astype(np.float32)
        features.append(np.concatenate((raw, mask), axis=-1))
        target_values = series.values[target_index]
        p95 = target_values[:, p95_index]
        targets.append(
            np.stack(
                (
                    target_values[:, request_index],
                    target_values[:, cpu_index],
                    p95,
                    np.where(
                        np.isfinite(p95),
                        (p95 > sla_latency_ms).astype(np.float32),
                        np.nan,
                    ),
                ),
                axis=-1,
            )
        )
        target_timestamps.append(int(series.timestamps[target_index]))
        input_indices.append(np.arange(start, stop, dtype=np.int64))
    return SequenceData(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        target_timestamps=np.asarray(target_timestamps, dtype=np.int64),
        input_frame_indices=tuple(input_indices),
    )


def fit_training_scaler(
    series: AlignedSeries,
    sequences: SequenceData,
    train_sample_indices: np.ndarray,
) -> NodeFeatureScaler:
    frame_indices = sorted(
        {
            int(frame)
            for sample_index in train_sample_indices
            for frame in sequences.input_frame_indices[int(sample_index)]
        }
    )
    timestamps = series.timestamps[frame_indices]
    return NodeFeatureScaler.fit(
        series.values[frame_indices],
        series.missing[frame_indices],
        fitted_frame_timestamps=timestamps,
    )


def apply_scaler_to_sequences(sequences: SequenceData, scaler: NodeFeatureScaler) -> np.ndarray:
    base_count = len(BASE_FEATURES)
    scaled = scaler.transform(sequences.features[..., :base_count])
    return np.concatenate((scaled, sequences.features[..., base_count:]), axis=-1).astype(np.float32)


def bounded_forward_fill(
    series: AlignedSeries,
    max_age_seconds: int = 30,
) -> AlignedSeries:
    """Fill missing values from the last observation without clearing masks."""
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    values = np.asarray(series.values, dtype=np.float32).copy()
    missing = np.asarray(series.missing, dtype=bool).copy()
    last_values = np.full(values.shape[1:], np.nan, dtype=np.float32)
    last_observed_at = np.full(values.shape[1:], -1, dtype=np.int64)
    for frame_index, timestamp in enumerate(series.timestamps):
        observed = ~missing[frame_index] & np.isfinite(values[frame_index])
        last_values[observed] = values[frame_index][observed]
        last_observed_at[observed] = int(timestamp)
        age = int(timestamp) - last_observed_at
        fillable = (
            missing[frame_index]
            & (last_observed_at >= 0)
            & (age <= max_age_seconds)
            & np.isfinite(last_values)
        )
        values[frame_index][fillable] = last_values[fillable]
        values[frame_index][missing[frame_index] & ~fillable] = np.nan
    return AlignedSeries(
        timestamps=np.asarray(series.timestamps, dtype=np.int64).copy(),
        values=values,
        missing=missing,
    )


def estimate_capacities(
    series: AlignedSeries,
    profile: ServiceProfile,
    training_frame_indices: Iterable[int],
    sla_latency_ms: float = 1000.0,
    cpu_limit: float = 0.8,
) -> dict[str, Any]:
    replicas_i = BASE_FEATURES.index("replicas")
    request_i = BASE_FEATURES.index("request_rate")
    cpu_usage_i = BASE_FEATURES.index("cpu_usage_cores")
    cpu_util_i = BASE_FEATURES.index("cpu_utilization")
    p95_i = BASE_FEATURES.index("p95_latency_ms")
    selected = np.asarray(sorted({int(value) for value in training_frame_indices}), dtype=np.int64)
    services: dict[str, Any] = {}
    for node_index, node in enumerate(profile.nodes):
        values = series.values[selected, node_index]
        missing = series.missing[selected, node_index]
        healthy = (
            ~missing[:, replicas_i]
            & ~missing[:, request_i]
            & ~missing[:, cpu_usage_i]
            & ~missing[:, cpu_util_i]
            & ~missing[:, p95_i]
            & (values[:, replicas_i] > 0)
            & (values[:, p95_i] <= sla_latency_ms)
            & (values[:, cpu_util_i] <= cpu_limit)
        )
        per_pod_request = values[healthy, request_i] / values[healthy, replicas_i]
        per_pod_cpu = values[healthy, cpu_usage_i] / values[healthy, replicas_i]
        positive_request = per_pod_request[np.isfinite(per_pod_request) & (per_pod_request > 0)]
        positive_cpu = per_pod_cpu[np.isfinite(per_pod_cpu) & (per_pod_cpu > 0)]
        services[node.name] = {
            "request_rate_per_pod": float(np.quantile(positive_request, 0.10))
            if len(positive_request)
            else 1.0,
            "cpu_cores_per_pod": float(np.quantile(positive_cpu, 0.10))
            if len(positive_cpu)
            else 1.0,
            "healthy_samples": int(np.count_nonzero(healthy)),
        }
    return {
        "schema_version": "headlight.capacity/v1",
        "estimator": "healthy-window-conservative",
        "sla_latency_ms": sla_latency_ms,
        "cpu_utilization_limit": cpu_limit,
        "services": services,
    }


def _float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _normalise_service(value: str) -> str:
    return value.removesuffix("-service")


def _row_feature(row: dict[str, Any], name: str, legacy: str | None = None) -> float:
    missing_value = str(row.get(f"{name}_missing", "")).strip().lower()
    if missing_value in {"1", "true", "yes"}:
        return float("nan")
    return _float_or_nan(row.get(name, row.get(legacy) if legacy else None))


def _aggregate_feature_group(group: list[float], requests: list[float], feature_name: str) -> float | None:
    finite = np.asarray([value for value in group if math.isfinite(value)], dtype=np.float32)
    if not len(finite):
        return None
    if feature_name in {"replicas", "request_rate", "cpu_usage_cores"}:
        return float(np.sum(finite))
    if feature_name in {"cpu_utilization", "p95_latency_ms"}:
        return float(np.max(finite))
    weights = np.asarray(
        [request for value, request in zip(group, requests) if math.isfinite(value) and math.isfinite(request) and request > 0],
        dtype=np.float32,
    )
    if len(weights) == len(finite) and float(np.sum(weights)) > 0:
        return float(np.average(finite, weights=weights))
    return float(np.mean(finite))


def _populate_calibration_values(
    values: np.ndarray,
    missing: np.ndarray,
    grouped: dict[tuple[int, str], list[dict[str, Any]]],
    timestamps: list[int],
    profile: ServiceProfile,
) -> None:
    timestamp_index = {value: index for index, value in enumerate(timestamps)}
    for (timestamp, service), service_rows in grouped.items():
        node_index = profile.node_order.index(service)
        target = values[timestamp_index[timestamp], node_index]
        target_missing = missing[timestamp_index[timestamp], node_index]
        means = [
            _row_feature(row, "mean_latency_ms")
            if row.get("mean_latency_ms") not in (None, "")
            else _float_or_nan(row.get("response_time_seconds")) * 1000.0
            for row in service_rows
        ]
        feature_groups = (
            [_row_feature(row, "replicas", "pod_count") for row in service_rows],
            [_row_feature(row, "request_rate") for row in service_rows],
            [_row_feature(row, "cpu_usage_cores") for row in service_rows],
            [_row_feature(row, "cpu_utilization") for row in service_rows],
            means,
            [_row_feature(row, "p95_latency_ms") for row in service_rows],
            [_row_feature(row, "error_rate") for row in service_rows],
        )
        requests = feature_groups[1]
        for feature_index, group in enumerate(feature_groups):
            aggregate = _aggregate_feature_group(group, requests, BASE_FEATURES[feature_index])
            if aggregate is None:
                continue
            target[feature_index] = aggregate
            target_missing[feature_index] = False


def load_calibration_csv(path: str | Path, profile: ServiceProfile) -> AlignedSeries:
    """Load canonical or legacy calibration rows and aggregate clusters per node."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"calibration CSV is empty: {csv_path}")
    nodes = set(profile.node_order)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        timestamp_value = _float_or_nan(row.get("timestamp"))
        service = _normalise_service(str(row.get("service") or row.get("workload") or ""))
        if not math.isfinite(timestamp_value) or service not in nodes:
            continue
        grouped.setdefault((int(timestamp_value), service), []).append(row)
    timestamps = sorted({key[0] for key in grouped})
    if not timestamps:
        raise ValueError("calibration CSV has no rows matching the selected service profile")
    values = np.full((len(timestamps), len(profile.nodes), len(BASE_FEATURES)), np.nan, dtype=np.float32)
    missing = np.ones_like(values, dtype=bool)
    _populate_calibration_values(values, missing, grouped, timestamps, profile)
    return bounded_forward_fill(
        AlignedSeries(
            timestamps=np.asarray(timestamps, dtype=np.int64),
            values=values,
            missing=missing,
        ),
        max_age_seconds=30,
    )


def prometheus_queries(namespace: str, window: str = "60s") -> dict[str, str]:
    labels = f'namespace="{namespace}"'
    destination = f'destination_workload_namespace="{namespace}"'
    return {
        "replicas": f"sum by (deployment) (kube_deployment_status_replicas_ready{{{labels}}})",
        "request_rate": (
            "sum by (destination_workload) "
            f"(rate(istio_requests_total{{{destination}}}[{window}]))"
        ),
        "cpu_usage_cores": (
            "sum by (pod) "
            f'(rate(container_cpu_usage_seconds_total{{{labels},container!="",pod!=""}}[{window}]))'
        ),
        "cpu_utilization": (
            "sum by (pod) "
            f'(rate(container_cpu_usage_seconds_total{{{labels},container!="",pod!=""}}[{window}])) '
            "/ clamp_min(sum by (pod) "
            f'(kube_pod_container_resource_requests{{{labels},resource="cpu"}}), 0.001)'
        ),
        "mean_latency_ms": (
            "sum by (destination_workload) "
            f"(rate(istio_request_duration_milliseconds_sum{{{destination}}}[{window}])) "
            "/ clamp_min(sum by (destination_workload) "
            f"(rate(istio_request_duration_milliseconds_count{{{destination}}}[{window}])), 0.001)"
        ),
        "p95_latency_ms": (
            "histogram_quantile(0.95, sum by (le, destination_workload) "
            f"(rate(istio_request_duration_milliseconds_bucket{{{destination}}}[{window}])))"
        ),
        "error_rate": (
            "sum by (destination_workload) "
            f'(rate(istio_requests_total{{{destination},response_code!~"2..|3.."}}[{window}])) '
            "/ clamp_min(sum by (destination_workload) "
            f"(rate(istio_requests_total{{{destination}}}[{window}])), 0.001)"
        ),
    }


class PrometheusMetricCollector:
    """Collect one aligned feature frame and apply bounded forward fill."""

    def __init__(
        self,
        base_url: str,
        profile: ServiceProfile,
        namespace: str | None = None,
        forward_fill_seconds: int = 30,
        timeout_seconds: float = 5.0,
        query_function: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.base_urls = tuple(
            endpoint.strip().rstrip("/")
            for endpoint in base_url.split(",")
            if endpoint.strip()
        )
        if not self.base_urls:
            raise ValueError("at least one Prometheus URL is required")
        self.base_url = self.base_urls[0]
        self.profile = profile
        self.namespace = namespace or profile.namespace
        self.forward_fill_seconds = int(forward_fill_seconds)
        self.timeout_seconds = timeout_seconds
        self.query_function = query_function
        self.session = requests.Session()
        shape = (len(profile.nodes), len(BASE_FEATURES))
        self.last_values = np.full(shape, np.nan, dtype=np.float32)
        self.last_observed_at = np.full(shape, -1, dtype=np.int64)

    def _query(self, query: str, timestamp: int) -> list[dict[str, Any]]:
        if self.query_function is not None:
            return self.query_function(query, timestamp)
        params = urllib.parse.urlencode({"query": query, "time": timestamp})
        combined: list[dict[str, Any]] = []
        for base_url in self.base_urls:
            response = self.session.get(
                f"{base_url}/api/v1/query?{params}",
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed at {base_url}: {payload}")
            combined.extend(payload.get("data", {}).get("result", []))
        return combined

    def _result_node(self, metric: dict[str, Any]) -> str | None:
        name = str(
            metric.get("destination_workload")
            or metric.get("deployment")
            or metric.get("workload")
            or ""
        )
        name = _normalise_service(name)
        if name in self.profile.node_order:
            return name
        pod = str(metric.get("pod", ""))
        for candidate in sorted(self.profile.node_order, key=len, reverse=True):
            if pod.startswith(f"{candidate}-") or pod == candidate:
                return candidate
        return None

    def _collect_feature(self, feature_name: str, query: str, timestamp: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(len(self.profile.nodes), np.nan, dtype=np.float32)
        missing = np.ones(len(self.profile.nodes), dtype=bool)
        try:
            result = self._query(query, timestamp)
        except Exception:
            return values, missing
        aggregates: dict[int, list[float]] = {}
        for series in result:
            node = self._result_node(series.get("metric", {}))
            if node is None:
                continue
            raw_value = series.get("value", [None, None])
            value = _float_or_nan(raw_value[1] if len(raw_value) > 1 else None)
            if math.isfinite(value):
                aggregates.setdefault(self.profile.node_order.index(node), []).append(value)
        for node_index, node_values in aggregates.items():
            aggregate = sum(node_values) if feature_name in {"replicas", "request_rate", "cpu_usage_cores"} else max(node_values)
            values[node_index] = float(aggregate)
            missing[node_index] = False
        return values, missing

    def collect(self, timestamp: int | None = None) -> MetricSample:
        aligned_timestamp = int(timestamp if timestamp is not None else time.time())
        values = np.full_like(self.last_values, np.nan)
        missing = np.ones_like(values, dtype=bool)
        queries = prometheus_queries(self.namespace)
        for feature_index, feature_name in enumerate(BASE_FEATURES):
            feature_values, feature_missing = self._collect_feature(feature_name, queries[feature_name], aligned_timestamp)
            values[:, feature_index] = feature_values
            missing[:, feature_index] = feature_missing
            observed = ~feature_missing
            self.last_values[observed, feature_index] = feature_values[observed]
            self.last_observed_at[observed, feature_index] = aligned_timestamp
        ages = np.where(
            self.last_observed_at >= 0,
            aligned_timestamp - self.last_observed_at,
            np.iinfo(np.int32).max,
        )
        fillable = missing & (ages <= self.forward_fill_seconds) & np.isfinite(self.last_values)
        values[fillable] = self.last_values[fillable]
        stale = missing & ~fillable
        values[stale] = np.nan
        required_predictive = [
            BASE_FEATURES.index("request_rate"),
            BASE_FEATURES.index("cpu_usage_cores"),
            BASE_FEATURES.index("p95_latency_ms"),
        ]
        stale_nodes = np.any(stale[:, required_predictive], axis=1)
        return MetricSample(
            timestamp=aligned_timestamp,
            values=values,
            missing=missing,
            ages_seconds=ages.astype(np.int64),
            stale_nodes=stale_nodes,
        )


def save_scaler(path: str | Path, scaler: NodeFeatureScaler) -> None:
    Path(path).write_text(json.dumps(scaler.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_scaler(path: str | Path) -> NodeFeatureScaler:
    return NodeFeatureScaler.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
