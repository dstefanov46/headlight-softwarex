#!/usr/bin/env python3
"""Build AdapGLA calibration histories from exported Prometheus range data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.paths import resolve_output_path, resolve_repo_path


FEATURE_COLUMNS = ["pod_count", "response_time_seconds", "request_rate", "cpu_usage_cores"]
CANONICAL_FEATURE_COLUMNS = [
    "replicas",
    "request_rate",
    "cpu_usage_cores",
    "cpu_utilization",
    "mean_latency_ms",
    "p95_latency_ms",
    "error_rate",
]


def normalise_service_name(name: str) -> str:
    return name.replace("-service", "")


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def write_npy_float32_matrix(path: Path, rows: list[list[float]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty NumPy matrix")
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError("all NumPy matrix rows must have the same width")

    header = {
        "descr": "<f4",
        "fortran_order": False,
        "shape": (len(rows), column_count),
    }
    header_text = repr(header)
    header_len = len(header_text) + 1
    padding = (16 - ((10 + header_len) % 16)) % 16
    encoded_header = (header_text + (" " * padding) + "\n").encode("latin1")

    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(struct.pack("<H", len(encoded_header)))
        handle.write(encoded_header)
        for row in rows:
            handle.write(struct.pack(f"<{column_count}f", *row))


def load_services(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    names = payload.get("node_order") or payload.get("service_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} is neither a service profile nor a service-name file")
    return [normalise_service_name(str(name)) for name in names]


def result_series(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return payload.get(key, {}).get("response", {}).get("data", {}).get("result", [])


def service_from_pod(pod: str, services: list[str]) -> str | None:
    candidates = sorted(services, key=len, reverse=True)
    for service in candidates:
        if pod.startswith(service):
            return service
    return None


def read_cluster_payloads(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in payload.items():
        if isinstance(value, dict) and "cluster" in value and "metric" in value:
            clusters[str(value["cluster"])][str(value["metric"])] = value
        elif ":" in key:
            cluster, metric = key.split(":", 1)
            clusters[cluster][metric] = value
    if clusters:
        return dict(clusters)
    return {"cluster": payload}


def add_series_values(target: dict[tuple[str, int], float], service: str, series: dict[str, Any], factor: float = 1.0) -> None:
    for timestamp, value in series.get("values", []):
        target[(service, int(float(timestamp)))] += safe_float(value) * factor


def max_series_values(target: dict[tuple[str, int], float], service: str, series: dict[str, Any]) -> None:
    for timestamp, value in series.get("values", []):
        key = (service, int(float(timestamp)))
        target[key] = max(target.get(key, 0.0), safe_float(value))


def _series_service(series: dict[str, Any], services: list[str], label: str) -> str | None:
    metric = series.get("metric", {})
    service = normalise_service_name(metric.get(label, ""))
    return service or service_from_pod(metric.get("pod", ""), services)


def _canonical_values(
    metrics: dict[str, dict[tuple[str, int], float]], key: tuple[str, int]
) -> dict[str, float | None]:
    response_times = metrics["response_time_seconds"]
    return {
        "replicas": metrics["pod_count"].get(key),
        "request_rate": metrics["request_rate"].get(key),
        "cpu_usage_cores": metrics["cpu_usage_cores"].get(key),
        "cpu_utilization": metrics["cpu_utilization"].get(key),
        "mean_latency_ms": response_times.get(key) * 1000.0 if key in response_times else None,
        "p95_latency_ms": metrics["p95_latency_ms"].get(key),
        "error_rate": metrics["error_rate"].get(key),
    }


def _write_row(
    writer: csv.DictWriter,
    cluster: str,
    service: str,
    timestamp: int,
    canonical_values: dict[str, float | None],
) -> tuple[dict[str, Any], list[float]]:
    row = {
        "timestamp": timestamp,
        "cluster": cluster,
        "service": service,
        **{
            name: "" if value is None else value
            for name, value in canonical_values.items()
        },
        **{
            f"{name}_missing": int(value is None)
            for name, value in canonical_values.items()
        },
    }
    writer.writerow(row)
    history = [
        canonical_values["replicas"] or 0.0,
        (canonical_values["mean_latency_ms"] or 0.0) / 1000.0,
        canonical_values["request_rate"] or 0.0,
        canonical_values["cpu_usage_cores"] or 0.0,
    ]
    return row, history


def _collect_metric(
    payload: dict[str, Any],
    metric_name: str,
    services: list[str],
    label: str,
    target: dict[tuple[str, int], float],
    maximum: bool = False,
) -> None:
    for series in result_series(payload, metric_name):
        service = _series_service(series, services, label)
        if service not in services:
            continue
        (max_series_values if maximum else add_series_values)(target, service, series)


def extract_metrics(cluster_payload: dict[str, Any], services: list[str]) -> tuple[set[int], dict[str, dict[tuple[str, int], float]]]:
    metrics: dict[str, dict[tuple[str, int], float]] = {
        "pod_count": defaultdict(float),
        "request_rate": defaultdict(float),
        "response_time_seconds": defaultdict(float),
        "cpu_usage_cores": defaultdict(float),
        "cpu_utilization": defaultdict(float),
        "p95_latency_ms": defaultdict(float),
        "error_rate": defaultdict(float),
    }
    latency_sum: dict[tuple[str, int], float] = defaultdict(float)
    latency_count: dict[tuple[str, int], float] = defaultdict(float)
    timestamps: set[int] = set()

    for metric_name in ("pod_count", "cpu_usage_cores", "cpu_utilization"):
        _collect_metric(cluster_payload, metric_name, services, "deployment", metrics[metric_name], maximum=metric_name == "cpu_utilization")
    for metric_name in ("request_rate", "response_time_seconds", "p95_latency_ms", "error_rate"):
        _collect_metric(cluster_payload, metric_name, services, "destination_workload", metrics[metric_name])
    _collect_metric(cluster_payload, "response_time_sum_ms", services, "destination_workload", latency_sum)
    for key in latency_sum:
        latency_sum[key] *= 0.001
    _collect_metric(cluster_payload, "response_time_count", services, "destination_workload", latency_count)

    for key, total_seconds in latency_sum.items():
        count = latency_count.get(key, 0.0)
        if count > 0:
            metrics["response_time_seconds"][key] = total_seconds / count

    for metric_values in metrics.values():
        timestamps.update(timestamp for _, timestamp in metric_values.keys())
    return timestamps, metrics


def _collect_edges(payload: dict[str, Any], services: list[str], edge_weights: dict[tuple[str, str], float]) -> None:
    for series in result_series(payload, "request_edges"):
        labels = series.get("metric", {})
        source = normalise_service_name(labels.get("source_workload", ""))
        destination = normalise_service_name(labels.get("destination_workload", ""))
        if source not in services or destination not in services:
            continue
        for _timestamp, raw_value in series.get("values", []):
            edge_weights[(source, destination)] += safe_float(raw_value)


def _write_cluster_rows(
    writer: csv.DictWriter,
    cluster: str,
    services: list[str],
    timestamps: set[int],
    metrics: dict[str, dict[tuple[str, int], float]],
    warmup_seconds: int,
    history_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    histories_written: list[str] = []
    if not timestamps:
        return rows, histories_written
    kept_timestamps = sorted(t for t in timestamps if t >= min(timestamps) + warmup_seconds)
    for service in services:
        history: list[list[float]] = []
        for timestamp in kept_timestamps:
            key = (service, timestamp)
            canonical_values = _canonical_values(metrics, key)
            row, history_values = _write_row(writer, cluster, service, timestamp, canonical_values)
            rows.append(row)
            history.append(history_values)
        if history:
            filename = f"{service}_{cluster}_history.npy"
            write_npy_float32_matrix(history_dir / filename, history)
            histories_written.append(filename)
    return rows, histories_written


def build_dataset(
    prometheus_path: Path,
    service_file: Path,
    output_csv: Path,
    history_dir: Path,
    warmup_seconds: int,
    edge_csv_output: Path | None = None,
) -> dict[str, Any]:
    services = load_services(service_file)
    clusters = read_cluster_payloads(prometheus_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    histories_written: list[str] = []
    edge_weights: dict[tuple[str, str], float] = defaultdict(float)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "timestamp",
            "cluster",
            "service",
            *CANONICAL_FEATURE_COLUMNS,
            *(f"{name}_missing" for name in CANONICAL_FEATURE_COLUMNS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster in sorted(clusters):
            timestamps, metrics = extract_metrics(clusters[cluster], services)
            _collect_edges(clusters[cluster], services, edge_weights)
            cluster_rows, cluster_histories = _write_cluster_rows(writer, cluster, services, timestamps, metrics, warmup_seconds, history_dir)
            rows.extend(cluster_rows)
            histories_written.extend(cluster_histories)

    if edge_csv_output:
        edge_csv_output.parent.mkdir(parents=True, exist_ok=True)
        with edge_csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_workload", "destination_workload", "request_rate"],
            )
            writer.writeheader()
            for (source, destination), weight in sorted(edge_weights.items()):
                writer.writerow(
                    {
                        "source_workload": source,
                        "destination_workload": destination,
                        "request_rate": weight,
                    }
                )

    return {
        "csv": str(output_csv),
        "feature_history_dir": str(history_dir),
        "rows": len(rows),
        "history_files": len(histories_written),
        "services": len(services),
        "clusters": sorted(clusters),
        "warmup_seconds": warmup_seconds,
        "feature_order": CANONICAL_FEATURE_COLUMNS,
        "legacy_history_feature_order": FEATURE_COLUMNS,
        "edge_csv": str(edge_csv_output) if edge_csv_output else None,
        "observed_edges": len(edge_weights),
    }


def duration_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)([smh]?)", value.strip())
    if not match:
        raise ValueError(f"invalid duration: {value}")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Headlight calibration dataset")
    parser.add_argument("--prometheus-range-data", required=True, type=Path)
    parser.add_argument(
        "--service-file",
        required=True,
        type=Path,
        help="service profile or legacy JSON object containing service_names",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--feature-history-dir", required=True, type=Path)
    parser.add_argument("--warmup", default="300s")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--edge-csv-output", type=Path)
    args = parser.parse_args()
    args.prometheus_range_data = resolve_repo_path(args.prometheus_range_data, must_exist=True)
    args.service_file = resolve_repo_path(args.service_file, must_exist=True)
    args.output_csv = resolve_output_path(args.output_csv)
    args.feature_history_dir = resolve_output_path(args.feature_history_dir)
    if args.manifest_output:
        args.manifest_output = resolve_output_path(args.manifest_output)
    if args.edge_csv_output:
        args.edge_csv_output = resolve_output_path(args.edge_csv_output)

    summary = build_dataset(
        args.prometheus_range_data,
        args.service_file,
        args.output_csv,
        args.feature_history_dir,
        duration_seconds(args.warmup),
        args.edge_csv_output,
    )
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
