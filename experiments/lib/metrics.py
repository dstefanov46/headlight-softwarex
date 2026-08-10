"""Metric formulas used by measured artifact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def sla_violation_rate(latency_ms: Iterable[float], threshold_ms: float) -> float:
    values = list(latency_ms)
    if not values:
        return 0.0
    return sum(1 for value in values if value > threshold_ms) / len(values)


def excess_response_time_seconds(latency_ms: Iterable[float], threshold_ms: float) -> float:
    return sum(max(0.0, value - threshold_ms) for value in latency_ms) / 1000.0


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_pod_count(samples: Iterable[float]) -> float:
    values = list(samples)
    return mean(values) if values else 0.0


def _prometheus_values(payload: dict, metric_name: str) -> list[float]:
    metric = payload.get(metric_name, {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    values: list[float] = []
    for series in result:
        for sample in series.get("values", []):
            if len(sample) < 2:
                continue
            try:
                values.append(float(sample[1]))
            except (TypeError, ValueError):
                continue
    return values


def pod_seconds(payload: dict, metric_name: str = "pod_count") -> float:
    metric = payload.get(metric_name, {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    total = 0.0
    for series in result:
        samples: list[tuple[float, float]] = []
        for sample in series.get("values", []):
            try:
                samples.append((float(sample[0]), float(sample[1])))
            except (IndexError, TypeError, ValueError):
                continue
        if not samples:
            continue
        steps = [
            samples[index][0] - samples[index - 1][0]
            for index in range(1, len(samples))
            if samples[index][0] > samples[index - 1][0]
        ]
        interval = sorted(steps)[len(steps) // 2] if steps else 15.0
        total += sum(value for _timestamp, value in samples) * interval
    return total


def action_churn(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("batch_successful")
            and row.get("desired_replicas") != row.get("previous_replicas")
        ):
            count += 1
    return float(count)


def request_latencies_ms(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values: list[float] = []
        for row in reader:
            if row.get("exception"):
                continue
            status = row.get("status", "")
            if status and not status.startswith(("2", "3")):
                continue
            try:
                values.append(float(row["response_time_ms"]))
            except (KeyError, TypeError, ValueError):
                continue
        return values


def request_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def compute_summary_metrics(
    request_csv: Path,
    prometheus_json: Path,
    sla_latency_ms: float,
    action_log: Path | None = None,
) -> dict[str, float]:
    latencies = request_latencies_ms(request_csv)
    prometheus = json.loads(prometheus_json.read_text(encoding="utf-8"))
    cpu_samples = _prometheus_values(prometheus, "cpu_seconds")
    pod_samples = _prometheus_values(prometheus, "pod_count")
    return {
        "sla_violation_rate": sla_violation_rate(latencies, sla_latency_ms),
        "excess_response_time_seconds": excess_response_time_seconds(latencies, sla_latency_ms),
        "cpu_seconds": sum(cpu_samples),
        "mean_pod_count": mean_pod_count(pod_samples),
        "pod_seconds": pod_seconds(prometheus),
        "p95_latency_ms": percentile(latencies, 0.95),
        "action_churn": action_churn(action_log),
    }
