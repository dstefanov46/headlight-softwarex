"""Metric formulas used by paper artifact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SUCCESS_STATUS_MIN = 200
SUCCESS_STATUS_MAX_EXCLUSIVE = 400


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


def is_success_status(status: str | int | None) -> bool:
    try:
        value = int(str(status))
    except (TypeError, ValueError):
        return False
    return SUCCESS_STATUS_MIN <= value < SUCCESS_STATUS_MAX_EXCLUSIVE


def _in_window(timestamp: float, started_at: float | None, finished_at: float | None) -> bool:
    if started_at is not None and timestamp < started_at:
        return False
    if finished_at is not None and timestamp > finished_at:
        return False
    return True


def _prometheus_values(payload: dict[str, Any], metric_name: str) -> list[float]:
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


def _prometheus_totals_by_timestamp(payload: dict[str, Any], metric_name: str) -> list[float]:
    """Sum a gauge split across labeled series at each scrape timestamp."""
    metric = payload.get(metric_name, {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    totals: dict[float, float] = {}
    for series in result:
        for sample in series.get("values", []):
            try:
                timestamp = float(sample[0])
                value = float(sample[1])
            except (IndexError, TypeError, ValueError):
                continue
            totals[timestamp] = totals.get(timestamp, 0.0) + value
    return [totals[timestamp] for timestamp in sorted(totals)]


def _series_samples(series: dict[str, Any]) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for sample in series.get("values", []):
        try:
            samples.append((float(sample[0]), float(sample[1])))
        except (IndexError, TypeError, ValueError):
            continue
    return sorted(samples)


def counter_delta(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    total = 0.0
    previous = samples[0][1]
    for _timestamp, value in samples[1:]:
        if value >= previous:
            total += value - previous
        else:
            total += value
        previous = value
    return total


def counter_seconds(
    payload: dict[str, Any],
    metric_name: str = "cpu_seconds",
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> float:
    metric = payload.get(metric_name, {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    total = 0.0
    for series in result:
        samples = [
            sample
            for sample in _series_samples(series)
            if _in_window(sample[0], started_at, finished_at)
        ]
        total += counter_delta(samples)
    return total


def _integrate_gauge_samples(samples: list[tuple[float, float]], started_at: float, finished_at: float) -> float:
    if finished_at <= started_at or not samples:
        return 0.0
    ordered = sorted(samples)
    clipped = [(timestamp, value) for timestamp, value in ordered if started_at <= timestamp <= finished_at]
    prior = [item for item in ordered if item[0] < started_at]
    if prior:
        clipped.insert(0, (started_at, prior[-1][1]))
    if not clipped:
        return 0.0
    if clipped[-1][0] < finished_at:
        clipped.append((finished_at, clipped[-1][1]))
    total = 0.0
    for index in range(1, len(clipped)):
        previous_timestamp, previous_value = clipped[index - 1]
        timestamp, _value = clipped[index]
        if timestamp > previous_timestamp:
            total += previous_value * (timestamp - previous_timestamp)
    return total


def pod_seconds(
    payload: dict[str, Any],
    metric_name: str = "pod_count",
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> float:
    metric = payload.get(metric_name, {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    total = 0.0
    for series in result:
        samples = _series_samples(series)
        if not samples:
            continue
        if started_at is not None and finished_at is not None:
            total += _integrate_gauge_samples(samples, started_at, finished_at)
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


def request_latencies_ms(
    path: Path,
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values: list[float] = []
        for row in reader:
            try:
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                timestamp = None
            if timestamp is not None and not _in_window(timestamp, started_at, finished_at):
                continue
            if row.get("exception"):
                continue
            if not is_success_status(row.get("status")):
                continue
            try:
                values.append(float(row["response_time_ms"]))
            except (KeyError, TypeError, ValueError):
                continue
        return values


def request_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def request_metrics(
    path: Path,
    sla_latency_ms: float,
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> dict[str, int | float | str]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _in_window(timestamp, started_at, finished_at):
                continue
            rows.append(row)
    successful_rows = [
        row
        for row in rows
        if not row.get("exception") and is_success_status(row.get("status"))
    ]
    latencies: list[float] = []
    for row in successful_rows:
        try:
            latencies.append(float(row["response_time_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
    exception_count = sum(bool(row.get("exception")) for row in rows)
    timeout_count = sum(
        "timeout" in str(row.get("exception", "")).lower()
        or str(row.get("status", "")).strip() in {"0", "408", "504"}
        for row in rows
    )
    status_distribution: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "")).strip() or "missing"
        status_distribution[status] = status_distribution.get(status, 0) + 1
    total = len(rows)
    successful = len(successful_rows)
    failed = total - successful
    if started_at is not None and finished_at is not None:
        window_seconds = max(0.0, finished_at - started_at)
    else:
        timestamps = []
        for row in rows:
            try:
                timestamps.append(float(row["timestamp"]))
            except (KeyError, TypeError, ValueError):
                pass
        window_seconds = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    return {
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "success_rate": successful / total if total else 0.0,
        "successful_throughput_rps": successful / window_seconds if window_seconds > 0 else 0.0,
        "offered_throughput_rps": total / window_seconds if window_seconds > 0 else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "sla_violation_rate": sla_violation_rate(latencies, sla_latency_ms),
        "timeout_count": int(timeout_count),
        "exception_count": exception_count,
        "http_status_distribution": json.dumps(status_distribution, sort_keys=True),
    }


def compute_summary_metrics(
    request_csv: Path,
    prometheus_json: Path,
    sla_latency_ms: float,
    action_log: Path | None = None,
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> dict[str, float | int | str]:
    requests = request_metrics(
        request_csv,
        sla_latency_ms,
        started_at=started_at,
        finished_at=finished_at,
    )
    prometheus = json.loads(prometheus_json.read_text(encoding="utf-8"))
    pod_samples = _prometheus_totals_by_timestamp(prometheus, "pod_count")
    return {
        **requests,
        "excess_response_time_seconds": excess_response_time_seconds(
            request_latencies_ms(request_csv, started_at=started_at, finished_at=finished_at),
            sla_latency_ms,
        ),
        "cpu_seconds": counter_seconds(
            prometheus,
            "cpu_seconds",
            started_at=started_at,
            finished_at=finished_at,
        ),
        "mean_pod_count": mean_pod_count(pod_samples),
        "pod_seconds": pod_seconds(prometheus, started_at=started_at, finished_at=finished_at),
        "action_churn": action_churn(action_log),
        "cpu_calculation_method": "counter_delta_with_reset_handling",
        "cpu_calculation_window": json.dumps({"started_at_unix": started_at, "finished_at_unix": finished_at}, sort_keys=True),
        "pod_seconds_calculation_method": "stepwise_gauge_integral_exact_window",
    }
