#!/usr/bin/env python3
"""Bounded fixed-arrival TrainTicket load driver with explicit accounting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import requests


VERIFY_CODE = "025080CF8BA94594B09E283F17815444"
EVENT_FIELDS = (
    "phase",
    "sequence",
    "scheduled_epoch",
    "enqueued_epoch",
    "actual_start_epoch",
    "completed_epoch",
    "queue_delay_ms",
    "response_time_ms",
    "status_code",
    "outcome",
    "error",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def authenticate(base_url: str, timeout_seconds: float) -> tuple[str, dict[str, Any]]:
    started = time.time()
    response = requests.post(
        f"{base_url}/api/v1/users/login",
        headers={"Content-Type": "application/json", "X-Verify-Code": VERIFY_CODE},
        json={"username": "fdse_microservice", "password": "111111"},
        timeout=timeout_seconds,
    )
    elapsed_ms = (time.time() - started) * 1000.0
    response.raise_for_status()
    payload = response.json()
    token = str(payload.get("data", {}).get("token", ""))
    if not token:
        raise RuntimeError("authentication succeeded without returning a token")
    return token, {
        "started_epoch": started,
        "completed_epoch": time.time(),
        "response_time_ms": elapsed_ms,
        "status_code": response.status_code,
    }


def run_fixed_arrival(
    *,
    phase: str,
    rate_rps: float,
    duration_seconds: float,
    max_concurrency: int,
    queue_capacity: int,
    request_timeout_seconds: float,
    drain_timeout_seconds: float,
    request: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Schedule every arrival independently of response completion.

    ``request`` is a zero-argument callable returning an object with a
    ``status_code`` attribute. Queue saturation is an explicit dropped arrival.
    Worker threads are daemonized so the bounded drain deadline remains a hard
    upper bound even if a client violates its timeout contract.
    """
    if rate_rps <= 0 or duration_seconds <= 0:
        raise ValueError("rate and duration must be positive")
    if max_concurrency < 1 or queue_capacity < 1:
        raise ValueError("concurrency and queue capacity must be positive")
    scheduled_count = int(math.ceil(rate_rps * duration_seconds - 1e-12))
    work: queue.Queue[int] = queue.Queue(maxsize=queue_capacity)
    records: list[dict[str, Any]] = [
        {
            "phase": phase,
            "sequence": sequence,
            "scheduled_epoch": None,
            "enqueued_epoch": None,
            "actual_start_epoch": None,
            "completed_epoch": None,
            "queue_delay_ms": None,
            "response_time_ms": None,
            "status_code": None,
            "outcome": None,
            "error": "",
        }
        for sequence in range(scheduled_count)
    ]
    lock = threading.Lock()
    stop_workers = threading.Event()
    concurrency = {"active": 0, "peak_active": 0, "peak_queued": 0, "peak_outstanding": 0}

    def worker() -> None:
        while not stop_workers.is_set():
            try:
                sequence = work.get(timeout=0.1)
            except queue.Empty:
                continue
            row = records[sequence]
            started_epoch = time.time()
            with lock:
                if row["outcome"] is not None:
                    work.task_done()
                    continue
                row["actual_start_epoch"] = started_epoch
                row["queue_delay_ms"] = (started_epoch - float(row["scheduled_epoch"])) * 1000.0
                concurrency["active"] += 1
                concurrency["peak_active"] = max(concurrency["peak_active"], concurrency["active"])
                concurrency["peak_outstanding"] = max(
                    concurrency["peak_outstanding"], concurrency["active"] + work.qsize()
                )
            request_started = time.monotonic()
            try:
                response = request()
                status_code = int(response.status_code)
                outcome = "success" if 200 <= status_code < 400 else "http_error"
                error = ""
            except requests.Timeout as exc:
                status_code = None
                outcome = "timeout"
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # recorded as a delivery failure
                status_code = None
                outcome = "error"
                error = f"{type(exc).__name__}: {exc}"
            completed_epoch = time.time()
            with lock:
                if row["outcome"] is None:
                    row.update(
                        {
                            "completed_epoch": completed_epoch,
                            "response_time_ms": (time.monotonic() - request_started) * 1000.0,
                            "status_code": status_code,
                            "outcome": outcome,
                            "error": error,
                        }
                    )
                concurrency["active"] -= 1
            work.task_done()

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max_concurrency)]
    for thread in workers:
        thread.start()

    start_monotonic = time.monotonic()
    start_epoch = time.time()
    for sequence in range(scheduled_count):
        due_monotonic = start_monotonic + sequence / rate_rps
        remaining = due_monotonic - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        scheduled_epoch = start_epoch + sequence / rate_rps
        enqueued_epoch = time.time()
        row = records[sequence]
        row["scheduled_epoch"] = scheduled_epoch
        try:
            work.put_nowait(sequence)
            row["enqueued_epoch"] = enqueued_epoch
            with lock:
                concurrency["peak_queued"] = max(concurrency["peak_queued"], work.qsize())
                concurrency["peak_outstanding"] = max(
                    concurrency["peak_outstanding"], concurrency["active"] + work.qsize()
                )
        except queue.Full:
            row.update(
                {
                    "completed_epoch": enqueued_epoch,
                    "queue_delay_ms": (enqueued_epoch - scheduled_epoch) * 1000.0,
                    "outcome": "dropped",
                    "error": "bounded dispatch queue was full",
                }
            )

    scheduling_finished_epoch = time.time()
    with lock:
        outstanding_at_drain_start = sum(row["outcome"] is None for row in records)
    drain_deadline = time.monotonic() + drain_timeout_seconds
    while time.monotonic() < drain_deadline:
        with lock:
            outstanding = [row for row in records if row["outcome"] is None]
        if not outstanding:
            break
        time.sleep(0.05)

    drain_finished_epoch = time.time()
    with lock:
        for row in records:
            if row["outcome"] is None:
                row.update(
                    {
                        "completed_epoch": drain_finished_epoch,
                        "outcome": "incomplete",
                        "error": "request remained outstanding at bounded drain deadline",
                    }
                )
    stop_workers.set()

    counts = {name: sum(row["outcome"] == name for row in records) for name in (
        "success", "http_error", "timeout", "error", "dropped", "incomplete"
    )}
    dispatched = sum(row["actual_start_epoch"] is not None for row in records)
    completed = sum(row["outcome"] not in {"dropped", "incomplete"} for row in records)
    queue_delays = [float(row["queue_delay_ms"]) for row in records if row["queue_delay_ms"] is not None]
    response_times = [float(row["response_time_ms"]) for row in records if row["response_time_ms"] is not None]
    accounted = sum(counts.values())
    summary = {
        "phase": phase,
        "rate_rps": rate_rps,
        "duration_seconds": duration_seconds,
        "max_concurrency": max_concurrency,
        "queue_capacity": queue_capacity,
        "request_timeout_seconds": request_timeout_seconds,
        "drain_timeout_seconds": drain_timeout_seconds,
        "scheduled_arrivals": scheduled_count,
        "dispatched_requests": dispatched,
        "completed_requests": completed,
        **{f"{name}_requests": value for name, value in counts.items()},
        "delivery_failures": scheduled_count - counts["success"],
        "peak_active_requests": concurrency["peak_active"],
        "peak_queued_requests": concurrency["peak_queued"],
        "peak_outstanding_requests": concurrency["peak_outstanding"],
        "outstanding_at_drain_start": outstanding_at_drain_start,
        "outstanding_at_drain_end": counts["incomplete"],
        "accounted_arrivals": accounted,
        "accounting_balanced": accounted == scheduled_count,
        "dispatch_fraction": dispatched / scheduled_count if scheduled_count else 0.0,
        "success_fraction_of_arrivals": counts["success"] / scheduled_count if scheduled_count else 0.0,
        "achieved_successful_rps": counts["success"] / duration_seconds,
        "queue_delay_p95_ms": percentile(queue_delays, 0.95),
        "queue_delay_max_ms": max(queue_delays, default=None),
        "response_time_p50_ms": percentile(response_times, 0.50),
        "response_time_p95_ms": percentile(response_times, 0.95),
        "response_time_p99_ms": percentile(response_times, 0.99),
        "response_time_max_ms": max(response_times, default=None),
        "measurement_started_epoch": start_epoch,
        "scheduling_finished_epoch": scheduling_finished_epoch,
        "drain_finished_epoch": drain_finished_epoch,
    }
    return records, summary


def write_events(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--rate-rps", type=float, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--queue-capacity", type=int, default=16)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=125.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/headlight-open-loop-events.csv"))
    args = parser.parse_args()

    auth_url = f"http://ts-auth-service.{args.namespace}.svc.cluster.local:12340"
    travel_url = f"http://ts-travel-service.{args.namespace}.svc.cluster.local:12346"
    token, authentication = authenticate(auth_url, args.request_timeout_seconds)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "departureTime": "2021-07-15",
        "startingPlace": "Shang Hai",
        "endPlace": "Su Zhou",
    }
    thread_local = threading.local()

    def request() -> requests.Response:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session.post(
            f"{travel_url}/api/v1/travelservice/trips/left",
            headers=headers,
            json=payload,
            timeout=args.request_timeout_seconds,
        )

    records, summary = run_fixed_arrival(
        phase=args.phase,
        rate_rps=args.rate_rps,
        duration_seconds=args.duration_seconds,
        max_concurrency=args.max_concurrency,
        queue_capacity=args.queue_capacity,
        request_timeout_seconds=args.request_timeout_seconds,
        drain_timeout_seconds=args.drain_timeout_seconds,
        request=request,
    )
    summary["authentication"] = authentication
    write_events(args.output, records)
    print("__HEADLIGHT_OPEN_LOOP_EVENTS_BEGIN__", flush=True)
    print(args.output.read_text(encoding="utf-8"), end="", flush=True)
    print("__HEADLIGHT_OPEN_LOOP_EVENTS_END__", flush=True)
    print("__HEADLIGHT_OPEN_LOOP_SUMMARY_BEGIN__", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("__HEADLIGHT_OPEN_LOOP_SUMMARY_END__", flush=True)
    return 0 if summary["accounting_balanced"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
