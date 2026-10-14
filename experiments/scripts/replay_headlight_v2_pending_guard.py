#!/usr/bin/env python3
"""Pointwise replay the pending-cluster guard over a saved v2 action trace."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.policy_v2 import PolicyV2Config  # noqa: E402


DEFAULT_RUN = REPO_ROOT / "results/headlight-v2-live/matched-c4-short-seed-101/headlight-v2/seed-101/raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=DEFAULT_RUN / "controller_actions.jsonl")
    parser.add_argument("--prometheus-range", type=Path, default=DEFAULT_RUN / "prometheus_range_data.json")
    parser.add_argument("--wait-seconds", type=int, default=PolicyV2Config().reactive_pending_wait_seconds)
    return parser.parse_args()


def load_actions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"action trace is empty: {path}")
    if any(row.get("schema_version") != "headlight.controller-action/v2" for row in rows):
        raise ValueError("action trace contains an unsupported controller-action schema")
    return sorted(rows, key=lambda row: (float(row["timestamp"]), str(row["service"])))


def load_ready_series(path: Path) -> dict[tuple[str, str], tuple[list[float], list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["pod_count"]["response"]["data"]["result"]
    series: dict[tuple[str, str], tuple[list[float], list[int]]] = {}
    for item in result:
        metric = item.get("metric", {})
        key = (str(metric.get("deployment", "")), str(metric.get("cluster", "")))
        if not all(key):
            continue
        samples = sorted((float(timestamp), int(float(value))) for timestamp, value in item.get("values", []))
        series[key] = ([sample[0] for sample in samples], [sample[1] for sample in samples])
    if not series:
        raise ValueError("Prometheus range archive has no per-deployment ready-pod series")
    return series


def ready_at(
    series: dict[tuple[str, str], tuple[list[float], list[int]]],
    service: str,
    cluster: str,
    timestamp: float,
) -> int:
    timestamps, values = series.get((service, cluster), ([], []))
    index = bisect.bisect_right(timestamps, timestamp) - 1
    return values[index] if index >= 0 else 0


def main() -> int:
    args = parse_args()
    if args.wait_seconds <= 0:
        raise ValueError("--wait-seconds must be positive")
    actions = args.actions.resolve()
    metrics = args.prometheus_range.resolve()
    input_hashes = {str(actions): sha256(actions), str(metrics): sha256(metrics)}
    rows = load_actions(actions)
    ready_series = load_ready_series(metrics)

    pending_since: dict[tuple[str, str], float] = {}
    last_timeout_bucket: dict[tuple[str, str], int] = {}
    replayed: list[dict[str, Any]] = []
    for row in rows:
        service = str(row["service"])
        timestamp = float(row["timestamp"])
        previous = row["previous_replicas"]
        desired = row["desired_replicas"]
        pending: dict[str, int] = {}
        ages: dict[str, float] = {}
        for cluster in previous:
            count = max(0, int(previous[cluster]) - ready_at(ready_series, service, cluster, timestamp))
            key = (service, cluster)
            pending[cluster] = count
            if count == 0:
                pending_since.pop(key, None)
                last_timeout_bucket.pop(key, None)
                ages[cluster] = 0.0
            else:
                pending_since.setdefault(key, timestamp)
                ages[cluster] = max(0.0, timestamp - pending_since[key])

        reason = str(row.get("policy_reason", ""))
        increased = [cluster for cluster in previous if int(desired[cluster]) > int(previous[cluster])]
        if "reactive-emergency" not in reason or not row.get("changed") or not increased:
            continue
        if len(increased) != 1:
            raise ValueError("reactive scale-out did not identify exactly one target cluster")
        cluster = increased[0]
        key = (service, cluster)
        wait_reason = None
        if pending[cluster] > 0:
            if ages[cluster] < args.wait_seconds:
                wait_reason = "pending-readiness"
            else:
                bucket = int(ages[cluster] // args.wait_seconds)
                if bucket <= last_timeout_bucket.get(key, 0):
                    wait_reason = "pending-timeout-cooldown"
                else:
                    last_timeout_bucket[key] = bucket
                    wait_reason = "pending-timeout-allow"
        replayed.append({
            "timestamp": int(timestamp),
            "service": service,
            "target_cluster": cluster,
            "previous_replicas": int(previous[cluster]),
            "ready_replicas": ready_at(ready_series, service, cluster, timestamp),
            "pending_replicas": pending[cluster],
            "pending_age_seconds": round(ages[cluster], 1),
            "guard_outcome": wait_reason or "allow-no-target-pending",
        })

    if input_hashes != {str(actions): sha256(actions), str(metrics): sha256(metrics)}:
        raise RuntimeError("an input artifact changed during replay")
    held = [row for row in replayed if row["guard_outcome"] in {"pending-readiness", "pending-timeout-cooldown"}]
    allowed = len(replayed) - len(held)
    print(json.dumps({
        "replay_type": "pointwise-observed-trace-not-closed-loop-resimulation",
        "inputs_sha256": input_hashes,
        "wait_seconds": args.wait_seconds,
        "reactive_scaleout_records": len(replayed),
        "guarded_holds": len(held),
        "allowed_records": allowed,
        "records": replayed,
        "actuator_called": False,
        "source_unchanged": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
