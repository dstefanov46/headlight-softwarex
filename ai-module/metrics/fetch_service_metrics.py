#!/usr/bin/env python3
"""Collect aligned 15-second Prometheus feature frames for one service profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


AI_MODULE = Path(__file__).resolve().parents[1] / "ai-module"
sys.path.insert(0, str(AI_MODULE))

from headlight.features import BASE_FEATURES, PrometheusMetricCollector
from headlight.profile import load_service_profile


def record(sample, profile) -> dict[str, object]:
    nodes = {}
    for node_index, node in enumerate(profile.nodes):
        nodes[node.name] = {
            "values": {
                feature: (
                    float(sample.values[node_index, feature_index])
                    if math.isfinite(float(sample.values[node_index, feature_index]))
                    else None
                )
                for feature_index, feature in enumerate(BASE_FEATURES)
            },
            "missing": {
                feature: bool(sample.missing[node_index, feature_index])
                for feature_index, feature in enumerate(BASE_FEATURES)
            },
            "ages_seconds": {
                feature: int(sample.ages_seconds[node_index, feature_index])
                for feature_index, feature in enumerate(BASE_FEATURES)
            },
            "predictive_stale": bool(sample.stale_nodes[node_index]),
        }
    return {
        "schema_version": "headlight.metric-sample/v1",
        "timestamp": sample.timestamp,
        "profile_hash": profile.profile_hash,
        "node_order": list(profile.node_order),
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-profile",
        default=os.environ.get("HEADLIGHT_SERVICE_PROFILE"),
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get(
            "HEADLIGHT_PROMETHEUS_URL",
            "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
        ),
    )
    parser.add_argument("--namespace", default=os.environ.get("HEADLIGHT_NAMESPACE"))
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--forward-fill-seconds", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--service-file", help=argparse.SUPPRESS)
    parser.add_argument("--scaling-service", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.service_profile:
        parser.error("--service-profile or HEADLIGHT_SERVICE_PROFILE is required")
    profile = load_service_profile(args.service_profile)
    if args.interval_seconds != profile.sampling_interval_seconds:
        parser.error("collector interval must equal the service profile sampling interval")
    collector = PrometheusMetricCollector(
        args.prometheus_url,
        profile,
        namespace=args.namespace,
        forward_fill_seconds=args.forward_fill_seconds,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    next_tick = math.ceil(time.time() / args.interval_seconds) * args.interval_seconds
    while True:
        delay = next_tick - time.time()
        if delay > 0 and not args.once:
            time.sleep(delay)
        payload = record(collector.collect(int(next_tick)), profile)
        line = json.dumps(payload, sort_keys=True)
        if args.output:
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(line, flush=True)
        if args.once:
            return 0
        next_tick += args.interval_seconds


if __name__ == "__main__":
    raise SystemExit(main())
