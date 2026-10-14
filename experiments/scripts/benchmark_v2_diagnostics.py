#!/usr/bin/env python3
"""Measure offline instrumentation overhead; no workload, training or actuation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-module/ai-module"))
from headlight.policy_v2 import PolicyV2Config, ServiceRuntimeStateV2  # noqa: E402
from headlight.profile import load_service_profile  # noqa: E402
from headlight.runtime_v2 import HeadlightV2Runtime  # noqa: E402
from replay_v2_diagnostics import replay  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/headlight-v2-live/bundle-full-seed-601")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = load_service_profile(ROOT / "ai-module/configs/service_profiles/advanced-ticket.json")
    def runtime(diagnostics):
        result = HeadlightV2Runtime.from_bundle(args.bundle, runtime_profile=profile,
            prometheus_url="http://cluster2.invalid,http://cluster3.invalid", prometheus_clusters=("cluster2", "cluster3"),
            action_handler_url="http://unused.invalid", policy_config=PolicyV2Config(), run_id="offline-overhead",
            seed=401, shadow_mode=True, device=torch.device("cpu"),
            diagnostic_dir=args.output / "diagnostics" if diagnostics else None)
        steps, nodes = result.lookback, len(profile.nodes)
        result.prime(np.zeros((steps, nodes, len(result.bundle.model_config["node_feature_names"])), dtype=np.float32),
                     np.zeros((steps, nodes, nodes, len(result.bundle.model_config["edge_feature_names"])), dtype=np.float32))
        return result
    baseline, observed = runtime(False), runtime(True)
    placement = {"cluster2": 1, "cluster3": 1}
    states = {node.name: ServiceRuntimeStateV2(desired_replicas=placement, ready_replicas=placement,
        pending_replicas=dict.fromkeys(placement, 0), pending_age_seconds=dict.fromkeys(placement, 0.),
        demand_rps=1., cpu_utilization=.95 if node.name == "ts-auth" else .3, queue_depth=0., p95_latency_ms=100.,
        error_rate=0., predictive_fresh=True, time_since_last_action_seconds=600., cluster_cpu_utilization={"cluster2": .95})
        for node in profile.nodes}
    measurements = {"baseline_wall_ms": [], "diagnostic_wall_ms": [], "baseline_cpu_ms": [], "diagnostic_cpu_ms": []}
    for index in range(args.intervals):
        results = {}
        # Alternate timing order to reduce systematic first-run effects.
        order = (("baseline", baseline), ("diagnostic", observed))
        if index % 2:
            order = order[::-1]
        for name, controller in order:
            wall, cpu = time.perf_counter(), time.process_time()
            with contextlib.redirect_stdout(io.StringIO()):
                results[name] = controller.infer_states(states, timestamp=1000 + index * 15)
            measurements[f"{name}_wall_ms"].append((time.perf_counter() - wall) * 1000)
            measurements[f"{name}_cpu_ms"].append((time.process_time() - cpu) * 1000)
        if results["baseline"]["decisions"] != results["diagnostic"]["decisions"] or baseline.policy.memory != observed.policy.memory:
            raise RuntimeError("instrumentation changed hybrid control behavior")
    records = [json.loads(line) for line in (args.output / "diagnostics/decisions.jsonl").read_text().splitlines()]
    sample = next(r for r in records if r["decision"]["service"] == "ts-auth")
    (args.output / "sample-record.json").write_text(json.dumps(sample, indent=2) + "\n")
    verified = replay(sample, args.output / "diagnostics", args.bundle)
    report = {"scope": "offline synthetic zero-input windows with the frozen checkpoint; not live workload overhead",
        "intervals": args.intervals, "torch_version": torch.__version__, "measurements": measurements,
        "median_ms": {key: statistics.median(values) for key, values in measurements.items()},
        "decisions_and_memory_identical": True, "replay": verified,
        "diagnostic_bytes": observed.diagnostic_store.used_bytes}
    (args.output / "overhead.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "measurements"}, indent=2))


if __name__ == "__main__":
    main()
