#!/usr/bin/env python3
"""Isolated-process v1 recording benchmark with simulated services and real weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    path = ROOT / "ai-module/tests/test_v1_diagnostics.py"
    spec = importlib.util.spec_from_file_location("v1_verification_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def worker(enabled, steps, directory):
    helpers = fixture()
    bundle, model = helpers.load_model()
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        root = Path(temp)
        runtime, cluster, env = helpers.make_runtime(root, bundle, model, enabled)
        durations, rss = [], []
        # A real flushed output file exercises encoding/file IO, but is not a
        # Kubernetes log pipe or concurrent remote fsyncing consumer.
        with (root / "stdout.log").open("w", encoding="utf-8") as output, patch.dict(os.environ, env), redirect_stdout(output):
            for tick in range(steps):
                started = time.perf_counter()
                runtime.run_once(1000000+tick*15)
                durations.append((time.perf_counter()-started)*1000)
                rss.append(current_rss_kib())
        warm = durations[:23]
        steady = durations[23:]
        def distribution(values):
            return {"samples": len(values), "median_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, .95)), "max_ms": max(values)}
        result = {"recording_enabled": enabled, "threads": 2, "steps": steps,
            "warmup": distribution(warm), "steady": distribution(steady),
            "all_intervals": distribution(durations),
            "durations_ms": durations, "observed_rss_max_kib": max(v for v in rss if v),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "artifact_bytes": {p.name: p.stat().st_size for p in root.iterdir() if p.is_file()},
            "action_request_count": len(cluster.calls), "diagnostics_disabled": bool(runtime.diagnostics and runtime.diagnostics.disabled),
            "bundle_id": bundle.bundle_id}
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--storage-dir", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--worker", choices=["off", "on"])
    args = parser.parse_args()
    if args.steps < 24 or args.repetitions < 1:
        parser.error("at least 24 steps and one repetition required")
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker == "on", args.steps, args.storage_dir)
        return
    if not args.output:
        parser.error("--output required")
    cases = []
    for repeat in range(args.repetitions):
        order = ["off", "on"] if repeat % 2 == 0 else ["on", "off"]
        for mode in order:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                "--steps", str(args.steps), "--storage-dir", str(args.storage_dir)],
                cwd=ROOT, text=True, capture_output=True, check=True, timeout=120)
            result = json.loads(completed.stdout)
            result["repetition"] = repeat
            cases.append(result)
            print(f"repetition={repeat+1} recording={mode} steady_p95_ms={result['steady']['p95_ms']:.3f}", flush=True)
    pooled = {}
    for enabled in [False, True]:
        selected = [c for c in cases if c["recording_enabled"] == enabled]
        intervals = [d for c in selected for d in c["durations_ms"][23:]]
        pooled["on" if enabled else "off"] = {
            "steady_intervals": len(intervals), "median_ms": float(np.median(intervals)),
            "p95_ms": float(np.quantile(intervals, .95)), "max_ms": max(intervals),
            "mean_max_rss_kib": float(np.mean([c["observed_rss_max_kib"] for c in selected])),
        }
    sizes = [c["artifact_bytes"].get("diagnostics.jsonl", 0) / args.steps for c in cases if c["recording_enabled"]]
    summary = {"schema_version": "headlight.v1-recording-benchmark/v1",
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "storage_directory": str(args.storage_dir),
        "python": sys.version, "cases": cases, "pooled": pooled,
        "incremental_median_ms": pooled["on"]["median_ms"]-pooled["off"]["median_ms"],
        "on_p95_fraction_of_15_second_interval": pooled["on"]["p95_ms"]/15000,
        "observed_diagnostic_bytes_per_interval": float(np.mean(sizes)),
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
            "ai-module/tests/test_v1_diagnostics.py", "experiments/scripts/benchmark_v1_recording.py",
            "ai-module/ai-module/headlight/diagnostics_v1.py", "ai-module/ai-module/headlight/runtime.py",
            "ai-module/ai-module/headlight/features.py"]},
        "limitations": ["Telemetry, Kubernetes state and action endpoint are simulated; checkpoint and filesystem writes are real.",
            "Each case is a separate process; modes alternate order, but host contention and cache effects remain.",
            "stdout writes to a flushed local file, not the live Kubernetes logging transport or concurrent host consumer.",
            "No networking, throttled container CPU, live workload memory, or pod filesystem was qualified.",
            "Synthetic inputs assess instrumentation cost, not forecast accuracy or capacity adequacy.",
            "A 15-second maximum check is an offline timing observation, not a live cadence guarantee."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"pooled": pooled, "incremental_median_ms": summary["incremental_median_ms"],
        "diagnostic_bytes_per_interval": summary["observed_diagnostic_bytes_per_interval"]}, indent=2))


if __name__ == "__main__":
    main()
