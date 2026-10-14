#!/usr/bin/env python3
"""External host-memory watchdog for the bounded local TrainTicket run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


GIB = 1024**3
TESTBED_CONTAINERS = (
    "cluster1-control-plane",
    "cluster2-control-plane",
    "cluster3-control-plane",
    "karmada-host-control-plane",
    "headlight-karmada-api-proxy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not expose MemAvailable")


@dataclass
class PressureState:
    low_samples: int = 0
    load_stopped: bool = False
    continued_low_samples: int = 0
    testbed_stopped: bool = False

    def observe(
        self,
        available: int,
        threshold: int,
        sustained_samples: int,
        stop_testbed_samples: int,
    ) -> list[str]:
        if available >= threshold:
            self.low_samples = 0
            self.continued_low_samples = 0
            return []

        self.low_samples += 1
        if not self.load_stopped and self.low_samples >= sustained_samples:
            self.load_stopped = True
            self.continued_low_samples = 0
            return ["stop_load"]

        if self.load_stopped and not self.testbed_stopped:
            self.continued_low_samples += 1
            if self.continued_low_samples >= stop_testbed_samples:
                self.testbed_stopped = True
                return ["stop_testbed"]
        return []


def run_capture(command: list[str], output: Path) -> None:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.write_text(result.stdout, encoding="utf-8")


def preserve_diagnostics(args: argparse.Namespace, label: str) -> None:
    directory = args.output_dir / f"pressure-{label}-{int(time.time())}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meminfo.txt").write_text(Path("/proc/meminfo").read_text(encoding="utf-8"), encoding="utf-8")
    run_capture(["docker", "stats", "--no-stream"], directory / "docker-stats.txt")
    run_capture(["docker", "ps", "--format", "{{json .}}"], directory / "docker-ps.jsonl")
    for cluster in ("cluster2", "cluster3"):
        prefix = ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", cluster]
        run_capture(prefix + ["get", "pods", "-A", "-o", "wide"], directory / f"{cluster}-pods.txt")
        run_capture(prefix + ["get", "events", "-A", "--sort-by=.lastTimestamp"], directory / f"{cluster}-events.txt")
        run_capture(prefix + ["top", "pods", "-A", "--containers"], directory / f"{cluster}-top.txt")


def append_event(path: Path, event: str, available: int, phase: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": utc_now(),
                    "event": event,
                    "phase": phase,
                    "mem_available_bytes": available,
                },
                sort_keys=True,
            )
            + "\n"
        )


def stop_load(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(args.kubeconfig),
            "--context",
            args.load_cluster,
            "-n",
            args.load_namespace or args.namespace,
            "delete",
            "job",
            args.load_job,
            "--ignore-not-found",
            "--wait=false",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_testbed() -> None:
    existing = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    ).stdout.splitlines()
    targets = [name for name in TESTBED_CONTAINERS if name in existing]
    if targets:
        subprocess.run(["docker", "stop", *targets], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--abort-file", type=Path, required=True)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--load-namespace")
    parser.add_argument("--load-cluster", default="cluster2")
    parser.add_argument("--load-job", default="headlight-local-fixed-load")
    parser.add_argument("--threshold-bytes", type=int, default=3 * GIB)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--sustained-samples", type=int, default=3)
    parser.add_argument("--stop-testbed-samples", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = args.output_dir / "watchdog-events.jsonl"
    samples = args.output_dir / "watchdog-memory.jsonl"
    state = PressureState()
    append_event(events, "watchdog_started", memory_available_bytes(), "preflight")

    while not args.stop_file.exists():
        available = memory_available_bytes()
        phase = args.phase_file.read_text(encoding="utf-8").strip() if args.phase_file.exists() else "unknown"
        with samples.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": utc_now(),
                        "phase": phase,
                        "mem_available_bytes": available,
                        "threshold_bytes": args.threshold_bytes,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        for action in state.observe(
            available,
            args.threshold_bytes,
            args.sustained_samples,
            args.stop_testbed_samples,
        ):
            preserve_diagnostics(args, action)
            append_event(events, action, available, phase)
            args.abort_file.write_text(f"{action}: available={available}\n", encoding="utf-8")
            if action == "stop_load":
                stop_load(args)
            elif action == "stop_testbed":
                stop_testbed()
        time.sleep(args.interval_seconds)

    append_event(events, "watchdog_stopped", memory_available_bytes(), "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
