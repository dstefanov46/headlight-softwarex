#!/usr/bin/env python3
"""Guard a live qualification against sustained host memory or /tmp pressure.

The memory threshold is treated as a warning first. If MemAvailable remains
below it for the grace period, the configured testbed containers are stopped.
OOM kills, severe full-memory PSI, and critically low /tmp space abort at once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TESTBED_CONTAINERS = {
    "karmada-host-control-plane",
    "cluster1-control-plane",
    "cluster2-control-plane",
    "cluster3-control-plane",
    "headlight-karmada-api-proxy",
}


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        fields = value.split()
        if fields and key in {"MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem"}:
            values[key] = int(fields[0]) * 1024
    return values


def read_psi() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    path = Path("/proc/pressure/memory")
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        result[fields[0]] = {
            key: float(value)
            for field in fields[1:]
            if "=" in field
            for key, value in [field.split("=", 1)]
            if key.startswith("avg")
        }
    return result


def read_vmstat() -> dict[str, int]:
    wanted = {"oom_kill", "oom_kill_allocating_task", "pgmajfault"}
    result: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in wanted:
            result[fields[0]] = int(fields[1])
    return result


def stat_tmp(path: str) -> dict[str, int]:
    stats = os.statvfs(path)
    return {
        "available": stats.f_bavail * stats.f_frsize,
        "total": stats.f_blocks * stats.f_frsize,
    }


def command_output(args: list[str], timeout: int = 5) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def detailed_snapshot() -> dict[str, Any]:
    ps = command_output(
        ["ps", "-eo", "pid,ppid,comm,rss,%mem", "--sort=-rss"], timeout=3
    )
    docker_stats = command_output(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"], timeout=5
    )
    parsed_stats: list[Any] = []
    if docker_stats.get("returncode") == 0:
        for line in docker_stats.get("stdout", "").splitlines():
            try:
                parsed_stats.append(json.loads(line))
            except json.JSONDecodeError:
                parsed_stats.append({"unparsed": line})
    return {
        "meminfo": read_meminfo(),
        "memory_pressure": read_psi(),
        "vmstat": read_vmstat(),
        "top_processes": ps.get("stdout", "").splitlines()[:16],
        "top_processes_error": ps.get("error") or ps.get("stderr"),
        "docker_stats": parsed_stats,
        "docker_stats_error": docker_stats.get("error") or docker_stats.get("stderr"),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mem-limit-gib", type=float, default=6.0)
    parser.add_argument("--mem-grace-seconds", type=float, default=30.0)
    parser.add_argument("--tmp-limit-gib", type=float, default=2.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--psi-full-avg10-abort-percent", type=float, default=10.0)
    parser.add_argument("--stop-timeout-seconds", type=int, default=10)
    args = parser.parse_args()

    if args.mem_limit_gib <= 0 or args.mem_grace_seconds < 0 or args.tmp_limit_gib <= 0:
        parser.error("memory and /tmp limits must be positive; grace cannot be negative")
    if args.interval_seconds <= 0 or args.psi_full_avg10_abort_percent < 0:
        parser.error("interval must be positive and PSI threshold cannot be negative")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    abort_path = output_dir / "ABORT"
    stop_path = output_dir / "monitor-stop"
    if abort_path.exists():
        print(f"Refusing to start: {abort_path} already exists", file=sys.stderr)
        return 2

    mem_limit = int(args.mem_limit_gib * 1024**3)
    tmp_limit = int(args.tmp_limit_gib * 1024**3)
    initial_oom_kill = read_vmstat().get("oom_kill", 0)
    config = {
        "started_at": time.time(),
        "mem_limit_bytes": mem_limit,
        "mem_limit_gib": args.mem_limit_gib,
        "mem_grace_seconds": args.mem_grace_seconds,
        "tmp_limit_bytes": tmp_limit,
        "tmp_limit_gib": args.tmp_limit_gib,
        "sample_interval_seconds": args.interval_seconds,
        "psi_full_avg10_abort_percent": args.psi_full_avg10_abort_percent,
        "oom_kill_baseline": initial_oom_kill,
        "testbed_containers": sorted(TESTBED_CONTAINERS),
    }
    (output_dir / "memory-guard-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    host_log = output_dir / "host-memory.jsonl"
    detail_log = output_dir / "host-memory-diagnostics.jsonl"
    below_since: float | None = None
    previous_oom_kill = initial_oom_kill

    while not stop_path.exists():
        now = time.time()
        meminfo = read_meminfo()
        tmp = stat_tmp("/tmp")
        psi = read_psi()
        vmstat = read_vmstat()
        available = meminfo.get("MemAvailable", 0)
        oom_kill = vmstat.get("oom_kill", previous_oom_kill)
        full_avg10 = psi.get("full", {}).get("avg10", 0.0)
        low_started_at = below_since
        low_duration = 0.0

        immediate_reasons: list[str] = []
        if tmp["available"] < tmp_limit:
            immediate_reasons.append("tmp headroom below hard limit")
        if oom_kill > previous_oom_kill:
            immediate_reasons.append("host oom_kill counter increased")
        if full_avg10 >= args.psi_full_avg10_abort_percent:
            immediate_reasons.append("severe full-memory PSI pressure")

        state = "ok"
        abort_reason: str | None = None
        if available < mem_limit:
            if below_since is None:
                below_since = now
            elapsed = now - below_since
            low_duration = elapsed
            state = "memory-warning" if elapsed < args.mem_grace_seconds else "memory-limit-persistent"
            if elapsed >= args.mem_grace_seconds:
                abort_reason = "MemAvailable remained below limit for grace period"
        elif below_since is not None:
            low_duration = now - below_since
            state = "memory-recovered"
            below_since = None

        if immediate_reasons:
            abort_reason = "; ".join(immediate_reasons)
            state = "immediate-abort"

        event: dict[str, Any] = {
            "time": now,
            "state": state,
            "mem_available": available,
            "mem_limit": mem_limit,
            "mem_below_since": low_started_at,
            "mem_below_duration_seconds": low_duration,
            "mem_grace_seconds": args.mem_grace_seconds,
            "shmem": meminfo.get("Shmem"),
            "tmp_available": tmp["available"],
            "tmp_limit": tmp_limit,
            "memory_pressure": psi,
            "vmstat": vmstat,
        }

        if state != "ok":
            append_jsonl(
                detail_log,
                {"time": now, "state": state, "abort_reason": abort_reason, **detailed_snapshot()},
            )

        if abort_reason is not None:
            event["abort"] = abort_reason
            abort_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            names = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            running = names.stdout.splitlines() if names.returncode == 0 else []
            selected = sorted(TESTBED_CONTAINERS.intersection(running))
            stop_result: dict[str, Any] = {"selected_containers": selected}
            if names.returncode != 0:
                stop_result["list_error"] = names.stderr
            elif selected:
                stopped = subprocess.run(
                    ["docker", "stop", "--time", str(args.stop_timeout_seconds), *selected],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(60, args.stop_timeout_seconds * len(selected) + 10),
                )
                stop_result.update(
                    returncode=stopped.returncode,
                    stdout=stopped.stdout,
                    stderr=stopped.stderr,
                )
            event["stop_result"] = stop_result

        append_jsonl(host_log, event)
        previous_oom_kill = oom_kill
        if abort_reason is not None:
            return 1
        time.sleep(args.interval_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
