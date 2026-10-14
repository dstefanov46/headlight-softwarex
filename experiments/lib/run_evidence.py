"""Bounded subprocess supervision and durable evidence for live attempts."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


ACTION_MARKER = "HEADLIGHT_ACTION_JSON "


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stop_process(process: subprocess.Popen) -> None:
    if os.name != "posix" and process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=2)
    except ProcessLookupError:
        pass
    finally:
        # An exited leader does not imply its children have left the group.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_logged_process(
    args: list[str],
    log_path: Path | None = None,
    *,
    timeout: float = 60,
    health_check: Callable[[], None] | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Keep draining output while checking process health and a wall-clock deadline."""
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8") if log_path else None
    process = None
    lines: list[str] = []
    reader = None
    messages: queue.Queue[str | None] = queue.Queue()
    try:
        if handle:
            handle.write("$ " + " ".join(args) + "\n")
            handle.flush()
        process = subprocess.Popen(
            args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1, start_new_session=os.name == "posix",
        )

        def read_output():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        while True:
            # Descendants can keep stdout busy after the command leader exits.
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(args, timeout, output="".join(lines))
            if process.poll() is None:
                if health_check:
                    health_check()
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(args, timeout, output="".join(lines))
                continue
            if line is None:
                # A command can close stdout while still running. Keep supervising it.
                while process.poll() is None:
                    if health_check:
                        health_check()
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(args, timeout, output="".join(lines))
                    time.sleep(0.1)
                break
            lines.append(line)
            if handle:
                handle.write(line)
                handle.flush()
        if handle:
            handle.write(f"[exit {process.returncode}]\n")
        return subprocess.CompletedProcess(args, process.returncode, stdout="".join(lines))
    except BaseException as exc:
        if handle:
            handle.write(f"[interrupted {type(exc).__name__}: {exc}]\n")
            handle.flush()
        raise
    finally:
        if process is not None:
            stop_process(process)
            if reader:
                reader.join(timeout=2)
                if reader.is_alive() and os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    reader.join(timeout=2)
            # Never block on BufferedReader.close() while another thread owns its IO lock.
            if process.stdout and (reader is None or not reader.is_alive()):
                process.stdout.close()
        if handle:
            handle.close()


class GatewayWatch:
    def __init__(self, process: subprocess.Popen, log_path: Path, max_staleness: float):
        self.process = process
        self.log_path = log_path
        self.max_staleness = max_staleness
        self.started_at = time.time()

    def check(self) -> None:
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(f"gateway weight updater exited during workload: {code}")
        modified = self.log_path.stat().st_mtime if self.log_path.exists() and self.log_path.stat().st_size else self.started_at
        if time.time() - modified > self.max_staleness:
            raise RuntimeError(f"gateway weights have no verified update for {self.max_staleness:g} seconds")


class ControllerActionStream:
    """Reconnect log following and copy flushed decision records onto the host."""

    def __init__(self, seed_dir: Path, command: list[str]):
        self.seed_dir = seed_dir
        self.command = command
        self.stopped = threading.Event()
        self.process: subprocess.Popen | None = None
        self.thread = threading.Thread(target=self._follow, daemon=True)
        self.seen: set[str] = set()
        self.closed = False

    def start(self) -> None:
        self.thread.start()

    def _follow(self) -> None:
        raw = self.seed_dir / "raw"
        logs = self.seed_dir / "logs"
        raw.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        with (raw / "controller_actions_live.jsonl").open("a", encoding="utf-8") as actions, (raw / "controller_proposals_live.jsonl").open("a", encoding="utf-8") as proposals, (raw / "controller_diagnostics_live.jsonl").open("a", encoding="utf-8") as diagnostics, (logs / "controller-stream.log").open("a", encoding="utf-8") as logs_out:
            while not self.stopped.is_set():
                try:
                    self.process = subprocess.Popen(self.command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, start_new_session=os.name == "posix")
                    # stop() can race with process creation.
                    if self.stopped.is_set():
                        stop_process(self.process)
                    for line in self.process.stdout:
                        logs_out.write(line)
                        logs_out.flush()
                        marker = next((m for m in (ACTION_MARKER, "HEADLIGHT_PROPOSAL_JSON ", "HEADLIGHT_V1_DIAGNOSTIC_JSON ", "HEADLIGHT_V1_DIAGNOSTIC_ERROR ") if m in line), None)
                        if marker is None:
                            continue
                        try:
                            record = json.loads(line.split(marker, 1)[1])
                            required = {"timestamp"} if marker == "HEADLIGHT_V1_DIAGNOSTIC_ERROR " else {"timestamp", "controller"}
                            if marker in {ACTION_MARKER, "HEADLIGHT_PROPOSAL_JSON "}:
                                required.add("service")
                            if not isinstance(record, dict) or not required <= record.keys():
                                raise ValueError("missing action identity")
                            encoded = json.dumps(record, sort_keys=True)
                            key = hashlib.sha256((marker + encoded).encode()).hexdigest()
                            if key not in self.seen:
                                target = actions if marker == ACTION_MARKER else proposals if marker == "HEADLIGHT_PROPOSAL_JSON " else diagnostics
                                target.write(encoded + "\n")
                                target.flush()
                                os.fsync(target.fileno())
                                self.seen.add(key)
                        except (ValueError, TypeError) as exc:
                            logs_out.write(f"[action decode error: {exc}]\n")
                    self.process.wait()
                    logs_out.write(f"[stream exit {self.process.returncode}; reconnecting]\n")
                    logs_out.flush()
                except Exception as exc:
                    logs_out.write(f"[stream error {type(exc).__name__}: {exc}]\n")
                    logs_out.flush()
                finally:
                    if self.process:
                        stop_process(self.process)
                        if self.process.stdout:
                            self.process.stdout.close()
                self.stopped.wait(1)

    def stop(self) -> None:
        if self.closed:
            return
        self.stopped.set()
        if self.process:
            stop_process(self.process)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("controller log stream did not stop")
        merge_controller_actions(self.seed_dir)
        self.closed = True


def merge_controller_actions(seed_dir: Path, collected_text: str = "") -> None:
    raw = seed_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    texts = [collected_text]
    for name in ["controller_actions.jsonl", "controller_actions_live.jsonl"]:
        path = raw / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8"))
    for text in texts:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("malformed controller action evidence") from exc
            if not isinstance(record, dict):
                raise ValueError("controller action evidence must contain JSON objects")
            records[json.dumps(record, sort_keys=True)] = record
    path = raw / "controller_actions.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sorted(records.values(), key=lambda r: float(r.get("timestamp", 0))):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RunEvidence:
    def __init__(self, seed_dir: Path, warmup_seconds: int, measurement_seconds: int,
                 observation_tail_seconds: int = 0,
                 observation_tail_mode: str | None = None,
                 load_phases: list[dict[str, Any]] | None = None):
        self.seed_dir = seed_dir
        self.warmup_seconds = warmup_seconds
        self.measurement_seconds = measurement_seconds
        self.observation_tail_seconds = observation_tail_seconds
        self.observation_tail_mode = observation_tail_mode or (
            "continue_workload" if observation_tail_seconds else "disabled"
        )
        self.load_phases = list(load_phases or [])
        self.phases: dict[str, dict[str, int]] = {}
        self.current_phase = "initialization"
        self.started_at = int(time.time())
        self.workload_started_at: int | None = None
        self.workload_finished_at: int | None = None
        self.status = {"technical_validity": "running", "started_at_unix": self.started_at}
        self.action_stream: ControllerActionStream | None = None
        self.failure_collected = False
        self.locust_context = "cluster1"
        self.monitor_stop = threading.Event()
        self.monitor: threading.Thread | None = None
        atomic_json(seed_dir / "raw/attempt_status.json", self.status)
        self._append_status_event()

    def _append_status_event(self) -> None:
        with (self.seed_dir / "raw/attempt_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.status, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def start_host_recording(self) -> None:
        def record():
            with (self.seed_dir / "raw/host_resources.jsonl").open("a", encoding="utf-8") as handle:
                while not self.monitor_stop.is_set():
                    sample = {"timestamp": time.time(), "cpu_count": os.cpu_count()}
                    try:
                        sample["memory_kib"] = {
                            line.split(":", 1)[0]: int(line.split()[1])
                            for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.split(":", 1)[0] in {"MemTotal", "MemAvailable", "Shmem", "SwapTotal", "SwapFree"}
                        }
                        sample["memory_pressure"] = Path("/proc/pressure/memory").read_text().strip()
                        sample["filesystems"] = {}
                        for path in ["/", "/tmp"]:
                            stats = os.statvfs(path)
                            sample["filesystems"][path] = {"total_bytes": stats.f_blocks * stats.f_frsize, "available_bytes": stats.f_bavail * stats.f_frsize}
                    except (OSError, AttributeError, ValueError) as exc:
                        sample["collection_error"] = str(exc)
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.monitor_stop.wait(5)
        self.monitor = threading.Thread(target=record, daemon=True)
        self.monitor.start()

    def stop_host_recording(self) -> None:
        self.monitor_stop.set()
        if self.monitor:
            self.monitor.join(timeout=5)

    def phase(self, name: str, event: str) -> None:
        self.current_phase = name
        self.phases.setdefault(name, {})[event] = int(time.time())
        atomic_json(self.seed_dir / "raw/phase_timestamps.json", self.phases)

    def begin_workload(self, timestamp: int) -> None:
        self.workload_started_at = timestamp
        self.phase("warmup", "started_at_unix")
        self._window(None)

    def finish_workload(self, timestamp: int) -> None:
        self.workload_finished_at = timestamp
        self._window(timestamp)

    def check_measurement_phase(self) -> None:
        if self.workload_started_at is None:
            return
        boundary = self.workload_started_at + self.warmup_seconds
        if int(time.time()) >= boundary and "measurement" not in self.phases:
            self.current_phase = "measurement"
            self.phases["warmup"]["finished_at_unix"] = boundary
            self.phases["measurement"] = {"started_at_unix": boundary}
            atomic_json(self.seed_dir / "raw/phase_timestamps.json", self.phases)

    def _window(self, finished_at: int | None) -> None:
        start = self.workload_started_at
        measurement_start = start + self.warmup_seconds if start is not None else None
        payload = {
            "locust_started_at_unix": start, "locust_finished_at_unix": finished_at,
            "started_at_unix": measurement_start,
            "finished_at_unix": min(finished_at, measurement_start + self.measurement_seconds) if finished_at is not None and measurement_start is not None and finished_at >= measurement_start else None,
            "warmup_seconds": self.warmup_seconds, "measurement_seconds": self.measurement_seconds,
            "locust_run_seconds": self.warmup_seconds + self.measurement_seconds + self.observation_tail_seconds,
            "diagnostic_observation_tail_seconds": self.observation_tail_seconds,
            "observation_tail_mode": self.observation_tail_mode,
        }
        if self.load_phases:
            payload["load_phases"] = []
            for phase in self.load_phases:
                resolved = dict(phase)
                if measurement_start is not None:
                    resolved["started_at_unix"] = measurement_start + int(phase["start_offset_seconds"])
                    resolved["finished_at_unix"] = measurement_start + int(phase["end_offset_seconds"])
                payload["load_phases"].append(resolved)
        atomic_json(self.seed_dir / "raw/measurement_window.json", payload)

    def fail(self, exc: BaseException) -> None:
        if self.workload_started_at is not None:
            if self.current_phase == "warmup":
                self.check_measurement_phase()
            if self.workload_finished_at is None:
                self.finish_workload(int(time.time()))
        self.status.update(technical_validity="invalid", failure_phase=self.current_phase,
                           invalidity_reason=f"{type(exc).__name__}: {exc}", finished_at_unix=int(time.time()))
        atomic_json(self.seed_dir / "raw/attempt_status.json", self.status)
        self._append_status_event()

    def complete(self) -> None:
        self.status.update(technical_validity="valid", finished_at_unix=int(time.time()))
        atomic_json(self.seed_dir / "raw/attempt_status.json", self.status)
        self._append_status_event()


def collect_failed_attempt(manifest: dict, evidence: RunEvidence, queries: dict[str, str]) -> None:
    """Collect independent diagnostics within a shared deadline; retain each error."""
    if evidence.failure_collected:
        return
    evidence.failure_collected = True
    seed_dir = evidence.seed_dir
    namespace = manifest["spec"]["topology"]["namespace"]
    controller = manifest["spec"]["controller"]["controller"]
    controller_context = os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3")
    controller_namespace = os.environ.get("HEADLIGHT_CONTROLLER_NAMESPACE", namespace)
    deployment = {"headlight": "ai-module", "headlight-v2": "ai-module-v2", "lstm": "lstm-autoscaler", "threshold": "threshold-autoscaler"}.get(controller)
    locust_context = evidence.locust_context
    deadline = time.monotonic() + float(os.environ.get("HEADLIGHT_FAILURE_COLLECTION_TIMEOUT_SECONDS", "30"))
    results: dict[str, dict] = {}

    def collect(name: str, args: list[str], destination: Path | None = None) -> str | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results[name] = {"status": "skipped", "reason": "collection deadline exhausted"}
            return None
        try:
            result = run_logged_process(args, seed_dir / "logs" / ("failure-collection-" + name + ".log"), timeout=min(5, remaining))
            results[name] = {"status": "collected" if result.returncode == 0 else "failed", "exit_code": result.returncode}
            if result.returncode != 0:
                results[name]["error"] = result.stdout[-2000:]
                return None
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(result.stdout, encoding="utf-8")
            return result.stdout
        except Exception as exc:
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            return None

    for name in ["meminfo", "vmstat", "pressure/memory"]:
        try:
            source = Path("/proc") / name
            (seed_dir / "raw" / ("failure-host-" + name.replace("/", "-") + ".txt")).write_text(source.read_text(), encoding="utf-8")
            results["host_" + name] = {"status": "collected"}
        except OSError as exc:
            results["host_" + name] = {"status": "failed", "error": str(exc)}
    telemetry: dict = {}

    def locust_diagnostics() -> None:
        # Snapshot partial requests before deleting the Job's ephemeral pod.
        if evidence.workload_started_at is not None:
            collect("locust_partial_requests", ["kubectl", "--context", locust_context, "-n", namespace, "exec", "job/headlight-paper-locust", "--", "cat", "/app/headlight_requests.csv"], seed_dir / "raw/request_events_partial.csv")
            collect("locust_stop", ["kubectl", "--context", locust_context, "-n", namespace, "delete", "job/headlight-paper-locust", "--ignore-not-found", "--wait=false"])

    def controller_actions() -> None:
        actions = collect("controller_actions", ["kubectl", "--context", controller_context, "-n", controller_namespace, "exec", "deploy/" + deployment, "--", "sh", "-lc", "cat /artifacts/actions-v2.jsonl 2>/dev/null || cat /artifacts/actions.jsonl"])
        try:
            merge_controller_actions(seed_dir, collected_text=actions or "")
            results["controller_action_merge"] = {"status": "collected"}
        except Exception as exc:
            results["controller_action_merge"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def prometheus_diagnostics(context: str, name: str, query: str) -> None:
        interval = manifest["spec"]["run"].get("sampling_interval_seconds", 15)
        service = os.environ.get("HEADLIGHT_PROMETHEUS_SERVICE", "prometheus-kube-prometheus-prometheus")
        prom_namespace = os.environ.get("HEADLIGHT_PROMETHEUS_NAMESPACE", "monitoring")
        params = urllib.parse.urlencode({"query": query, "start": evidence.workload_started_at, "end": int(time.time()), "step": f"{interval}s"})
        route = f"/api/v1/namespaces/{prom_namespace}/services/http:{service}:9090/proxy/api/v1/query_range?{params}"
        stage = "prometheus_" + context + "_" + name
        response = collect(stage, ["kubectl", "--context", context, "get", "--raw=" + route])
        if response:
            try:
                telemetry.setdefault(context, {})[name] = {"query": query, "response": json.loads(response)}
            except ValueError as exc:
                results[stage] = {"status": "failed", "error": str(exc)}

    try:
        # Dead APIs must not consume the entire budget before independent host or
        # healthy member diagnostics get a chance to run.
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = []

            def submit(name: str, callback: Callable, *args) -> None:
                tasks.append((name, executor.submit(callback, *args)))

            submit("kernel", collect, "kernel", ["journalctl", "-k", "--since=@" + str(evidence.started_at), "--no-pager", "-o", "short-iso"], seed_dir / "logs/kernel-failure.log")
            submit("docker", collect, "docker", ["docker", "stats", "--no-stream", "--format", "{{json .}}"], seed_dir / "logs/docker-memory-failure.log")
            for context in ["karmada-apiserver", *manifest["spec"]["topology"]["active_clusters"]]:
                resource = "controlledreplicapropagations" if context == "karmada-apiserver" else "pods,deployments,events"
                submit("state_" + context, collect, "state_" + context, ["kubectl", "--context", context, "-n", namespace, "get", resource, "-o", "json"], seed_dir / "raw" / ("failure-state-" + context + ".json"))
            if evidence.workload_started_at is not None:
                submit("locust_diagnostics", locust_diagnostics)
            if deployment and evidence.action_stream is not None:
                submit("controller_actions", controller_actions)
                submit("controller_logs", collect, "controller_logs", ["kubectl", "--context", controller_context, "-n", controller_namespace, "logs", "deployment/" + deployment, "--tail=2000"], seed_dir / "logs/controller-failure.log")
            if evidence.workload_started_at is not None:
                submit("gateway_logs", collect, "gateway_logs", ["kubectl", "--context", "cluster1", "-n", namespace, "logs", "deployment/trainticket-gateway", "--tail=2000"], seed_dir / "logs/gateway-failure.log")
                submit("actions_handler_logs", collect, "actions_handler_logs", ["kubectl", "--context", "karmada-host", "-n", "actions-handler-system", "logs", "-l", "app.kubernetes.io/name=actions-handler", "--tail=2000"], seed_dir / "logs/actions-handler-failure.log")
                for name, query in queries.items():
                    for context in manifest["spec"]["topology"]["active_clusters"]:
                        submit("prometheus_" + context + "_" + name, prometheus_diagnostics, context, name, query)
            for name, task in tasks:
                try:
                    task.result()
                except Exception as exc:
                    results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if evidence.workload_started_at is not None:
            atomic_json(seed_dir / "raw/prometheus_partial_data.json", telemetry)
        try:
            merge_controller_actions(seed_dir)
            results["action_merge"] = {"status": "collected"}
        except Exception as exc:
            results["action_merge"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(seed_dir / "raw/failure_collection.json", {"stages": results, "complete": all(r["status"] == "collected" for r in results.values())})
