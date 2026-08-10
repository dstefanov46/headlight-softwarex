#!/usr/bin/env python3
"""Qualify safe load, run selected controllers for one seed, and power off.

This is intentionally a single outer process: it owns status persistence,
signal handling, failure diagnostics, and the opt-in shutdown decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT, resolve_run, write_yaml
from experiments.lib.quality import BundleQualityError, capacity_is_healthy, request_summary
from experiments.run_experiment import environment, run_one_seed


SEED = 101
PILOT_LEVELS = (5, 6, 8, 10)
REQUIRED_CONTEXTS = ("karmada-apiserver", "karmada-host", "cluster1", "cluster2", "cluster3")
MEMBER_CONTEXTS = ("karmada-host", "cluster1", "cluster2", "cluster3")
TRAINTICKET_PATHS = ("login-route", "advanced-ticket", "full")
RESOURCE_PROFILES = ("local-tiny", "capacity-vm", "source")
LOCUST_MODE_BY_PATH = {
    "login-route": "login-route",
    "advanced-ticket": "advanced-ticket",
    "full": "full-session",
}
RUN_CONFIGS = {
    "headlight": "experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml",
    "hpa": "experiments/configs/runs/hpa-gradual-ramp-multi-cluster-headlight.yaml",
    "keda": "experiments/configs/runs/keda-gradual-ramp-multi-cluster-headlight.yaml",
    "lstm": "experiments/configs/runs/lstm-gradual-ramp-multi-cluster-headlight.yaml",
}
PILOT_CONFIG = "experiments/configs/runs/static-gradual-ramp-multi-cluster-static.yaml"


def _stable_order_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CampaignOptions:
    output_dir: Path
    dry_run: bool = False
    smoke: bool = False
    trainticket_path: str = "advanced-ticket"
    resource_profile: str = "capacity-vm"
    pilot_duration: int = 360
    qualification_duration: int = 1200
    cooldown_seconds: int = 30
    seed: int = SEED
    candidate_metadata: dict[str, Any] | None = None
    skip_pilots: bool = False
    selected_capacity_override: int | None = None
    controllers: tuple[str, ...] = tuple(RUN_CONFIGS)

    @property
    def locust_mode(self) -> str:
        return LOCUST_MODE_BY_PATH[self.trainticket_path]


class CampaignFailure(RuntimeError):
    pass


class PreflightFailure(RuntimeError):
    pass


def default_kubeconfig() -> Path:
    return REPO_ROOT / ".kube" / "headlight-testbed.config"


def headlight_candidate_metadata(env: dict[str, str] | None = None) -> dict[str, Any]:
    source = env or os.environ
    metadata: dict[str, Any] = {}
    if value := source.get("HEADLIGHT_CANDIDATE"):
        metadata["candidate"] = value
    numeric_fields = {
        "HEADLIGHT_STEP_NUM_IN": ("step_num_in", int),
        "HEADLIGHT_NUM_BLOCKS": ("num_blocks", int),
        "HEADLIGHT_HIDDEN_WIDTH": ("hidden_width", int),
        "HEADLIGHT_LEARNING_RATE": ("learning_rate", float),
        "HEADLIGHT_RESIDUAL_BOUND": ("residual_bound", float),
    }
    for env_key, (field, caster) in numeric_fields.items():
        value = source.get(env_key)
        if value in (None, ""):
            continue
        try:
            metadata[field] = caster(value)
        except ValueError as exc:
            raise CampaignFailure(f"{env_key} must be a {caster.__name__}: {value}") from exc
    return metadata


def _completed(command: list[str], runner: Callable[..., subprocess.CompletedProcess[str]]) -> subprocess.CompletedProcess[str]:
    return runner(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _kubeconfig_context_clusters(kubeconfig: Path) -> dict[str, str]:
    data = yaml.safe_load(kubeconfig.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for item in data.get("contexts", []) or []:
        name = item.get("name")
        cluster = item.get("context", {}).get("cluster")
        if name and cluster:
            mapping[name] = cluster
    return mapping


def _kubeconfig_server_ports(kubeconfig: Path) -> dict[str, int]:
    data = yaml.safe_load(kubeconfig.read_text(encoding="utf-8")) or {}
    ports: dict[str, int] = {}
    for item in data.get("clusters", []) or []:
        name = item.get("name")
        server = item.get("cluster", {}).get("server", "")
        match = re.search(r"https?://(?:127\.0\.0\.1|localhost|\[::1\]):(\d+)", server)
        if name and match:
            ports[name] = int(match.group(1))
    return ports


def _docker_kind_ports(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, set[int]]:
    result = _completed(["docker", "ps", "--format", "{{.Names}} {{.Ports}}"], runner)
    if result.returncode != 0:
        raise PreflightFailure(f"cannot inspect Docker Kind ports; run ./experiments/rebuild_local_testbed.sh\n{result.stdout.strip()}")
    ports: dict[str, set[int]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, exposed = line.partition(" ")
        context = name.removesuffix("-control-plane")
        found = {int(port) for port in re.findall(r"(?:127\.0\.0\.1|0\.0\.0\.0|\[::\]):(\d+)->6443/tcp", exposed)}
        if found:
            ports[context] = found
    return ports


def preflight_testbed(
    kubeconfig: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    recovery = "./experiments/rebuild_local_testbed.sh"
    if not kubeconfig.exists():
        raise PreflightFailure(f"missing kubeconfig {kubeconfig}; run {recovery}")
    os.environ.setdefault("HEADLIGHT_KUBECONFIG", str(kubeconfig))
    os.environ.setdefault("KUBECONFIG", str(kubeconfig))

    contexts_result = _completed(["kubectl", "--kubeconfig", str(kubeconfig), "config", "get-contexts", "-o", "name"], runner)
    if contexts_result.returncode != 0:
        raise PreflightFailure(f"cannot read kubeconfig contexts from {kubeconfig}; run {recovery}\n{contexts_result.stdout.strip()}")
    contexts = set(contexts_result.stdout.splitlines())
    missing = [context for context in REQUIRED_CONTEXTS if context not in contexts]
    if missing:
        raise PreflightFailure(f"missing kube contexts: {', '.join(missing)}; run {recovery}")

    context_clusters = _kubeconfig_context_clusters(kubeconfig)
    server_ports = _kubeconfig_server_ports(kubeconfig)
    docker_ports = _docker_kind_ports(runner)
    stale: list[str] = []
    for context in MEMBER_CONTEXTS:
        cluster_name = context_clusters.get(context)
        configured_port = server_ports.get(cluster_name or "")
        live_ports = docker_ports.get(context, set())
        if configured_port and live_ports and configured_port not in live_ports:
            stale.append(f"{context} kubeconfig_port={configured_port} live_ports={','.join(str(port) for port in sorted(live_ports))}")
    if stale:
        raise PreflightFailure(f"stale Kind API ports detected: {'; '.join(stale)}; run {recovery}")

    for context in REQUIRED_CONTEXTS:
        result = _completed(["kubectl", "--kubeconfig", str(kubeconfig), "--context", context, "get", "--raw=/readyz"], runner)
        if result.returncode != 0:
            if context == "karmada-apiserver":
                raise PreflightFailure(
                    f"Karmada API TLS/auth appears stale for {context}; run {recovery}\n{result.stdout.strip()}"
                )
            raise PreflightFailure(f"context {context} is not reachable; run {recovery}\n{result.stdout.strip()}")

    return {"kubeconfig": str(kubeconfig), "contexts": list(REQUIRED_CONTEXTS)}


def preflight_monitoring(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    namespace = os.environ.get("HEADLIGHT_PROMETHEUS_NAMESPACE", "monitoring")
    service = os.environ.get("HEADLIGHT_PROMETHEUS_SERVICE", "prometheus-kube-prometheus-prometheus")
    cluster = os.environ.get("HEADLIGHT_PROMETHEUS_CLUSTER", "cluster2")
    result = _completed(["kubectl", "--context", cluster, "-n", namespace, "get", "service", service], runner)
    if result.returncode != 0:
        raise PreflightFailure(
            f"missing Prometheus service {service}.{namespace} in {cluster}; "
            "run HEADLIGHT_KUBECONFIG=/root/headlight/.kube/headlight-testbed.config "
            "./experiments/install_prometheus_monitoring.sh\n"
            f"{result.stdout.strip()}"
        )
    return {"cluster": cluster, "namespace": namespace, "service": service}


def pilot_manifest(
    users: int,
    duration: int,
    output_dir: Path,
    trainticket_path: str = "advanced-ticket",
    resource_profile: str = "capacity-vm",
    seed: int = SEED,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locust_mode = LOCUST_MODE_BY_PATH[trainticket_path]
    manifest = copy.deepcopy(resolve_run(Path(PILOT_CONFIG)))
    manifest["metadata"]["name"] = f"capacity-pilot-{users}-users"
    manifest["spec"]["workload"].update(
        {
            "pattern": "gradual-ramp",
            "duration_seconds": duration,
            "warmup_seconds": 0,
            "locust": {
                "spawn_rate": 2,
                "start_users": users,
                "end_users": users,
                "ramp_seconds": 1,
                "hold_seconds": duration - 1,
            },
        }
    )
    manifest["spec"]["run"].update(
        {
            "duration_seconds": duration,
            "warmup_seconds": 0,
            "seeds": [seed],
            "output_dir": str(output_dir),
            "qualification_overrides": {
                "kind": "capacity-pilot",
                "users": users,
                "seed": seed,
                "trainticket_path": trainticket_path,
                "transaction_mix": locust_mode,
                "resource_profile": resource_profile,
                "headlight_candidate": candidate_metadata or {},
            },
        }
    )
    return manifest


def qualification_manifest(
    controller: str,
    user_cap: int,
    duration: int,
    output_dir: Path,
    trainticket_path: str = "advanced-ticket",
    resource_profile: str = "capacity-vm",
    seed: int = SEED,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locust_mode = LOCUST_MODE_BY_PATH[trainticket_path]
    manifest = copy.deepcopy(resolve_run(Path(RUN_CONFIGS[controller])))
    if "service_profile" in manifest["spec"]["controller"]:
        manifest["spec"]["controller"]["service_profile"] = (
            f"ai-module/configs/service_profiles/{trainticket_path}.json"
        )
    common_topology = resolve_run(Path(RUN_CONFIGS["headlight"]))["spec"]["topology"]
    manifest["spec"]["topology"] = copy.deepcopy(common_topology)
    ramp_seconds = min(900, max(1, duration * 3 // 4))
    start_users = max(1, int(math.ceil(user_cap * 0.60)))
    end_users = max(start_users + 1, int(math.ceil(user_cap * 1.60)))
    manifest["metadata"]["name"] = f"qualification-{controller}-seed-{seed}"
    manifest["spec"]["workload"].update(
        {
            "pattern": "gradual-ramp",
            "duration_seconds": duration,
            "warmup_seconds": 0,
            "locust": {
                "spawn_rate": 2,
                "start_users": start_users,
                "end_users": end_users,
                "ramp_seconds": ramp_seconds,
                "hold_seconds": duration - ramp_seconds,
            },
        }
    )
    manifest["spec"]["run"].update(
        {
            "duration_seconds": duration,
            "warmup_seconds": 0,
            "seeds": [seed],
            "output_dir": str(output_dir),
            "qualification_overrides": {
                "kind": "capacity-qualification",
                "publishable_as_softwarex_five_seed_result": False,
                "seed": seed,
                "selected_user_cap": user_cap,
                "qualification_start_users": start_users,
                "qualification_end_users": end_users,
                "trainticket_path": trainticket_path,
                "transaction_mix": locust_mode,
                "resource_profile": resource_profile,
                "headlight_candidate": candidate_metadata or {},
                "topology": copy.deepcopy(common_topology),
                "resource_bounds": {
                    "min_replicas_per_cluster": 1,
                    "cluster_max_replicas": {"cluster2": 3, "cluster3": 5},
                    "max_total_replicas_per_service": 8,
                },
            },
        }
    )
    return manifest


class CapacityCampaign:
    def __init__(
        self,
        options: CampaignOptions,
        execute: Callable[[dict[str, Any], Path, int, bool], None] = run_one_seed,
        sleeper: Callable[[float], None] = time.sleep,
        summarize_pilot: Callable[[Path], dict[str, Any]] | None = None,
    ) -> None:
        self.options = options
        self.execute = execute
        self.sleeper = sleeper
        self.summarize_pilot = summarize_pilot
        self.completed_stage = "initializing"
        self.selected_capacity: int | None = None
        self.controller_order: list[str] = []
        self.started_at = utc_now()
        self.log_path = options.output_dir / "campaign.log"

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _execute_manifest(self, manifest: dict[str, Any], output: Path) -> None:
        if self.options.smoke:
            manifest["spec"]["run"]["smoke_validation"] = True
            manifest["spec"]["run"].setdefault("reset_policy", {})["delete_pods"] = False
        output.mkdir(parents=True, exist_ok=True)
        write_yaml(output / "manifest.yaml", manifest)
        (output / "environment.json").write_text(
            json.dumps(environment(self.options.dry_run), indent=2) + "\n", encoding="utf-8"
        )
        self.execute(manifest, output, self.options.seed, self.options.dry_run)

    def _cooldown(self) -> None:
        if not self.options.dry_run and self.options.cooldown_seconds:
            self.log(f"cooldown seconds={self.options.cooldown_seconds}")
            self.sleeper(self.options.cooldown_seconds)

    def _pilot_users(self) -> tuple[int, ...]:
        if self.options.skip_pilots:
            if self.options.selected_capacity_override is None:
                raise CampaignFailure("--skip-pilots requires --selected-capacity")
            self.selected_capacity = self.options.selected_capacity_override
            self.completed_stage = "pilots-skipped"
            self.log(f"skipping capacity pilots selected_capacity={self.selected_capacity}")
            return ()
        return PILOT_LEVELS

    def _run_pilot(self, users: int, baseline_p95: float | None) -> dict[str, Any]:
        self.completed_stage = f"pilot-{users}-starting"
        output = self.options.output_dir / "pilots" / f"{users}-users"
        manifest = pilot_manifest(
            users,
            self.options.pilot_duration,
            output,
            self.options.trainticket_path,
            self.options.resource_profile,
            self.options.seed,
            self.options.candidate_metadata,
        )
        self.log(
            f"running capacity pilot users={users} seed={self.options.seed} "
            f"trainticket_path={self.options.trainticket_path} locust_mode={self.options.locust_mode}"
        )
        knee_from_requests = False
        knee_reason: str | None = None
        try:
            self._execute_manifest(manifest, output)
        except BundleQualityError as exc:
            if not exc.issues or not all(issue.capacity_knee for issue in exc.issues):
                raise
            knee_from_requests = True
            knee_reason = "; ".join(issue.message for issue in exc.issues)
            self.log(f"capacity knee at {users} users: {exc}")

        seed_dir = output / f"seed-{self.options.seed}"
        if self.summarize_pilot is not None:
            summary = self.summarize_pilot(seed_dir)
        elif self.options.dry_run:
            summary = {"requests": 0, "successful": 0, "success_rate": 1.0, "exceptions": 0, "p95_latency_ms": 100.0}
        else:
            summary = request_summary(seed_dir / "raw" / "request_events.csv")
        healthy, reason = capacity_is_healthy(summary, baseline_p95)
        healthy = healthy and not knee_from_requests
        if knee_reason:
            reason = knee_reason
        record = {"users": users, "healthy": healthy, "reason": reason, **summary}
        (output / "pilot_summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self.completed_stage = f"pilot-{users}-complete"
        return record

    def _run_pilots(self) -> list[dict[str, Any]]:
        baseline_p95: float | None = None
        pilot_results: list[dict[str, Any]] = []
        for users in self._pilot_users():
            record = self._run_pilot(users, baseline_p95)
            pilot_results.append(record)
            if users == PILOT_LEVELS[0] and not record["healthy"]:
                raise CampaignFailure(f"5-user capacity baseline failed: {record['reason']}")
            if not record["healthy"]:
                self.log(f"expected relative capacity knee reached: {record['reason']}")
                break
            self.selected_capacity = users
            if users == PILOT_LEVELS[0]:
                baseline_p95 = float(record["p95_latency_ms"])
            self._cooldown()

        if self.selected_capacity is None:
            raise CampaignFailure("no safe pilot capacity was selected")
        (self.options.output_dir / "capacity_summary.json").write_text(
            json.dumps(
                {
                    "baseline_users": 5,
                    "selected_capacity": self.selected_capacity,
                    "pilots": pilot_results,
                    "pilots_skipped": self.options.skip_pilots,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return pilot_results

    def _run_qualifications(self) -> None:
        self.controller_order = sorted(
            self.options.controllers,
            key=lambda controller: _stable_order_key(self.options.seed, controller),
        )
        (self.options.output_dir / "controller_order.json").write_text(
            json.dumps({"seed": self.options.seed, "order": self.controller_order}, indent=2) + "\n", encoding="utf-8"
        )
        for controller in self.controller_order:
            self.completed_stage = f"qualification-{controller}-starting"
            output = self.options.output_dir / "qualification" / controller
            manifest = qualification_manifest(
                controller,
                self.selected_capacity,
                self.options.qualification_duration,
                output,
                self.options.trainticket_path,
                self.options.resource_profile,
                self.options.seed,
                self.options.candidate_metadata if controller == "headlight" else None,
            )
            self.log(f"running qualification controller={controller} users={self.selected_capacity} seed={self.options.seed}")
            self._execute_manifest(manifest, output)
            self.completed_stage = f"qualification-{controller}-complete"
            self._cooldown()

    def run(self) -> None:
        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        self.log("campaign started; no pre-existing experiment process is managed by this campaign")
        self._run_pilots()
        self._run_qualifications()
        self.completed_stage = "complete"

    def status(self, success: bool, exit_code: int, error: str | None = None) -> dict[str, Any]:
        return {
            "status": "success" if success else "failure",
            "completed_stage": self.completed_stage,
            "selected_capacity": self.selected_capacity,
            "controller_order": self.controller_order,
            "seed": self.options.seed,
            "dry_run": self.options.dry_run,
            "smoke": self.options.smoke,
            "trainticket_path": self.options.trainticket_path,
            "locust_mode": self.options.locust_mode,
            "resource_profile": self.options.resource_profile,
            "headlight_candidate": self.options.candidate_metadata or {},
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "exit_code": exit_code,
            "error": error,
        }


def capture_failure_diagnostics(output_dir: Path, log: Callable[[str], None]) -> None:
    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    commands = {
        "host-state.txt": ["sh", "-c", "uptime; free -h; df -h; ps -eo pid,ppid,stat,%cpu,%mem,cmd --sort=-%cpu | head -n 80"],
    }
    for context in ["karmada-apiserver", "karmada-host", "cluster1", "cluster2", "cluster3"]:
        commands[f"{context}-nodes.txt"] = ["kubectl", "--context", context, "get", "nodes", "-o", "wide"]
        commands[f"{context}-pods.txt"] = ["kubectl", "--context", context, "get", "pods", "-A", "-o", "wide"]
        commands[f"{context}-events.txt"] = ["kubectl", "--context", context, "get", "events", "-A", "--sort-by=.lastTimestamp"]
        commands[f"{context}-deployments.txt"] = ["kubectl", "--context", context, "get", "deployments", "-A", "-o", "wide"]
    for selector in ["app=ai-module", "app=lstm-autoscaler"]:
        safe_name = selector.split("=", 1)[1]
        commands[f"controller-{safe_name}.txt"] = [
            "kubectl", "--context", "cluster1", "-n", "trainticket", "logs", "-l", selector, "--tail=-1"
        ]
    for filename, command in commands.items():
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=4, check=False)
            (diagnostics / filename).write_text(f"$ {shlex.join(command)}\n{result.stdout}\n[exit {result.returncode}]\n", encoding="utf-8")
        except Exception as exc:
            (diagnostics / filename).write_text(f"diagnostic capture failed: {exc}\n", encoding="utf-8")
            log(f"diagnostic command failed command={shlex.join(command)} error={exc}")


class ShutdownOnce:
    def __init__(self, enabled: bool, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
        self.enabled = enabled
        self.runner = runner
        self.requested = False

    def request(self, log: Callable[[str], None]) -> None:
        if self.requested or not self.enabled:
            return
        self.requested = True
        override = os.environ.get("HEADLIGHT_SHUTDOWN_COMMAND")
        primary = shlex.split(override) if override else ["systemctl", "poweroff"]
        try:
            result = self.runner(primary, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        except Exception as exc:
            result = subprocess.CompletedProcess(primary, 127, stdout=str(exc))
        if result.returncode == 0:
            return
        log(f"poweroff command failed exit={result.returncode} output={result.stdout.strip()}")
        fallback_override = os.environ.get("HEADLIGHT_SHUTDOWN_FALLBACK_COMMAND")
        fallback = shlex.split(fallback_override) if fallback_override else ["/usr/sbin/shutdown", "-h", "now"]
        try:
            fallback_result = self.runner(fallback, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        except Exception as exc:
            fallback_result = subprocess.CompletedProcess(fallback, 127, stdout=str(exc))
        if fallback_result.returncode != 0:
            log(f"fallback shutdown failed exit={fallback_result.returncode} output={fallback_result.stdout.strip()}")


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "results" / "capacity-qualification" / stamp


def _record_campaign_failure(
    exc: BaseException,
    campaign: CapacityCampaign,
    output: Path,
    options: CampaignOptions,
) -> tuple[str, int]:
    error = f"{type(exc).__name__}: {exc}"
    exit_code = 130 if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 1
    campaign.log(f"campaign failed stage={campaign.completed_stage} exit_code={exit_code} error={error}")
    with (output / "campaign.log").open("a", encoding="utf-8") as handle:
        traceback.print_exc(file=handle)
    if not options.dry_run:
        capture_failure_diagnostics(output, campaign.log)
    return error, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Capacity qualification and seed-101 controller campaign")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="write all contracts without touching Kubernetes or powering off")
    parser.add_argument("--smoke", action="store_true", help="run live 30-second stages with shutdown forcibly disabled")
    parser.add_argument("--pilot-duration", type=int, default=360)
    parser.add_argument("--qualification-duration", type=int, default=1200)
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trainticket-path", choices=TRAINTICKET_PATHS, default="advanced-ticket")
    parser.add_argument("--resource-profile", choices=RESOURCE_PROFILES, default="capacity-vm")
    parser.add_argument("--skip-pilots", action="store_true", help="skip capacity search and use --selected-capacity")
    parser.add_argument("--selected-capacity", type=int, default=None, help="user cap to use with --skip-pilots")
    parser.add_argument(
        "--controllers",
        default=",".join(RUN_CONFIGS),
        help="comma-separated subset/order of controllers to qualify",
    )
    args = parser.parse_args()
    if args.skip_pilots and args.selected_capacity is None:
        parser.error("--skip-pilots requires --selected-capacity")
    controllers = tuple(controller.strip() for controller in args.controllers.split(",") if controller.strip())
    unknown_controllers = [controller for controller in controllers if controller not in RUN_CONFIGS]
    if not controllers:
        parser.error("--controllers must include at least one controller")
    if unknown_controllers:
        parser.error(f"unknown controllers: {', '.join(unknown_controllers)}")
    output = (args.output_dir or default_output_dir()).resolve()
    options = CampaignOptions(
        output_dir=output,
        dry_run=args.dry_run,
        smoke=args.smoke,
        trainticket_path=args.trainticket_path,
        resource_profile=args.resource_profile,
        pilot_duration=30 if args.smoke else args.pilot_duration,
        qualification_duration=30 if args.smoke else args.qualification_duration,
        cooldown_seconds=0 if args.smoke or args.dry_run else args.cooldown_seconds,
        seed=args.seed,
        candidate_metadata=headlight_candidate_metadata(),
        skip_pilots=args.skip_pilots,
        selected_capacity_override=args.selected_capacity,
        controllers=controllers,
    )
    kubeconfig = Path(os.environ.get("HEADLIGHT_KUBECONFIG", str(default_kubeconfig()))).resolve()
    os.environ.update(
        {
            "HEADLIGHT_KUBECONFIG": str(kubeconfig),
            "KUBECONFIG": str(kubeconfig),
            "HEADLIGHT_TRAINTICKET_PATH": options.trainticket_path,
            "HEADLIGHT_TRAINTICKET_LOCUST_MODE": options.locust_mode,
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": options.resource_profile,
            "HEADLIGHT_STRICT_FINAL_READINESS": "1",
        }
    )
    os.environ.setdefault("HEADLIGHT_AI_FAST_TIMING", "1")
    os.environ.setdefault("HEADLIGHT_AI_FAST_SLEEP_INTERVAL", "15")
    campaign = CapacityCampaign(options)
    shutdown = ShutdownOnce(
        enabled=os.environ.get("HEADLIGHT_SHUTDOWN_ON_EXIT") == "1" and not options.dry_run and not options.smoke
    )
    interrupted: dict[str, str | None] = {"signal": None}

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted["signal"] = signal.Signals(signum).name
        raise InterruptedError(f"received {interrupted['signal']}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)

    success = False
    exit_code = 1
    error: str | None = None
    try:
        if not options.dry_run:
            campaign.completed_stage = "preflight"
            campaign.log(f"running testbed preflight kubeconfig={kubeconfig}")
            preflight = preflight_testbed(kubeconfig)
            preflight["monitoring"] = preflight_monitoring()
            (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
            campaign.log("testbed preflight passed")
        campaign.run()
        success = True
        exit_code = 0
    except (KeyboardInterrupt, InterruptedError) as exc:
        error, exit_code = _record_campaign_failure(exc, campaign, output, options)
    except Exception as exc:
        error, exit_code = _record_campaign_failure(exc, campaign, output, options)
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "campaign_status.json").write_text(
            json.dumps(campaign.status(success, exit_code, error), indent=2) + "\n", encoding="utf-8"
        )
        campaign.log(f"campaign status persisted success={success} exit_code={exit_code}")
        subprocess.run(["sync"], check=False)
        shutdown.request(campaign.log)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
