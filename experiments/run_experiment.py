#!/usr/bin/env python3
"""Run or dry-run a resolved Headlight experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.capacity_contract import CapacityContractError, contract_path_from_env, validate_capacity_contract
from experiments.lib.baseline_extension import TRACE_HASHES as EVALUATION_TRACE_HASHES
from experiments.lib.config import REPO_ROOT, resolve_run, write_yaml
from experiments.lib.freeze import validate_freeze_manifest, validate_threshold_evaluation_inputs
from experiments.lib.gateway_protocol import gateway_protocol_from_env
from experiments.lib.matched_comparison import validate_matched_baseline_pair
from experiments.lib.metrics import compute_summary_metrics
from experiments.lib.prometheus import export_queries
from experiments.lib.protocol import (
    capacity_seeds,
    calibration_seeds,
    common_topology,
    evaluation_seeds,
    load_protocol,
    primary_controllers,
    primary_workloads,
    protocol_reference,
    protocol_timing,
    replica_limit_for_cluster,
    replica_limits,
    tuning_seeds,
)
from experiments.lib.quality import performance_outcome, validate_bundle
from experiments.lib.run_evidence import (
    ControllerActionStream, GatewayWatch, RunEvidence, atomic_json,
    collect_failed_attempt, merge_controller_actions, run_logged_process, stop_process,
)
from experiments.lib.static_plan import static_plan_path, validate_static_plan
from experiments.lib.workloads import generate_shape, workload_phase_windows, write_shape_csv
from experiments.scripts.render_minimal_trainticket import render as render_trainticket


REQUIRED_CONTEXTS = ["karmada-apiserver", "karmada-host", "cluster1", "cluster2", "cluster3"]
DEFAULT_TRAINTICKET_PATH = "advanced-ticket"
PROTOCOL = load_protocol()
GATEWAY_PROTOCOL = gateway_protocol_from_env()
PRIMARY_CONTROLLERS = primary_controllers(PROTOCOL)
PRIMARY_WORKLOADS = primary_workloads(PROTOCOL)
EVALUATION_SEEDS = evaluation_seeds(PROTOCOL)
CAPACITY_SEEDS = capacity_seeds(PROTOCOL)
CALIBRATION_SEEDS = calibration_seeds(PROTOCOL)
TUNING_SEEDS = tuning_seeds(PROTOCOL)
REPLICA_LIMITS = replica_limits(PROTOCOL)
REPLICA_CEILING = REPLICA_LIMITS["aggregate_max_replicas_per_service"]
DEFAULT_CLUSTER_REPLICA_SPLIT = REPLICA_LIMITS["per_cluster_max_replicas"]
RESOURCE_PROFILE_BY_TRAINTICKET_PATH = {
    "login-route": "local-tiny",
    "advanced-ticket": "capacity-vm",
    "full": "source",
}
REQUEST_EVENTS_FILE = "request_events.csv"
PROMETHEUS_RANGE_DATA_FILE = "prometheus_range_data.json"
CONTROLLER_ACTIONS_FILE = "controller_actions.jsonl"
DEFAULT_POLICY_CONFIG = REPO_ROOT / "ai-module" / "configs" / "policies" / "research-v3-forecast-policy.json"


configured_kubeconfig = os.environ.get(
    "KUBECONFIG",
    os.environ.get("HEADLIGHT_KUBECONFIG", str(REPO_ROOT / ".kube" / "headlight-testbed.config")),
)
# Controller setup scripts use their own working directories. Resolve every
# kubeconfig component before spawning them so relative VM paths stay valid.
os.environ["KUBECONFIG"] = os.pathsep.join(
    str(Path(component).expanduser().resolve())
    for component in configured_kubeconfig.split(os.pathsep)
    if component
)


def parse_seeds(value: str, configured: list[int]) -> list[int]:
    if value == "all":
        return configured
    return [int(part) for part in value.split(",") if part]


def protocol_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    workload = manifest["spec"]["workload"]
    run = manifest["spec"]["run"]
    timing = protocol_timing(PROTOCOL)
    warmup = int(run.get("warmup_seconds", workload.get("warmup_seconds", 0)))
    measurement = int(run["duration_seconds"])
    observation_tail = int(run.get("diagnostic_observation_tail_seconds", 0))
    metadata = {
        "reference": protocol_reference(PROTOCOL),
        "primary_controllers": list(PRIMARY_CONTROLLERS),
        "primary_workloads": list(PRIMARY_WORKLOADS),
        "capacity_seeds": list(CAPACITY_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "headlight_tuning_seeds": list(TUNING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "replica_ceiling_per_service": REPLICA_CEILING,
        "cluster_replica_split": dict(DEFAULT_CLUSTER_REPLICA_SPLIT),
        "warmup_seconds": warmup,
        "measurement_seconds": measurement,
        "cooldown_seconds_between_attempts": int(timing["cooldown_seconds"]),
        "diagnostic_observation_tail_seconds": observation_tail,
        "observation_tail_mode": _observation_tail_mode(workload, observation_tail),
        "locust_run_seconds": warmup + measurement + observation_tail,
        "capacity_normalization": {
            "base": "0.2C",
            "peak": "C",
            "capacity_env": "HEADLIGHT_EVALUATION_CAPACITY",
            "fallback_capacity_users": int(
                workload.get("capacity_users", workload.get("locust", {}).get("capacity_users", 10))
            ),
        },
        "technical_invalidity_only": True,
        "statistical_unit": "seed-run",
    }
    if GATEWAY_PROTOCOL is not None:
        metadata["effective_version"] = GATEWAY_PROTOCOL["spec"]["version"]
        metadata["gateway_protocol"] = GATEWAY_PROTOCOL["reference"]
        metadata["prior_research_v3_cells_directly_comparable"] = False
    if run.get("evaluation_trial"):
        metadata["experiment_scope"] = "evaluation_trial"
        metadata["matched_comparison_eligible"] = False
        metadata["capacity_qualified"] = False
    return metadata


def apply_protocol_metadata(manifest: dict[str, Any]) -> None:
    manifest["spec"]["run"]["protocol"] = protocol_metadata(manifest)


def _rendered_workload_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    topology = manifest["spec"]["topology"]
    path = trainticket_path()
    resource_profile = trainticket_resource_profile(path)
    namespace = topology["namespace"]
    mongo_storage = trainticket_mongo_storage()
    data_plane_docs = render_trainticket(path, namespace, resource_profile, include_policies=False, mongo_storage=mongo_storage)
    karmada_docs = render_trainticket(path, namespace, resource_profile, include_policies=True, mongo_storage=mongo_storage)
    return {
        "trainticket_path": path,
        "resource_profile": resource_profile,
        "mongo_storage": mongo_storage,
        "namespace": namespace,
        "data_plane_no_policies_sha256": hashlib.sha256(
            yaml.safe_dump_all(data_plane_docs, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "karmada_with_policies_sha256": hashlib.sha256(
            yaml.safe_dump_all(karmada_docs, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _resolved_topology(manifest: dict[str, Any], workload_hashes: dict[str, str]) -> dict[str, Any]:
    controller = manifest["spec"]["controller"]["controller"]
    protocol_topology = common_topology(PROTOCOL)
    native_controller = controller in {"hpa", "keda"}
    gateway_spec = GATEWAY_PROTOCOL["spec"] if GATEWAY_PROTOCOL is not None else None
    topology = {
        "protocol": protocol_reference(PROTOCOL),
        "manifest_topology": manifest["spec"]["topology"],
        "common_topology": protocol_topology,
        "controller": controller,
        "controller_context": os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", protocol_topology.get("controller_cluster", "cluster3")),
        "load_generator_cluster": gateway_spec["load_generator_cluster"] if gateway_spec else os.environ.get(
            "HEADLIGHT_LOCUST_CLUSTER", str(protocol_topology["load_generator_cluster"])
        ),
        "ingress_cluster": gateway_spec["gateway_cluster"] if gateway_spec else protocol_topology["ingress_cluster"],
        "deploy_mode": "single-cluster" if native_controller else "karmada",
        "hpa_controller_path_note": (
            "HPA uses native member-cluster kube-controller-manager behavior, not Actions Handler or CRP patches."
            if controller == "hpa"
            else ""
        ),
        "workload_manifest_hashes": workload_hashes,
    }
    if GATEWAY_PROTOCOL is not None:
        topology["gateway_protocol"] = GATEWAY_PROTOCOL["reference"]
        topology["gateway_url"] = gateway_spec["gateway_url"]
    return topology


def configure_evaluation_trial(
    manifest: dict[str, Any], seeds: list[int], output_dir: Path,
    matched_baseline_manifest: Path | None,
    *, campaign_cell: bool = False,
) -> None:
    """Configure one configured Threshold observation, separate from comparison."""
    if matched_baseline_manifest is not None:
        raise ValueError("evaluation trials cannot claim matched baseline evidence")
    if manifest["spec"]["controller"]["controller"] != "threshold" or GATEWAY_PROTOCOL is None:
        raise ValueError("evaluation trials require Threshold and the shared gateway protocol")
    workload = manifest["spec"]["workload"]["pattern"]
    allowed_cells = (
        {(seed, pattern) for seed in (101, 102, 103, 104, 105)
         for pattern in EVALUATION_TRACE_HASHES}
        if campaign_cell else {(104, "gradual-ramp")}
    )
    if len(seeds) != 1 or workload not in EVALUATION_TRACE_HASHES or (seeds[0], workload) not in allowed_cells:
        raise ValueError("evaluation Threshold cell is not in the authorized seed/workload set")
    expected_duration = {"gradual-ramp": 1200, "diurnal": 2400, "flash-crowd": 1200}[workload]
    if manifest["spec"]["run"]["duration_seconds"] != expected_duration:
        raise ValueError(f"evaluation {workload} cells require the registered {expected_duration}s measurement")
    if os.environ.get("HEADLIGHT_EVALUATION_CAPACITY") != "4":
        raise ValueError("evaluation trials require explicit, configured C=4")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("evaluation trial output directory must be new and empty")
    manifest["spec"]["run"]["seeds"] = seeds
    manifest["spec"]["run"]["evaluation_trial"] = {
        "schema_version": "headlight.evaluation-trial/v1",
        "seed": seeds[0], "workload": workload, "capacity_users": 4,
        "capacity_qualified": False,
        "matched_comparison_eligible": False,
        "historical_gateway_image_verified": False,
        "comparison_status": "evaluation_not_matched",
        "load_trace_sha256": EVALUATION_TRACE_HASHES[workload],
        "controller_image": os.environ.get("HEADLIGHT_THRESHOLD_IMAGE"),
        "controller_image_digest": os.environ.get("HEADLIGHT_THRESHOLD_IMAGE_ID"),
        "open_gates": ["gateway_capacity_qualification", "historical_gateway_image_provenance"],
    }
    evaluation_trial_metadata(manifest)


def evaluation_trial_metadata(manifest: dict[str, Any]) -> dict[str, Any] | None:
    trial = manifest["spec"]["run"].get("evaluation_trial")
    if trial is None:
        return None
    if (not isinstance(trial, dict)
            or trial.get("schema_version") != "headlight.evaluation-trial/v1"
            or trial.get("comparison_status") != "evaluation_not_matched"
            or trial.get("capacity_qualified") is not False
            or trial.get("matched_comparison_eligible") is not False
            or trial.get("capacity_users") != 4
            or os.environ.get("HEADLIGHT_EVALUATION_CAPACITY") != "4"
            or manifest["spec"]["run"]["seeds"] != [trial.get("seed")]
            or (trial.get("seed"), trial.get("workload", manifest["spec"]["workload"]["pattern"])) not in {
                (seed, pattern)
                for seed in (101, 102, 103, 104, 105)
                for pattern in EVALUATION_TRACE_HASHES
            }
            or manifest["spec"]["run"].get("matched_baseline_manifest") is not None
            or manifest["spec"]["controller"]["controller"] != "threshold"
            or manifest["spec"]["workload"]["pattern"] != trial.get("workload", manifest["spec"]["workload"]["pattern"])
            or manifest["spec"]["run"]["warmup_seconds"] != 480
            or manifest["spec"]["run"]["duration_seconds"] != {"gradual-ramp": 1200, "diurnal": 2400, "flash-crowd": 1200}.get(
                trial.get("workload", manifest["spec"]["workload"]["pattern"])
            )
            or manifest["spec"]["run"]["sampling_interval_seconds"] != 15
            or manifest["spec"]["run"].get("diagnostic_observation_tail_seconds", 0) != 0
            or trainticket_path() != "advanced-ticket"
            or trainticket_resource_profile("advanced-ticket") != "capacity-vm"
            or trainticket_mongo_storage() != "image-defined"
            or trial.get("load_trace_sha256") != EVALUATION_TRACE_HASHES.get(
                trial.get("workload", manifest["spec"]["workload"]["pattern"])
            )
            or GATEWAY_PROTOCOL is None):
        raise ValueError("invalid evaluation trial metadata or runtime capacity")
    return trial


def _is_primary_evaluation(manifest: dict[str, Any]) -> bool:
    controller = manifest["spec"]["controller"]["controller"]
    workload = manifest["spec"]["workload"]["pattern"]
    overrides = manifest["spec"]["run"].get("qualification_overrides", {})
    return (
        controller in PRIMARY_CONTROLLERS
        and workload in PRIMARY_WORKLOADS
        and evaluation_trial_metadata(manifest) is None
        and overrides.get("kind") not in {"capacity-pilot", "capacity-qualification", "forecast-qualification"}
    )


def _observation_tail_mode(workload: dict[str, Any], tail_seconds: int) -> str:
    if tail_seconds <= 0:
        return "disabled"
    if workload.get("pattern") == "multi-phase" and workload.get("continuation_mode") == "hold_endpoint":
        return "hold_final_load"
    return "continue_workload"


def _measurement_load_phases(
    workload: dict[str, Any], measurement_started_at: int | float | None = None,
) -> list[dict[str, Any]]:
    if workload.get("pattern") != "multi-phase":
        return []
    windows = workload_phase_windows(workload.get("phases", []))
    if measurement_started_at is not None:
        for window in windows:
            window["started_at_unix"] = measurement_started_at + window["start_offset_seconds"]
            window["finished_at_unix"] = measurement_started_at + window["end_offset_seconds"]
    return windows


def prepare_capacity_contract(
    manifest: dict[str, Any],
    dry_run: bool,
    seed_dir: Path,
    *,
    matched_baselines: bool = False,
) -> dict[str, Any] | None:
    if not _is_primary_evaluation(manifest):
        return None
    if dry_run:
        return None
    path = trainticket_path()
    resource_profile = trainticket_resource_profile(path)
    contract_path = contract_path_from_env()
    expected_gateway = GATEWAY_PROTOCOL["reference"] if GATEWAY_PROTOCOL is not None else None
    try:
        contract, warnings = validate_capacity_contract(
            contract_path,
            expected_trainticket_path=path,
            expected_resource_profile=resource_profile,
            expected_gateway_protocol=expected_gateway,
        )
    except CapacityContractError:
        if (not matched_baselines or os.environ.get("HEADLIGHT_CAPACITY_CONTRACT")
                or manifest["spec"]["controller"]["controller"] == "threshold"):
            raise
        # Keep the old frozen resource-bound check when paired, measured
        # HPA/KEDA traces establish the same C=4 load through this gateway.
        contract, warnings = validate_capacity_contract(
            contract_path,
            expected_trainticket_path=path,
            expected_resource_profile=resource_profile,
            expected_gateway_protocol=None,
        )
        if int(contract.get("selected_capacity", 0)) != 4:
            raise CapacityContractError(
                "matched HPA/KEDA baselines use C=4, but the retained capacity contract does not"
            )
    configured = os.environ.get("HEADLIGHT_EVALUATION_CAPACITY")
    selected = str(int(contract["selected_capacity"]))
    if configured and configured != selected:
        raise CapacityContractError(
            f"HEADLIGHT_EVALUATION_CAPACITY={configured} conflicts with capacity contract C={selected}"
        )
    os.environ["HEADLIGHT_EVALUATION_CAPACITY"] = selected
    if warnings:
        warning_text = "\n".join(warnings) + "\n"
        (seed_dir / "logs" / "capacity-contract-warning.log").write_text(warning_text, encoding="utf-8")
    return contract


def trainticket_path() -> str:
    return os.environ.get("HEADLIGHT_TRAINTICKET_PATH", DEFAULT_TRAINTICKET_PATH)


def trainticket_resource_profile(path: str) -> str:
    return os.environ.get(
        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE",
        RESOURCE_PROFILE_BY_TRAINTICKET_PATH.get(path, "capacity-vm"),
    )


def trainticket_mongo_storage() -> str:
    mode = os.environ.get("HEADLIGHT_TRAINTICKET_MONGO_STORAGE", "pod-emptydir")
    if mode not in {"pod-emptydir", "image-defined"}:
        raise ValueError(f"unsupported Mongo storage mode: {mode}")
    return mode


def tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def policy_config_path() -> Path:
    configured = os.environ.get("HEADLIGHT_POLICY_CONFIG")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / path
    return DEFAULT_POLICY_CONFIG


def environment(dry_run: bool) -> dict[str, Any]:
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).strip()
    except Exception:
        commit = None
    contexts: list[str] = []
    try:
        contexts = subprocess.check_output(
            ["kubectl", "config", "get-contexts", "-o", "name"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).splitlines()
    except Exception:
        contexts = []
    return {
        "captured_at_unix": int(time.time()),
        "repo_root": str(REPO_ROOT),
        "git_commit": commit,
        "source_tree_sha256": None if commit else tree_sha256(list((REPO_ROOT / "experiments").rglob("*"))),
        "harness_sources_sha256": {
            name: file_sha256(REPO_ROOT / name)
            for name in ["experiments/run_experiment.py", "experiments/lib/run_evidence.py",
                         "experiments/scripts/gateway_weights.py", "ai-module/ai_module_setup.sh",
                         "ai-module/ai-module/headlight/runtime.py",
                         "ai-module/ai-module/headlight/features.py",
                         "ai-module/ai-module/headlight/diagnostics_v1.py"]
        },
        "dry_run": dry_run,
        "required_contexts": REQUIRED_CONTEXTS,
        "available_contexts": contexts,
        "image_tags": {
            "ai_module": os.environ.get("HEADLIGHT_AI_IMAGE", "local/headlight-ai-module:dev"),
            "threshold": os.environ.get("HEADLIGHT_THRESHOLD_IMAGE", "local/headlight-ai-module:dev"),
            "actions_handler": os.environ.get(
                "HEADLIGHT_ACTIONS_IMAGE",
                "ghcr.io/vidbregar/actions-handler:0.2.0",
            ),
            "locust": os.environ.get("HEADLIGHT_LOCUST_IMAGE", "dstefanov46/locust-test:0.66.0"),
        },
    }


def ensure_contexts() -> None:
    contexts = run_cmd(["kubectl", "config", "get-contexts", "-o", "name"]).stdout.splitlines()
    missing = [context for context in REQUIRED_CONTEXTS if context not in contexts]
    if missing:
        raise RuntimeError(f"missing kube contexts: {', '.join(missing)}")
    clusters = kubectl_json(["get", "clusters", "-o", "json"], "karmada-apiserver")
    readiness = {
        item.get("metadata", {}).get("name"): next(
            (
                condition.get("status") == "True"
                for condition in item.get("status", {}).get("conditions", [])
                if condition.get("type") == "Ready"
            ),
            False,
        )
        for item in clusters.get("items", [])
    }
    unhealthy = [cluster for cluster in ["cluster1", "cluster2", "cluster3"] if not readiness.get(cluster, False)]
    if unhealthy:
        raise RuntimeError(f"Karmada member clusters are not Ready: {', '.join(unhealthy)}")


def run_cmd(
    args: list[str],
    log_path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    check: bool = True,
    timeout: float | None = None,
    health_check: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if timeout is None:
        timeout = float(os.environ.get("HEADLIGHT_API_COMMAND_TIMEOUT_SECONDS", "60")) if Path(args[0]).name == "kubectl" else 3600
        # Long-lived Kubernetes waits retain their declared workload/readiness deadline.
        for argument in args:
            if argument.startswith("--timeout="):
                value = argument.split("=", 1)[1]
                multiplier = 60 if value.endswith("m") else 1
                timeout = float(value.rstrip("sm")) * multiplier + 30
    try:
        completed = run_logged_process(args, log_path, env=merged_env, cwd=cwd,
                                       timeout=timeout, health_check=health_check)
    except subprocess.TimeoutExpired as exc:
        if check:
            raise RuntimeError(f"command timed out after {timeout:g}s: {' '.join(args)}") from exc
        completed = subprocess.CompletedProcess(args, 124, stdout=exc.output or "")
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(args)}")
    return completed


def kubectl_json(args: list[str], context: str, log_path: Path | None = None, check: bool = True) -> dict[str, Any]:
    completed = run_cmd(["kubectl", "--context", context, *args], log_path, check=check)
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    return json.loads(completed.stdout)


def write_seed_summary(
    path: Path,
    seed: int,
    manifest: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sla_ms = manifest["spec"]["run"]["sla"]["p95_latency_ms"]
    candidate = manifest["spec"]["run"].get("qualification_overrides", {}).get("headlight_candidate", {}) or {}
    values: dict[str, Any] = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "success_rate": 0.0,
        "successful_throughput_rps": 0.0,
        "offered_throughput_rps": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "sla_violation_rate": 0.0,
        "timeout_count": 0,
        "exception_count": 0,
        "http_status_distribution": "{}",
        "excess_response_time_seconds": 0.0,
        "cpu_seconds": 0.0,
        "mean_pod_count": 0.0,
        "pod_seconds": 0.0,
        "action_churn": 0.0,
        "cpu_calculation_method": "",
        "cpu_calculation_window": "{}",
        "pod_seconds_calculation_method": "",
        "technical_validity": "dry_run" if metrics is None else "valid",
        "performance_outcome": "not_measured" if metrics is None else "measured",
        "invalidity_reason": "",
        "retry_allowed": False,
        "retry_rule": "",
    }
    if metrics:
        values.update(metrics)
    if manifest["spec"]["run"].get("evaluation_trial"):
        values.update({"experiment_scope": "evaluation_trial", "matched_comparison_eligible": False,
                       "capacity_qualified": False})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "candidate",
                "headlight_step_num_in",
                "headlight_num_blocks",
                "headlight_hidden_width",
                "headlight_learning_rate",
                "headlight_residual_bound",
                "controller",
                "workload",
                "sla_latency_ms",
                "total_requests",
                "successful_requests",
                "failed_requests",
                "success_rate",
                "successful_throughput_rps",
                "offered_throughput_rps",
                "p50_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "sla_violation_rate",
                "timeout_count",
                "exception_count",
                "http_status_distribution",
                "excess_response_time_seconds",
                "cpu_seconds",
                "mean_pod_count",
                "pod_seconds",
                "action_churn",
                "cpu_calculation_method",
                "cpu_calculation_window",
                "pod_seconds_calculation_method",
                "technical_validity",
                "performance_outcome",
                "invalidity_reason",
                "retry_allowed",
                "retry_rule",
                *(["experiment_scope", "matched_comparison_eligible", "capacity_qualified"]
                  if manifest["spec"]["run"].get("evaluation_trial") else []),
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": seed,
                "candidate": candidate.get("candidate", ""),
                "headlight_step_num_in": candidate.get("step_num_in", ""),
                "headlight_num_blocks": candidate.get("num_blocks", ""),
                "headlight_hidden_width": candidate.get("hidden_width", ""),
                "headlight_learning_rate": candidate.get("learning_rate", ""),
                "headlight_residual_bound": candidate.get("residual_bound", ""),
                "controller": manifest["spec"]["controller"]["controller"],
                "workload": manifest["spec"]["workload"]["pattern"],
                "sla_latency_ms": sla_ms,
                **values,
            }
        )


def extract_marked_block(text: str, begin: str, end: str) -> str:
    if begin not in text or end not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0].strip() + "\n"


def reset_state(manifest: dict[str, Any], seed_dir: Path) -> None:
    namespace = manifest["spec"]["topology"]["namespace"]
    clusters = manifest["spec"]["topology"]["active_clusters"]
    log_path = seed_dir / "logs" / "reset.log"
    run_cmd(
        [
            "kubectl",
            "--context",
            "karmada-apiserver",
            "-n",
            namespace,
            "delete",
            "controlledreplicapropagations,propagationpolicies,deployments,services",
            "--all",
            "--ignore-not-found",
        ],
        log_path,
        check=False,
    )
    for cluster in clusters:
        run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "job", "headlight-paper-locust", "--ignore-not-found"], log_path)
        run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "configmap", "headlight-paper-locust", "--ignore-not-found"], log_path)
        run_cmd(
            [
                "kubectl",
                "--context",
                cluster,
                "-n",
                namespace,
                "delete",
                "deployments",
                "--all",
                "--ignore-not-found",
            ],
            log_path,
            check=False,
        )
        if manifest["spec"]["run"].get("reset_policy", {}).get("delete_pods"):
            run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "pod", "--all", "--ignore-not-found"], log_path, check=False)
    for cluster in clusters:
        run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "hpa", "--all", "--ignore-not-found"], log_path, check=False)
        run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "scaledobject", "--all", "--ignore-not-found"], log_path, check=False)
    # Controllers are mutually exclusive evidence sources. Remove predictive
    # deployments left by an earlier campaign stage before installing the next
    # controller (HPA and KEDA objects are removed above).
    for context in ["cluster1", *clusters]:
        run_cmd(
            [
                "kubectl", "--context", context, "-n", namespace, "delete", "deployment",
                "ai-module", "ai-module-v2", "lstm-autoscaler", "threshold-autoscaler", "kubernetes-autoscaler-eval", "--ignore-not-found",
            ],
            log_path,
            check=False,
        )


def deploy_workload(manifest: dict[str, Any], seed_dir: Path) -> None:
    path = trainticket_path()
    topology = manifest["spec"]["topology"]
    controller = manifest["spec"]["controller"]["controller"]
    native_controller = controller in {"hpa", "keda"}
    deploy_mode = (
        "single-cluster"
        if topology["mode"] == "single-cluster" or native_controller
        else "karmada"
    )
    resource_profile = trainticket_resource_profile(path)
    rendered = seed_dir / "raw" / f"rendered-trainticket-{path}.yaml"
    run_cmd(
        [str(REPO_ROOT / "experiments" / "deploy_minimal_trainticket.sh")],
        seed_dir / "logs" / "deploy-trainticket.log",
        env={
            "HEADLIGHT_TRAINTICKET_NAMESPACE": topology["namespace"],
            "HEADLIGHT_TRAINTICKET_PATH": path,
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": resource_profile,
            "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": trainticket_mongo_storage(),
            "HEADLIGHT_TRAINTICKET_DEPLOY_MODE": deploy_mode,
            "HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS": " ".join(topology["active_clusters"]),
            "HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST": str(rendered),
            "HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS": os.environ.get(
                "HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS",
                "",
            ),
        },
    )


def install_controller(manifest: dict[str, Any], seed_dir: Path, seed: int | None = None) -> None:
    setup = REPO_ROOT / manifest["spec"]["controller"]["setup_script"]
    topology = manifest["spec"]["topology"]
    env = {
        "HOME": str(REPO_ROOT),
        "HEADLIGHT_AI_IMAGE": os.environ.get("HEADLIGHT_AI_IMAGE", "local/headlight-ai-module:dev"),
        "HEADLIGHT_CONTROLLER_CONTEXT": os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3"),
        "HEADLIGHT_SKIP_SMO": os.environ.get("HEADLIGHT_SKIP_SMO", "1"),
        "HEADLIGHT_SKIP_LEGACY_LOCUST": os.environ.get("HEADLIGHT_SKIP_LEGACY_LOCUST", "1"),
        "HEADLIGHT_TRAINTICKET_NAMESPACE": topology["namespace"],
        "HEADLIGHT_TRAINTICKET_PATH": trainticket_path(),
        "HEADLIGHT_HPA_CONTEXTS": " ".join(topology["active_clusters"]),
        "HEADLIGHT_KEDA_CONTEXT": topology["active_clusters"][0],
        "HEADLIGHT_SKIP_CONTROLLER_WORKLOAD_DEPLOY": "1",
        "HEADLIGHT_MATCH_TOPOLOGY_CAPACITY": "1",
        # Controller setup scripts run with cwd=setup.parent. Keep the output
        # location absolute so relative experiment output paths don't resolve
        # under ai-module/ instead of the repository root.
        "HEADLIGHT_RENDERED_HPA_DIR": str((seed_dir / "raw" / "hpa").resolve()),
        "HEADLIGHT_STATIC_PLAN_FILE": os.environ.get("HEADLIGHT_STATIC_PLAN_FILE", str(static_plan_path())),
    }
    if seed is not None and manifest["spec"]["controller"]["controller"] in {"headlight", "headlight-v2"}:
        env["HEADLIGHT_RUN_ID"] = f"{manifest['metadata']['name']}__seed-{seed}"
        env["HEADLIGHT_SEED"] = str(seed)
    for key in (
        "HEADLIGHT_V2_IMAGE",
        "HEADLIGHT_V2_BUNDLE_DIR",
        "HEADLIGHT_V2_POLICY_CONFIG",
        "HEADLIGHT_V2_RUNTIME_LOCK",
        "HEADLIGHT_V2_ENABLE_ACTUATION",
        "HEADLIGHT_V2_SHADOW_EVIDENCE",
        "HEADLIGHT_CONTROLLER_PROMETHEUS_URL",
        "HEADLIGHT_V2_PROMETHEUS_CLUSTERS",
        "HEADLIGHT_V2_METRIC_DIAGNOSTICS",
        "HEADLIGHT_V2_CPU_RATE_WINDOW",
        "HEADLIGHT_V2_POLICY_REVISION",
        "HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY",
        "HEADLIGHT_V2_DIAGNOSTIC_DIR",
        "HEADLIGHT_V2_DIAGNOSTIC_MAX_BYTES",
        "HEADLIGHT_V1_DIAGNOSTIC_LOG",
        "HEADLIGHT_V1_DIAGNOSTIC_MAX_BYTES",
        "HEADLIGHT_V1_PROMETHEUS_CLUSTERS",
        "HEADLIGHT_CODE_REVISION",
        "HEADLIGHT_IMAGE_REVISION",
        "HEADLIGHT_POD_KUBECONFIG",
        "HEADLIGHT_THRESHOLD_IMAGE",
        "HEADLIGHT_PROMETHEUS_URL",
        "HEADLIGHT_ACTION_HANDLER_URL",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    workload = manifest["spec"].get("workload", {})
    locust = workload.get("locust", {})
    overrides = manifest["spec"].get("run", {}).get("qualification_overrides", {})
    bootstrap_fields = {
        "HEADLIGHT_BOOTSTRAP_DURATION_SECONDS": workload.get("duration_seconds"),
        "HEADLIGHT_BOOTSTRAP_START_USERS": locust.get("start_users"),
        "HEADLIGHT_BOOTSTRAP_END_USERS": locust.get("end_users"),
        "HEADLIGHT_BOOTSTRAP_RAMP_SECONDS": locust.get("ramp_seconds"),
        "HEADLIGHT_BOOTSTRAP_HOLD_SECONDS": locust.get("hold_seconds"),
        "HEADLIGHT_BOOTSTRAP_SPAWN_RATE": locust.get("spawn_rate"),
        "HEADLIGHT_BOOTSTRAP_TRAINTICKET_MODE": overrides.get("transaction_mix") or os.environ.get(
            "HEADLIGHT_TRAINTICKET_LOCUST_MODE",
            trainticket_path(),
        ),
    }
    env.update({key: str(value) for key, value in bootstrap_fields.items() if value is not None})
    for key in [
        "HEADLIGHT_CANDIDATE",
        "HEADLIGHT_STEP_NUM_IN",
        "HEADLIGHT_NUM_BLOCKS",
        "HEADLIGHT_HIDDEN_WIDTH",
        "HEADLIGHT_LEARNING_RATE",
        "HEADLIGHT_RESIDUAL_BOUND",
    ]:
        if os.environ.get(key):
            env[key] = os.environ[key]
    controller_name = manifest["spec"]["controller"]["controller"]
    if controller_name in {'static-matched','lstm-matched'}:
        from experiments.lib.matched_runtime_artifacts import capture_matched_artifacts
        configured_amendment_dir = manifest["spec"]["run"].get("static_amendment_dir") or manifest["spec"]["run"].get("lstm_amendment_dir")
        configured_freeze_dir = os.environ.get("HEADLIGHT_MATCHED_FREEZE_DIR")
        if configured_amendment_dir:
            configured_amendment_dir = str((REPO_ROOT / configured_amendment_dir).resolve())
            if configured_freeze_dir and Path(configured_freeze_dir).resolve() != Path(configured_amendment_dir):
                raise ValueError("HEADLIGHT_MATCHED_FREEZE_DIR differs from the manifest amendment directory")
            os.environ["HEADLIGHT_MATCHED_FREEZE_DIR"] = configured_amendment_dir
        capture_matched_artifacts(controller_name, seed_dir/'raw')
        freeze_dir=Path(os.environ['HEADLIGHT_MATCHED_FREEZE_DIR']).resolve()
        env.update(HEADLIGHT_MATCHED_FREEZE_DIR=str(freeze_dir),HEADLIGHT_MATCHED_RUN_ID=f"{manifest['metadata']['name']}__seed-{seed}",HEADLIGHT_MATCHED_SEED=str(seed))
        if controller_name=='static-matched':
            env.update(HEADLIGHT_STATIC_PLAN_FILE=str(freeze_dir/'static-plan.json'),HEADLIGHT_STATIC_MATCHED='1')
        if manifest["spec"]["run"].get("lstm_amendment_dir"):
            env["HEADLIGHT_MATCHED_EVALUATION_MODE"] = "1"
        for key in ('HEADLIGHT_MATCHED_LSTM_IMAGE','HEADLIGHT_MATCHED_LIVE_POLICY','HEADLIGHT_SERVICE_PROFILE_FILE'):
            if os.environ.get(key): env[key]=os.environ[key]
    qualification_kind = overrides.get("kind")
    capacity_pilot_bootstrap = qualification_kind == "capacity-pilot"
    if controller_name == "static":
        if capacity_pilot_bootstrap:
            env["HEADLIGHT_CAPACITY_PILOT_BOOTSTRAP"] = "1"
            bootstrap = {
                "schema_version": "headlight.capacity-pilot-bootstrap/v1",
                "mode": "profile-minimum-fixed-placement",
                "static_plan_required": False,
                "placements": {"cluster2": 1, "cluster3": 1},
            }
            (seed_dir / "raw" / "static_plan.json").write_text(
                json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            plan = validate_static_plan(Path(env["HEADLIGHT_STATIC_PLAN_FILE"]))
            (seed_dir / "raw" / "static_plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if controller_name == "keda":
        for cluster in topology["active_clusters"]:
            cluster_max = str(DEFAULT_CLUSTER_REPLICA_SPLIT.get(cluster, REPLICA_CEILING // 2))
            run_cmd(
                ["bash", str(setup)],
                seed_dir / "logs" / "controller-setup.log",
                env={**env, "HEADLIGHT_KEDA_CONTEXT": cluster, "HEADLIGHT_KEDA_MAX_REPLICAS": cluster_max},
                cwd=setup.parent,
            )
    else:
        run_cmd(["bash", str(setup)], seed_dir / "logs" / "controller-setup.log", env=env, cwd=setup.parent)
    if controller_name == "hpa":
        (seed_dir / "raw" / "hpa_controller_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "headlight.hpa-controller-metadata/v1",
                    "native_controller_path": True,
                    "sync_period_seconds": {
                        "expected": 15,
                        "observed": None,
                        "source": "Kubernetes kube-controller-manager default unless cluster configuration exposes a different value",
                    },
                    "cpu_target_average_utilization": PROTOCOL["spec"]["hpa_fairness"]["cpu_target_average_utilization"],
                    "scale_up_behavior": "kubernetes-default",
                    "scale_down_behavior": PROTOCOL["spec"]["hpa_fairness"]["downscale_behavior"],
                    "initialization_behavior": PROTOCOL["spec"]["hpa_fairness"]["initialization_behavior"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def stop_controller(manifest: dict[str, Any], seed_dir: Path, *, check: bool = True) -> None:
    """Stop the cell's controller without changing the workload placement."""
    controller = manifest["spec"]["controller"]["controller"]
    topology = manifest["spec"]["topology"]
    namespace = topology["namespace"]
    log_path = seed_dir / "logs" / "controller-stop.log"
    deployment = {
        "headlight": "ai-module",
        "headlight-v2": "ai-module-v2",
        "lstm": "lstm-autoscaler",
        "lstm-matched": "lstm-matched-autoscaler",
        "threshold": "threshold-autoscaler",
    }.get(controller)
    if deployment is not None:
        context = os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3")
        run_cmd(
            [
                "kubectl", "--context", context, "-n", namespace,
                "scale", "deployment", deployment, "--replicas=0",
            ],
            log_path,
            check=check,
        )
        return
    if controller == "hpa":
        for context in topology["active_clusters"]:
            run_cmd(
                ["kubectl", "--context", context, "-n", namespace, "delete", "hpa", "--all", "--ignore-not-found"],
                log_path,
                check=check,
            )
        return
    if controller == "keda":
        for context in topology["active_clusters"]:
            run_cmd(
                [
                    "kubectl", "--context", context, "-n", namespace,
                    "delete", "scaledobject", "--all", "--ignore-not-found",
                ],
                log_path,
                check=check,
            )
            run_cmd(
                ["kubectl", "--context", context, "-n", namespace, "delete", "hpa", "--all", "--ignore-not-found"],
                log_path,
                check=check,
            )
        return
    log_path.write_text(f"controller={controller} has no active controller resource to stop\n", encoding="utf-8")


def validate_frozen_controller_bundle(manifest: dict[str, Any], seed_dir: Path) -> None:
    controller = manifest["spec"]["controller"]["controller"]
    if controller in {'static-matched','lstm-matched'}:
        from experiments.lib.matched_runtime_artifacts import capture_matched_artifacts
        capture_matched_artifacts(controller,seed_dir/'raw')
        return
    if controller == "headlight-v2":
        bundle = Path(os.environ["HEADLIGHT_V2_BUNDLE_DIR"])
        policy_config = Path(os.environ["HEADLIGHT_V2_POLICY_CONFIG"])
        python = REPO_ROOT / ".venv" / "bin" / "python"
        if not python.exists():
            python = Path(sys.executable)
        run_cmd(
            [
                str(python),
                "-c",
                "from headlight.bundle_v2 import HeadlightV2Bundle; import sys; HeadlightV2Bundle.load(sys.argv[1], expected_model_type='headlight-v2')",
                str(bundle),
            ],
            seed_dir / "logs" / "bundle-preflight.log",
            env={"PYTHONPATH": str(REPO_ROOT / "ai-module" / "ai-module")},
        )
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        policy_payload = json.loads(policy_config.read_text(encoding="utf-8"))
        provenance = {
            "schema_version": "headlight.model-policy-provenance/v2",
            "controller": controller,
            "bundle_path": str(bundle),
            "bundle_id": metadata.get("bundle_id"),
            "bundle_sha256": directory_sha256(bundle),
            "bundle_metadata": metadata,
            "policy_config_path": str(policy_config),
            "policy_config_sha256": file_sha256(policy_config),
            "no_online_retraining": metadata.get("runtime_reload") is False,
        }
        (seed_dir / "raw" / "model_bundle_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (seed_dir / "raw" / "policy_config.json").write_text(
            json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if controller not in {"headlight", "lstm"}:
        return
    path = trainticket_path()
    profile = Path(
        os.environ.get(
            "HEADLIGHT_SERVICE_PROFILE_FILE",
            str(REPO_ROOT / "ai-module" / "configs" / "service_profiles" / f"{path}.json"),
        )
    )
    bundle_env = "HEADLIGHT_MODEL_BUNDLE_DIR" if controller == "headlight" else "HEADLIGHT_LSTM_BUNDLE_DIR"
    bundle = Path(
        os.environ.get(
            bundle_env,
            str(REPO_ROOT / "results" / "calibration" / f"{controller}-bundle"),
        )
    )
    policy_config = policy_config_path()
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    run_cmd(
        [
            str(python),
            str(REPO_ROOT / "ai-module" / "ai-module" / "validate_model_bundle.py"),
            "--bundle",
            str(bundle),
            "--service-profile",
            str(profile),
            "--controller",
            controller,
            "--policy-config",
            str(policy_config),
        ],
        seed_dir / "logs" / "bundle-preflight.log",
    )
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    policy_payload = json.loads(policy_config.read_text(encoding="utf-8"))
    provenance = {
        "schema_version": "headlight.model-policy-provenance/v1",
        "controller": controller,
        "bundle_path": str(bundle),
        "bundle_id": metadata.get("bundle_id"),
        "bundle_sha256": directory_sha256(bundle),
        "bundle_metadata": metadata,
        "bundle_directory_read_only": not bool(bundle.stat().st_mode & 0o222),
        "policy_config_path": str(policy_config),
        "policy_id": policy_payload.get("policy_id"),
        "policy_configuration_hash": policy_payload.get("policy_configuration_hash"),
        "policy_config_sha256": file_sha256(policy_config),
        "policy_provenance": policy_payload.get("provenance", {}),
        "no_online_retraining": metadata.get("runtime_reload") is False,
    }
    (seed_dir / "raw" / "model_bundle_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / "policy_config.json").write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_bundle_unchanged(manifest: dict[str, Any], seed_dir: Path) -> None:
    controller = manifest["spec"]["controller"]["controller"]
    if controller not in {"headlight", "headlight-v2", "lstm", "lstm-matched"}:
        return
    provenance_path = seed_dir / "raw" / "model_bundle_provenance.json"
    if not provenance_path.exists():
        return
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    bundle = Path(provenance["bundle_path"])
    after_hash = directory_sha256(bundle)
    provenance["bundle_sha256_after_evaluation"] = after_hash
    provenance["bundle_unchanged_during_evaluation"] = after_hash == provenance.get("bundle_sha256")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _locust_settings() -> tuple[str, str, str]:
    default_cluster = (
        GATEWAY_PROTOCOL["spec"]["load_generator_cluster"]
        if GATEWAY_PROTOCOL is not None
        else str(common_topology(PROTOCOL)["load_generator_cluster"])
    )
    cluster = os.environ.get("HEADLIGHT_LOCUST_CLUSTER", default_cluster)
    gateway_url = os.environ.get("HEADLIGHT_TRAINTICKET_GATEWAY_URL", "")
    if GATEWAY_PROTOCOL is not None:
        gateway_spec = GATEWAY_PROTOCOL["spec"]
        if cluster != gateway_spec["load_generator_cluster"]:
            raise RuntimeError("gateway protocol requires Locust in cluster1")
        if gateway_url and gateway_url != gateway_spec["gateway_url"]:
            raise RuntimeError("gateway URL differs from the gateway protocol")
        gateway_url = gateway_spec["gateway_url"]
    elif gateway_url:
        raise RuntimeError("gateway URL requires HEADLIGHT_GATEWAY_PROTOCOL_PATH")
    path = trainticket_path()
    mode = os.environ.get(
        "HEADLIGHT_TRAINTICKET_LOCUST_MODE",
        "full-session" if path == "full" else path,
    )
    if GATEWAY_PROTOCOL is not None and (path != "advanced-ticket" or mode != "advanced-ticket"):
        raise RuntimeError("gateway protocol requires the advanced-ticket workload and Locust mode")
    return cluster, gateway_url, mode


def validate_matched_runtime_settings() -> None:
    if os.environ.get("HEADLIGHT_SHADOW_MODE", "0") not in {"", "0"}:
        raise RuntimeError("matched v1 comparison requires active actuation, not shadow mode")
    _locust_settings()


def verify_matched_controller_active(manifest: dict[str, Any], seed_dir: Path,
                                     evidence_name: str = "matched_controller_runtime.json") -> None:
    controller = manifest["spec"].get("controller", {}).get("controller", "headlight")
    deployment_name, container_name = ("threshold-autoscaler", "autoscaler") if controller == "threshold" else ("ai-module", "ai-module")
    deployment = kubectl_json(
        ["-n", manifest["spec"]["topology"]["namespace"], "get", "deployment", deployment_name, "-o", "json"],
        os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3"),
        seed_dir / "logs" / "controller-setup.log",
    )
    containers = deployment.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    container = next((item for item in containers if item.get("name") == container_name), {})
    shadow_setting = next((item for item in container.get("env", []) if item.get("name") == "HEADLIGHT_SHADOW_MODE"), {})
    if shadow_setting.get("value") != "0" or "valueFrom" in shadow_setting:
        raise RuntimeError(f"deployed {controller} controller is not explicitly configured for active actuation")
    if int(deployment.get("spec", {}).get("replicas", 0)) < 1 or int(deployment.get("status", {}).get("availableReplicas", 0)) < 1:
        raise RuntimeError(f"matched {controller} controller has no available replica")
    pod_images = []
    if controller == "threshold":
        expected_image = os.environ.get("HEADLIGHT_THRESHOLD_IMAGE")
        expected_digest = os.environ.get("HEADLIGHT_THRESHOLD_IMAGE_ID")
        if not expected_image or container.get("image") != expected_image or not expected_digest:
            raise RuntimeError("matched Threshold requires explicit expected image and image digest")
        pods = kubectl_json(
            ["-n", manifest["spec"]["topology"]["namespace"], "get", "pods", "-l", "app=threshold-autoscaler", "-o", "json"],
            os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3"), seed_dir / "logs/controller-setup.log",
        )
        for pod in pods.get("items", []):
            if pod.get("metadata", {}).get("deletionTimestamp"):
                continue
            for status in pod.get("status", {}).get("containerStatuses", []):
                if status.get("name") == container_name and status.get("ready") is True:
                    pod_images.append({"pod": pod["metadata"]["name"], "image_id": status.get("imageID", "")})
        if not pod_images or any(item["image_id"].split("@")[-1] != expected_digest for item in pod_images):
            raise RuntimeError("deployed Threshold image digest differs from the qualified smoke image digest")
    (seed_dir / "raw" / evidence_name).write_text(
        json.dumps({"controller": controller, "active_actuation": True, "shadow_mode": "0", "image": container.get("image"),
                    "pod_images": pod_images,
                    "expected_image_digest": os.environ.get("HEADLIGHT_THRESHOLD_IMAGE_ID") if controller == "threshold" else None,
                    "deployment": deployment.get("metadata", {})}, indent=2) + "\n", encoding="utf-8"
    )


def run_locust(manifest: dict[str, Any], seed_dir: Path, duration_seconds: int,
               health_check: Callable[[], None] | None = None) -> None:
    namespace = manifest["spec"]["topology"]["namespace"]
    cluster, gateway_url, mode = _locust_settings()
    image = os.environ.get("HEADLIGHT_LOCUST_IMAGE", "dstefanov46/locust-test:0.66.0")
    spawn_rate = str(manifest["spec"]["workload"]["locust"]["spawn_rate"])
    wait_timeout = os.environ.get("HEADLIGHT_LOCUST_WAIT_TIMEOUT", f"{duration_seconds + 900}s")
    log_path = seed_dir / "logs" / "locust.log"
    run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "job", "headlight-paper-locust", "--ignore-not-found"], log_path)
    run_cmd(["kubectl", "--context", cluster, "-n", namespace, "delete", "configmap", "headlight-paper-locust", "--ignore-not-found"], log_path)
    run_cmd(
        [
            "kubectl",
            "--context",
            cluster,
            "-n",
            namespace,
            "create",
            "configmap",
            "headlight-paper-locust",
            f"--from-file=trainticket_paper.py={REPO_ROOT / 'experiments' / 'testbed' / 'trainticket-paper' / 'trainticket_paper.py'}",
            f"--from-file=load_profile.csv={seed_dir / 'raw' / 'load_profile.csv'}",
        ],
        log_path,
    )
    locust_env = [
        {"name": "HEADLIGHT_REQUEST_CSV", "value": "/app/headlight_requests.csv"},
        {"name": "HEADLIGHT_LOAD_SHAPE", "value": "/app/load_profile.csv"},
        {"name": "HEADLIGHT_LOAD_SHAPE_DURATION_SECONDS", "value": str(duration_seconds)},
        {"name": "HEADLIGHT_TRAINTICKET_MODE", "value": mode},
        {"name": "HEADLIGHT_TRAINTICKET_NAMESPACE", "value": namespace},
    ]
    if gateway_url:
        locust_env.append({"name": "HEADLIGHT_TRAINTICKET_GATEWAY_URL", "value": gateway_url})
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "headlight-paper-locust", "namespace": namespace},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "locust",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-lc"],
                            "args": [
                                "\n".join(
                                    [
                                        "/opt/venv/bin/locust --headless -f /app/trainticket_paper.py "
                                        f'--spawn-rate "{spawn_rate}" --run-time "{duration_seconds}s" '
                                        "--exit-code-on-error 0 --csv /app/headlight-locust",
                                        'rc="$?"',
                                        "echo __HEADLIGHT_REQUEST_CSV_BEGIN__",
                                        "cat /app/headlight_requests.csv 2>/dev/null || true",
                                        "echo __HEADLIGHT_REQUEST_CSV_END__",
                                        "echo __HEADLIGHT_LOCUST_STATS_BEGIN__",
                                        "cat /app/headlight-locust_stats.csv 2>/dev/null || true",
                                        "echo __HEADLIGHT_LOCUST_STATS_END__",
                                        'exit "$rc"',
                                    ]
                                )
                            ],
                            "env": locust_env,
                            "volumeMounts": [
                                {"name": "locustfile", "mountPath": "/app/trainticket_paper.py", "subPath": "trainticket_paper.py"},
                                {"name": "locustfile", "mountPath": "/app/load_profile.csv", "subPath": "load_profile.csv"},
                            ],
                        }
                    ],
                    "volumes": [{"name": "locustfile", "configMap": {"name": "headlight-paper-locust"}}],
                },
            },
        },
    }
    job_path = seed_dir / "raw" / "locust-job.yaml"
    write_yaml(job_path, job_manifest)
    run_cmd(["kubectl", "--context", cluster, "apply", "-f", str(job_path)], log_path)
    wait_result = run_cmd(
        ["kubectl", "--context", cluster, "-n", namespace, "wait", "--for=condition=Complete", "job/headlight-paper-locust", f"--timeout={wait_timeout}"],
        log_path,
        check=False,
        health_check=health_check,
    )
    pod = run_cmd(
        ["kubectl", "--context", cluster, "-n", namespace, "get", "pod", "-l", "job-name=headlight-paper-locust", "-o", "jsonpath={.items[0].metadata.name}"],
        log_path,
        check=False,
    ).stdout.strip()
    if wait_result.returncode != 0:
        if pod:
            describe = run_cmd(["kubectl", "--context", cluster, "-n", namespace, "describe", "pod", pod], log_path, check=False).stdout
            (seed_dir / "logs" / "locust-pod-describe.txt").write_text(describe, encoding="utf-8")
            pod_logs = run_cmd(["kubectl", "--context", cluster, "-n", namespace, "logs", pod], log_path, check=False).stdout
            (seed_dir / "logs" / "locust-pod.log").write_text(pod_logs, encoding="utf-8")
        raise RuntimeError(f"Locust job did not complete within {wait_timeout}")
    if not pod:
        raise RuntimeError("Locust job completed but no pod was found")
    logs = run_cmd(["kubectl", "--context", cluster, "-n", namespace, "logs", pod], log_path).stdout
    (seed_dir / "logs" / "locust-pod.log").write_text(logs, encoding="utf-8")
    (seed_dir / "raw" / REQUEST_EVENTS_FILE).write_text(
        extract_marked_block(logs, "__HEADLIGHT_REQUEST_CSV_BEGIN__", "__HEADLIGHT_REQUEST_CSV_END__"),
        encoding="utf-8",
    )
    (seed_dir / "raw" / "locust_stats.csv").write_text(
        extract_marked_block(logs, "__HEADLIGHT_LOCUST_STATS_BEGIN__", "__HEADLIGHT_LOCUST_STATS_END__"),
        encoding="utf-8",
    )


def prometheus_query_range(port: int, query: str, start: int, end: int, step: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/query_range?{params}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _merge_prometheus_responses(responses: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    result_type = "matrix"
    for cluster, response in responses:
        if response.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed in {cluster}: {response}")
        data = response.get("data", {})
        result_type = str(data.get("resultType", result_type))
        for series in data.get("result", []):
            merged = dict(series)
            labels = dict(series.get("metric", {}))
            labels["cluster"] = cluster
            merged["metric"] = labels
            result.append(merged)
    return {"status": "success", "data": {"resultType": result_type, "result": result}}


def collect_prometheus(manifest: dict[str, Any], seed_dir: Path, started_at: int, finished_at: int, interval: int) -> None:
    cluster_override = os.environ.get("HEADLIGHT_PROMETHEUS_CLUSTER")
    clusters = [cluster_override] if cluster_override else list(manifest["spec"]["topology"]["active_clusters"])
    namespace = manifest["spec"]["topology"]["namespace"]
    prometheus_namespace = os.environ.get("HEADLIGHT_PROMETHEUS_NAMESPACE", "monitoring")
    prometheus_service = os.environ.get("HEADLIGHT_PROMETHEUS_SERVICE", "prometheus-kube-prometheus-prometheus")
    prometheus_port = int(os.environ.get("HEADLIGHT_PROMETHEUS_PORT", "39090"))
    log_path = seed_dir / "logs" / "prometheus.log"
    queries = export_queries(namespace, f"{interval}s")
    responses: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in queries}
    port_forward_log = seed_dir / "logs" / "prometheus-port-forward.log"
    port_forward_log.write_text("", encoding="utf-8")
    for cluster in clusters:
        with port_forward_log.open("a", encoding="utf-8") as forward_log:
            port_forward = subprocess.Popen(
                [
                    "kubectl",
                    "--context",
                    cluster,
                    "-n",
                    prometheus_namespace,
                    "port-forward",
                    f"svc/{prometheus_service}",
                    f"{prometheus_port}:9090",
                ],
                cwd=REPO_ROOT,
                stdout=forward_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{prometheus_port}/-/ready", timeout=1).read()
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError(f"Prometheus port-forward did not become ready for {cluster}")
                for name, query in queries.items():
                    response = prometheus_query_range(prometheus_port, query, started_at, finished_at, f"{interval}s")
                    responses[name].append((cluster, response))
            finally:
                port_forward.send_signal(signal.SIGINT)
                try:
                    port_forward.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    port_forward.kill()
                    port_forward.wait(timeout=10)

    payload: dict[str, Any] = {"source_clusters": clusters}
    nonempty = False
    for name, query in queries.items():
        response = _merge_prometheus_responses(responses[name])
        payload[name] = {"query": query, "response": response}
        nonempty = nonempty or bool(response.get("data", {}).get("result"))
    (seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not nonempty:
        raise RuntimeError("Prometheus returned no time series")
    log_path.write_text(
        f"queried {prometheus_service}.{prometheus_namespace} on {', '.join(clusters)}\n",
        encoding="utf-8",
    )


def collect_workload_snapshot(manifest: dict[str, Any], seed_dir: Path, phase: str) -> None:
    """Capture readiness and restart counters at a measurement boundary."""
    namespace = manifest["spec"]["topology"]["namespace"]
    clusters = manifest["spec"]["topology"]["active_clusters"]
    pods: dict[str, Any] = {}
    deployments: dict[str, Any] = {}
    for cluster in clusters:
        log_path = seed_dir / "logs" / f"kubectl-{cluster}.log"
        pods[cluster] = kubectl_json(["-n", namespace, "get", "pods", "-o", "json"], cluster, log_path, check=False)
        deployments[cluster] = kubectl_json(["-n", namespace, "get", "deployments", "-o", "json"], cluster, log_path, check=False)
    (seed_dir / "raw" / f"pods_{phase}.json").write_text(json.dumps(pods, indent=2) + "\n", encoding="utf-8")
    (seed_dir / "raw" / f"deployments_{phase}.json").write_text(json.dumps(deployments, indent=2) + "\n", encoding="utf-8")
    crps = kubectl_json(
        ["-n", namespace, "get", "controlledreplicapropagations", "-o", "json"],
        "karmada-apiserver",
        seed_dir / "logs" / f"crp-{phase}.log",
        check=False,
    )
    (seed_dir / "raw" / f"crps_{phase}.json").write_text(
        json.dumps(crps, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_inference_timings(log_text: str, controller: str, variant: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    marker = "HEADLIGHT_INFERENCE_TIMING "
    for line in log_text.splitlines():
        if controller == 'lstm-matched' and '"event":"LSTM_MATCHED_FORECAST"' in line:
            try:
                payload = json.loads(line[line.index("{"):])
                rows.append({"timestamp":"", "controller":controller,
                             "model_variant":f"{variant}:forecaster",
                             "service_count":str(payload['service_count']),
                             "duration_ms":str(payload['duration_ms']), "applicable":"true"})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            continue
        if '"event":"HEADLIGHT_V2_INFERENCE"' in line:
            try:
                payload = json.loads(line[line.index("{"):])
                rows.append({
                    "timestamp": "",
                    "controller": controller,
                    "model_variant": f"{variant}:scorer" if controller == 'lstm-matched' else variant,
                    "service_count": "1",
                    "duration_ms": str(payload["duration_ms"]),
                    "applicable": "true",
                })
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            continue
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
            rows.append(
                {
                    "timestamp": str(payload["timestamp"]),
                    "controller": str(payload.get("controller", controller)),
                    "model_variant": str(payload.get("model_variant", variant)),
                    "service_count": str(payload["service_count"]),
                    "duration_ms": str(payload["duration_ms"]),
                    "applicable": "true",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def write_inference_timings(path: Path, log_text: str, controller: str, variant: str) -> None:
    fields = ["timestamp", "controller", "model_variant", "service_count", "duration_ms", "applicable"]
    rows = parse_inference_timings(log_text, controller, variant)
    if controller == "headlight" and not rows and "Num. of replicas for " in log_text:
        rows = [
            {
                "timestamp": "",
                "controller": controller,
                "model_variant": f"{variant}:legacy-uninstrumented",
                "service_count": "",
                "duration_ms": "",
                "applicable": "legacy-uninstrumented",
            }
        ]
    if controller not in {"headlight", "headlight-v2", "lstm", "lstm-matched"}:
        rows = [{"timestamp": "", "controller": controller, "model_variant": variant, "service_count": "", "duration_ms": "", "applicable": "false"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_kubernetes_state(manifest: dict[str, Any], seed_dir: Path) -> None:
    namespace = manifest["spec"]["topology"]["namespace"]
    clusters = manifest["spec"]["topology"]["active_clusters"]
    events: dict[str, Any] = {}
    pods: dict[str, Any] = {}
    deployments: dict[str, Any] = {}
    for cluster in clusters:
        log_path = seed_dir / "logs" / f"kubectl-{cluster}.log"
        events[cluster] = kubectl_json(["-n", namespace, "get", "events", "-o", "json"], cluster, log_path, check=False)
        pods[cluster] = kubectl_json(["-n", namespace, "get", "pods", "-o", "json"], cluster, log_path, check=False)
        deployments[cluster] = kubectl_json(["-n", namespace, "get", "deployments", "-o", "json"], cluster, log_path, check=False)
    (seed_dir / "raw" / "kubernetes_events.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    (seed_dir / "raw" / "pod_restarts.json").write_text(json.dumps(pods, indent=2) + "\n", encoding="utf-8")
    (seed_dir / "raw" / "deployment_status.json").write_text(json.dumps(deployments, indent=2) + "\n", encoding="utf-8")
    crps = kubectl_json(["-n", namespace, "get", "controlledreplicapropagations", "-o", "json"], "karmada-apiserver", seed_dir / "logs" / "crp.log", check=False)
    (seed_dir / "raw" / "crp_transitions.json").write_text(json.dumps(crps, indent=2) + "\n", encoding="utf-8")
    actions_selector = os.environ.get("HEADLIGHT_ACTIONS_HANDLER_SELECTOR", "app.kubernetes.io/name=actions-handler")
    actions_logs = run_cmd(
        ["kubectl", "--context", "karmada-host", "-n", "actions-handler-system", "logs", "-l", actions_selector, "--tail=-1"],
        seed_dir / "logs" / "actions-handler.log",
        check=False,
    ).stdout
    (seed_dir / "raw" / "actions_handler_batches.json").write_text(json.dumps({"logs": actions_logs.splitlines()}, indent=2) + "\n", encoding="utf-8")
    controller = manifest["spec"]["controller"]["controller"]
    variant = manifest["spec"]["controller"].get("controller_variant", controller)
    controller_context = os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3")
    controller_namespace = os.environ.get("HEADLIGHT_CONTROLLER_NAMESPACE", namespace)
    selector = {
        "headlight": "app=ai-module",
        "headlight-v2": "app=ai-module-v2",
    }.get(controller, f"app={controller}-autoscaler")
    ai_logs = run_cmd(
        ["kubectl", "--context", controller_context, "-n", controller_namespace, "logs", "-l", selector, "--tail=-1"],
        seed_dir / "logs" / "controller.log",
        check=False,
    ).stdout
    if controller == "headlight":
        file_logs = run_cmd(
            [
                "kubectl",
                "--context",
                controller_context,
                "-n",
                controller_namespace,
                "exec",
                "deploy/ai-module",
                "--",
                "sh",
                "-lc",
                "cat /app/logs/__main__.log 2>/dev/null || true",
            ],
            seed_dir / "logs" / "controller-file.log",
            check=False,
        ).stdout
        if file_logs:
            ai_logs = ai_logs + "\n" + file_logs
    if controller == "headlight" and os.environ.get("HEADLIGHT_V1_DIAGNOSTIC_LOG"):
        pods = json.loads(run_cmd(
            ["kubectl", "--context", controller_context, "-n", controller_namespace,
             "get", "pods", "-l", "app=ai-module", "-o", "json"],
            seed_dir / "logs" / "v1-diagnostic-pod-collection.log",
        ).stdout)
        running = [pod for pod in pods["items"] if pod.get("status", {}).get("phase") == "Running"
                   and not pod["metadata"].get("deletionTimestamp")]
        if len(running) != 1:
            raise RuntimeError("exactly one running v1 pod is required to archive diagnostics")
        pod = running[0]
        (seed_dir / "raw" / "controller_pod_provenance.json").write_text(json.dumps(pod, indent=2) + "\n")
        run_cmd(
            ["kubectl", "--context", controller_context, "-n", controller_namespace,
             "cp", "-c", "ai-module",
             f"{pod['metadata']['name']}:{os.environ['HEADLIGHT_V1_DIAGNOSTIC_LOG']}",
             str(seed_dir / "raw" / "controller_diagnostics.jsonl")],
            seed_dir / "logs" / "v1-diagnostic-artifact-collection.log",
        )
    if controller == "headlight-v2" and os.environ.get("HEADLIGHT_V2_DIAGNOSTIC_DIR"):
        pods = json.loads(run_cmd(
            ["kubectl", "--context", controller_context, "-n", controller_namespace,
             "get", "pods", "-l", "app=ai-module-v2", "-o", "json"],
            seed_dir / "logs" / "diagnostic-pod-collection.log",
        ).stdout)
        running = [pod for pod in pods["items"] if pod.get("status", {}).get("phase") == "Running"
                   and not pod["metadata"].get("deletionTimestamp")]
        if len(running) != 1:
            raise RuntimeError("exactly one running Headlight pod is required to archive diagnostics")
        pod = running[0]
        (seed_dir / "raw" / "controller_pod_provenance.json").write_text(json.dumps(pod, indent=2) + "\n")
        run_cmd(
            ["kubectl", "--context", controller_context, "-n", controller_namespace,
             "cp", "-c", "ai-module-v2",
             f"{pod['metadata']['name']}:{os.environ['HEADLIGHT_V2_DIAGNOSTIC_DIR']}",
             str(seed_dir / "raw" / "diagnostics")],
            seed_dir / "logs" / "diagnostic-artifact-collection.log",
        )
    deployment_name = {
        "headlight": "ai-module",
        "headlight-v2": "ai-module-v2",
        "lstm": "lstm-autoscaler",
        "lstm-matched": "lstm-matched-autoscaler",
        "threshold": "threshold-autoscaler",
    }.get(controller, "lstm-autoscaler")
    if controller in {"headlight", "headlight-v2", "lstm", "lstm-matched", "threshold"}:
        action_log = run_cmd(
            [
                "kubectl",
                "--context",
                controller_context,
                "-n",
                controller_namespace,
                "exec",
                f"deploy/{deployment_name}",
                "--",
                "sh",
                "-lc",
                "cat /artifacts/actions-v2.jsonl 2>/dev/null || cat /artifacts/actions.jsonl 2>/dev/null || true",
            ],
            seed_dir / "logs" / "controller-actions.log",
            check=False,
        ).stdout
        merge_controller_actions(seed_dir, collected_text=action_log)
        if controller in {"headlight", "headlight-v2", "lstm", "lstm-matched"}:
            bundle_metadata = run_cmd(
                [
                    "kubectl",
                    "--context",
                    controller_context,
                    "-n",
                    controller_namespace,
                    "exec",
                    f"deploy/{deployment_name}",
                    "--",
                    "sh",
                    "-lc",
                    "cat /models/bundle/metadata.json",
                ],
                seed_dir / "logs" / "model-bundle.log",
                check=False,
            ).stdout
            (seed_dir / "raw" / "model_bundle_metadata.json").write_text(
                bundle_metadata,
                encoding="utf-8",
            )
    write_inference_timings(seed_dir / "raw" / "ai_inference_times.csv", ai_logs, controller, variant)
    (seed_dir / "logs" / "controller.log").write_text(ai_logs, encoding="utf-8")


def assert_deployments_ready(seed_dir: Path, *, strict: bool = True) -> None:
    data = json.loads((seed_dir / "raw" / "deployment_status.json").read_text(encoding="utf-8"))
    allowed_unready = {
        name
        for name in os.environ.get("HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS", "").split()
        if name
    }
    not_ready: list[str] = []
    for cluster, payload in data.items():
        for item in payload.get("items", []):
            name = item.get("metadata", {}).get("name")
            if name in allowed_unready:
                continue
            spec_replicas = int(item.get("spec", {}).get("replicas", 1))
            ready = int(item.get("status", {}).get("readyReplicas", 0))
            if ready < spec_replicas:
                not_ready.append(f"{cluster}/{name} {ready}/{spec_replicas}")
    if not_ready:
        message = f"deployments not Ready: {', '.join(not_ready)}"
        (seed_dir / "logs" / "deployment-readiness-warning.log").write_text(message + "\n", encoding="utf-8")
        if strict:
            raise RuntimeError(message)
    else:
        (seed_dir / "logs" / "deployment-readiness-warning.log").unlink(missing_ok=True)


def validate_seed_bundle(seed_dir: Path, dry_run: bool, manifest: dict[str, Any]) -> None:
    validate_bundle(seed_dir, manifest, dry_run=dry_run)


def write_measurement_request_csv(source: Path, target: Path, started_at: float, finished_at: float) -> None:
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = reader.fieldnames or ["timestamp", "method", "endpoint", "status", "response_time_ms", "exception"]
        with target.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    timestamp = float(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                if started_at <= timestamp <= finished_at:
                    writer.writerow(row)


def _prometheus_pod_count_samples(payload: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    metric = payload.get("pod_count", {})
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    totals: dict[str, dict[float, float]] = {}
    for series in result:
        labels = series.get("metric", {})
        # Only named workload series are useful for ceiling binding.  An
        # unlabeled aggregate must never become a synthetic service named
        # "unknown", because that turns ordinary aggregate pod counts into a
        # false ceiling hit and downgrades otherwise valid runs.
        deployment = labels.get("label_app") or labels.get("app") or labels.get("deployment")
        if not deployment:
            continue
        deployment = str(deployment)
        service_totals = totals.setdefault(deployment, {})
        for value in series.get("values", []):
            try:
                timestamp = float(value[0])
                service_totals[timestamp] = service_totals.get(timestamp, 0.0) + float(value[1])
            except (IndexError, TypeError, ValueError):
                continue
    return {
        deployment: sorted(service_totals.items())
        for deployment, service_totals in totals.items()
        if service_totals
    }


def write_ceiling_binding_diagnostic(
    manifest: dict[str, Any],
    seed_dir: Path,
    started_at: float,
    finished_at: float,
) -> None:
    duration = max(0.0, finished_at - started_at)
    aggregate_ceiling = REPLICA_CEILING
    prometheus = json.loads((seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE).read_text(encoding="utf-8"))
    source_clusters = [
        str(cluster)
        for cluster in prometheus.get("source_clusters", [])
        if str(cluster) in DEFAULT_CLUSTER_REPLICA_SPLIT
    ]
    observed_ceiling = (
        sum(int(DEFAULT_CLUSTER_REPLICA_SPLIT[cluster]) for cluster in source_clusters)
        if source_clusters
        else aggregate_ceiling
    )
    services_at_ceiling: list[str] = []
    service_fractions: dict[str, float] = {}
    for service, samples in _prometheus_pod_count_samples(prometheus).items():
        in_window = [(timestamp, value) for timestamp, value in samples if started_at <= timestamp <= finished_at]
        if not in_window:
            continue
        ceiling_samples = sum(1 for _timestamp, value in in_window if value >= observed_ceiling)
        fraction = ceiling_samples / len(in_window)
        service_fractions[service] = fraction
        if ceiling_samples:
            services_at_ceiling.append(service)

    threshold = PROTOCOL["spec"]["replica_bounds"]["uninformative_threshold"]
    service_fraction = len(services_at_ceiling) / max(1, len(service_fractions))
    time_fraction = max(service_fractions.values(), default=0.0)
    uninformative = (
        service_fraction >= float(threshold["minimum_fraction_of_services_at_ceiling"])
        and time_fraction >= float(threshold["minimum_fraction_of_measurement_at_ceiling"])
    )
    payload = {
        "schema_version": "headlight.ceiling-binding/v1",
        "controller": manifest["spec"]["controller"]["controller"],
        "workload": manifest["spec"]["workload"]["pattern"],
        "aggregate_ceiling": aggregate_ceiling,
        "per_cluster_ceiling": dict(DEFAULT_CLUSTER_REPLICA_SPLIT),
        "source_clusters": source_clusters,
        "observed_ceiling": observed_ceiling,
        "measurement_seconds": duration,
        "services_at_ceiling": sorted(services_at_ceiling),
        "service_ceiling_sample_fraction": service_fractions,
        "time_at_ceiling_seconds": {
            service: round(duration * fraction, 6)
            for service, fraction in sorted(service_fractions.items())
            if fraction > 0
        },
        "ceiling_hit_invalidates_run": False,
        "ceiling_makes_run_uninformative": uninformative,
    }
    (seed_dir / "raw" / "ceiling_binding.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _execute_seed(
    manifest: dict[str, Any],
    output_dir: Path,
    seed: int,
    dry_run: bool,
    matched_baseline_manifest: Path | None = None,
    evidence: RunEvidence | None = None,
) -> None:
    run = manifest["spec"]["run"]
    topology = manifest["spec"]["topology"]
    workload = manifest["spec"]["workload"]
    interval = int(run.get("sampling_interval_seconds", workload.get("sampling_interval_seconds", 15)))
    warmup_seconds = int(run.get("warmup_seconds", workload.get("warmup_seconds", 0)))
    measurement_seconds = int(run["duration_seconds"])
    diagnostic_observation_tail_seconds = run.get("diagnostic_observation_tail_seconds", 0)
    if (isinstance(diagnostic_observation_tail_seconds, bool)
            or not isinstance(diagnostic_observation_tail_seconds, int)
            or diagnostic_observation_tail_seconds < 0):
        raise ValueError("diagnostic_observation_tail_seconds must be a non-negative integer")
    locust_duration_seconds = warmup_seconds + measurement_seconds + diagnostic_observation_tail_seconds
    cooldown_seconds = int(protocol_timing(PROTOCOL)["cooldown_seconds"])
    seed_dir = output_dir / f"seed-{seed}"
    for subdir in ["logs", "raw", "summaries", "figures"]:
        (seed_dir / subdir).mkdir(parents=True, exist_ok=True)
    if not dry_run:
        for relative in [
            "logs/reset.log",
            "logs/deploy-trainticket.log",
            "logs/controller-setup.log",
            "logs/controller-stop.log",
            "logs/locust.log",
            "logs/locust-pod.log",
            "logs/locust-pod-describe.txt",
            "logs/prometheus.log",
            "logs/prometheus-port-forward.log",
            "logs/kubectl-cluster1.log",
            "logs/kubectl-cluster2.log",
            "logs/kubectl-cluster3.log",
            "logs/crp.log",
            "logs/actions-handler.log",
            "logs/controller.log",
        ]:
            (seed_dir / relative).unlink(missing_ok=True)

    controller_name = manifest["spec"]["controller"]["controller"]
    evaluation = evaluation_trial_metadata(manifest)
    if evaluation is not None:
        if matched_baseline_manifest is not None or seed != evaluation["seed"]:
            raise RuntimeError("evaluation trial cannot claim matched evidence or run another seed")
        validate_matched_runtime_settings()
        validate_threshold_evaluation_inputs()
        if not dry_run and (not evaluation.get("controller_image") or not evaluation.get("controller_image_digest")
                            or evaluation["controller_image"] != os.environ.get("HEADLIGHT_THRESHOLD_IMAGE")
                            or evaluation["controller_image_digest"] != os.environ.get("HEADLIGHT_THRESHOLD_IMAGE_ID")):
            raise RuntimeError("evaluation trial requires explicit, consistent Threshold image and digest")
        configured_policy = os.environ.get("HEADLIGHT_POLICY_CONFIG")
        if configured_policy and Path(configured_policy).resolve() != DEFAULT_POLICY_CONFIG.resolve():
            raise RuntimeError("evaluation trial requires the unchanged frozen policy configuration")
    if matched_baseline_manifest is not None and (
        controller_name not in {"headlight", "threshold"} or GATEWAY_PROTOCOL is None
    ):
        raise RuntimeError("matched comparison requires Headlight-v1 or Threshold and the shared gateway protocol")
    if matched_baseline_manifest is not None:
        validate_matched_runtime_settings()
    if not dry_run and evaluation is None and controller_name in PRIMARY_CONTROLLERS and seed in EVALUATION_SEEDS:
        if GATEWAY_PROTOCOL is not None:
            if matched_baseline_manifest is None:
                raise RuntimeError(
                    "research-v4-gateway is evaluation: evaluation seeds require a separately "
                    "qualified gateway capacity contract and freeze manifest, or explicit "
                    "matched HPA/KEDA baseline evidence"
                )
            validate_freeze_manifest()
            expected_bundle = (REPO_ROOT / "results/calibration/headlight-bundle").resolve()
            configured_bundle = os.environ.get("HEADLIGHT_MODEL_BUNDLE_DIR")
            if controller_name == "headlight" and configured_bundle and Path(configured_bundle).resolve() != expected_bundle:
                raise RuntimeError("matched v1 comparison requires the frozen Headlight-v1 model bundle")
            expected_policy = DEFAULT_POLICY_CONFIG.resolve()
            configured_policy = os.environ.get("HEADLIGHT_POLICY_CONFIG")
            if configured_policy and Path(configured_policy).resolve() != expected_policy:
                raise RuntimeError("matched v1 comparison requires the frozen v1 policy configuration")
            if trainticket_path() != "advanced-ticket" or trainticket_resource_profile("advanced-ticket") != "capacity-vm":
                raise RuntimeError("matched v1 comparison requires the advanced-ticket capacity-vm profile")
        else:
            validate_freeze_manifest()

    supported_matched_seeds = {101, 102, 103} if controller_name == "headlight" else set(EVALUATION_SEEDS)
    if matched_baseline_manifest is not None and not dry_run and seed not in supported_matched_seeds:
        raise RuntimeError(f"matched {controller_name} comparison only supports evaluation seeds {sorted(supported_matched_seeds)}")

    apply_protocol_metadata(manifest)
    if evaluation is not None:
        (seed_dir / "raw" / "evaluation_trial.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    capacity_contract = prepare_capacity_contract(
        manifest,
        dry_run,
        seed_dir,
        matched_baselines=matched_baseline_manifest is not None,
    )
    workload_hashes = _rendered_workload_hashes(manifest)
    if matched_baseline_manifest is not None:
        run_config = manifest["spec"]["run"]
        capacity = int(capacity_contract["selected_capacity"]) if capacity_contract else 4
        matched_comparison = validate_matched_baseline_pair(
            matched_baseline_manifest,
            controller=controller_name,
            workload_pattern=str(workload["pattern"]),
            seed=seed,
            capacity=capacity,
            workload=workload,
            measurement_seconds=measurement_seconds,
            warmup_seconds=warmup_seconds,
            sampling_interval_seconds=interval,
            sla=run_config["sla"],
            gateway_reference=GATEWAY_PROTOCOL["reference"],
            workload_manifest_hashes=workload_hashes,
        )
        contract_path = contract_path_from_env()
        matched_comparison["capacity_contract"] = {
            "path": str(contract_path),
            "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()
            if capacity_contract is not None else None,
            "gateway_protocol_matches": bool(
                capacity_contract is not None
                and capacity_contract.get("gateway_protocol") == GATEWAY_PROTOCOL["reference"]
            ),
            "topology": capacity_contract.get("topology") if capacity_contract is not None else None,
        }
        manifest["spec"]["run"]["matched_baseline_manifest"] = str(matched_baseline_manifest)
        (seed_dir / "raw" / "matched_baseline_comparison.json").write_text(
            json.dumps(matched_comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    topology_metadata = _resolved_topology(manifest, workload_hashes)
    (seed_dir / "raw" / "protocol.json").write_text(
        json.dumps({"protocol": protocol_reference(PROTOCOL), "spec": PROTOCOL["spec"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    if GATEWAY_PROTOCOL is not None:
        (seed_dir / "raw" / "gateway_protocol.json").write_text(
            json.dumps(GATEWAY_PROTOCOL, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (seed_dir / "raw" / "topology.json").write_text(json.dumps(topology_metadata, indent=2) + "\n", encoding="utf-8")
    (seed_dir / "raw" / "workload_manifest_hashes.json").write_text(
        json.dumps(workload_hashes, indent=2) + "\n",
        encoding="utf-8",
    )
    if capacity_contract is not None:
        (seed_dir / "raw" / "capacity_contract.json").write_text(
            json.dumps(capacity_contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    points = generate_shape(
        workload,
        seed,
        interval,
        warmup_seconds=warmup_seconds,
        duration_seconds=measurement_seconds,
        capacity_users=capacity if matched_baseline_manifest is not None else None,
        continuation_seconds=diagnostic_observation_tail_seconds,
    )
    write_shape_csv(seed_dir / "raw" / "load_profile.csv", points)
    if evaluation is not None and file_sha256(seed_dir / "raw/load_profile.csv") != evaluation["load_trace_sha256"]:
        raise RuntimeError("evaluation trial requires the registered load trace for its workload")
    if matched_baseline_manifest is not None and file_sha256(seed_dir / "raw" / "load_profile.csv") != matched_comparison["load_trace_sha256"]:
        raise RuntimeError("generated load trace differs from the validated matched baseline evidence")
    queries = export_queries(topology["namespace"], f"{interval}s")
    (seed_dir / "raw" / "prometheus_queries.json").write_text(
        json.dumps(queries, indent=2) + "\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / "resource_calculation.json").write_text(
        json.dumps(
            {
                "cpu_seconds": {
                    "method": "counter_delta_with_reset_handling",
                    "query": queries["cpu_seconds"],
                    "measurement_window": {"started_at_unix": None, "finished_at_unix": None},
                },
                "pod_seconds": {
                    "method": "stepwise_gauge_integral_exact_window",
                    "query": queries["pod_count"],
                    "measurement_window": {"started_at_unix": None, "finished_at_unix": None},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / REQUEST_EVENTS_FILE).write_text(
        "timestamp,method,endpoint,status,response_time_ms,exception\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / "request_events_measurement.csv").write_text(
        "timestamp,method,endpoint,status,response_time_ms,exception\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / "kubernetes_events.json").write_text("[]\n", encoding="utf-8")
    (seed_dir / "raw" / "crp_transitions.json").write_text("[]\n", encoding="utf-8")
    (seed_dir / "raw" / "actions_handler_batches.json").write_text("[]\n", encoding="utf-8")
    write_inference_timings(
        seed_dir / "raw" / "ai_inference_times.csv",
        "",
        manifest["spec"]["controller"]["controller"],
        manifest["spec"]["controller"].get("controller_variant", ""),
    )
    (seed_dir / "raw" / "model_checkpoints").mkdir(exist_ok=True)
    (seed_dir / "raw" / CONTROLLER_ACTIONS_FILE).write_text("", encoding="utf-8")
    (seed_dir / "raw" / "model_bundle_metadata.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "model_bundle_provenance.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "policy_config.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE).write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "ceiling_binding.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "hpa_controller_metadata.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "static_plan.json").write_text("{}\n", encoding="utf-8")
    (seed_dir / "raw" / "phase_timestamps.json").write_text(
        json.dumps(
            {
                "dry_run": dry_run,
                "planned": {
                    "warmup_seconds": warmup_seconds,
                    "measurement_seconds": measurement_seconds,
                    "cooldown_seconds": cooldown_seconds,
                    "diagnostic_observation_tail_seconds": diagnostic_observation_tail_seconds,
                    "observation_tail_mode": _observation_tail_mode(workload, diagnostic_observation_tail_seconds),
                    "scored_prediction_window_seconds": measurement_seconds,
                    "locust_run_seconds": locust_duration_seconds,
                    "load_phases": _measurement_load_phases(workload),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / "measurement_window.json").write_text(
        json.dumps(
            {
                "locust_started_at_unix": None,
                "started_at_unix": None,
                "finished_at_unix": None,
                "locust_finished_at_unix": None,
                "warmup_seconds": warmup_seconds,
                "measurement_seconds": measurement_seconds,
                "locust_run_seconds": locust_duration_seconds,
                "cooldown_seconds": cooldown_seconds,
                "diagnostic_observation_tail_seconds": diagnostic_observation_tail_seconds,
                "observation_tail_mode": _observation_tail_mode(workload, diagnostic_observation_tail_seconds),
                "scored_prediction_window_seconds": measurement_seconds,
                "load_phases": _measurement_load_phases(workload),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (seed_dir / "logs" / "runner.log").write_text(
        f"dry_run={dry_run} seed={seed} controller={manifest['spec']['controller']['controller']}\n",
        encoding="utf-8",
    )
    write_seed_summary(seed_dir / "summaries" / "summary.csv", seed, manifest,
                       None if dry_run else {"technical_validity": "running", "performance_outcome": "not_measured"})

    if not dry_run:
        phases: dict[str, dict[str, int]] = {}
        def mark_phase(name: str, event: str) -> None:
            phases.setdefault(name, {})[event] = int(time.time())
            if evidence:
                evidence.phase(name, event)

        mark_phase("preflight", "started_at_unix")
        validate_frozen_controller_bundle(manifest, seed_dir)
        ensure_contexts()
        mark_phase("preflight", "finished_at_unix")
        mark_phase("reset_and_deployment", "started_at_unix")
        reset_state(manifest, seed_dir)
        deploy_workload(manifest, seed_dir)
        mark_phase("reset_and_deployment", "finished_at_unix")
        mark_phase("readiness", "started_at_unix")
        try:
            install_controller(manifest, seed_dir, seed=seed)
            if evidence and controller_name in {"headlight", "lstm", "threshold"}:
                context = os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster3")
                controller_namespace = os.environ.get("HEADLIGHT_CONTROLLER_NAMESPACE", topology["namespace"])
                selector = {"headlight": "app=ai-module", "lstm": "app=lstm-autoscaler",
                            "threshold": "app=threshold-autoscaler"}[controller_name]
                since = datetime.fromtimestamp(evidence.started_at, timezone.utc).isoformat()
                evidence.action_stream = ControllerActionStream(seed_dir, [
                    "kubectl", "--context", context, "-n", controller_namespace, "logs",
                    "-l", selector, "--follow", "--prefix=true", "--timestamps=true", "--tail=-1",
                    "--max-log-requests=5", "--since-time=" + since,
                ])
                evidence.action_stream.start()
            if matched_baseline_manifest is not None:
                verify_matched_controller_active(manifest, seed_dir)
            elif evaluation is not None:
                verify_matched_controller_active(manifest, seed_dir, "controller_runtime.json")
        except BaseException as exc:
            if evidence:
                record_run_failure(manifest, seed, evidence, exc)
            cleanup_run_component(evidence, "controller_shutdown",
                                  lambda: stop_controller(manifest, seed_dir, check=False),
                                  suppress_errors=True)
            raise
        try:
            settle_seconds = int(os.environ.get("HEADLIGHT_PRE_MEASUREMENT_SETTLE_SECONDS", "15"))
            if settle_seconds:
                time.sleep(settle_seconds)
            mark_phase("readiness", "finished_at_unix")
            collect_workload_snapshot(manifest, seed_dir, "before")
        except BaseException as exc:
            if evidence:
                record_run_failure(manifest, seed, evidence, exc)
            cleanup_run_component(evidence, "controller_shutdown",
                                  lambda: stop_controller(manifest, seed_dir, check=False),
                                  suppress_errors=True)
            raise
        try:
            if GATEWAY_PROTOCOL is not None:
                run_cmd(
                    [sys.executable, str(REPO_ROOT / "experiments/scripts/gateway_weights.py"), "--once"],
                    seed_dir / "logs" / "gateway-weights-initial.log",
                )
                weight_log_path = seed_dir / "logs" / "gateway-weights.jsonl"
                with weight_log_path.open("w", encoding="utf-8") as weight_log:
                    watcher = subprocess.Popen(
                        [
                            sys.executable,
                            str(REPO_ROOT / "experiments/scripts/gateway_weights.py"),
                            "--interval",
                            str(GATEWAY_PROTOCOL["spec"]["update_interval_seconds"]),
                        ],
                        cwd=REPO_ROOT,
                        env=os.environ.copy(),
                        stdout=weight_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        locust_started_at = int(time.time())
                        if evidence:
                            evidence.locust_context = _locust_settings()[0]
                            evidence.begin_workload(locust_started_at)
                        max_staleness = float(os.environ.get("HEADLIGHT_GATEWAY_MAX_STALENESS_SECONDS", "60"))
                        watch = GatewayWatch(watcher, weight_log_path, max_staleness)
                        def check_workload_health() -> None:
                            if evidence:
                                evidence.check_measurement_phase()
                            watch.check()
                        run_locust(manifest, seed_dir, locust_duration_seconds, health_check=check_workload_health)
                        locust_finished_at = int(time.time())
                        if evidence:
                            evidence.finish_workload(locust_finished_at)
                        watcher_early_exit = watcher.poll()
                        watch.check()
                    finally:
                        stop_process(watcher)
                if watcher_early_exit is not None:
                    raise RuntimeError(f"gateway weight updater exited during workload: {watcher_early_exit}")
                since_time = datetime.fromtimestamp(locust_started_at, timezone.utc).isoformat()
                gateway_logs = run_cmd(
                    [
                        "kubectl", "--context", "cluster1", "-n", topology["namespace"],
                        "logs", "deployment/trainticket-gateway", f"--since-time={since_time}",
                    ],
                    seed_dir / "logs" / "gateway-log-collection.log",
                )
                (seed_dir / "raw" / "gateway_requests.log").write_text(gateway_logs.stdout, encoding="utf-8")
            else:
                locust_started_at = int(time.time())
                if evidence:
                    evidence.locust_context = _locust_settings()[0]
                    evidence.begin_workload(locust_started_at)
                run_locust(manifest, seed_dir, locust_duration_seconds)
                locust_finished_at = int(time.time())
                if evidence:
                    evidence.finish_workload(locust_finished_at)
            started_at = locust_started_at + warmup_seconds
            finished_at = min(locust_finished_at, started_at + measurement_seconds)
            phases["warmup"] = {"started_at_unix": locust_started_at, "finished_at_unix": started_at}
            phases["measurement"] = {"started_at_unix": started_at, "finished_at_unix": finished_at}
            if evidence:
                evidence.phases.update({"warmup": phases["warmup"], "measurement": phases["measurement"]})
                atomic_json(seed_dir / "raw/phase_timestamps.json", evidence.phases)
            if diagnostic_observation_tail_seconds:
                tail_phase = {
                    "started_at_unix": finished_at,
                    "finished_at_unix": locust_finished_at,
                    "mode": _observation_tail_mode(workload, diagnostic_observation_tail_seconds),
                }
                phases["diagnostic_observation_tail"] = tail_phase
                if evidence:
                    evidence.phases["diagnostic_observation_tail"] = tail_phase
                    atomic_json(seed_dir / "raw/phase_timestamps.json", evidence.phases)
            collect_workload_snapshot(manifest, seed_dir, "after")
            collect_kubernetes_state(manifest, seed_dir)
        finally:
            failure = sys.exc_info()[1]
            if failure is not None and evidence:
                record_run_failure(manifest, seed, evidence, failure)
            try:
                mark_phase("controller_shutdown", "started_at_unix")
                cleanup_run_component(
                    evidence, "controller_shutdown",
                    lambda: stop_controller(manifest, seed_dir, check=failure is None),
                    suppress_errors=failure is not None,
                )
                mark_phase("controller_shutdown", "finished_at_unix")
            finally:
                if evidence and evidence.action_stream:
                    cleanup_run_component(
                        evidence, "controller_stream", evidence.action_stream.stop,
                        suppress_errors=failure is not None or sys.exc_info()[1] is not None,
                    )
        mark_phase("cooldown", "started_at_unix")
        if cooldown_seconds:
            time.sleep(cooldown_seconds)
        mark_phase("cooldown", "finished_at_unix")
        mark_phase("collection_and_validation", "started_at_unix")
        (seed_dir / "raw" / "measurement_window.json").write_text(
            json.dumps(
                {
                    "locust_started_at_unix": locust_started_at,
                    "started_at_unix": started_at,
                    "finished_at_unix": finished_at,
                    "locust_finished_at_unix": locust_finished_at,
                    "warmup_seconds": warmup_seconds,
                    "measurement_seconds": measurement_seconds,
                    "locust_run_seconds": locust_duration_seconds,
                    "cooldown_seconds": cooldown_seconds,
                    "diagnostic_observation_tail_seconds": diagnostic_observation_tail_seconds,
                    "observation_tail_mode": _observation_tail_mode(workload, diagnostic_observation_tail_seconds),
                    "scored_prediction_window_end_unix": finished_at,
                    "observation_tail_start_unix": finished_at if diagnostic_observation_tail_seconds else None,
                    "observation_tail_end_unix": locust_finished_at if diagnostic_observation_tail_seconds else None,
                    "load_phases": _measurement_load_phases(workload, started_at),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_measurement_request_csv(
            seed_dir / "raw" / REQUEST_EVENTS_FILE,
            seed_dir / "raw" / "request_events_measurement.csv",
            started_at,
            finished_at,
        )
        verify_bundle_unchanged(manifest, seed_dir)
        collect_prometheus(manifest, seed_dir, started_at, finished_at, interval)
        queries = export_queries(topology["namespace"], f"{interval}s")
        (seed_dir / "raw" / "resource_calculation.json").write_text(
            json.dumps(
                {
                    "cpu_seconds": {
                        "method": "counter_delta_with_reset_handling",
                        "query": queries["cpu_seconds"],
                        "measurement_window": {
                            "started_at_unix": started_at,
                            "finished_at_unix": finished_at,
                        },
                    },
                    "pod_seconds": {
                        "method": "stepwise_gauge_integral_exact_window",
                        "query": queries["pod_count"],
                        "measurement_window": {
                            "started_at_unix": started_at,
                            "finished_at_unix": finished_at,
                        },
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_ceiling_binding_diagnostic(manifest, seed_dir, started_at, finished_at)
        metrics = compute_summary_metrics(
            seed_dir / "raw" / REQUEST_EVENTS_FILE,
            seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE,
            float(run["sla"]["p95_latency_ms"]),
            seed_dir / "raw" / CONTROLLER_ACTIONS_FILE,
            started_at=started_at,
            finished_at=finished_at,
        )
        ceiling = json.loads((seed_dir / "raw" / "ceiling_binding.json").read_text(encoding="utf-8"))
        metrics.update(
            {
                "technical_validity": "valid",
                "performance_outcome": performance_outcome(metrics, run, ceiling),
                "invalidity_reason": "",
                "retry_allowed": False,
                "retry_rule": "",
            }
        )
        write_seed_summary(seed_dir / "summaries" / "summary.csv", seed, manifest, metrics)
        mark_phase("collection_and_validation", "finished_at_unix")
        (seed_dir / "raw" / "phase_timestamps.json").write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
    if evidence and evidence.action_stream:
        cleanup_run_component(evidence, "controller_stream", evidence.action_stream.stop)
    validate_seed_bundle(seed_dir, dry_run, manifest)


def cleanup_run_component(
    evidence: RunEvidence | None, name: str, cleanup: Callable[[], None], *,
    suppress_errors: bool = False,
) -> None:
    """Record cleanup errors without replacing an existing experiment failure."""
    try:
        cleanup()
    except Exception as exc:
        if evidence:
            atomic_json(evidence.seed_dir / "raw" / ("cleanup-error-" + name + ".json"), {
                "error": f"{type(exc).__name__}: {exc}",
            })
        if not suppress_errors:
            raise


def record_run_failure(manifest: dict[str, Any], seed: int, evidence: RunEvidence, exc: BaseException) -> None:
    if evidence.status.get("technical_validity") != "invalid":
        evidence.fail(exc)
    write_seed_summary(evidence.seed_dir / "summaries/summary.csv", seed, manifest, {
        "technical_validity": "invalid", "performance_outcome": "not_measured",
        "invalidity_reason": evidence.status["invalidity_reason"], "retry_allowed": False,
    })
    try:
        queries = export_queries(manifest["spec"]["topology"]["namespace"],
                                 f"{manifest['spec']['run'].get('sampling_interval_seconds', 15)}s")
        collect_failed_attempt(manifest, evidence, queries)
    except Exception as collection_error:
        atomic_json(evidence.seed_dir / "raw/failure_collection_error.json", {
            "error": f"{type(collection_error).__name__}: {collection_error}",
        })


def run_one_seed(
    manifest: dict[str, Any], output_dir: Path, seed: int, dry_run: bool,
    matched_baseline_manifest: Path | None = None,
) -> None:
    if dry_run:
        _execute_seed(manifest, output_dir, seed, True, matched_baseline_manifest)
        return
    run = manifest["spec"]["run"]
    evidence = RunEvidence(output_dir / f"seed-{seed}",
                           int(run.get("warmup_seconds", manifest["spec"]["workload"].get("warmup_seconds", 0))),
                           int(run["duration_seconds"]),
                           int(run.get("diagnostic_observation_tail_seconds", 0)),
                           observation_tail_mode=_observation_tail_mode(
                               manifest["spec"]["workload"],
                               int(run.get("diagnostic_observation_tail_seconds", 0)),
                           ),
                           load_phases=_measurement_load_phases(manifest["spec"]["workload"]))
    try:
        evidence.start_host_recording()
        _execute_seed(manifest, output_dir, seed, False, matched_baseline_manifest, evidence)
        if evidence.action_stream:
            cleanup_run_component(evidence, "controller_stream", evidence.action_stream.stop)
        evidence.complete()
    except BaseException as exc:
        record_run_failure(manifest, seed, evidence, exc)
        raise
    finally:
        try:
            if evidence.action_stream:
                cleanup_run_component(evidence, "controller_stream", evidence.action_stream.stop,
                                      suppress_errors=sys.exc_info()[1] is not None)
        finally:
            evidence.stop_host_recording()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Headlight experiment")
    parser.add_argument("--run", required=True, help="repository-relative ExperimentRun YAML")
    parser.add_argument("--seeds", default="all", help="all or comma-separated integer seeds")
    parser.add_argument("--dry-run", action="store_true", help="create bundle without touching Kubernetes")
    parser.add_argument("--evaluation-trial", action="store_true",
                        help="run the seed104 Threshold gradual-ramp gateway observation at configured C=4, excluded from matched comparison")
    parser.add_argument("--evaluation-campaign-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-configured-static-amendment",
        action="store_true",
        help="explicitly allow execution of a static-matched manifest backed by a configured amended selection",
    )
    parser.add_argument(
        "--allow-configured-lstm-amendment",
        action="store_true",
        help="explicitly allow execution of an lstm-matched manifest backed by a configured amended selection",
    )
    parser.add_argument(
        "--matched-baseline-manifest",
        type=Path,
        help="require matching HPA and KEDA evidence for a Headlight-v1 or Threshold gateway evaluation",
    )
    args = parser.parse_args()
    if args.evaluation_campaign_cell and not args.evaluation_trial:
        parser.error("--evaluation-campaign-cell requires --evaluation-trial")

    manifest = resolve_run(Path(args.run))
    apply_protocol_metadata(manifest)
    run_spec = manifest["spec"]["run"]
    amendment_dir_value = run_spec.get("static_amendment_dir")
    lstm_amendment_dir_value = run_spec.get("lstm_amendment_dir")
    if amendment_dir_value and lstm_amendment_dir_value:
        raise ValueError("a run may use only one matched amendment")
    if amendment_dir_value:
        if manifest["spec"]["controller"]["controller"] != "static-matched":
            raise ValueError("static_amendment_dir is valid only for static-matched")
        from experiments.lib.matched_static_amendment import validate_static_amendment
        amendment_dir = (REPO_ROOT / amendment_dir_value).resolve()
        amendment = validate_static_amendment(amendment_dir)
        configured_protocol = (REPO_ROOT / run_spec["static_amendment_protocol"]).resolve()
        if configured_protocol != Path(amendment["amendment_config_path"]).resolve():
            raise ValueError("run manifest amendment protocol does not match its selection artifact")
        if not args.dry_run and not args.allow_configured_static_amendment:
            parser.error("this run uses a configured static amendment; pass --allow-configured-static-amendment to execute it")
    if lstm_amendment_dir_value:
        if manifest["spec"]["controller"]["controller"] != "lstm-matched":
            raise ValueError("lstm_amendment_dir is valid only for lstm-matched")
        from experiments.lib.matched_lstm_amendment import validate_lstm_amendment
        amendment_dir = (REPO_ROOT / lstm_amendment_dir_value).resolve()
        amendment = validate_lstm_amendment(amendment_dir)
        configured_protocol = (REPO_ROOT / run_spec["lstm_amendment_protocol"]).resolve()
        if configured_protocol != Path(amendment["amendment_config_path"]).resolve():
            raise ValueError("run manifest amendment protocol does not match its LSTM selection artifact")
        if not args.dry_run and not args.allow_configured_lstm_amendment:
            parser.error("this run uses a configured LSTM amendment; pass --allow-configured-lstm-amendment to execute it")
    if args.allow_configured_static_amendment and not amendment_dir_value:
        parser.error("--allow-configured-static-amendment requires a static amendment run manifest")
    if args.allow_configured_lstm_amendment and not lstm_amendment_dir_value:
        parser.error("--allow-configured-lstm-amendment requires an LSTM amendment run manifest")
    if amendment_dir_value or lstm_amendment_dir_value:
        configured_freeze = os.environ.get("HEADLIGHT_MATCHED_FREEZE_DIR")
        if configured_freeze and Path(configured_freeze).resolve() != amendment_dir:
            raise ValueError("HEADLIGHT_MATCHED_FREEZE_DIR differs from the manifest amendment directory")
        # Bundle preflight runs before install_controller. Bind the manifest's
        # artifact directory now so a clean VM needs no implicit prior export.
        os.environ["HEADLIGHT_MATCHED_FREEZE_DIR"] = str(amendment_dir)
    seeds = parse_seeds(args.seeds, manifest["spec"]["run"]["seeds"])
    output_dir_value = os.environ.get("HEADLIGHT_EXPERIMENT_OUTPUT_DIR", manifest["spec"]["run"]["output_dir"])
    output_dir_path = Path(output_dir_value)
    output_dir = output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path
    if args.evaluation_trial:
        configure_evaluation_trial(
            manifest, seeds, output_dir, args.matched_baseline_manifest,
            campaign_cell=args.evaluation_campaign_cell,
        )
        apply_protocol_metadata(manifest)
    if args.matched_baseline_manifest is not None:
        controller = manifest["spec"]["controller"]["controller"]
        workload_pattern = manifest["spec"]["workload"]["pattern"]
        if controller not in {"headlight", "threshold"}:
            raise ValueError("--matched-baseline-manifest is only supported for Headlight-v1 or Threshold")
        if workload_pattern not in {"gradual-ramp", "flash-crowd", "diurnal"}:
            raise ValueError("--matched-baseline-manifest requires a registered paper workload")
        if not GATEWAY_PROTOCOL:
            raise ValueError("--matched-baseline-manifest requires HEADLIGHT_GATEWAY_PROTOCOL_PATH")
        validate_matched_runtime_settings()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("matched comparison output directory must be new and empty")
        args.matched_baseline_manifest = args.matched_baseline_manifest.resolve()
        if not args.matched_baseline_manifest.is_file():
            raise FileNotFoundError(f"matched baseline source manifest is absent: {args.matched_baseline_manifest}")
        supported_seeds = {101, 102, 103} if controller == "headlight" else set(EVALUATION_SEEDS)
        if not seeds or any(seed not in supported_seeds for seed in seeds):
            permitted = "101, 102, and 103" if controller == "headlight" else "101 through 105"
            raise ValueError(f"matched v1 comparison supports only evaluation seeds {permitted}")
        if len(seeds) != len(set(seeds)):
            raise ValueError("matched v1 comparison seed list must not contain duplicates")
        manifest["spec"]["run"]["matched_baseline_manifest"] = str(args.matched_baseline_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["logs", "raw", "summaries", "figures"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    try:
        manifest["spec"]["run"]["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        manifest["spec"]["run"]["output_dir"] = str(output_dir)
    write_yaml(output_dir / "manifest.yaml", manifest)
    (output_dir / "environment.json").write_text(json.dumps(environment(args.dry_run), indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        run_one_seed(manifest, output_dir, seed, args.dry_run, args.matched_baseline_manifest)
    try:
        display_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        display_output = output_dir
    print(f"wrote result bundle: {display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
