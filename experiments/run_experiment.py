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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT, resolve_run, write_yaml
from experiments.lib.metrics import compute_summary_metrics
from experiments.lib.prometheus import export_queries
from experiments.lib.quality import validate_bundle
from experiments.lib.workloads import generate_shape, write_shape_csv


REQUIRED_CONTEXTS = ["karmada-apiserver", "karmada-host", "cluster1", "cluster2", "cluster3"]
DEFAULT_TRAINTICKET_PATH = "advanced-ticket"
RESOURCE_PROFILE_BY_TRAINTICKET_PATH = {
    "login-route": "local-tiny",
    "advanced-ticket": "capacity-vm",
    "full": "source",
}
REQUEST_EVENTS_FILE = "request_events.csv"
PROMETHEUS_RANGE_DATA_FILE = "prometheus_range_data.json"
CONTROLLER_ACTIONS_FILE = "controller_actions.jsonl"


os.environ.setdefault(
    "KUBECONFIG",
    os.environ.get("HEADLIGHT_KUBECONFIG", str(REPO_ROOT / ".kube" / "headlight-testbed.config")),
)


def parse_seeds(value: str, configured: list[int]) -> list[int]:
    if value == "all":
        return configured
    return [int(part) for part in value.split(",") if part]


def trainticket_path() -> str:
    return os.environ.get("HEADLIGHT_TRAINTICKET_PATH", DEFAULT_TRAINTICKET_PATH)


def trainticket_resource_profile(path: str) -> str:
    return os.environ.get(
        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE",
        RESOURCE_PROFILE_BY_TRAINTICKET_PATH.get(path, "capacity-vm"),
    )


def tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(dry_run: bool) -> dict[str, Any]:
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    contexts: list[str] = []
    try:
        contexts = subprocess.check_output(
            ["kubectl", "config", "get-contexts", "-o", "name"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except Exception:
        contexts = []
    return {
        "captured_at_unix": int(time.time()),
        "repo_root": str(REPO_ROOT),
        "git_commit": commit,
        "source_tree_sha256": None if commit else tree_sha256(list((REPO_ROOT / "experiments").rglob("*"))),
        "dry_run": dry_run,
        "required_contexts": REQUIRED_CONTEXTS,
        "available_contexts": contexts,
        "image_tags": {
            "ai_module": os.environ.get("HEADLIGHT_AI_IMAGE", "local/headlight-ai:dev"),
            "actions_handler": os.environ.get("HEADLIGHT_ACTIONS_IMAGE", "local/actions-handler:dev"),
            "locust": os.environ.get("HEADLIGHT_LOCUST_IMAGE", "local/headlight-locust:dev"),
        },
    }


def ensure_contexts() -> None:
    contexts = subprocess.check_output(["kubectl", "config", "get-contexts", "-o", "name"], text=True).splitlines()
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
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(args)}\n")
            handle.flush()
            try:
                process = subprocess.Popen(
                    args,
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=merged_env,
                    bufsize=1,
                )
            except Exception as exc:
                handle.write(f"[exec error {exc}]\n")
                raise
            stdout_parts: list[str] = []
            if process.stdout:
                for chunk in process.stdout:
                    stdout_parts.append(chunk)
                    handle.write(chunk)
                    handle.flush()
            returncode = process.wait()
            stdout = "".join(stdout_parts)
            if stdout and not stdout.endswith("\n"):
                handle.write("\n")
            handle.write(f"[exit {returncode}]\n")
            completed = subprocess.CompletedProcess(args, returncode, stdout=stdout)
    else:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=merged_env,
            check=False,
        )
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
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sla_ms = manifest["spec"]["run"]["sla"]["p95_latency_ms"]
    candidate = manifest["spec"]["run"].get("qualification_overrides", {}).get("headlight_candidate", {}) or {}
    values = metrics or {
        "sla_violation_rate": 0.0,
        "excess_response_time_seconds": 0.0,
        "cpu_seconds": 0.0,
        "mean_pod_count": 0.0,
        "pod_seconds": 0.0,
        "p95_latency_ms": 0.0,
        "action_churn": 0.0,
    }
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
                "sla_violation_rate",
                "excess_response_time_seconds",
                "cpu_seconds",
                "mean_pod_count",
                "pod_seconds",
                "p95_latency_ms",
                "action_churn",
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
                "ai-module", "lstm-autoscaler", "kubernetes-autoscaler-eval", "--ignore-not-found",
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
            "HEADLIGHT_TRAINTICKET_DEPLOY_MODE": deploy_mode,
            "HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS": " ".join(topology["active_clusters"]),
            "HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST": str(rendered),
            "HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS": os.environ.get(
                "HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS",
                "",
            ),
        },
    )


def install_controller(manifest: dict[str, Any], seed_dir: Path) -> None:
    setup = REPO_ROOT / manifest["spec"]["controller"]["setup_script"]
    topology = manifest["spec"]["topology"]
    env = {
        "HOME": str(REPO_ROOT),
        "HEADLIGHT_SKIP_SMO": os.environ.get("HEADLIGHT_SKIP_SMO", "1"),
        "HEADLIGHT_SKIP_LEGACY_LOCUST": os.environ.get("HEADLIGHT_SKIP_LEGACY_LOCUST", "1"),
        "HEADLIGHT_TRAINTICKET_NAMESPACE": topology["namespace"],
        "HEADLIGHT_TRAINTICKET_PATH": trainticket_path(),
        "HEADLIGHT_HPA_CONTEXTS": " ".join(topology["active_clusters"]),
        "HEADLIGHT_KEDA_CONTEXT": topology["active_clusters"][0],
        "HEADLIGHT_SKIP_CONTROLLER_WORKLOAD_DEPLOY": "1",
        "HEADLIGHT_MATCH_TOPOLOGY_CAPACITY": "1",
    }
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
    if manifest["spec"]["controller"]["controller"] == "keda":
        for cluster in topology["active_clusters"]:
            cluster_max = {"cluster2": "3", "cluster3": "5"}.get(cluster, "4")
            run_cmd(
                ["bash", str(setup)],
                seed_dir / "logs" / "controller-setup.log",
                env={**env, "HEADLIGHT_KEDA_CONTEXT": cluster, "HEADLIGHT_KEDA_MAX_REPLICAS": cluster_max},
                cwd=setup.parent,
            )
    else:
        run_cmd(["bash", str(setup)], seed_dir / "logs" / "controller-setup.log", env=env, cwd=setup.parent)


def validate_frozen_controller_bundle(manifest: dict[str, Any], seed_dir: Path) -> None:
    controller = manifest["spec"]["controller"]["controller"]
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
        ],
        seed_dir / "logs" / "bundle-preflight.log",
    )


def run_locust(manifest: dict[str, Any], seed_dir: Path, duration_seconds: int) -> None:
    namespace = manifest["spec"]["topology"]["namespace"]
    cluster = os.environ.get("HEADLIGHT_LOCUST_CLUSTER", manifest["spec"]["topology"]["active_clusters"][0])
    image = os.environ.get("HEADLIGHT_LOCUST_IMAGE", "dstefanov46/locust-test:0.66.0")
    spawn_rate = str(manifest["spec"]["workload"]["locust"]["spawn_rate"])
    path = trainticket_path()
    mode = os.environ.get(
        "HEADLIGHT_TRAINTICKET_LOCUST_MODE",
        "full-session" if path == "full" else path,
    )
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
                            "env": [
                                {"name": "HEADLIGHT_REQUEST_CSV", "value": "/app/headlight_requests.csv"},
                                {"name": "HEADLIGHT_LOAD_SHAPE", "value": "/app/load_profile.csv"},
                                {"name": "HEADLIGHT_TRAINTICKET_MODE", "value": mode},
                                {"name": "HEADLIGHT_TRAINTICKET_NAMESPACE", "value": namespace},
                            ],
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


def collect_prometheus(manifest: dict[str, Any], seed_dir: Path, started_at: int, finished_at: int, interval: int) -> None:
    cluster = os.environ.get("HEADLIGHT_PROMETHEUS_CLUSTER", manifest["spec"]["topology"]["active_clusters"][0])
    namespace = manifest["spec"]["topology"]["namespace"]
    prometheus_namespace = os.environ.get("HEADLIGHT_PROMETHEUS_NAMESPACE", "monitoring")
    prometheus_service = os.environ.get("HEADLIGHT_PROMETHEUS_SERVICE", "prometheus-kube-prometheus-prometheus")
    prometheus_port = int(os.environ.get("HEADLIGHT_PROMETHEUS_PORT", "39090"))
    log_path = seed_dir / "logs" / "prometheus.log"
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
        stdout=(seed_dir / "logs" / "prometheus-port-forward.log").open("w", encoding="utf-8"),
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
            raise RuntimeError("Prometheus port-forward did not become ready")
        payload: dict[str, Any] = {}
        nonempty = False
        for name, query in export_queries(namespace, f"{interval}s").items():
            response = prometheus_query_range(prometheus_port, query, started_at, finished_at, f"{interval}s")
            payload[name] = {"query": query, "response": response}
            nonempty = nonempty or bool(response.get("data", {}).get("result"))
        (seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not nonempty:
            raise RuntimeError("Prometheus returned no time series")
    finally:
        port_forward.send_signal(signal.SIGINT)
        try:
            port_forward.wait(timeout=10)
        except subprocess.TimeoutExpired:
            port_forward.kill()
            port_forward.wait(timeout=10)
        log_path.write_text(f"queried {prometheus_service}.{prometheus_namespace} on {cluster}\n", encoding="utf-8")


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
    if controller not in {"headlight", "lstm"}:
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
    controller_context = os.environ.get("HEADLIGHT_CONTROLLER_CONTEXT", "cluster1")
    controller_namespace = os.environ.get("HEADLIGHT_CONTROLLER_NAMESPACE", namespace)
    selector = "app=ai-module" if controller == "headlight" else f"app={controller}-autoscaler"
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
    deployment_name = "ai-module" if controller == "headlight" else "lstm-autoscaler"
    if controller in {"headlight", "lstm"}:
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
                "cat /artifacts/actions.jsonl 2>/dev/null || true",
            ],
            seed_dir / "logs" / "controller-actions.log",
            check=False,
        ).stdout
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
        (seed_dir / "raw" / CONTROLLER_ACTIONS_FILE).write_text(action_log, encoding="utf-8")
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


def run_one_seed(manifest: dict[str, Any], output_dir: Path, seed: int, dry_run: bool) -> None:
    run = manifest["spec"]["run"]
    topology = manifest["spec"]["topology"]
    workload = manifest["spec"]["workload"]
    interval = int(run.get("sampling_interval_seconds", workload.get("sampling_interval_seconds", 15)))
    seed_dir = output_dir / f"seed-{seed}"
    for subdir in ["logs", "raw", "summaries", "figures"]:
        (seed_dir / subdir).mkdir(parents=True, exist_ok=True)
    if not dry_run:
        for relative in [
            "logs/reset.log",
            "logs/deploy-trainticket.log",
            "logs/controller-setup.log",
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

    points = generate_shape(workload, seed, interval)
    write_shape_csv(seed_dir / "raw" / "load_profile.csv", points)
    (seed_dir / "raw" / "prometheus_queries.json").write_text(
        json.dumps(export_queries(topology["namespace"], f"{interval}s"), indent=2) + "\n",
        encoding="utf-8",
    )
    (seed_dir / "raw" / REQUEST_EVENTS_FILE).write_text(
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
    (seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE).write_text("{}\n", encoding="utf-8")
    (seed_dir / "logs" / "runner.log").write_text(
        f"dry_run={dry_run} seed={seed} controller={manifest['spec']['controller']['controller']}\n",
        encoding="utf-8",
    )
    write_seed_summary(seed_dir / "summaries" / "summary.csv", seed, manifest)

    if not dry_run:
        validate_frozen_controller_bundle(manifest, seed_dir)
        ensure_contexts()
        reset_state(manifest, seed_dir)
        deploy_workload(manifest, seed_dir)
        install_controller(manifest, seed_dir)
        settle_seconds = int(os.environ.get("HEADLIGHT_PRE_MEASUREMENT_SETTLE_SECONDS", "15"))
        if settle_seconds:
            time.sleep(settle_seconds)
        collect_workload_snapshot(manifest, seed_dir, "before")
        started_at = int(time.time())
        run_locust(manifest, seed_dir, int(run["duration_seconds"]))
        finished_at = int(time.time())
        collect_workload_snapshot(manifest, seed_dir, "after")
        (seed_dir / "raw" / "measurement_window.json").write_text(
            json.dumps({"started_at_unix": started_at, "finished_at_unix": finished_at}, indent=2) + "\n",
            encoding="utf-8",
        )
        collect_kubernetes_state(manifest, seed_dir)
        collect_prometheus(manifest, seed_dir, started_at, finished_at, interval)
        metrics = compute_summary_metrics(
            seed_dir / "raw" / REQUEST_EVENTS_FILE,
            seed_dir / "raw" / PROMETHEUS_RANGE_DATA_FILE,
            float(run["sla"]["p95_latency_ms"]),
            seed_dir / "raw" / CONTROLLER_ACTIONS_FILE,
        )
        write_seed_summary(seed_dir / "summaries" / "summary.csv", seed, manifest, metrics)
    validate_seed_bundle(seed_dir, dry_run, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Headlight experiment")
    parser.add_argument("--run", required=True, help="repository-relative ExperimentRun YAML")
    parser.add_argument("--seeds", default="all", help="all or comma-separated integer seeds")
    parser.add_argument("--dry-run", action="store_true", help="create bundle without touching Kubernetes")
    args = parser.parse_args()

    manifest = resolve_run(Path(args.run))
    seeds = parse_seeds(args.seeds, manifest["spec"]["run"]["seeds"])
    output_dir_value = os.environ.get("HEADLIGHT_EXPERIMENT_OUTPUT_DIR", manifest["spec"]["run"]["output_dir"])
    output_dir_path = Path(output_dir_value)
    output_dir = output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path
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
        run_one_seed(manifest, output_dir, seed, args.dry_run)
    try:
        display_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        display_output = output_dir
    print(f"wrote result bundle: {display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
