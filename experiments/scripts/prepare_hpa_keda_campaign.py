#!/usr/bin/env python3
"""Discover the rebuilt testbed and prepare a separate HPA/KEDA extension."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.baseline_extension import audit_baselines, baseline_environment
from experiments.scripts import run_seed102_103_campaign as campaign


def check_prometheus_ready(endpoint: str) -> None:
    with urllib.request.urlopen(endpoint + "/-/ready", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Prometheus is not ready: {endpoint}")


def prepare_environment(root: Path) -> None:
    root = campaign.validate_extension_root(root)
    root.mkdir(parents=True, exist_ok=True)
    campaign.LEGACY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (root / "campaign.lock", campaign.LEGACY_STATE_DIR / "campaign.lock"):
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another campaign process holds the testbed lock") from exc
        _prepare_environment(root)


def _prepare_environment(root: Path) -> None:
    environment = baseline_environment(root)
    os.environ["HEADLIGHT_TRAINTICKET_MONGO_STORAGE"] = "image-defined"
    audit = audit_baselines()
    for name in ("HEADLIGHT_KUBECONFIG", "HEADLIGHT_INTERNAL_KUBECONFIG"):
        if not Path(environment[name]).is_file():
            raise ValueError(f"missing current kubeconfig: {environment[name]}")

    def kubectl(context: str, *args: str) -> str:
        result = subprocess.run([
            "kubectl", "--kubeconfig", environment["HEADLIGHT_KUBECONFIG"],
            "--context", context, "--request-timeout=20s", *args,
        ], cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=30)
        return result.stdout

    def verify_gateway() -> None:
        configmap = json.loads(kubectl(
            "cluster1", "-n", "trainticket", "get", "configmap",
            "trainticket-gateway-config", "-o", "json",
        ))
        config = configmap.get("data", {}).get("haproxy.cfg", "")
        for context, service_port in (("cluster2", 80), ("cluster3", 80)):
            node = json.loads(kubectl(context, "get", "nodes", "-o", "json"))["items"][0]
            address = next(a["address"] for a in node["status"]["addresses"]
                           if a["type"] == "InternalIP")
            service = json.loads(kubectl(
                context, "-n", "istio-system", "get", "service",
                "istio-ingressgateway", "-o", "json",
            ))
            ports = service["spec"]["ports"]
            http_port = next(p["nodePort"] for p in ports if p["port"] == service_port)
            status_port = next(p["nodePort"] for p in ports if p["port"] == 15021)
            expected = f"server {context} {address}:{http_port} check port {status_port} weight 0"
            if config.count(expected) < 3:
                raise RuntimeError(
                    f"gateway config is stale for {context}; refresh it with "
                    "experiments/scripts/deploy_trainticket_gateway.sh"
                )

        result = subprocess.run([
            "kubectl", "--kubeconfig", environment["HEADLIGHT_KUBECONFIG"],
            "--context", "cluster1", "-n", "trainticket", "exec", "-i",
            "deployment/trainticket-gateway", "-c", "haproxy", "--", "nc",
            "-w", "2", "127.0.0.1", "9999",
        ], cwd=REPO_ROOT, input="show stat\n", check=False,
            capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(f"cannot inspect HAProxy gateway health: {result.stderr.strip()}")
        rows = csv.DictReader(io.StringIO(result.stdout))
        backends = {"auth", "route", "travel"}
        members = {"cluster2", "cluster3"}
        states = {
            (row["# pxname"], row["svname"]): row
            for row in rows
            if row.get("# pxname") in backends and row.get("svname") in members
        }
        expected_servers = {(backend, member) for backend in backends for member in members}
        down = sorted(
            f"{backend}/{member}={states.get((backend, member), {}).get('status', 'missing')}"
            for backend, member in expected_servers
            if states.get((backend, member), {}).get("status") != "UP"
        )
        if down:
            raise RuntimeError(
                "shared gateway backends are not healthy (" + ", ".join(down) + "); "
                "refresh with experiments/scripts/deploy_trainticket_gateway.sh and "
                "qualify with experiments/scripts/smoke_trainticket_gateway.sh"
            )

    for context in ("karmada-host", "karmada-apiserver", "cluster1", "cluster2", "cluster3"):
        kubectl(context, "get", "--raw=/readyz")
    verify_gateway()

    urls = []
    node_provenance = {}
    for context in ("cluster2", "cluster3"):
        service_name = "prometheus-kube-prometheus-prometheus"
        service = json.loads(kubectl(context, "-n", "monitoring", "get", "service", service_name, "-o", "json"))
        if service["spec"].get("type") != "NodePort":
            kubectl(context, "-n", "monitoring", "patch", "service", service_name,
                    "--type=merge", "-p", '{"spec":{"type":"NodePort"}}')
            service = json.loads(kubectl(context, "-n", "monitoring", "get", "service", service_name, "-o", "json"))
        node = json.loads(kubectl(context, "get", "nodes", "-o", "json"))["items"][0]
        ip = next(a["address"] for a in node["status"]["addresses"] if a["type"] == "InternalIP")
        port = next(p["nodePort"] for p in service["spec"]["ports"] if p["port"] == 9090)
        endpoint = f"http://{ip}:{port}"
        check_prometheus_ready(endpoint)
        urls.append(endpoint)
        node_provenance[context] = {"address": ip, "node_info": node["status"].get("nodeInfo"),
                                    "capacity": node["status"].get("capacity"),
                                    "allocatable": node["status"].get("allocatable")}
    environment["HEADLIGHT_PREFLIGHT_PROMETHEUS_URL"] = ",".join(urls)
    environment["HEADLIGHT_CONTROLLER_PROMETHEUS_URL"] = ",".join(urls)
    docker = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
                                      capture_output=True, text=True, check=True, timeout=20).stdout)
    docker_root = Path(docker["DockerRootDir"])
    filesystem = subprocess.run(["findmnt", "-T", str(docker_root), "-n", "-o", "FSTYPE"],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    provenance = {
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_count": os.cpu_count(),
        "memory_total_kib": next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")),
        "docker": {key: docker.get(key) for key in ("DockerRootDir", "Driver", "ServerVersion", "CgroupVersion", "NCPU", "MemTotal")},
        "docker_filesystem": filesystem,
        "nodes": node_provenance, "environment": environment, "baseline_audit": audit,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime").mkdir(exist_ok=True)
    snapshot = root / "environment-snapshots" / f"preparation-{time.time_ns()}.json"
    provenance["snapshot_path"] = str(snapshot.relative_to(REPO_ROOT))
    campaign.write_json(snapshot, provenance)
    campaign.write_json(root / "environment-provenance.json", provenance)
    env_file = root / "campaign-env.sh"
    # Create privately before writing any settings.
    descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(env_file, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("#!/usr/bin/env bash\n# Regenerate after a testbed rebuild.\n")
        for name, value in sorted(environment.items()):
            handle.write(f"export {name}={shlex.quote(value)}\n")
    print(f"CAMPAIGN_ENV_READY {env_file}\nPrometheus: {','.join(urls)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_environment(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
