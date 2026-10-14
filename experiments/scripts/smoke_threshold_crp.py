#!/usr/bin/env python3
"""Isolated Threshold deployment/CRP smoke using synthetic metrics, never evaluation traffic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.lib.config import REPO_ROOT

CRP = "controlledreplicapropagations.policy.nephele-project.eu"


def confirm_actuation(actions: list[dict], crp: dict, deployments: dict) -> bool:
    desired = {"cluster2": 1, "cluster3": 2}
    action_seen = any(
        row.get("controller") == "threshold" and row.get("service") == "ts-auth"
        and row.get("policy_reason") == "threshold-scale-up"
        and row.get("previous_replicas") == {"cluster2": 1, "cluster3": 1}
        and row.get("desired_replicas") == desired
        and row.get("batch_attempted") is True and row.get("batch_successful") is True
        and row.get("batch_status_code") == 202 for row in actions
    )
    placements = {item["clusterName"]: item["replicas"]
                  for item in crp.get("spec", {}).get("placements", [])}
    return action_seen and placements == desired and all(
        deployments.get(cluster, {}).get("spec", {}).get("replicas") == replicas
        and deployments.get(cluster, {}).get("status", {}).get("readyReplicas", 0) == replicas
        for cluster, replicas in desired.items()
    )


METRICS_FIXTURE = '''import json, time
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"status":"success", "data":{"resultType":"vector", "result":[
            {"metric":{"destination_workload":"ts-auth"}, "value":[time.time(), "0.95"]}]}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
HTTPServer(("0.0.0.0", 9090), Handler).serve_forever()
'''


def execute_smoke(image: str, output: Path, timeout: int) -> dict:
    if output.exists():
        raise ValueError(f"smoke output must be new: {output}")
    output.mkdir(parents=True, mode=0o700)
    namespace = f"threshold-smoke-{time.time_ns()}"
    env = dict(os.environ)
    env["KUBECONFIG"] = env.get("HEADLIGHT_KUBECONFIG", str(REPO_ROOT / ".kube/headlight-testbed.config"))

    def run(command: list[str], *, input_text: str | None = None, check: bool = True) -> str:
        result = subprocess.run(command, input=input_text, text=True, capture_output=True,
                                env=env, timeout=timeout, cwd=REPO_ROOT)
        with (output / "commands.log").open("a") as handle:
            # Commands contain paths and resource identities, never kubeconfig contents.
            handle.write(json.dumps(command) + "\n" + result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {command}: {result.stderr[-1500:]}")
        return result.stdout

    def kubectl(context: str, *args: str, **kwargs) -> str:
        return run(["kubectl", "--context", context, "--request-timeout=20s", *args], **kwargs)

    def apply(context: str, value: dict) -> None:
        kubectl(context, "apply", "-f", "-", input_text=json.dumps(value))

    def resource(context: str, kind: str, name: str) -> dict:
        return json.loads(kubectl(context, "-n", namespace, "get", kind, name, "-o", "json"))

    def save(name: str, value) -> None:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    report = {"schema_version": "headlight.threshold-crp-smoke/v1", "namespace": namespace,
              "image": image, "synthetic_metrics": True, "evaluation_run": False, "passed": False}
    contexts = ("karmada-apiserver", "cluster2", "cluster3")
    try:
        for context in contexts:
            apply(context, {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}})
        deployment = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "ts-auth", "namespace": namespace},
                      "spec": {"replicas": 2, "selector": {"matchLabels": {"app": "ts-auth"}},
                               "template": {"metadata": {"labels": {"app": "ts-auth"},
                                                         "annotations": {"sidecar.istio.io/inject": "false"}},
                                            "spec": {"containers": [{"name": "pause", "image": "registry.k8s.io/pause:3.10",
                                                                      "resources": {"requests": {"cpu": "10m", "memory": "16Mi"},
                                                                                    "limits": {"cpu": "100m", "memory": "32Mi"}}}]}}}}
        apply("karmada-apiserver", deployment)
        apply("karmada-apiserver", {
            "apiVersion": "policy.nephele-project.eu/v1alpha1", "kind": "ControlledReplicaPropagation",
            "metadata": {"name": "ts-auth", "namespace": namespace},
            "spec": {"targetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "ts-auth"},
                     "placements": [{"clusterName": cluster, "replicas": 1} for cluster in ("cluster2", "cluster3")],
                     "strategy": {"maxUnavailable": 0}},
        })
        deadline = time.monotonic() + timeout
        while True:
            ready = True
            for cluster in ("cluster2", "cluster3"):
                try:
                    member = resource(cluster, "deployment", "ts-auth")
                    ready &= member.get("spec", {}).get("replicas") == 1 and member.get("status", {}).get("readyReplicas") == 1
                except (RuntimeError, json.JSONDecodeError):
                    ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("initial CRP placements did not become ready")
            time.sleep(2)
        save("crp-before.json", resource("karmada-apiserver", CRP, "ts-auth"))
        for cluster in ("cluster2", "cluster3"):
            save(f"{cluster}-before.json", resource(cluster, "deployment", "ts-auth"))

        apply("cluster3", {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "threshold-metrics", "namespace": namespace},
                           "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "threshold-metrics"}},
                                    "template": {"metadata": {"labels": {"app": "threshold-metrics"}},
                                                 "spec": {"containers": [{"name": "metrics", "image": image,
                                                                           "imagePullPolicy": "IfNotPresent",
                                                                           "command": ["python3", "-u", "-c", METRICS_FIXTURE],
                                                                           "resources": {"requests": {"cpu": "10m", "memory": "32Mi"},
                                                                                         "limits": {"cpu": "100m", "memory": "128Mi"}}}]}}}})
        apply("cluster3", {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "threshold-metrics", "namespace": namespace},
                           "spec": {"selector": {"app": "threshold-metrics"}, "ports": [{"port": 9090, "targetPort": 9090}]}})
        kubectl("cluster3", "-n", namespace, "rollout", "status", "deployment/threshold-metrics", f"--timeout={timeout}s")
        profile = json.loads((REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json").read_text())
        profile.pop("profile_hash")
        profile["namespace"] = namespace
        profile["profile_hash"] = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        profile_path = output / "smoke-profile.json"
        save(profile_path.name, profile)
        env.update({"HEADLIGHT_TRAINTICKET_NAMESPACE": namespace, "HEADLIGHT_THRESHOLD_IMAGE": image,
                    "HEADLIGHT_SERVICE_PROFILE_FILE": str(profile_path), "HEADLIGHT_SHADOW_MODE": "0",
                    "HEADLIGHT_THRESHOLD_RENDERED_MANIFEST": str(output / "threshold-deployment.yaml"),
                    "HEADLIGHT_PROMETHEUS_URL": f"http://threshold-metrics.{namespace}.svc.cluster.local:9090",
                    "HEADLIGHT_CONTROLLER_CONTEXT": "cluster3", "HEADLIGHT_CONTROLLER_WAIT_TIMEOUT": f"{timeout}s"})
        run(["bash", "ai-module/threshold_controller_setup.sh"])
        save("controller-deployment.json", resource("cluster3", "deployment", "threshold-autoscaler"))
        save("controller-pods.json", json.loads(kubectl("cluster3", "-n", namespace, "get", "pods", "-l", "app=threshold-autoscaler", "-o", "json")))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = kubectl("cluster3", "-n", namespace, "exec", "deployment/threshold-autoscaler", "--", "cat", "/artifacts/actions.jsonl", check=False)
            actions = [json.loads(line) for line in data.splitlines() if line.strip()]
            crp = resource("karmada-apiserver", CRP, "ts-auth")
            members = {cluster: resource(cluster, "deployment", "ts-auth") for cluster in ("cluster2", "cluster3")}
            (output / "controller-actions.jsonl").write_text(data)
            save("crp-after.json", crp)
            save("members-after.json", members)
            if confirm_actuation(actions, crp, members):
                report["passed"] = True
                report["placements_before"] = {"cluster2": 1, "cluster3": 1}
                report["placements_after"] = {"cluster2": 1, "cluster3": 2}
                break
            time.sleep(2)
        if not report["passed"]:
            raise RuntimeError("Threshold request, CRP placements, and ready member replicas did not all confirm actuation")
        return report
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        # Always attempt every cleanup step, even if collecting evidence fails.
        cleanup_errors = []
        try:
            (output / "controller.log").write_text(kubectl("cluster3", "-n", namespace, "logs", "deployment/threshold-autoscaler", check=False))
        except Exception as exc:
            report["log_collection_error"] = str(exc)
        try:
            kubectl("cluster3", "-n", namespace, "delete", "deployment", "threshold-autoscaler",
                    "--ignore-not-found", "--cascade=foreground", "--wait=true", "--timeout=45s")
        except Exception as exc:
            cleanup_errors.append(f"stop controller: {exc}")
        for context in contexts:
            try:
                kubectl(context, "delete", "namespace", namespace, "--ignore-not-found", "--wait=true", "--timeout=45s")
            except Exception as exc:
                cleanup_errors.append(f"{context}: {exc}")
        report["cleanup_errors"] = cleanup_errors
        save("report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--execute", action="store_true", help="create temporary resources and test real CRP actuation")
    args = parser.parse_args()
    if not args.execute:
        print("Plan only: use --execute to deploy an isolated metric fixture and verify Threshold -> Actions Handler -> CRP -> ready replicas.")
        return 0
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    report = execute_smoke(args.image, args.output_dir.resolve(), args.timeout_seconds)
    if report["cleanup_errors"]:
        raise RuntimeError(f"smoke resources need cleanup: {report['cleanup_errors']}")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
