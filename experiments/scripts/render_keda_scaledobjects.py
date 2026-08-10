#!/usr/bin/env python3
"""Render KEDA ScaledObjects for TrainTicket application Deployments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.render_minimal_trainticket import render as render_trainticket_slice
from experiments.lib.paths import resolve_output_path, resolve_repo_path

APP_MANIFEST = REPO_ROOT / "ai-module" / "deployments" / "application-services-deployment.yaml"


def load_yaml(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def app_deployments(source: Path) -> list[str]:
    names: list[str] = []
    for doc in load_yaml(source):
        if doc.get("kind") == "Deployment":
            name = doc.get("metadata", {}).get("name")
            if name and not name.endswith(("-mongo", "-mysql")):
                names.append(name)
    return sorted(set(names))


def slice_app_deployments(path: str, namespace: str, resource_profile: str) -> list[str]:
    names: list[str] = []
    for doc in render_trainticket_slice(path, namespace, resource_profile, include_policies=False):
        if doc.get("kind") != "Deployment":
            continue
        name = doc.get("metadata", {}).get("name")
        if name and not name.endswith(("-mongo", "-mysql")):
            names.append(name)
    return sorted(set(names))


def scaled_object(
    name: str,
    namespace: str,
    prometheus_address: str,
    min_replicas: int,
    max_replicas: int,
    cpu_target: int,
    request_rate_threshold: str,
) -> dict[str, Any]:
    query = (
        "sum(rate(istio_requests_total"
        f'{{destination_workload_namespace="{namespace}",destination_workload="{name}"}}[1m]))'
    )
    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {
            "name": f"{name}-keda",
            "namespace": namespace,
            "labels": {"app.kubernetes.io/part-of": "headlight-paper"},
        },
        "spec": {
            "scaleTargetRef": {"name": name},
            "pollingInterval": 15,
            "cooldownPeriod": 60,
            "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas,
            "triggers": [
                {
                    "type": "cpu",
                    "metricType": "Utilization",
                    "metadata": {
                        "value": str(cpu_target),
                    },
                },
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": prometheus_address,
                        "metricName": f"{name.replace('-', '_')}_request_rate",
                        "threshold": request_rate_threshold,
                        "query": query,
                    },
                },
            ],
        },
    }


def render(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = (
        app_deployments(args.source)
        if args.source
        else slice_app_deployments(args.trainticket_path, args.namespace, args.resource_profile)
    )
    return [
        scaled_object(
            name,
            args.namespace,
            args.prometheus_address,
            args.min_replicas,
            args.max_replicas,
            args.cpu_target,
            args.request_rate_threshold,
        )
        for name in names
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--trainticket-path",
        choices=["login-route", "advanced-ticket", "full"],
        default=os.environ.get("HEADLIGHT_TRAINTICKET_PATH", "full"),
    )
    parser.add_argument(
        "--resource-profile",
        choices=["local-tiny", "capacity-vm", "source"],
        default=os.environ.get("HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE", "capacity-vm"),
    )
    parser.add_argument(
        "--prometheus-address",
        default="http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
    )
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=8)
    parser.add_argument("--cpu-target", type=int, default=70)
    parser.add_argument("--request-rate-threshold", default="10")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source:
        args.source = resolve_repo_path(args.source, must_exist=True)
    if args.output:
        args.output = resolve_output_path(args.output)

    text = yaml.safe_dump_all(render(args), sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
