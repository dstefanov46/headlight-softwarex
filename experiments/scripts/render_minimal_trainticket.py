#!/usr/bin/env python3
"""Render a TrainTicket slice from the checked-in manifests."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_MANIFEST = REPO_ROOT / "ai-module" / "deployments" / "application-services-deployment.yaml"
DB_MANIFEST = REPO_ROOT / "ai-module" / "deployments" / "db-services-deployment.yaml"
APP_POLICY_MANIFEST = REPO_ROOT / "ai-module" / "propagation-policies" / "propagation-policy-application.yaml"
DB_DEPLOY_POLICY_MANIFEST = REPO_ROOT / "ai-module" / "propagation-policies" / "propagation-policy-db-deployments.yaml"
DB_SERVICE_POLICY_MANIFEST = REPO_ROOT / "ai-module" / "propagation-policies" / "propagation-policy-db-services.yaml"

LOGIN_ROUTE_APPS = {"ts-auth", "ts-route"}
LOGIN_ROUTE_DBS = {"ts-auth-mongo", "ts-route-mongo"}

ADVANCED_APPS = {
    "ts-travel-plan",
    "ts-travel",
    "ts-travel2",
    "ts-route-plan",
    "ts-basic",
    "ts-station",
    "ts-ticketinfo",
    "ts-train",
    "ts-price",
}
ADVANCED_DBS = {
    "ts-travel-mongo",
    "ts-travel2-mongo",
    "ts-station-mongo",
    "ts-train-mongo",
    "ts-price-mongo",
}

NAMESPACED_KINDS = {
    "ConfigMap",
    "ControlledReplicaPropagation",
    "Deployment",
    "PropagationPolicy",
    "ResourceQuota",
    "Secret",
    "Service",
    "ServiceAccount",
}

LOCAL_24GB_ADMISSION_QUOTA_NAME = "local-24gb-tested-admission"
LOCAL_24GB_ADMISSION_MEMORY = "2078Mi"
MONGO_STORAGE_MODES = ("pod-emptydir", "image-defined")
DEFAULT_MONGO_STORAGE = "pod-emptydir"

RESOURCE_PROFILES = {
    "source": None,
    # Full service graph sized for the 32-vCPU/126-GiB qualification VM.  The
    # source requests over-reserve the local Kind nodes; local-tiny is useful
    # for smoke tests but permits unrealistic density and CPU throttling.
    "capacity-vm": {
        "app": {
            "requests": {"cpu": "50m", "memory": "128Mi", "ephemeral-storage": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi", "ephemeral-storage": "1Gi"},
        },
        "db": {
            "requests": {"cpu": "25m", "memory": "512Mi", "ephemeral-storage": "256Mi"},
            "limits": {"cpu": "200m", "memory": "1Gi", "ephemeral-storage": "2Gi"},
        },
        "sidecar": {
            "sidecar.istio.io/proxyCPU": "20m",
            "sidecar.istio.io/proxyMemory": "64Mi",
            "sidecar.istio.io/proxyCPULimit": "200m",
            "sidecar.istio.io/proxyMemoryLimit": "192Mi",
        },
    },
    "local-tiny": {
        "app": {
            "requests": {"cpu": "10m", "memory": "96Mi", "ephemeral-storage": "128Mi"},
            "limits": {"cpu": "150m", "memory": "512Mi", "ephemeral-storage": "1Gi"},
        },
        "db": {
            "requests": {"cpu": "10m", "memory": "64Mi", "ephemeral-storage": "256Mi"},
            "limits": {"cpu": "200m", "memory": "384Mi", "ephemeral-storage": "2Gi"},
        },
        "sidecar": {
            "sidecar.istio.io/proxyCPU": "10m",
            "sidecar.istio.io/proxyMemory": "64Mi",
            "sidecar.istio.io/proxyCPULimit": "100m",
            "sidecar.istio.io/proxyMemoryLimit": "128Mi",
        },
    },
    # Feasibility profile for the 24-GiB-class local machine. Application and
    # database resources were re-verified against the upstream deployment
    # manifests at the recorded revision. Keep the already exercised local
    # Istio proxy override visible here: it is part of each application pod's
    # effective reservation and limit, not hidden injector overhead.
    "local-24gb-tested": {
        "metadata": {
            "workload_path": "advanced-ticket",
            "upstream_deployments_url": (
                "https://gitlab.eclipse.org/eclipse-research-labs/nephele-project/"
                "opencall-2/headlight/ai-module/-/tree/main/deployments"
            ),
            "upstream_revision": "2d1ae2c3e255f464c993b980b207e88dced7f796",
            "verified_on": "2026-09-16",
            "sidecar_basis": "preserved from local-tiny",
            "admission_quota_basis": "measured 2026-09-17",
            "admission_requests_memory_per_cluster": LOCAL_24GB_ADMISSION_MEMORY,
            "mongo_limits": "intentionally unspecified after idle-only feasibility measurement",
        },
        "app": {
            "requests": {"cpu": "25m", "memory": "80Mi"},
            "limits": {"cpu": "200m", "memory": "500Mi"},
        },
        "db": {
            "requests": {"cpu": "25m", "memory": "50Mi"},
        },
        "sidecar": {
            "sidecar.istio.io/proxyCPU": "10m",
            "sidecar.istio.io/proxyMemory": "64Mi",
            "sidecar.istio.io/proxyCPULimit": "100m",
            "sidecar.istio.io/proxyMemoryLimit": "128Mi",
        },
    },
}


def load_yaml(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def by_name(paths: list[Path], names: set[str]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for path in paths:
        for doc in load_yaml(path):
            if doc.get("metadata", {}).get("name") in names:
                rendered.append(copy.deepcopy(doc))
    return rendered


def service_names(workload_names: set[str]) -> set[str]:
    return {f"{name}-service" for name in workload_names}


def names_by_kind(path: Path, kind: str) -> set[str]:
    return {doc["metadata"]["name"] for doc in load_yaml(path) if doc.get("kind") == kind}


def _normalize_deployment(
    doc: dict[str, Any],
    name: str | None,
    db_names: set[str],
    app_names: set[str],
    resource_profile: str,
    mongo_storage: str,
) -> None:
    doc.setdefault("spec", {})["replicas"] = 1
    template_spec = doc.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})
    template_spec["automountServiceAccountToken"] = False
    template_metadata = doc.setdefault("spec", {}).setdefault("template", {}).setdefault("metadata", {})
    annotations = template_metadata.setdefault("annotations", {})
    if name in db_names:
        annotations["sidecar.istio.io/inject"] = "false"
        profile_key = "db"
    elif name in app_names:
        annotations["sidecar.istio.io/inject"] = "true"
        profile = RESOURCE_PROFILES[resource_profile]
        if profile:
            annotations.update(profile["sidecar"])
        profile_key = "app"
    else:
        profile_key = None
    profile = RESOURCE_PROFILES[resource_profile]
    if profile_key and profile:
        for container in template_spec.get("containers", []):
            container["resources"] = profile[profile_key]
    if name in db_names and name.endswith("-mongo") and mongo_storage == "pod-emptydir":
        _configure_mongo_storage(template_spec)
        annotations["headlight.research/mongo-storage"] = "pod-emptydir-v1"


def _configure_mongo_storage(pod_spec: dict[str, Any]) -> None:
    # Both paths are declared VOLUMEs by mongo:4.0. Explicit, node-filesystem
    # emptyDirs belong to the pod, so container restarts reuse its journals
    # instead of allocating another pair of implicit containerd volumes.
    mounts = [
        {"name": "mongo-data", "mountPath": "/data/db"},
        {"name": "mongo-configdb", "mountPath": "/data/configdb"},
    ]
    volumes = [{"name": m["name"], "emptyDir": {}} for m in mounts]
    existing_volumes = pod_spec.setdefault("volumes", [])
    for volume in volumes:
        existing = next((v for v in existing_volumes if v["name"] == volume["name"]), None)
        if existing is not None and existing != volume:
            raise ValueError(f"existing Mongo volume {volume['name']} conflicts with pod-emptydir storage")
        if existing is None:
            existing_volumes.append(volume)
    for container in pod_spec.get("containers", []):
        if not container.get("name", "").endswith("-mongo"):
            continue
        existing_mounts = container.setdefault("volumeMounts", [])
        for mount in mounts:
            existing = next((m for m in existing_mounts if m["mountPath"] == mount["mountPath"]), None)
            if existing is not None and existing != mount:
                raise ValueError(f"existing Mongo mount {mount['mountPath']} conflicts with pod-emptydir storage")
            if existing is None:
                existing_mounts.append(dict(mount))


def _normalize_propagation(doc: dict[str, Any], namespace: str) -> None:
    for selector in doc.setdefault("spec", {}).setdefault("resourceSelectors", []):
        selector["namespace"] = namespace
    placement = doc.setdefault("spec", {}).setdefault("placement", {})
    placement.setdefault("clusterAffinity", {})["clusterNames"] = ["cluster2", "cluster3"]


def _normalize_crp(doc: dict[str, Any]) -> None:
    placements = doc.setdefault("spec", {}).setdefault("placements", [])
    for placement in placements:
        if placement.get("clusterName") in {"cluster2", "cluster3"}:
            placement["replicas"] = 1
    doc["spec"]["placements"] = [
        placement for placement in placements if placement.get("clusterName") in {"cluster2", "cluster3"}
    ]


def normalize(
    doc: dict[str, Any],
    namespace: str,
    db_names: set[str],
    app_names: set[str],
    resource_profile: str,
    mongo_storage: str = DEFAULT_MONGO_STORAGE,
) -> dict[str, Any]:
    metadata = doc.setdefault("metadata", {})
    kind = doc.get("kind")
    name = metadata.get("name")
    if kind in NAMESPACED_KINDS:
        metadata["namespace"] = namespace

    if kind == "Service":
        spec = doc.setdefault("spec", {})
        if spec.get("type") == "LoadBalancer":
            spec["type"] = "ClusterIP"

    if kind == "Deployment":
        _normalize_deployment(doc, name, db_names, app_names, resource_profile, mongo_storage)

    if kind == "PropagationPolicy":
        _normalize_propagation(doc, namespace)

    if kind == "ControlledReplicaPropagation":
        _normalize_crp(doc)

    return doc


def render(
    path: str,
    namespace: str,
    resource_profile: str,
    include_policies: bool = True,
    *,
    mongo_storage: str = DEFAULT_MONGO_STORAGE,
) -> list[dict[str, Any]]:
    if mongo_storage not in MONGO_STORAGE_MODES:
        raise ValueError(f"unsupported Mongo storage mode: {mongo_storage}")
    app_names = set(LOGIN_ROUTE_APPS)
    db_names = set(LOGIN_ROUTE_DBS)
    if path == "advanced-ticket":
        app_names |= ADVANCED_APPS
        db_names |= ADVANCED_DBS
    elif path == "full":
        app_names = names_by_kind(APP_MANIFEST, "Deployment")
        db_names = names_by_kind(DB_MANIFEST, "Deployment")

    if path == "full":
        app_services = names_by_kind(APP_MANIFEST, "Service")
        db_services = names_by_kind(DB_MANIFEST, "Service")
    else:
        app_services = service_names(app_names)
        db_services = set(db_names)

    docs: list[dict[str, Any]] = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace,
                "labels": {"istio-injection": "enabled"},
            },
        }
    ]
    docs.extend(by_name([DB_MANIFEST], db_names | db_services))
    docs.extend(by_name([APP_MANIFEST], app_names | app_services))
    if include_policies:
        docs.extend(by_name([DB_DEPLOY_POLICY_MANIFEST], db_names))
        docs.extend(by_name([APP_POLICY_MANIFEST], app_names | app_services))
        docs.extend(by_name([DB_SERVICE_POLICY_MANIFEST], db_services))

    wanted = db_names | db_services | app_names | app_services
    rendered = [docs[0]]
    rendered.extend(normalize(doc, namespace, db_names, app_names, resource_profile, mongo_storage) for doc in docs[1:])

    if path == "advanced-ticket" and resource_profile == "local-24gb-tested":
        rendered.append(
            {
                "apiVersion": "v1",
                "kind": "ResourceQuota",
                "metadata": {
                    "name": LOCAL_24GB_ADMISSION_QUOTA_NAME,
                    "namespace": namespace,
                },
                "spec": {"hard": {"requests.memory": LOCAL_24GB_ADMISSION_MEMORY}},
            }
        )
        if include_policies:
            rendered.append(
                {
                    "apiVersion": "policy.karmada.io/v1alpha1",
                    "kind": "PropagationPolicy",
                    "metadata": {
                        "name": LOCAL_24GB_ADMISSION_QUOTA_NAME,
                        "namespace": namespace,
                    },
                    "spec": {
                        "resourceSelectors": [
                            {
                                "apiVersion": "v1",
                                "kind": "ResourceQuota",
                                "name": LOCAL_24GB_ADMISSION_QUOTA_NAME,
                                "namespace": namespace,
                            }
                        ],
                        "placement": {
                            "clusterAffinity": {"clusterNames": ["cluster2", "cluster3"]}
                        },
                    },
                }
            )

    missing = sorted(wanted - {doc.get("metadata", {}).get("name") for doc in rendered})
    if missing:
        raise SystemExit(f"source manifests did not contain expected objects: {', '.join(missing)}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--path", choices=["login-route", "advanced-ticket", "full"], default="login-route")
    parser.add_argument("--resource-profile", choices=sorted(RESOURCE_PROFILES))
    parser.add_argument(
        "--mongo-storage", choices=MONGO_STORAGE_MODES, default=DEFAULT_MONGO_STORAGE,
        help="pod-emptydir survives container restarts and resets on pod replacement; image-defined reproduces historical storage",
    )
    parser.add_argument("--no-policies", action="store_true", help="render only Kubernetes-native objects")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    resource_profile = args.resource_profile or ("source" if args.path == "full" else "local-tiny")
    text = yaml.safe_dump_all(render(args.path, args.namespace, resource_profile, include_policies=not args.no_policies, mongo_storage=args.mongo_storage), sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
