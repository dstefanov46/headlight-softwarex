#!/usr/bin/env python3
"""Generate versioned service profiles from rendered TrainTicket manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from experiments.scripts.render_minimal_trainticket import RESOURCE_PROFILES, render
from experiments.lib.paths import resolve_output_path
from experiments.lib.protocol import replica_limits
from headlight.profile import build_profile_payload


PROFILE_PATHS = ("login-route", "advanced-ticket", "full")
OUTPUT_DIR = REPO_ROOT / "ai-module" / "configs" / "service_profiles"


def _is_database(name: str) -> bool:
    return name.endswith(("-mongo", "-mysql"))


def _service_for(workload: str, service_names: set[str], database: bool) -> str:
    candidates = [workload, f"{workload}-service"] if database else [f"{workload}-service", workload]
    for candidate in candidates:
        if candidate in service_names:
            return candidate
    raise ValueError(f"rendered manifest has no Service for Deployment {workload}")


def profile_from_rendered(
    profile_name: str,
    namespace: str = "trainticket",
    resource_profile: str = "capacity-vm",
    cluster_max_replicas: dict[str, int] | None = None,
) -> dict[str, Any]:
    docs = render(profile_name, namespace, resource_profile)
    deployments = sorted(
        {
            str(doc["metadata"]["name"])
            for doc in docs
            if doc.get("kind") == "Deployment"
        }
    )
    service_names = {
        str(doc["metadata"]["name"])
        for doc in docs
        if doc.get("kind") == "Service"
    }
    nodes: list[dict[str, Any]] = []
    for workload in deployments:
        database = _is_database(workload)
        nodes.append(
            {
                "name": workload,
                "workload": workload,
                "service_name": _service_for(workload, service_names, database),
                "service_type": "database" if database else "application",
                "scalable": not database,
            }
        )
    rendered_bytes = yaml.safe_dump_all(docs, sort_keys=True).encode("utf-8")
    maxima = cluster_max_replicas or {
        cluster: int(max_replicas)
        for cluster, max_replicas in replica_limits()["per_cluster_max_replicas"].items()
    }
    return build_profile_payload(
        profile_name=profile_name,
        namespace=namespace,
        nodes=nodes,
        cluster_limits={
            cluster: {
                "min_replicas": 1,
                "initial_replicas": 1,
                "max_replicas": int(max_replicas),
            }
            for cluster, max_replicas in maxima.items()
        },
        source={
            "generator": "experiments/scripts/generate_service_profiles.py",
            "rendered_manifest_sha256": hashlib.sha256(rendered_bytes).hexdigest(),
            "resource_profile": resource_profile,
        },
    )


def write_profiles(
    output_dir: Path,
    paths: tuple[str, ...] = PROFILE_PATHS,
    namespace: str = "trainticket",
    resource_profile: str | None = None,
    cluster_max_replicas: dict[str, int] | None = None,
    output_name: str | None = None,
) -> list[Path]:
    if output_name and len(paths) != 1:
        raise ValueError("output_name requires exactly one path")
    output_dir = resolve_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for profile_name in paths:
        selected_resource_profile = resource_profile or {
            "login-route": "local-tiny",
            "advanced-ticket": "capacity-vm",
            "full": "source",
        }[profile_name]
        payload = profile_from_rendered(
            profile_name,
            namespace,
            selected_resource_profile,
            cluster_max_replicas,
        )
        output = output_dir / (output_name or f"{profile_name}.json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--resource-profile", choices=sorted(RESOURCE_PROFILES))
    parser.add_argument("--path", action="append", choices=PROFILE_PATHS)
    parser.add_argument("--cluster2-max", type=int)
    parser.add_argument("--cluster3-max", type=int)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    paths = tuple(args.path) if args.path else PROFILE_PATHS
    if (args.cluster2_max is None) != (args.cluster3_max is None):
        parser.error("--cluster2-max and --cluster3-max must be supplied together")
    maxima = None
    if args.cluster2_max is not None:
        maxima = {"cluster2": args.cluster2_max, "cluster3": args.cluster3_max}
    outputs = write_profiles(
        args.output_dir,
        paths,
        args.namespace,
        args.resource_profile,
        maxima,
        args.output_name,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
