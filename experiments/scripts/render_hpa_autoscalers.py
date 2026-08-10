#!/usr/bin/env python3
"""Render HPA objects for the selected TrainTicket slice."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.render_minimal_trainticket import render as render_trainticket_slice
from experiments.lib.paths import resolve_output_path, resolve_repo_path


HPA_MANIFEST = REPO_ROOT / "ai-module" / "kubernetes-autoscaler" / "hpa.yaml"


def load_yaml(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def slice_deployments(path: str, namespace: str, resource_profile: str) -> set[str]:
    return {
        doc["metadata"]["name"]
        for doc in render_trainticket_slice(path, namespace, resource_profile, include_policies=False)
        if doc.get("kind") == "Deployment"
        and not doc["metadata"]["name"].endswith(("-mongo", "-mysql"))
    }


def render(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = slice_deployments(args.trainticket_path, args.namespace, args.resource_profile)
    docs: list[dict[str, Any]] = []
    for doc in load_yaml(args.source):
        if doc.get("kind") != "HorizontalPodAutoscaler":
            continue
        target = doc.get("spec", {}).get("scaleTargetRef", {}).get("name")
        if target not in wanted:
            continue
        rendered = copy.deepcopy(doc)
        rendered.setdefault("metadata", {})["namespace"] = args.namespace
        rendered.setdefault("spec", {})["maxReplicas"] = args.max_replicas
        docs.append(rendered)
    missing = sorted(wanted - {doc.get("spec", {}).get("scaleTargetRef", {}).get("name") for doc in docs})
    if missing:
        raise SystemExit(f"HPA source did not contain expected targets: {', '.join(missing)}")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--source", type=Path, default=HPA_MANIFEST)
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
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
