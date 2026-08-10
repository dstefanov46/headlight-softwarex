"""Configuration loading and manifest resolution for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.validate_configs import REPO_ROOT, ValidationError, load_yaml, validate_all


def resolve_run(run_path: Path) -> dict[str, Any]:
    """Return the resolved manifest for one ExperimentRun YAML file."""
    target = run_path if run_path.is_absolute() else REPO_ROOT / run_path
    target = target.resolve()
    manifests, _ = validate_all()
    for manifest in manifests:
        for path in (REPO_ROOT / "experiments" / "configs" / "runs").glob("*.yaml"):
            if path.resolve() == target:
                data = load_yaml(path)
                if data["metadata"]["name"] == manifest["metadata"]["name"]:
                    return manifest
    raise ValidationError(f"unknown run config `{run_path}`")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
