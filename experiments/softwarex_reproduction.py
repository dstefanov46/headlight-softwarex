#!/usr/bin/env python3
"""Deterministic SoftwareX reproduction workflow driver."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.paths import resolve_repo_path

CONFIG_SCHEMA = "headlight.softwarex-reproduction/v1"
STAGES = (
    "validate-env",
    "bootstrap-cluster",
    "deploy-monitoring",
    "deploy-headlight",
    "collect-or-load-data",
    "train-or-load-model",
    "run-workload",
    "run-controllers",
    "validate-results",
    "generate-compact-output",
)


class ReproductionConfigError(ValueError):
    pass


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ReproductionConfigError(f"{config_path}: top-level YAML value must be a mapping")
    if data.get("schema_version") != CONFIG_SCHEMA:
        raise ReproductionConfigError(f"{config_path}: unsupported schema_version")
    seeds = data.get("seeds", {})
    for key in ("training", "workload", "controller_order"):
        if not isinstance(seeds.get(key), int):
            raise ReproductionConfigError(f"{config_path}: seeds.{key} must be an integer")
    workflow = data.get("workflow", {})
    for key in ("service_profile", "workload_profile", "run_config", "output_root"):
        if not workflow.get(key):
            raise ReproductionConfigError(f"{config_path}: workflow.{key} is required")
    pins = data.get("pinned_environment", {})
    serialized_pins = json.dumps(pins)
    if ":latest" in serialized_pins or '"latest"' in serialized_pins:
        raise ReproductionConfigError(f"{config_path}: pinned_environment must not use latest")
    configured_stages = data.get("stages", {})
    missing = [stage for stage in STAGES if stage not in configured_stages]
    if missing:
        raise ReproductionConfigError(f"{config_path}: missing stages: {', '.join(missing)}")
    return data


def run_command(command: list[str] | str, log_path: Path, *, dry_run: bool) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(f"DRY-RUN would execute: {command}\n", encoding="utf-8")
        return {"returncode": 0, "dry_run": True, "command": command}
    args = command if isinstance(command, list) else ["bash", "-lc", command]
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"stage command failed: {command}")
    return {"returncode": completed.returncode, "dry_run": False, "command": command}


def output_root(config: dict[str, Any], suffix: str | None = None) -> Path:
    root = resolve(config["workflow"]["output_root"])
    if suffix:
        return root / suffix
    return root


def stage_validate_env(run_dir: Path, dry_run: bool) -> dict[str, Any]:
    output = run_dir / "environment_versions.json"
    return run_command(
        [sys.executable, "experiments/record_versions.py", "--config", "experiments/config/softwarex_reproduction.yaml", "--output", str(output)],
        run_dir / "logs" / "validate-env.log",
        dry_run=dry_run,
    )


def stage_collect_or_load_data(config: dict[str, Any], run_dir: Path, dry_run: bool) -> dict[str, Any]:
    data = config["sample_data"] if dry_run else config["real_data"]
    source = resolve(data["calibration_csv"])
    if not dry_run and not source.exists():
        raise FileNotFoundError(f"missing measured calibration data: {source}")
    record = {
        "label": data["label"],
        "calibration_csv": str(source.relative_to(REPO_ROOT) if source.exists() and source.is_relative_to(REPO_ROOT) else source),
        "synthetic_or_sample": dry_run,
        "validation_status": "dry-run" if dry_run else "measured-input-present",
    }
    (run_dir / "data_source.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def stage_generate_compact_output(config: dict[str, Any], run_dir: Path, dry_run: bool) -> dict[str, Any]:
    path = run_dir / "compact_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["controller", "workload", "seed", "data_label", "validation_status", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "controller": "headlight",
                "workload": "gradual-ramp",
                "seed": config["seeds"]["workload"],
                "data_label": "sample" if dry_run else "measured",
                "validation_status": "dry-run" if dry_run else "requires-artifact-validator",
                "note": "Dry-run contains no comparative measurement" if dry_run else "Generated from validated measured bundle only",
            }
        )
    return {"path": str(path), "validation_status": "dry-run" if dry_run else "pending"}


def execute_stage(config: dict[str, Any], stage: str, run_dir: Path, dry_run: bool) -> dict[str, Any]:
    if stage not in STAGES:
        raise ReproductionConfigError(f"unknown stage: {stage}")
    stage_config = config["stages"][stage]
    if stage == "validate-env":
        result = stage_validate_env(run_dir, dry_run)
    elif stage == "collect-or-load-data":
        result = stage_collect_or_load_data(config, run_dir, dry_run)
    elif stage == "generate-compact-output":
        result = stage_generate_compact_output(config, run_dir, dry_run)
    elif dry_run:
        result = {
            "dry_run": True,
            "validation_status": "planned-not-executed",
            "description": stage_config["description"],
            "commands": stage_config.get("commands") or [stage_config.get("command", "")],
        }
        (run_dir / "logs" / f"{stage}.log").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs" / f"{stage}.log").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        commands = stage_config.get("commands") or [stage_config.get("command")]
        if not commands or not commands[0]:
            result = {"validation_status": "manual-extension-required", "description": stage_config["description"]}
        else:
            results = [
                run_command(command, run_dir / "logs" / f"{stage}-{index}.log", dry_run=False)
                for index, command in enumerate(commands, start=1)
            ]
            result = {"commands": results, "validation_status": "executed"}
    return {"stage": stage, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/config/softwarex_reproduction.yaml")
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = args.run_id or ("dry-run" if args.dry_run else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = output_root(config, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)
    results = [execute_stage(config, stage, run_dir, args.dry_run) for stage in stages]
    summary = {
        "schema_version": "headlight.softwarex-reproduction-run/v1",
        "dry_run": args.dry_run,
        "run_id": run_id,
        "stages": results,
        "validation_status": "dry-run" if args.dry_run else "executed",
    }
    (run_dir / "reproduction_run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
