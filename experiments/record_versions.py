#!/usr/bin/env python3
"""Record repository, language, Kubernetes, and container/tool versions."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.paths import resolve_output_path, resolve_repo_path


def run_version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )
    except FileNotFoundError:
        return {"available": False, "command": command, "output": ""}
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "command": command, "output": f"timeout: {exc}"}
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def git_commit() -> str | None:
    result = run_version(["git", "rev-parse", "HEAD"])
    return result["output"] if result["available"] else None


def load_pins(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return data.get("pinned_environment", {})


def collect(config: Path) -> dict[str, Any]:
    return {
        "schema_version": "headlight.version-record/v1",
        "captured_at_unix": int(time.time()),
        "git_commit": git_commit(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "pinned_environment": load_pins(config),
        "commands": {
            "python": run_version([sys.executable, "--version"]),
            "go": run_version(["go", "version"]),
            "kubectl": run_version(["kubectl", "version", "--client=true"]),
            "helm": run_version(["helm", "version", "--short"]),
            "kind": run_version(["kind", "version"]),
            "karmadactl": run_version(["karmadactl", "version"]),
            "istioctl": run_version(["istioctl", "version", "--remote=false"]),
            "docker": run_version(["docker", "version", "--format", "{{json .Client.Version}}"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/config/softwarex_reproduction.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = resolve_repo_path(args.config, must_exist=True)
    output = resolve_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collect(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
