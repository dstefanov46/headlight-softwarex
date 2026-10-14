#!/usr/bin/env python3
"""Build the frozen research-v3 static-baseline plan from calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.lib.config import REPO_ROOT
from experiments.lib.protocol import calibration_seeds
from experiments.lib.static_plan import build_static_plan, validate_static_plan


sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))
from headlight.profile import load_service_profile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-contract", default="results/capacity-qualification/capacity-contract.json")
    parser.add_argument("--service-profile", default="ai-module/configs/service_profiles/advanced-ticket.json")
    parser.add_argument("--calibration-trace", required=True)
    parser.add_argument("--output", default="results/capacity-qualification/static-plan.json")
    args = parser.parse_args()
    contract = resolve(Path(args.capacity_contract))
    profile_path = resolve(Path(args.service_profile))
    trace = resolve(Path(args.calibration_trace))
    output = resolve(Path(args.output))
    profile = load_service_profile(profile_path)
    plan = build_static_plan(
        service_names=[node.workload for node in profile.scalable_nodes],
        capacity_contract_path=contract,
        calibration_trace_name=str(trace.relative_to(REPO_ROOT) if trace.is_relative_to(REPO_ROOT) else trace),
        calibration_trace_hash=file_sha256(trace),
        calibration_seeds=list(calibration_seeds()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    validate_static_plan(output)
    print(f"wrote frozen static plan: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
