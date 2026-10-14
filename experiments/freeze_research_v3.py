#!/usr/bin/env python3
"""Create or check the tracked manifest for the external frozen release bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT
from experiments.lib.freeze import FREEZE_MANIFEST_PATH, create_freeze_manifest, validate_freeze_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the existing tracked manifest")
    parser.add_argument("--output", type=Path, default=FREEZE_MANIFEST_PATH)
    parser.add_argument("--release-root", default="external/research-v3-release")
    parser.add_argument("--capacity-contract", type=Path, default=REPO_ROOT / "results/capacity-qualification/capacity-contract.json")
    parser.add_argument("--static-plan", type=Path, default=REPO_ROOT / "results/capacity-qualification/static-plan.json")
    parser.add_argument("--headlight-bundle", type=Path, default=REPO_ROOT / "results/calibration/headlight-bundle")
    parser.add_argument("--lstm-bundle", type=Path, default=REPO_ROOT / "results/calibration/lstm-bundle")
    parser.add_argument("--policy", type=Path, default=REPO_ROOT / "ai-module/configs/policies/research-v3-forecast-policy.json")
    args = parser.parse_args()
    try:
        payload = validate_freeze_manifest(args.output) if args.check else create_freeze_manifest(
            capacity_contract=args.capacity_contract,
            static_plan=args.static_plan,
            headlight_bundle=args.headlight_bundle,
            lstm_bundle=args.lstm_bundle,
            policy=args.policy,
            output=args.output,
            release_root=args.release_root,
        )
    except Exception as exc:
        print(f"freeze validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "manifest": str(args.output), "protocol": payload["protocol"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
