#!/usr/bin/env python3
"""Run one short seed-401 shadow canary with the frozen v2 bundle and diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_experiment import apply_protocol_metadata, environment, run_one_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("canary output must be a new directory")
    if os.environ.get("HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY") != "1" or not os.environ.get("HEADLIGHT_V2_DIAGNOSTIC_DIR"):
        raise ValueError("emergency-disabled mode and diagnostic directory must be explicit")
    os.environ["HEADLIGHT_V2_ENABLE_ACTUATION"] = "0"
    os.environ.setdefault("HEADLIGHT_V2_BUNDLE_DIR", str(ROOT / "results/headlight-v2-live/bundle-full-seed-601"))
    os.environ.setdefault("HEADLIGHT_V2_POLICY_CONFIG", str(ROOT / "experiments/configs/models/headlight-v2-closed-loop-policy.json"))
    reference = ROOT / "results/headlight-v2-live/guarded-gradual-ramp-c4-seed-101-20260928/manifest.yaml"
    manifest = copy.deepcopy(yaml.safe_load(reference.read_text()))
    manifest["metadata"]["name"] = "headlight-v2-emergency-disabled-diagnostic-canary"
    manifest["spec"]["run"].update(seeds=[401], warmup_seconds=390, duration_seconds=60, output_dir=str(output))
    manifest["spec"]["run"]["qualification_overrides"]["kind"] = "headlight-v2-diagnostic-canary"
    manifest["spec"]["workload"].update(duration_seconds=60, warmup_seconds=390, capacity_users=4)
    manifest["spec"]["workload"]["locust"].update(start_users=1, end_users=1, ramp_seconds=1, hold_seconds=60)
    manifest["spec"]["workload"]["shape_parameters"] = dict(manifest["spec"]["workload"]["locust"])
    apply_protocol_metadata(manifest)
    output.mkdir()
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "environment.json").write_text(json.dumps(environment(False), indent=2) + "\n")
    run_one_seed(manifest, output, 401, False)
    records = [json.loads(line) for line in (output / "seed-401/raw/diagnostics/decisions.jsonl").read_text().splitlines()]
    ready = [r for r in records if r.get("model_status") == "ok"]
    if not ready or any(r.get("capture_error") or r.get("record_storage_error") for r in ready):
        raise RuntimeError("canary has no complete replayable model diagnostics")
    if any(r.get("local_override") or r.get("interval_override") for r in ready):
        raise RuntimeError("emergency intervention occurred during disabled canary")
    if any(r["effective_policy"]["reactive_emergency_enabled"] for r in ready):
        raise RuntimeError("deployed policy has emergencies enabled")
    metadata = json.loads((ROOT / "results/headlight-v2-live/bundle-full-seed-601/metadata.json").read_text())
    (output / "shadow-smoke.ok").write_text(f"bundle_id={metadata['bundle_id']}\nstatus=passed\nseed=401\n")
    print(json.dumps({"canary": "passed", "scored_service_records": len(ready), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
