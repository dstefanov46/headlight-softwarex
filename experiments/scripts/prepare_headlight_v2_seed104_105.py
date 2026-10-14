#!/usr/bin/env python3
"""Rebuild an idle VM testbed and prepare fresh Headlight extension evidence."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.scripts.frozen_headlight_v2_source import verify_assets

SHARED_LOCK = REPO_ROOT / "results/headlight-v2-live/qualification-seed102-103-20261004/campaign.lock"
BUNDLE_ID = "c37050a44dd2fc876bf6523aaeabdd364af7608ffeee5e1586a6cb8bbaaaeb4a"


def prepare(root: Path) -> None:
    root = root.resolve()
    results = REPO_ROOT / "results"
    if not root.is_relative_to(results) or root == results:
        raise ValueError("campaign directory must be a subdirectory of repository results")
    if any(root.is_relative_to(results / name) for name in
           ("headlight-v2-live", "socc26", "capacity-qualification")):
        raise ValueError("refusing to prepare a historical results directory")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("preparation requires a fresh empty campaign directory; existing artifacts are preserved")
    # Acquire the lock used by all campaign launchers BEFORE any rebuild, build,
    # deployment, or new campaign artifact. Do not truncate existing lock files.
    SHARED_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        shared = stack.enter_context(SHARED_LOCK.open("a"))
        try:
            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another campaign process holds the testbed lock; preparation did nothing") from exc
        verify_assets(REPO_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        local = stack.enter_context((root / "campaign.lock").open("a"))
        fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script_dir = REPO_ROOT / "experiments/scripts"
        smoke_dir = root / "shadow-smoke"
        environment_file = root / "campaign-env.sh"
        environment = {**os.environ,
            "HEADLIGHT_CAMPAIGN_STATE_DIR": str(root),
            "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
            "HEADLIGHT_V2_ENABLE_ACTUATION": "0",
            "HEADLIGHT_V2_SHADOW_EVIDENCE": str(smoke_dir / "shadow-smoke.ok"),
            "HEADLIGHT_SMOKE_ARTIFACT_DIR": str(smoke_dir),
            "HEADLIGHT_V2_IMAGE_ARTIFACT_DIR": str(root / "image-build"),
        }
        options = {"cwd": REPO_ROOT, "env": environment, "check": True,
                   "pass_fds": (shared.fileno(), local.fileno())}
        subprocess.run(["bash", str(script_dir / "prepare_seed102_103_testbed.sh"),
                        str(environment_file)], **options)
        subprocess.run(["bash", "-c",
                        'set -euo pipefail; source "$1"; export HEADLIGHT_V2_ENABLE_ACTUATION=0; exec bash "$2"',
                        "headlight-v2-shadow-preparation", str(environment_file),
                        str(script_dir / "smoke_headlight_v2_shadow.sh")], **options)
        marker = smoke_dir / "shadow-smoke.ok"
        fields = dict(line.split("=", 1) for line in marker.read_text().splitlines() if "=" in line)
        if fields.get("bundle_id") != BUNDLE_ID or fields.get("status") != "passed":
            raise RuntimeError("new testbed did not produce matching successful shadow smoke evidence")
        evidence = {"schema_version": "headlight.v2-extension-preparation/v1", "status": "ready",
                    "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "campaign_environment": str(environment_file), "shadow_smoke": str(marker),
                    "image_build_provenance": str(root / "image-build/image-build-provenance.json"),
                    "environment_equivalence_established": False}
        (root / "preparation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"HEADLIGHT_V2_EXTENSION_PREPARED campaign={root} environment={environment_file}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    prepare(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
