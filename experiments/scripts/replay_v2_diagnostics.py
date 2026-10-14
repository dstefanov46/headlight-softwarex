#!/usr/bin/env python3
"""Verify captured artifacts and replay one scored service with the frozen model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-module/ai-module"))
from headlight.bundle_v2 import HeadlightV2Bundle, load_v2_checkpoint  # noqa: E402
from headlight.models_v2 import build_headlight_v2_model  # noqa: E402


def replay(record, directory: Path, bundle_path: Path):
    bundle = HeadlightV2Bundle.load(bundle_path, expected_model_type="headlight-v2")
    if bundle.bundle_id != record["decision"]["bundle_id"]:
        raise ValueError("replay bundle differs from diagnostic bundle")
    arrays = {}
    for name, reference in record["replay"].items():
        path = directory / reference["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"replay artifact hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays[name] = {key: archive[key] for key in archive.files}
    device = torch.device("cpu")
    model = build_headlight_v2_model(bundle.model_config,
                                    torch.tensor(bundle.adjacency["matrix"], dtype=torch.float32),
                                    torch.tensor(bundle.adjacency["observed_mask"], dtype=torch.float32))
    model.load_state_dict(load_v2_checkpoint(bundle, device), strict=True)
    model.eval()
    def tensor(array):
        return torch.from_numpy(array)
    with torch.no_grad():
        output = model(tensor(arrays["window"]["node_window"]), tensor(arrays["window"]["edge_window"]),
                       tensor(arrays["candidates"]["replicas"]), tensor(arrays["candidates"]["placement"]),
                       tensor(arrays["candidates"]["readiness"]))
    for name, expected in arrays["outputs"].items():
        actual = output[name] if name in output else output["candidate_performance"][name]
        torch.testing.assert_close(actual, tensor(expected), rtol=1e-5, atol=1e-6)
    return {"bundle_id": bundle.bundle_id, "service": record["decision"]["service"],
            "outputs_verified": sorted(arrays["outputs"]), "device": "cpu", "rtol": 1e-5, "atol": 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--service", default="ts-auth")
    parser.add_argument("--timestamp", type=float)
    args = parser.parse_args()
    records = (json.loads(line) for line in args.records.read_text().splitlines())
    record = next((r for r in records if r.get("replay") and r["decision"]["service"] == args.service
                   and (args.timestamp is None or r["decision"]["timestamp"] == args.timestamp)), None)
    if record is None:
        raise ValueError("no matching replayable service record")
    print(json.dumps(replay(record, args.records.parent, args.bundle), indent=2))


if __name__ == "__main__":
    main()
