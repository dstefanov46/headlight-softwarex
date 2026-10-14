#!/usr/bin/env python3
"""Read a completed seed-104/105 campaign without altering its artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.lib.config import REPO_ROOT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(campaign: Path) -> dict:
    campaign = campaign.resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
        if not path.is_relative_to(campaign):
            raise ValueError(f"artifact escapes campaign: {path}")
        return path

    def relative(path: Path) -> str:
        return str(path.relative_to(REPO_ROOT))

    status_path = campaign / "campaign-status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        raise ValueError("baseline campaign is not complete")
    expected = {(seed, workload, controller) for seed in (104, 105)
                for workload in ("gradual-ramp", "flash-crowd", "diurnal")
                for controller in ("hpa", "keda")}
    seen = set()
    sources = []
    for cell in status["cells"]:
        identity = (int(cell["seed"]), cell["workload"], cell["controller"])
        if identity not in expected or identity in seen or cell.get("status") != "valid":
            raise ValueError(f"unexpected, duplicate, or invalid campaign cell: {identity}")
        seen.add(identity)
        seed, workload, controller = identity
        output = resolve(cell["valid_output_dir"])
        proofs = []
        for candidate in sorted((output.parent / "status").glob("*.json")):
            proof = json.loads(candidate.read_text())
            if (proof.get("status") == "valid" and proof.get("output_dir")
                    and resolve(proof["output_dir"]) == output
                    and proof.get("cell") == {"seed": seed, "workload": workload, "controller": controller}):
                proofs.append((candidate, proof))
        if len(proofs) != 1:
            raise ValueError(f"expected one valid attempt proof for {identity}")
        proof_path, proof = proofs[0]
        snapshot_ref = proof["environment_provenance"]
        snapshot = resolve(snapshot_ref["path"])
        if sha256(snapshot) != snapshot_ref["sha256"]:
            raise ValueError(f"environment snapshot hash mismatch: {snapshot}")
        seed_dir = output / f"seed-{seed}"
        trace = seed_dir / "raw/load_profile.csv"
        trace_hash = sha256(trace)
        if trace_hash != proof["expected_trace_sha256"]:
            raise ValueError(f"attempt load trace hash mismatch: {trace}")
        summary_path = seed_dir / "summaries/summary.csv"
        with summary_path.open(newline="") as handle:
            summaries = list(csv.DictReader(handle))
        if len(summaries) != 1:
            raise ValueError(f"expected one summary row: {summary_path}")
        summary = summaries[0]
        if ((int(summary["seed"]), summary["workload"], summary["controller"]) != identity
                or summary.get("technical_validity") != "valid"
                or summary.get("performance_outcome") not in {"measured", "poor_but_valid"}):
            raise ValueError(f"summary does not confirm a valid cell: {summary_path}")
        sources.append({
            "seed": seed, "workload": workload, "controller": controller,
            "output_dir": relative(output), "load_trace_sha256": trace_hash,
            "measurement_window_json": relative(seed_dir / "raw/measurement_window.json"),
            "request_events_csv": relative(seed_dir / "raw/request_events.csv"),
            "summary_csv": relative(summary_path),
            "technical_validity": "valid", "performance_outcome": summary["performance_outcome"],
            "environment_provenance": snapshot_ref,
            "status_reference": {"path": relative(proof_path), "sha256": sha256(proof_path)},
        })
    if seen != expected:
        raise ValueError(f"missing campaign cells: {sorted(expected - seen)}")
    return {
        "schema_version": "headlight.threshold-baseline-sources/v1",
        "campaign_status": {"path": relative(status_path), "sha256": sha256(status_path)},
        "cohort": "seed104-105-persistent-docker", "capacity": 4,
        "environment_equivalence_established": False,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.campaign_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    args.output.chmod(0o444)
    print(f"Validated {len(manifest['sources'])} baseline source cells: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
