from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.validate_artifacts import (
    ArtifactValidationError,
    ZERO_GUARD_FIELDS,
    assert_artifacts_not_stale,
    _summary_has_invalid_zeroes,
    validated_summary_rows,
)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2) + "\n")


def measured_bundle(root: Path, name: str = "run-a", seed: int = 1, *, zero_summary: bool = False) -> Path:
    bundle = root / name
    manifest = {
        "apiVersion": "headlight.experiments/v1alpha1",
        "kind": "ResolvedExperimentManifest",
        "metadata": {"name": "hpa__gradual-ramp__multi-cluster-headlight"},
        "spec": {
            "controller": {"controller": "hpa"},
            "workload": {"pattern": "gradual-ramp", "sampling_interval_seconds": 5},
            "topology": {"namespace": "trainticket", "active_clusters": ["cluster2"]},
            "run": {
                "seeds": [seed],
                "duration_seconds": 10,
                "sampling_interval_seconds": 5,
                "sla": {"p95_latency_ms": 1000, "minimum_success_rate": 0.90},
            },
        },
    }
    write_text(bundle / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    write_json(bundle / "environment.json", {"dry_run": False, "git_commit": "abc123"})
    seed_dir = bundle / f"seed-{seed}"
    write_text(seed_dir / "raw/load_profile.csv", "elapsed_seconds,users,spawn_rate\n0,1,1\n")
    write_json(seed_dir / "raw/prometheus_queries.json", {"pod_count": "query"})
    write_text(
        seed_dir / "raw/request_events.csv",
        "timestamp,method,endpoint,status,response_time_ms,exception\n"
        "0,GET,/,200,100,\n"
        "10,GET,/,200,200,\n",
    )
    write_json(seed_dir / "raw/kubernetes_events.json", {})
    write_json(seed_dir / "raw/crp_transitions.json", [])
    write_json(seed_dir / "raw/actions_handler_batches.json", [])
    write_text(seed_dir / "raw/ai_inference_times.csv", "timestamp,controller,model_variant,service_count,duration_ms,applicable\n,hpa,native,,,false\n")
    prometheus = {
        name: {"response": {"data": {"result": [{"values": [[0, "1"], [5, "1"], [10, "1"]]}]}}}
        for name in ("cpu_seconds", "memory_working_set_bytes", "pod_count")
    }
    write_json(seed_dir / "raw/prometheus_range_data.json", prometheus)
    write_json(seed_dir / "raw/measurement_window.json", {"started_at_unix": 0, "finished_at_unix": 10})
    write_json(seed_dir / "raw/pods_before.json", {})
    write_json(seed_dir / "raw/pods_after.json", {})
    write_json(seed_dir / "raw/deployments_after.json", {})
    write_json(seed_dir / "raw/crps_before.json", {"items": []})
    write_json(seed_dir / "raw/crps_after.json", {"items": []})
    write_text(seed_dir / "logs/runner.log", "ok\n")
    if zero_summary:
        metrics = "0,0,0,0,0,0"
    else:
        metrics = "0.0,0.0,3.0,1.0,15.0,190.0"
    write_text(
        seed_dir / "summaries/summary.csv",
        "seed,controller,workload,sla_latency_ms,sla_violation_rate,excess_response_time_seconds,cpu_seconds,mean_pod_count,pod_seconds,p95_latency_ms\n"
        f"{seed},hpa,gradual-ramp,1000,{metrics}\n",
    )
    return bundle


class ArtifactValidationTests(unittest.TestCase):
    def test_accepts_complete_measured_singleton_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root)
            rows, validation = validated_summary_rows(
                root,
                controllers={"hpa"},
                workloads={"gradual-ramp"},
                seeds={1},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["validation_status"], "validated")
            self.assertEqual(rows[0]["commit_hash"], "abc123")
            self.assertEqual(validation["validation_status"], "validated")

    def test_rejects_invalid_zero_valued_measured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root, zero_summary=True)
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("invalid_zero_run", {issue["code"] for issue in ctx.exception.issues})

    def test_zero_summary_guard_uses_tolerance(self) -> None:
        near_zero = {field: "1e-13" for field in ZERO_GUARD_FIELDS}
        meaningful = {field: "1e-13" for field in ZERO_GUARD_FIELDS}
        meaningful["p95_latency_ms"] = "1e-6"
        self.assertTrue(_summary_has_invalid_zeroes(near_zero))
        self.assertFalse(_summary_has_invalid_zeroes(meaningful))

    def test_rejects_duplicate_measured_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root, "run-a")
            measured_bundle(root, "run-b")
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("duplicate_run", {issue["code"] for issue in ctx.exception.issues})

    def test_rejects_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root)
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1, 2})
            self.assertIn("incomplete_matrix", {issue["code"] for issue in ctx.exception.issues})

    def test_rejects_stale_existing_generated_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_json(output / "artifact_validation_manifest.json", {"source_digest": "old"})
            with self.assertRaises(ArtifactValidationError):
                assert_artifacts_not_stale(output, "new", replace_stale=False)

    def test_generate_paper_artifacts_writes_validated_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir="results") as tmp:
            root = Path(tmp)
            measured_bundle(root)
            output = root / "artifacts"
            subprocess.run(
                [
                    sys.executable,
                    "experiments/generate_paper_artifacts.py",
                    "--results-root",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--allow-incomplete-matrix",
                ],
                check=True,
            )
            manifest = json.loads((output / "artifact_validation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["validation_status"], "validated")
            self.assertIn("commit_hash", (output / "main_comparison.csv").read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
