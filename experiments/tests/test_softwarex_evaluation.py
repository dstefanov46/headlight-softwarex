"""The published matrix must be complete and backed by intact, valid summaries."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.validate_softwarex_evaluation import validate_evaluation


class SoftwareXEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        runs = []
        for controller in ("headlight-v2", "hpa", "keda", "threshold"):
            for workload in ("gradual-ramp", "flash-crowd", "diurnal"):
                for seed in range(101, 106):
                    path = root / f"{controller}-{workload}-{seed}.csv"
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["controller", "workload", "seed", "technical_validity", "performance_outcome", "p95_latency_ms", "sla_violation_rate", "pod_seconds"])
                        writer.writeheader()
                        writer.writerow(dict(controller=controller, workload=workload, seed=seed, technical_validity="valid", performance_outcome="measured", p95_latency_ms=150, sla_violation_rate=0.001, pod_seconds=50000))
                    runs.append(dict(controller=controller, workload=workload, seed=seed, technical_validity="valid", performance_outcome="measured", artifacts={"summary": dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)}))
        return {"runs": runs}

    def test_accepts_the_complete_valid_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_evaluation(self.fixture(root), root)
            self.assertEqual(result["validation_status"], "validated")
            self.assertEqual(result["validated_run_count"], 60)
            self.assertEqual(len(result["aggregates"]), 12)

    def test_rejects_missing_and_duplicate_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            index["runs"][-1] = index["runs"][0]
            with self.assertRaisesRegex(ValueError, "matrix"):
                validate_evaluation(index, root)

    def test_rejects_modified_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            (root / index["runs"][0]["artifacts"]["summary"]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_evaluation(index, root)

    def test_rejects_failed_run_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            index["runs"][0]["technical_validity"] = "invalid"
            with self.assertRaisesRegex(ValueError, "valid"):
                validate_evaluation(index, root)

    def test_rejects_dry_run_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            index["runs"][0]["performance_outcome"] = "dry-run"
            with self.assertRaisesRegex(ValueError, "measured"):
                validate_evaluation(index, root)

    def test_rejects_dry_run_environment_with_valid_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.fixture(root)
            path = root / "environment.json"
            path.write_text(json.dumps({"dry_run": True}))
            index["runs"][0]["artifacts"]["environment"] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "dry-run"):
                validate_evaluation(index, root)


if __name__ == "__main__":
    unittest.main()
