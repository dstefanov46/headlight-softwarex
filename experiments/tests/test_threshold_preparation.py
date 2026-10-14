from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ThresholdPreparationTests(unittest.TestCase):
    def test_manifest_retains_valid_attempt_and_verifies_snapshot(self):
        from experiments.scripts import prepare_threshold_baselines as preparation
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            campaign = repo / "campaign"
            snapshot = campaign / "environment-snapshots/preparation.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text('{"docker_filesystem":"ext4"}')
            reference = {"path": str(snapshot.relative_to(repo)),
                         "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
            cells = []
            for seed in (104, 105):
                for workload in ("gradual-ramp", "flash-crowd", "diurnal"):
                    for controller in ("hpa", "keda"):
                        parent = campaign / "cells" / f"{workload}-seed-{seed}"
                        output = parent / f"{controller}-attempt-2"
                        seed_dir = output / f"seed-{seed}"
                        (seed_dir / "raw").mkdir(parents=True)
                        (seed_dir / "summaries").mkdir()
                        trace = seed_dir / "raw/load_profile.csv"
                        trace.write_text("elapsed_seconds,users,spawn_rate\n0,1,15\n")
                        trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
                        summary = seed_dir / "summaries/summary.csv"
                        summary.write_text("seed,controller,workload,technical_validity,performance_outcome\n"
                                           f"{seed},{controller},{workload},valid,poor_but_valid\n")
                        status = {"cell": {"seed": seed, "controller": controller, "workload": workload},
                                  "status": "valid", "output_dir": str(output.relative_to(repo)),
                                  "environment_provenance": reference, "expected_trace_sha256": trace_hash}
                        (parent / "status").mkdir(exist_ok=True)
                        (parent / "status" / f"{controller}-attempt-2.json").write_text(json.dumps(status))
                        cells.append({**status["cell"], "status": "valid", "valid_output_dir": status["output_dir"]})
            (campaign / "campaign-status.json").write_text(json.dumps({"status": "complete", "cells": cells}))
            with patch.object(preparation, "REPO_ROOT", repo):
                manifest = preparation.build_manifest(campaign)
                self.assertEqual(len(manifest["sources"]), 12)
                self.assertTrue(all(row["output_dir"].endswith("attempt-2") for row in manifest["sources"]))
                self.assertTrue(all(row["performance_outcome"] == "poor_but_valid" for row in manifest["sources"]))
                snapshot.write_text('{}')
                with self.assertRaisesRegex(ValueError, "snapshot hash"):
                    preparation.build_manifest(campaign)

    def test_smoke_confirmation_requires_successful_controller_request_and_ready_placements(self):
        from experiments.scripts.smoke_threshold_crp import confirm_actuation
        actions = [{"controller": "threshold", "service": "ts-auth", "policy_reason": "threshold-scale-up",
                    "previous_replicas": {"cluster2": 1, "cluster3": 1},
                    "desired_replicas": {"cluster2": 1, "cluster3": 2},
                    "batch_attempted": True, "batch_successful": True, "batch_status_code": 202}]
        crp = {"spec": {"placements": [{"clusterName": "cluster2", "replicas": 1},
                                        {"clusterName": "cluster3", "replicas": 2}]}}
        deployments = {cluster: {"spec": {"replicas": replicas}, "status": {"readyReplicas": replicas}}
                       for cluster, replicas in (("cluster2", 1), ("cluster3", 2))}
        self.assertTrue(confirm_actuation(actions, crp, deployments))
        deployments["cluster3"]["status"]["readyReplicas"] = 1
        self.assertFalse(confirm_actuation(actions, crp, deployments))
        deployments["cluster3"]["status"]["readyReplicas"] = 2
        actions[0]["batch_successful"] = False
        self.assertFalse(confirm_actuation(actions, crp, deployments))
