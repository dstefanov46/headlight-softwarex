from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import yaml

from experiments.lib.config import REPO_ROOT, resolve_run


@contextmanager
def read_only_checkout_model_bundles():
    """Model a clean checkout, then apply the read-only state required by release validation."""
    roots = [REPO_ROOT / "results/calibration/headlight-bundle"]
    paths = [path for root in roots for path in (root, *root.rglob("*"))]
    paths.append(REPO_ROOT / "results/capacity-qualification/capacity-contract.json")
    original_modes = {path: stat.S_IMODE(path.stat().st_mode) for path in paths}
    try:
        for path in sorted(paths, key=lambda item: len(item.parts)):
            mode = original_modes[path] | stat.S_IWUSR
            if path.is_dir():
                mode |= stat.S_IXUSR
            path.chmod(mode)
        for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        yield
    finally:
        directories = [path for path in paths if path.is_dir()]
        for path in sorted(directories, key=lambda item: len(item.parts)):
            path.chmod(original_modes[path] | stat.S_IWUSR | stat.S_IXUSR)
        for path in paths:
            if not path.is_dir():
                path.chmod(original_modes[path])
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            path.chmod(original_modes[path])


class ThresholdEvaluationTests(unittest.TestCase):
    def test_selected_seed_set_expands_to_all_registered_workloads(self):
        from experiments.scripts import run_threshold_evaluation_campaign as campaign

        cells = campaign.campaign_cells_for_seeds([101, 102, 103])

        self.assertEqual(len(cells), 9)
        self.assertEqual({seed for seed, _ in cells}, {101, 102, 103})
        self.assertEqual(
            {(seed, workload) for seed, workload in cells},
            {
                (seed, workload)
                for seed in (101, 102, 103)
                for workload in ("gradual-ramp", "diurnal", "flash-crowd")
            },
        )

    def test_selected_campaign_cells_keep_evaluation_unmatched_metadata(self):
        from experiments import run_experiment as runner

        env = {
            "HEADLIGHT_EVALUATION_CAPACITY": "4",
            "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
            "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "GATEWAY_PROTOCOL", {"reference": {}}
        ), patch.dict(os.environ, env):
            for seed in (101, 102, 103):
                output = Path(directory) / f"seed-{seed}"
                manifest = resolve_run(
                    Path("experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml")
                )
                runner.configure_evaluation_trial(
                    manifest, [seed], output, None, campaign_cell=True
                )
                evidence = runner.evaluation_trial_metadata(manifest)
                self.assertEqual(evidence["seed"], seed)
                self.assertFalse(evidence["capacity_qualified"])
                self.assertFalse(evidence["matched_comparison_eligible"])

    def test_campaign_import_does_not_load_runner_before_prepared_environment(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; "
                "import experiments.scripts.run_threshold_evaluation_campaign; "
                "assert 'experiments.run_experiment' not in sys.modules"
            )],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_records_configured_single_cell_and_retains_exact_ramp_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial"
            env = dict(os.environ)
            env.update({"HEADLIGHT_EXPERIMENT_OUTPUT_DIR": str(output),
                        "HEADLIGHT_GATEWAY_PROTOCOL_PATH": "experiments/configs/gateway-protocols/research-v4-gateway.yaml",
                        "HEADLIGHT_EVALUATION_CAPACITY": "4",
                        "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
                        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
                        "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined"})
            with read_only_checkout_model_bundles():
                result = subprocess.run([sys.executable, "experiments/run_experiment.py", "--run",
                                         "experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml",
                                         "--seeds", "104", "--evaluation-trial", "--dry-run"],
                                        cwd=REPO_ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            seed = output / "seed-104"
            evidence = json.loads((seed / "raw/evaluation_trial.json").read_text())
            self.assertFalse(evidence["capacity_qualified"])
            self.assertFalse(evidence["matched_comparison_eligible"])
            self.assertEqual(evidence["seed"], 104)
            self.assertEqual(evidence["capacity_users"], 4)
            self.assertFalse((seed / "raw/capacity_contract.json").exists())
            self.assertFalse((seed / "raw/matched_baseline_comparison.json").exists())
            self.assertEqual(hashlib.sha256((seed / "raw/load_profile.csv").read_bytes()).hexdigest(),
                             "b1a3d85fce1cd8a4ce44bef382794c8c12fcb2647f389ae789a284cc19cb00ca")
            with (seed / "summaries/summary.csv").open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["experiment_scope"], "evaluation_trial")
            self.assertEqual(summary["matched_comparison_eligible"], "False")
            # Losing the evaluation marker must make the bundle unusable.
            from experiments.lib.quality import BundleQualityError, validate_bundle
            manifest = yaml.safe_load((output / "manifest.yaml").read_text())
            evidence["matched_comparison_eligible"] = True
            (seed / "raw/evaluation_trial.json").write_text(json.dumps(evidence))
            with self.assertRaises(BundleQualityError):
                validate_bundle(seed, manifest, dry_run=True)
            (seed / "raw/evaluation_trial.json").write_text(json.dumps(manifest["spec"]["run"]["evaluation_trial"]))
            del manifest["spec"]["run"]["evaluation_trial"]
            with self.assertRaises(BundleQualityError):
                validate_bundle(seed, manifest, dry_run=True)

    def test_evaluation_mode_rejects_matched_evidence_multiple_cells_and_existing_output(self):
        from experiments import run_experiment as runner
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "GATEWAY_PROTOCOL", {"reference": {}}), patch.dict(os.environ, {"HEADLIGHT_EVALUATION_CAPACITY": "4"}):
            output = Path(directory) / "trial"
            for seeds, matched in (([104, 105], None), ([105], None), ([104], Path("baseline.json"))):
                with self.assertRaises(ValueError):
                    runner.configure_evaluation_trial(resolve_run(Path("experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml")), seeds, output, matched)
            output.mkdir()
            (output / "keep.txt").write_text("existing evidence")
            with self.assertRaises(ValueError):
                runner.configure_evaluation_trial(resolve_run(Path("experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml")), [104], output, None)
            self.assertEqual((output / "keep.txt").read_text(), "existing evidence")

    def test_only_explicit_evaluation_mode_omits_live_capacity_contract(self):
        from experiments import run_experiment as runner
        from experiments.lib.capacity_contract import CapacityContractError
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "GATEWAY_PROTOCOL", {"reference": {}}), patch.dict(os.environ, {"HEADLIGHT_EVALUATION_CAPACITY": "4", "HEADLIGHT_CAPACITY_CONTRACT": str(Path(directory) / "missing.json"), "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket", "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm", "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined"}):
            manifest = resolve_run(Path("experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml"))
            with self.assertRaises(CapacityContractError):
                runner.prepare_capacity_contract(manifest, False, Path(directory))
            runner.configure_evaluation_trial(manifest, [104], Path(directory) / "trial", None)
            self.assertIsNone(runner.prepare_capacity_contract(manifest, False, Path(directory)))

    def test_evaluation_trial_rejects_changed_application_profile_or_ramp_timing(self):
        from experiments import run_experiment as runner
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "GATEWAY_PROTOCOL", {"reference": {}}), patch.dict(os.environ, {"HEADLIGHT_EVALUATION_CAPACITY": "4", "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket", "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "local-tiny", "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined"}):
            manifest = resolve_run(Path("experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml"))
            with self.assertRaises(ValueError):
                runner.configure_evaluation_trial(manifest, [104], Path(directory) / "trial", None)
            os.environ["HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE"] = "capacity-vm"
            manifest["spec"]["run"]["warmup_seconds"] = 0
            with self.assertRaises(ValueError):
                runner.configure_evaluation_trial(manifest, [104], Path(directory) / "trial", None)
