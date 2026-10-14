from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.scripts import run_seed102_103_campaign as campaign
from experiments import validate_headlight_v2_live_configs as validation


class HeadlightV2ExtensionTests(unittest.TestCase):
    def test_extension_selects_only_six_v2_cells_and_fresh_attempts(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            with patch.multiple(campaign, SEEDS=campaign.SEEDS, CONTROLLERS=campaign.CONTROLLERS,
                                CAMPAIGN_STATE_DIR=campaign.CAMPAIGN_STATE_DIR,
                                BASELINE_EXTENSION=False, HEADLIGHT_V2_EXTENSION=False, create=True):
                args = campaign.parse_args(["--headlight-v2-extension", "--campaign-dir", directory,
                                            "--dry-run"])
                campaign.configure_campaign(args)
                with patch.dict(os.environ, {}, clear=True):
                    plan = campaign.extension_plan(True)
                identities = {(c["seed"], c["workload"], c["controller"]) for c in plan["cells"]}
                self.assertEqual(identities, {(seed, workload, "headlight-v2")
                    for seed in (104, 105) for workload in ("gradual-ramp", "diurnal", "flash-crowd")})
                for cell in plan["cells"]:
                    self.assertIn("experiments/run_headlight_v2_live.py", cell["command"])
                    self.assertTrue((campaign.REPO_ROOT / cell["output_dir"]).is_relative_to(Path(directory)))
                self.assertEqual(plan["minimum_runtime_seconds"], 12840)
                self.assertFalse(plan["environment_equivalence_established"])

    def test_selected_extension_config_rejects_changed_frozen_inputs(self):
        path = campaign.REPO_ROOT / "experiments/configs/headlight-v2-extension/headlight-v2-emergency-disabled-gradual-ramp-seed-104.yaml"
        self.assertTrue(path.is_file(), "seed-104 Headlight run config is missing")
        self.assertEqual(validation.validate_extension_config(path)["spec"]["seed"], 104)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "run.yaml"
            for key, value in (("seed", 106), ("policy_sha256", "wrong"),
                               ("mode", "reactive-emergency-enabled"), ("measurement_seconds", 60),
                               ("workload_profile_sha256", "wrong"),
                               ("expected_comparator_trace_sha256", "wrong"),
                               ("gateway_protocol_sha256", "wrong"),
                               ("capacity_users", 2), ("shadow_smoke_required", False)):
                with self.subTest(key=key):
                    data = yaml.safe_load(path.read_text())
                    data["spec"][key] = value
                    changed.write_text(yaml.safe_dump(data))
                    with self.assertRaises(ValueError):
                        validation.validate_extension_config(changed)

    def test_wrapper_generates_all_six_original_traces_without_live_resources(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            env = {key: value for key, value in os.environ.items() if not key.startswith("HEADLIGHT_")}
            result = subprocess.run([
                "bash", "experiments/scripts/run_headlight_v2_seed104_105.sh", directory, "--dry-run",
            ], cwd=campaign.REPO_ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = Path(directory)
            plan = json.loads((root / "campaign-plan.json").read_text())
            self.assertEqual(len(plan["cells"]), 6)
            hashes = {
                "gradual-ramp": "b1a3d85fce1cd8a4ce44bef382794c8c12fcb2647f389ae789a284cc19cb00ca",
                "diurnal": "ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2",
                "flash-crowd": "015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116",
            }
            for cell in plan["cells"]:
                output = campaign.REPO_ROOT / cell["output_dir"]
                trace = output / f"seed-{cell['seed']}/raw/load_profile.csv"
                self.assertEqual(hashlib.sha256(trace.read_bytes()).hexdigest(), hashes[cell["workload"]])
                manifest = yaml.safe_load((output / "manifest.yaml").read_text())["spec"]
                self.assertEqual(manifest["controller"]["controller"], "headlight-v2")
                self.assertEqual(manifest["run"]["seeds"], [cell["seed"]])
                self.assertFalse(manifest["run"]["qualification_overrides"]["prior_comparator_traces_directly_comparable"])
            self.assertFalse((root / "campaign-status.json").exists())
            self.assertFalse((root / "guards").exists())

    def test_wrapper_uses_system_python_when_checkout_has_no_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "experiments/scripts"
            scripts.mkdir(parents=True)
            wrapper = scripts / "run_headlight_v2_seed104_105.sh"
            shutil.copy2(campaign.REPO_ROOT / "experiments/scripts/run_headlight_v2_seed104_105.sh", wrapper)
            runner = scripts / "run_seed102_103_campaign.py"
            runner.write_text(
                "import sys\n"
                "print('PYTHON_FALLBACK_OK ' + ' '.join(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            env = {key: value for key, value in os.environ.items() if not key.startswith("HEADLIGHT_")}
            result = subprocess.run(
                ["bash", str(wrapper), "results/dry-run", "--dry-run"],
                cwd=root, env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PYTHON_FALLBACK_OK", result.stdout)
            self.assertIn("--headlight-v2-extension", result.stdout)

    def test_resume_skips_valid_v2_output_and_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            root = Path(directory)
            with patch.multiple(campaign, CAMPAIGN_STATE_DIR=root,
                                BASELINE_EXTENSION=False, HEADLIGHT_V2_EXTENSION=True, create=True):
                output = campaign.attempt_dir(104, "gradual-ramp", "headlight-v2", 1)
                self.assertTrue(output.is_relative_to(root))
                trace = output / "seed-104/raw/load_profile.csv"
                trace.parent.mkdir(parents=True)
                trace.write_text("timestamp,users\n0,1\n")
                summary = output / "seed-104/summaries/summary.csv"
                summary.parent.mkdir(parents=True)
                with summary.open("w") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["seed", "controller", "workload", "technical_validity", "performance_outcome"])
                    writer.writerow([104, "headlight-v2", "gradual-ramp", "valid", "poor_but_valid"])
                with patch.dict(campaign.EXPECTED_TRACES, {"gradual-ramp": campaign.sha256(trace)}):
                    self.assertEqual(campaign.valid_attempt(104, "gradual-ramp", "headlight-v2"), output)
                    status = campaign.status_path(104, "gradual-ramp", "headlight-v2", 1)
                    record = json.loads(status.read_text())
                    record["status"] = "invalid"
                    status.write_text(json.dumps(record))
                    self.assertIsNone(campaign.valid_attempt(104, "gradual-ramp", "headlight-v2"))
                    number, retry = campaign.next_attempt(104, "gradual-ramp", "headlight-v2")
                    self.assertEqual(number, 2)
                    self.assertEqual(retry.name, "headlight-v2-attempt-2")
                    self.assertTrue(summary.is_file())

    def test_v2_extension_rejects_historical_directories(self):
        for directory in ("results/headlight-v2-live/qualification-seed102-103-20261004",
                          "results/socc26", "results/capacity-qualification"):
            with self.subTest(directory=directory):
                args = campaign.parse_args(["--headlight-v2-extension", "--campaign-dir", directory])
                with self.assertRaisesRegex(ValueError, "historical"):
                    campaign.configure_campaign(args)

    def test_live_extension_requires_retained_v2_settings_before_any_output(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            with patch.multiple(campaign, SEEDS=campaign.SEEDS, CONTROLLERS=campaign.CONTROLLERS,
                                CAMPAIGN_STATE_DIR=campaign.CAMPAIGN_STATE_DIR,
                                BASELINE_EXTENSION=False, HEADLIGHT_V2_EXTENSION=False), \
                 patch.dict(os.environ, {}, clear=True):
                args = campaign.parse_args(["--headlight-v2-extension", "--campaign-dir", directory])
                campaign.configure_campaign(args)
                with self.assertRaisesRegex(ValueError, "retained Headlight settings"):
                    campaign.extension_plan(False)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_direct_extension_run_rejects_wrong_runtime_protocol_before_output(self):
        config = "experiments/configs/headlight-v2-extension/headlight-v2-emergency-disabled-gradual-ramp-seed-104.yaml"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "should-not-exist"
            env = {key: value for key, value in os.environ.items() if not key.startswith("HEADLIGHT_")}
            result = subprocess.run([sys.executable, "experiments/run_headlight_v2_live.py",
                                     "--run", config, "--output-dir", str(output), "--dry-run"],
                                    cwd=campaign.REPO_ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("v2 extension requires the shared gateway", result.stderr)
            self.assertFalse(output.exists())

    def test_resume_rejects_changed_campaign_inputs_before_recovery(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            root = Path(directory)
            snapshot = root / "campaign-inputs.json"
            retained = {"environment": {"image": "retained"}, "configs": [],
                        "seeds": [104, 105], "controllers": ["headlight-v2"]}
            snapshot.write_text(json.dumps(retained))
            args = campaign.parse_args(["--headlight-v2-extension", "--campaign-dir", directory, "--resume"])
            changed = {**retained, "environment": {"image": "changed"}}
            with patch.multiple(campaign, SEEDS=campaign.SEEDS, CONTROLLERS=campaign.CONTROLLERS,
                                CAMPAIGN_STATE_DIR=campaign.CAMPAIGN_STATE_DIR,
                                BASELINE_EXTENSION=False, HEADLIGHT_V2_EXTENSION=False,
                                LEGACY_STATE_DIR=root / "shared-lock"), \
                 patch.object(campaign, "parse_args", return_value=args), \
                 patch.object(campaign, "extension_plan", return_value=changed), \
                 patch.object(campaign, "validate_environment"):
                with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                    campaign.main()
                self.assertEqual(json.loads(snapshot.read_text()), retained)
                self.assertFalse((root / "initial-recovery.json").exists())
                self.assertFalse((root / "cells").exists())


if __name__ == "__main__":
    unittest.main()
