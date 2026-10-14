"""Offline checks of Threshold's fixed policy, evidence, and deployment rendering."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-module/ai-module"))
spec = importlib.util.spec_from_file_location("threshold_baseline", ROOT / "ai-module/baselines/threshold_controller.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from headlight.features import BASE_FEATURES, MetricSample
from headlight.policy import ServiceRuntimeState


class ThresholdTests(unittest.TestCase):
    def test_streamed_actions_match_file_and_fixed_registered_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            action_log = Path(directory) / "actions.jsonl"
            controller = module.ThresholdController(
                str(ROOT / "ai-module/configs/service_profiles/advanced-ticket.json"),
                "http://fixture", "http://fixture/placements", "trainticket", str(action_log),
            )
            values = np.zeros((len(controller.profile.nodes), len(BASE_FEATURES)), dtype=np.float32)
            values[:, BASE_FEATURES.index("cpu_utilization")] = .95
            sample = MetricSample(123, values, np.zeros_like(values, dtype=bool),
                                  np.zeros_like(values, dtype=np.int64), np.zeros(len(values), dtype=bool))
            states = {node.name: ServiceRuntimeState({"cluster2": 1, "cluster3": 1},
                      {"cluster2": 1, "cluster3": 1}, None, None, False, False)
                      for node in controller.profile.nodes}
            controller.collector = SimpleNamespace(collect=lambda timestamp: sample)
            controller.state_reader = SimpleNamespace(read=lambda profile: states)
            session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=202, text=""))
            controller.actuator.session = session
            output = io.StringIO()
            with redirect_stdout(output):
                controller.run_once(123)
            recorded = [json.loads(line) for line in action_log.read_text().splitlines()]
            streamed = [json.loads(line.split("HEADLIGHT_ACTION_JSON ", 1)[1])
                        for line in output.getvalue().splitlines() if line.startswith("HEADLIGHT_ACTION_JSON ")]
            self.assertEqual(streamed, recorded)
            self.assertTrue(any(row["batch_attempted"] and row["batch_successful"] for row in recorded))
            self.assertTrue(all(row["desired_replicas"] == {"cluster2": 1, "cluster3": 2} for row in recorded))
            policy = yaml.safe_load((ROOT / "experiments/configs/baselines/threshold.yaml").read_text())["spec"]["policy"]
            expected = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(controller.policy_hash, expected)

    def test_setup_renders_selected_namespace_and_runtime_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kubectl = root / "kubectl"
            kubectl.write_text("#!/bin/sh\nexit 0\n")
            kubectl.chmod(0o755)
            config = root / "kubeconfig"
            config.write_text("# Offline test fixture\n")
            rendered = root / "deployment.yaml"
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "HEADLIGHT_TRAINTICKET_NAMESPACE": "threshold-smoke",
                   "HEADLIGHT_POD_KUBECONFIG": str(config),
                   "HEADLIGHT_THRESHOLD_RENDERED_MANIFEST": str(rendered),
                   "HEADLIGHT_THRESHOLD_IMAGE": "local/threshold:source-hash",
                   "HEADLIGHT_PROMETHEUS_URL": "http://fixture",
                   "HEADLIGHT_ACTION_HANDLER_URL": "http://fixture/placements"}
            result = subprocess.run(["bash", str(ROOT / "ai-module/threshold_controller_setup.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            deployment = yaml.safe_load(rendered.read_text())
            self.assertEqual(deployment["metadata"]["namespace"], "threshold-smoke")
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            variables = {row["name"]: row["value"] for row in container["env"]}
            self.assertEqual(variables["HEADLIGHT_NAMESPACE"], "threshold-smoke")
            self.assertEqual(container["image"], "local/threshold:source-hash")
