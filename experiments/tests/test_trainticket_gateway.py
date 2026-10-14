from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiments.lib.config import REPO_ROOT
from experiments.lib.capacity_contract import (
    CapacityContractError,
    build_capacity_contract,
    validate_capacity_contract,
    write_capacity_contract,
)
from experiments.lib.gateway_protocol import gateway_protocol_from_env
from experiments.scripts.gateway_weights import ready_endpoint_count


class GatewayWeightTests(unittest.TestCase):
    def test_counts_only_distinct_ready_pod_addresses(self) -> None:
        payload = {
            "items": [
                {"endpoints": [
                    {"addresses": ["10.0.0.1"], "conditions": {"ready": True}, "targetRef": {"kind": "Pod"}},
                    {"addresses": ["10.0.0.2"], "conditions": {"ready": False}, "targetRef": {"kind": "Pod"}},
                    {"addresses": ["10.0.0.3"], "conditions": {"ready": True}, "targetRef": {"kind": "Node"}},
                ]},
                {"endpoints": [
                    {"addresses": ["10.0.0.1"], "conditions": {"ready": True}, "targetRef": {"kind": "Pod"}},
                    {"addresses": ["10.0.0.4"], "conditions": {"ready": True}, "targetRef": {"kind": "Pod"}},
                ]},
            ]
        }
        self.assertEqual(ready_endpoint_count(payload), 2)

    def test_missing_or_unknown_readiness_is_not_routable(self) -> None:
        payload = {"items": [{"endpoints": [
            {"addresses": ["10.0.0.1"], "conditions": {}, "targetRef": {"kind": "Pod"}},
            {"addresses": ["10.0.0.2"], "conditions": {"ready": None}, "targetRef": {"kind": "Pod"}},
        ]}]}
        self.assertEqual(ready_endpoint_count(payload), 0)


class GatewayProtocolTests(unittest.TestCase):
    def test_protocol_is_opt_in_and_pinned_to_base_protocol(self) -> None:
        with patch.dict(os.environ, {"HEADLIGHT_GATEWAY_PROTOCOL_PATH": ""}):
            self.assertIsNone(gateway_protocol_from_env())
        with patch.dict(os.environ, {
            "HEADLIGHT_GATEWAY_PROTOCOL_PATH": "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
        }):
            protocol = gateway_protocol_from_env()
        self.assertEqual(protocol["spec"]["load_generator_cluster"], "cluster1")
        self.assertEqual(protocol["spec"]["workload_clusters"], ["cluster2", "cluster3"])
        self.assertEqual(protocol["reference"]["base_protocol_sha256"], protocol["spec"]["base_protocol_sha256"])

    def test_runner_dry_run_records_effective_gateway_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle"
            env = os.environ.copy()
            env["HEADLIGHT_GATEWAY_PROTOCOL_PATH"] = "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
            env["HEADLIGHT_EXPERIMENT_OUTPUT_DIR"] = str(output)
            subprocess.run(
                [sys.executable, "experiments/run_experiment.py", "--run",
                 "experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml",
                 "--seeds", "11", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            topology = json.loads((output / "seed-11/raw/topology.json").read_text(encoding="utf-8"))
            protocol = json.loads((output / "seed-11/raw/gateway_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(topology["load_generator_cluster"], "cluster1")
            self.assertEqual(topology["ingress_cluster"], "cluster1")
            self.assertEqual(topology["gateway_protocol"]["sha256"], protocol["reference"]["sha256"])
            self.assertEqual(topology["workload_manifest_hashes"]["trainticket_path"], "advanced-ticket")

    def test_runner_resolves_relative_kubeconfig_for_setup_scripts(self) -> None:
        env = os.environ.copy()
        env["KUBECONFIG"] = ".kube/headlight-gateway.config"
        output = subprocess.check_output(
            [sys.executable, "-c", "import os; import experiments.run_experiment; print(os.environ['KUBECONFIG'])"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
        ).strip()
        self.assertEqual(output, str(REPO_ROOT / ".kube/headlight-gateway.config"))

    def test_gateway_capacity_contract_cannot_be_used_as_old_contract(self) -> None:
        with patch.dict(os.environ, {
            "HEADLIGHT_GATEWAY_PROTOCOL_PATH": "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
        }):
            reference = gateway_protocol_from_env()["reference"]
        payload = build_capacity_contract(
            selected_capacity=4,
            seed=11,
            qualification_inputs={},
            environment={},
            pilot_results=[],
            workload_manifest_hashes={},
            gateway_protocol=reference,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity-contract.json"
            write_capacity_contract(path, payload)
            validated, _ = validate_capacity_contract(path, expected_gateway_protocol=reference)
            self.assertEqual(validated["topology"]["ingress_cluster"], "cluster1")
            with self.assertRaisesRegex(CapacityContractError, "gateway protocol reference"):
                validate_capacity_contract(path)

    def test_v2_gateway_run_requires_separate_output_and_marks_noncomparability(self) -> None:
        env = os.environ.copy()
        env["HEADLIGHT_GATEWAY_PROTOCOL_PATH"] = "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
        command = [
            sys.executable, "experiments/run_headlight_v2_live.py", "--run",
            "experiments/configs/live-v2/headlight-v2-gradual-ramp-seed-101.yaml", "--dry-run",
        ]
        refused = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("requires --output-dir distinct", refused.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            accepted = subprocess.run(
                [*command, "--output-dir", str(Path(tmp) / "gateway-v2")],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            manifest = yaml.safe_load((Path(tmp) / "gateway-v2/manifest.yaml").read_text(encoding="utf-8"))
            run = manifest["spec"]["run"]
            self.assertFalse(run["qualification_overrides"]["prior_comparator_traces_directly_comparable"])
            self.assertEqual(run["protocol"]["effective_version"], "research-v4-gateway-v1")


if __name__ == "__main__":
    unittest.main()
