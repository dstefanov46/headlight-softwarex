from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.lib.quality import (
    capacity_is_healthy,
    duration_is_covered,
    inference_timing_issues,
    is_success_status,
    percentile,
    prometheus_coverage_issues,
    restart_increases,
    select_safe_capacity,
)
from experiments.run_capacity_qualification import (
    CampaignFailure,
    CampaignOptions,
    CapacityCampaign,
    LOCUST_MODE_BY_PATH,
    PreflightFailure,
    ShutdownOnce,
    headlight_candidate_metadata,
    main,
    pilot_manifest,
    preflight_monitoring,
    preflight_testbed,
    qualification_manifest,
)
from experiments.run_experiment import install_controller


def summary(p95: float, success_rate: float = 1.0, exceptions: int = 0) -> dict[str, float | int]:
    return {
        "requests": 100,
        "successful": int(100 * success_rate),
        "success_rate": success_rate,
        "exceptions": exceptions,
        "p95_latency_ms": p95,
    }


class QualityGateTests(unittest.TestCase):
    def test_http_status_classification(self) -> None:
        self.assertTrue(is_success_status("200"))
        self.assertTrue(is_success_status(302))
        self.assertFalse(is_success_status("404"))
        self.assertFalse(is_success_status(""))

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)

    def test_relative_knee_and_safe_selection(self) -> None:
        self.assertEqual(capacity_is_healthy(summary(199), 100), (True, "healthy"))
        self.assertFalse(capacity_is_healthy(summary(201), 100)[0])
        self.assertEqual(select_safe_capacity([(5, True), (10, True), (20, False), (40, True)]), 10)
        self.assertIsNone(select_safe_capacity([(5, False)]))

    def test_duration_coverage(self) -> None:
        self.assertTrue(duration_is_covered(100, 190, 100))
        self.assertFalse(duration_is_covered(100, 189, 100))

    def test_prometheus_requires_full_window_and_samples(self) -> None:
        values = [[timestamp, "1"] for timestamp in range(100, 201, 10)]
        payload = {
            name: {"response": {"data": {"result": [{"values": values}]}}}
            for name in [
                "cpu_seconds",
                "memory_working_set_bytes",
                "pod_count",
            ]
        }
        self.assertEqual(prometheus_coverage_issues(payload, 100, 200, 10), [])
        payload["cpu_seconds"]["response"]["data"]["result"][0]["values"] = values[:2]
        self.assertTrue(any(issue.code == "prometheus_coverage" for issue in prometheus_coverage_issues(payload, 100, 200, 10)))

    def test_restart_detection(self) -> None:
        def pods(count: int) -> dict:
            return {"cluster2": {"items": [{"metadata": {"name": "pod-a"}, "status": {"containerStatuses": [{"name": "app", "restartCount": count}]}}]}}

        self.assertEqual(restart_increases(pods(1), pods(2)), {"cluster2/pod-a/app": 1})
        self.assertEqual(restart_increases(pods(2), pods(2)), {})

    def test_predictive_timing_requirement_and_na_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timings.csv"
            path.write_text("timestamp,controller,model_variant,service_count,duration_ms,applicable\n", encoding="utf-8")
            self.assertEqual(inference_timing_issues(path, "headlight")[0].code, "inference_timing")
            path.write_text(
                "timestamp,controller,model_variant,service_count,duration_ms,applicable\n,hpa,native,,,false\n",
                encoding="utf-8",
            )
            self.assertEqual(inference_timing_issues(path, "hpa"), [])
            path.write_text(
                "timestamp,controller,model_variant,service_count,duration_ms,applicable\n"
                ",headlight,headlight:legacy-uninstrumented,,,legacy-uninstrumented\n",
                encoding="utf-8",
            )
            self.assertEqual(inference_timing_issues(path, "headlight"), [])


class CampaignTests(unittest.TestCase):
    def run_campaign(self, summaries: list[dict], fail_controller: str | None = None) -> tuple[CapacityCampaign, list[tuple[str, int]]]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls: list[tuple[str, int]] = []

        def execute(manifest: dict, _output: Path, seed: int, _dry: bool) -> None:
            controller = manifest["spec"]["controller"]["controller"]
            calls.append((manifest["metadata"]["name"], seed))
            if fail_controller and manifest["metadata"]["name"].startswith(f"qualification-{fail_controller}"):
                raise RuntimeError("controller failed")

        iterator = iter(summaries)
        campaign = CapacityCampaign(
            CampaignOptions(Path(tmp.name), cooldown_seconds=0),
            execute=execute,
            summarize_pilot=lambda _path: next(iterator),
        )
        return campaign, calls

    def test_all_pilots_and_four_qualifications_complete(self) -> None:
        campaign, calls = self.run_campaign([summary(100), summary(150), summary(180), summary(190)])
        campaign.run()
        self.assertEqual(campaign.selected_capacity, 10)
        self.assertEqual(len(calls), 8)
        self.assertCountEqual(campaign.controller_order, ["headlight", "lstm", "keda", "hpa"])

    def test_knee_uses_last_safe_level(self) -> None:
        campaign, calls = self.run_campaign([summary(100), summary(180), summary(201)])
        campaign.run()
        self.assertEqual(campaign.selected_capacity, 6)
        qualification_calls = [name for name, _seed in calls if name.startswith("qualification-")]
        self.assertEqual(len(qualification_calls), 4)

    def test_skip_pilots_uses_selected_capacity(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls: list[str] = []

        def execute(manifest: dict, _output: Path, _seed: int, _dry: bool) -> None:
            calls.append(manifest["metadata"]["name"])

        campaign = CapacityCampaign(
            CampaignOptions(Path(tmp.name), cooldown_seconds=0, skip_pilots=True, selected_capacity_override=5),
            execute=execute,
        )
        campaign.run()
        self.assertEqual(campaign.selected_capacity, 5)
        self.assertTrue(all(name.startswith("qualification-") for name in calls))
        self.assertEqual(len(calls), 4)

    def test_controller_subset_limits_qualifications(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls: list[str] = []

        def execute(manifest: dict, _output: Path, _seed: int, _dry: bool) -> None:
            calls.append(manifest["metadata"]["name"])

        campaign = CapacityCampaign(
            CampaignOptions(
                Path(tmp.name),
                cooldown_seconds=0,
                skip_pilots=True,
                selected_capacity_override=5,
                controllers=("hpa",),
            ),
            execute=execute,
        )
        campaign.run()
        self.assertEqual(calls, ["qualification-hpa-seed-101"])
        self.assertEqual(campaign.controller_order, ["hpa"])

    def test_seed_and_headlight_candidate_metadata_are_persisted(self) -> None:
        metadata = headlight_candidate_metadata(
            {
                "HEADLIGHT_CANDIDATE": "h4",
                "HEADLIGHT_STEP_NUM_IN": "6",
                "HEADLIGHT_NUM_BLOCKS": "4",
                "HEADLIGHT_LEARNING_RATE": "0.0001",
            }
        )
        manifest = qualification_manifest(
            "headlight",
            5,
            300,
            Path("results/example"),
            seed=202,
            candidate_metadata=metadata,
        )
        overrides = manifest["spec"]["run"]["qualification_overrides"]
        self.assertEqual(manifest["metadata"]["name"], "qualification-headlight-seed-202")
        self.assertEqual(manifest["spec"]["run"]["seeds"], [202])
        self.assertEqual(overrides["seed"], 202)
        self.assertEqual(
            overrides["headlight_candidate"],
            {"candidate": "h4", "step_num_in": 6, "num_blocks": 4, "learning_rate": 0.0001},
        )

    def test_baseline_failure_skips_qualifications(self) -> None:
        campaign, calls = self.run_campaign([summary(1001)])
        with self.assertRaises(CampaignFailure):
            campaign.run()
        self.assertEqual(len(calls), 1)

    def test_controller_failure_preserves_stage(self) -> None:
        campaign, _calls = self.run_campaign([summary(100)] * 4, fail_controller="lstm")
        with self.assertRaisesRegex(RuntimeError, "controller failed"):
            campaign.run()
        self.assertEqual(campaign.completed_stage, "qualification-lstm-starting")

    def test_qualification_contract_is_not_paper_evidence(self) -> None:
        manifest = qualification_manifest("hpa", 20, 900, Path("results/example"))
        overrides = manifest["spec"]["run"]["qualification_overrides"]
        self.assertFalse(overrides["publishable_as_softwarex_five_seed_result"])
        self.assertEqual(manifest["spec"]["topology"]["active_clusters"], ["cluster2", "cluster3"])

    def test_campaign_defaults_to_advanced_ticket_contract(self) -> None:
        manifest = pilot_manifest(20, 180, Path("results/example"))
        override = manifest["spec"]["run"]["qualification_overrides"]
        self.assertEqual(override["trainticket_path"], "advanced-ticket")
        self.assertEqual(override["transaction_mix"], "advanced-ticket")
        self.assertEqual(override["resource_profile"], "capacity-vm")
        self.assertEqual(manifest["spec"]["workload"]["locust"]["start_users"], 20)
        self.assertEqual(manifest["spec"]["workload"]["locust"]["end_users"], 20)

    def test_train_ticket_path_maps_to_locust_mode(self) -> None:
        self.assertEqual(LOCUST_MODE_BY_PATH["full"], "full-session")
        self.assertEqual(LOCUST_MODE_BY_PATH["login-route"], "login-route")
        self.assertEqual(LOCUST_MODE_BY_PATH["advanced-ticket"], "advanced-ticket")
        full_manifest = qualification_manifest("headlight", 20, 900, Path("results/example"), "full")
        login_manifest = qualification_manifest("headlight", 20, 900, Path("results/example"), "login-route")
        self.assertEqual(full_manifest["spec"]["run"]["qualification_overrides"]["transaction_mix"], "full-session")
        self.assertEqual(login_manifest["spec"]["run"]["qualification_overrides"]["transaction_mix"], "login-route")
        self.assertEqual(
            full_manifest["spec"]["controller"]["service_profile"],
            "ai-module/configs/service_profiles/full.json",
        )
        self.assertEqual(
            login_manifest["spec"]["controller"]["service_profile"],
            "ai-module/configs/service_profiles/login-route.json",
        )

    def test_headlight_controller_setup_gets_bootstrap_workload_shape(self) -> None:
        manifest = qualification_manifest("headlight", 20, 300, Path("results/example"), "advanced-ticket")
        captured: dict[str, str] = {}

        def fake_run_cmd(_command: list[str], _log_path: Path, env: dict[str, str], **_kwargs) -> None:
            captured.update(env)

        with tempfile.TemporaryDirectory() as tmp, patch("experiments.run_experiment.run_cmd", side_effect=fake_run_cmd):
            install_controller(manifest, Path(tmp))

        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_TRAINTICKET_MODE"], "advanced-ticket")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_DURATION_SECONDS"], "300")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_START_USERS"], "12")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_END_USERS"], "32")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_RAMP_SECONDS"], "225")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_HOLD_SECONDS"], "75")
        self.assertEqual(captured["HEADLIGHT_BOOTSTRAP_SPAWN_RATE"], "2")

    def test_status_records_slice_profile_and_order(self) -> None:
        campaign = CapacityCampaign(
            CampaignOptions(
                Path("results/example"),
                trainticket_path="full",
                resource_profile="source",
                candidate_metadata={"candidate": "h0", "step_num_in": 3},
            )
        )
        campaign.selected_capacity = 20
        campaign.controller_order = ["hpa", "headlight"]
        status = campaign.status(True, 0)
        self.assertEqual(status["trainticket_path"], "full")
        self.assertEqual(status["locust_mode"], "full-session")
        self.assertEqual(status["resource_profile"], "source")
        self.assertEqual(status["selected_capacity"], 20)
        self.assertEqual(status["controller_order"], ["hpa", "headlight"])
        self.assertEqual(status["headlight_candidate"], {"candidate": "h0", "step_num_in": 3})

    def test_shutdown_disabled_and_fallback(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, stdout="failed")

        disabled = ShutdownOnce(False, runner)
        disabled.request(lambda _message: None)
        self.assertEqual(calls, [])
        enabled = ShutdownOnce(True, runner)
        enabled.request(lambda _message: None)
        enabled.request(lambda _message: None)
        self.assertEqual(calls, [["systemctl", "poweroff"], ["/usr/sbin/shutdown", "-h", "now"]])

    def test_preflight_reports_stale_member_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kubeconfig = Path(tmp) / "config"
            kubeconfig.write_text(
                """
apiVersion: v1
clusters:
- name: kind-karmada-apiserver
  cluster:
    server: https://127.0.0.1:10000
- name: kind-karmada-host
  cluster:
    server: https://127.0.0.1:10001
- name: kind-cluster1
  cluster:
    server: https://127.0.0.1:10002
- name: kind-cluster2
  cluster:
    server: https://127.0.0.1:10003
- name: kind-cluster3
  cluster:
    server: https://127.0.0.1:10004
contexts:
- name: karmada-apiserver
  context:
    cluster: kind-karmada-apiserver
- name: karmada-host
  context:
    cluster: kind-karmada-host
- name: cluster1
  context:
    cluster: kind-cluster1
- name: cluster2
  context:
    cluster: kind-cluster2
- name: cluster3
  context:
    cluster: kind-cluster3
""",
                encoding="utf-8",
            )

            def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["kubectl", "--kubeconfig", str(kubeconfig)] and command[-3:] == ["get-contexts", "-o", "name"]:
                    return subprocess.CompletedProcess(command, 0, stdout="karmada-apiserver\nkarmada-host\ncluster1\ncluster2\ncluster3\n")
                if command[:2] == ["docker", "ps"]:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "karmada-host-control-plane 127.0.0.1:10001->6443/tcp\n"
                            "cluster1-control-plane 127.0.0.1:10002->6443/tcp\n"
                            "cluster2-control-plane 127.0.0.1:20003->6443/tcp\n"
                            "cluster3-control-plane 127.0.0.1:10004->6443/tcp\n"
                        ),
                    )
                return subprocess.CompletedProcess(command, 0, stdout="ok")

            with self.assertRaisesRegex(PreflightFailure, "stale Kind API ports.*rebuild_local_testbed.sh"):
                preflight_testbed(kubeconfig, runner)

    def test_preflight_monitoring_reports_missing_prometheus(self) -> None:
        def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
            self.assertEqual(
                command,
                [
                    "kubectl",
                    "--context",
                    "cluster2",
                    "-n",
                    "monitoring",
                    "get",
                    "service",
                    "prometheus-kube-prometheus-prometheus",
                ],
            )
            return subprocess.CompletedProcess(command, 1, stdout='Error from server (NotFound): services "prometheus" not found')

        with self.assertRaisesRegex(PreflightFailure, "install_prometheus_monitoring.sh"):
            preflight_monitoring(runner)

    def test_preflight_failure_stops_before_campaign_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "campaign_status.json"
            with patch.object(sys, "argv", ["run_capacity_qualification.py", "--smoke", "--output-dir", tmp]), \
                patch.dict(os.environ, {"HEADLIGHT_SHUTDOWN_ON_EXIT": "1"}, clear=False), \
                patch("experiments.run_capacity_qualification.preflight_testbed", side_effect=PreflightFailure("stale Kind API ports; run ./experiments/rebuild_local_testbed.sh")), \
                patch("experiments.run_capacity_qualification.preflight_monitoring"), \
                patch.object(CapacityCampaign, "run") as campaign_run, \
                patch("experiments.run_capacity_qualification.capture_failure_diagnostics"), \
                patch("experiments.run_capacity_qualification.subprocess.run", return_value=subprocess.CompletedProcess(["sync"], 0, stdout="")):
                self.assertEqual(main(), 1)
            campaign_run.assert_not_called()
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["completed_stage"], "preflight")
            self.assertEqual(status["trainticket_path"], "advanced-ticket")


if __name__ == "__main__":
    unittest.main()
