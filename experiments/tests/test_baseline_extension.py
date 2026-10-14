from __future__ import annotations

import csv
import fcntl
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.scripts import run_seed102_103_campaign as campaign


def complete_local_archive() -> bool:
    source = campaign.REPO_ROOT / "results/headlight-v2-live/seed101-103-aggregate-20261005/source-manifest.json"
    if not source.is_file():
        return False
    return all((campaign.REPO_ROOT / cell["output_dir"] / f"seed-{cell['seed']}" / "raw/workload_manifest_hashes.json").is_file()
               for cell in json.loads(source.read_text())["sources"] if cell["controller"] in {"hpa", "keda"})


class BaselineExtensionTests(unittest.TestCase):
    def test_guard_shutdown_kills_descendants_before_releasing_scope(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            root = Path(directory)
            child_pid_file = root / "child.pid"
            real_popen = subprocess.Popen
            waits = []
            guard_code = (
                "import signal,subprocess,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
                f"Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
                "root=Path(sys.argv[1]); (root/'ABORT').write_text('{}'); "
                "(root/'memory-guard-config.json').write_text('{}'); time.sleep(60)"
            )

            def launch(args, **kwargs):
                process = real_popen([sys.executable, "-c", guard_code, args[args.index("--output-dir") + 1]], **kwargs)
                real_wait = process.wait
                def wait(timeout=None):
                    waits.append(timeout)
                    if len(waits) <= 2:
                        raise subprocess.TimeoutExpired(args, timeout)
                    return real_wait(timeout=5)
                process.wait = wait
                return process

            with patch.multiple(campaign, CAMPAIGN_STATE_DIR=root,
                                _memory_guard=None, _memory_guard_dir=None), \
                 patch.object(campaign.subprocess, "Popen", side_effect=launch):
                with ExitStack() as stack:
                    campaign.start_memory_guard(stack)
                child_pid = int(child_pid_file.read_text())
            child_state = Path(f"/proc/{child_pid}/stat")
            deadline = time.monotonic() + 5
            while True:
                try:
                    state = child_state.read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    state = None
                if state in (None, "Z") or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            self.assertIn(state, (None, "Z"))
            self.assertEqual(waits[0], 90)

    def test_second_launcher_cannot_change_status_while_campaign_lock_is_held(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            root = Path(directory)
            status = root / "campaign-status.json"
            status.write_text('{"status":"running","sentinel":"preserve"}\n')
            with (root / "campaign.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run([
                    sys.executable, "-c",
                    "from experiments.lib import baseline_extension as b; "
                    "from experiments.scripts import run_seed102_103_campaign as c; "
                    "b.audit_baselines=lambda: {}; raise SystemExit(c.main())",
                    "--baseline-extension",
                    "--campaign-dir", directory, "--dry-run",
                ], cwd=campaign.REPO_ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(status.read_text()), {"status": "running", "sentinel": "preserve"})
            self.assertFalse((root / "dry-run-status.json").exists())

    def test_guard_exit_terminates_active_subprocess_and_classifies_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probes = []
            def guard_poll():
                probes.append(1)
                if len(probes) >= 2:
                    (root / "ABORT").write_text('{"abort":"low memory"}')
                return None
            with patch.multiple(campaign, BASELINE_EXTENSION=True,
                                _memory_guard=SimpleNamespace(poll=guard_poll), _memory_guard_dir=root):
                code, timed_out = campaign.stream_process(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    env=os.environ.copy(), log_path=root / "cell.log", timeout_seconds=120)
                self.assertEqual(code, 1)
                self.assertFalse(timed_out)
                self.assertEqual(campaign.classify_failure(root / "cell.log", code, timed_out), "host_resource_guard_abort")
                with self.assertRaisesRegex(RuntimeError, "before cell launch"):
                    campaign.stream_process([sys.executable, "-c", "print('must not start')"],
                        env=os.environ.copy(), log_path=root / "never-started.log", timeout_seconds=10)
                self.assertFalse((root / "never-started.log").exists())

    def test_extension_selects_twelve_cells_in_new_directories(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            with patch.multiple(campaign, SEEDS=campaign.SEEDS, CONTROLLERS=campaign.CONTROLLERS,
                                CAMPAIGN_STATE_DIR=campaign.CAMPAIGN_STATE_DIR,
                                BASELINE_EXTENSION=False):
                args = campaign.parse_args(["--baseline-extension", "--campaign-dir", directory])
                campaign.configure_campaign(args)
                self.assertEqual(campaign.SEEDS, (104, 105))
                self.assertEqual(campaign.CONTROLLERS, ("hpa", "keda"))
                self.assertEqual(len(campaign.SEEDS) * len(campaign.WORKLOADS) * len(campaign.CONTROLLERS), 12)
                output = campaign.attempt_dir(104, "gradual-ramp", "hpa", 1)
                self.assertTrue(output.is_relative_to(Path(directory)))
                self.assertNotIn("20261004", str(output))

    def test_extension_rejects_historical_output_directory(self):
        args = campaign.parse_args(["--baseline-extension", "--campaign-dir",
                                    "results/headlight-v2-live/qualification-seed102-103-20261004"])
        with self.assertRaisesRegex(ValueError, "historical"):
            campaign.configure_campaign(args)

    @unittest.skipUnless(complete_local_archive(), "requires the complete retained VM baseline archive")
    def test_baseline_audit_requires_historical_storage_and_valid_archive(self):
        from experiments.lib.baseline_extension import audit_baselines
        with patch.dict(os.environ, {"HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined"}):
            audit = audit_baselines()
        self.assertEqual(len(audit["sources"]), 18)
        self.assertFalse(audit["environment_equivalence_established"])
        with patch.dict(os.environ, {"HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "pod-emptydir"}):
            with self.assertRaisesRegex(ValueError, "image-defined"):
                audit_baselines()

    def test_summary_with_wrong_identity_is_never_accepted_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "seed-104/summaries/summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text("seed,controller,workload,technical_validity,performance_outcome\n105,hpa,diurnal,valid,measured\n")
            valid, reason = campaign.summary_result(root, 104, "unused", controller="hpa", workload="diurnal")
            self.assertFalse(valid)
            self.assertIn("identity", reason)

    def test_resume_accepts_poor_but_valid_without_accepting_invalid_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "seed-104/summaries/summary.csv"
            trace = root / "seed-104/raw/load_profile.csv"
            summary.parent.mkdir(parents=True)
            trace.parent.mkdir(parents=True)
            trace.write_text("timestamp,users\n0,1\n")
            expected_trace = campaign.sha256(trace)
            for validity, outcome, accepted in (
                ("valid", "measured", True),
                ("valid", "poor_but_valid", True),
                ("invalid", "poor_but_valid", False),
                ("valid", "unmeasured", False),
            ):
                with self.subTest(validity=validity, outcome=outcome):
                    summary.write_text(
                        "seed,controller,workload,technical_validity,performance_outcome\n"
                        f"104,hpa,diurnal,{validity},{outcome}\n"
                    )
                    valid, _ = campaign.summary_result(
                        root, 104, expected_trace, controller="hpa", workload="diurnal"
                    )
                    self.assertEqual(valid, accepted)
                    if accepted:
                        valid, reason = campaign.summary_result(
                            root, 104, "wrong-trace", controller="hpa", workload="diurnal"
                        )
                        self.assertFalse(valid)
                        self.assertTrue(reason.startswith("trace_mismatch:"))

    @unittest.skipUnless(complete_local_archive(), "requires the complete retained VM baseline archive")
    def test_dry_run_emits_twelve_matching_bundles_without_live_gate(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            env = {key: value for key, value in os.environ.items() if not key.startswith("HEADLIGHT_")}
            env["HEADLIGHT_TRAINTICKET_MONGO_STORAGE"] = "image-defined"
            result = subprocess.run([
                sys.executable,
                str(campaign.REPO_ROOT / "experiments/scripts/run_seed102_103_campaign.py"),
                "--baseline-extension", "--campaign-dir", directory, "--dry-run",
            ], cwd=campaign.REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads((Path(directory) / "campaign-plan.json").read_text())
            self.assertEqual(len(plan["cells"]), 12)
            self.assertEqual(plan["minimum_runtime_seconds"], 25680)
            for cell in plan["cells"]:
                root = campaign.REPO_ROOT / cell["output_dir"] / f"seed-{cell['seed']}"
                with (root / "summaries/summary.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["technical_validity"], "dry_run")
                self.assertEqual(row["controller"], cell["controller"])
                self.assertEqual(campaign.sha256(root / "raw/load_profile.csv"), campaign.EXPECTED_TRACES[cell["workload"]])

    def _preparation_command(self, args, **kwargs):
        if args[0] == "docker":
            value = '{"DockerRootDir":"/var/lib/docker","Driver":"overlay2"}'
        elif "--raw=/readyz" in args:
            value = "ok"
        elif args[0] == "findmnt":
            value = "ext4"
        elif "configmap" in args:
            config = "\n".join(
                f"server {cluster} {ip}:30080 check port 30021 weight 0"
                for _ in range(3)
                for cluster, ip in (("cluster2", "172.18.0.10"), ("cluster3", "172.18.0.11"))
            )
            value = json.dumps({"data": {"haproxy.cfg": config}})
        elif "exec" in args:
            self.assertEqual(kwargs["input"], "show stat\n")
            value = "# pxname,svname,status\n" + "".join(
                f"{backend},{member},UP\n"
                for backend in ("auth", "route", "travel")
                for member in ("cluster2", "cluster3")
            )
        elif "istio-ingressgateway" in args:
            value = json.dumps({"spec": {"ports": [
                {"port": 80, "nodePort": 30080}, {"port": 15021, "nodePort": 30021}
            ]}})
        elif "service" in args:
            value = json.dumps({"spec": {"type": "NodePort", "ports": [{"port": 9090, "nodePort": 32090}]}})
        elif "nodes" in args:
            cluster = args[args.index("--context") + 1]
            ip = "172.18.0.10" if cluster == "cluster2" else "172.18.0.11"
            value = json.dumps({"items": [{"status": {"addresses": [{"type": "InternalIP", "address": ip}],
                                                       "nodeInfo": {"kubeletVersion": "v1.32.0"}}}]})
        else:
            self.fail(f"unexpected external command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    def test_preparation_discovers_current_endpoints_and_writes_private_environment(self):
        from experiments.scripts.prepare_hpa_keda_campaign import prepare_environment

        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            kubeconfig = Path(directory) / "test-kubeconfig"
            kubeconfig.write_text("# External kubectl calls are replaced by test responses.\n")
            with patch.object(campaign, "LEGACY_STATE_DIR", Path(directory) / "shared-lock"), \
                 patch("experiments.scripts.prepare_hpa_keda_campaign.subprocess.run", side_effect=self._preparation_command), \
                 patch("experiments.scripts.prepare_hpa_keda_campaign.check_prometheus_ready"), \
                 patch("experiments.scripts.prepare_hpa_keda_campaign.audit_baselines", return_value={"sources": []}), \
                 patch.dict(os.environ, {"HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
                                         "HEADLIGHT_KUBECONFIG": str(kubeconfig),
                                         "HEADLIGHT_INTERNAL_KUBECONFIG": str(kubeconfig)}):
                prepare_environment(Path(directory))
            env_file = Path(directory) / "campaign-env.sh"
            env_text = env_file.read_text()
            self.assertIn("172.18.0.10:32090,http://172.18.0.11:32090", env_text)
            self.assertNotIn("HEADLIGHT_V2", env_text)
            self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)
            provenance = json.loads((Path(directory) / "environment-provenance.json").read_text())
            self.assertEqual(provenance["docker"]["DockerRootDir"], "/var/lib/docker")
            snapshot = campaign.REPO_ROOT / provenance["snapshot_path"]
            self.assertEqual(json.loads(snapshot.read_text()), provenance)

    def test_preparation_rejects_stale_or_unhealthy_gateway_before_writing_environment(self):
        from experiments.scripts.prepare_hpa_keda_campaign import prepare_environment

        for scenario, message in (
            ("stale", "gateway config is stale"),
            ("down", "route/cluster3=DOWN"),
            ("missing", "route/cluster3=missing"),
            ("exec-error", "cannot inspect HAProxy gateway health"),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
                root = Path(directory)
                kubeconfig = root / "test-kubeconfig"
                kubeconfig.write_text("# Mocked Kubernetes responses only.\n")

                def command(args, **kwargs):
                    result = self._preparation_command(args, **kwargs)
                    if scenario == "stale" and "configmap" in args:
                        result.stdout = result.stdout.replace("172.18.0.10", "172.18.0.99")
                    elif "exec" in args:
                        if scenario == "down":
                            result.stdout = result.stdout.replace("route,cluster3,UP", "route,cluster3,DOWN")
                        elif scenario == "missing":
                            result.stdout = result.stdout.replace("route,cluster3,UP\n", "")
                        elif scenario == "exec-error":
                            result.returncode, result.stderr = 1, "gateway unavailable"
                    return result

                with patch.object(campaign, "LEGACY_STATE_DIR", root / "shared-lock"), \
                     patch("experiments.scripts.prepare_hpa_keda_campaign.subprocess.run", side_effect=command), \
                     patch("experiments.scripts.prepare_hpa_keda_campaign.check_prometheus_ready"), \
                     patch("experiments.scripts.prepare_hpa_keda_campaign.audit_baselines", return_value={"sources": []}), \
                     patch.dict(os.environ, {"HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
                                             "HEADLIGHT_KUBECONFIG": str(kubeconfig),
                                             "HEADLIGHT_INTERNAL_KUBECONFIG": str(kubeconfig)}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        prepare_environment(root)
                self.assertFalse((root / "campaign-env.sh").exists())
                self.assertFalse((root / "environment-provenance.json").exists())

    def test_extension_stops_after_first_technical_failure(self):
        with tempfile.TemporaryDirectory(dir=campaign.REPO_ROOT / "results") as directory:
            args = campaign.parse_args(["--baseline-extension", "--campaign-dir", directory])
            with patch.multiple(campaign, SEEDS=campaign.SEEDS, CONTROLLERS=campaign.CONTROLLERS,
                                CAMPAIGN_STATE_DIR=campaign.CAMPAIGN_STATE_DIR, BASELINE_EXTENSION=False,
                                LEGACY_STATE_DIR=Path(directory) / "shared-lock"), \
                 patch.object(campaign, "parse_args", return_value=args), \
                 patch.object(campaign, "extension_plan", return_value={"cells": []}), \
                 patch.object(campaign, "validate_environment"), \
                 patch.object(campaign, "cleanup_after_failure", return_value=[]), \
                 patch.object(campaign, "start_memory_guard"), \
                 patch.object(campaign, "recovery_gate", return_value={"status": "passed"}), \
                 patch.object(campaign, "run_cell", return_value=(False, True)) as run_cell:
                self.assertEqual(campaign.main(), 1)
                self.assertEqual(run_cell.call_count, 1)
                status = json.loads((Path(directory) / "campaign-status.json").read_text())
                self.assertEqual(status["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
