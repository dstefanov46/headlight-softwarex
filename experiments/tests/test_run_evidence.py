"""Regression coverage for failures which previously lost live-run evidence."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from experiments.lib.run_evidence import (
    ControllerActionStream,
    GatewayWatch,
    RunEvidence,
    collect_failed_attempt,
    merge_controller_actions,
    run_logged_process,
)


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_command_health_failure_stops_a_blocked_process(self):
        started = time.monotonic()

        def unhealthy():
            raise RuntimeError("gateway updater exited: 1")

        with self.assertRaisesRegex(RuntimeError, "gateway updater exited"):
            run_logged_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                self.root / "wait.log",
                timeout=20,
                health_check=unhealthy,
            )
        self.assertLess(time.monotonic() - started, 3)

    def test_command_deadline_retains_output_and_terminates(self):
        path = self.root / "command.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            run_logged_process(
                [sys.executable, "-u", "-c", "import time; print('before outage'); time.sleep(30)"],
                path,
                timeout=0.3,
            )
        self.assertIn("before outage", path.read_text())

    @unittest.skipUnless(os.name == "posix", "POSIX process-group regression")
    def test_deadline_kills_children_after_the_command_leader_exits(self):
        child = "import os,time; pid=os.fork();\nif pid: os._exit(0)\nprint('GROUP='+str(os.getpgrp()),flush=True); time.sleep(30)"
        self._assert_descendant_deadline(child)

    def _assert_descendant_deadline(self, child):
        path = self.root / "child.log"
        outer = (
            "import subprocess,sys; from pathlib import Path; "
            "from experiments.lib.run_evidence import run_logged_process\n"
            f"try: run_logged_process([sys.executable,'-u','-c',{child!r}],Path({str(path)!r}),timeout=0.2)\n"
            "except subprocess.TimeoutExpired: print('deadline preserved')\n"
        )
        try:
            result = subprocess.run([sys.executable, "-c", outer], capture_output=True, text=True, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("deadline preserved", result.stdout)
        finally:
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.startswith("GROUP="):
                        try:
                            os.killpg(int(line.split("=", 1)[1]), signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    @unittest.skipUnless(os.name == "posix", "POSIX process-group regression")
    def test_deadline_applies_after_leader_exit_even_with_continuous_stdout(self):
        child = "import os; pid=os.fork();\nif pid: os._exit(0)\nprint('GROUP='+str(os.getpgrp()),flush=True)\nwhile True: print('still running',flush=True)"
        self._assert_descendant_deadline(child)

    def test_controller_decisions_survive_stream_exit_and_collection_failure(self):
        record = {"timestamp": 123, "controller": "headlight", "service": "ts-auth", "desired_replicas": {"cluster2": 2}}
        source = "HEADLIGHT_ACTION_JSON " + json.dumps(record)
        command = [sys.executable, "-u", "-c", f"print({source!r})"]
        stream = ControllerActionStream(self.root, command)
        stream.start()
        try:
            deadline = time.monotonic() + 3
            live = self.root / "raw/controller_actions_live.jsonl"
            while time.monotonic() < deadline and (not live.exists() or not live.read_text()):
                time.sleep(0.01)
            self.assertEqual(json.loads(live.read_text().splitlines()[0]), record)
        finally:
            stream.stop()
        merge_controller_actions(self.root, collected_text="")
        self.assertEqual(json.loads((self.root / "raw/controller_actions.jsonl").read_text()), record)

    def test_merge_preserves_live_and_final_records_without_duplicates(self):
        raw = self.root / "raw"
        raw.mkdir()
        first = '{"timestamp":1,"service":"auth","controller":"headlight"}'
        second = '{"timestamp":2,"service":"auth","controller":"headlight"}'
        (raw / "controller_actions_live.jsonl").write_text(first + "\n")
        (raw / "controller_actions.jsonl").write_text(first + "\n" + second + "\n")
        merge_controller_actions(self.root, collected_text=second + "\n")
        self.assertEqual([json.loads(line)["timestamp"] for line in (raw / "controller_actions.jsonl").read_text().splitlines()], [1, 2])

    def test_live_phases_and_invalid_outcome_are_persisted_before_return(self):
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        evidence.phase("preflight", "started_at_unix")
        phases = json.loads((self.root / "raw/phase_timestamps.json").read_text())
        self.assertIn("started_at_unix", phases["preflight"])
        evidence.fail(RuntimeError("Karmada API unavailable"))
        status = json.loads((self.root / "raw/attempt_status.json").read_text())
        self.assertEqual(status["technical_validity"], "invalid")
        self.assertEqual(status["failure_phase"], "preflight")
        self.assertIn("Karmada API unavailable", status["invalidity_reason"])

    def test_environment_records_actual_harness_bytes_even_with_a_git_commit(self):
        from experiments import run_experiment as runner
        with patch.object(runner.subprocess, "check_output", side_effect=["commit123\n", "cluster1\n"]):
            metadata = runner.environment(False)
        self.assertEqual(metadata["git_commit"], "commit123")
        name = "experiments/lib/run_evidence.py"
        self.assertEqual(metadata["harness_sources_sha256"][name], runner.file_sha256(runner.REPO_ROOT / name))

    def test_gateway_watch_rejects_an_exited_updater_and_stale_weights(self):
        class Process:
            code = None
            def poll(self):
                return self.code

        process = Process()
        watch = GatewayWatch(process, self.root / "weights.log", max_staleness=60)
        process.code = 1
        with self.assertRaisesRegex(RuntimeError, "updater exited"):
            watch.check()
        process.code = None
        watch.started_at = time.time() - 61
        with self.assertRaisesRegex(RuntimeError, "no verified update"):
            watch.check()

    def test_preflight_failure_is_invalid_without_losing_the_original_error(self):
        from experiments import run_experiment as runner
        from experiments.lib.config import resolve_run

        manifest = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))
        def fail_before_deployment(*args, **kwargs):
            raise RuntimeError("Karmada preflight unavailable")

        with patch.object(runner, "_execute_seed", side_effect=fail_before_deployment), patch.object(runner, "collect_failed_attempt", side_effect=RuntimeError("diagnostic API unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Karmada preflight unavailable"):
                runner.run_one_seed(manifest, self.root, 101, False)
        import csv
        summary = list(csv.DictReader((self.root / "seed-101/summaries/summary.csv").open()))[0]
        self.assertEqual(summary["technical_validity"], "invalid")
        self.assertIn("Karmada preflight unavailable", summary["invalidity_reason"])
        collection_error = json.loads((self.root / "seed-101/raw/failure_collection_error.json").read_text())
        self.assertIn("diagnostic API unavailable", collection_error["error"])

    def test_failure_collection_preserves_streamed_actions_when_api_is_unavailable(self):
        from experiments.lib.config import resolve_run
        manifest = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        evidence.action_stream = ControllerActionStream(self.root, [])
        (self.root / "raw/controller_actions_live.jsonl").write_text('{"timestamp":123,"service":"ts-auth","controller":"headlight"}\n')
        def failed_api(args, *unused, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="API unavailable")
        with patch("experiments.lib.run_evidence.run_logged_process", side_effect=failed_api):
            collect_failed_attempt(manifest, evidence, {})
        actions = (self.root / "raw/controller_actions.jsonl").read_text()
        self.assertEqual(json.loads(actions)["service"], "ts-auth")
        report = json.loads((self.root / "raw/failure_collection.json").read_text())
        self.assertEqual(report["stages"]["controller_actions"]["status"], "failed")
        self.assertFalse(report["complete"])

    def test_malformed_final_actions_do_not_replace_durable_evidence(self):
        raw = self.root / "raw"
        raw.mkdir()
        original = '{"timestamp":123,"service":"ts-auth","controller":"headlight"}\n'
        (raw / "controller_actions.jsonl").write_text(original)
        with self.assertRaisesRegex(ValueError, "malformed controller action"):
            merge_controller_actions(self.root, collected_text="corrupted record")
        self.assertEqual((raw / "controller_actions.jsonl").read_text(), original)

    def test_failure_after_warmup_retains_partial_measurement_and_phase(self):
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        with patch("experiments.lib.run_evidence.time.time", return_value=1000):
            evidence.begin_workload(1000)
        with patch("experiments.lib.run_evidence.time.time", return_value=1600):
            evidence.fail(RuntimeError("gateway failure"))
        window = json.loads((self.root / "raw/measurement_window.json").read_text())
        self.assertEqual(window["started_at_unix"], 1480)
        self.assertEqual(window["finished_at_unix"], 1600)
        self.assertEqual(evidence.status["failure_phase"], "measurement")

    def test_workload_window_preserves_multiphase_boundaries_and_steady_tail_mode(self):
        evidence = RunEvidence(
            self.root,
            warmup_seconds=480,
            measurement_seconds=1200,
            observation_tail_seconds=180,
            observation_tail_mode="hold_final_load",
            load_phases=[
                {"pattern": "gradual-ramp", "start_offset_seconds": 0, "end_offset_seconds": 405},
                {"pattern": "flash-crowd", "start_offset_seconds": 405, "end_offset_seconds": 795},
                {"pattern": "diurnal", "start_offset_seconds": 795, "end_offset_seconds": 1200},
            ],
        )

        evidence.begin_workload(1000)
        window = json.loads((self.root / "raw/measurement_window.json").read_text())

        self.assertEqual(window["observation_tail_mode"], "hold_final_load")
        self.assertEqual(window["load_phases"][0]["started_at_unix"], 1480)
        self.assertEqual(window["load_phases"][1]["started_at_unix"], 1885)
        self.assertEqual(window["load_phases"][2]["finished_at_unix"], 2680)

    def test_late_validation_failure_preserves_actual_workload_finish(self):
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        evidence.begin_workload(1000)
        evidence.finish_workload(2680)
        evidence.phase("collection_and_validation", "started_at_unix")
        with patch("experiments.lib.run_evidence.time.time", return_value=3000):
            evidence.fail(RuntimeError("validation failed"))
        window = json.loads((self.root / "raw/measurement_window.json").read_text())
        self.assertEqual(window["locust_finished_at_unix"], 2680)
        self.assertEqual(window["finished_at_unix"], 2680)

    def test_cleanup_failure_preserves_original_error_and_stops_host_recording(self):
        from experiments import run_experiment as runner
        from experiments.lib.config import resolve_run

        manifest = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))
        class BrokenStream:
            def stop(self):
                raise ValueError("malformed controller action evidence")

        def fail_workload(*args):
            evidence = args[-1]
            evidence.action_stream = BrokenStream()
            raise RuntimeError("original gateway failure")

        with patch.object(runner, "_execute_seed", side_effect=fail_workload), patch.object(runner, "collect_failed_attempt"), patch.object(RunEvidence, "start_host_recording"), patch.object(RunEvidence, "stop_host_recording") as stop_host:
            with self.assertRaisesRegex(RuntimeError, "original gateway failure"):
                runner.run_one_seed(manifest, self.root, 101, False)
            stop_host.assert_called_once()
        error = json.loads((self.root / "seed-101/raw/cleanup-error-controller_stream.json").read_text())
        self.assertIn("malformed controller action", error["error"])

    def test_malformed_controller_collection_does_not_skip_other_diagnostics(self):
        from experiments.lib.config import resolve_run
        manifest = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        evidence.action_stream = ControllerActionStream(self.root, [])
        def fake_command(args, *unused, **kwargs):
            output = "corrupted record" if "exec" in args else "{}"
            return subprocess.CompletedProcess(args, 0, stdout=output)
        with patch("experiments.lib.run_evidence.run_logged_process", side_effect=fake_command):
            collect_failed_attempt(manifest, evidence, {})
        stages = json.loads((self.root / "raw/failure_collection.json").read_text())["stages"]
        self.assertEqual(stages["controller_action_merge"]["status"], "failed")
        self.assertEqual(stages["kernel"]["status"], "collected")
        self.assertEqual(stages["state_cluster2"]["status"], "collected")

    def test_stalled_api_does_not_starve_host_or_healthy_member_diagnostics(self):
        from experiments.lib.config import resolve_run
        manifest = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))
        evidence = RunEvidence(self.root, warmup_seconds=480, measurement_seconds=1200)
        evidence.action_stream = ControllerActionStream(self.root, [])
        evidence.begin_workload(int(time.time()))
        def fake_command(args, *unused, **kwargs):
            if "--context" in args and args[args.index("--context") + 1] != "cluster2":
                time.sleep(kwargs["timeout"])
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            return subprocess.CompletedProcess(args, 0, stdout="{}")
        with patch.dict(os.environ, {"HEADLIGHT_FAILURE_COLLECTION_TIMEOUT_SECONDS": "0.25"}), patch("experiments.lib.run_evidence.run_logged_process", side_effect=fake_command):
            collect_failed_attempt(manifest, evidence, {"pod_count": "count(pods)"})
        stages = json.loads((self.root / "raw/failure_collection.json").read_text())["stages"]
        self.assertEqual(stages["kernel"]["status"], "collected")
        self.assertEqual(stages["docker"]["status"], "collected")
        self.assertEqual(stages["state_cluster2"]["status"], "collected")
        self.assertEqual(stages["prometheus_cluster2_pod_count"]["status"], "collected")


if __name__ == "__main__":
    unittest.main()
