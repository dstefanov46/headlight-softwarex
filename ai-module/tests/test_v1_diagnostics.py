"""Offline checks of optional v1 recording; no Kubernetes or HTTP services."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-module/ai-module"))
sys.path.insert(0, str(ROOT))
from headlight.bundle import ModelBundle, load_checkpoint
from headlight.diagnostics_v1 import V1DiagnosticRecorder
from headlight.features import BASE_FEATURES, MetricSample, PrometheusMetricCollector
from headlight.models import build_model
from headlight.policy import ForecastPlacementPolicy, PlacementActuator, ServiceRuntimeState
from headlight.runtime import ControllerRuntime
from experiments.lib.run_evidence import ControllerActionStream


class NullOutput:
    def __init__(self):
        self.bytes = 0

    def write(self, value):
        self.bytes += len(value.encode())
        return len(value)

    def flush(self):
        pass


class Cluster:
    def __init__(self, profile):
        self.profile = profile
        self.placements = {n.name: {c: 1 for c in profile.cluster_limits} for n in profile.nodes}
        self.ready = copy.deepcopy(self.placements)
        self.lag = 0
        self.tick = 0
        self.calls = []

    def read(self, profile):
        self.tick += 1
        if self.lag:
            self.lag -= 1
        else:
            self.ready = copy.deepcopy(self.placements)
        return {n.name: ServiceRuntimeState(dict(self.placements[n.name]), dict(self.ready[n.name]),
            None, None, False, False, exists=self.tick % 19 != 17) for n in profile.nodes}

    def post(self, url, json, headers, timeout):
        self.calls.append(copy.deepcopy({"url": url, "json": json, "headers": headers, "timeout": timeout}))
        status = 500 if len(self.calls) == 1 else 202
        if status == 202:
            for entry in json:
                self.placements[entry["workload"]["name"]][entry["clusterName"]] = entry["replicas"]
            self.lag = 2
        return type("Response", (), {"status_code": status, "text": "simulated failure" if status == 500 else ""})()


class Samples:
    namespace = "trainticket"
    cpu_rate_window = "60s"
    observed_zero_error_rate = False
    forward_fill_seconds = 30

    def __init__(self, profile):
        self.profile = profile
        self.tick = 0
        self.observation_evidence = {}

    def collect(self, timestamp):
        self.tick += 1
        count = len(self.profile.nodes)
        values = np.zeros((count, len(BASE_FEATURES)), dtype=np.float32)
        values[:, 0] = 2
        values[:, 1] = .3 + np.arange(count)*.04 + .1*np.sin(self.tick/3)
        values[:, 2] = .015 + np.arange(count)*.002
        values[:, 3] = .95 if self.tick % 13 == 0 else .35
        values[:, 4] = 80
        values[:, 5] = 1200 if self.tick % 23 == 0 else 300
        missing = np.zeros_like(values, dtype=bool)
        ages = np.zeros_like(values, dtype=np.int64)
        stale = np.zeros(count, dtype=bool)
        if self.tick % 11 == 0:
            missing[0, 5] = True
            ages[0, 5] = 15  # Filled p95, not a fresh accuracy label.
        if self.tick % 17 == 0:
            missing[1, 1] = True
            values[1, 1] = np.nan
            ages[1, 1] = 60
            stale[1] = True
        self.observation_evidence = {}
        for fi, feature in enumerate(BASE_FEATURES):
            endpoints = []
            for cluster in self.profile.cluster_limits:
                result = []
                for ni, node in enumerate(self.profile.nodes):
                    if missing[ni, fi]:
                        continue
                    value = float(values[ni, fi]) / (2 if fi in [0, 1, 2] else 1)
                    result.append({"metric": {"destination_workload": node.name},
                        "value": [timestamp, str(value)]})
                endpoints.append({"endpoint": f"http://{cluster}", "cluster": cluster,
                    "status": "ok", "result": result})
            self.observation_evidence[feature] = {"timestamp": timestamp, "query": feature, "endpoints": endpoints}
        return MetricSample(int(timestamp), values, missing, ages, stale)


def load_model():
    torch.set_num_threads(2)
    bundle = ModelBundle.load(ROOT / "results/calibration/headlight-bundle")
    model = build_model(bundle.model_config, bundle.adjacency, torch.device("cpu"))
    model.load_state_dict(load_checkpoint(bundle, torch.device("cpu")), strict=True)
    model.eval()
    return bundle, model


def make_runtime(directory, bundle, model, enabled, *, limit=134217728, diagnostic_path=None, shadow=False):
    directory.mkdir(parents=True, exist_ok=True)
    cluster = Cluster(bundle.profile)
    env = {"HEADLIGHT_V1_DIAGNOSTIC_LOG": str(diagnostic_path or directory / "diagnostics.jsonl") if enabled else "",
        "HEADLIGHT_V1_DIAGNOSTIC_MAX_BYTES": str(limit), "HEADLIGHT_RUN_ID": "offline-verification",
        "HEADLIGHT_SEED": "21000"}
    with patch.dict(os.environ, env):
        runtime = ControllerRuntime(bundle, model, torch.device("cpu"), Samples(bundle.profile), cluster,
            ForecastPlacementPolicy(bundle.profile, bundle.capacity),
            PlacementActuator("http://simulated-actions/placements", "trainticket", session=cluster, shadow=shadow),
            "headlight", directory / "actions.jsonl")
    return runtime, cluster, env


def exercise(directory, bundle, model, enabled, *, steps=48, **kwargs):
    runtime, cluster, env = make_runtime(directory, bundle, model, enabled, **kwargs)
    summaries = []
    sink = NullOutput()
    with patch.dict(os.environ, env), redirect_stdout(sink):
        for tick in range(steps):
            result = runtime.run_once(1000000 + tick*15)
            summaries.append({"timestamp": result["timestamp"], "decisions": result["decisions"], "batch": result["batch"]})
    return runtime, cluster, summaries, sink


def outcome_module():
    spec = importlib.util.spec_from_file_location("v1_outcomes", ROOT / "experiments/scripts/analyze_v1_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.model = load_model()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def assert_parity(self, baseline, recorded):
        _, off_cluster, off, _ = baseline
        runtime, on_cluster, on, _ = recorded
        self.assertEqual(off, on)
        self.assertEqual(off_cluster.calls, on_cluster.calls)
        self.assertGreater(len(on_cluster.calls), 1)  # Includes failed request and retry.
        self.assertEqual((self.root / "off/actions.jsonl").read_text(), (self.root / "on/actions.jsonl").read_text())
        self.assertEqual((self.root / "off/proposals.jsonl").read_text(), (self.root / "on/proposals.jsonl").read_text())
        return runtime

    def test_real_checkpoint_predictions_actions_and_inputs_identical_on_off(self):
        off = exercise(self.root / "off", self.bundle, self.model, False)
        on = exercise(self.root / "on", self.bundle, self.model, True)
        runtime = self.assert_parity(off, on)
        records = [json.loads(l) for l in (self.root / "on/diagnostics.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 48*3)
        self.assertEqual([r["phase"] for r in records[:3]], ["observation", "decision", "actuation"])
        self.assertFalse(runtime.diagnostics.disabled)
        last = [r for r in records if r["phase"] == "decision"][-1]
        np.testing.assert_array_equal(np.asarray(last["scaled_model_input"], dtype=np.float32), runtime._input_tensor().numpy())
        self.assertEqual(last["input_timestamps"], list(range(1000000+24*15, 1000000+48*15, 15)))
        for name, value in last["services"].items():
            if value["prediction"]:
                f = on[2][-1]["decisions"][self.bundle.profile.node_order.index(name)].prediction
                if f is not None:
                    self.assertEqual(value["prediction"], f)
        self.assertTrue(any(r["phase"] == "observation" and any(any(m) for m in r["missing"]) for r in records))

    def test_cap_exhaustion_does_not_change_predictions_or_actions(self):
        off = exercise(self.root / "off", self.bundle, self.model, False)
        on = exercise(self.root / "on", self.bundle, self.model, True, limit=1)
        self.assertTrue(self.assert_parity(off, on).diagnostics.disabled)

    def test_unwritable_file_does_not_change_predictions_or_actions(self):
        directory = self.root / "is-a-directory"
        directory.mkdir()
        off = exercise(self.root / "off", self.bundle, self.model, False)
        on = exercise(self.root / "on", self.bundle, self.model, True, diagnostic_path=directory)
        self.assertTrue(self.assert_parity(off, on).diagnostics.disabled)

    def test_broken_stdout_error_reporting_is_nonfatal(self):
        recorder = V1DiagnosticRecorder(self.root / "record.jsonl")
        with patch("builtins.print", side_effect=BrokenPipeError("closed")):
            recorder.error(OSError("disk full"))
        self.assertTrue(recorder.disabled)

    def test_cadence_uses_identical_15_second_ticks_with_measured_work_cost(self):
        class Stop(BaseException):
            pass
        def run(enabled):
            runtime, cluster, env = make_runtime(self.root / ("on" if enabled else "off"), self.bundle, self.model, enabled)
            clock = [1000000.1]
            ticks, sleeps, costs = [], [], []
            original = runtime.run_once
            def step(timestamp):
                if len(ticks) == 40:
                    raise Stop()
                ticks.append(timestamp)
                started = time.perf_counter()
                result = original(timestamp)
                cost = time.perf_counter()-started
                costs.append(cost); clock[0] += cost
                return result
            def sleep(delay):
                self.assertGreaterEqual(delay, 0)
                sleeps.append(delay); clock[0] += delay
            with patch.dict(os.environ, env), redirect_stdout(NullOutput()), patch.object(runtime, "run_once", side_effect=step), patch("headlight.runtime.time.time", side_effect=lambda: clock[0]), patch("headlight.runtime.time.sleep", side_effect=sleep):
                with self.assertRaises(Stop):
                    runtime.run_forever()
            self.assertLess(max(costs), 15)
            self.assertEqual(len(sleeps), 41)
            return ticks, cluster.calls
        self.assertEqual(run(False), run(True))

    def test_endpoint_capture_preserves_metric_values_masks_and_query_count(self):
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"status": "success", "data": {"result": [{"metric": {"destination_workload": "ts-auth"}, "value": [1000000, "1.5"]}]}}
        class Session:
            def __init__(self):
                self.calls = []
            def get(self, url, timeout):
                self.calls.append((url, timeout)); return Response()
        samples, calls = [], []
        for enabled in [False, True]:
            collector = PrometheusMetricCollector("http://cluster2,http://cluster3", self.bundle.profile,
                capture_observations=enabled, cluster_names=("cluster2", "cluster3") if enabled else None)
            collector.session = Session()
            samples.append(collector.collect(1000000)); calls.append(collector.session.calls)
            if enabled:
                self.assertEqual(len(collector.observation_evidence), 7)
                self.assertTrue(all(len(v["endpoints"]) == 2 for v in collector.observation_evidence.values()))
        for name in ["values", "missing", "ages_seconds", "stale_nodes"]:
            np.testing.assert_array_equal(getattr(samples[0], name), getattr(samples[1], name))
        self.assertEqual(calls[0], calls[1]); self.assertEqual(len(calls[0]), 14)

    def test_offline_outcomes_exact_join_duplicate_merge_and_member_filter(self):
        exercise(self.root / "on", self.bundle, self.model, True)
        source = self.root / "on/diagnostics.jsonl"
        out = self.root / "out"
        with patch.object(sys, "argv", ["analyze", "--diagnostics", str(source), "--diagnostics", str(source), "--output-dir", str(out)]), redirect_stdout(NullOutput()):
            outcome_module().main()
        result = json.loads((out / "summary.json").read_text())
        self.assertEqual(result["unique_records"], 48*3)
        self.assertGreater(result["services"]["ts-auth"]["request_q90"]["primary_observed"], 0)
        self.assertIsNotNone(result["services"]["ts-auth"]["sla_violation_probability"]["brier"])
        records = [json.loads(l) for l in source.read_text().splitlines()]
        observation = next(r for r in records if r["phase"] == "observation")
        module = outcome_module()
        self.assertTrue(module.complete_members(observation, "request_rate", "ts-auth"))
        observation["endpoint_observations"]["request_rate"]["endpoints"].pop()
        self.assertFalse(module.complete_members(observation, "request_rate", "ts-auth"))

    def test_host_stream_retains_diagnostics_and_error_without_action_pollution(self):
        record = {"schema_version": "headlight.v1-diagnostics/v1", "timestamp": 1, "controller": "headlight", "phase": "observation"}
        error = {"timestamp": 2, "error": "recording disabled"}
        command = [sys.executable, "-u", "-c", "print('HEADLIGHT_V1_DIAGNOSTIC_JSON '+"+repr(json.dumps(record))+");print('HEADLIGHT_V1_DIAGNOSTIC_ERROR '+"+repr(json.dumps(error))+")"]
        stream = ControllerActionStream(self.root, command)
        stream.start()
        try:
            target = self.root / "raw/controller_diagnostics_live.jsonl"
            deadline = time.monotonic()+3
            while time.monotonic() < deadline:
                if target.exists() and len(target.read_text().splitlines()) >= 2:
                    break
                time.sleep(.01)
            self.assertEqual([json.loads(l) for l in target.read_text().splitlines()], [record, error])
            self.assertEqual((self.root / "raw/controller_actions_live.jsonl").read_text(), "")
        finally:
            stream.stop()
