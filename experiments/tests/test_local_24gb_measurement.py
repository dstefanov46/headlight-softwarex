from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.run_local_24gb_measurement import (
    category,
    quantity_bytes,
    quantity_cpu,
    summarize_kubernetes_containers,
)
from experiments.scripts.local_measurement_watchdog import GIB, PressureState


class WatchdogTests(unittest.TestCase):
    def test_stops_load_before_testbed_on_sustained_pressure(self) -> None:
        state = PressureState()
        actions = []
        for _ in range(3):
            actions.extend(state.observe(2 * GIB, 3 * GIB, 3, 2))
        self.assertEqual(actions, ["stop_load"])
        self.assertFalse(state.testbed_stopped)
        self.assertEqual(state.observe(2 * GIB, 3 * GIB, 3, 2), [])
        self.assertEqual(state.observe(2 * GIB, 3 * GIB, 3, 2), ["stop_testbed"])

    def test_recovery_resets_consecutive_pressure(self) -> None:
        state = PressureState()
        self.assertEqual(state.observe(2 * GIB, 3 * GIB, 2, 2), [])
        self.assertEqual(state.observe(4 * GIB, 3 * GIB, 2, 2), [])
        self.assertEqual(state.observe(2 * GIB, 3 * GIB, 2, 2), [])


class MeasurementParsingTests(unittest.TestCase):
    def test_resource_quantities(self) -> None:
        self.assertEqual(quantity_cpu("12m"), 12)
        self.assertEqual(quantity_cpu("0.5"), 500)
        self.assertEqual(quantity_bytes("64Mi"), 64 * 1024**2)
        self.assertEqual(quantity_bytes("1Gi"), 1024**3)

    def test_container_categories(self) -> None:
        self.assertEqual(category("trainticket", "ts-auth"), "application")
        self.assertEqual(category("trainticket", "ts-auth-mongo"), "database")
        self.assertEqual(category("trainticket", "istio-proxy"), "sidecar")
        self.assertEqual(category("monitoring", "prometheus"), "monitoring")
        self.assertEqual(category("kube-system", "kube-apiserver"), "infrastructure")

    def test_per_container_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kubernetes-container-series.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["timestamp", "timestamp_iso", "phase", "cluster", "namespace", "pod", "container",
                     "cpu_millicores", "memory_bytes"]
                )
                writer.writerow(["1", "now", "idle", "cluster2", "trainticket", "ts-auth-a", "ts-auth", "10", "100"])
                writer.writerow(["2", "now", "idle", "cluster2", "trainticket", "ts-auth-a", "ts-auth", "25", "90"])
            summary = summarize_kubernetes_containers(Path(directory))["idle"][0]
            self.assertEqual(summary["cpu_peak_millicores"], 25)
            self.assertEqual(summary["memory_peak_bytes"], 100)
            self.assertEqual(summary["samples"], 2)


if __name__ == "__main__":
    unittest.main()
