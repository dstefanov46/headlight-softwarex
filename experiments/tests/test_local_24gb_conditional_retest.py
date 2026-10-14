from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.run_local_24gb_conditional_retest import (
    CRP_PATCH,
    ordered_queue,
    projected_extra_replicas,
    summarize_swap,
)


class ConditionalRetestTests(unittest.TestCase):
    def test_admission_queue_is_oldest_then_service_then_id(self) -> None:
        entries = [
            {"id": "z", "decision_epoch": 2.0, "service": "ts-route"},
            {"id": "b", "decision_epoch": 1.0, "service": "ts-travel"},
            {"id": "a", "decision_epoch": 1.0, "service": "ts-route"},
        ]
        self.assertEqual([item["id"] for item in ordered_queue(entries)], ["a", "b", "z"])

    def test_per_cluster_budget_counts_transition_target(self) -> None:
        current = {"ts-route": 1, "ts-travel": 1}
        self.assertEqual(projected_extra_replicas(current, "ts-route", 2), 1)
        current["ts-travel"] = 2
        self.assertEqual(projected_extra_replicas(current, "ts-route", 2), 2)

    def test_swap_summary_reports_phase_average_and_short_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swap.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "timestamp", "phase", "pswpin", "pswpout"])
                writer.writerow([0, "t0", "load", 10, 20])
                writer.writerow([5, "t1", "load", 10, 30])
                writer.writerow([10, "t2", "load", 10, 30])
            summary = summarize_swap(path)["load"]
            self.assertEqual(summary["average_total_pages_per_second"], 1.0)
            self.assertEqual(summary["peak_short_window_total_pages_per_second"], 2.0)

    def test_crp_patch_blocks_replica_update_until_binding_is_suspended(self) -> None:
        patch = CRP_PATCH.read_text(encoding="utf-8")
        wait = patch.index("Waiting for ResourceBinding to observe dispatch suspension")
        update = patch.index("Patching Deployment with new replicas") if "Patching Deployment with new replicas" in patch else len(patch)
        self.assertLess(wait, update)
        self.assertIn("TestScaleOutWaitsForResourceBindingDispatchSuspension", patch)
        self.assertIn("SchedulerObservedGeneration < binding.Generation", patch)


if __name__ == "__main__":
    unittest.main()
