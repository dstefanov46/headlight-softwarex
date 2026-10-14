from __future__ import annotations

import unittest

from experiments.run_headlight_v2_closed_loop import (
    PAIRED_METRICS,
    SUMMARY_METRICS,
    _mean_std,
    paired_rows,
    summarize,
)


def episode(controller: str, case: str, seed: int, offset: float) -> dict[str, object]:
    row: dict[str, object] = {
        "controller": controller,
        "case": case,
        "seed": seed,
        "trace_hash": f"{case}-{seed}",
        "failed": False,
    }
    row.update({metric: 10.0 + offset for metric in SUMMARY_METRICS})
    return row


class ClosedLoopV2ReportingTests(unittest.TestCase):
    def test_descriptive_summary_contains_median_and_interval(self) -> None:
        result = _mean_std((1.0, 2.0, 3.0))
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["median"], 2.0)
        self.assertLess(result["ci95_low"], result["mean"])
        self.assertGreater(result["ci95_high"], result["mean"])

    def test_summary_groups_each_workload(self) -> None:
        rows = [
            episode("headlight-v2", "gradual-ramp", 701, 0.0),
            episode("headlight-v2", "gradual-ramp", 702, 2.0),
        ]
        result = summarize(rows)
        metric = result["controllers"]["headlight-v2"]["cases"]["gradual-ramp"]
        self.assertEqual(metric["success_rate"]["count"], 2)
        self.assertEqual(metric["success_rate"]["median"], 11.0)

    def test_paired_differences_are_reported_overall_and_by_workload(self) -> None:
        rows = []
        for case in ("gradual-ramp", "flash-crowd"):
            for seed in (701, 702):
                rows.append(episode("headlight-v2", case, seed, 0.0))
                rows.append(episode("hpa-cpu-emulation", case, seed, 2.0))
        comparisons = paired_rows(rows)
        selected = {
            (row["workload"], row["metric"]): row
            for row in comparisons
            if row["comparator"] == "hpa-cpu-emulation"
        }
        self.assertEqual(selected[("overall", "success_rate")]["pairs"], 4)
        self.assertEqual(selected[("gradual-ramp", "success_rate")]["pairs"], 2)
        self.assertEqual(
            selected[("flash-crowd", "success_rate")]["median_difference"], 2.0
        )
        self.assertEqual(
            len(selected),
            len(PAIRED_METRICS) * 3,
        )


if __name__ == "__main__":
    unittest.main()
