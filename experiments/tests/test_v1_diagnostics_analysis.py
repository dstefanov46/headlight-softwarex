from __future__ import annotations

import unittest

from experiments.scripts.analyze_v1_diagnostics import scored_decisions


class V1DiagnosticAnalysisTests(unittest.TestCase):
    def test_predictions_at_or_after_scored_window_end_are_excluded(self) -> None:
        records = [
            {"phase": "decision", "timestamp": 100},
            {"phase": "observation", "timestamp": 280},
            {"phase": "decision", "timestamp": 280},
            {"phase": "decision", "timestamp": 300},
        ]

        self.assertEqual(scored_decisions(records, 280), [records[0]])


if __name__ == "__main__":
    unittest.main()
