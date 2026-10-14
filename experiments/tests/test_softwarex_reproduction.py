from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.softwarex_reproduction import execute_stage, load_config


class SoftwareXReproductionTests(unittest.TestCase):
    def test_loads_pinned_reproduction_config(self) -> None:
        config = load_config("experiments/config/softwarex_reproduction.yaml")
        self.assertEqual(config["schema_version"], "headlight.softwarex-reproduction/v1")
        self.assertEqual(config["seeds"]["workload"], 101)
        self.assertNotIn("latest", str(config["pinned_environment"]))

    def test_dry_run_collect_and_compact_output_are_labelled_dry_run(self) -> None:
        config = load_config("experiments/config/softwarex_reproduction.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            data = execute_stage(config, "collect-or-load-data", run_dir, dry_run=True)
            compact = execute_stage(config, "generate-compact-output", run_dir, dry_run=True)
            self.assertEqual(data["validation_status"], "dry-run")
            self.assertTrue((run_dir / "data_source.json").exists())
            self.assertTrue(Path(compact["path"]).exists())
            self.assertIn("dry-run", Path(compact["path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
