from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.softwarex_reproduction import (
    DRY_RUN_ONLY_MESSAGE,
    ReproductionConfigError,
    execute_stage,
    load_config,
)


class SoftwareXReproductionTests(unittest.TestCase):
    def test_loads_pinned_reproduction_config(self) -> None:
        config = load_config("experiments/config/softwarex_reproduction.yaml")
        self.assertEqual(config["schema_version"], "headlight.softwarex-reproduction/v1")
        self.assertEqual(config["seeds"]["workload"], 101)
        self.assertNotIn("latest", str(config["pinned_environment"]))

    def test_dry_run_collect_and_compact_output_are_labelled_unvalidated(self) -> None:
        config = load_config("experiments/config/softwarex_reproduction.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            data = execute_stage(config, "collect-or-load-data", run_dir, dry_run=True)
            compact = execute_stage(config, "generate-compact-output", run_dir, dry_run=True)
            self.assertEqual(data["validation_status"], "dry-run-unvalidated")
            self.assertTrue((run_dir / "data_source.json").exists())
            self.assertTrue(Path(compact["path"]).exists())
            self.assertIn("dry-run-unvalidated", Path(compact["path"]).read_text(encoding="utf-8"))

    def test_real_mode_stage_execution_is_rejected(self) -> None:
        config = load_config("experiments/config/softwarex_reproduction.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ReproductionConfigError, "dry-run/contract-only"):
                execute_stage(config, "generate-compact-output", Path(tmp), dry_run=False)

    def test_real_mode_cli_is_rejected(self) -> None:
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                "experiments/softwarex_reproduction.py",
                "--config",
                "experiments/config/softwarex_reproduction.yaml",
                "--stage",
                "all",
                "--run-id",
                "real-mode-test",
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(DRY_RUN_ONLY_MESSAGE, completed.stderr)
        self.assertIn("docs/reproduction.md", completed.stderr)


if __name__ == "__main__":
    unittest.main()
