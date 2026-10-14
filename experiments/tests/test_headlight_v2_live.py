from __future__ import annotations

import sys
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.quality import PREDICTIVE_CONTROLLERS
from experiments.run_experiment import parse_inference_timings
from experiments.validate_headlight_v2_live_configs import main as validate_live_configs


class HeadlightV2LiveTests(unittest.TestCase):
    def test_frozen_run_config_hashes(self) -> None:
        self.assertEqual(validate_live_configs(), 0)

    def test_v2_inference_timing_is_parsed_as_predictive(self) -> None:
        rows = parse_inference_timings(
            '{"event":"HEADLIGHT_V2_INFERENCE","service":"ts-auth","candidate_count":48,"duration_ms":12.5}',
            "headlight-v2",
            "full-action-conditioned",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["duration_ms"], "12.5")
        self.assertEqual(rows[0]["applicable"], "true")
        self.assertIn("headlight-v2", PREDICTIVE_CONTROLLERS)

    def test_vm_preflight_captures_stable_checksummed_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "bin"
            artifacts = root / "artifacts"
            binaries.mkdir()
            kubectl = binaries / "kubectl"
            kubectl.write_text("#!/bin/sh\necho fake-kubectl-ok\n", encoding="utf-8")
            kubectl.chmod(0o755)
            curl = binaries / "curl"
            curl.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *api/v1/query*) echo '{\"status\":\"success\",\"data\":{\"result\":[{\"value\":[0,\"18\"]}]}}' ;;\n"
                "  *) echo 'Prometheus is Ready.' ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            curl.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=f"{binaries}:{env['PATH']}",
                HEADLIGHT_PREFLIGHT_PROMETHEUS_URL="http://prometheus.test",
                HEADLIGHT_PREFLIGHT_ARTIFACT_DIR=str(artifacts),
            )
            result = subprocess.run(
                ["bash", str(REPO_ROOT / "experiments/scripts/preflight_headlight_v2_vm.sh")],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            report = next(artifacts.glob("preflight-*.log"))
            declared = Path(f"{report}.sha256").read_text(encoding="utf-8").split()[0]
            self.assertEqual(declared, hashlib.sha256(report.read_bytes()).hexdigest())
            self.assertIn("PREFLIGHT_OK", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
