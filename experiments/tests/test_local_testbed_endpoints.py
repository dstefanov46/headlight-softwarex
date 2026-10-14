from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from experiments.run_capacity_qualification import RESOURCE_PROFILES


REPO_ROOT = Path(__file__).resolve().parents[2]


class LocalTestbedEndpointTests(unittest.TestCase):
    def test_endpoint_scripts_are_syntactically_valid(self) -> None:
        scripts = [
            REPO_ROOT / "experiments" / "install_host_inotify_limits.sh",
            REPO_ROOT / "experiments" / "repair_local_testbed_endpoints.sh",
            REPO_ROOT / "experiments" / "rebuild_local_testbed.sh",
            REPO_ROOT / "experiments" / "start_paper_testbed.sh",
        ]
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)

    def test_rebuild_does_not_persist_raw_docker_ips(self) -> None:
        rebuild = (REPO_ROOT / "experiments" / "rebuild_local_testbed.sh").read_text(encoding="utf-8")
        self.assertNotIn(".NetworkSettings.Networks", rebuild)
        self.assertIn('https://${member}-control-plane:6443', rebuild)
        self.assertIn("repair_local_testbed_endpoints.sh", rebuild)

    def test_external_karmada_endpoint_is_localhost_tls(self) -> None:
        repair = (REPO_ROOT / "experiments" / "repair_local_testbed_endpoints.sh").read_text(encoding="utf-8")
        self.assertIn('https://localhost:${proxy_port}', repair)
        self.assertIn('127.0.0.1:${proxy_port}:32443', repair)
        self.assertIn("--security-opt no-new-privileges", repair)
        self.assertIn("--read-only", repair)

    def test_capacity_campaign_accepts_tested_profile(self) -> None:
        self.assertIn("local-24gb-tested", RESOURCE_PROFILES)


if __name__ == "__main__":
    unittest.main()
