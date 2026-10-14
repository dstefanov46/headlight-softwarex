from __future__ import annotations

import time
import unittest
import importlib.util
import json
import threading
import tempfile
import csv
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import requests
from requests import Timeout

SCRIPT = Path(__file__).resolve().parents[1] / "testbed" / "trainticket-local-measurement" / "open_loop_load.py"
SPEC = importlib.util.spec_from_file_location("headlight_open_loop_load", SCRIPT)
assert SPEC and SPEC.loader
open_loop_load = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(open_loop_load)

from experiments import run_local_24gb_corrected_retest as corrected


class OpenLoopLoadTests(unittest.TestCase):
    def test_swap_summary_assigns_phase_boundary_interval(self) -> None:
        fields = corrected.DiagnosticSampler.HOST_FIELDS
        rows = []
        for epoch, phase, pin, pout in ((0, "a", 10, 20), (5, "a", 15, 25), (10, "b", 25, 35)):
            row = {field: 0 for field in fields}
            row.update({
                "epoch": epoch, "timestamp": str(epoch), "phase": phase,
                "mem_total_bytes": 1000, "mem_available_bytes": 500,
                "swap_used_bytes": 100, "pswpin": pin, "pswpout": pout,
            })
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            summary = corrected.summarize_swap(path)
        self.assertEqual(1.0, summary["a"]["swap_in_pages_per_second"])
        self.assertEqual(2.0, summary["b"]["swap_in_pages_per_second"])
        self.assertEqual(2.0, summary["b"]["swap_out_pages_per_second"])
        self.assertEqual(10, summary["b"]["pages_in"])

    def test_local_http_fixture_authenticates_and_accounts_arrivals(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if self.path.endswith("/api/v1/users/login"):
                    body = json.dumps({"data": {"token": "fixture-token"}}).encode()
                else:
                    body = b'{"status":1}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            token, authentication = open_loop_load.authenticate(base_url, 1.0)
            self.assertEqual("fixture-token", token)
            self.assertEqual(200, authentication["status_code"])

            def request() -> requests.Response:
                return requests.post(f"{base_url}/travel", json={"fixture": True}, timeout=1.0)

            _, summary = open_loop_load.run_fixed_arrival(
                phase="http-fixture",
                rate_rps=20.0,
                duration_seconds=0.25,
                max_concurrency=2,
                queue_capacity=2,
                request_timeout_seconds=1.0,
                drain_timeout_seconds=1.0,
                request=request,
            )
            self.assertEqual(5, summary["scheduled_arrivals"])
            self.assertEqual(5, summary["success_requests"])
            self.assertEqual(0, summary["outstanding_at_drain_end"])
        finally:
            server.shutdown()
            server.server_close()

    def test_fixed_arrivals_are_fully_accounted(self) -> None:
        def request() -> SimpleNamespace:
            time.sleep(0.005)
            return SimpleNamespace(status_code=200)

        records, summary = open_loop_load.run_fixed_arrival(
            phase="fixture",
            rate_rps=20.0,
            duration_seconds=0.25,
            max_concurrency=2,
            queue_capacity=2,
            request_timeout_seconds=1.0,
            drain_timeout_seconds=1.0,
            request=request,
        )
        self.assertEqual(5, summary["scheduled_arrivals"])
        self.assertEqual(5, summary["success_requests"])
        self.assertEqual(5, summary["accounted_arrivals"])
        self.assertTrue(summary["accounting_balanced"])
        self.assertTrue(all(row["scheduled_epoch"] is not None for row in records))
        self.assertTrue(all(row["actual_start_epoch"] is not None for row in records))
        self.assertTrue(all(row["completed_epoch"] is not None for row in records))

    def test_queue_saturation_is_an_explicit_delivery_failure(self) -> None:
        def request() -> SimpleNamespace:
            time.sleep(0.15)
            return SimpleNamespace(status_code=200)

        _, summary = open_loop_load.run_fixed_arrival(
            phase="fixture-saturation",
            rate_rps=100.0,
            duration_seconds=0.20,
            max_concurrency=1,
            queue_capacity=1,
            request_timeout_seconds=1.0,
            drain_timeout_seconds=1.0,
            request=request,
        )
        self.assertGreater(summary["dropped_requests"], 0)
        self.assertEqual(summary["scheduled_arrivals"], summary["accounted_arrivals"])
        self.assertEqual(
            summary["scheduled_arrivals"] - summary["success_requests"],
            summary["delivery_failures"],
        )

    def test_timeouts_are_counted_as_failures(self) -> None:
        def request() -> SimpleNamespace:
            raise Timeout("fixture timeout")

        _, summary = open_loop_load.run_fixed_arrival(
            phase="fixture-timeout",
            rate_rps=10.0,
            duration_seconds=0.20,
            max_concurrency=1,
            queue_capacity=1,
            request_timeout_seconds=0.1,
            drain_timeout_seconds=1.0,
            request=request,
        )
        self.assertEqual(2, summary["timeout_requests"])
        self.assertEqual(2, summary["delivery_failures"])
        self.assertTrue(summary["accounting_balanced"])


if __name__ == "__main__":
    unittest.main()
