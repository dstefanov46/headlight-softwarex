"""One-user, one-request-per-task fixed-rate TrainTicket workload."""

from __future__ import annotations

import csv
import json
import os
import time

from locust import HttpUser, LoadTestShape, constant_pacing, events, task


NAMESPACE = os.environ.get("HEADLIGHT_TRAINTICKET_NAMESPACE", "trainticket")
TARGET_RPS = float(os.environ.get("HEADLIGHT_TARGET_RPS", "1.0"))
REQUEST_CSV_PATH = os.environ.get("HEADLIGHT_REQUEST_CSV", "/app/headlight_requests.csv")
VERIFY_CODE = "025080CF8BA94594B09E283F17815444"
TRACE_SEGMENTS = json.loads(os.environ.get("HEADLIGHT_TRACE_SEGMENTS", "[]"))

request_handle = open(REQUEST_CSV_PATH, "a", encoding="utf-8", newline="")
request_csv = csv.DictWriter(
    request_handle,
    fieldnames=["timestamp", "method", "endpoint", "status", "response_time_ms", "exception"],
)
if request_handle.tell() == 0:
    request_csv.writeheader()


@events.request.add_listener
def record_request(request_type, name, response_time, response_length, response=None, exception=None, start_time=None, **_kwargs):
    request_csv.writerow(
        {
            "timestamp": start_time or time.time(),
            "method": request_type,
            "endpoint": name,
            "status": getattr(response, "status_code", "") if response is not None else "",
            "response_time_ms": response_time,
            "exception": str(exception) if exception else "",
        }
    )
    request_handle.flush()


def service_url(service: str, port: int, path: str) -> str:
    return f"http://{service}.{NAMESPACE}.svc.cluster.local:{port}{path}"


class FixedRateUser(HttpUser):
    host = service_url("ts-travel-service", 12346, "")
    wait_time = constant_pacing(1.0 / TARGET_RPS)

    def on_start(self) -> None:
        response = self.client.post(
            service_url("ts-auth-service", 12340, "/api/v1/users/login"),
            headers={"Content-Type": "application/json", "X-Verify-Code": VERIFY_CODE},
            json={"username": "fdse_microservice", "password": "111111"},
            name="auth.login",
        )
        try:
            token = response.json().get("data", {}).get("token", "")
        except Exception:
            token = ""
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    @task
    def travel_left(self) -> None:
        self.client.post(
            service_url("ts-travel-service", 12346, "/api/v1/travelservice/trips/left"),
            headers=self.headers,
            json={
                "departureTime": "2021-07-15",
                "startingPlace": "Shang Hai",
                "endPlace": "Su Zhou",
            },
            name="travel.left",
        )


if TRACE_SEGMENTS:
    class InfrastructureQualificationShape(LoadTestShape):
        def tick(self):
            elapsed = self.get_run_time()
            boundary = 0.0
            for segment in TRACE_SEGMENTS:
                boundary += float(segment["duration_seconds"])
                if elapsed < boundary:
                    users = int(segment["users"])
                    return users, max(1.0, float(users))
            return None
