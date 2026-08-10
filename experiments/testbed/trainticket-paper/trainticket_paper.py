"""Paper-scale TrainTicket Locust workload using Kubernetes service DNS."""

from __future__ import annotations

import csv
import os
import secrets
import time
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, events, task


NAMESPACE = os.environ.get("HEADLIGHT_TRAINTICKET_NAMESPACE", "trainticket")
MODE = os.environ.get("HEADLIGHT_TRAINTICKET_MODE", "login-route")
REQUEST_CSV_PATH = os.environ.get("HEADLIGHT_REQUEST_CSV", "headlight_requests.csv")
REQUEST_CSV_DIR = os.path.dirname(REQUEST_CSV_PATH)
if REQUEST_CSV_DIR:
    os.makedirs(REQUEST_CSV_DIR, exist_ok=True)
LOAD_SHAPE_PATH = os.environ.get("HEADLIGHT_LOAD_SHAPE", "/app/load_profile.csv")
VERIFY_CODE = "025080CF8BA94594B09E283F17815444"
ACCOUNT_ID = "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
DEFAULT_TRAVEL_DATE = os.environ.get("HEADLIGHT_TRAINTICKET_TRAVEL_DATE", "2021-07-15")
SHANG_HAI = "Shang Hai"
SU_ZHOU = "Su Zhou"
NAN_JING = "Nan Jing"
ADVANCED_TICKET_PAIRS = ((SHANG_HAI, SU_ZHOU), (SU_ZHOU, SHANG_HAI), (NAN_JING, SHANG_HAI))
FULL_SESSION_PAIRS = ((SHANG_HAI, SU_ZHOU), (NAN_JING, SHANG_HAI))


request_csv_handle = open(REQUEST_CSV_PATH, "a", encoding="utf-8", newline="")
request_csv = csv.DictWriter(
    request_csv_handle,
    fieldnames=["timestamp", "method", "endpoint", "status", "response_time_ms", "exception"],
)
if request_csv_handle.tell() == 0:
    request_csv.writeheader()


@events.request.add_listener
def record_request(request_type, name, response_time, response_length, response=None, exception=None, start_time=None, url=None, **kwargs):
    request_csv.writerow(
        {
            "timestamp": start_time or time.time(),
            "method": request_type,
            "endpoint": name or url,
            "status": getattr(response, "status_code", "") if response is not None else "",
            "response_time_ms": response_time,
            "exception": str(exception) if exception else "",
        }
    )
    request_csv_handle.flush()


def service_url(service: str, port: int, path: str) -> str:
    return f"http://{service}.{NAMESPACE}.svc.cluster.local:{port}{path}"


def today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def travel_date() -> str:
    return DEFAULT_TRAVEL_DATE


class TrainTicketPaperUser(HttpUser):
    wait_time = between(0.2, 1.2)
    host = service_url("ts-auth-service", 12340, "")

    def login(self) -> str:
        response = self.client.post(
            service_url("ts-auth-service", 12340, "/api/v1/users/login"),
            headers={"Content-Type": "application/json", "X-Verify-Code": VERIFY_CODE},
            json={"username": "fdse_microservice", "password": "111111"},
            name="auth.login",
        )
        try:
            return response.json().get("data", {}).get("token", "")
        except Exception:
            return ""

    def headers(self) -> dict[str, str]:
        token = self.login()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @task(4)
    def login_route(self) -> None:
        headers = self.headers()
        self.client.get(
            service_url("ts-route-service", 11178, "/api/v1/routeservice/routes"),
            headers=headers,
            name="route.list",
        )

    @task(3)
    def advanced_ticket(self) -> None:
        if MODE not in {"advanced-ticket", "full-session"}:
            return
        pair = secrets.choice(ADVANCED_TICKET_PAIRS)
        self.client.post(
            service_url("ts-travel-service", 12346, "/api/v1/travelservice/trips/left"),
            headers=self.headers(),
            json={"departureTime": travel_date(), "startingPlace": pair[0], "endPlace": pair[1]},
            name="travel.left",
        )

    @task(2)
    def full_session_queries(self) -> None:
        if MODE != "full-session":
            return
        headers = self.headers()
        pair = secrets.choice(FULL_SESSION_PAIRS)
        self.client.post(
            service_url("ts-travel-service", 12346, "/api/v1/travelservice/trips/left"),
            headers=headers,
            json={"departureTime": travel_date(), "startingPlace": pair[0], "endPlace": pair[1]},
            name="travel.left",
        )
        self.client.get(
            service_url("ts-assurance-service", 18888, "/api/v1/assuranceservice/assurances/types"),
            headers=headers,
            name="assurance.types",
        )
        self.client.get(
            service_url("ts-contacts-service", 12347, f"/api/v1/contactservice/contacts/account/{ACCOUNT_ID}"),
            headers=headers,
            name="contacts.account",
        )
        self.client.get(
            service_url("ts-food-service", 18856, f"/api/v1/foodservice/foods/{today()}/{pair[0]}/{pair[1]}/D1345"),
            headers=headers,
            name="food.query",
        )


class PaperLoadShape(LoadTestShape):
    def __init__(self) -> None:
        super().__init__()
        self.points = self._load_points(Path(LOAD_SHAPE_PATH))

    @staticmethod
    def _load_points(path: Path) -> list[tuple[int, int, int]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                (int(row["elapsed_seconds"]), int(row["users"]), int(row["spawn_rate"]))
                for row in reader
            ]

    def tick(self):
        elapsed = int(self.get_run_time())
        selected = self.points[-1]
        for point in self.points:
            if elapsed < point[0]:
                break
            selected = point
        if elapsed > self.points[-1][0]:
            return None
        return selected[1], selected[2]
