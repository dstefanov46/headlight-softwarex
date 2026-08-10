import csv
import os
import time

from locust import HttpUser, between, events, task


AUTH_BASE = os.environ.get("HEADLIGHT_AUTH_BASE", "http://ts-auth-service:12340")
ROUTE_BASE = os.environ.get("HEADLIGHT_ROUTE_BASE", "http://ts-route-service:11178")
ROUTE_ID = os.environ.get("HEADLIGHT_ROUTE_ID", "92708982-77af-4318-be25-57ccb0ff69ad")
REQUEST_CSV_PATH = os.environ.get("HEADLIGHT_REQUEST_CSV", "headlight_requests.csv")
REQUEST_CSV_DIR = os.path.dirname(REQUEST_CSV_PATH)
if REQUEST_CSV_DIR:
    os.makedirs(REQUEST_CSV_DIR, exist_ok=True)

request_csv_handle = open(REQUEST_CSV_PATH, "a", encoding="utf-8", newline="")
request_csv = csv.DictWriter(
    request_csv_handle,
    fieldnames=["timestamp", "method", "endpoint", "status", "response_time_ms", "exception"],
)
if request_csv_handle.tell() == 0:
    request_csv.writeheader()


@events.request.add_listener
def record_request(
    request_type,
    name,
    response_time,
    response_length,
    response=None,
    context=None,
    exception=None,
    start_time=None,
    url=None,
    **kwargs,
):
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


class MinimalTrainTicketUser(HttpUser):
    wait_time = between(0.5, 1.5)
    host = AUTH_BASE

    def login(self):
        headers = {
            "Content-Type": "application/json",
            "X-Verify-Code": "025080CF8BA94594B09E283F17815444",
        }
        payload = {"username": "fdse_microservice", "password": "111111"}
        with self.client.post(
            f"{AUTH_BASE}/api/v1/users/login",
            headers=headers,
            json=payload,
            name="login",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"login returned HTTP {response.status_code}")
                return None
            data = response.json().get("data") or {}
            token = data.get("token")
            if not token:
                response.failure("login response did not include a token")
                return None
            return token

    @task
    def login_then_route_query(self):
        token = self.login()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with self.client.get(
            f"{ROUTE_BASE}/api/v1/routeservice/routes/{ROUTE_ID}",
            headers=headers,
            name="route_query",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"route query returned HTTP {response.status_code}")
