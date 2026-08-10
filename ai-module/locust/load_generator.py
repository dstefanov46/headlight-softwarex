import string
import subprocess
import logging
import secrets
import time
import sys
import csv
import math
import os
sys.path.append('/app')

from typing import List
from locust import HttpUser, TaskSet, events, task, constant
from locust import LoadTestShape

namespace = "trainticket"
secure_random = secrets.SystemRandom()
AWK_NODE_IP = " | awk '{print $3}' "
JSON_CONTENT_TYPE = "application/json"
AUTHORIZATION_PREFIX = "Bearer "
DEFAULT_COOKIE = "JSESSIONID=823B2652E3F5B64A1C94C924A05D80AF; YsbCaptcha=2E037F4AB09D49FA9EE3BE4E737EAFD2"
SHANG_HAI = "Shang Hai"
SU_ZHOU = "Su Zhou"
NAN_JING = "Nan Jing"
request_csv_path = os.environ.get("HEADLIGHT_REQUEST_CSV", "headlight_requests.csv")
request_csv_dir = os.path.dirname(request_csv_path)
if request_csv_dir:
    os.makedirs(request_csv_dir, exist_ok=True)
request_csv_handle = open(request_csv_path, "a", encoding="utf-8", newline="")
request_csv = csv.DictWriter(
    request_csv_handle,
    fieldnames=["timestamp", "method", "endpoint", "status", "response_time_ms", "exception"],
)
if request_csv_handle.tell() == 0:
    request_csv.writeheader()


@events.request.add_listener
def record_request(request_type, name, response_time, response_length, response=None, context=None, exception=None, start_time=None, url=None, **kwargs):
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


def get_hostname():
    try:
        result = subprocess.check_output(["hostname"], text=True).strip()
        print(f"[INFO] Pod hostname: {result}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get hostname: {e}")
        return None


def get_kube_contexts():
    try:
        contexts = subprocess.check_output(
            ["kubectl", "config", "get-contexts", "-o", "name"],
            text=True
        ).strip().splitlines()
        return contexts
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get kube contexts: {e}")
        return []


def search_pod_in_contexts(pod_name):
    for context in get_kube_contexts():
        print(f"[INFO] Checking context: {context}")
        try:
            output = subprocess.check_output(
                ["kubectl", "--context", context, "get", "pods", "--all-namespaces", "--no-headers"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            for line in output.strip().splitlines():
                if f"{pod_name} " in line or f"{pod_name}\t" in line:
                    print(f"✅ Found pod '{pod_name}' in context: {context}")
                    return context
        except subprocess.CalledProcessError:
            continue
    print(f"Pod '{pod_name}' not found in any context.")
    return None


# Utility function to execute a kubectl command and capture the output
def run_kubectl_command(command: str):
    try:
        # Start the process using Popen
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()  # Capture output and error

        if process.returncode != 0:
            logging.error(f"Command failed: {stderr.decode()}")
            return None
        return stdout.decode().strip()
    except Exception:
        logging.exception("Error executing command")
        return None


pod_name = get_hostname()
context = search_pod_in_contexts(pod_name)


class TrainTicketUserTasks(TaskSet):
    def __init__(self, parent):
        super().__init__(parent)

    def random_boolean(self) -> bool:
        return secure_random.choice([True, False])

    def random_from_list(self, l: List):
        return secure_random.choice(l)

    def random_from_weighted(self, d: dict):
        total = sum(d.values())
        ra = secure_random.uniform(0, total)
        curr_sum = 0
        ret = None

        keys = d.keys()
        for k in keys:
            curr_sum += d[k]
            if ra <= curr_sum:
                ret = k
                break

        return ret

    def random_str(self):
        return ''.join(secure_random.choices(string.ascii_letters, k=secure_random.randint(4, 10)))

    def random_phone(self):
        return ''.join(secure_random.choices(string.digits, k=secure_random.randint(8, 15)))

    logger = logging.getLogger("query_and_preserve")
    uuid = "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
    date = time.strftime("%Y-%m-%d", time.localtime())

    def login(self):
        username = "fdse_microservice"
        password = "111111"
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-auth-service" + AWK_NODE_IP)
        url = f"http://{ip}:12340/api/v1/users/login"

        headers = {
            "Content-Type": JSON_CONTENT_TYPE,
            'X-Verify-Code': '025080CF8BA94594B09E283F17815444'
        }

        data = '{"username":"' + username + '","password":"' + password + '"}'
        r = self.client.post(url=url, headers=headers, data=data, verify=False)
        if r.status_code == 200:
            data = r.json().get("data")
            token = data.get("token")
            return token
        return None

    @task(1)
    def query_admin_basic_config_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.query_admin_basic_config(headers=headers)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_advanced_ticket_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token,
                   "Content-Type": JSON_CONTENT_TYPE}

        place_pairs = [(SHANG_HAI, SU_ZHOU),
                       (SU_ZHOU, SHANG_HAI),
                       (NAN_JING, SHANG_HAI)]
        plan_type = "quickest"
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-plan-service | "
                                 f"awk '{{print $3}}'")
        url = f"http://{ip}:14322/api/v1/travelplanservice/travelPlan/" + plan_type
        place_pair = secure_random.choice(place_pairs)

        payload = {
            "departureTime": time.strftime("%Y-%m-%d", time.localtime()),
            "startingPlace": place_pair[0],
            "endPlace": place_pair[1],
        }

        self.client.post(url, headers=headers, json=payload)

    @task(1)
    def query_and_preserve_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token,
                   "Content-Type": JSON_CONTENT_TYPE}

        high_speed = self.random_boolean()
        if high_speed:
            start = SHANG_HAI
            end = SU_ZHOU
            high_speed_place_pair = (start, end)
            trip_ids = self.query_high_speed_ticket(place_pair=high_speed_place_pair, headers=headers,
                                                    time=time.strftime("%Y-%m-%d", time.localtime()))
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-preserve-service"
                                     + AWK_NODE_IP)
            PRESERVE_URL = f"http://{ip}:14568/api/v1/preserveservice/preserve"
        else:
            start = SHANG_HAI
            end = NAN_JING
            other_place_pair = (start, end)
            trip_ids = self.query_normal_ticket(place_pair=other_place_pair, headers=headers,
                                                time=time.strftime("%Y-%m-%d", time.localtime()))
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-preserve-other-service"
                                     + AWK_NODE_IP)
            PRESERVE_URL = f"http://{ip}:14569/api/v1/preserveotherservice/preserveOther"

        _ = self.query_assurances(headers=headers)
        food_result = self.query_food(headers=headers)
        contacts_result = self.query_contacts(headers=headers)

        base_preserve_payload = {
            "accountId": self.uuid,
            "assurance": "0",
            "contactsId": "",
            "date": time.strftime("%Y-%m-%d", time.localtime()),
            "from": start,
            "to": end,
            "tripId": ""
        }

        trip_id = self.random_from_list(trip_ids)
        base_preserve_payload["tripId"] = trip_id

        need_food = self.random_boolean()
        if need_food:
            self.logger.info("need food")
            food_dict = self.random_from_list(food_result)
            base_preserve_payload.update(food_dict)
        else:
            self.logger.info("not need food")
            base_preserve_payload["foodType"] = "0"

        need_assurance = self.random_boolean()
        if need_assurance:
            base_preserve_payload["assurance"] = 1

        contacts_id = self.random_from_list(contacts_result)
        base_preserve_payload["contactsId"] = contacts_id

        seat_type = self.random_from_list(["2", "3"])
        base_preserve_payload["seatType"] = seat_type

        need_consign = self.random_boolean()
        if need_consign:
            consign = {
                "consigneeName": self.random_str(),
                "consigneePhone": self.random_phone(),
                "consigneeWeight": secure_random.randint(1, 10),
                "handleDate": time.strftime("%Y-%m-%d", time.localtime())
            }
            base_preserve_payload.update(consign)

        print("payload: " + str(base_preserve_payload))

        print(f"choices: preserve_high: {high_speed} need_food: {need_food}  need_consign: {need_consign}"
              f" need_assurance: {need_assurance}")

        res = self.client.post(url=PRESERVE_URL,
                               headers=headers,
                               json=base_preserve_payload)

        print(res.json())
        if res.json()["data"] != "Success":
            raise RuntimeError(res.json() + " not success")
        print()

    @task(1)
    def query_admin_basic_price_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.query_admin_basic_price(headers=headers)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_and_cancel_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}

        self.query_one_and_cancel(headers=headers, uuid=self.uuid)

    @task(1)
    def query_and_collect_ticket_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}
        self.query_and_collect_ticket(headers=headers)

    @task(1)
    def query_and_enter_station_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}
        self.query_and_enter_station(headers=headers)

    @task(1)
    def query_and_put_consign_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        pairs = self.query_orders_all_info(headers=headers)
        pairs2 = self.query_orders_all_info(headers=headers, query_other=True)
        pairs = pairs + pairs2
        self.query_one_and_put_consign(headers=headers, pairs=pairs)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_and_rebook_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.query_and_rebook(headers=headers)

        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_food_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.query_food(headers=headers)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_order_and_pay_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token, "Content-Type": JSON_CONTENT_TYPE}
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        pairs = self.query_orders(headers=headers, types=(0, 1))
        pairs2 = self.query_orders(headers=headers, types=(0, 1), query_other=True)

        pairs = pairs + pairs2

        for i in range(330):
            try:
                self.query_order_and_pay(headers=headers, pairs=pairs)
                print("***************************** INDEX: " + str(i))
            except Exception as e:
                print(e)

        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_route_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token,
                   "Content-Type": JSON_CONTENT_TYPE}

        self.query_route(headers=headers)

    @task(1)
    def query_travel_left_parallel_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token,
                   "Content-Type": JSON_CONTENT_TYPE}

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"start: {start_time}")
        self.query_travel_left_parallel(headers=headers)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    @task(1)
    def query_travel_left_use(self):
        token = self.login()
        headers = {"Cookie": DEFAULT_COOKIE,
                   "Authorization": AUTHORIZATION_PREFIX + token,
                   "Content-Type": JSON_CONTENT_TYPE}
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"start: {start_time}")

        self.query_travel_left(headers=headers)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print(f"start: {start_time} end: {end_time}")

    def query_high_speed_ticket(self,place_pair: tuple = (SHANG_HAI, SU_ZHOU), headers: dict = {},
                                time: str = "2021-07-15") -> list[str] | None:
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:12346/api/v1/travelservice/trips/left"
        payload = {
            "departureTime": time,
            "startingPlace": place_pair[0],
            "endPlace": place_pair[1],
        }

        response = self.client.post(url=url,
                                    headers=headers,
                                    json=payload)

        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"request for {url} failed. response data is {response.text}")
            return None

        data = response.json().get("data")

        trip_ids = []
        for d in data:
            trip_id = d.get("tripId").get("type") + d.get("tripId").get("number")
            trip_ids.append(trip_id)
        return trip_ids

    def query_normal_ticket(self,place_pair: tuple = (NAN_JING, SHANG_HAI), headers: dict = {},
                            time: str = "2021-07-15") -> list[str] | None:
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel2-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:16346/api/v1/travel2service/trips/left"
        payload = {
            "departureTime": time,
            "startingPlace": place_pair[0],
            "endPlace": place_pair[1],
        }

        response = self.client.post(url=url,
                                    headers=headers,
                                    json=payload)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"request for {url} failed. response data is {response.json()}")
            return None

        data = response.json().get("data")

        trip_ids = []
        for d in data:
            trip_id = d.get("tripId").get("type") + d.get("tripId").get("number")
            trip_ids.append(trip_id)
        return trip_ids

    def query_assurances(self, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-assurance-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18888/api/v1/assuranceservice/assurances/types"
        response = self.client.get(url=url, headers=headers)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"query assurance failed, response data is {response.json()}")
            return None

        return [{"assurance": "1"}]

    def query_food(self,place_pair: tuple = (SHANG_HAI, SU_ZHOU), train_num: str = "D1345", headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-food-service"
                                 + AWK_NODE_IP)
        url = (f"http://{ip}:18856/api/v1/foodservice/foods/2021-07-14/"
               f"{place_pair[0]}/{place_pair[1]}/{train_num}")

        response = self.client.get(url=url, headers=headers)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"query food failed, response data is {response}")
            return None
        return [{
            "foodName": "Soup",
            "foodPrice": 3.7,
            "foodType": 2,
            "stationName": SU_ZHOU,
            "storeName": "Roman Holiday"
        }]

    def query_contacts(self,headers: dict = {}) -> list[str] | None:
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-contacts-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:12347/api/v1/contactservice/contacts/account/{self.uuid}"
        response = self.client.get(url=url, headers=headers)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"query contacts failed, response data is {response.json()}")
            return None

        data = response.json().get("data")

        ids = [d.get("id") for d in data if d.get("id") is not None]

        return ids

    def query_admin_basic_config(self,headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-admin-basic-info-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18767/api/v1/adminbasicservice/adminbasic/configs"
        response = self.client.get(url=url,headers=headers)
        if response.status_code == 200:
            print("config success")
            return response
        else:
            print("config failed")
            return None

    def admin_login(self):
        return self.login()

    def query_high_speed_ticket_parallel(self,place_pair: tuple = (SHANG_HAI, SU_ZHOU), headers: dict = {},
                                         time: str = "2021-07-15") -> list[str] | None:
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:12346/api/v1/travelservice/trips/left_parallel"
        payload = {
            "departureTime": time,
            "startingPlace": place_pair[0],
            "endPlace": place_pair[1],
        }

        response = self.client.post(url=url,
                                    headers=headers,
                                    json=payload)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"request for {url} failed. response data is {response.text}")
            return None

        data = response.json().get("data")

        trip_ids = []
        for d in data:
            trip_id = d.get("tripId").get("type") + d.get("tripId").get("number")
            trip_ids.append(trip_id)
        return trip_ids

    def query_advanced_ticket(self, place_pair: tuple = (NAN_JING, SHANG_HAI), headers: dict = {},
                              time: str = "2021-07-15", plan_type: str = "cheapest") -> list[str] | None:
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-plan-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:14322/api/v1/travelplanservice/travelPlan/" + plan_type

        payload = {
            "departureTime": time,
            "startingPlace": place_pair[0],
            "endPlace": place_pair[1],
        }

        response = self.client.post(url=url,
                                    headers=headers,
                                    json=payload)

        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"request for {url} failed. response data is {response.json()}")
            return None

        data = response.json().get("data")

        trip_ids = []
        for d in data:
            trip_id = d.get("tripId")
            trip_ids.append(trip_id)
        return trip_ids

    def query_orders(self, headers: dict = {}, types: tuple = (0,), query_other: bool = False) -> list[tuple] | None:
        if query_other:
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-order-other-service"
                                     + AWK_NODE_IP)
            url = f"http://{ip}:12032/api/v1/orderOtherService/orderOther/refresh"
        else:
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-order-service"
                                     + AWK_NODE_IP)
            url = f"http://{ip}:12031/api/v1/orderservice/order/refresh"

        payload = {
            "loginId": self.uuid,
        }

        response = self.client.post(url=url, headers=headers, json=payload)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"query orders failed, response data is {response.text}")
            return None

        data = response.json().get("data")
        pairs = []
        for d in data:
            # status = 0: not paid
            # status = 1: paid not collect
            # status = 2: collected
            if d.get("status") in types:
                order_id = d.get("id")
                trip_id = d.get("trainNumber")
                pairs.append((order_id, trip_id))
        print(f"queried {len(pairs)} orders")

        return pairs

    def query_orders_all_info(self,headers: dict = {}, query_other: bool = False) -> list[tuple] | None:
        if query_other:
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-order-other-service"
                                     + AWK_NODE_IP)
            url = f"http://{ip}:12032/api/v1/orderOtherService/orderOther/refresh"
        else:
            ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-order-service"
                                     + AWK_NODE_IP)
            url = f"http://{ip}:12031/api/v1/orderservice/order/refresh"

        payload = {
            "loginId": self.uuid,
        }

        response = self.client.post(url=url, headers=headers, json=payload)
        if response.status_code != 200 or response.json().get("data") is None:
            self.logger.warning(f"query orders failed, response data is {response.text}")
            return None

        data = response.json().get("data")
        pairs = []
        for d in data:
            result = {"accountId": d.get("accountId"),
                      "targetDate": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                      "orderId": d.get("id"), "from": d.get("from"), "to": d.get("to")}
            pairs.append(result)
        print(f"queried {len(pairs)} orders")

        return pairs

    def put_consign(self, result, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-consign-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:16111/api/v1/consignservice/consigns"
        consignload = {
            "accountId": result["accountId"],
            "handleDate": time.strftime('%Y-%m-%d', time.localtime(time.time())),
            "targetDate": result["targetDate"],
            "from": result["from"],
            "to": result["to"],
            "orderId": result["orderId"],
            "consignee": "32",
            "phone": "12345677654",
            "weight": "32",
            "id": "",
            "isWithin": False
        }
        response = self.client.put(url=url, headers=headers, json=consignload)

        order_id = result["orderId"]
        if response.status_code == 200 or response.status_code == 201:
            print(f"{order_id} put consign success")
        else:
            print(f"{order_id} failed!")
            return None

        return order_id

    def query_route(self,route_id: str = '92708982-77af-4318-be25-57ccb0ff69ad', headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-route-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:11178/api/v1/routeservice/routes/{route_id}"

        res = self.client.get(url=url, headers=headers)

        if res.status_code == 200:
            print(f"query {route_id} success")
        else:
            print(f"query {route_id} fail")

    def pay_one_order(self,order_id, trip_id, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-inside-payment-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18673/api/v1/inside_pay_service/inside_payment"
        payload = {
            "orderId": order_id,
            "tripId": trip_id
        }

        response = self.client.post(url=url, headers=headers, json=payload)

        if response.status_code == 200:
            print(f"{order_id} pay success")
        else:
            print(f"pay {order_id} failed!")
            return None

        return order_id

    def cancel_one_order(self, order_id, uuid, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-cancel-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18885/api/v1/cancelservice/cancel/{order_id}/{uuid}"
        response = self.client.get(url=url,
                                   headers=headers)

        if response.status_code == 200:
            print(f"{order_id} cancel success")
        else:
            print(f"{order_id} cancel failed")

        return order_id

    def collect_one_order(self, order_id, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-execute-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:12386/api/v1/executeservice/execute/collected/{order_id}"
        response = self.client.get(url=url,
                                   headers=headers)
        if response.status_code == 200:
            print(f"{order_id} collect success")
        else:
            print(f"{order_id} collect failed")

        return order_id

    def enter_station(self,order_id, headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-execute-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:12386/api/v1/executeservice/execute/execute/{order_id}"
        response = self.client.get(url=url,
                                   headers=headers)
        if response.status_code == 200:
            print(f"{order_id} enter station success")
        else:
            print(f"{order_id} enter station failed")

        return order_id

    def query_cheapest(self,date="2021-12-31", headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-plan-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:14322/api/v1/travelplanservice/travelPlan/cheapest"

        payload = {
            "departureTime": date,
            "endPlace": SHANG_HAI,
            "startingPlace": NAN_JING
        }

        r = self.client.post(url=url, json=payload, headers=headers)
        if r.status_code == 200:
            print("query cheapest success")
        else:
            print("query cheapest failed")

    def query_min_station(self,date="2021-12-31", headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-plan-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:14322/api/v1/travelplanservice/travelPlan/minStation"

        payload = {
            "departureTime": date,
            "endPlace": SHANG_HAI,
            "startingPlace": NAN_JING
        }

        r = self.client.post(url=url, json=payload, headers=headers)
        if r.status_code == 200:
            print("query min station success")
        else:
            print("query min station failed")

    def query_quickest(self,date="2021-12-31", headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-travel-plan-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:14322/api/v1/travelplanservice/travelPlan/quickest"

        payload = {
            "departureTime": date,
            "endPlace": SHANG_HAI,
            "startingPlace": NAN_JING
        }

        r = self.client.post(url=url, json=payload, headers=headers)
        if r.status_code == 200:
            print("query quickest success")
        else:
            print("query quickest failed")

    def query_admin_basic_price(self,headers: dict = {}):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-admin-basic-info-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18767/api/v1/adminbasicservice/adminbasic/prices"
        response = self.client.get(url=url,
                                   headers=headers)

        if response.status_code == 200:
            print("price success")
            return response
        else:
            print("price failed")
            return None

    def rebook_ticket(self,old_order_id, old_trip_id, new_trip_id, new_date, new_seat_type, headers):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-rebook-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:18886/api/v1/rebookservice/rebook"

        payload = {
            "oldTripId": old_trip_id,
            "orderId": old_order_id,
            "tripId": new_trip_id,
            "date": new_date,
            "seatType": new_seat_type
        }
        print(payload)
        r = self.client.post(url=url, json=payload, headers=headers)
        if r.status_code == 200:
            print(r.text)
        else:
            print(f"Request Failed: status code: {r.status_code}")
            print(r.text)

    def query_admin_travel(self, headers):
        ip = run_kubectl_command(f"kubectl get service -n {namespace} --context={context} | grep ts-admin-travel-service"
                                 + AWK_NODE_IP)
        url = f"http://{ip}:16114/api/v1/admintravelservice/admintravel"

        r = self.client.get(url=url, headers=headers)
        if r.status_code == 200 and r.json()["status"] == 1:
            print("success to query admin travel")
        else:
            print(f"failed to query admin travel with status_code: {r.status_code}")

    def query_one_and_cancel(self, headers, uuid="4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"):
        pairs = self.query_orders(headers=headers, types=(0,))
        pairs2 = self.query_orders(headers=headers, types=(0,), query_other=True)

        if not pairs and not pairs2:
            return

        pairs = pairs + pairs2

        # (orderId, tripId) pair
        pair = self.random_from_list(pairs)
        print("...............................")
        order_id = self.cancel_one_order(order_id=pair[0], uuid=self.uuid, headers=headers)
        if not order_id:
            return

        print(f"{order_id} queried and canceled")

    def query_and_collect_ticket(self, headers):

        pairs = self.query_orders(headers=headers, types=(0,))
        pairs2 = self.query_orders(headers=headers, types=(0,), query_other=True)

        if not pairs and not pairs2:
            return

        pairs = pairs + pairs2

        # (orderId, tripId)
        pair = self.random_from_list(pairs)

        order_id = self.collect_one_order(order_id=pair[0], headers=headers)
        if not order_id:
            return

        print(f"{order_id} queried and collected")

    def query_and_enter_station(self, headers):
        pairs = self.query_orders(headers=headers, types=(0,))
        pairs2 = self.query_orders(headers=headers, types=(0,), query_other=True)

        if not pairs and not pairs2:
            return

        pairs = pairs + pairs2

        # (orderId, tripId)
        pair = self.random_from_list(pairs)

        order_id = self.enter_station(order_id=pair[0], headers=headers)
        if not order_id:
            return

        print(f"{order_id} queried and entered station")

    def query_one_and_put_consign(self,headers, pairs):
        pair = self.random_from_list(pairs)

        order_id = self.put_consign(result=pair, headers=headers)
        if not order_id:
            return

        print(f"{order_id} queried and put consign")

    def query_and_rebook(self,headers):
        pairs = self.query_orders(headers=headers, types=(0,))
        pairs2 = self.query_orders(headers=headers, types=(1,), query_other=True)

        if not pairs and not pairs2:
            return

        pairs = pairs + pairs2

        # (orderId, tripId)
        new_trip_id = "D1345"
        new_date = time.strftime("%Y-%m-%d", time.localtime())
        new_seat_type = "3"

        for pair in pairs:
            self.rebook_ticket(old_order_id=pair[0], old_trip_id=pair[1], new_trip_id=new_trip_id, new_date=new_date,
                               new_seat_type=new_seat_type, headers=headers)

    def query_order_and_pay(self,headers, pairs):
        # (orderId, tripId) pair
        pair = self.random_from_list(pairs)

        order_id = self.pay_one_order(pair[0], pair[1], headers=headers)
        if not order_id:
            return

        print(f"{order_id} queried and paid")

    def query_travel_left_parallel(self,headers):
        start = SU_ZHOU
        end = SHANG_HAI
        high_speed_place_pair = (start, end)
        self.query_high_speed_ticket_parallel(place_pair=high_speed_place_pair, headers=headers,
                                              time=time.strftime("%Y-%m-%d", time.localtime()))

    def query_travel_left(self,headers):
        start = SHANG_HAI
        end = NAN_JING
        other_place_pair = (start, end)
        self.query_normal_ticket(place_pair=other_place_pair, headers=headers,
                                 time=time.strftime("%Y-%m-%d", time.localtime()))


class WebsiteUser(HttpUser):
    def on_start(self):
        return super().on_start()

    def on_stop(self):
        return super().on_stop()
    host = "http://localhost:30001"
    wait_time = constant(1)
    tasks = [TrainTicketUserTasks]


class StagesShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.pattern = os.environ.get("HEADLIGHT_WORKLOAD_PATTERN", "gradual-ramp")
        self.duration = int(os.environ.get("HEADLIGHT_DURATION_SECONDS", "3000"))
        self.spawn_rate = int(os.environ.get("HEADLIGHT_SPAWN_RATE", "15"))

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.duration:
            return None

        if self.pattern == "gradual-ramp":
            start = int(os.environ.get("HEADLIGHT_START_USERS", "20"))
            end = int(os.environ.get("HEADLIGHT_END_USERS", "320"))
            ramp = int(os.environ.get("HEADLIGHT_RAMP_SECONDS", "1200"))
            hold = int(os.environ.get("HEADLIGHT_HOLD_SECONDS", str(max(0, self.duration - (2 * ramp)))))
            if run_time <= ramp:
                users = start + (end - start) * (run_time / max(1, ramp))
            elif run_time <= ramp + hold:
                users = end
            else:
                users = end + (start - end) * ((run_time - ramp - hold) / max(1, self.duration - ramp - hold))
        elif self.pattern == "flash-crowd":
            base = int(os.environ.get("HEADLIGHT_BASE_USERS", "30"))
            spike = int(os.environ.get("HEADLIGHT_SPIKE_USERS", "400"))
            starts = [int(v) for v in os.environ.get("HEADLIGHT_SPIKE_START_SECONDS", "600,1500").split(",")]
            spike_duration = int(os.environ.get("HEADLIGHT_SPIKE_DURATION_SECONDS", "180"))
            users = base
            for start_time in starts:
                if start_time <= run_time < start_time + spike_duration:
                    users = spike
                    break
        elif self.pattern == "diurnal":
            minimum = int(os.environ.get("HEADLIGHT_MIN_USERS", "25"))
            maximum = int(os.environ.get("HEADLIGHT_MAX_USERS", "250"))
            cycle = int(os.environ.get("HEADLIGHT_CYCLE_SECONDS", str(self.duration)))
            phase = int(os.environ.get("HEADLIGHT_PHASE_OFFSET_SECONDS", "0"))
            amplitude = (maximum - minimum) / 2
            midpoint = minimum + amplitude
            users = midpoint + amplitude * math.sin(2 * math.pi * (run_time + phase) / max(1, cycle) - math.pi / 2)
        else:
            users = int(os.environ.get("HEADLIGHT_BASE_USERS", "25"))
        return (max(1, int(round(users))), self.spawn_rate)
