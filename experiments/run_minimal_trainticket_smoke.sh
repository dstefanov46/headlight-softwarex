#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
cluster="${HEADLIGHT_SMOKE_CLUSTER:-cluster2}"
duration="${HEADLIGHT_SMOKE_DURATION:-5m}"
users="${HEADLIGHT_SMOKE_USERS:-1}"
spawn_rate="${HEADLIGHT_SMOKE_SPAWN_RATE:-1}"
image="${HEADLIGHT_LOCUST_IMAGE:-dstefanov46/locust-test:0.66.0}"
job="headlight-minimal-smoke"
result_dir="${HEADLIGHT_SMOKE_RESULT_DIR:-$repo_root/results/local-smoke/$(date +%Y%m%d-%H%M%S)}"
require_prometheus="${HEADLIGHT_REQUIRE_PROMETHEUS:-1}"
prometheus_namespace="${HEADLIGHT_PROMETHEUS_NAMESPACE:-monitoring}"
prometheus_service="${HEADLIGHT_PROMETHEUS_SERVICE:-prometheus-kube-prometheus-prometheus}"
prometheus_port="${HEADLIGHT_PROMETHEUS_PORT:-39090}"

export KUBECONFIG="$kubeconfig"
mkdir -p "$result_dir/raw" "$result_dir/logs"

kubectl --context="$cluster" -n "$namespace" delete job "$job" --ignore-not-found
kubectl --context="$cluster" -n "$namespace" delete configmap headlight-minimal-locust --ignore-not-found
kubectl --context="$cluster" -n "$namespace" create configmap headlight-minimal-locust \
  --from-file=minimal_smoke.py="$repo_root/experiments/testbed/trainticket-minimal/minimal_smoke.py"

kubectl --context="$cluster" -n "$namespace" apply -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
spec:
  backoffLimit: 0
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      containers:
      - name: locust
        image: $image
        imagePullPolicy: IfNotPresent
        command:
        - /bin/sh
        - -lc
        args:
        - |
          /opt/venv/bin/locust --headless -f /app/minimal_smoke.py \
            --users "$users" \
            --spawn-rate "$spawn_rate" \
            --run-time "$duration" \
            --csv /tmp/headlight-smoke
          rc="\$?"
          echo __HEADLIGHT_REQUEST_CSV_BEGIN__
          cat /app/headlight_requests.csv 2>/dev/null || true
          echo __HEADLIGHT_REQUEST_CSV_END__
          echo __HEADLIGHT_LOCUST_STATS_BEGIN__
          cat /tmp/headlight-smoke_stats.csv 2>/dev/null || true
          echo __HEADLIGHT_LOCUST_STATS_END__
          exit "\$rc"
        volumeMounts:
        - name: locustfile
          mountPath: /app/minimal_smoke.py
          subPath: minimal_smoke.py
        env:
        - name: HEADLIGHT_REQUEST_CSV
          value: /app/headlight_requests.csv
      volumes:
      - name: locustfile
        configMap:
          name: headlight-minimal-locust
YAML

if ! kubectl --context="$cluster" -n "$namespace" wait --for=condition=Complete "job/$job" --timeout="${HEADLIGHT_SMOKE_WAIT_TIMEOUT:-15m}"; then
  pod="$(kubectl --context="$cluster" -n "$namespace" get pod -l job-name="$job" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$pod" ]]; then
    kubectl --context="$cluster" -n "$namespace" describe pod "$pod" > "$result_dir/logs/smoke-pod-describe.txt" || true
    kubectl --context="$cluster" -n "$namespace" logs "$pod" > "$result_dir/logs/smoke.log" || true
  fi
  exit 1
fi

pod="$(kubectl --context="$cluster" -n "$namespace" get pod -l job-name="$job" -o jsonpath='{.items[0].metadata.name}')"
kubectl --context="$cluster" -n "$namespace" logs "$pod" > "$result_dir/logs/smoke.log"
awk '/__HEADLIGHT_REQUEST_CSV_BEGIN__/{flag=1;next}/__HEADLIGHT_REQUEST_CSV_END__/{flag=0}flag' \
  "$result_dir/logs/smoke.log" > "$result_dir/raw/request_events.csv"
awk '/__HEADLIGHT_LOCUST_STATS_BEGIN__/{flag=1;next}/__HEADLIGHT_LOCUST_STATS_END__/{flag=0}flag' \
  "$result_dir/logs/smoke.log" > "$result_dir/raw/locust_stats.csv"
kubectl --context="$cluster" -n "$namespace" get events -o json > "$result_dir/raw/kubernetes_events.json"

if kubectl --context="$cluster" -n "$prometheus_namespace" get service "$prometheus_service" >/dev/null 2>&1; then
  kubectl --context="$cluster" -n "$prometheus_namespace" port-forward "svc/$prometheus_service" "$prometheus_port:9090" \
    > "$result_dir/logs/prometheus-port-forward.log" 2>&1 &
  pf_pid="$!"
  trap 'kill "$pf_pid" >/dev/null 2>&1 || true' EXIT

  python3 - "$prometheus_port" "$namespace" "$duration" "$result_dir/raw/prometheus_range_data.json" <<'PY'
import json
import sys
import time
import urllib.parse
import urllib.request


def duration_seconds(value: str) -> int:
    value = value.strip()
    if value.endswith("m"):
        return int(value[:-1]) * 60
    if value.endswith("s"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 3600
    return int(value)


port, namespace, duration, output = sys.argv[1:5]
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try:
        urllib.request.urlopen(f"{base}/-/ready", timeout=1).read()
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Prometheus port-forward did not become ready")

end = int(time.time())
start = end - duration_seconds(duration) - 60
queries = {
    "up": "up",
    "istio_request_rate": f'sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}"}}[1m]))',
    "istio_p95_latency_ms": f'histogram_quantile(0.95, sum by (le) (rate(istio_request_duration_milliseconds_bucket{{destination_workload_namespace="{namespace}"}}[1m])))',
}
payload = {}
nonempty = False
for name, query in queries.items():
    params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": "15s"})
    with urllib.request.urlopen(f"{base}/api/v1/query_range?{params}", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    payload[name] = {"query": query, "response": body}
    nonempty = nonempty or bool(body.get("data", {}).get("result"))

with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

if not nonempty:
    raise SystemExit("Prometheus returned no time series")
PY
else
  if [[ "$require_prometheus" = "1" ]]; then
    echo "required Prometheus service $prometheus_service.$prometheus_namespace is missing in $cluster" >&2
    exit 1
  fi
  printf '{}\n' > "$result_dir/raw/prometheus_range_data.json"
fi

if [[ "$(wc -l < "$result_dir/raw/request_events.csv")" -lt 2 ]]; then
  echo "smoke run did not record any request rows" >&2
  exit 1
fi

echo "Minimal smoke result bundle: $result_dir"
