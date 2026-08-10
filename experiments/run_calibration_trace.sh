#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
phase="${HEADLIGHT_CALIBRATION_PHASE:-paper}"
case "$phase" in
  debug) default_duration=1h ;;
  paper) default_duration=3h ;;
  *) echo "HEADLIGHT_CALIBRATION_PHASE must be debug or paper" >&2; exit 2 ;;
esac
duration="${HEADLIGHT_CALIBRATION_DURATION:-$default_duration}"
warmup="${HEADLIGHT_CALIBRATION_WARMUP:-300s}"
path="${HEADLIGHT_CALIBRATION_TRAINTICKET_PATH:-advanced-ticket}"
clusters="${HEADLIGHT_CALIBRATION_ACTIVE_CLUSTERS:-cluster2 cluster3}"
locust_cluster="${HEADLIGHT_CALIBRATION_LOCUST_CLUSTER:-cluster2}"
default_locust_mode="$path"
if [[ "$path" = "full" ]]; then
  default_locust_mode="full-session"
fi
locust_mode="${HEADLIGHT_CALIBRATION_LOCUST_MODE:-$default_locust_mode}"
locust_image="${HEADLIGHT_LOCUST_IMAGE:-dstefanov46/locust-test:0.66.0}"
if [[ "$phase" = "debug" ]]; then
  default_result_dir="$repo_root/results/calibration/debug"
else
  default_result_dir="$repo_root/results/calibration"
fi
result_dir="${HEADLIGHT_CALIBRATION_RESULT_DIR:-$default_result_dir}"
prometheus_namespace="${HEADLIGHT_PROMETHEUS_NAMESPACE:-monitoring}"
prometheus_service="${HEADLIGHT_PROMETHEUS_SERVICE:-prometheus-kube-prometheus-prometheus}"
prometheus_port="${HEADLIGHT_PROMETHEUS_PORT:-39090}"
sampling_interval="${HEADLIGHT_CALIBRATION_SAMPLING_INTERVAL:-15}"
seed="${HEADLIGHT_CALIBRATION_SEED:-$([[ "$phase" = debug ]] && printf 17 || printf 101)}"
user_scale="${HEADLIGHT_CALIBRATION_USER_SCALE:-1.0}"
max_users="${HEADLIGHT_CALIBRATION_MAX_USERS:-}"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

export KUBECONFIG="$kubeconfig"
mkdir -p "$result_dir/raw" "$result_dir/logs" "$result_dir/feature_history" "$result_dir/models"

duration_seconds() {
  local duration="$1"
  python3 - "$duration" <<'PY'
import re
import sys
value = sys.argv[1]
match = re.fullmatch(r"(\d+)([smh]?)", value)
if not match:
    raise SystemExit(f"invalid duration: {value}")
amount = int(match.group(1))
print(amount * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)])
PY
}

duration_s="$(duration_seconds "$duration")"
warmup_s="$(duration_seconds "$warmup")"
retention_s=$((duration_s + warmup_s + 3600))
retention_h=$(((retention_s + 3599) / 3600))
export HEADLIGHT_PROMETHEUS_RETENTION="${HEADLIGHT_PROMETHEUS_RETENTION:-${retention_h}h}"

echo "== Installing Prometheus with retention $HEADLIGHT_PROMETHEUS_RETENTION =="
HEADLIGHT_PROMETHEUS_CONTEXTS="$clusters" "$repo_root/experiments/install_prometheus_monitoring.sh" \
  > "$result_dir/logs/install-prometheus.log" 2>&1

echo "== Deploying TrainTicket path $path =="
HEADLIGHT_TRAINTICKET_PATH="$path" \
HEADLIGHT_TRAINTICKET_DEPLOY_MODE=karmada \
HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS="$clusters" \
HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST="$result_dir/raw/rendered-trainticket-${path}.yaml" \
  "$repo_root/experiments/deploy_minimal_trainticket.sh" \
  > "$result_dir/logs/deploy-trainticket.log" 2>&1

"$python_bin" - "$duration_s" "$sampling_interval" "$seed" "$result_dir/raw/load_profile.csv" "$user_scale" "$max_users" <<'PY'
import csv
import math
import random
import sys
duration, interval, seed, output = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
user_scale = float(sys.argv[5])
max_users = int(sys.argv[6]) if sys.argv[6] else None
rng = random.Random(seed)
with open(output, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "users", "spawn_rate"])
    writer.writeheader()
    for elapsed in range(0, duration + interval, interval):
        ramp = elapsed / max(duration, 1)
        wave = 0.5 + 0.5 * math.sin((2 * math.pi * elapsed / max(duration, 1)) - math.pi / 2)
        burst = 8 if rng.random() < 0.08 else 0
        users = max(1, int(round((3 + 8 * ramp + 8 * wave + burst) * user_scale)))
        if max_users is not None:
            users = min(users, max_users)
        writer.writerow({"elapsed_seconds": min(elapsed, duration), "users": users, "spawn_rate": max(1, min(users, 5))})
PY

echo "== Running calibration workload for $duration =="
job="headlight-calibration-locust"
kubectl --context="$locust_cluster" -n "$namespace" delete job "$job" --ignore-not-found > "$result_dir/logs/locust.log" 2>&1
kubectl --context="$locust_cluster" -n "$namespace" delete configmap headlight-calibration-locust --ignore-not-found >> "$result_dir/logs/locust.log" 2>&1
kubectl --context="$locust_cluster" -n "$namespace" create configmap headlight-calibration-locust \
  --from-file=trainticket_paper.py="$repo_root/experiments/testbed/trainticket-paper/trainticket_paper.py" \
  --from-file=load_profile.csv="$result_dir/raw/load_profile.csv" >> "$result_dir/logs/locust.log" 2>&1
kubectl --context="$locust_cluster" -n "$namespace" apply -f - >> "$result_dir/logs/locust.log" 2>&1 <<YAML
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
        image: $locust_image
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh", "-lc"]
        args:
        - |
          /opt/venv/bin/locust --headless -f /app/trainticket_paper.py --run-time "${duration_s}s" --exit-code-on-error 0 --csv /tmp/headlight-calibration
          rc="\$?"
          echo __HEADLIGHT_REQUEST_CSV_BEGIN__
          cat /app/headlight_requests.csv 2>/dev/null || true
          echo __HEADLIGHT_REQUEST_CSV_END__
          echo __HEADLIGHT_LOCUST_STATS_BEGIN__
          cat /tmp/headlight-calibration_stats.csv 2>/dev/null || true
          echo __HEADLIGHT_LOCUST_STATS_END__
          exit "\$rc"
        env:
        - name: HEADLIGHT_REQUEST_CSV
          value: /app/headlight_requests.csv
        - name: HEADLIGHT_LOAD_SHAPE
          value: /app/load_profile.csv
        - name: HEADLIGHT_TRAINTICKET_MODE
          value: "$locust_mode"
        - name: HEADLIGHT_TRAINTICKET_NAMESPACE
          value: "$namespace"
        volumeMounts:
        - name: locustfile
          mountPath: /app/trainticket_paper.py
          subPath: trainticket_paper.py
        - name: locustfile
          mountPath: /app/load_profile.csv
          subPath: load_profile.csv
      volumes:
      - name: locustfile
        configMap:
          name: headlight-calibration-locust
YAML
started_at="$(date +%s)"
if ! kubectl --context="$locust_cluster" -n "$namespace" wait --for=condition=Complete "job/$job" --timeout="${HEADLIGHT_CALIBRATION_WAIT_TIMEOUT:-$((duration_s + 1800))s}" >> "$result_dir/logs/locust.log" 2>&1; then
  pod="$(kubectl --context="$locust_cluster" -n "$namespace" get pod -l job-name="$job" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$pod" ]]; then
    kubectl --context="$locust_cluster" -n "$namespace" describe pod "$pod" > "$result_dir/logs/locust-pod-describe.txt" || true
    kubectl --context="$locust_cluster" -n "$namespace" logs "$pod" > "$result_dir/logs/locust-pod.log" || true
  fi
  exit 1
fi
finished_at="$(date +%s)"
pod="$(kubectl --context="$locust_cluster" -n "$namespace" get pod -l job-name="$job" -o jsonpath='{.items[0].metadata.name}')"
kubectl --context="$locust_cluster" -n "$namespace" logs "$pod" > "$result_dir/logs/locust-pod.log"
awk '/__HEADLIGHT_REQUEST_CSV_BEGIN__/{flag=1;next}/__HEADLIGHT_REQUEST_CSV_END__/{flag=0}flag' "$result_dir/logs/locust-pod.log" > "$result_dir/raw/request_events.csv"
awk '/__HEADLIGHT_LOCUST_STATS_BEGIN__/{flag=1;next}/__HEADLIGHT_LOCUST_STATS_END__/{flag=0}flag' "$result_dir/logs/locust-pod.log" > "$result_dir/raw/locust_stats.csv"

echo "== Collecting Kubernetes and Prometheus telemetry =="
for cluster in $clusters; do
  kubectl --context="$cluster" -n "$namespace" get events -o json > "$result_dir/raw/kubernetes_events_${cluster}.json" || true
  kubectl --context="$cluster" -n "$namespace" get pods -o json > "$result_dir/raw/pods_${cluster}.json" || true
  kubectl --context="$cluster" -n "$namespace" get deployments -o json > "$result_dir/raw/deployments_${cluster}.json" || true
done
"$python_bin" - "$clusters" "$prometheus_namespace" "$prometheus_service" "$prometheus_port" "$namespace" "$started_at" "$finished_at" "$sampling_interval" "$result_dir/raw/prometheus_range_data.json" <<'PY'
import json
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request

clusters, prometheus_namespace, prometheus_service, base_port, namespace, start, end, interval, output = sys.argv[1:10]
payload = {}
queries = {
    "pod_count": f'kube_pod_status_ready{{namespace="{namespace}",condition="true"}} == 1',
    "cpu_usage_cores": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!="",pod!=""}}[60s]))',
    "cpu_utilization": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!="",pod!=""}}[60s])) / clamp_min(sum by (pod) (kube_pod_container_resource_requests{{namespace="{namespace}",resource="cpu"}}), 0.001)',
    "request_rate": f'sum by (destination_workload) (rate(istio_requests_total{{destination_workload_namespace="{namespace}"}}[60s]))',
    "response_time_sum_ms": f'sum by (destination_workload) (rate(istio_request_duration_milliseconds_sum{{destination_workload_namespace="{namespace}"}}[60s]))',
    "response_time_count": f'sum by (destination_workload) (rate(istio_request_duration_milliseconds_count{{destination_workload_namespace="{namespace}"}}[60s]))',
    "p95_latency_ms": f'histogram_quantile(0.95, sum by (le, destination_workload) (rate(istio_request_duration_milliseconds_bucket{{destination_workload_namespace="{namespace}"}}[60s])))',
    "error_rate": f'sum by (destination_workload) (rate(istio_requests_total{{destination_workload_namespace="{namespace}",response_code!~"2..|3.."}}[60s])) / clamp_min(sum by (destination_workload) (rate(istio_requests_total{{destination_workload_namespace="{namespace}"}}[60s])), 0.001)',
    "request_edges": f'sum by (source_workload, destination_workload) (rate(istio_requests_total{{destination_workload_namespace="{namespace}",source_workload!="unknown",destination_workload!="unknown"}}[60s]))',
}
for index, cluster in enumerate(clusters.split()):
    port = int(base_port) + index
    log = open(output.replace("prometheus_range_data.json", f"prometheus-port-forward-{cluster}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["kubectl", "--context", cluster, "-n", prometheus_namespace, "port-forward", f"svc/{prometheus_service}", f"{port}:9090"],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/-/ready", timeout=1).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Prometheus port-forward did not become ready for {cluster}")
        for metric, query in queries.items():
            params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": f"{interval}s"})
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/query_range?{params}", timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            payload[f"{cluster}:{metric}"] = {"cluster": cluster, "metric": metric, "query": query, "response": body}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

"$python_bin" "$repo_root/experiments/scripts/build_calibration_dataset.py" \
  --prometheus-range-data "$result_dir/raw/prometheus_range_data.json" \
  --service-file "$repo_root/ai-module/configs/service_profiles/${path}.json" \
  --output-csv "$result_dir/train-ticket-calibration.csv" \
  --feature-history-dir "$result_dir/feature_history" \
  --edge-csv-output "$result_dir/istio-edges.csv" \
  --warmup "$warmup" \
  --manifest-output "$result_dir/calibration_manifest.json" \
  > "$result_dir/logs/build-calibration-dataset.log"

"$python_bin" "$repo_root/experiments/scripts/build_static_adjacency.py" \
  --input "$result_dir/istio-edges.csv" \
  --service-profile "$repo_root/ai-module/configs/service_profiles/${path}.json" \
  --output "$result_dir/static-adjacency.json" \
  > "$result_dir/logs/build-static-adjacency.log"

"$python_bin" - "$result_dir/calibration_manifest.json" "$duration" "$warmup" "$path" "$clusters" "$locust_mode" "$started_at" "$finished_at" "$HEADLIGHT_PROMETHEUS_RETENTION" "$phase" "$seed" <<'PY'
import json
import sys
path, duration, warmup, ticket_path, clusters, locust_mode, started, finished, retention, phase, seed = sys.argv[1:12]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest.update({
    "duration": duration,
    "warmup": warmup,
    "trainticket_path": ticket_path,
    "active_clusters": clusters.split(),
    "locust_mode": locust_mode,
    "started_at_unix": int(started),
    "finished_at_unix": int(finished),
    "prometheus_retention": retention,
    "phase": phase,
    "seed": int(seed),
    "models_frozen_during_experiments": True,
})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY

echo "Calibration result bundle: $result_dir"
