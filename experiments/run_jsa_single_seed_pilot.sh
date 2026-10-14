#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
run_config="${HEADLIGHT_JSA_RUN_CONFIG:-experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml}"
seed="${HEADLIGHT_JSA_SEED:-101}"
trainticket_path="${HEADLIGHT_JSA_TRAINTICKET_PATH:-login-route}"
resource_profile="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-local-tiny}"

# This is a live pilot, not the final one-hour paper run. Override when ready:
#   HEADLIGHT_JSA_PILOT_DURATION=60m HEADLIGHT_JSA_PILOT_USERS=25 ./experiments/run_jsa_single_seed_pilot.sh
duration="${HEADLIGHT_JSA_PILOT_DURATION:-10m}"
users="${HEADLIGHT_JSA_PILOT_USERS:-5}"
spawn_rate="${HEADLIGHT_JSA_PILOT_SPAWN_RATE:-2}"
expected_pod_count="${HEADLIGHT_ADVANCED_EXPECTED_POD_COUNT:-18}"

export HEADLIGHT_KUBECONFIG="$kubeconfig"
export KUBECONFIG="$kubeconfig"

cd "$repo_root"

run_name="$(
  python3 - "$run_config" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path.cwd() / path
data = yaml.safe_load(path.read_text(encoding="utf-8"))
print(data["metadata"]["name"])
PY
)"
timestamp="$(date +%Y%m%d-%H%M%S)"
result_dir="${HEADLIGHT_JSA_PILOT_RESULT_DIR:-$repo_root/results/jsa-pilot/$run_name/seed-$seed-$timestamp}"

if [[ "$trainticket_path" = "advanced-ticket" ]]; then
  workload_selector='app in (ts-auth,ts-route,ts-travel-plan,ts-travel,ts-travel2,ts-route-plan,ts-basic,ts-station,ts-ticketinfo,ts-train,ts-price,ts-auth-mongo,ts-route-mongo,ts-travel-mongo,ts-travel2-mongo,ts-station-mongo,ts-train-mongo,ts-price-mongo)'
  expected_pod_count="${HEADLIGHT_JSA_EXPECTED_POD_COUNT:-18}"
elif [[ "$trainticket_path" = "login-route" ]]; then
  workload_selector='app in (ts-auth,ts-route,ts-auth-mongo,ts-route-mongo)'
  expected_pod_count="${HEADLIGHT_JSA_EXPECTED_POD_COUNT:-4}"
else
  echo "unsupported HEADLIGHT_JSA_TRAINTICKET_PATH: $trainticket_path" >&2
  exit 1
fi

require_tool() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required tool not found: $command_name" >&2
    exit 1
  fi
}

require_context() {
  local context_name="$1"
  if ! kubectl config get-contexts -o name | grep -qx "$context_name"; then
    echo "missing kube context: $context_name" >&2
    exit 1
  fi
}

wait_for_workload_pods() {
  local cluster="$1"
  echo "Waiting for at least $expected_pod_count TrainTicket pods in $cluster/$namespace..."
  for _ in $(seq 1 120); do
    count="$(kubectl --context="$cluster" -n "$namespace" get pods \
      -l "$workload_selector" \
      --no-headers 2>/dev/null | wc -l)"
    if [[ "$count" -ge "$expected_pod_count" ]]; then
      break
    fi
    sleep 5
  done

  count="$(kubectl --context="$cluster" -n "$namespace" get pods \
    -l "$workload_selector" \
    --no-headers 2>/dev/null | wc -l)"
  if [[ "$count" -lt "$expected_pod_count" ]]; then
    echo "Expected at least $expected_pod_count pods in $cluster/$namespace, found $count" >&2
    kubectl --context="$cluster" -n "$namespace" get pods -o wide >&2 || true
    exit 1
  fi

  kubectl --context="$cluster" -n "$namespace" wait pod \
    --for=condition=Ready \
    -l "$workload_selector" \
    --timeout=10m
}

echo "== JSA single-seed live pilot =="
echo "Run config: $run_config"
echo "Run name:   $run_name"
echo "Seed:       $seed"
echo "TT path:    $trainticket_path"
echo "Resources:  $resource_profile"
echo "Duration:   $duration"
echo "Users:      $users"
echo "Result dir: $result_dir"
echo

require_tool kubectl
require_tool python3
require_tool helm
require_tool curl

for context in karmada-apiserver karmada-host cluster1 cluster2 cluster3; do
  require_context "$context"
done

echo "== Cluster health =="
kubectl --context=karmada-apiserver get clusters
for cluster in cluster2 cluster3; do
  kubectl --context="$cluster" get nodes
done

echo
echo "== Ensuring TrainTicket slice =="
HEADLIGHT_TRAINTICKET_PATH="$trainticket_path" \
HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE="$resource_profile" \
HEADLIGHT_KUBECONFIG="$kubeconfig" \
  "$repo_root/experiments/deploy_minimal_trainticket.sh"

for cluster in cluster2 cluster3; do
  wait_for_workload_pods "$cluster"
  kubectl --context="$cluster" -n "$namespace" get pods \
    -l "$workload_selector" \
    -o custom-columns='NAME:.metadata.name,READY:.status.containerStatuses[*].ready,CONTAINERS:.spec.containers[*].name'
done

echo
echo "== Ensuring Prometheus and Istio sidecar scraping =="
HEADLIGHT_KUBECONFIG="$kubeconfig" "$repo_root/experiments/install_prometheus_monitoring.sh"
kubectl --context=cluster2 -n monitoring get podmonitor trainticket-istio-sidecars

mkdir -p "$result_dir/raw" "$result_dir/logs" "$result_dir/summaries"

echo
echo "== Writing pilot metadata and planned load profile =="
python3 - "$run_config" "$seed" "$duration" "$users" "$spawn_rate" "$result_dir" "$trainticket_path" "$resource_profile" <<'PY'
import json
import sys
from pathlib import Path

from experiments.lib.config import resolve_run, write_yaml
from experiments.lib.workloads import generate_shape, write_shape_csv

run_config, seed, duration, users, spawn_rate, result_dir, trainticket_path, resource_profile = sys.argv[1:9]
seed = int(seed)
result = Path(result_dir)
manifest = resolve_run(Path(run_config))
interval = int(manifest["spec"]["run"].get("sampling_interval_seconds", 15))
points = generate_shape(manifest["spec"]["workload"], seed, interval)

write_yaml(result / "manifest.yaml", manifest)
write_shape_csv(result / "raw" / "configured_load_profile.csv", points)
(result / "pilot_metadata.json").write_text(
    json.dumps(
        {
            "mode": "live-minimal-trainticket-login-route",
            "note": (
                "This is a measured JSA pilot on the stable local testbed. "
                "It validates live request, Kubernetes event, and Prometheus capture before full paper runs."
            ),
            "run_name": manifest["metadata"]["name"],
            "seed": seed,
            "pilot_duration": duration,
            "pilot_users": int(users),
            "pilot_spawn_rate": int(spawn_rate),
            "trainticket_path": trainticket_path,
            "trainticket_resource_profile": resource_profile,
            "configured_run_duration_seconds": manifest["spec"]["run"]["duration_seconds"],
            "configured_workload": manifest["spec"]["workload"]["pattern"],
            "configured_controller": manifest["spec"]["controller"]["controller"],
            "configured_topology": manifest["spec"]["topology"]["mode"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo
echo "== Running live TrainTicket pilot load =="
HEADLIGHT_KUBECONFIG="$kubeconfig" \
HEADLIGHT_SMOKE_DURATION="$duration" \
HEADLIGHT_SMOKE_USERS="$users" \
HEADLIGHT_SMOKE_SPAWN_RATE="$spawn_rate" \
HEADLIGHT_SMOKE_RESULT_DIR="$result_dir" \
  "$repo_root/experiments/run_minimal_trainticket_smoke.sh"

echo
echo "== Summarizing pilot bundle =="
python3 - "$result_dir" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

result = Path(sys.argv[1])
request_csv = result / "raw" / "request_events.csv"
prometheus_json = result / "raw" / "prometheus_range_data.json"
summary_json = result / "summaries" / "pilot_summary.json"

rows = list(csv.DictReader(request_csv.open(encoding="utf-8")))
latencies = [float(row["response_time_ms"]) for row in rows if row.get("response_time_ms")]
failures = [row for row in rows if row.get("exception")]
statuses = {}
for row in rows:
    status = row.get("status") or "unknown"
    statuses[status] = statuses.get(status, 0) + 1

prom = json.loads(prometheus_json.read_text(encoding="utf-8"))
prometheus_series = {}
for name, item in prom.items():
    result_set = item.get("response", {}).get("data", {}).get("result", [])
    prometheus_series[name] = {
        "series": len(result_set),
        "samples": sum(len(series.get("values", [])) for series in result_set),
    }

summary = {
    "request_rows": len(rows),
    "failure_rows": len(failures),
    "statuses": statuses,
    "latency_ms": {
        "min": min(latencies) if latencies else 0.0,
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
    },
    "prometheus": prometheus_series,
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
if len(rows) < 2:
    raise SystemExit("pilot produced no request rows")
if failures:
    raise SystemExit("pilot produced failed request rows")
for required in ["up", "istio_request_rate", "istio_p95_latency_ms"]:
    if prometheus_series.get(required, {}).get("samples", 0) == 0:
        raise SystemExit(f"pilot missing Prometheus samples for {required}")
PY

echo
echo "JSA single-seed live pilot completed: $result_dir"
