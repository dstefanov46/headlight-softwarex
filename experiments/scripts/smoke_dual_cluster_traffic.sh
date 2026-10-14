#!/usr/bin/env bash
# Short, opt-in two-cluster TrainTicket traffic smoke. No controller actuation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
duration="${HEADLIGHT_DUAL_SMOKE_DURATION:-90s}"
users_per_cluster="${HEADLIGHT_DUAL_SMOKE_USERS_PER_CLUSTER:-2}"
result_dir="${HEADLIGHT_DUAL_SMOKE_RESULT_DIR:-$repo_root/results/headlight-v2-live/dual-cluster-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$duration" =~ ^[1-9][0-9]*s$ ]]; then
  echo "HEADLIGHT_DUAL_SMOKE_DURATION must be a positive number of seconds" >&2
  exit 2
fi
if [[ ! "$users_per_cluster" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEADLIGHT_DUAL_SMOKE_USERS_PER_CLUSTER must be a positive integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists: $result_dir" >&2
  exit 2
fi
if [[ ! -f "$kubeconfig" ]]; then
  echo "Kubeconfig not found: $kubeconfig" >&2
  exit 2
fi
export KUBECONFIG="$kubeconfig"

# The reused smoke runner has fixed Job/ConfigMap names and deletes them before
# creation. Refuse to touch either cluster if those names already exist.
for cluster in cluster2 cluster3; do
  kubectl --context="$cluster" -n "$namespace" get service ts-auth-service ts-route-service >/dev/null
  kubectl --context="$cluster" -n monitoring get service prometheus-kube-prometheus-prometheus >/dev/null
  for resource in job/headlight-minimal-smoke configmap/headlight-minimal-locust; do
    if [[ -n "$(kubectl --context="$cluster" -n "$namespace" get "$resource" --ignore-not-found -o name)" ]]; then
      echo "$resource already exists in $cluster/$namespace; refusing to replace it" >&2
      exit 2
    fi
  done
done

mkdir -p "$result_dir"
for cluster in cluster2 cluster3; do
  mkdir -p "$result_dir/$cluster"
done

(
  HEADLIGHT_SMOKE_CLUSTER=cluster2 \
  HEADLIGHT_SMOKE_DURATION="$duration" \
  HEADLIGHT_SMOKE_USERS="$users_per_cluster" \
  HEADLIGHT_SMOKE_WAIT_TIMEOUT=5m \
  HEADLIGHT_REQUIRE_PROMETHEUS=1 \
  HEADLIGHT_PROMETHEUS_PORT=39120 \
  HEADLIGHT_SMOKE_RESULT_DIR="$result_dir/cluster2" \
    "$repo_root/experiments/run_minimal_trainticket_smoke.sh"
) >"$result_dir/cluster2/runner.log" 2>&1 &
cluster2_pid=$!
(
  HEADLIGHT_SMOKE_CLUSTER=cluster3 \
  HEADLIGHT_SMOKE_DURATION="$duration" \
  HEADLIGHT_SMOKE_USERS="$users_per_cluster" \
  HEADLIGHT_SMOKE_WAIT_TIMEOUT=5m \
  HEADLIGHT_REQUIRE_PROMETHEUS=1 \
  HEADLIGHT_PROMETHEUS_PORT=39121 \
  HEADLIGHT_SMOKE_RESULT_DIR="$result_dir/cluster3" \
    "$repo_root/experiments/run_minimal_trainticket_smoke.sh"
) >"$result_dir/cluster3/runner.log" 2>&1 &
cluster3_pid=$!

cluster2_status=0
cluster3_status=0
wait "$cluster2_pid" || cluster2_status=$?
wait "$cluster3_pid" || cluster3_status=$?
if (( cluster2_status != 0 || cluster3_status != 0 )); then
  echo "Dual-cluster smoke failed: cluster2=$cluster2_status cluster3=$cluster3_status; logs: $result_dir" >&2
  exit 1
fi

for cluster in cluster2 cluster3; do
  request_csv="$result_dir/$cluster/raw/request_events.csv"
  prometheus_json="$result_dir/$cluster/raw/prometheus_range_data.json"
  for endpoint in login route_query; do
    successes="$(awk -F, -v endpoint="$endpoint" 'NR > 1 && $3 == endpoint && $4 ~ /^2[0-9][0-9]$/ && $6 == "" {count++} END {print count+0}' "$request_csv")"
    if (( successes == 0 )); then
      echo "$cluster had no successful $endpoint requests; inspect $request_csv" >&2
      exit 1
    fi
    echo "DUAL_CLUSTER_REQUEST_PASS cluster=$cluster endpoint=$endpoint successful_requests=$successes"
  done
  if ! jq -e 'any(.istio_request_rate.response.data.result[]?.values[]; (try (.[1] | tonumber) catch 0) > 0)' "$prometheus_json" >/dev/null; then
    echo "$cluster had no positive Istio request-rate sample; inspect $prometheus_json" >&2
    exit 1
  fi
  echo "DUAL_CLUSTER_TRAFFIC_PASS cluster=$cluster istio_request_rate=positive"
done
echo "Evidence: $result_dir"
