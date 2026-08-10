#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
expected_pod_count="${HEADLIGHT_ADVANCED_EXPECTED_POD_COUNT:-16}"

export HEADLIGHT_KUBECONFIG="$kubeconfig"
export KUBECONFIG="$kubeconfig"

cd "$repo_root"

echo "Using kubeconfig: $kubeconfig"
echo

echo "== Baseline cluster health =="
kubectl --context=karmada-apiserver get clusters
for cluster in cluster2 cluster3; do
  echo
  echo "Pods in $cluster/$namespace:"
  kubectl --context="$cluster" -n "$namespace" get pods
done

echo
echo "== Deploying advanced TrainTicket slice =="
HEADLIGHT_TRAINTICKET_PATH=advanced-ticket \
HEADLIGHT_KUBECONFIG="$kubeconfig" \
  "$repo_root/experiments/deploy_minimal_trainticket.sh"

advanced_selector='app in (ts-auth,ts-route,ts-travel-plan,ts-travel,ts-travel2,ts-route-plan,ts-station,ts-train,ts-price,ts-auth-mongo,ts-route-mongo,ts-travel-mongo,ts-travel2-mongo,ts-station-mongo,ts-train-mongo,ts-price-mongo)'

echo
echo "== Waiting for advanced TrainTicket pods =="
for cluster in cluster2 cluster3; do
  echo "Waiting for at least $expected_pod_count selected pods in $cluster/$namespace..."
  for _ in $(seq 1 120); do
    count="$(kubectl --context="$cluster" -n "$namespace" get pods \
      -l "$advanced_selector" \
      --no-headers 2>/dev/null | wc -l)"
    if [[ "$count" -ge "$expected_pod_count" ]]; then
      break
    fi
    sleep 5
  done

  count="$(kubectl --context="$cluster" -n "$namespace" get pods \
    -l "$advanced_selector" \
    --no-headers 2>/dev/null | wc -l)"
  if [[ "$count" -lt "$expected_pod_count" ]]; then
    echo "Expected at least $expected_pod_count pods in $cluster/$namespace, found $count" >&2
    kubectl --context="$cluster" -n "$namespace" get pods -o wide >&2 || true
    exit 1
  fi

  kubectl --context="$cluster" -n "$namespace" wait pod \
    --for=condition=Ready \
    -l "$advanced_selector" \
    --timeout=10m

  kubectl --context="$cluster" -n "$namespace" get pods \
    -l "$advanced_selector" \
    -o custom-columns='NAME:.metadata.name,READY:.status.containerStatuses[*].ready,CONTAINERS:.spec.containers[*].name'
done

echo
echo "== Refreshing Prometheus monitoring =="
HEADLIGHT_KUBECONFIG="$kubeconfig" "$repo_root/experiments/install_prometheus_monitoring.sh"

echo
echo "== Running constrained minimal smoke =="
HEADLIGHT_KUBECONFIG="$kubeconfig" "$repo_root/experiments/run_minimal_trainticket_smoke.sh"

echo
echo "== Dry-running configured experiment =="
python3 "$repo_root/experiments/run_experiment.py" \
  --run "$repo_root/experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml" \
  --seeds all \
  --dry-run

echo
echo "Next testbed steps completed."
