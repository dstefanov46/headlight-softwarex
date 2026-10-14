#!/usr/bin/env bash
set -euo pipefail

# Restart the paper testbed after a host reboot. Existing Kind containers are
# reused when possible; the destructive rebuild is used only when the testbed
# is missing or cannot be recovered.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
kind_node_image="${KIND_NODE_IMAGE:-kindest/node:v1.32.0}"
clusters=(karmada-host cluster1 cluster2 cluster3)

export HEADLIGHT_KUBECONFIG="$kubeconfig"
export KUBECONFIG="$kubeconfig"
cd "$repo_root"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "$1 is required" >&2
    exit 1
  }
}

for command_name in docker kind kubectl; do
  need_cmd "$command_name"
done

mkdir -p "$(dirname "$kubeconfig")"

instances="$(cat /proc/sys/fs/inotify/max_user_instances)"
watches="$(cat /proc/sys/fs/inotify/max_user_watches)"
if [[ "$instances" -lt 8192 || "$watches" -lt 1048576 ]]; then
  echo "host inotify limits are too low: instances=$instances watches=$watches" >&2
  echo "run: sudo $repo_root/experiments/install_host_inotify_limits.sh" >&2
  exit 1
fi

kind_cluster_exists() {
  kind get clusters 2>/dev/null | grep -Fqx "$1"
}

start_existing_containers() {
  local cluster state
  for cluster in "${clusters[@]}"; do
    if ! kind_cluster_exists "$cluster"; then
      return 1
    fi
    state="$(docker inspect -f '{{.State.Running}}' "${cluster}-control-plane" 2>/dev/null || true)"
    if [[ "$state" == "false" ]]; then
      echo "Starting stopped container ${cluster}-control-plane..."
      docker start "${cluster}-control-plane" >/dev/null
    elif [[ "$state" != "true" ]]; then
      return 1
    fi
  done

  # Refresh endpoint addresses after a restart and restore the paper context
  # names expected by the experiment scripts.
  for cluster in "${clusters[@]}"; do
    kind export kubeconfig --name "$cluster" --kubeconfig "$kubeconfig" >/dev/null
    kubectl --kubeconfig "$kubeconfig" config rename-context "kind-$cluster" "$cluster" >/dev/null 2>&1 || true
  done
  return 0
}

testbed_ready() {
  local cluster
  for cluster in "${clusters[@]}"; do
    kubectl --kubeconfig "$kubeconfig" --context="$cluster" get --raw=/readyz >/dev/null 2>&1 || return 1
  done
  kubectl --kubeconfig "$kubeconfig" --context=karmada-apiserver get --raw=/readyz >/dev/null 2>&1 || return 1
  return 0
}

if ! start_existing_containers; then
  echo "Existing Kind/Karmada testbed is missing or not recoverable. Rebuilding it..."
  HEADLIGHT_KUBECONFIG="$kubeconfig" \
    KIND_NODE_IMAGE="$kind_node_image" \
    "$repo_root/experiments/rebuild_local_testbed.sh"
else
  HEADLIGHT_KUBECONFIG="$kubeconfig" \
    "$repo_root/experiments/repair_local_testbed_endpoints.sh"
  if ! testbed_ready; then
    echo "Existing Kind/Karmada testbed did not recover after endpoint repair." >&2
    echo "Refusing an implicit destructive rebuild; inspect it or run rebuild_local_testbed.sh explicitly." >&2
    exit 1
  fi
  echo "Reused and repaired existing Kind/Karmada testbed."
fi

echo "== Waiting for required control-plane contexts =="
for context in karmada-apiserver karmada-host cluster1 cluster2 cluster3; do
  kubectl --kubeconfig "$kubeconfig" --context="$context" get --raw=/readyz >/dev/null
done

echo "== Refreshing Prometheus monitoring =="
HEADLIGHT_KUBECONFIG="$kubeconfig" \
  "$repo_root/experiments/install_prometheus_monitoring.sh"

echo "== Deploying advanced-ticket TrainTicket prerequisites =="
HEADLIGHT_KUBECONFIG="$kubeconfig" \
HEADLIGHT_TRAINTICKET_PATH=advanced-ticket \
HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE=capacity-vm \
  "$repo_root/experiments/deploy_minimal_trainticket.sh"

if [[ "${HEADLIGHT_BUILD_LOCAL_IMAGES:-1}" == "1" ]]; then
  echo "== Building and loading Headlight runtime image =="
  docker build \
    -f ai-module/ai-module/Dockerfile \
    -t local/headlight-ai-module:dev \
    ai-module
  kind load docker-image local/headlight-ai-module:dev --name cluster3

  echo "== Building and loading LSTM runtime image =="
  docker build \
    -f ai-module/baselines/Dockerfile.lstm \
    -t local/headlight-lstm:dev \
    .
  kind load docker-image local/headlight-lstm:dev --name cluster3
fi

echo "== Refreshing the shared TrainTicket gateway for current Kind node addresses =="
HEADLIGHT_KUBECONFIG="$kubeconfig" \
  "$repo_root/experiments/scripts/deploy_trainticket_gateway.sh"

echo "== Checking the shared gateway end to end =="
HEADLIGHT_KUBECONFIG="$kubeconfig" \
  "$repo_root/experiments/scripts/smoke_trainticket_gateway.sh"

echo
echo "Paper testbed is ready."
echo "KUBECONFIG=$kubeconfig"
echo "Next Headlight retry:"
echo "  python3 experiments/validate_softwarex_evaluation.py"
