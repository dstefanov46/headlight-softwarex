#!/usr/bin/env bash
set -euo pipefail

# Bootstrap helper for the local measurement testbed. It creates/validates the Kind
# member clusters expected by the experiment runner and leaves Karmada
# installation to the checked-in dist manifests.

contexts=(karmada-apiserver karmada-host cluster1 cluster2 cluster3)
kind_node_image="${KIND_NODE_IMAGE:-kindest/node:v1.32.0}"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$(pwd)/.kube/headlight-testbed.config}"

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is required" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

inotify_instances="$(cat /proc/sys/fs/inotify/max_user_instances)"
if [[ "$inotify_instances" -lt 1024 ]]; then
  echo "fs.inotify.max_user_instances=$inotify_instances is too low for a multi-cluster Kind testbed." >&2
  echo "Raise it on the host first, for example: sudo sysctl fs.inotify.max_user_instances=1024" >&2
  exit 1
fi

mkdir -p "$(dirname "$kubeconfig")"
export KUBECONFIG="$kubeconfig"

for cluster in karmada-host cluster1 cluster2 cluster3; do
  if ! kind get clusters | grep -qx "$cluster"; then
    kind create cluster --name "$cluster" --image "$kind_node_image" --kubeconfig "$kubeconfig" --wait 180s
  fi
  kubectl config rename-context "kind-$cluster" "$cluster" >/dev/null 2>&1 || true
done

echo "Expected contexts:"
printf '  %s\n' "${contexts[@]}"
echo
echo "KUBECONFIG=$kubeconfig"
echo "Install Karmada host/apiserver using controlled-replica-propagation/dist and actions-handler/dist, then rerun:"
echo "  kubectl config get-contexts -o name"
