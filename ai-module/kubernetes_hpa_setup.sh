#!/bin/sh
set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
contexts="${HEADLIGHT_HPA_CONTEXTS:-cluster2 cluster3}"
primary_context="${contexts%% *}"
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-full}"
resource_profile="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-capacity-vm}"
rendered_hpa_dir="${HEADLIGHT_RENDERED_HPA_DIR:-}"

control_plane_ip() {
  name="$1"
  if command -v docker >/dev/null 2>&1; then
    docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$name" 2>/dev/null || true
  fi
}

control_plane_server() {
  name="$1"
  ip=$(control_plane_ip "$name")
  if [ -n "$ip" ]; then
    printf 'https://%s:6443\n' "$ip"
  fi
}

set_cluster_server() {
  kubeconfig="$1"
  cluster="$2"
  server="$3"
  if [ -n "$server" ]; then
    kubectl --kubeconfig="$kubeconfig" config set-cluster "$cluster" --server="$server" >/dev/null
  fi
}

prepare_pod_kubeconfig() {
  default_source_config="$repo_root/.kube/headlight-members-internal.config"
  if [ ! -f "$default_source_config" ]; then
    default_source_config="${HEADLIGHT_KUBECONFIG:-$HOME/.kube/config}"
  fi
  source_config="${HEADLIGHT_HOST_KUBECONFIG:-${HEADLIGHT_POD_KUBECONFIG:-$default_source_config}}"
  pod_config="$1"
  if [ ! -f "$source_config" ]; then
    echo "Missing source kubeconfig for controller pod: $source_config" >&2
    return 1
  fi
  cp "$source_config" "$pod_config"

  cluster1_server="${HEADLIGHT_POD_CLUSTER1_SERVER:-}"
  cluster2_server="${HEADLIGHT_POD_CLUSTER2_SERVER:-}"
  cluster3_server="${HEADLIGHT_POD_CLUSTER3_SERVER:-}"
  karmada_host_server="${HEADLIGHT_POD_KARMADA_HOST_SERVER:-}"

  if [ -z "$cluster1_server" ]; then cluster1_server=$(control_plane_server cluster1-control-plane); fi
  if [ -z "$cluster2_server" ]; then cluster2_server=$(control_plane_server cluster2-control-plane); fi
  if [ -z "$cluster3_server" ]; then cluster3_server=$(control_plane_server cluster3-control-plane); fi
  if [ -z "$karmada_host_server" ]; then karmada_host_server=$(control_plane_server karmada-host-control-plane); fi

  set_cluster_server "$pod_config" kind-cluster1 "$cluster1_server"
  set_cluster_server "$pod_config" kind-cluster2 "$cluster2_server"
  set_cluster_server "$pod_config" kind-cluster3 "$cluster3_server"
  set_cluster_server "$pod_config" kind-karmada-host "$karmada_host_server"
  set_cluster_server "$pod_config" karmada-apiserver "${HEADLIGHT_POD_KARMADA_APISERVER_SERVER:-}"

  for context in cluster1 cluster2 cluster3; do
    if ! kubectl --kubeconfig="$pod_config" --context="$context" get --raw=/readyz >/dev/null 2>&1; then
      echo "Generated controller pod kubeconfig cannot reach $context; set HEADLIGHT_POD_*_SERVER overrides or rebuild the testbed." >&2
      kubectl --kubeconfig="$pod_config" --context="$context" get --raw=/readyz >&2 || true
      return 1
    fi
  done
}

echo "Creating namespace and enabling Istio injection..."
kubectl create namespace trainticket --context=karmada-apiserver || true
kubectl label namespace trainticket --context=karmada-apiserver istio-injection=enabled --overwrite

if [ "${HEADLIGHT_SKIP_CONTROLLER_WORKLOAD_DEPLOY:-0}" != "1" ]; then
for cluster in $contexts; do
  echo "Switching context to $cluster..."

  echo "Applying DB services in $cluster..."
  kubectl --context="$cluster" apply -f deployments/db-services-deployment.yaml -n trainticket

  echo "Applying application services in $cluster..."
  kubectl --context="$cluster" apply -f deployments/application-services-deployment.yaml -n trainticket

  echo "Applying Istio gateway in $cluster..."
  kubectl --context="$cluster" apply -f deployments/istio-gateway.yaml -n trainticket
done
else
echo "Keeping the workload deployment prepared by the experiment runner"
fi

echo "Applying roles..."
for cluster in $contexts; do
  echo "Applying roles in $cluster..."
  for role_file in roles/*.yaml; do
    sed "s/namespace: nephele/namespace: trainticket/g" "$role_file" | kubectl apply --context="$cluster" -f -
  done
done

pod_kubeconfig=$(mktemp)
trap 'rm -f "$pod_kubeconfig"' EXIT
prepare_pod_kubeconfig "$pod_kubeconfig"

if [ "${HEADLIGHT_SKIP_LEGACY_LOCUST:-1}" != "1" ]; then
echo "Deploying Locust master and workers..."
for cluster in $contexts; do
  kubectl --context="$cluster" create secret generic pod-kubeconfig-locust --from-file=config="$pod_kubeconfig" -n trainticket --dry-run=client -o yaml | kubectl --context="$cluster" apply -f -
done
helm --kube-context "$primary_context" upgrade --install locust ./locust-helm-chart --namespace trainticket
else
echo "Skipping legacy Locust Helm deployment because HEADLIGHT_SKIP_LEGACY_LOCUST=1"
fi

for cluster in $contexts; do
  echo "Switching context to $cluster for autoscaler setup..."
  kubectl --context="$cluster" create secret generic pod-kubeconfig-trainticket --from-file=config="$pod_kubeconfig" -n trainticket --dry-run=client -o yaml | kubectl --context="$cluster" apply -f -

  echo "Applying HPA config in $cluster..."
  if [ "${HEADLIGHT_MATCH_TOPOLOGY_CAPACITY:-0}" = "1" ]; then
    cluster_max_replicas=$(python3 - "$cluster" "$repo_root" <<'PY'
import sys

sys.path.insert(0, sys.argv[2])
from experiments.lib.protocol import replica_limit_for_cluster, replica_limits

cluster = sys.argv[1]
try:
    print(replica_limit_for_cluster(cluster))
except KeyError:
    print(int(replica_limits()["aggregate_max_replicas_per_service"]) // 2)
PY
)
    if [ -n "$rendered_hpa_dir" ]; then
      mkdir -p "$rendered_hpa_dir"
      rendered_hpa="$rendered_hpa_dir/hpa-$cluster.yaml"
      python3 "$repo_root/experiments/scripts/render_hpa_autoscalers.py" \
        --namespace trainticket \
        --trainticket-path "$trainticket_path" \
        --resource-profile "$resource_profile" \
        --max-replicas "$cluster_max_replicas" \
        --output "$rendered_hpa"
      kubectl --context="$cluster" apply -f "$rendered_hpa" -n trainticket
    else
      python3 "$repo_root/experiments/scripts/render_hpa_autoscalers.py" \
        --namespace trainticket \
        --trainticket-path "$trainticket_path" \
        --resource-profile "$resource_profile" \
        --max-replicas "$cluster_max_replicas" | \
        kubectl --context="$cluster" apply -f - -n trainticket
    fi
  else
    kubectl --context="$cluster" apply -f kubernetes-autoscaler/hpa.yaml -n trainticket
  fi

  if [ "${HEADLIGHT_SKIP_HPA_EVAL_DEPLOYMENT:-1}" = "1" ]; then
    echo "Skipping autoscaler evaluation deployment because HEADLIGHT_SKIP_HPA_EVAL_DEPLOYMENT=1"
    continue
  fi

  echo "Applying autoscaler evaluation deployment in $cluster..."
  kubectl --context="$cluster" apply -f deployments/kubernetes-autoscaler-eval-deployment.yaml -n trainticket
done

echo "All deployments completed successfully."
