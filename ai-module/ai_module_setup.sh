#!/bin/sh
set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

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

expose_actions_handler() {
  kubectl --context=karmada-host create namespace actions-handler-system --dry-run=client -o yaml | kubectl --context=karmada-host apply -f - >/dev/null
  cat <<'EOF' | kubectl --context=karmada-host apply -f - >/dev/null
apiVersion: v1
kind: Service
metadata:
  name: actions-handler-nodeport
  namespace: actions-handler-system
spec:
  type: NodePort
  selector:
    app.kubernetes.io/name: actions-handler
  ports:
  - name: http
    port: 8080
    targetPort: 8080
EOF
  kubectl --context=karmada-host -n actions-handler-system get service actions-handler-nodeport -o jsonpath='{.spec.ports[0].nodePort}'
}

expose_prometheus() {
  prometheus_cluster="$1"
  prometheus_namespace="${HEADLIGHT_PROMETHEUS_NAMESPACE:-monitoring}"
  prometheus_service="${HEADLIGHT_PROMETHEUS_SERVICE:-prometheus-kube-prometheus-prometheus}"
  kubectl --context="$prometheus_cluster" -n "$prometheus_namespace" expose service "$prometheus_service" \
    --name=headlight-prometheus-nodeport \
    --type=NodePort \
    --port=9090 \
    --target-port=9090 \
    --dry-run=client \
    -o yaml | kubectl --context="$prometheus_cluster" apply -f - >/dev/null
  kubectl --context="$prometheus_cluster" -n "$prometheus_namespace" get service headlight-prometheus-nodeport -o jsonpath='{.spec.ports[0].nodePort}'
}

echo "Creating namespace and enabling Istio injection..."
kubectl create namespace trainticket --context=karmada-apiserver || true
kubectl label namespace trainticket --context=karmada-apiserver istio-injection=enabled --overwrite

if [ "${HEADLIGHT_SKIP_SMO:-0}" != "1" ]; then
echo "SMO port forwarding..."
kubectl --context=cluster1 -n smo port-forward service/smo-service 8000:8000 &

echo "Deploying DB services..."
curl -X POST "localhost:8000/project/trainticket/graphs" \
     -H "Content-Type: application/json" \
     --data '{"artifact": "ghcr.io/dstefanov46/trainticket/trainticket-db-graph:2.0.1"}'

echo "Deploying application services..."
curl -X POST "localhost:8000/project/trainticket/graphs" \
     -H "Content-Type: application/json" \
     --data '{"artifact": "ghcr.io/dstefanov46/trainticket/trainticket-app-graph:7.0.1"}'
else
echo "Skipping SMO graph deployment because HEADLIGHT_SKIP_SMO=1"
fi

echo "Deploying istio gateway..."
for cluster in cluster2 cluster3; do
  echo "Setting up Istio gateway in $cluster..."
  kubectl apply --context="$cluster" -f "$repo_root/ai-module/deployments/istio-gateway.yaml" -n trainticket
done

echo "Applying roles..."
for cluster in cluster2 cluster3; do
  echo "Applying roles in $cluster..."
  for role_file in "$repo_root"/ai-module/roles/*.yaml; do
    sed "s/namespace: nephele/namespace: trainticket/g" "$role_file" | kubectl apply --context="$cluster" -f -
  done
done

pod_kubeconfig=$(mktemp)
trap 'rm -f "$pod_kubeconfig"' EXIT
prepare_pod_kubeconfig "$pod_kubeconfig"

actions_handler_url="${HEADLIGHT_ACTION_HANDLER_URL:-}"
if [ -z "$actions_handler_url" ]; then
  karmada_host_ip=$(control_plane_ip karmada-host-control-plane)
  if [ -n "$karmada_host_ip" ]; then
    actions_handler_node_port=$(expose_actions_handler)
    actions_handler_url="http://$karmada_host_ip:$actions_handler_node_port/placements"
  fi
fi
prometheus_url="${HEADLIGHT_PROMETHEUS_URL:-}"
if [ -z "$prometheus_url" ]; then
  for prometheus_cluster in cluster2 cluster3; do
    prometheus_host_ip=$(control_plane_ip "$prometheus_cluster-control-plane")
    if [ -n "$prometheus_host_ip" ]; then
      prometheus_node_port=$(expose_prometheus "$prometheus_cluster")
      endpoint="http://$prometheus_host_ip:$prometheus_node_port"
      if [ -n "$prometheus_url" ]; then
        prometheus_url="$prometheus_url,$endpoint"
      else
        prometheus_url="$endpoint"
      fi
    fi
  done
fi

if [ "${HEADLIGHT_SKIP_LEGACY_LOCUST:-1}" != "1" ]; then
echo "Deploying legacy Locust master and workers..."
kubectl --context=cluster2 create secret generic pod-kubeconfig-locust --from-file=config="$pod_kubeconfig" -n trainticket --dry-run=client -o yaml | kubectl --context=cluster2 apply -f -
kubectl --context=cluster3 create secret generic pod-kubeconfig-locust --from-file=config="$pod_kubeconfig" -n trainticket --dry-run=client -o yaml | kubectl --context=cluster3 apply -f -
helm --kube-context cluster2 upgrade --install locust "$repo_root/ai-module/locust-helm-chart" --namespace trainticket
else
echo "Skipping legacy Locust Helm deployment because HEADLIGHT_SKIP_LEGACY_LOCUST=1"
fi

echo "Deploying AI module..."
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"
model_bundle="${HEADLIGHT_MODEL_BUNDLE_DIR:-$repo_root/results/calibration/headlight-bundle}"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
  python_bin=python3
fi
"$python_bin" "$repo_root/ai-module/ai-module/validate_model_bundle.py" \
  --bundle "$model_bundle" \
  --service-profile "$service_profile" \
  --controller headlight
kubectl create namespace trainticket --context=cluster1 || true
kubectl label namespace trainticket --context=cluster1 istio-injection=enabled --overwrite
for role_file in "$repo_root"/ai-module/roles/*.yaml; do
  sed "s/namespace: nephele/namespace: trainticket/g" "$role_file" | kubectl apply --context=cluster1 -f -
done
kubectl --context=cluster1 create secret generic pod-kubeconfig-trainticket --from-file=config="$pod_kubeconfig" -n trainticket --dry-run=client -o yaml | kubectl --context=cluster1 apply -f -
kubectl --context=cluster1 -n trainticket create secret generic headlight-model-bundle \
  --from-file=checkpoint.pt="$model_bundle/checkpoint.pt" \
  --from-file=scaler.json="$model_bundle/scaler.json" \
  --from-file=feature_schema.json="$model_bundle/feature_schema.json" \
  --from-file=service_profile.json="$model_bundle/service_profile.json" \
  --from-file=adjacency.json="$model_bundle/adjacency.json" \
  --from-file=capacity.json="$model_bundle/capacity.json" \
  --from-file=model_config.json="$model_bundle/model_config.json" \
  --from-file=learned_edges.json="$model_bundle/learned_edges.json" \
  --from-file=metadata.json="$model_bundle/metadata.json" \
  --dry-run=client -o yaml | kubectl --context=cluster1 apply -f -
kubectl --context=cluster1 -n trainticket create configmap headlight-service-profile \
  --from-file=profile.json="$service_profile" \
  --dry-run=client -o yaml | kubectl --context=cluster1 apply -f -
kubectl apply -f "$repo_root/ai-module/deployments/ai-module-deployment.yaml" -n trainticket --context=cluster1
if [ -n "${HEADLIGHT_AI_IMAGE:-}" ]; then
  kubectl --context=cluster1 -n trainticket set image deployment/ai-module ai-module="$HEADLIGHT_AI_IMAGE"
fi
if [ -n "$actions_handler_url" ]; then
  kubectl --context=cluster1 -n trainticket set env deployment/ai-module HEADLIGHT_ACTION_HANDLER_URL="$actions_handler_url"
fi
if [ -n "$prometheus_url" ]; then
  kubectl --context=cluster1 -n trainticket set env deployment/ai-module HEADLIGHT_PROMETHEUS_URL="$prometheus_url"
fi
kubectl --context=cluster1 -n trainticket set env deployment/ai-module \
  HEADLIGHT_SHADOW_MODE="${HEADLIGHT_SHADOW_MODE:-0}"
kubectl --context=cluster1 -n trainticket rollout restart deployment/ai-module
kubectl --context=cluster1 -n trainticket rollout status deployment/ai-module --timeout="${HEADLIGHT_CONTROLLER_WAIT_TIMEOUT:-600s}"

echo "Deployment script completed successfully."
