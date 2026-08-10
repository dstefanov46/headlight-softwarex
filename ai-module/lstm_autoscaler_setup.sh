#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
context="${HEADLIGHT_CONTROLLER_CONTEXT:-cluster1}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
image="${HEADLIGHT_LSTM_IMAGE:-local/headlight-lstm:dev}"
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"
model_bundle="${HEADLIGHT_LSTM_BUNDLE_DIR:-$repo_root/results/calibration/lstm-bundle}"
pod_kubeconfig="${HEADLIGHT_POD_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
rendered="${HEADLIGHT_LSTM_RENDERED_MANIFEST:-$repo_root/results/lstm-autoscaler-deployment.yaml}"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi

control_plane_ip() {
  local container_name="$1"
  docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container_name" 2>/dev/null || true
}

actions_handler_url="${HEADLIGHT_ACTION_HANDLER_URL:-}"
if [[ -z "$actions_handler_url" ]]; then
  actions_host_ip="$(control_plane_ip karmada-host-control-plane)"
  if [[ -n "$actions_host_ip" ]]; then
    kubectl --context=karmada-host create namespace actions-handler-system --dry-run=client -o yaml | \
      kubectl --context=karmada-host apply -f - >/dev/null
    kubectl --context=karmada-host apply -f - >/dev/null <<'YAML'
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
YAML
    actions_port="$(kubectl --context=karmada-host -n actions-handler-system get service actions-handler-nodeport -o jsonpath='{.spec.ports[0].nodePort}')"
    actions_handler_url="http://$actions_host_ip:$actions_port/placements"
  fi
fi

prometheus_url="${HEADLIGHT_PROMETHEUS_URL:-}"
if [[ -z "$prometheus_url" ]]; then
  for member in cluster2 cluster3; do
    member_ip="$(control_plane_ip "$member-control-plane")"
    if [[ -z "$member_ip" ]]; then
      continue
    fi
    kubectl --context="$member" -n monitoring expose service prometheus-kube-prometheus-prometheus \
      --name=headlight-prometheus-nodeport --type=NodePort --port=9090 --target-port=9090 \
      --dry-run=client -o yaml | kubectl --context="$member" apply -f - >/dev/null
    prometheus_port="$(kubectl --context="$member" -n monitoring get service headlight-prometheus-nodeport -o jsonpath='{.spec.ports[0].nodePort}')"
    endpoint="http://$member_ip:$prometheus_port"
    prometheus_url="${prometheus_url:+$prometheus_url,}$endpoint"
  done
fi

"$python_bin" "$repo_root/ai-module/ai-module/validate_model_bundle.py" \
  --bundle "$model_bundle" \
  --service-profile "$service_profile" \
  --controller lstm
if [[ ! -f "$pod_kubeconfig" ]]; then
  echo "Missing pod-reachable kubeconfig: $pod_kubeconfig" >&2
  exit 1
fi

kubectl --context="$context" get namespace "$namespace" >/dev/null 2>&1 || \
  kubectl --context="$context" create namespace "$namespace"
kubectl --context="$context" -n "$namespace" create secret generic pod-kubeconfig-trainticket \
  --from-file=config="$pod_kubeconfig" \
  --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" create secret generic lstm-model-bundle \
  --from-file=checkpoint.pt="$model_bundle/checkpoint.pt" \
  --from-file=scaler.json="$model_bundle/scaler.json" \
  --from-file=feature_schema.json="$model_bundle/feature_schema.json" \
  --from-file=service_profile.json="$model_bundle/service_profile.json" \
  --from-file=adjacency.json="$model_bundle/adjacency.json" \
  --from-file=capacity.json="$model_bundle/capacity.json" \
  --from-file=model_config.json="$model_bundle/model_config.json" \
  --from-file=learned_edges.json="$model_bundle/learned_edges.json" \
  --from-file=metadata.json="$model_bundle/metadata.json" \
  --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" create configmap headlight-service-profile \
  --from-file=profile.json="$service_profile" \
  --dry-run=client -o yaml | kubectl --context="$context" apply -f -

sed "s#\${HEADLIGHT_LSTM_IMAGE}#$image#g" \
  "$repo_root/ai-module/deployments/lstm-autoscaler-deployment.yaml" > "$rendered"
kubectl --context="$context" -n "$namespace" apply -f "$rendered"
kubectl --context="$context" -n "$namespace" set env deployment/lstm-autoscaler \
  HEADLIGHT_ACTION_HANDLER_URL="${actions_handler_url:-http://actions-handler.actions-handler-system.svc.cluster.local:8080/placements}" \
  HEADLIGHT_PROMETHEUS_URL="${prometheus_url:-http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090}" \
  HEADLIGHT_SHADOW_MODE="${HEADLIGHT_SHADOW_MODE:-0}"
kubectl --context="$context" -n "$namespace" rollout status deployment/lstm-autoscaler \
  --timeout="${HEADLIGHT_CONTROLLER_WAIT_TIMEOUT:-600s}"

echo "LSTM controller deployed in $context/$namespace with frozen bundle $model_bundle"
