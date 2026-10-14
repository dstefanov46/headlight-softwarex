#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
context="${HEADLIGHT_CONTROLLER_CONTEXT:-cluster3}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
image="${HEADLIGHT_THRESHOLD_IMAGE:-local/headlight-ai-module:dev}"
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"
policy_config="${HEADLIGHT_POLICY_CONFIG:-$repo_root/ai-module/configs/policies/research-v3-forecast-policy.json}"
pod_kubeconfig="${HEADLIGHT_POD_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
rendered="${HEADLIGHT_THRESHOLD_RENDERED_MANIFEST:-$repo_root/results/threshold-autoscaler-deployment.yaml}"

if [[ ! -f "$pod_kubeconfig" ]]; then
  echo "Missing pod-reachable kubeconfig: $pod_kubeconfig" >&2
  exit 1
fi

control_plane_ip() {
  docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$1" 2>/dev/null || true
}

actions_handler_url="${HEADLIGHT_ACTION_HANDLER_URL:-}"
if [[ -z "$actions_handler_url" ]]; then
  host_ip="$(control_plane_ip karmada-host-control-plane)"
  if [[ -n "$host_ip" ]]; then
    kubectl --context=karmada-host create namespace actions-handler-system --dry-run=client -o yaml | kubectl --context=karmada-host apply -f - >/dev/null
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
    port="$(kubectl --context=karmada-host -n actions-handler-system get service actions-handler-nodeport -o jsonpath='{.spec.ports[0].nodePort}')"
    actions_handler_url="http://${host_ip}:${port}/placements"
  fi
fi

prometheus_url="${HEADLIGHT_PROMETHEUS_URL:-}"
if [[ -z "$prometheus_url" ]]; then
  for member in cluster2 cluster3; do
    member_ip="$(control_plane_ip "$member-control-plane")"
    [[ -z "$member_ip" ]] && continue
    kubectl --context="$member" -n monitoring expose service prometheus-kube-prometheus-prometheus \
      --name=headlight-prometheus-nodeport --type=NodePort --port=9090 --target-port=9090 \
      --dry-run=client -o yaml | kubectl --context="$member" apply -f - >/dev/null
    port="$(kubectl --context="$member" -n monitoring get service headlight-prometheus-nodeport -o jsonpath='{.spec.ports[0].nodePort}')"
    endpoint="http://${member_ip}:${port}"
    prometheus_url="${prometheus_url:+$prometheus_url,}$endpoint"
  done
fi

kubectl --context="$context" get namespace "$namespace" >/dev/null 2>&1 || kubectl --context="$context" create namespace "$namespace"
kubectl --context="$context" -n "$namespace" create secret generic pod-kubeconfig-trainticket \
  --from-file=config="$pod_kubeconfig" --dry-run=client -o yaml | kubectl --context="$context" apply -f - >/dev/null
kubectl --context="$context" -n "$namespace" create configmap headlight-service-profile \
  --from-file=profile.json="$service_profile" --dry-run=client -o yaml | kubectl --context="$context" apply -f - >/dev/null
kubectl --context="$context" -n "$namespace" create configmap headlight-policy-config \
  --from-file=policy.json="$policy_config" --dry-run=client -o yaml | kubectl --context="$context" apply -f - >/dev/null

mkdir -p "$(dirname "$rendered")"
python3 - "$repo_root/ai-module/deployments/threshold-autoscaler-deployment.yaml" "$rendered" "$image" "$namespace" <<'PY'
from pathlib import Path
import sys
import yaml
source, output, image, namespace = sys.argv[1:]
deployment = yaml.safe_load(Path(source).read_text())
deployment["metadata"]["namespace"] = namespace
container = deployment["spec"]["template"]["spec"]["containers"][0]
container["image"] = image
for variable in container["env"]:
    if variable["name"] == "HEADLIGHT_NAMESPACE":
        variable["value"] = namespace
Path(output).write_text(yaml.safe_dump(deployment, sort_keys=False))
PY
kubectl --context="$context" -n "$namespace" apply -f "$rendered"
kubectl --context="$context" -n "$namespace" set env deployment/threshold-autoscaler \
  HEADLIGHT_ACTION_HANDLER_URL="${actions_handler_url:-http://actions-handler.actions-handler-system.svc.cluster.local:8080/placements}" \
  HEADLIGHT_PROMETHEUS_URL="${prometheus_url:-http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090}" \
  HEADLIGHT_SHADOW_MODE="${HEADLIGHT_SHADOW_MODE:-0}"
kubectl --context="$context" -n "$namespace" rollout status deployment/threshold-autoscaler --timeout="${HEADLIGHT_CONTROLLER_WAIT_TIMEOUT:-600s}"

echo "Threshold controller deployed in $context/$namespace using the shared Actions Handler/CRP path"
