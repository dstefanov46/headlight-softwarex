#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
context=${HEADLIGHT_CONTROLLER_CONTEXT:-cluster3}
namespace=${HEADLIGHT_NAMESPACE:-trainticket}
bundle=${HEADLIGHT_V2_BUNDLE_DIR:-$repo_root/results/headlight-v2-live/bundle-full-seed-601}
policy=${HEADLIGHT_V2_POLICY_CONFIG:-$repo_root/experiments/configs/models/headlight-v2-closed-loop-policy.json}
profile=${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/advanced-ticket.json}
lock=${HEADLIGHT_V2_RUNTIME_LOCK:-$repo_root/experiments/configs/live-v2/runtime-lock.json}
image=${HEADLIGHT_V2_IMAGE:-headlight-ai:v2-live}
pod_kubeconfig=${HEADLIGHT_POD_KUBECONFIG:-${HEADLIGHT_HOST_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}}
if [[ ! -f "$pod_kubeconfig" ]]; then
  pod_kubeconfig=${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}
fi

bundle_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle"]["bundle_id"])' "$lock")
policy_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"]["sha256"])' "$lock")
actual_policy_sha=$(sha256sum "$policy" | cut -d' ' -f1)
[[ "$actual_policy_sha" == "$policy_sha" ]] || { echo "policy checksum mismatch" >&2; exit 1; }
actual_bundle_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_id"])' "$bundle/metadata.json")
[[ "$actual_bundle_id" == "$bundle_id" ]] || { echo "bundle ID mismatch" >&2; exit 1; }

actions_handler_url=${HEADLIGHT_ACTION_HANDLER_URL:-}
if [[ -z "$actions_handler_url" ]]; then
  karmada_host_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' karmada-host-control-plane 2>/dev/null || true)
  [[ -n "$karmada_host_ip" ]] || {
    echo "cannot determine karmada-host-control-plane IP; set HEADLIGHT_ACTION_HANDLER_URL explicitly" >&2
    exit 1
  }
  kubectl --context=karmada-host -n actions-handler-system rollout status statefulset/actions-handler --timeout=180s
  kubectl --context=karmada-host apply -f - <<'YAML'
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
    protocol: TCP
YAML
  actions_handler_node_port=$(kubectl --context=karmada-host -n actions-handler-system get service actions-handler-nodeport -o jsonpath='{.spec.ports[0].nodePort}')
  actions_handler_endpoints=$(kubectl --context=karmada-host -n actions-handler-system get endpoints actions-handler-nodeport -o jsonpath='{.subsets[*].addresses[*].ip}')
  [[ -n "$actions_handler_node_port" && -n "$actions_handler_endpoints" ]] || {
    echo "Actions Handler NodePort has no ready endpoint" >&2
    exit 1
  }
  actions_handler_url="http://${karmada_host_ip}:${actions_handler_node_port}/placements"
fi

kubectl --context="$context" create namespace "$namespace" --dry-run=client -o yaml | kubectl --context="$context" apply -f -
[[ -f "$pod_kubeconfig" ]] || { echo "missing pod kubeconfig: $pod_kubeconfig" >&2; exit 1; }
kubectl --context="$context" -n "$namespace" create secret generic pod-kubeconfig-trainticket \
  --from-file=config="$pod_kubeconfig" --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" create secret generic headlight-v2-model-bundle \
  --from-file="$bundle" --dry-run=client -o yaml | kubectl --context="$context" apply --server-side -f -
kubectl --context="$context" -n "$namespace" create configmap headlight-v2-policy-config \
  --from-file=policy.json="$policy" --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" create configmap headlight-service-profile \
  --from-file=profile.json="$profile" --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" create configmap headlight-v2-runtime-lock \
  --from-literal=bundle_id="$bundle_id" \
  --from-literal=policy_sha256="$policy_sha" \
  --dry-run=client -o yaml | kubectl --context="$context" apply -f -
kubectl --context="$context" -n "$namespace" apply -f "$repo_root/ai-module/deployments/ai-module-v2-deployment.yaml"
kubectl --context="$context" -n "$namespace" set image deployment/ai-module-v2 ai-module-v2="$image"
if [[ "${HEADLIGHT_V2_ENABLE_ACTUATION:-0}" == "1" ]]; then
  evidence=${HEADLIGHT_V2_SHADOW_EVIDENCE:-$repo_root/results/headlight-v2-live/shadow-smoke/shadow-smoke.ok}
  grep -q "bundle_id=$bundle_id" "$evidence" || {
    echo "refusing actuation: missing matching shadow-smoke evidence at $evidence" >&2
    exit 1
  }
  kubectl --context="$context" -n "$namespace" patch deployment/ai-module-v2 --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--enable-actuation"}]'
fi
if [[ -n "${HEADLIGHT_CONTROLLER_PROMETHEUS_URL:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_PROMETHEUS_URL="$HEADLIGHT_CONTROLLER_PROMETHEUS_URL"
fi
if [[ -n "${HEADLIGHT_V2_PROMETHEUS_CLUSTERS:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_V2_PROMETHEUS_CLUSTERS="$HEADLIGHT_V2_PROMETHEUS_CLUSTERS"
fi
if [[ -n "${HEADLIGHT_V2_METRIC_DIAGNOSTICS:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_V2_METRIC_DIAGNOSTICS="$HEADLIGHT_V2_METRIC_DIAGNOSTICS"
fi
if [[ -n "${HEADLIGHT_V2_CPU_RATE_WINDOW:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_V2_CPU_RATE_WINDOW="$HEADLIGHT_V2_CPU_RATE_WINDOW"
fi
if [[ -n "${HEADLIGHT_V2_POLICY_REVISION:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_V2_POLICY_REVISION="$HEADLIGHT_V2_POLICY_REVISION"
fi
if [[ -n "${HEADLIGHT_RUN_ID:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_RUN_ID="$HEADLIGHT_RUN_ID"
fi
if [[ -n "${HEADLIGHT_SEED:-}" ]]; then
  kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
    HEADLIGHT_SEED="$HEADLIGHT_SEED"
fi
kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 \
  HEADLIGHT_ACTION_HANDLER_URL="$actions_handler_url"
for setting in HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY HEADLIGHT_V2_DIAGNOSTIC_DIR HEADLIGHT_V2_DIAGNOSTIC_MAX_BYTES HEADLIGHT_CODE_REVISION HEADLIGHT_IMAGE_REVISION; do
  if [[ -n "${!setting:-}" ]]; then
    kubectl --context="$context" -n "$namespace" set env deployment/ai-module-v2 "$setting=${!setting}"
  fi
done
kubectl --context="$context" -n "$namespace" rollout status deployment/ai-module-v2 --timeout=600s

echo "HEADLIGHT_V2_DEPLOYED context=$context image=$image bundle_id=$bundle_id actuation=${HEADLIGHT_V2_ENABLE_ACTUATION:-0} action_handler_url=$actions_handler_url"
