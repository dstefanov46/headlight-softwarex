#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
context="${HEADLIGHT_KEDA_CONTEXT:-cluster3}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
manifest="${HEADLIGHT_KEDA_SCALEDOBJECTS:-$repo_root/results/keda-scaledobjects.yaml}"
keda_chart_version="${HEADLIGHT_KEDA_CHART_VERSION:-2.19.0}"
prometheus_address="${HEADLIGHT_KEDA_PROMETHEUS_ADDRESS:-http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090}"
min_replicas="${HEADLIGHT_KEDA_MIN_REPLICAS:-1}"
max_replicas="${HEADLIGHT_KEDA_MAX_REPLICAS:-8}"
cpu_target="${HEADLIGHT_KEDA_CPU_TARGET:-70}"
request_rate_threshold="${HEADLIGHT_KEDA_REQUEST_RATE_THRESHOLD:-10}"
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-full}"
resource_profile="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-capacity-vm}"

export KUBECONFIG="$kubeconfig"

helm repo add kedacore https://kedacore.github.io/charts --force-update >/dev/null
helm repo update kedacore >/dev/null
helm --kube-context "$context" upgrade --install keda kedacore/keda \
  --version "$keda_chart_version" \
  --namespace keda \
  --create-namespace
kubectl --context="$context" -n keda rollout status deployment/keda-operator --timeout="${HEADLIGHT_KEDA_WAIT_TIMEOUT:-300s}"
kubectl --context="$context" -n keda rollout status deployment/keda-operator-metrics-apiserver --timeout="${HEADLIGHT_KEDA_WAIT_TIMEOUT:-300s}"

python3 "$repo_root/experiments/scripts/render_keda_scaledobjects.py" \
  --namespace "$namespace" \
  --trainticket-path "$trainticket_path" \
  --resource-profile "$resource_profile" \
  --prometheus-address "$prometheus_address" \
  --min-replicas "$min_replicas" \
  --max-replicas "$max_replicas" \
  --cpu-target "$cpu_target" \
  --request-rate-threshold "$request_rate_threshold" \
  --output "$manifest"

kubectl --context="$context" apply -f "$manifest"
echo "Rendered KEDA ScaledObjects: $manifest"
