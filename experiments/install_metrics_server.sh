#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
contexts="${HEADLIGHT_METRICS_SERVER_CONTEXTS:-cluster2 cluster3}"
version="${HEADLIGHT_METRICS_SERVER_VERSION:-v0.7.2}"
manifest_url="${HEADLIGHT_METRICS_SERVER_MANIFEST:-https://github.com/kubernetes-sigs/metrics-server/releases/download/${version}/components.yaml}"

export KUBECONFIG="$kubeconfig"

for context in $contexts; do
  echo "Installing metrics-server in $context..."
  kubectl --context="$context" apply -f "$manifest_url"
  kubectl --context="$context" -n kube-system patch deployment metrics-server \
    --type=json \
    -p='[
      {
        "op": "replace",
        "path": "/spec/strategy",
        "value": {"type": "Recreate"}
      },
      {
        "op": "replace",
        "path": "/spec/template/spec/containers/0/args",
        "value": [
          "--cert-dir=/tmp",
          "--secure-port=10250",
          "--kubelet-preferred-address-types=InternalIP,Hostname,ExternalIP",
          "--kubelet-use-node-status-port",
          "--kubelet-request-timeout=25s",
          "--metric-resolution=30s",
          "--kubelet-insecure-tls"
        ]
      },
      {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/timeoutSeconds", "value": 10},
      {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/failureThreshold", "value": 6},
      {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/timeoutSeconds", "value": 10}
    ]'
  kubectl --context="$context" -n kube-system rollout status deployment/metrics-server --timeout=180s
  kubectl --context="$context" wait apiservice/v1beta1.metrics.k8s.io --for=condition=Available --timeout=180s

  for attempt in $(seq 1 24); do
    if kubectl --context="$context" top nodes >/dev/null 2>&1; then
      echo "metrics.k8s.io is ready in $context"
      break
    fi
    if [[ "$attempt" -eq 24 ]]; then
      echo "metrics.k8s.io did not return node metrics in $context" >&2
      exit 1
    fi
    sleep 5
  done
done
