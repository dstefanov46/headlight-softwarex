#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
contexts="${HEADLIGHT_PROMETHEUS_CONTEXTS:-cluster2 cluster3}"
namespace="${HEADLIGHT_PROMETHEUS_NAMESPACE:-monitoring}"
release="${HEADLIGHT_PROMETHEUS_RELEASE:-prometheus}"
chart="${HEADLIGHT_PROMETHEUS_CHART:-prometheus-community/kube-prometheus-stack}"
chart_version="${HEADLIGHT_PROMETHEUS_CHART_VERSION:-61.7.2}"
trainticket_namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
retention="${HEADLIGHT_PROMETHEUS_RETENTION:-2h}"

export KUBECONFIG="$kubeconfig"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo update prometheus-community >/dev/null

for context in $contexts; do
  helm upgrade --install "$release" "$chart" \
    --version "$chart_version" \
    --kube-context "$context" \
    --namespace "$namespace" \
    --create-namespace \
    --set grafana.enabled=false \
    --set alertmanager.enabled=false \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.retention="$retention" \
    --set prometheus.prometheusSpec.resources.requests.cpu=100m \
    --set prometheus.prometheusSpec.resources.requests.memory=1Gi \
    --set prometheus.prometheusSpec.resources.limits.cpu=500m \
    --set prometheus.prometheusSpec.resources.limits.memory=4Gi \
    --set prometheusOperator.resources.requests.cpu=50m \
    --set prometheusOperator.resources.requests.memory=128Mi \
    --set prometheusOperator.resources.limits.cpu=200m \
    --set prometheusOperator.resources.limits.memory=256Mi \
    --set kube-state-metrics.resources.requests.cpu=25m \
    --set kube-state-metrics.resources.requests.memory=64Mi \
    --set kube-state-metrics.resources.limits.cpu=100m \
    --set kube-state-metrics.resources.limits.memory=128Mi

  kubectl --context="$context" -n "$namespace" wait \
    "prometheus.monitoring.coreos.com/${release}-kube-prometheus-prometheus" \
    --for=condition=Available \
    --timeout=600s
  kubectl --context="$context" -n "$namespace" rollout status \
    "statefulset/prometheus-${release}-kube-prometheus-prometheus" \
    --timeout=600s
  kubectl --context="$context" -n "$namespace" get service prometheus-kube-prometheus-prometheus >/dev/null

  kubectl --context="$context" apply -f - <<YAML
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: trainticket-istio-sidecars
  namespace: $namespace
  labels:
    release: $release
spec:
  namespaceSelector:
    matchNames:
    - $trainticket_namespace
  selector:
    matchExpressions:
    - key: app
      operator: In
      values:
      - ts-auth
      - ts-basic
      - ts-ticketinfo
      - ts-route
      - ts-travel-plan
      - ts-travel
      - ts-travel2
      - ts-route-plan
      - ts-station
      - ts-train
      - ts-price
  podMetricsEndpoints:
  - port: http-envoy-prom
    path: /stats/prometheus
    interval: 15s
YAML
done

echo "Prometheus service name is prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
