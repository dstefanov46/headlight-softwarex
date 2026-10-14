#!/usr/bin/env bash
set -euo pipefail

namespace=${HEADLIGHT_NAMESPACE:-trainticket}
artifact_dir=${HEADLIGHT_PREFLIGHT_ARTIFACT_DIR:-results/headlight-v2-live/preflight}
prometheus_url=${HEADLIGHT_PREFLIGHT_PROMETHEUS_URL:-}
mkdir -p "$artifact_dir"
report="$artifact_dir/preflight-$(date -u +%Y%m%dT%H%M%SZ).log"
exec 3>&1 4>&2
exec > >(tee "$report" >&3) 2>&1

check() {
  label=$1
  shift
  echo "CHECK $label"
  "$@"
}

check_prometheus_metrics() {
  endpoint=$1
  response=$(curl --get --fail --silent --show-error \
    --data-urlencode "query=$query" "$endpoint/api/v1/query")
  python3 -c 'import json,sys; p=json.loads(sys.argv[1]); r=p.get("data",{}).get("result",[]); assert p.get("status")=="success" and r and float(r[0]["value"][1])>0, "no TrainTicket deployment metrics"' "$response"
  echo "$response"
}

check karmada-host-ready kubectl --context=karmada-host get --raw=/readyz
check karmada-api-ready kubectl --context=karmada-apiserver get --raw=/readyz
for cluster in cluster2 cluster3; do
  check "$cluster-api-ready" kubectl --context="$cluster" get --raw=/readyz
  check "$cluster-member-ready" kubectl --context=karmada-apiserver get cluster "$cluster"
  check "$cluster-trainticket-deployments" kubectl --context="$cluster" -n "$namespace" wait deployment --all --for=condition=Available --timeout=300s
done

check crp-crd kubectl --context=karmada-apiserver get crd controlledreplicapropagations.policy.nephele-project.eu
check crp-controller kubectl --context=karmada-host -n crp-system rollout status deployment/crp-controller-manager --timeout=180s
check actions-handler kubectl --context=karmada-host -n actions-handler-system rollout status statefulset/actions-handler --timeout=180s
check actions-handler-service kubectl --context=karmada-host -n actions-handler-system get service/actions-handler

if [[ -z "$prometheus_url" ]]; then
  echo "HEADLIGHT_PREFLIGHT_PROMETHEUS_URL must point to a VM-reachable Prometheus endpoint" >&2
  exit 1
fi
query='count(kube_deployment_status_replicas{namespace="trainticket"})'
IFS=',' read -r -a prometheus_endpoints <<< "$prometheus_url"
for index in "${!prometheus_endpoints[@]}"; do
  endpoint=${prometheus_endpoints[$index]%/}
  check "prometheus-$index-ready" curl --fail --silent --show-error "$endpoint/-/ready"
  check "prometheus-$index-trainticket-series" check_prometheus_metrics "$endpoint"
done

echo "PREFLIGHT_OK report=$report"
exec 1>&3 2>&4
wait
test -s "$report"
sha256sum "$report" > "$report.sha256"
echo "PREFLIGHT_CAPTURED report=$report checksum=$(cut -d' ' -f1 "$report.sha256")"
