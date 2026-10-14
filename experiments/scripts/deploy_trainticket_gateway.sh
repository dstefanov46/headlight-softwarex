#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
image="${HEADLIGHT_GATEWAY_IMAGE:-haproxy:3.0-alpine}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
if [[ "$namespace" != trainticket ]]; then
  echo "gateway protocol currently supports only the trainticket namespace" >&2
  exit 2
fi
template_dir="$repo_root/experiments/testbed/trainticket-gateway"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT

for command_name in docker kind kubectl jq python3 sha256sum; do
  command -v "$command_name" >/dev/null || { echo "$command_name is required" >&2; exit 1; }
done

for cluster in cluster2 cluster3; do
  kubectl --context="$cluster" -n "$namespace" apply -f "$template_dir/local-ingress.yaml"
  kubectl --context="$cluster" -n istio-system rollout status deployment/istio-ingressgateway --timeout=120s
  node_ip="$(kubectl --context="$cluster" get node -o json | jq -er '.items[0].status.addresses[] | select(.type == "InternalIP") | .address' | head -1)"
  service_json="$(kubectl --context="$cluster" -n istio-system get service istio-ingressgateway -o json)"
  http_port="$(jq -er '.spec.ports[] | select(.port == 80) | .nodePort' <<<"$service_json")"
  status_port="$(jq -er '.spec.ports[] | select(.port == 15021) | .nodePort' <<<"$service_json")"
  docker exec cluster1-control-plane curl -fsS --connect-timeout 3 --max-time 5 \
    "http://$node_ip:$status_port/healthz/ready" >/dev/null
  if [[ "$cluster" == cluster2 ]]; then
    cluster2_ip="$node_ip"
    cluster2_http_port="$http_port"
    cluster2_status_port="$status_port"
  else
    cluster3_ip="$node_ip"
    cluster3_http_port="$http_port"
    cluster3_status_port="$status_port"
  fi
  echo "$cluster ingress=$node_ip:$http_port health=$node_ip:$status_port"
done

python3 - "$template_dir/haproxy.cfg.in" "$temporary/haproxy.cfg" \
  "$cluster2_ip" "$cluster2_http_port" "$cluster2_status_port" \
  "$cluster3_ip" "$cluster3_http_port" "$cluster3_status_port" <<'PY'
from pathlib import Path
import sys

source, output, c2_ip, c2_http, c2_status, c3_ip, c3_http, c3_status = sys.argv[1:]
config = Path(source).read_text(encoding="utf-8")
replacements = {
    "__CLUSTER2_IP__": c2_ip,
    "__CLUSTER2_HTTP_PORT__": c2_http,
    "__CLUSTER2_STATUS_PORT__": c2_status,
    "__CLUSTER3_IP__": c3_ip,
    "__CLUSTER3_HTTP_PORT__": c3_http,
    "__CLUSTER3_STATUS_PORT__": c3_status,
}
for old, new in replacements.items():
    config = config.replace(old, new)
if "__CLUSTER" in config:
    raise SystemExit("unresolved HAProxy template placeholder")
Path(output).write_text(config, encoding="utf-8")
PY

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker pull "$image"
fi
docker run --rm -v "$temporary/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro" \
  "$image" haproxy -c -f /usr/local/etc/haproxy/haproxy.cfg
kind load docker-image "$image" --name cluster1

config_sha256="$(sha256sum "$temporary/haproxy.cfg" | awk '{print $1}')"
kubectl --context=cluster1 get namespace "$namespace" >/dev/null 2>&1 || \
  kubectl --context=cluster1 create namespace "$namespace"
kubectl --context=cluster1 -n "$namespace" create configmap trainticket-gateway-config \
  --from-file=haproxy.cfg="$temporary/haproxy.cfg" --dry-run=client -o yaml | \
  kubectl --context=cluster1 apply -f -
python3 - "$template_dir/deployment.yaml.in" "$temporary/deployment.yaml" \
  "$config_sha256" "$image" <<'PY'
from pathlib import Path
import sys

source, output, config_sha256, image = sys.argv[1:]
manifest = Path(source).read_text(encoding="utf-8")
manifest = manifest.replace("__CONFIG_SHA256__", config_sha256)
manifest = manifest.replace("__GATEWAY_IMAGE__", image)
Path(output).write_text(manifest, encoding="utf-8")
PY
kubectl --context=cluster1 apply -f "$temporary/deployment.yaml"
kubectl --context=cluster1 -n "$namespace" rollout status deployment/trainticket-gateway --timeout=180s
"$repo_root/experiments/scripts/gateway_weights.py" --once
echo "TRAINTICKET_GATEWAY_READY url=http://trainticket-gateway.$namespace.svc.cluster.local:8080 config_sha256=$config_sha256"
