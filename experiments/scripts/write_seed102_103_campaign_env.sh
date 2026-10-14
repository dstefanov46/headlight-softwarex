#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
campaign_state_dir="${HEADLIGHT_CAMPAIGN_STATE_DIR:-$repo_root/results/headlight-v2-live/qualification-seed102-103-20261004}"
output="${1:-$campaign_state_dir/campaign-env.sh}"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
internal_kubeconfig="${HEADLIGHT_INTERNAL_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
karmada_data="${HEADLIGHT_KARMADA_DATA:-$repo_root/.karmada}"
reference_environment="$repo_root/experiments/assets/headlight-v2/execution-environment.json"
prometheus_namespace="${HEADLIGHT_PROMETHEUS_NAMESPACE:-monitoring}"
prometheus_service="${HEADLIGHT_PROMETHEUS_SERVICE:-prometheus-kube-prometheus-prometheus}"

for command_name in kubectl curl python3; do
  command -v "$command_name" >/dev/null || { echo "$command_name is required" >&2; exit 1; }
done
for required in "$kubeconfig" "$internal_kubeconfig" "$reference_environment"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

export KUBECONFIG="$kubeconfig"
prometheus_urls=()
for cluster in cluster2 cluster3; do
  kubectl --context="$cluster" -n "$prometheus_namespace" patch service "$prometheus_service" \
    --type=merge -p '{"spec":{"type":"NodePort"}}' >/dev/null

  node_ip="$(kubectl --context="$cluster" get node \
    -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
  node_port="$(kubectl --context="$cluster" -n "$prometheus_namespace" get service "$prometheus_service" \
    -o jsonpath='{.spec.ports[?(@.port==9090)].nodePort}')"
  if [[ -z "$node_ip" || -z "$node_port" ]]; then
    echo "Could not discover the Prometheus NodePort for $cluster" >&2
    exit 1
  fi

  url="http://${node_ip}:${node_port}"
  curl --fail --silent --show-error "$url/-/ready" >/dev/null
  prometheus_urls+=("$url")
  echo "PROMETHEUS_DISCOVERED cluster=$cluster url=$url"
done
joined_prometheus_urls="$(IFS=,; echo "${prometheus_urls[*]}")"

mkdir -p "$(dirname -- "$output")" "$campaign_state_dir/runtime"
python3 - \
  "$reference_environment" \
  "$output" \
  "$repo_root" \
  "$kubeconfig" \
  "$internal_kubeconfig" \
  "$karmada_data" \
  "$joined_prometheus_urls" \
  "$campaign_state_dir" <<'PY'
import json
import os
from pathlib import Path
import shlex
import sys

(
    reference_path,
    output_path,
    repo_root,
    kubeconfig,
    internal_kubeconfig,
    karmada_data,
    prometheus_urls,
    campaign_state_dir,
) = sys.argv[1:]

environment = json.loads(Path(reference_path).read_text(encoding="utf-8"))
environment.update(
    {
        "KUBECONFIG": kubeconfig,
        "HEADLIGHT_KUBECONFIG": kubeconfig,
        "HEADLIGHT_INTERNAL_KUBECONFIG": internal_kubeconfig,
        "HEADLIGHT_POD_KUBECONFIG": internal_kubeconfig,
        "HEADLIGHT_KARMADA_DATA": karmada_data,
        "HEADLIGHT_PREFLIGHT_PROMETHEUS_URL": prometheus_urls,
        "HEADLIGHT_CONTROLLER_PROMETHEUS_URL": prometheus_urls,
        "HEADLIGHT_GATEWAY_PROTOCOL_PATH": str(
            Path(repo_root) / "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
        ),
        "HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST": str(
            Path(campaign_state_dir) / "runtime/rendered-advanced-ticket.yaml"
        ),
        "HEADLIGHT_PREFLIGHT_ARTIFACT_DIR": str(Path(campaign_state_dir) / "preflight"),
        "HEADLIGHT_V2_SHADOW_EVIDENCE": str(
            Path(os.environ.get("HEADLIGHT_V2_SHADOW_EVIDENCE", str(
                Path(repo_root)
                / "results/headlight-v2-live/emergency-disabled-canary-seed-401-20261004/shadow-smoke.ok"
            )))
        ),
        "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
    }
)

required_values = {
    "HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY": "1",
    "HEADLIGHT_EVALUATION_CAPACITY": "4",
    "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
    "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
    "HEADLIGHT_V2_PROMETHEUS_CLUSTERS": "cluster2,cluster3",
}
for name, expected in required_values.items():
    actual = environment.get(name)
    if actual != expected:
        raise SystemExit(f"retained {name} must be {expected!r}, got {actual!r}")

output = Path(output_path)
temporary = output.with_name(f".{output.name}.tmp")
lines = ["#!/usr/bin/env bash", "# Generated for the current rebuilt testbed; do not reuse after another rebuild."]
for name in sorted(environment):
    lines.append(f"export {name}={shlex.quote(str(environment[name]))}")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
temporary.replace(output)
PY

echo "CAMPAIGN_ENV_READY file=$output prometheus=$joined_prometheus_urls"
