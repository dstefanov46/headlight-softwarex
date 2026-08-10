#!/usr/bin/env bash
set -euo pipefail

# Static pilots intentionally install no autoscaler. They do, however, need to
# apply the fixed provisioning plan described by the static controller profile
# instead of leaving the experiment runner's 1/1 bootstrap placements in place.
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
contexts="${HEADLIGHT_STATIC_CONTEXTS:-cluster2 cluster3}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"
cluster2_replicas="${HEADLIGHT_STATIC_CLUSTER2_REPLICAS:-3}"
cluster3_replicas="${HEADLIGHT_STATIC_CLUSTER3_REPLICAS:-5}"

for context in $contexts; do
  kubectl --context="$context" -n "$namespace" delete hpa --all --ignore-not-found
  if kubectl --context="$context" api-resources --api-group=keda.sh 2>/dev/null | grep -q '^scaledobjects[[:space:]]'; then
    kubectl --context="$context" -n "$namespace" delete scaledobject --all --ignore-not-found
  fi
done

scalable_workloads=$("$python_bin" - "$service_profile" <<'PY'
import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in profile.get("nodes", []):
    if node.get("scalable", True):
        print(node.get("workload") or node["name"])
PY
)

for workload in $scalable_workloads; do
  kubectl --context=karmada-apiserver -n "$namespace" patch \
    controlledreplicapropagation.policy.nephele-project.eu "$workload" \
    --type=merge \
    -p "{\"spec\":{\"placements\":[{\"clusterName\":\"cluster2\",\"replicas\":$cluster2_replicas},{\"clusterName\":\"cluster3\",\"replicas\":$cluster3_replicas}]}}"
done

for context in $contexts; do
  for workload in $scalable_workloads; do
    kubectl --context="$context" -n "$namespace" rollout status deployment/"$workload" --timeout="${HEADLIGHT_STATIC_ROLLOUT_TIMEOUT:-900s}"
  done
done

echo "Static baseline active with fixed placements cluster2=$cluster2_replicas cluster3=$cluster3_replicas for: $scalable_workloads"
