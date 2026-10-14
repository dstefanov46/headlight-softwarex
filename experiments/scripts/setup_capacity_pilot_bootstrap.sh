#!/usr/bin/env bash
set -euo pipefail

# Capacity discovery must precede the calibration-derived static plan. Keep
# every scalable workload at the service-profile minima during pilot runs.
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
contexts="${HEADLIGHT_STATIC_CONTEXTS:-cluster2 cluster3}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
trainticket_path="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"

placement_rows=$("$python_bin" - "$service_profile" <<'PY'
import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
limits = profile["cluster_limits"]
cluster2 = int(limits["cluster2"]["min_replicas"])
cluster3 = int(limits["cluster3"]["min_replicas"])
for node in profile.get("nodes", []):
    if node.get("scalable", True):
        print(f"{node.get('workload') or node['name']}\t{cluster2}\t{cluster3}")
PY
)

for context in $contexts; do
  kubectl --context="$context" -n "$namespace" delete hpa --all --ignore-not-found
  if kubectl --context="$context" api-resources --api-group=keda.sh 2>/dev/null | grep -q '^scaledobjects[[:space:]]'; then
    kubectl --context="$context" -n "$namespace" delete scaledobject --all --ignore-not-found
  fi
done

while IFS=$'\t' read -r workload cluster2_replicas cluster3_replicas; do
  [[ -z "$workload" ]] && continue
  kubectl --context=karmada-apiserver -n "$namespace" patch \
    controlledreplicapropagation.policy.nephele-project.eu "$workload" \
    --type=merge \
    -p "{\"spec\":{\"placements\":[{\"clusterName\":\"cluster2\",\"replicas\":$cluster2_replicas},{\"clusterName\":\"cluster3\",\"replicas\":$cluster3_replicas}]}}"
done <<< "$placement_rows"

for context in $contexts; do
  while IFS=$'\t' read -r workload _cluster2_replicas _cluster3_replicas; do
    [[ -z "$workload" ]] && continue
    kubectl --context="$context" -n "$namespace" rollout status deployment/"$workload" \
      --timeout="${HEADLIGHT_CAPACITY_PILOT_ROLLOUT_TIMEOUT:-600s}"
  done <<< "$placement_rows"
done

echo "Capacity pilot bootstrap active at service-profile minimum placements"
