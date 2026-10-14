#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
context=${HEADLIGHT_CONTROLLER_CONTEXT:-cluster3}
namespace=${HEADLIGHT_NAMESPACE:-trainticket}
artifact_dir=${HEADLIGHT_SMOKE_ARTIFACT_DIR:-$repo_root/results/headlight-v2-live/shadow-smoke}
mkdir -p "$artifact_dir"

"$repo_root/experiments/scripts/preflight_headlight_v2_vm.sh"
"$repo_root/experiments/scripts/deploy_headlight_v2_shadow.sh"

deadline=$((SECONDS + 600))
while (( SECONDS < deadline )); do
  logs=$(kubectl --context="$context" -n "$namespace" logs deployment/ai-module-v2 --tail=-1 2>&1 || true)
  if grep -q 'HEADLIGHT_V2_DECISION_INTERVAL' <<<"$logs"; then
    break
  fi
  sleep 15
done
kubectl --context="$context" -n "$namespace" logs deployment/ai-module-v2 --tail=-1 > "$artifact_dir/controller.log"
grep -q 'HEADLIGHT_V2_RUNTIME_READY' "$artifact_dir/controller.log"
grep -q 'HEADLIGHT_V2_DECISION_INTERVAL' "$artifact_dir/controller.log"
if grep 'HEADLIGHT_V2_DECISION_INTERVAL' "$artifact_dir/controller.log" | grep -vq '"shadow_mode":true'; then
  echo "non-shadow decision found during shadow smoke" >&2
  exit 1
fi
if grep 'HEADLIGHT_V2_DECISION_INTERVAL' "$artifact_dir/controller.log" | grep -vq '"actuator_called":false'; then
  echo "actuator was called during shadow smoke" >&2
  exit 1
fi
pod=$(kubectl --context="$context" -n "$namespace" get pod -l app=ai-module-v2 -o jsonpath='{.items[0].metadata.name}')
kubectl --context="$context" -n "$namespace" cp "$pod:/artifacts/actions-v2.jsonl" "$artifact_dir/actions-v2.jsonl"
kubectl --context=karmada-host -n actions-handler-system logs statefulset/actions-handler --tail=-1 > "$artifact_dir/actions-handler.log"
kubectl --context=karmada-apiserver get controlledreplicapropagations -A -o yaml > "$artifact_dir/crp-snapshot.yaml"
bundle_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle"]["bundle_id"])' \
  "$repo_root/experiments/configs/live-v2/runtime-lock.json")
printf 'bundle_id=%s\nstatus=passed\n' "$bundle_id" > "$artifact_dir/shadow-smoke.ok"
sha256sum "$artifact_dir"/actions-v2.jsonl "$artifact_dir"/actions-handler.log \
  "$artifact_dir"/controller.log "$artifact_dir"/crp-snapshot.yaml "$artifact_dir"/shadow-smoke.ok \
  > "$artifact_dir/SHA256SUMS"
echo "HEADLIGHT_V2_SHADOW_SMOKE_OK artifacts=$artifact_dir"
