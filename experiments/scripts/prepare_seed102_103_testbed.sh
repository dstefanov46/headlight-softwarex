#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
campaign_state_dir="${HEADLIGHT_CAMPAIGN_STATE_DIR:-$repo_root/results/headlight-v2-live/qualification-seed102-103-20261004}"
environment_file="${1:-$campaign_state_dir/campaign-env.sh}"

export HEADLIGHT_KUBECONFIG="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
export HEADLIGHT_INTERNAL_KUBECONFIG="${HEADLIGHT_INTERNAL_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
export HEADLIGHT_KARMADA_DATA="${HEADLIGHT_KARMADA_DATA:-$repo_root/.karmada}"
export HEADLIGHT_TRAINTICKET_PATH=advanced-ticket
export HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE=capacity-vm
export HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST="$campaign_state_dir/runtime/rendered-advanced-ticket.yaml"
export KUBECONFIG="$HEADLIGHT_KUBECONFIG"

mkdir -p "$campaign_state_dir/runtime"
cd "$repo_root"

echo "== Verifying retained Headlight source provenance =="
"$script_dir/verify_and_build_headlight_v2_image.sh" --verify-only

echo "== Rebuilding the Kind/Karmada testbed and redeploying advanced-ticket =="
"$repo_root/experiments/rebuild_local_testbed.sh"

echo "== Installing Prometheus monitoring =="
"$repo_root/experiments/install_prometheus_monitoring.sh"

echo "== Rebuilding and loading the pinned Headlight image =="
"$script_dir/verify_and_build_headlight_v2_image.sh"

echo "== Deploying the shared TrainTicket gateway =="
"$script_dir/deploy_trainticket_gateway.sh"

echo "== Recreating the campaign environment for this testbed =="
"$script_dir/write_seed102_103_campaign_env.sh" "$environment_file"

# shellcheck disable=SC1090
source "$environment_file"
echo "== Running the campaign preflight =="
"$script_dir/preflight_headlight_v2_vm.sh"

echo "TESTBED_PREPARATION_COMPLETE environment=$environment_file"
