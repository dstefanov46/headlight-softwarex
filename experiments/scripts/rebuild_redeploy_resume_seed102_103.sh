#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
campaign_state_dir="$repo_root/results/headlight-v2-live/qualification-seed102-103-20261004"
environment_file="${HEADLIGHT_CAMPAIGN_ENV_FILE:-$campaign_state_dir/campaign-env.sh}"

cd "$repo_root"
"$script_dir/prepare_seed102_103_testbed.sh" "$environment_file"

echo "== Resuming the seed 102/103 campaign =="
echo "Existing valid cells will be skipped; incomplete cells get fresh attempt directories."
export HEADLIGHT_CAMPAIGN_ENV_FILE="$environment_file"
exec "$script_dir/run_seed102_103_all_workloads.sh" --resume
