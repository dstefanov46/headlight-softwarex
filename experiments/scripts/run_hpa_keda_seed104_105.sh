#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign_dir="${1:?Usage: bash experiments/scripts/run_hpa_keda_seed104_105.sh CAMPAIGN_DIR [--dry-run|--resume]}"
shift
cd "$repo_root"
if [[ -f "$campaign_dir/campaign-env.sh" ]]; then
  source "$campaign_dir/campaign-env.sh"
elif [[ " ${*} " != *" --dry-run "* ]]; then
  echo "Prepare this campaign with prepare_hpa_keda_campaign.py first." >&2
  exit 1
fi
exec "$repo_root/.venv/bin/python" experiments/scripts/run_seed102_103_campaign.py \
  --baseline-extension --campaign-dir "$campaign_dir" "$@"
