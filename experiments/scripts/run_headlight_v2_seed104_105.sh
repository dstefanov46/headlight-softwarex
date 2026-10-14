#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign_dir="${1:?Usage: bash experiments/scripts/run_headlight_v2_seed104_105.sh CAMPAIGN_DIR [--dry-run|--resume]}"
shift
cd "$repo_root"
environment_file="${HEADLIGHT_CAMPAIGN_ENV_FILE:-$campaign_dir/campaign-env.sh}"
if [[ " ${*} " != *" --dry-run "* && " ${*} " != *" --help "* ]]; then
  if [[ ! -f "$environment_file" ]]; then
    echo "Missing campaign environment: $environment_file" >&2
    echo "Generate it for this testbed with write_seed102_103_campaign_env.sh (see experiments/README.md)." >&2
    exit 1
  fi
  source "$environment_file"
fi
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
exec "$python_bin" experiments/scripts/run_seed102_103_campaign.py \
  --headlight-v2-extension --campaign-dir "$campaign_dir" "$@"
