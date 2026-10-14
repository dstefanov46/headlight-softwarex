#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${HEADLIGHT_CAMPAIGN_ENV_FILE:-$REPO_ROOT/results/headlight-v2-live/qualification-seed102-103-20261004/campaign-env.sh}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing campaign environment file: $ENV_FILE" >&2
  echo "Recreate it for the current testbed, or set HEADLIGHT_CAMPAIGN_ENV_FILE." >&2
  exit 1
fi

cd "$REPO_ROOT"
source "$ENV_FILE"
exec .venv/bin/python experiments/scripts/run_seed102_103_campaign.py "$@"
