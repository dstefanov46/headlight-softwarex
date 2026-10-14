#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="/tmp/headlight-ablation-env.sh"
CANARY_EVIDENCE="$REPO_ROOT/results/headlight-v2-live/emergency-disabled-canary-seed-401-20261004/shadow-smoke.ok"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 1
fi

cd "$REPO_ROOT"
source "$ENV_FILE"

export HEADLIGHT_V2_SHADOW_EVIDENCE="$CANARY_EVIDENCE"

if [[ "${HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY:-}" != "1" ]]; then
  echo "HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY must be 1" >&2
  exit 1
fi
if [[ "${HEADLIGHT_EVALUATION_CAPACITY:-}" != "4" ]]; then
  echo "HEADLIGHT_EVALUATION_CAPACITY must be 4" >&2
  exit 1
fi
if [[ "${HEADLIGHT_GATEWAY_PROTOCOL_PATH:-}" != "$REPO_ROOT/experiments/configs/gateway-protocols/research-v4-gateway.yaml" ]]; then
  echo "The research-v4 gateway protocol must be configured" >&2
  exit 1
fi
if [[ ! -s "$HEADLIGHT_V2_SHADOW_EVIDENCE" ]]; then
  echo "Missing matching Headlight shadow canary evidence: $HEADLIGHT_V2_SHADOW_EVIDENCE" >&2
  exit 1
fi

OUTPUTS=(
  "$REPO_ROOT/results/headlight-v2-live/qualification-diurnal-c4-seed-101-20261004"
  "$REPO_ROOT/results/socc26/hpa-diurnal-multi-cluster-headlight"
  "$REPO_ROOT/results/socc26/keda-diurnal-multi-cluster-headlight"
  "$REPO_ROOT/results/headlight-v2-live/qualification-flash-crowd-c4-seed-101-20261004"
  "$REPO_ROOT/results/socc26/hpa-flash-crowd-multi-cluster-headlight"
  "$REPO_ROOT/results/socc26/keda-flash-crowd-multi-cluster-headlight"
)

for output in "${OUTPUTS[@]}"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output path: $output" >&2
    echo "Move or archive that result before starting a fresh six-cell run." >&2
    exit 1
  fi
done

echo "Running Diurnal: Headlight, HPA, KEDA"
.venv/bin/python experiments/run_headlight_v2_live.py \
  --run experiments/configs/qualification/headlight-v2-emergency-disabled-diurnal-seed-101-20261004.yaml \
  --output-dir results/headlight-v2-live/qualification-diurnal-c4-seed-101-20261004

.venv/bin/python experiments/run_experiment.py \
  --run experiments/configs/runs/hpa-diurnal-multi-cluster-headlight.yaml \
  --seeds 101

.venv/bin/python experiments/run_experiment.py \
  --run experiments/configs/runs/keda-diurnal-multi-cluster-headlight.yaml \
  --seeds 101

echo "Checking the three Diurnal load traces"
sha256sum --check <<'EOF'
ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2  results/headlight-v2-live/qualification-diurnal-c4-seed-101-20261004/seed-101/raw/load_profile.csv
ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2  results/socc26/hpa-diurnal-multi-cluster-headlight/seed-101/raw/load_profile.csv
ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2  results/socc26/keda-diurnal-multi-cluster-headlight/seed-101/raw/load_profile.csv
EOF

echo "Running Flash-Crowd: Headlight, HPA, KEDA"
.venv/bin/python experiments/run_headlight_v2_live.py \
  --run experiments/configs/qualification/headlight-v2-emergency-disabled-flash-crowd-seed-101-20261004.yaml \
  --output-dir results/headlight-v2-live/qualification-flash-crowd-c4-seed-101-20261004

.venv/bin/python experiments/run_experiment.py \
  --run experiments/configs/runs/hpa-flash-crowd-multi-cluster-headlight.yaml \
  --seeds 101

.venv/bin/python experiments/run_experiment.py \
  --run experiments/configs/runs/keda-flash-crowd-multi-cluster-headlight.yaml \
  --seeds 101

echo "Checking the three Flash-Crowd load traces"
sha256sum --check <<'EOF'
015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116  results/headlight-v2-live/qualification-flash-crowd-c4-seed-101-20261004/seed-101/raw/load_profile.csv
015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116  results/socc26/hpa-flash-crowd-multi-cluster-headlight/seed-101/raw/load_profile.csv
015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116  results/socc26/keda-flash-crowd-multi-cluster-headlight/seed-101/raw/load_profile.csv
EOF

echo "All six seed-101 cells completed and both workload trace checks passed."
