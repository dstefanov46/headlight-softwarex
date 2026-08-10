#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
progress_file="${HEADLIGHT_MATRIX_PROGRESS:-$repo_root/results/softwarex/matrix-progress.tsv}"
seeds="${HEADLIGHT_MATRIX_SEEDS:-101,102,103,104,105}"
export HEADLIGHT_TRAINTICKET_PATH="${HEADLIGHT_TRAINTICKET_PATH:-advanced-ticket}"
export HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-capacity-vm}"
winner_file="${HEADLIGHT_FROZEN_WINNER_FILE:-$repo_root/results/headlight-tuning/winner.json}"

if [[ ! -f "$winner_file" ]]; then
  echo "Missing frozen Headlight tuning winner: $winner_file" >&2
  echo "Run experiments/run_headlight_tuning.py before the measured matrix." >&2
  exit 1
fi
mapfile -t winner_values < <(
  python3 - "$winner_file" "${HEADLIGHT_MODEL_BUNDLE_DIR:-}" <<'PY'
import json
import sys
from pathlib import Path

winner_file = Path(sys.argv[1]).resolve()
override = sys.argv[2]
payload = json.loads(winner_file.read_text(encoding="utf-8"))
if payload.get("frozen_before_paper_seeds") is not True:
    raise SystemExit(f"{winner_file} is not marked frozen before measured seeds")
winner = payload.get("winner") or {}
source_bundle = Path(str(winner.get("bundle", ""))).resolve()
metadata_path = source_bundle / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"winning bundle is missing metadata: {metadata_path}")
source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if source_metadata.get("frozen") is not True:
    raise SystemExit(f"winning bundle is not immutable: {source_bundle}")
selected_bundle = Path(override).resolve() if override else source_bundle
selected_metadata_path = selected_bundle / "metadata.json"
if not selected_metadata_path.is_file():
    raise SystemExit(f"selected bundle is missing metadata: {selected_metadata_path}")
selected_metadata = json.loads(selected_metadata_path.read_text(encoding="utf-8"))
if selected_metadata.get("bundle_id") != source_metadata.get("bundle_id"):
    raise SystemExit("HEADLIGHT_MODEL_BUNDLE_DIR does not identify the frozen tuning winner")
for value in (
    selected_bundle,
    winner.get("candidate", ""),
    winner.get("lookback", ""),
    winner.get("graph_blocks", ""),
    winner.get("hidden_width", ""),
    winner.get("learning_rate", ""),
    winner.get("residual_bound", ""),
):
    print(value)
PY
)
if [[ "${#winner_values[@]}" -ne 7 ]]; then
  echo "Frozen winner metadata is incomplete: $winner_file" >&2
  exit 1
fi
export HEADLIGHT_MODEL_BUNDLE_DIR="${winner_values[0]}"
export HEADLIGHT_CANDIDATE="${winner_values[1]}"
export HEADLIGHT_STEP_NUM_IN="${winner_values[2]}"
export HEADLIGHT_NUM_BLOCKS="${winner_values[3]}"
export HEADLIGHT_HIDDEN_WIDTH="${winner_values[4]}"
export HEADLIGHT_LEARNING_RATE="${winner_values[5]}"
export HEADLIGHT_RESIDUAL_BOUND="${winner_values[6]}"

default_runs=(
  experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml
  experiments/configs/runs/headlight-flash-crowd-multi-cluster-headlight.yaml
  experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml
  experiments/configs/runs/hpa-diurnal-multi-cluster-headlight.yaml
  experiments/configs/runs/hpa-flash-crowd-multi-cluster-headlight.yaml
  experiments/configs/runs/hpa-gradual-ramp-multi-cluster-headlight.yaml
  experiments/configs/runs/keda-diurnal-multi-cluster-headlight.yaml
  experiments/configs/runs/keda-flash-crowd-multi-cluster-headlight.yaml
  experiments/configs/runs/keda-gradual-ramp-multi-cluster-headlight.yaml
  experiments/configs/runs/lstm-diurnal-multi-cluster-headlight.yaml
  experiments/configs/runs/lstm-flash-crowd-multi-cluster-headlight.yaml
  experiments/configs/runs/lstm-gradual-ramp-multi-cluster-headlight.yaml
  experiments/configs/runs/static-diurnal-multi-cluster-static.yaml
  experiments/configs/runs/static-flash-crowd-multi-cluster-static.yaml
  experiments/configs/runs/static-gradual-ramp-multi-cluster-static.yaml
  experiments/configs/runs/threshold-diurnal-multi-cluster-headlight.yaml
  experiments/configs/runs/threshold-flash-crowd-multi-cluster-headlight.yaml
  experiments/configs/runs/threshold-gradual-ramp-multi-cluster-headlight.yaml
)

if [[ -n "${HEADLIGHT_MATRIX_RUNS:-}" ]]; then
  IFS=',' read -r -a runs <<< "$HEADLIGHT_MATRIX_RUNS"
else
  runs=("${default_runs[@]}")
fi
IFS=',' read -r -a seed_list <<< "$seeds"

mkdir -p "$(dirname "$progress_file")"
touch "$progress_file"

if [[ "${HEADLIGHT_SKIP_METRICS_SERVER_INSTALL:-0}" != "1" ]]; then
  "$repo_root/experiments/install_metrics_server.sh"
fi

completed() {
  local run="$1"
  local seed="$2"
  [[ "${HEADLIGHT_RESUME:-0}" = "1" ]] || return 1
  if validate_bundle "$run" "$seed" >/dev/null 2>&1; then
    if ! grep -Fqx "$run	$seed	complete" "$progress_file"; then
      printf '%s\t%s\tcomplete\n' "$run" "$seed" >> "$progress_file"
    fi
    return 0
  fi
  if grep -Fqx "$run	$seed	complete" "$progress_file"; then
    echo "Progress says complete but bundle is invalid; rerunning $run seed $seed" >&2
  fi
  return 1
}

validate_bundle() {
  local run="$1"
  local seed="$2"
  python3 - "$run" "$seed" <<'PY'
import sys
from pathlib import Path

from experiments.lib.config import REPO_ROOT, resolve_run
from experiments.run_experiment import validate_seed_bundle

manifest = resolve_run(Path(sys.argv[1]))
seed = sys.argv[2]
seed_dir = REPO_ROOT / manifest["spec"]["run"]["output_dir"] / f"seed-{seed}"
validate_seed_bundle(seed_dir, dry_run=False, manifest=manifest)
PY
}

for seed in "${seed_list[@]}"; do
  if [[ "${HEADLIGHT_RANDOMIZE_ORDER:-1}" = "1" ]]; then
    mapfile -t seed_runs < <(printf '%s\n' "${runs[@]}" | python3 -c 'import random,sys; values=[line.strip() for line in sys.stdin if line.strip()]; random.Random(int(sys.argv[1])).shuffle(values); print("\\n".join(values))' "$seed")
  else
    seed_runs=("${runs[@]}")
  fi
  for run in "${seed_runs[@]}"; do
    if completed "$run" "$seed"; then
      echo "Skipping completed $run seed $seed"
      continue
    fi
    echo "Running $run seed $seed"
    python3 "$repo_root/experiments/run_experiment.py" --run "$run" --seeds "$seed"
    validate_bundle "$run" "$seed"
    printf '%s\t%s\tcomplete\n' "$run" "$seed" >> "$progress_file"
  done
  if [[ "$seed" = "101" ]] && [ "${HEADLIGHT_SKIP_SEED101_PROMOTION_GATE:-0}" != "1" ]]; then
    gate_args=()
    for run in "${runs[@]}"; do
      gate_args+=(--run "$run")
    done
    python3 "$repo_root/experiments/check_seed_promotion.py" --seed 101 "${gate_args[@]}"
  fi
done

echo "Matrix complete. Progress: $progress_file"
