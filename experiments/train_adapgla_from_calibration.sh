#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
calibration_dir="${HEADLIGHT_CALIBRATION_RESULT_DIR:-$repo_root/results/calibration}"
metrics_output="${HEADLIGHT_CALIBRATION_TRAINING_METRICS:-$calibration_dir/training_metrics.json}"
dataset="${HEADLIGHT_CALIBRATION_DATASET:-$calibration_dir/train-ticket-calibration.csv}"
trainticket_path="${HEADLIGHT_CALIBRATION_TRAINTICKET_PATH:-advanced-ticket}"
service_profile="${HEADLIGHT_SERVICE_PROFILE_FILE:-$repo_root/ai-module/configs/service_profiles/${trainticket_path}.json}"
adjacency="${HEADLIGHT_CALIBRATION_ADJACENCY:-$calibration_dir/static-adjacency.json}"
controller="${HEADLIGHT_TRAIN_CONTROLLER:-headlight}"
bundle_output="${HEADLIGHT_MODEL_BUNDLE_DIR:-$calibration_dir/${controller}-bundle}"
python_bin="${HEADLIGHT_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

mkdir -p "$(dirname "$bundle_output")"
cd "$repo_root/ai-module/ai-module"
"$python_bin" train_model.py \
  --once \
  --controller "$controller" \
  --dataset "$dataset" \
  --service-profile "$service_profile" \
  --adjacency "$adjacency" \
  --bundle-output "$bundle_output" \
  --seed "${HEADLIGHT_TRAIN_SEED:-20260705}" \
  --lookback "${HEADLIGHT_LOOKBACK:-24}" \
  --horizon 12 \
  --graph-blocks "${HEADLIGHT_GRAPH_BLOCKS:-2}" \
  --hidden-width "${HEADLIGHT_HIDDEN_WIDTH:-32}" \
  --learning-rate "${HEADLIGHT_LEARNING_RATE:-0.0003}" \
  --residual-bound "${HEADLIGHT_RESIDUAL_BOUND:-0.3}" \
  --variant "${HEADLIGHT_MODEL_VARIANT:-full}" \
  --epochs "${HEADLIGHT_TRAIN_EPOCHS:-100}" \
  --overwrite \
  --metrics-output "$metrics_output"

echo "Frozen $controller bundle: $bundle_output"
echo "Training metrics: $metrics_output"
