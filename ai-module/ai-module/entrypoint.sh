#!/bin/sh
set -eu

bundle="${HEADLIGHT_MODEL_BUNDLE:-/models/bundle}"
profile="${HEADLIGHT_SERVICE_PROFILE:-/profiles/profile.json}"

python validate_model_bundle.py \
  --bundle "$bundle" \
  --service-profile "$profile" \
  --controller headlight

exec python live_inference.py \
  --bundle "$bundle" \
  --service-profile "$profile" \
  --action-log "${HEADLIGHT_ACTION_LOG:-/artifacts/actions.jsonl}"
