#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
mode="${1:-build}"

if [[ "$mode" != "build" && "$mode" != "--verify-only" ]]; then
  echo "Usage: $0 [--verify-only]" >&2
  exit 2
fi

helper="$script_dir/frozen_headlight_v2_source.py"
context_args=()
if [[ "$mode" == "build" ]]; then
  temporary="$(mktemp -d)"
  trap 'rm -rf -- "$temporary"' EXIT
  mkdir "$temporary/context"
  context_args=(--context "$temporary/context")
fi
# Verify the capsule and frozen model without changing the current checkout.
metadata="$(python3 "$helper" "${context_args[@]}")"
mapfile -t fields <<<"$metadata"
image="${fields[0]}"
code_revision="${fields[1]}"
image_revision="${fields[2]}"
file_count="${fields[3]}"

echo "HEADLIGHT_V2_SOURCE_OK files=$file_count image=$image code_revision=$code_revision"
if [[ "$mode" == "--verify-only" ]]; then
  exit 0
fi

for command_name in docker kind; do
  command -v "$command_name" >/dev/null || { echo "$command_name is required" >&2; exit 1; }
done

docker build \
  --file "$temporary/context/ai-module/ai-module/Dockerfile" \
  --tag "$image" \
  --build-arg "HEADLIGHT_CODE_REVISION=$code_revision" \
  --build-arg "HEADLIGHT_IMAGE_REVISION=$image_revision" \
  "$temporary/context/ai-module"
docker image inspect "$image" > "$temporary/image-inspect.json"
artifact_dir="${HEADLIGHT_V2_IMAGE_ARTIFACT_DIR:-$repo_root/results/headlight-v2-image-builds/$(date -u +%Y%m%dT%H%M%S)-$$}"
python3 "$helper" --record-image "$temporary/image-inspect.json" --output "$artifact_dir" >/dev/null
kind load docker-image "$image" --name cluster3
echo "HEADLIGHT_V2_IMAGE_READY image=$image cluster=cluster3 provenance=$artifact_dir/image-build-provenance.json"
