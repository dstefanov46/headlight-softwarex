#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${HEADLIGHT_V2_IMAGE:-headlight-ai:v2-live}

docker build --file "$repo_root/ai-module/ai-module/Dockerfile" --tag "$image" "$repo_root/ai-module"
docker image inspect "$image" --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}}'
