#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
internal_kubeconfig="${HEADLIGHT_INTERNAL_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
proxy_name="${HEADLIGHT_KARMADA_PROXY_NAME:-headlight-karmada-api-proxy}"
proxy_image="${HEADLIGHT_KARMADA_PROXY_IMAGE:-python:3.11-slim}"
proxy_port="${HEADLIGHT_KARMADA_PROXY_PORT:-32443}"
members=(cluster1 cluster2 cluster3)

usage() {
  cat <<'EOF'
Usage: repair_local_testbed_endpoints.sh [--verify-only]

Migrate the local Karmada endpoint to https://localhost:32443 and member
registrations to certificate-valid Docker DNS names. The operation snapshots
the previous kubeconfigs and Cluster endpoints under results/ before editing.
EOF
}

verify_only=0
case "${1:-}" in
  "") ;;
  --verify-only) verify_only=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

export KUBECONFIG="$kubeconfig"

require_inotify() {
  local instances watches
  instances="$(cat /proc/sys/fs/inotify/max_user_instances)"
  watches="$(cat /proc/sys/fs/inotify/max_user_watches)"
  if [[ "$instances" -lt 8192 || "$watches" -lt 1048576 ]]; then
    echo "host inotify limits are not ready: instances=$instances watches=$watches" >&2
    echo "run: sudo $repo_root/experiments/install_host_inotify_limits.sh" >&2
    exit 1
  fi
}

context_cluster() {
  kubectl --kubeconfig "$1" config view \
    -o jsonpath="{.contexts[?(@.name=='$2')].context.cluster}"
}

set_context_server() {
  local config="$1" context="$2" server="$3" cluster
  cluster="$(context_cluster "$config" "$context")"
  if [[ -z "$cluster" ]]; then
    echo "missing context $context in $config" >&2
    exit 1
  fi
  kubectl --kubeconfig "$config" config set-cluster "$cluster" --server="$server" >/dev/null
}

ensure_proxy() {
  if docker inspect "$proxy_name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$proxy_name")" != "true" ]]; then
      docker start "$proxy_name" >/dev/null
    fi
    return
  fi

  docker run -d \
    --name "$proxy_name" \
    --label eu.nephele-project.headlight.role=karmada-api-proxy \
    --restart unless-stopped \
    --network kind \
    --publish "127.0.0.1:${proxy_port}:32443" \
    --memory 64m \
    --cpus 0.10 \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --mount "type=bind,source=$repo_root/experiments/scripts/karmada_api_proxy.py,target=/karmada_api_proxy.py,readonly" \
    "$proxy_image" \
    python /karmada_api_proxy.py \
      --listen-port 32443 \
      --target-host karmada-host-control-plane \
      --target-port 32443 >/dev/null
}

wait_for_karmada() {
  for _ in $(seq 1 60); do
    if kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver get --raw=/readyz >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Karmada API did not become ready through https://localhost:${proxy_port}" >&2
  return 1
}

verify_endpoints() {
  kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver get --raw=/readyz >/dev/null
  for member in "${members[@]}"; do
    endpoint="$(kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver \
      get "clusters.cluster.karmada.io/$member" -o jsonpath='{.spec.apiEndpoint}')"
    expected="https://${member}-control-plane:6443"
    if [[ "$endpoint" != "$expected" ]]; then
      echo "$member endpoint is $endpoint, expected $expected" >&2
      return 1
    fi
    kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver \
      wait "clusters.cluster.karmada.io/$member" --for=condition=Ready --timeout=300s
  done
}

require_inotify
ensure_proxy

if [[ "$verify_only" -eq 0 ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  state_dir="$repo_root/results/local-24gb-feasibility/endpoint-migration-$stamp"
  mkdir -p "$state_dir"
  cp "$kubeconfig" "$state_dir/headlight-testbed.config.before"
  if [[ -f "$internal_kubeconfig" ]]; then
    cp "$internal_kubeconfig" "$state_dir/headlight-members-internal.config.before"
  else
    cp "$kubeconfig" "$internal_kubeconfig"
  fi

  # Bootstrap through a temporary host-cluster tunnel because the old Karmada
  # context may still contain an obsolete Docker IP.
  pf_log="$state_dir/karmada-bootstrap-port-forward.log"
  kubectl --kubeconfig "$kubeconfig" --context karmada-host -n karmada-system \
    port-forward svc/karmada-apiserver 18443:5443 >"$pf_log" 2>&1 &
  pf_pid=$!
  cleanup() {
    kill "$pf_pid" >/dev/null 2>&1 || true
    wait "$pf_pid" 2>/dev/null || true
  }
  trap cleanup EXIT
  for _ in $(seq 1 30); do
    grep -q 'Forwarding from' "$pf_log" && break
    sleep 1
  done

  kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver \
    --server=https://localhost:18443 get clusters.cluster.karmada.io -o json \
    > "$state_dir/karmada-clusters.before.json"

  for member in "${members[@]}"; do
    kubectl --kubeconfig "$kubeconfig" --context karmada-apiserver \
      --server=https://localhost:18443 patch "clusters.cluster.karmada.io/$member" \
      --type merge \
      -p "{\"spec\":{\"apiEndpoint\":\"https://${member}-control-plane:6443\"}}"
    set_context_server "$internal_kubeconfig" "$member" "https://${member}-control-plane:6443"
  done

  set_context_server "$kubeconfig" karmada-apiserver "https://localhost:${proxy_port}"
  set_context_server "$internal_kubeconfig" karmada-apiserver "https://localhost:${proxy_port}"
  cleanup
  trap - EXIT
fi

wait_for_karmada
verify_endpoints
echo "stable Karmada endpoint: https://localhost:${proxy_port}"
echo "all Karmada members use certificate-valid Docker DNS endpoints and report Ready"
