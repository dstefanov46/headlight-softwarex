#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kind_node_image="${KIND_NODE_IMAGE:-kindest/node:v1.32.0}"
kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
internal_kubeconfig="${HEADLIGHT_INTERNAL_KUBECONFIG:-$repo_root/.kube/headlight-members-internal.config}"
karmada_data="${HEADLIGHT_KARMADA_DATA:-$repo_root/.karmada}"
kind_config="${HEADLIGHT_KIND_CONFIG:-$repo_root/experiments/testbed/kind/single-node.yaml}"
clusters=(karmada-host cluster1 cluster2 cluster3)
members=(cluster1 cluster2 cluster3)

export KUBECONFIG="$kubeconfig"

need_cmd() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "$command_name is required" >&2
    exit 1
  }
}

preflight() {
  for cmd in docker kind kubectl karmadactl helm istioctl python3; do
    need_cmd "$cmd"
  done

  local instances watches
  instances="$(cat /proc/sys/fs/inotify/max_user_instances)"
  watches="$(cat /proc/sys/fs/inotify/max_user_watches)"
  if [[ "$instances" -lt 8192 || "$watches" -lt 1048576 ]]; then
    if [[ "$(id -u)" -eq 0 ]]; then
      sysctl -w fs.inotify.max_user_instances=8192
      sysctl -w fs.inotify.max_user_watches=1048576
    else
      echo "host inotify limits are too low:" >&2
      echo "  fs.inotify.max_user_instances=$instances, need >=8192" >&2
      echo "  fs.inotify.max_user_watches=$watches, need >=1048576" >&2
      echo "raise them first: sudo sysctl fs.inotify.max_user_instances=8192 fs.inotify.max_user_watches=1048576" >&2
      exit 1
    fi
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not reachable. Start Docker before rebuilding the Kind testbed." >&2
    exit 1
  fi
}

clean_state() {
  for cluster in "${clusters[@]}"; do
    if kind get clusters 2>/dev/null | grep -qx "$cluster"; then
      kind delete cluster --name "$cluster"
    fi
  done
  rm -f "$kubeconfig" "$internal_kubeconfig"
  rm -rf "$karmada_data"
  mkdir -p "$(dirname "$kubeconfig")" "$karmada_data"
}

create_kind_clusters() {
  for cluster in "${clusters[@]}"; do
    kind create cluster \
      --name "$cluster" \
      --image "$kind_node_image" \
      --config "$kind_config" \
      --kubeconfig "$kubeconfig" \
      --wait 180s
    kubectl --kubeconfig "$kubeconfig" config rename-context "kind-$cluster" "$cluster" >/dev/null 2>&1 || true
    kubectl --kubeconfig "$kubeconfig" --context "$cluster" wait node --all --for=condition=Ready --timeout=180s
    kubectl --kubeconfig "$kubeconfig" --context "$cluster" -n kube-system rollout status deployment/coredns --timeout=180s
  done
}

merge_karmada_context() {
  local merged
  merged="$(mktemp)"
  KUBECONFIG="$kubeconfig:$karmada_data/karmada-apiserver.config" kubectl config view --flatten > "$merged"
  mv "$merged" "$kubeconfig"
  chmod 600 "$kubeconfig"
}

install_karmada() {
  karmadactl init \
    --kubeconfig "$kubeconfig" \
    --context karmada-host \
    --karmada-data "$karmada_data" \
    --karmada-pki "$karmada_data/pki" \
    --wait-component-ready-timeout 600
  merge_karmada_context
  kubectl --context=karmada-host -n karmada-system wait pod --all --for=condition=Ready --timeout=600s
  kubectl --context=karmada-apiserver get --raw=/readyz >/dev/null
}

refresh_internal_member_kubeconfig() {
  cp "$kubeconfig" "$internal_kubeconfig"
  chmod 600 "$internal_kubeconfig"
}

join_members() {
  refresh_internal_member_kubeconfig
  for member in "${members[@]}"; do
    karmadactl join "$member" \
      --kubeconfig "$kubeconfig" \
      --karmada-context karmada-apiserver \
      --cluster-kubeconfig "$internal_kubeconfig" \
      --cluster-context "$member"
    # Karmadactl must use the host-published API endpoint while joining, but
    # Karmada controllers run inside the Docker network. Store the stable node
    # DNS name in the registration; it is present in the member API server's
    # certificate and survives Docker IP reassignment.
    kubectl --context=karmada-apiserver patch "clusters.cluster.karmada.io/$member" \
      --type merge \
      -p "{\"spec\":{\"apiEndpoint\":\"https://${member}-control-plane:6443\"}}"
    local cluster_entry
    cluster_entry="$(kubectl --kubeconfig "$internal_kubeconfig" config view \
      -o jsonpath="{.contexts[?(@.name==\"$member\")].context.cluster}")"
    kubectl --kubeconfig "$internal_kubeconfig" config set-cluster "$cluster_entry" \
      --server="https://${member}-control-plane:6443" >/dev/null
  done

  for member in "${members[@]}"; do
    kubectl --context=karmada-apiserver wait "clusters.cluster.karmada.io/$member" \
      --for=condition=Ready \
      --timeout=300s
  done
}

write_sa_kubeconfig() {
  local namespace="$1"
  local service_account="$2"
  local output="$3"
  local server ca token
  # Host-cluster controllers should use the stable in-cluster Service DNS. A
  # Docker node IP can change after a container restart and the merged
  # kubeconfig may expose the host-cluster CA rather than Karmada's own CA.
  server="${HEADLIGHT_POD_KARMADA_SERVER:-https://karmada-apiserver.karmada-system.svc.cluster.local:5443}"
  ca="$(base64 -w0 "$karmada_data/pki/ca.crt")"
  token="$(kubectl --context=karmada-apiserver -n "$namespace" create token "$service_account" --duration=31536000s)"
  cat > "$output" <<YAML
apiVersion: v1
kind: Config
clusters:
- name: karmada-apiserver
  cluster:
    certificate-authority-data: $ca
    server: $server
contexts:
- name: ${service_account}@karmada-apiserver
  context:
    cluster: karmada-apiserver
    namespace: $namespace
    user: $service_account
current-context: ${service_account}@karmada-apiserver
users:
- name: $service_account
  user:
    token: $token
YAML
  chmod 600 "$output"
}

install_headlight_control() {
  kubectl --context=karmada-apiserver apply -f "$repo_root/controlled-replica-propagation/dist/karmada-install.yaml"
  kubectl --context=karmada-apiserver apply -f "$repo_root/actions-handler/dist/karmada-install.yaml"

  kubectl --context=karmada-apiserver -n crp-system get serviceaccount/crp-controller-manager >/dev/null
  kubectl --context=karmada-apiserver -n actions-handler-system get serviceaccount/actions-handler-manager >/dev/null

  local crp_config ah_config
  crp_config="$(mktemp)"
  ah_config="$(mktemp)"
  write_sa_kubeconfig crp-system crp-controller-manager "$crp_config"
  write_sa_kubeconfig actions-handler-system actions-handler-manager "$ah_config"

  kubectl --context=karmada-host apply -f "$repo_root/controlled-replica-propagation/dist/host-cluster-install.yaml"
  kubectl --context=karmada-host apply -f "$repo_root/actions-handler/dist/host-cluster-install.yaml"

  kubectl --context=karmada-host -n crp-system create secret generic karmada-crp-config \
    --from-file=karmada.config="$crp_config" \
    --dry-run=client -o yaml | kubectl --context=karmada-host apply -f -
  kubectl --context=karmada-host -n actions-handler-system create secret generic karmada-ah-config \
    --from-file=karmada.config="$ah_config" \
    --dry-run=client -o yaml | kubectl --context=karmada-host apply -f -
  rm -f "$crp_config" "$ah_config"

  kubectl --context=karmada-host -n crp-system rollout status deployment/crp-controller-manager --timeout=300s
  kubectl --context=karmada-host -n actions-handler-system rollout status statefulset/actions-handler --timeout=300s
}

install_istio() {
  for member in cluster2 cluster3; do
    istioctl install --context "$member" --set profile=default --skip-confirmation --readiness-timeout 5m
    kubectl --context="$member" -n istio-system wait pod --all --for=condition=Ready --timeout=300s
  done
}

main() {
  preflight
  clean_state
  create_kind_clusters
  install_karmada
  join_members
  "$repo_root/experiments/repair_local_testbed_endpoints.sh"
  install_headlight_control
  install_istio
  "$repo_root/experiments/install_metrics_server.sh"
  "$repo_root/experiments/deploy_minimal_trainticket.sh"

  echo
  echo "Staged local testbed is ready."
  echo "KUBECONFIG=$kubeconfig"
  echo "Internal member kubeconfig refreshed at $internal_kubeconfig"
  echo "Run the minimal smoke test with:"
  echo "  HEADLIGHT_KUBECONFIG=$kubeconfig $repo_root/experiments/run_minimal_trainticket_smoke.sh"
}

main "$@"
