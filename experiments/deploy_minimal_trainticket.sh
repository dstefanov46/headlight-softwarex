#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/dimi/bin:$PATH"

kubeconfig="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
path="${HEADLIGHT_TRAINTICKET_PATH:-login-route}"
deploy_mode="${HEADLIGHT_TRAINTICKET_DEPLOY_MODE:-karmada}"
active_clusters="${HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS:-cluster2 cluster3}"
if [[ "$path" = "full" ]]; then
  resource_profile="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-source}"
else
  resource_profile="${HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE:-local-tiny}"
fi
rendered="${HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST:-$repo_root/experiments/testbed/trainticket-minimal/rendered-${path}.yaml}"
allow_unready=" ${HEADLIGHT_WORKLOAD_ALLOW_UNREADY_DEPLOYMENTS:-} "
seed_only="${HEADLIGHT_TRAINTICKET_SEED_ONLY:-0}"
mongo_storage="${HEADLIGHT_TRAINTICKET_MONGO_STORAGE:-pod-emptydir}"

export KUBECONFIG="$kubeconfig"

if [[ "$seed_only" != "1" ]]; then
render_args=(
  --namespace "$namespace"
  --path "$path"
  --resource-profile "$resource_profile"
  --mongo-storage "$mongo_storage"
  --output "$rendered"
)
if [[ "$deploy_mode" = "single-cluster" ]]; then
  render_args+=(--no-policies)
fi

python3 "$repo_root/experiments/scripts/render_minimal_trainticket.py" "${render_args[@]}"

if [[ "$deploy_mode" = "single-cluster" ]]; then
  for cluster in $active_clusters; do
    kubectl --context="$cluster" get namespace "$namespace" >/dev/null 2>&1 || \
      kubectl --context="$cluster" create namespace "$namespace"
    kubectl --context="$cluster" label namespace "$namespace" istio-injection=enabled --overwrite
    kubectl --context="$cluster" apply -f "$rendered"
  done
else
  kubectl --context=karmada-apiserver apply -f "$rendered"
fi

mapfile -t deployments < <(python3 - "$rendered" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    for doc in yaml.safe_load_all(handle):
        if doc and doc.get("kind") == "Deployment":
            print(doc["metadata"]["name"])
PY
)

mapfile -t sidecar_deployments < <(python3 - "$rendered" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    for doc in yaml.safe_load_all(handle):
        if doc and doc.get("kind") == "Deployment":
            annotations = doc.get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {})
            if annotations.get("sidecar.istio.io/inject") == "true":
                print(doc["metadata"]["name"])
PY
)

refresh_crp_dispatch() {
  if [[ "$deploy_mode" == "single-cluster" ]]; then
    return
  fi

  # The CRP controller briefly suspends its generated PropagationPolicy while
  # updating replicas. Karmada can occasionally retain that suspension in the
  # ResourceBinding after the policy is unsuspended, leaving one member
  # without a deployment. Touch the CRPs first, wait for their policies to be
  # unsuspended, then touch the policies so Karmada refreshes the bindings.
  local deployment policy dispatch pending reconcile_token
  reconcile_token="$(date +%s%N)"
  for deployment in "${deployments[@]}"; do
    kubectl --context=karmada-apiserver -n "$namespace" annotate \
      controlledreplicapropagation "$deployment" \
      headlight.nephele-project.eu/reconcile-at="$reconcile_token" \
      --overwrite >/dev/null 2>&1 || true
  done

  for _ in $(seq 1 30); do
    pending=0
    for deployment in "${deployments[@]}"; do
      policy="crp-owned-$deployment"
      if ! kubectl --context=karmada-apiserver -n "$namespace" get \
        propagationpolicy "$policy" >/dev/null 2>&1; then
        pending=1
        break
      fi
      dispatch="$(kubectl --context=karmada-apiserver -n "$namespace" get \
        propagationpolicy "$policy" \
        -o jsonpath='{.spec.suspension.dispatching}' 2>/dev/null || true)"
      if [[ "$dispatch" == "true" ]]; then
        pending=1
        break
      fi
    done
    if [[ "$pending" -eq 0 ]]; then
      break
    fi
    sleep 1
  done

  reconcile_token="$(date +%s%N)"
  for deployment in "${deployments[@]}"; do
    policy="crp-owned-$deployment"
    kubectl --context=karmada-apiserver -n "$namespace" annotate \
      propagationpolicy "$policy" \
      headlight.nephele-project.eu/dispatch-at="$reconcile_token" \
      --overwrite >/dev/null 2>&1 || true
  done
}

if [[ "$deploy_mode" != "single-cluster" ]]; then
  for cluster in $active_clusters; do
    kubectl --context="$cluster" get namespace "$namespace" >/dev/null 2>&1 || \
      kubectl --context="$cluster" create namespace "$namespace"
    kubectl --context="$cluster" label namespace "$namespace" istio-injection=enabled --overwrite
  done
fi

wait_for_workload() {
  for cluster in $active_clusters; do
    for deployment in "${deployments[@]}"; do
      for _ in $(seq 1 120); do
        if kubectl --context="$cluster" -n "$namespace" get deployment "$deployment" >/dev/null 2>&1; then
          break
        fi
        sleep 5
      done
      if ! kubectl --context="$cluster" -n "$namespace" get deployment "$deployment" >/dev/null 2>&1; then
        echo "Deployment $cluster/$namespace/$deployment was not propagated in time" >&2
        return 1
      fi
      if kubectl --context="$cluster" -n "$namespace" rollout status \
        "deployment/$deployment" \
        --timeout="${HEADLIGHT_WORKLOAD_WAIT_TIMEOUT:-600s}"; then
        continue
      fi
      if [[ "$allow_unready" == *" $deployment "* ]]; then
        echo "Allowing unready deployment $deployment in $cluster"
        continue
      fi
      echo "Deployment $cluster/$namespace/$deployment did not become ready" >&2
      return 1
    done

    echo "Workload pods in $cluster:"
    kubectl --context="$cluster" -n "$namespace" get pods \
      -o custom-columns='NAME:.metadata.name,READY:.status.containerStatuses[*].ready,CONTAINERS:.spec.containers[*].name'
  done
}

ensure_istio_sidecars() {
  local cluster attempt pod_json deployment
  local -a missing_pods
  for cluster in $active_clusters; do
    for attempt in 1 2 3; do
      pod_json="$(kubectl --context="$cluster" -n "$namespace" get pods -o json)"
      mapfile -t missing_pods < <(python3 -c '
import json
import sys

expected = set(sys.argv[1:])
for pod in json.load(sys.stdin)["items"]:
    if pod["metadata"].get("labels", {}).get("app") not in expected:
        continue
    if "istio-proxy" not in {container["name"] for container in pod["spec"]["containers"]}:
        print(pod["metadata"]["name"])
' "${sidecar_deployments[@]}" <<<"$pod_json")
      if (( ${#missing_pods[@]} == 0 )); then
        break
      fi
      if (( attempt == 3 )); then
        echo "Istio injection missing after two repairs in $cluster: ${missing_pods[*]}" >&2
        return 1
      fi
      echo "Recreating pods without required Istio sidecars in $cluster: ${missing_pods[*]}" >&2
      kubectl --context="$cluster" -n "$namespace" delete pod "${missing_pods[@]}" --wait=true
      for deployment in "${sidecar_deployments[@]}"; do
        kubectl --context="$cluster" -n "$namespace" rollout status \
          "deployment/$deployment" --timeout="${HEADLIGHT_WORKLOAD_WAIT_TIMEOUT:-600s}"
      done
    done
  done
}

deployment_attempts="${HEADLIGHT_WORKLOAD_DEPLOY_ATTEMPTS:-3}"
refresh_crp_dispatch
for attempt in $(seq 1 "$deployment_attempts"); do
  if wait_for_workload; then
    break
  fi
  if [[ "$attempt" -eq "$deployment_attempts" ]]; then
    echo "TrainTicket deployment did not converge after $deployment_attempts attempts" >&2
    exit 1
  fi
  echo "TrainTicket propagation incomplete; reapplying manifest (attempt $((attempt + 1))/$deployment_attempts)" >&2
  if [[ "$deploy_mode" = "single-cluster" ]]; then
    for cluster in $active_clusters; do
      kubectl --context="$cluster" -n "$namespace" apply -f "$rendered" >/dev/null
    done
  else
    kubectl --context=karmada-apiserver apply -f "$rendered" >/dev/null
  fi
  refresh_crp_dispatch
  sleep 10
done

ensure_istio_sidecars

fi

seed_auth_user() {
  local cluster="$1"
  local local_port="$2"
  local payload
  payload='{"userId":"4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f","userName":"fdse_microservice","password":"111111"}'

  kubectl --context="$cluster" -n "$namespace" port-forward svc/ts-auth-service "$local_port:12340" >/tmp/headlight-auth-seed-"$cluster".log 2>&1 &
  local pf_pid="$!"
  trap 'kill "$pf_pid" >/dev/null 2>&1 || true' RETURN

  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$local_port/api/v1/auth/hello" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if curl -fsS \
    -H 'Content-Type: application/json' \
    -H 'X-Verify-Code: 025080CF8BA94594B09E283F17815444' \
    -d '{"username":"fdse_microservice","password":"111111"}' \
    "http://127.0.0.1:$local_port/api/v1/users/login" | grep -q '"token"'; then
    kill "$pf_pid" >/dev/null 2>&1 || true
    trap - RETURN
    return 0
  fi

  curl -fsS \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "http://127.0.0.1:$local_port/api/v1/auth" >/dev/null
  curl -fsS \
    -H 'Content-Type: application/json' \
    -H 'X-Verify-Code: 025080CF8BA94594B09E283F17815444' \
    -d '{"username":"fdse_microservice","password":"111111"}' \
    "http://127.0.0.1:$local_port/api/v1/users/login" | grep -q '"token"'
  kill "$pf_pid" >/dev/null 2>&1 || true
  trap - RETURN
}

seed_route_data() {
  local cluster="$1"
  local local_port="$2"
  local routes
  routes=$(
    cat <<'ROUTES'
92708982-77af-4318-be25-57ccb0ff69ad|shanghai|suzhou|0,100
aefcef3f-3f42-46e8-afd7-6cb2a928bd3d|suzhou|shanghai|0,100
a3f256c1-0e43-4f7d-9c21-121bf258101f|nanjing|shanghai|0,300
084837bb-53c8-4438-87c8-0321a4d09917|shanghai|nanjing|0,300
f3d4d4ef-693b-4456-8eed-59c0d717dd08|nanjing|suzhou|0,250
ROUTES
  )

  kubectl --context="$cluster" -n "$namespace" port-forward svc/ts-route-service "$local_port:11178" >/tmp/headlight-route-seed-"$cluster".log 2>&1 &
  local pf_pid="$!"
  trap 'kill "$pf_pid" >/dev/null 2>&1 || true' RETURN

  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$local_port/api/v1/routeservice/welcome" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  while IFS='|' read -r route_id start_station end_station distance_list; do
    [[ -n "$route_id" ]] || continue
    local station_list payload
    station_list="$start_station,$end_station"
    payload=$(
      python3 - "$route_id" "$start_station" "$end_station" "$station_list" "$distance_list" <<'PY'
import json
import sys
print(json.dumps({
    "id": sys.argv[1],
    "startStation": sys.argv[2],
    "endStation": sys.argv[3],
    "stationList": sys.argv[4],
    "distanceList": sys.argv[5],
}))
PY
    )
    local seeded=0
    for _ in $(seq 1 5); do
      local response existing
      response="$(
        curl -sS \
          -H 'Content-Type: application/json' \
          -d "$payload" \
          "http://127.0.0.1:$local_port/api/v1/routeservice/routes" || true
      )"
      if grep -q '"status":1' <<< "$response"; then
        seeded=1
        break
      fi
      existing="$(curl -sS "http://127.0.0.1:$local_port/api/v1/routeservice/routes/$route_id" || true)"
      if grep -q '"status":1' <<< "$existing"; then
        seeded=1
        break
      fi
      sleep 2
    done
    [[ "$seeded" = "1" ]]
  done <<< "$routes"

  curl -fsS "http://127.0.0.1:$local_port/api/v1/routeservice/routes/92708982-77af-4318-be25-57ccb0ff69ad" | grep -q '"status":1'
  kill "$pf_pid" >/dev/null 2>&1 || true
  trap - RETURN
}

auth_seed_port=12340
route_seed_port=11178
for cluster in $active_clusters; do
  seed_auth_user "$cluster" "$auth_seed_port"
  seed_route_data "$cluster" "$route_seed_port"
  auth_seed_port=$((auth_seed_port + 1))
  route_seed_port=$((route_seed_port + 1))
done

if [[ "$seed_only" != "1" ]]; then
  echo "Rendered minimal TrainTicket manifest: $rendered"
fi
