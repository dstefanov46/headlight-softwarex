#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="${HEADLIGHT_KUBECONFIG:-$repo_root/.kube/headlight-testbed.config}"
namespace="${HEADLIGHT_TRAINTICKET_NAMESPACE:-trainticket}"
if [[ "$namespace" != trainticket ]]; then
  echo "gateway protocol currently supports only the trainticket namespace" >&2
  exit 2
fi
seconds="${HEADLIGHT_GATEWAY_SMOKE_SECONDS:-90}"
users="${HEADLIGHT_GATEWAY_SMOKE_USERS:-2}"
image="${HEADLIGHT_LOCUST_IMAGE:-dstefanov46/locust-test:0.66.0}"
result_dir="${HEADLIGHT_GATEWAY_SMOKE_RESULT_DIR:-$repo_root/results/headlight-v2-live/gateway-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
gateway_url="http://trainticket-gateway.$namespace.svc.cluster.local:8080"
job="headlight-gateway-smoke-$(date -u +%H%M%S)"
configmap="$job"

if [[ ! "$seconds" =~ ^[1-9][0-9]*$ || ! "$users" =~ ^[1-9][0-9]*$ ]]; then
  echo "smoke seconds and users must be positive integers" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
for cluster in cluster2 cluster3; do
  kubectl --context="$cluster" -n "$namespace" get service \
    ts-auth-service ts-route-service ts-travel-service >/dev/null
done
kubectl --context=cluster1 -n "$namespace" get service trainticket-gateway >/dev/null

mkdir -p "$result_dir"
printf 'elapsed_seconds,users,spawn_rate\n0,%s,1\n%s,%s,1\n' \
  "$users" "$seconds" "$users" > "$result_dir/load_profile.csv"
kubectl --context=cluster1 -n "$namespace" create configmap "$configmap" \
  --from-file=trainticket_paper.py="$repo_root/experiments/testbed/trainticket-paper/trainticket_paper.py" \
  --from-file=load_profile.csv="$result_dir/load_profile.csv" > "$result_dir/configmap-create.log"

cat > "$result_dir/job.yaml" <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
  namespace: $namespace
spec:
  backoffLimit: 0
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      restartPolicy: Never
      containers:
        - name: locust
          image: $image
          imagePullPolicy: IfNotPresent
          command: [/bin/sh, -lc]
          args:
            - |
              /opt/venv/bin/locust --headless -f /app/trainticket_paper.py --run-time "${seconds}s" --exit-code-on-error 0 --csv /tmp/headlight-gateway
              rc=\$?
              echo __HEADLIGHT_REQUEST_CSV_BEGIN__
              cat /app/headlight_requests.csv 2>/dev/null || true
              echo __HEADLIGHT_REQUEST_CSV_END__
              exit "\$rc"
          env:
            - name: HEADLIGHT_REQUEST_CSV
              value: /app/headlight_requests.csv
            - name: HEADLIGHT_LOAD_SHAPE
              value: /app/load_profile.csv
            - name: HEADLIGHT_TRAINTICKET_MODE
              value: advanced-ticket
            - name: HEADLIGHT_TRAINTICKET_NAMESPACE
              value: $namespace
            - name: HEADLIGHT_TRAINTICKET_GATEWAY_URL
              value: $gateway_url
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          volumeMounts:
            - name: workload
              mountPath: /app/trainticket_paper.py
              subPath: trainticket_paper.py
            - name: workload
              mountPath: /app/load_profile.csv
              subPath: load_profile.csv
      volumes:
        - name: workload
          configMap:
            name: $configmap
YAML

"$repo_root/experiments/scripts/gateway_weights.py" --once > "$result_dir/initial-weights.jsonl"
"$repo_root/experiments/scripts/gateway_weights.py" --interval 5 > "$result_dir/weights.jsonl" 2> "$result_dir/weights.err" &
weights_pid=$!
trap 'kill "$weights_pid" >/dev/null 2>&1 || true' EXIT
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
kubectl --context=cluster1 apply -f "$result_dir/job.yaml" > "$result_dir/job-apply.log"
if ! kubectl --context=cluster1 -n "$namespace" wait --for=condition=Complete "job/$job" --timeout=4m; then
  kubectl --context=cluster1 -n "$namespace" logs "job/$job" > "$result_dir/locust.log" 2>&1 || true
  echo "gateway smoke Job did not complete; evidence: $result_dir" >&2
  exit 1
fi
kubectl --context=cluster1 -n "$namespace" logs "job/$job" > "$result_dir/locust.log"
awk '/__HEADLIGHT_REQUEST_CSV_BEGIN__/{flag=1;next}/__HEADLIGHT_REQUEST_CSV_END__/{flag=0}flag' \
  "$result_dir/locust.log" > "$result_dir/request_events.csv"
kubectl --context=cluster1 -n "$namespace" logs deployment/trainticket-gateway \
  --since-time="$started_at" > "$result_dir/gateway.log"
if ! kill -0 "$weights_pid" 2>/dev/null; then
  echo "gateway weight updater exited during smoke; evidence: $result_dir" >&2
  exit 1
fi

for endpoint in auth.login route.list travel.left; do
  count="$(awk -F, -v endpoint="$endpoint" 'NR > 1 && $3 == endpoint && $4 ~ /^[23][0-9][0-9]$/ && $6 == "" {n++} END {print n+0}' "$result_dir/request_events.csv")"
  if (( count == 0 )); then
    echo "no successful $endpoint requests; evidence: $result_dir" >&2
    exit 1
  fi
  echo "GATEWAY_WORKFLOW_PASS endpoint=$endpoint successes=$count"
done
for backend in auth route travel; do
  for cluster in cluster2 cluster3; do
    count="$(rg -c "${backend}/${cluster} .* [23][0-9][0-9] " "$result_dir/gateway.log" || true)"
    count="${count:-0}"
    if (( count == 0 )); then
      echo "no successful $backend requests routed to $cluster; evidence: $result_dir" >&2
      exit 1
    fi
    echo "GATEWAY_DISTRIBUTION_PASS backend=$backend cluster=$cluster successes=$count"
  done
done
echo "GATEWAY_SMOKE_PASS evidence=$result_dir"
