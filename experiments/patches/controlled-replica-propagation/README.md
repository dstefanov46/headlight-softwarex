# Controlled Replica Propagation suspension barrier

This patch is pinned to upstream revision
`8d11e02ab7b3005e7c0f9c74a32b3aaa67f35c25`. It makes the controller wait
until Karmada's `ResourceBinding` has observed dispatch suspension before it
changes the aggregate Deployment replica count or placement weights. This
prevents the old weights from transiently dispatching a third member-cluster
pod during a `2/1 -> 2/2` transition.

The local qualification image has Docker image ID
`sha256:07d531df001fc446394b5ad5e64b217590199e694e2cb59e9a651569af87ddcb`.

## Build and focused test

```bash
workdir=$(mktemp -d)
git clone https://gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/controlled-replica-propagation.git "$workdir/crp"
git -C "$workdir/crp" checkout 8d11e02ab7b3005e7c0f9c74a32b3aaa67f35c25
git -C "$workdir/crp" apply /home/dimi/headlight-jsa/experiments/patches/controlled-replica-propagation/0001-wait-for-resourcebinding-suspension.patch
go -C "$workdir/crp" test ./cmd ./internal/controller -run TestScaleOutWaitsForResourceBindingDispatchSuspension -count=1
docker build -t headlight/controlled-replica-propagation:local-24gb-tested-rb-suspension "$workdir/crp"
docker image inspect headlight/controlled-replica-propagation:local-24gb-tested-rb-suspension --format '{{.Id}}'
```

## Install in the existing isolated testbed

These commands do not rebuild any cluster. The `ResourceBinding` permission is
added to the Karmada API ClusterRole, while the controller Deployment runs in
the `karmada-host` cluster.

```bash
export KUBECONFIG=/home/dimi/headlight-jsa/.kube/headlight-testbed.config
kind load docker-image --name karmada-host headlight/controlled-replica-propagation:local-24gb-tested-rb-suspension
kubectl --context karmada-apiserver patch clusterrole crp-manager-role --type=json -p='[{"op":"add","path":"/rules/-","value":{"apiGroups":["work.karmada.io"],"resources":["resourcebindings"],"verbs":["get","list","watch"]}}]'
kubectl --context karmada-host -n crp-system set image deployment/crp-controller-manager manager=headlight/controlled-replica-propagation:local-24gb-tested-rb-suspension
kubectl --context karmada-host -n crp-system rollout status deployment/crp-controller-manager --timeout=300s
```

The JSON patch is intentionally one-time. On an already-patched cluster,
inspect `crp-manager-role` first and do not append a duplicate rule.

## Qualification actuation path

Run the conditional retest through the repository runner so every scale action
uses the ConfigMap compare-and-swap admission queue and retains the queue lease
until all requested member replicas are Ready:

```bash
python3 /home/dimi/headlight-jsa/experiments/run_local_24gb_conditional_retest.py \
  --output-dir /home/dimi/headlight-jsa/results/research-v4-infrastructure-qualification/seed-201/conditional-retest-attempt-YYYYMMDDTHHMMSSZ/retest-run
```

The queue orders contenders by decision timestamp and then stable service/name,
admits one transition at a time, rejects projected per-cluster usage above one
extra application replica, accounts for starting and terminating pods, and
retains the 2078Mi namespace quota as a hard backstop.
