#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

context="karmada-apiserver"
cluster_name="karmada-apiserver"
address="$(kubectl --context "$context" config view --minify -o jsonpath='{.clusters[].cluster.server}')"

namespace="actions-handler-system"
service_account_name="actions-handler-manager"

ca="$(kubectl --context "$context" get configmap cluster-info --namespace kube-public -o json | yq '.data.kubeconfig' | yq '.clusters[0].cluster.certificate-authority-data')"
# 365 days
token="$(kubectl create token --duration 31536000s --namespace "$namespace" "$service_account_name")"

echo "apiVersion: v1
kind: Config
clusters:
- name: ${cluster_name}
  cluster:
    certificate-authority-data: ${ca}
    server: ${address}
contexts:
- name: ${service_account_name}@${cluster_name}
  context:
    cluster: ${cluster_name}
    namespace: ${namespace}
    user: ${service_account_name}
current-context: ${service_account_name}@${cluster_name}
users:
- name: ${service_account_name}
  user:
    token: ${token}" > sa.karmadaconfig
