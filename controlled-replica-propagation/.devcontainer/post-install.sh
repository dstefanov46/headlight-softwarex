#!/bin/bash
set -x

KIND_VERSION="${KIND_VERSION:-v0.23.0}"
KUBEBUILDER_VERSION="${KUBEBUILDER_VERSION:-4.3.1}"
curl --proto '=https' --tlsv1.2 --fail --show-error --location -o ./kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
chmod +x ./kind
mv ./kind /usr/local/bin/kind

curl --proto '=https' --tlsv1.2 --fail --show-error --location -o kubebuilder "https://github.com/kubernetes-sigs/kubebuilder/releases/download/v${KUBEBUILDER_VERSION}/kubebuilder_linux_amd64"
chmod +x kubebuilder
mv kubebuilder /usr/local/bin/

KUBECTL_VERSION=$(curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location https://dl.k8s.io/release/stable.txt)
curl --proto '=https' --tlsv1.2 --fail --show-error --location --remote-name "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl"
chmod +x kubectl
mv kubectl /usr/local/bin/kubectl

docker network create -d=bridge --subnet=172.19.0.0/24 kind

kind version
kubebuilder version
docker --version
go version
kubectl version --client
