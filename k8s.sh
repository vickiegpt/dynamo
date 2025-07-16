#!/usr/bin/env bash
set -euo pipefail

# 1. Install Homebrew if missing
if ! command -v brew &> /dev/null; then
  echo "Homebrew not found—installing prerequisites and Homebrew…"
  # Install build-time prerequisites
  apt-get update
  apt-get install -y build-essential procps curl file git

  # Non-interactive Homebrew install
  NONINTERACTIVE=1 \
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  # Load brew into this shell
  eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
else
  echo "Homebrew already installed, skipping."
fi

# 2. Ensure brew is up-to-date
echo "Updating Homebrew…"
brew update

# 3. Install Azure CLI
if ! command -v az &> /dev/null; then
  echo "Installing Azure CLI (az)…"
  brew install azure-cli
else
  echo "Azure CLI already installed, skipping."
fi

# 4. Install kubelogin
if ! command -v kubelogin &> /dev/null; then
  echo "Installing kubelogin…"
  brew install Azure/kubelogin/kubelogin
else
  echo "kubelogin already installed, skipping."
fi

# 5. Install kubectl
if ! command -v kubectl &> /dev/null; then
  echo "Installing kubectl (kubernetes-cli)…"
  brew install kubernetes-cli
else
  echo "kubectl already installed, skipping."
fi

echo "✅ All tools are installed and up-to-date."

echo >> /root/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

az login
az aks get-credentials --resource-group rg-aks-dynamo-dev --name aks-dynamo-dev
kubelogin convert-kubeconfig -l azurecli
kubectl auth can-i create deployments
