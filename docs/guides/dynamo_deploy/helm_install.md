# Manual Helm Deployment

Manual Helm chart deployment for Dynamo Platform.

## Prerequisites

- Kubernetes cluster
- Helm 3.x installed
- kubectl configured

## Installation Steps

### 1. Add Helm Repository

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

### 2. Install Dynamo Platform

```bash
helm install dynamo-platform nvidia/dynamo-platform \
  --namespace dynamo-system \
  --create-namespace
```

### 3. Verify Installation

```bash
kubectl get pods -n dynamo-system
```

## Configuration

### Custom Values

Create a `values.yaml` file for custom configuration:

```yaml
# Example values.yaml
replicaCount: 2
image:
  tag: "latest"
resources:
  limits:
    nvidia.com/gpu: 1
```

Install with custom values:

```bash
helm install dynamo-platform nvidia/dynamo-platform \
  --namespace dynamo-system \
  --create-namespace \
  -f values.yaml
```

## Upgrade

```bash
helm upgrade dynamo-platform nvidia/dynamo-platform \
  --namespace dynamo-system
```

## Uninstall

```bash
helm uninstall dynamo-platform --namespace dynamo-system
```