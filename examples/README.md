# Quickstart Guide

## Deploying examples on Kubernetes

This guide will help you quickly deploy and clean up the dynamo example services in Kubernetes.

### Prerequisites

- **Dynamo Cloud** is already deployed in your target Kubernetes namespace.
- You have `kubectl` access to your cluster and the correct namespace set in `$NAMESPACE`.


### Create a secret with huggingface token

```bash
export HF_TOKEN="huggingfacehub token with read permission to models"
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN -n $KUBE_NS || true
```

---

Choose the example you want to deploy or delete. The YAML files are located in `examples/<backend>/deploy/k8s/`.

### Deploy dynamo example

```bash
kubectl apply -f examples/<Backend-Folder>/deploy/k8s/<Example yaml file> -n $NAMESPACE
```

### Uninstall dynamo example


```bash
kubectl delete -f examples/<Backend-Folder>/deploy/k8s/<Example yaml file> -n $NAMESPACE
```

### Using a different dynamo container

To customize the container image used in your deployment, you will need to update the manifest before applying it.

You can use [`yq`](https://github.com/mikefarah/yq?tab=readme-ov-file#install), a portable command-line YAML processor.

Please follow the [installation instructions](https://github.com/mikefarah/yq?tab=readme-ov-file#install) for your platform if you do not already have `yq` installed. After installing `yq`, you can generate and apply your manifest as follows:

1. prepare your custom manifest file
```bash
export DYNAMO_IMAGE=my-registry/my-image:tag
yq '.spec.services.[].extraPodSpec.mainContainer.image = env(DYNAMO_IMAGE)' $EXAMPLE_FILE > my_example_manifest.yaml
```

2. install the dynamo example
```bash
kubectl apply -f my_example_manifest.yaml -n $NAMESPACE
```

3. uninstall the dynamo example
```bash
kubectl delete -f my_example_manifest.yaml -n $NAMESPACE
```