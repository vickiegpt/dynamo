# Kubernetes utilities for Dynamo

This directory contains small utilities and manifests used by benchmarking and profiling flows.

## Contents

- `setup_k8s_namespace.sh` — one-time per Kubernetes namespace. Creates namespace (if missing), applies common manifests, and installs the Dynamo operator. If `DOCKER_SERVER`/`IMAGE_TAG` are provided, it installs your custom operator image; otherwise it installs the default published image. If your registry is private, provide `DOCKER_USERNAME`/`DOCKER_PASSWORD` or respond to the prompt to create an image pull secret.
- `manifests/`
  - `serviceaccount.yaml` — ServiceAccount `dynamo-sa`
  - `role.yaml` — Role `dynamo-role`
  - `rolebinding.yaml` — RoleBinding `dynamo-binding`
  - `pvc.yaml` — PVC `dynamo-pvc`
  - `pvc-access-pod.yaml` — short‑lived pod for copying profiler results from the PVC
- `kubernetes.py` — helper used by tooling to apply/read resources (e.g., access pod for PVC downloads).

## Quick start

### Kubernetes Setup (one-time per namespace)

Use the helper script to prepare a Kubernetes namespace with the common manifests and install the operator.
This script creates a Kubernetes namespace with the given name if it does not yet exist. It then applies common manifests (serviceaccount, role, rolebinding, pvc), creates secrets, and deploys the Dynamo Cloud Operator to your namespace.
If your namespace is already set up, you can skip this step.

```bash
export HF_TOKEN=<HF_TOKEN>
export DOCKER_SERVER=<YOUR_DOCKER_SERVER>

NAMESPACE=benchmarking HF_TOKEN=$HF_TOKEN DOCKER_SERVER=$DOCKER_SERVER deploy/utils/setup_k8s_namespace.sh

# IF you want to build and push a new Docker image for the Dynamo Cloud Operator, include an IMAGE_TAG
# NAMESPACE=benchmarking HF_TOKEN=$HF_TOKEN DOCKER_SERVER=$DOCKER_SERVER IMAGE_TAG=latest deploy/utils/setup_k8s_namespace.sh
```

This script applies the following manifests:

- `deploy/utils/manifests/serviceaccount.yaml` - ServiceAccount `dynamo-sa`
- `deploy/utils/manifests/role.yaml` - Role `dynamo-role`
- `deploy/utils/manifests/rolebinding.yaml` - RoleBinding `dynamo-binding`
- `deploy/utils/manifests/pvc.yaml` - PVC `dynamo-pvc`

If `DOCKER_SERVER` and `IMAGE_TAG` are not both provided, the script deploys the operator using the default published image `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.0`.
To build/push and use a new image instead, pass both `DOCKER_SERVER` and `IMAGE_TAG`.

This script also installs the Dynamo CRDs if not present.

If the registry is private, either pass credentials or respond to the prompt:

```bash
NAMESPACE=benchmarking \
DOCKER_SERVER=my-registry.example.com \
IMAGE_TAG=latest \
DOCKER_USERNAME="$oauthtoken" \
DOCKER_PASSWORD=<token> \
deploy/utils/setup_k8s_namespace.sh
```

If `DOCKER_SERVER`/`IMAGE_TAG` are omitted, the script installs the default operator image `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.0`.

### PVC Manipulation Scripts

Inject a manifest into the PVC and download the PVC contents:

```bash
python3 deploy/utils/inject_manifest.py \
  --namespace $NAMESPACE \
  --src ./my-disagg.yaml \
  --dest /profiling_results/disagg.yaml

python3 benchmarks/profiler/download_pvc_results.py \
  --namespace $NAMESPACE \
  --output-dir ./all_pvc_files \
  --no-config
```

## Notes

- Benchmarking scripts (`benchmarks/benchmark.sh`, `benchmarks/deploy_benchmark.sh`) call this setup automatically when present.
- Profiling job manifest remains in `benchmarks/profiler/deploy/profile_sla_job.yaml` and now relies on the common ServiceAccount/PVC here.
