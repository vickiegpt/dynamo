# Building Dynamo for ARM

This document describes how to build Dynamo for ARM 64-bit architecture using PyTorch container 25.03. This works for GB200 machines.

## Building Dynamo for ARM with PyTorch container 25.03

### Prepare Dynamo repositories

Let's assume that you will used ``/tmp/dynamo`` as a source directory for Dynamo


Change directory to the source directory:

```bash
cd /tmp/
```

Clone the Dynamo repository:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
```

Change directory to the Dynamo repository:

```bash
cd dynamo
```

### Prepare Dynamo

Dynamo forces pytest version, which is not compatible with the latest version of PyTorch. You need to modify ``pyproject.toml`` to use ``pytest`` instead of ``pytest>=8.3.4``:

```diff
diff --git a/pyproject.toml b/pyproject.toml
index aaad9e1b..80472be2 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -25,7 +25,7 @@ license = { text = "Apache-2.0" }
 license-files = ["LICENSE"]
 requires-python = ">=3.10"
 dependencies = [
-    "pytest>=8.3.4",
+    "pytest",
     "bentoml==1.4.8",
     "types-psutil==7.0.0.20250218",
     "kubernetes==32.0.1",
```

This build dependecy leaks to runtime wheel, which can't be used in GB200 machine, which has different pytest version installed.

### Build Dynamo container with wheels

Build the Dynamo container for custom architecture and docker container:

```bash
bash container/build.sh --platform linux/arm64 \
    --framework vllm \
    --build-arg ARCH=arm64 \
    --build-arg ARCH_ALT=aarch64 \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch \
    --build-arg BASE_IMAGE_TAG=25.03-py3 \
    --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/pytorch \
    --build-arg RUNTIME_IMAGE_TAG=25.03-py3 \
    --build-arg MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_aarch64 \
    --build-arg MOFED_VERSION=24.10-2.1.8.0 \
    --nixl-repo piotrm-nvidia/nixl.git \
    --nixl-commit 8c4dcc1399c951632b6083303ce2e95dc7dcc7b9
```

### Run interactive container to get Dynamo wheels


Run the container with the following command to mount the source directories. You can't use workspace volume mount because it will hide folder ``/workspace/dist`` with pre-built vLLM wheel, which you need to install at system level instead of uv venv to use PyTorch from 25.03 container.

```
bash container/run.sh -it -v /tmp:/tmp
```

Copy the vLLM wheel to the host machine:

```
cp -r /workspace/dist /tmp/arm_wheels
```

Dynamo container creates its own virtual environment, which is active in the container. This can't work for you because only system level environment in container includes valid PyTorch working in ARM64 GB200 machine. Dynamo virtual environment includes only custom Dynamo dependencies not PyTorch.


## Using Dynamo vLLM patch in custom vLLM build

### Compile vLLM in PyTorch 25.03 container

#### Start container

You need to compile vLLM in the same container with PyTorch 25.03.

Run the following command to start the container:

```bash
docker run -ti \
    --gpus all \
    --network=host \
    --ulimit core=-1 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_PTRACE \
    --shm-size 2G \
    -v /tmp:/tmp \
    nvcr.io/nvidia/pytorch:25.03-py3  bash
```

#### Prepare Dynamo version of vLLM

Clone the vLLM repository:

```
git clone https://github.com/vllm-project/vllm.git
```

Checkout the vLLM version used as a base for Dynamo:

```
cd vllm
git checkout v0.8.4
```

Apply Dynamo patch to vLLM:

```
patch -p1 < /tmp/dynamo/container/deps/vllm/vllm_v0.8.4-dynamo-kv-disagg-patch.patch
```


#### Prepare vLLM for 25.03 PyTorch container


Remove all PyTorch dependencies from the vLLM requirements:

```
python use_existing_torch.py
```


Numba at version 0.61.2 is not supported by PyTorch 25.03 container:

```diff
diff --git a/requirements/neuron.txt b/requirements/neuron.txt
index 5f25bd054..783b95a87 100644
--- a/requirements/neuron.txt
root@gb-nvl-053-compute07:/tmp/vllm# git diff requirements/cuda.txt
diff --git a/requirements/cuda.txt b/requirements/cuda.txt
index cdc6ee75a..554b1d79d 100644
--- a/requirements/cuda.txt
+++ b/requirements/cuda.txt
@@ -1,13 +1,8 @@
 # Common dependencies
 -r common.txt

-numba == 0.60.0; python_version == '3.9' # v0.61 doesn't support Python 3.9. Required for N-gram speculative decoding
-numba == 0.61.2; python_version > '3.9'
+#numba == 0.60.0; python_version == '3.9' # v0.61 doesn't support Python 3.9. Required for N-gram speculative decoding
+#numba == 0.61.2; python_version > '3.9'

 # Dependencies for NVIDIA GPUs
 ray[cgraph]>=2.43.0, !=2.44.* # Ray Compiled Graph, required for pipeline parallelism in V1.
```

The original patch for Dynamo modifies version to use ``ai_dynamo_vllm`` instead of vllm. When you install vLLM with editable compilation mode the name of package is still vLLM so you need to modify ``vllm/platforms/__init__.py`` to use ``vllm`` instead of ``ai_dynamo_vllm``.

Change for platforms/__init__.py:

```diff
diff --git a/vllm/platforms/__init__.py b/vllm/platforms/__init__.py
index 08dbc0e78..8d48d488d 100644
--- a/vllm/platforms/__init__.py
+++ b/vllm/platforms/__init__.py
@@ -21,7 +21,7 @@ def vllm_version_matches_substr(substr: str) -> bool:
     from importlib.metadata import PackageNotFoundError, version
     try:
         logger.warning("Using ai_dynamo_vllm")
-        vllm_version = version("ai_dynamo_vllm")
+        vllm_version = version("vllm")
     except PackageNotFoundError as e:
         logger.warning(
             "The vLLM package was not found, so its version could not be "
```



#### Compile vLLM


Install all dependencies from the requirements/build.txt file:

```
pip install -r  requirements/build.txt
```

Run editable compilation of vLLM:

```
pip install -e . --no-build-isolation
```

This takes a while because CUDA kernels are compiled.




## Prepare Dynamo environment

You should use PyTorch 25.03 container executed above to install Dynamo wheels and dependencies with environment containing vLLM compiled for ARM64.

### Install dynamo wheels and dependencies

Install dynamo wheels:

```
pip install /tmp/arm_wheels/*.whl
```

Download nats-server:

```
wget https://github.com/nats-io/nats-server/releases/download/v2.10.24/nats-server-v2.10.24-arm64.deb
```

Install nats-server:

```
dpkg -i nats-server-v2.10.24-arm64.deb
```

Download etcd:

```
wget https://github.com/etcd-io/etcd/releases/download/v3.5.18/etcd-v3.5.18-linux-arm64.tar.gz
```

Install etcd:

```
tar -xvf etcd-v3.5.18-linux-arm64.tar.gz
```

#### Start services for Dynamo

Start nats-server:

```
nats-server -js
```

Start etcd:

```
etcd-v3.5.18-linux-arm64/etcd
```



## Run disaggregated inference

You can run disaggregated inference in containe which includes both Dynamo and vLLM compiled for ARM and CUDA.

Dynamo repo contains folder ``examples/llm`` with example of disaggregated inference.

```
cd examples/llm
```





