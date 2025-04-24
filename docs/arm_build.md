# Building Dynamo for ARM

This document describes how to build Dynamo for ARM 64-bit architecture using PyTorch container 25.03. This works for GB200 machines.

## Building Dynamo for ARM with PyTorch container 25.03

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


Build the Dynamo container for custom architecture and docker container:

```bash
bash container/build.sh --platform linux/arm64 \
    --framework vllm \
    --base-image nvcr.io/nvidia/pytorch \
    --base-image-tag 25.03-py3 \
    --runtime-image nvcr.io/nvidia/pytorch \
    --runtime-image-tag 25.03-py3 \
    --manylinux-image quay.io/pypa/manylinux_2_31_aarch64 \
    --build-arg MOFED_VERSION=24.10-2.1.8.0
```


wget https://content.mellanox.com/ofed/MLNX_OFED-24.10-2.1.8.0/MLNX_OFED_LINUX-24.10-2.1.8.0-ubuntu24.04-aarch64.tgz


    "https://content.mellanox.com/ofed/MLNX_OFED-24.10-2.1.8.0/MLNX_OFED_LINUX-24.10-2.1.8.0-ubuntu24.04-${ARCH_ALT}.tgz"


## Using Dynamo vLLM patch in custom vLLM build

### Prepare Dynamo version of vLLM

Clone the vLLM repository:

```
git clone https://github.com/vllm-project/vllm.git
```

Checkout the vLLM version used as a base for Dynamo:

```
cd vllm
git checkout v0.8.4
```

Apply the patch:

```
patch -p1 < /tmp/dynamo/container/deps/vllm/vllm_v0.8.4-dynamo-kv-disagg-patch.patch
```

### Run interactive container


Run the container with the following command to mount the source directories. You can't use workspace volume mount because it will hide folder ``/workspace/dist`` with pre-built vLLM wheel, which you need to install at system level instead of uv venv to use PyTorch from 25.03 container.

```
bash container/run.sh -it -v /tmp:/tmp
```


### Prepare vLLm for 25.03 PyTorch container


Remove all PyTorch dependencies from the vLLM requirements:

```
python use_existing_torch.py
```


Install all dependencies from the requirements/build.txt file:

```
pip install -r  requirements/build.txt
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
