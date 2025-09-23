#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to install vLLM and its dependencies
# If installing vLLM from a release tag, we will use pip to manage the install
# Otherwise, we will use git to checkout the vLLM source code and build it from source.
# The dependencies are installed in the following order:
# 1. vLLM
# 2. LMCache
# 3. DeepGEMM
# 5. EP kernels

set -euo pipefail

VLLM_REF="v0.10.2"

# Basic Configurations
ARCH=$(uname -m)
MAX_JOBS=16
INSTALLATION_DIR=/tmp

# VLLM and Dependency Configurations
TORCH_BACKEND="cu128"
TORCH_CUDA_ARCH_LIST="9.0;10.0" # For EP Kernels
DEEPGEMM_REF=""
CUDA_VERSION="12.8" # For DEEPGEMM

# These flags are applicable when installing vLLM from source code
EDITABLE=true
VLLM_GIT_URL="https://github.com/vllm-project/vllm.git"
FLASHINF_REF="v0.3.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --editable)
            EDITABLE=true
            shift
            ;;
        --no-editable)
            EDITABLE=false
            shift
            ;;
        --vllm-ref)
            VLLM_REF="$2"
            shift 2
            ;;
        --vllm-git-url)
            VLLM_GIT_URL="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --installation-dir)
            INSTALLATION_DIR="$2"
            shift 2
            ;;
        --deepgemm-ref)
            DEEPGEMM_REF="$2"
            shift 2
            ;;
        --flashinf-ref)
            FLASHINF_REF="$2"
            shift 2
            ;;
        --torch-backend)
            TORCH_BACKEND="$2"
            shift 2
            ;;
        --torch-cuda-arch-list)
            TORCH_CUDA_ARCH_LIST="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--editable|--no-editable] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--deepgemm-ref REF] [--flashinf-ref REF] [--torch-backend BACKEND] [--torch-cuda-arch-list LIST] [--cuda-version VERSION]"
            echo "Options:"
            echo "  --editable        Install vllm in editable mode (default)"
            echo "  --no-editable     Install vllm in non-editable mode"
            echo "  --vllm-ref REF    Git reference to checkout (default: ${VLLM_REF})"
            echo "  --max-jobs NUM    Maximum number of parallel jobs (default: ${MAX_JOBS})"
            echo "  --arch ARCH       Architecture (amd64|arm64, default: auto-detect)"
            echo "  --installation-dir DIR  Directory to install vllm (default: ${INSTALLATION_DIR})"
            echo "  --deepgemm-ref REF  Git reference for DeepGEMM (default: ${DEEPGEMM_REF})"
            echo "  --flashinf-ref REF  Git reference for Flash Infer (default: ${FLASHINF_REF})"
            echo "  --torch-backend BACKEND  Torch backend to use (default: ${TORCH_BACKEND})"
            echo "  --torch-cuda-arch-list LIST  CUDA architectures to compile for (default: ${TORCH_CUDA_ARCH_LIST})"
            echo "  --cuda-version VERSION  CUDA version to use (default: ${CUDA_VERSION})"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert x86_64 to amd64 for consistency with Docker ARG
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

export MAX_JOBS=$MAX_JOBS
export CUDA_HOME=/usr/local/cuda

echo "=== Installing prerequisites ==="
uv pip install pip cuda-python

echo "\n=== Configuration Summary ==="
echo "  VLLM_REF=$VLLM_REF | EDITABLE=$EDITABLE | ARCH=$ARCH"
echo "  MAX_JOBS=$MAX_JOBS | TORCH_BACKEND=$TORCH_BACKEND | CUDA_VERSION=$CUDA_VERSION"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  DEEPGEMM_REF=$DEEPGEMM_REF | FLASHINF_REF=$FLASHINF_REF"
echo "  INSTALLATION_DIR=$INSTALLATION_DIR | VLLM_GIT_URL=$VLLM_GIT_URL"

echo "\n=== Cloning vLLM repository ==="
cd $INSTALLATION_DIR
git clone $VLLM_GIT_URL vllm
cd vllm
git checkout $VLLM_REF

echo "\n=== Installing vLLM ==="

if [[ $VLLM_REF =~ ^v ]]; then
    # VLLM_REF starts with 'v' - use pip install with version tag
    echo "Installing vLLM $VLLM_REF from PyPI..."
    uv pip install vllm[flashinfer]==$VLLM_REF --torch-backend=$TORCH_BACKEND
else
    # VLLM_REF does not start with 'v' - use git checkout path
    if [ "$ARCH" = "arm64" ]; then
        echo "Building vLLM from source for ARM64 architecture..."

        # Try to install specific PyTorch version first, fallback to latest nightly
        echo "Attempting to install pinned PyTorch nightly versions..."
        if ! uv pip install torch==2.7.1+cu128 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl; then
            echo "Pinned versions failed"
            exit 1
            # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        fi

        python use_existing_torch.py
        uv pip install -r requirements/build.txt

        if [ "$EDITABLE" = "true" ]; then
            MAX_JOBS=${MAX_JOBS} uv pip install --no-build-isolation -e . -v
        else
            MAX_JOBS=${MAX_JOBS} uv pip install --no-build-isolation . -v
        fi

        echo "\n=== Installing FlashInfer from source ==="
        cd $INSTALLATION_DIR
        git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
        cd flashinfer
        git checkout $FLASHINF_REF
        uv pip install -v --no-build-isolation .

    else
        echo "Building vLLM from source for AMD64 architecture..."

        # When updating above VLLM_REF make sure precompiled wheel file URL is correct. Run this command:
        # aws s3 ls s3://vllm-wheels/${VLLM_REF}/ --region us-west-2 --no-sign-request
        export VLLM_PRECOMPILED_WHEEL_LOCATION="https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_REF}/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl"

        if [ "$EDITABLE" = "true" ]; then
            uv pip install -e . --torch-backend=$TORCH_BACKEND
        else
            uv pip install . --torch-backend=$TORCH_BACKEND
        fi

        echo "\n=== Installing FlashInfer from PyPI ==="
        uv pip install flashinfer-python==$FLASHINF_REF

    fi
fi

echo "✓ vLLM installation completed"

echo "\n=== Installing LMCache ==="
if [ "$ARCH" = "amd64" ]; then
    # LMCache installation currently fails on arm64 due to CUDA dependency issues:
    # OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
    # TODO: Re-enable for arm64 after verifying lmcache compatibility and resolving the build issue.
    uv pip install lmcache==0.3.3
    echo "✓ LMCache installed"
else
    echo "⚠ Skipping LMCache on ARM64 (compatibility issues)"
fi

echo "\n=== Installing DeepGEMM ==="
cd tools/
if [ -n "$DEEPGEMM_REF" ]; then
    bash install_deepgemm.sh --cuda-version "${CUDA_VERSION}" --ref "$DEEPGEMM_REF"
else
    bash install_deepgemm.sh --cuda-version "${CUDA_VERSION}"
fi
echo "✓ DeepGEMM installation completed"

echo "\n=== Installing EP Kernels (PPLX and DeepEP) ==="
cd ep_kernels/
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" bash install_python_libraries.sh

echo "\n✅ All installations completed successfully!"