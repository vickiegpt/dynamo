#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --patch PATH        Apply a patch file during installation"
    echo "  --ref REF          Specify the vLLM git reference (branch/tag/commit) to install"
    echo "  --install-cmd CMD  Specify the installation command (default: 'pip install')"
    echo "  --recompile        Recompile vLLM kernels"
    echo "  --installation-dir DIR  Specify the installation directory (default: 'vllm')"
    echo "  --flash-attn-src-dir DIR  Specify the flash-attn source directory (default: none)"
    echo "  --help             Show this help message"
}

# Default values
INSTALL_CMD="pip install"
VLLM_REF="main"
PATCH_PATH=""
RECOMPILE=false
FLASH_ATTN_SRC_DIR=""
# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch)
            PATCH_PATH="$2"
            shift 2
            ;;
        --ref)
            VLLM_REF="$2"
            shift 2
            ;;
        --install-cmd)
            INSTALL_CMD="$2"
            shift 2
            ;;
        --recompile)
            RECOMPILE="$2"
            shift 2
            ;;
        --installation-dir)
            INSTALLATION_DIR="$2"
            shift 2
            ;;
        --flash-attn-src-dir)
            FLASH_ATTN_SRC_DIR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create temp directory and clean it up on exit

# Convert patch path to absolute path if it's relative
if [[ ! "$PATCH_PATH" = /* ]]; then
    PATCH_PATH="$(pwd)/${PATCH_PATH}"
fi

# Clean up the installation directory
rm -r "$INSTALLATION_DIR"

# Clone vLLM repository
echo "Cloning vLLM repository at ref: $VLLM_REF"
git clone https://github.com/vllm-project/vllm.git "$INSTALLATION_DIR"
cd "$INSTALLATION_DIR"
git checkout "$VLLM_REF"

# Apply patch if provided
if [ -n "$PATCH_PATH" ]; then
    echo "Applying patch from: $PATCH_PATH"
    git apply "$PATCH_PATH"
fi

# Install using specified command
echo "Installing using: $INSTALL_CMD"
if [ "$RECOMPILE" = true ]; then
    echo "Recompiling vLLM"
    export VLLM_USE_PRECOMPILED=0
else
    export VLLM_USE_PRECOMPILED=1
fi

if [ -n "$FLASH_ATTN_SRC_DIR" ]; then
    export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_SRC_DIR"
else
    unset VLLM_FLASH_ATTN_SRC_DIR
fi
$INSTALL_CMD

echo "Installation complete!"
