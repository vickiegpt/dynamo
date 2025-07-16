#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# We've been having some trouble with the mooncake installation when we build
# the container. This script is ran before SGL starts up and allows us to use
# the mnnvl capabilites from mooncake main

cd /sgl-workspace

# Try to set this
export TORCH_CUDA_ARCH_LIST=10.0

echo $LD_LIBRARY_PATH

# Clone & build
# Once Mooncake main branch has fixed
# 1. proper g++ compilation
# 2. solved std::function call issue - we can swap back to ToT
# As of 7/16 10:20AM PST - I've been told its was solved but I have not been able to test it E2E
# So for now we will stay on my side branch
git clone https://github.com/ishandhanani/Mooncake.git
cd Mooncake
git checkout ishan/pr-571-diff-build
bash dependencies.sh -y
mkdir -p build
cd build
cmake .. -DUSE_MNNVL=ON
make -j

make install

chmod +x /usr/local/lib/python3.10/dist-packages/mooncake/nvlink_allocator.so

echo "Mooncake built and installed"
