#!/bin/bash

# We've been having some trouble with the mooncake installation when we build
# the container. This script is ran before SGL starts up and allows us to use
# the mnnvl capabilites from mooncake main

cd /sgl-workspace

# Try to set this
export TORCH_CUDA_ARCH_LIST=10.0

echo $LD_LIBRARY_PATH

# Uninstall any existing package
#pip install --break-system-packages mooncake-transfer-engine

# Clone & build
git clone https://github.com/ishandhanani/Mooncake.git
cd Mooncake
git checkout ishan/manual-nvl-installation
bash dependencies.sh -y
mkdir -p build
cd build
cmake .. -DUSE_MNNVL=ON
make -j

make install

chmod +x /usr/local/lib/python3.10/dist-packages/mooncake/nvlink_allocator.so

echo "Mooncake built and installed"
