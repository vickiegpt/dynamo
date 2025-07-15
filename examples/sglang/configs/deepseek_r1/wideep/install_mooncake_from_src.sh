#!/bin/bash

# We've been having some trouble with the mooncake installation when we build
# the container. This script is ran before SGL starts up and allows us to use
# the mnnvl capabilites from mooncake main

set -ex

cd /sgl-workspace

pip uninstall mooncake-transfer-engine

git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DUSE_MNNVL=ON
make -j
sudo make install