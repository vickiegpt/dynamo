#!/bin/bash

# We've been having some trouble with the mooncake installation when we build
# the container. This script is ran before SGL starts up and allows us to use
# the mnnvl capabilites from mooncake main
#
# Usage: ./install_mooncake.sh <dynamo|sglang>
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <dynamo|sglang>"
  exit 1
fi

MODE="$1"
case "$MODE" in
  dynamo)
    SUDO=""
    ;;
  sglang)
    SUDO="sudo"
    ;;
  *)
    echo "Error: invalid mode '$MODE'. Use 'dynamo' or 'sglang'."
    exit 1
    ;;
esac

cd /sgl-workspace

# Clean up previous build
$SUDO rm -rf Mooncake/

# Uninstall any existing package
pip uninstall -y mooncake-transfer-engine

# Clone & build
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh

mkdir -p build
cd build
cmake .. -DUSE_MNNVL=ON
make -j

# Install (with sudo if in sglang mode)
$SUDO make install

echo "Mooncake built and installed in '$MODE' mode."