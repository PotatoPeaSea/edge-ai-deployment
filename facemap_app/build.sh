#!/bin/sh
# Build libsnpe_shim.so on-device (aarch64 Ubuntu, g++ 9.4).
# Expects SNPE headers at $SNPE_INCLUDE and libSNPE.so under $SNPE_LIB.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SNPE_INCLUDE="${SNPE_INCLUDE:-/data/local/tmp/snpeexample/include/SNPE}"
SNPE_LIB="${SNPE_LIB:-/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib}"

if [ ! -d "$SNPE_INCLUDE" ]; then
    echo "SNPE_INCLUDE=$SNPE_INCLUDE does not exist."
    echo "Push the SDK's include/SNPE there first."
    exit 1
fi

g++ -O2 -fPIC -shared -std=c++17 \
    -I"$SNPE_INCLUDE" \
    -Wno-unused-parameter -Wno-unused-variable \
    "$DIR/snpe_shim.cpp" \
    -o "$DIR/libsnpe_shim.so" \
    -L"$SNPE_LIB" -lSNPE

echo "built: $DIR/libsnpe_shim.so"
file "$DIR/libsnpe_shim.so" || true
