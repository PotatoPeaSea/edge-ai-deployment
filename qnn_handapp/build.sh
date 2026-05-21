#!/bin/sh
# Build libqnn_shim.so on-device.
# Expects QNN headers at $QNN_INCLUDE (e.g. /data/local/tmp/snpeexample/include/QNN).
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
QNN_INCLUDE="${QNN_INCLUDE:-/data/local/tmp/snpeexample/include/QNN}"

if [ ! -d "$QNN_INCLUDE" ]; then
    echo "QNN_INCLUDE=$QNN_INCLUDE does not exist."
    echo "Set it to the path containing QnnInterface.h."
    exit 1
fi

g++ -O2 -fPIC -shared -std=c++17 \
    -I"$QNN_INCLUDE" \
    -Wno-unused-parameter -Wno-unused-variable \
    "$DIR/qnn_shim.cpp" \
    -o "$DIR/libqnn_shim.so" \
    -ldl

echo "built: $DIR/libqnn_shim.so"
file "$DIR/libqnn_shim.so" || true
