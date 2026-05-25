#!/bin/sh
# One-shot launcher for the EAR demo on device.
# Usage:  ./run_demo.sh [--video-index N] [--ear-threshold 0.20] [...]

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

mount -o remount,exec /data

export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"
export LD_LIBRARY_PATH="/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib:$LD_LIBRARY_PATH"

exec python3 "$DIR/ear_demo.py" "$@"
