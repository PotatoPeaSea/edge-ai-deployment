#!/bin/sh
# One-shot launcher: sets up the QNN environment and runs the hand demo.
# Usage:  ./run_demo.sh [--video-index N] [--image path] [...]
#
# Run this instead of sourcing VarSetup + python3 demo.py separately.
# Must be executed as root (needed for the remount).

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# /data is noexec by default; remount so .so files can be dlopen'd.
mount -o remount,exec /data

# FastRPC path for libQnnHtpV68Skel.so — must be first entry.
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"

exec python3 "$DIR/demo.py" "$@"
