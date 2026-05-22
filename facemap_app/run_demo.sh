#!/bin/sh
# One-shot launcher: sets up the SNPE environment and runs the facemap demo.
# Run this instead of sourcing VarSetup + python3 demo.py separately.
# Must be executed as root (needed for the /data remount).
#
# Usage:
#   ./run_demo.sh                        # webcam live. With a display -> window
#                                        # (press q). Over SSH/no display -> HEADLESS:
#                                        # writes facemap_live.jpg + facemap_live.mp4.
#   ./run_demo.sh --seconds 10           # headless live, auto-stop after 10s
#   ./run_demo.sh --image face.jpg       # single image -> facemap_out.jpg
#   ./run_demo.sh --image obama.jpg --check        # automated PASS/FAIL
#   ./run_demo.sh --image face.jpg --score-thresh 0.6
#
# Headless tip: scp the result off the device to view it, e.g.
#   scp root@<ip>:/data/local/tmp/facemap/facemap_live.jpg .

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SNPE_LIB="/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib"

# /data is noexec by default; remount so .so files can be loaded.
mount -o remount,exec /data 2>/dev/null || true

# libSNPE.so (linked by libsnpe_shim.so) lives here.
export LD_LIBRARY_PATH="$SNPE_LIB:$LD_LIBRARY_PATH"
# FastRPC path for the HTP skel library — must be first entry, semicolon-separated.
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;$SNPE_LIB;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"

exec python3 "$DIR/demo.py" "$@"
