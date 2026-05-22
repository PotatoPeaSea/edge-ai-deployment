#!/bin/bash
# HOST-SIDE script: push updated Python files to the target device.
# Run this from the host, NOT on device.
#
# Usage:  ./deploy.sh
#
# Uses sshpass if available; otherwise scp/ssh will prompt for the password.

DEVICE="root@172.30.101.184"
PASS="oelinux123"
REMOTE_DIR="/data/local/tmp/mediapipe_hand/qnn_handapp"
DIR="$(cd "$(dirname "$0")" && pwd)"

FILES=(
    "$DIR/demo.py"
    "$DIR/qnn_runtime.py"
    "$DIR/run_demo.sh"
)

if command -v sshpass >/dev/null 2>&1; then
    SCP="sshpass -p $PASS scp"
    SSH="sshpass -p $PASS ssh"
else
    echo "sshpass not found — you will be prompted for the password (oelinux123)."
    SCP="scp"
    SSH="ssh"
fi

echo "Pushing files to $DEVICE:$REMOTE_DIR ..."
$SCP "${FILES[@]}" "$DEVICE:$REMOTE_DIR/"

echo "Setting executable bit on run_demo.sh ..."
$SSH "$DEVICE" "chmod +x $REMOTE_DIR/run_demo.sh"

echo "Done. On device, run:"
echo "  $REMOTE_DIR/run_demo.sh --video-index 0"
