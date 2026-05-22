#!/bin/bash
# Deploy facemap_app to device.
# Usage: bash deploy.sh [device_ip]
#
# Requires facemap_3dmm_prepared.dlc to already exist in the sibling
# facemap_3dmm-qnn_dlc-w8a8/ folder.  If it doesn't exist, generate it first:
#
#   docker start qairt_dev
#   docker exec qairt_dev bash -c "
#     /workspace/qairt/2.46.0.260424/bin/x86_64-linux-clang/snpe-dlc-graph-prepare \
#       --input_dlc  /workspace/facemap/facemap_3dmm.dlc \
#       --output_dlc /workspace/facemap/facemap_3dmm_prepared.dlc \
#       --htp_socs   qcs6490
#   "
#   docker cp qairt_dev:/workspace/facemap/facemap_3dmm_prepared.dlc \
#     exportAssets/facemap_3dmm-qnn_dlc-w8a8/facemap_3dmm_prepared.dlc
set -e

DEVICE="${1:-172.30.101.184}"
REMOTE_DIR="/data/local/tmp/facemap"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DLC_DIR="$SCRIPT_DIR/../facemap_3dmm-qnn_dlc-w8a8"
SDK_INCLUDE="$SCRIPT_DIR/../../qairt/2.46.0.260424/include/SNPE"
REMOTE_INCLUDE="/data/local/tmp/snpeexample/include/SNPE"

echo "deploying to root@${DEVICE}:${REMOTE_DIR}"

ssh root@"$DEVICE" "mount -o remount,exec /data 2>/dev/null; mkdir -p $REMOTE_DIR"
scp "$SCRIPT_DIR/demo.py"                  root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/run_demo.sh"              root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/VarSetup"                 root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/snpe_shim.cpp"            root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/snpe_shim.h"              root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/snpe_runtime.py"          root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/build.sh"                 root@"$DEVICE":"$REMOTE_DIR/"
scp "$DLC_DIR/facemap_3dmm.dlc"            root@"$DEVICE":"$REMOTE_DIR/"
scp "$DLC_DIR/facemap_3dmm_prepared.dlc"   root@"$DEVICE":"$REMOTE_DIR/"
# Face detector (Ultra-Light RFB-320) DLCs — runs on DSP for the crop box.
scp "$SCRIPT_DIR/face_det_w8a8.dlc"          root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/face_det_w8a8_prepared.dlc" root@"$DEVICE":"$REMOTE_DIR/"
# 3DMM basis assets for the 68-landmark reconstruction.
scp "$SCRIPT_DIR/meanFace.npy"             root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/shapeBasis.npy"           root@"$DEVICE":"$REMOTE_DIR/"
scp "$SCRIPT_DIR/blendShape.npy"           root@"$DEVICE":"$REMOTE_DIR/"

# SNPE headers (needed once to build the shim on-device).
echo "pushing SNPE headers to $REMOTE_INCLUDE …"
ssh root@"$DEVICE" "mkdir -p $REMOTE_INCLUDE"
scp -r "$SDK_INCLUDE/." root@"$DEVICE":"$REMOTE_INCLUDE/"

# Build the in-process shim on the device.
echo "building libsnpe_shim.so on-device …"
ssh root@"$DEVICE" "cd $REMOTE_DIR && sh build.sh"

echo ""
echo "done — on device run:"
echo "  ssh root@$DEVICE"
echo "  cd $REMOTE_DIR && source VarSetup"
echo "  python3 demo.py --image /data/local/tmp/mediapipe_hand/zidane.jpg --check"
