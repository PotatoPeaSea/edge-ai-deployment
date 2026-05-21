# MediaPipe Hand — Edge AI Deployment on QCS6490

## Overview

This project deploys the [MediaPipe Hand Detection model](https://github.com/qualcomm/ai-hub-models/tree/v0.54.0/src/qai_hub_models/models/mediapipe_hand) from Qualcomm AI Hub Models onto a **QCS6490-based Qualcomm Linux device** using the **Hexagon v68 DSP/HTP** via the QNN runtime.

The model detects hands in images and outputs:
- Bounding boxes and keypoints (palm detector)
- 21 hand landmark coordinates (landmark detector)

---

## Device

| Field | Value |
|---|---|
| Chipset | QCS6490 |
| OS | Qualcomm Linux 1.6 |
| AI Runtime | QNN (libQnnHtp.so) |
| Compute | Hexagon v68 HTP (DSP, INT8) |

---

## What We Did

### 1. Ran the local PyTorch demo
Verified the model works end-to-end on the host machine using PyTorch:
```bash
python -m qai_hub_models.models.mediapipe_hand.demo --use-default-image
python -m qai_hub_models.models.mediapipe_hand.demo   # webcam mode
```

### 2. Identified the export constraint
The `qai_hub_models` export script blocks float-precision models on QCS6490 because the Hexagon v68 HTP does not support FP16. The solution was to bypass this client-side check by using the low-level `qai_hub` API directly and applying INT8 post-training quantization.

### 3. Compiled both model components for the QCS6490 HTP

The model has two sub-networks, each requiring a 3-step pipeline:

```
TorchScript  →  ONNX (compile)  →  INT8 QDQ ONNX (quantize)  →  QNN context binary (compile)
```

**Key compile options used:**
- `--target_runtime qnn_context_binary` — ahead-of-time compiled binary for HTP
- `--compute_unit all` — allows float I/O boundaries to be handled at compile time
- `--quantize_io` — quantizes model I/O for full INT8 execution on the HTP
- `--force_channel_last_input image` — NHWC memory layout for efficient HTP execution
- `--qnn_options context_enable_graphs=<name>` — required for AoT context binaries

**AI Hub job IDs (for reference):**

| Component | ONNX Compile | Quantize | QNN Compile |
|---|---|---|---|
| hand_detector | jp3qlxrx5 | jpr1momvg | jpe408075 |
| hand_landmark_detector | jgkrwo1y5 | jpvzye87g | jgzvq86zp |

**Profile jobs (on-device latency):**
- hand_detector: `jp8wd6nzp`
- hand_landmark_detector: `jp3qlxyx5`

### 4. Downloaded compiled binaries

```
export_assets/
├── mediapipe_hand_hand_detector.bin          # Palm detector QNN context binary
└── mediapipe_hand_hand_landmark_detector.bin # Landmark detector QNN context binary
```

---

## Files

| File | Description |
|---|---|
| `export_mediapipe_hand_rb3.py` | Full 3-step export pipeline using the low-level `qai_hub` API |
| `demo_rb3.py` | Live webcam demo to run directly on the QCS6490 device |
| `export_assets/*.bin` | Compiled QNN context binaries ready for deployment |

---

## Deployment

### Transfer to device
```powershell
# From Windows
scp export_assets\mediapipe_hand_hand_detector.bin user@<device-ip>:~/mediapipe_hand/
scp export_assets\mediapipe_hand_hand_landmark_detector.bin user@<device-ip>:~/mediapipe_hand/
scp demo_rb3.py user@<device-ip>:~/mediapipe_hand/
```

### Smoke test on device
```bash
python3 -c "import numpy as np; np.random.rand(1,256,256,3).astype('float32').tofile('test_input.raw')"
echo "test_input.raw" > input_list.txt
qnn-net-run \
  --model mediapipe_hand_hand_detector.bin \
  --backend /opt/qcom/aistack/qairt/*/lib/aarch64-oe-linux-gcc11/libQnnHtp.so \
  --input_list input_list.txt \
  --output_dir ./output
```

### Live webcam demo
```bash
pip3 install opencv-python numpy
python3 demo_rb3.py
```

---

---

## Direct QNN Inference via C++ Shim (Phase 2)

Instead of calling `qnn-net-run` as a subprocess (which incurs ~hundreds of ms of process startup overhead per frame), a thin C++ shim wraps the QNN C API directly and exposes a plain C ABI for Python to call via `ctypes`.

### Architecture

```
demo.py  →  qnn_runtime.py (ctypes)  →  libqnn_shim.so  →  libQnnHtp.so  →  Hexagon v68 DSP
```

### Files

| File | Description |
|---|---|
| `qnn_handapp/qnn_shim.h` | C ABI header — opaque handle, tensor info struct, six functions |
| `qnn_handapp/qnn_shim.cpp` | C++17 implementation — dlopen, context binary load, quantized execute |
| `qnn_handapp/qnn_runtime.py` | Python ctypes wrapper — `QnnRuntime`, `QnnModel`, `TensorMeta` |
| `qnn_handapp/demo.py` | End-to-end pipeline: preprocess → palm detector → landmark → annotate |
| `qnn_handapp/build.sh` | Builds `libqnn_shim.so` on-device with g++ |
| `qnn_handapp/VarSetup` | Shell script to `source` before running — sets `ADSP_LIBRARY_PATH` and remounts `/data` exec |

### Build on device

```sh
# Requires QNN headers on the device
export QNN_INCLUDE=/data/local/tmp/snpeexample/include/QNN
cd /data/local/tmp/mediapipe_hand/qnn_handapp
bash build.sh
```

### Run

```sh
source VarSetup
python3 demo.py --video-index 0          # live webcam, press q to quit
python3 demo.py --image hand.jpg         # static image, runs --frames times
```

### Key implementation details

- **`RTLD_GLOBAL`** when dlopening the backend lib so HTP internal symbols resolve across shared libs.
- **`ADSP_LIBRARY_PATH`** must include `.../dsp/lib` (where `libQnnHtpV68Skel.so` lives) **and** `.../aarch64-ubuntu-gcc9.4/lib`. Missing the dsp path gives `Failed to load skel, error: 1002`.
- **`/data` is mounted `noexec`** by default. `VarSetup` calls `mount -o remount,exec /data` — this reverts on reboot.
- **Input normalization**: model I/O is `UFIXED_POINT_8` with `scale=1/255, offset=0`. Feed RGB floats in `[0, 1]`; the shim quantizes to `uint8` on the way in and dequantizes back to `float32` on the way out.
- **Output tensor order is not guaranteed** — always index by name, not position (`{m.name: o for m, o in zip(model.outputs, outs)}`).
- **Landmark coordinates** are in normalized `[0, 1]` space (not pixels). Scale by crop size for drawing.

### Performance (QCS6490, Hexagon v68 HTP, INT8)

| Stage | Time |
|---|---|
| Preprocess (letterbox + normalize) | ~0.3 ms |
| Palm detector | ~3.1 ms |
| Landmark detector | ~1.6 ms |
| **Total** | **~5.0 ms (~200 fps)** |

---

## Next Steps

- [ ] **Proper anchor decoding** — palm detector outputs 2944 SSD anchors; decode and run NMS for accurate bounding boxes
- [ ] **Crop-then-landmark** — crop the detected palm region and feed it to the landmark model (currently feeds the full frame)
- [ ] **Zero-copy buffers** — use `QNN_TENSORMEMTYPE_MEMHANDLE` (ION/DMA buffers) to eliminate the quantization memcpy
- [ ] **Larger calibration set** — INT8 quantization used a single sample; a diverse real-hand calibration set will improve accuracy
- [ ] **systemd service** — wrap the demo in a service for auto-start on boot
