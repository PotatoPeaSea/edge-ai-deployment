# Edge AI Deployment — MediaPipe Hand on QCS6490

End-to-end deployment of the MediaPipe Hand model on a **Qualcomm QCS6490** device using the **Hexagon v68 HTP DSP** (INT8) via the QNN C API directly — no subprocess overhead, ~200 fps.

---

## What this is

Two MediaPipe sub-networks (palm detector + 21-point landmark detector) are compiled to QNN context binaries via [Qualcomm AI Hub](https://aihub.qualcomm.com) and run on-device through a thin C++ shim that wraps `libQnnHtp.so`. A Python ctypes layer sits on top, and a webcam demo draws live landmark annotations using `cv2.imshow`.

The full step-by-step deployment record (including every error hit and why each decision was made) is in [DEPLOYMENT_JOURNAL.md](DEPLOYMENT_JOURNAL.md).

---

## Hardware & software

| | |
|---|---|
| Device | Qualcomm QCS6490 ODK (`qcs6490-odk`) |
| OS | Ubuntu 20.04.3 LTS, kernel 5.4.219-perf, aarch64 |
| AI compute | Hexagon v68 HTP (DSP, INT8) |
| QNN runtime | QAIRT v2.46.0 — `libQnnHtp.so`, `libQnnHtpV68Skel.so` |
| Python | 3.8.10 |
| Compiler | g++ 9.4.0 |
| Host SDK path | `/data/local/tmp/snpeexample/` |

---

## Repository layout

```
.
├── NOTICE
├── DEPLOYMENT_JOURNAL.md          # full step-by-step deployment record
├── hand.jpg                       # sample test image
└── qnn_handapp/
    ├── qnn_shim.h                 # C ABI — opaque handle + 6 functions
    ├── qnn_shim.cpp               # C++17 — dlopen, binary load, quant execute
    ├── qnn_runtime.py             # Python ctypes wrapper (QnnRuntime, QnnModel)
    ├── demo.py                    # end-to-end webcam / image demo
    ├── build.sh                   # builds libqnn_shim.so on-device
    └── VarSetup                   # source this before running
```

> **Model binaries not included.** The `.bin` files are compiled QNN context binaries produced by Qualcomm AI Hub. Export your own:
> - [MediaPipe Hand — AI Hub](https://aihub.qualcomm.com/models/mediapipe_hand)
> - Target: `QCS6490 (Proxy)`, runtime: `qnn_context_binary`, quantization: INT8, `--quantize_io`

---

## Architecture

```
demo.py
  └─ qnn_runtime.py  (ctypes)
       └─ libqnn_shim.so  (C++17, built on-device)
            ├─ libQnnHtp.so       (RTLD_GLOBAL — QNN HTP backend)
            └─ libQnnSystem.so    (context binary inspection)
                  └─ Hexagon v68 HTP DSP
```

The shim loads a context binary, inspects tensor metadata, and exposes a single `qnn_execute()` call that accepts `float32` buffers — quantizing to `uint8` on the way in and dequantizing back on the way out.

---

## Setup

### 1. Transfer files to device

```sh
scp -r qnn_handapp/ root@<device-ip>:/data/local/tmp/mediapipe_hand/
scp mediapipe_hand_hand_detector.bin root@<device-ip>:/data/local/tmp/mediapipe_hand/
scp mediapipe_hand_hand_landmark_detector.bin root@<device-ip>:/data/local/tmp/mediapipe_hand/
```

### 2. Build the shim on-device

```sh
ssh root@<device-ip>
mount -o remount,exec /data        # /data is noexec by default — reverts on reboot
export QNN_INCLUDE=/data/local/tmp/snpeexample/include/QNN
cd /data/local/tmp/mediapipe_hand/qnn_handapp
bash build.sh
# → libqnn_shim.so
```

### 3. Install OpenCV (if not present)

```sh
apt-get install -y python3-opencv
```

---

## Running

```sh
cd /data/local/tmp/mediapipe_hand/qnn_handapp
source VarSetup          # sets ADSP_LIBRARY_PATH, remounts /data exec

# Live webcam (press q to quit)
python3 demo.py --video-index 0

# Static image (runs --frames times, saves annotated result)
python3 demo.py --image ../hand.jpg --frames 30 --out hand_out.jpg
```

`VarSetup` sets:

```sh
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;\
/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib;\
/vendor/lib/rfsa/adsp;\
/vendor/dsp/cdsp"
```

> **Critical:** `dsp/lib` must come first — that is where `libQnnHtpV68Skel.so` lives. Without it FastRPC fails with `Failed to load skel, error: 1002`.

---

## Performance

Measured on QCS6490 Hexagon v68 HTP, INT8, averaged over 30 runs:

| Stage | Latency |
|---|---|
| Preprocess (letterbox + normalize) | ~0.3 ms |
| Palm detector | ~3.1 ms |
| Landmark detector | ~1.6 ms |
| **Total** | **~5.0 ms (~200 fps)** |

Baseline with `qnn-net-run` subprocess: several hundred ms per frame (process startup dominates).

---

## Key gotchas

**`/data` is `noexec`** — both the `.so` shim and the QNN backend libs fail to load until remounted. Does not persist across reboots; `VarSetup` handles it.

**`RTLD_GLOBAL` on the backend lib** — `libQnnHtp.so` must be dlopened with `RTLD_GLOBAL` so its internal symbols are visible to the skel library loaded by FastRPC.

**Output tensor order is not guaranteed** — always look up tensors by name:
```python
out = {m.name: o for m, o in zip(model.outputs, results)}
landmarks = out["landmarks"].reshape(21, 3)
```

**Landmark coordinates are normalized `[0, 1]`** — not pixels. Multiply by crop size when drawing; do not divide.

**Deep-copy tensor names before freeing the system context** — the QNN system context binary info struct owns the name strings; free it too early and you get dangling pointers in the shim.

---

## Remaining work

- [ ] Proper SSD anchor decoding for the palm detector (2944 anchors, 18 channels each)
- [ ] Crop detected palm region before feeding landmark detector
- [ ] NMS for multi-hand scenes
- [ ] Zero-copy inference with `QNN_TENSORMEMTYPE_MEMHANDLE` (ION/DMA buffers)
- [ ] Larger INT8 calibration set for better quantized accuracy
- [ ] systemd service for auto-start on boot

---

## Attribution

See [NOTICE](NOTICE) for third-party attributions (Qualcomm QNN SDK, Google MediaPipe, Qualcomm AI Hub).
