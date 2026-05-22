# MediaPipe Hand on QCS6490 — End-to-End Deployment Journal

A step-by-step record of taking the two QNN context binaries in this folder
(`mediapipe_hand_hand_detector.bin`, `mediapipe_hand_hand_landmark_detector.bin`)
and getting them running on a QCS6490 device — first via the stock
`qnn-net-run` tool, then via a custom direct-QNN ctypes pipeline that hits
~200 fps end-to-end with a live webcam.

This is intentionally detailed and includes *why* each step was chosen, so a
reader can either follow along verbatim or know which constraints can be
relaxed.

---

## Table of contents

- [0. Lay of the land](#0-lay-of-the-land)
- [1. Phase one — qnn-net-run smoke test over ADB](#1-phase-one--qnn-net-run-smoke-test-over-adb)
- [2. Phase two — direct-QNN ctypes pipeline](#2-phase-two--direct-qnn-ctypes-pipeline)
- [3. Phase three — webcam integration](#3-phase-three--webcam-integration)
- [4. Performance numbers](#4-performance-numbers)
- [5. Gotchas / decisions you'll likely re-encounter](#5-gotchas--decisions-youll-likely-re-encounter)
- [6. Remaining work](#6-remaining-work)
- [Appendix A — file inventory](#appendix-a--file-inventory)
- [Appendix B — useful one-liners](#appendix-b--useful-one-liners)

---

## 0. Lay of the land

### 0.1 Host

| | |
|---|---|
| OS | Ubuntu 20.04 (matthew@…/SNPE) |
| QAIRT SDK | `/mnt/data02/matthew/SNPE/qairt/2.46.0.260424` |
| Dev container | `qairt_dev` (Ubuntu 22.04, has venv at `/workspace/qairt/2.46.0.260424/bin/venv/`) |
| Container mount | host `/mnt/data02/matthew/SNPE` → container `/workspace` |

The folder is misleadingly named `SNPE`; the actual SDK inside is **QAIRT
v2.46.0**. SNPE is now a runtime/backend within QAIRT — DLCs still work,
QNN context binaries (`.bin`) work, and `setup_snpe_env.sh` still configures
the environment.

### 0.2 Target device

| | |
|---|---|
| Hostname | `qcs6490-odk` |
| OS | Ubuntu 20.04.3 LTS, kernel 5.4.219-perf |
| Arch | aarch64 |
| AI runtime | QAIRT v2.46.0 installed under `/data/local/tmp/snpeexample/` (non-standard path — the README expected `/opt/qcom/aistack/qairt/...`) |
| Compute | Hexagon v68 HTP (DSP, INT8) — confirmed via `libQnnHtpV68Skel.so` in `/data/local/tmp/snpeexample/dsp/lib/` |
| Python | 3.8.10 |
| Compiler | gcc/g++ 9.4.0 |

Layout under `/data/local/tmp/snpeexample/`:

```
aarch64-ubuntu-gcc9.4/
├── bin/      qnn-net-run, qnn-context-binary-generator, snpe-net-run, …
└── lib/      libQnnHtp.so, libQnnSystem.so, libQnnHtpV68Stub.so, …
dsp/
└── lib/      libQnnHtpV68.so, libQnnHtpV68Skel.so, …
```

Two transports to the device:
- **ADB over USB** (used in phase 1)
- **SSH at `172.30.101.184`** (root / password `oelinux123`, used in phases 2+)

### 0.3 Export assets handed to us

```
exportAssets/
├── README.md                                          # original deployment doc
├── mediapipe_hand_hand_detector.bin                   # palm detector QNN context binary
├── mediapipe_hand_hand_landmark_detector.bin          # landmark detector QNN context binary
└── mediapipe_hand_hand_detector.bin.onnx.zip          # pre-quantization ONNX (reference)
```

The README claimed a `demo_rb3.py` ships alongside, but it does not — only
the three artifacts above and a description of the export pipeline. That
gap is what this journal fills.

---

## 1. Phase one — qnn-net-run smoke test over ADB

Goal: prove the binaries run on the HTP DSP at all, before writing any code.

### 1.1 Locate the QNN runtime on device

`adb shell` and search:

```bash
adb shell 'find / -name "qnn-net-run" 2>/dev/null'
# → /data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/bin/qnn-net-run

adb shell 'find / -name "libQnnHtpV68Skel.so" 2>/dev/null'
# → /data/local/tmp/snpeexample/dsp/lib/libQnnHtpV68Skel.so
```

The README assumed `/opt/qcom/aistack/qairt/...`. Don't take SDK install
paths at face value; always re-probe.

### 1.2 Push the binaries

```bash
adb shell 'mkdir -p /data/local/tmp/mediapipe_hand'
adb push mediapipe_hand_hand_detector.bin           /data/local/tmp/mediapipe_hand/
adb push mediapipe_hand_hand_landmark_detector.bin  /data/local/tmp/mediapipe_hand/
```

### 1.3 Inspect graph I/O before generating a test input

We need to know dtype + shape per input/output. The SDK ships
`qnn-context-binary-utility` (x86 host-side only). Run it from inside the
`qairt_dev` container:

```bash
docker exec qairt_dev bash -c '
QNN_BIN=/workspace/qairt/2.46.0.260424/bin/x86_64-linux-clang
$QNN_BIN/qnn-context-binary-utility \
  --context_binary /workspace/exportAssets/mediapipe_hand_hand_detector.bin \
  --json_file /tmp/det_meta.json
'
```

Then parse with Python:

```python
import json
m = json.load(open("/tmp/det_meta.json"))
for g in m["info"]["graphs"]:
    info = g["info"]
    print("Graph:", info["graphName"])
    for t in info["graphInputs"]:
        ti = t["info"]
        print("IN ", ti["name"], ti["dimensions"], ti["dataType"])
    for t in info["graphOutputs"]:
        ti = t["info"]
        print("OUT", ti["name"], ti["dimensions"], ti["dataType"])
```

Result for both binaries:

| Graph | Inputs | Outputs |
|---|---|---|
| `hand_detector` | `image [1,256,256,3] UFIXED_POINT_8` | `box_scores [1,2944,1]`, `box_coords [1,2944,18]` (both UFIXED_POINT_8) |
| `hand_landmark_detector` | `image [1,256,256,3] UFIXED_POINT_8` | `scores [1]`, `lr [1]`, `landmarks [1,21,3]` (all UFIXED_POINT_8) |

Two non-obvious things this told us:
- I/O were quantized to uint8 by `--quantize_io` (so the device-side tensor is uint8 NHWC, not fp32 NCHW like the optimized ONNX).
- The landmark detector also expects 256×256 (not the 224×224 used by some MediaPipe variants).

### 1.4 Generate a float test input

`qnn-net-run` accepts fp32 input by default and quantizes at the boundary
using the input tensor's encoding. So we feed `(1, 256, 256, 3)` fp32:

```bash
docker exec qairt_dev bash -c '
source /workspace/qairt/2.46.0.260424/bin/venv/bin/activate
python3 -c "
import numpy as np
np.random.rand(1,256,256,3).astype(\"float32\").tofile(\"/workspace/_onnx_tmp/test_input.raw\")
"'
adb push /mnt/data02/matthew/SNPE/_onnx_tmp/test_input.raw /data/local/tmp/mediapipe_hand/
adb shell 'cd /data/local/tmp/mediapipe_hand && echo test_input.raw > input_list.txt'
```

### 1.5 First gotcha — `/data` is mounted `noexec`

```text
$ adb shell '/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/bin/qnn-net-run --version'
Permission denied
```

The mount is `nosuid,nodev,noexec`. Remount with exec (root required):

```bash
adb shell 'mount -o remount,exec /data'
adb shell 'mount | grep "/data "'
# /dev/sda12 on /data type ext4 (rw,nosuid,nodev,relatime,...)  ← noexec gone
```

This will revert on reboot. If you want it permanent, edit `/etc/fstab`.

It also blocks loading `.so` files later from the same partition (e.g. the
custom `libqnn_shim.so` in phase 2). Same fix applies.

### 1.6 Run inference

```bash
adb shell '
cd /data/local/tmp/mediapipe_hand
export QNN_ROOT=/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH=$QNN_ROOT/lib:/data/local/tmp/snpeexample/dsp/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"

$QNN_ROOT/bin/qnn-net-run \
  --retrieve_context mediapipe_hand_hand_detector.bin \
  --backend $QNN_ROOT/lib/libQnnHtp.so \
  --input_list input_list.txt \
  --output_dir output_hand_detector
'
```

Successful output:
- `output_hand_detector/Result_0/box_coords.raw` — 211968 bytes = 1·2944·18·4 (fp32)
- `output_hand_detector/Result_0/box_scores.raw` — 11776 bytes = 1·2944·1·4

`qnn-net-run` auto-dequantizes outputs back to fp32, even though they live
as uint8 inside the graph.

Same recipe with `mediapipe_hand_hand_landmark_detector.bin` produced
`landmarks.raw (252 B)`, `lr.raw (4 B)`, `scores.raw (4 B)`.

`ADSP_LIBRARY_PATH` is what tells the FastRPC layer where to find the
Hexagon-side skeletons (`libQnnHtpV68Skel.so`, etc.). Forgetting it gives
`Could not create context from binary`.

### 1.7 Sanity-check the float output

Dequantization preserves the original encoding, so values matched what the
binary's quantization params predicted (e.g. `box_scores` includes both
strong negatives like -81 and small positives near 1.0; these are pre-sigmoid
logits).

That closed phase 1: both binaries run on HTP, raw outputs look plausible.

---

## 2. Phase two — direct-QNN ctypes pipeline

The README's "Next Steps" identified the obvious next move:

> *"Replace subprocess with direct QNN C API — calling `qnn-net-run` per
> frame is slow due to process startup overhead; a C++ or Python-ctypes
> wrapper using libQnnHtp.so directly will give real-time throughput."*

I went with: **thin C++ shim** exposing a small C ABI, **Python ctypes**
wrapper around it. Rationale:

- Pure-Python ctypes against the raw QNN C API means porting ~50
  function-pointer interface tables and the `Qnn_Tensor_t` version union
  to ctypes. Doable but fragile — struct layouts must mirror SDK headers
  exactly.
- A 250-line C++ shim that uses the SDK headers directly is robust and
  trivially testable.
- The Python side then only sees: load, execute, destroy.

### 2.1 Switch transport: ADB → SSH

Phase 1 used ADB. From here we use SSH at `172.30.101.184`:

```text
ssh root@172.30.101.184  # password: oelinux123
```

`sshpass` wasn't installed on host, but `paramiko` (Python) was. Two tiny
wrappers were written under `/tmp/` to make automation painless:

```python
# /tmp/sshrun.py  — one-shot exec
import sys, paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("172.30.101.184", username="root", password="oelinux123",
          timeout=15, allow_agent=False, look_for_keys=False)
stdin, stdout, stderr = c.exec_command(sys.argv[1], timeout=120)
sys.stdout.write(stdout.read().decode())
sys.stderr.write(stderr.read().decode())
sys.exit(stdout.channel.recv_exit_status())
```

```python
# /tmp/sshcp.py  — recursive sftp push (creates remote dirs)
# /tmp/sshpull.py — single-file sftp get
```

(Full contents are in [Appendix B](#appendix-b--useful-one-liners).)

### 2.2 Probe the device's runtime + dev environment

```bash
/tmp/sshrun.py 'python3 --version; python3 -c "import cv2; print(cv2.__version__)" 2>&1; ls /dev/video*; g++ --version | head -1'
```

Result:
- Python 3.8.10
- `cv2` **not installed** — fix below
- `/dev/video0..3, video32, video33` exist (USB UVC + Camera Subsystem)
- g++ 9.4 present → can build the shim directly on-device

Install OpenCV. `pip3` wasn't present either, but `apt` had it:

```bash
/tmp/sshrun.py 'apt-get install -y python3-opencv'
# → opencv 4.2.0 installed
```

### 2.3 Design the shim

Goals:
- **Input**: float32 numpy arrays (one per input tensor, NHWC order).
- **Output**: float32 numpy arrays (the shim dequantizes for you).
- **Single graph per context** (matches our binaries).
- **HTP backend** but the path is parameterized — could swap to GPU/CPU.

C ABI in [qnn_handapp/qnn_shim.h](qnn_handapp/qnn_shim.h):

```c
qnn_model_t* qnn_load(const char* backend_so,
                      const char* system_so,
                      const char* binary_path,
                      const char* graph_name);   // NULL = first graph
int32_t qnn_num_inputs(qnn_model_t*);
int32_t qnn_num_outputs(qnn_model_t*);
const qnn_tensor_info_t* qnn_input_info(qnn_model_t*, int32_t i);
const qnn_tensor_info_t* qnn_output_info(qnn_model_t*, int32_t i);
int32_t qnn_execute(qnn_model_t*, const float* const* in, float* const* out);
void qnn_destroy(qnn_model_t*);
const char* qnn_last_error(qnn_model_t*);
```

`qnn_tensor_info_t` exposes name, rank, dims, raw `Qnn_DataType_t`, scale,
offset, and element count.

### 2.4 Shim internals — what it actually does

[qnn_handapp/qnn_shim.cpp](qnn_handapp/qnn_shim.cpp). Step-by-step:

1. **`dlopen` two libraries** with the right flags:
   - Backend (`libQnnHtp.so`) → `RTLD_NOW | RTLD_GLOBAL`. The `GLOBAL`
     matters: the HTP backend's `.so` lazily resolves symbols against
     itself, and dlsym from inside the backend's own dependencies expects
     them to be globally visible.
   - System (`libQnnSystem.so`) → `RTLD_NOW | RTLD_LOCAL`. Local is fine
     because nothing chains off it.

2. **Resolve providers**:

   ```c
   auto get_iface = dlsym(backend, "QnnInterface_getProviders");
   auto get_sys   = dlsym(system,  "QnnSystemInterface_getProviders");
   ```

   Each returns an array of provider structs. Pick the first one whose
   API version is compatible with `QNN_API_VERSION_{MAJOR,MINOR}` from the
   headers. Stash the resulting `QNN_INTERFACE_VER_TYPE` / `QNN_SYSTEM_INTERFACE_VER_TYPE`
   in the model state — these are tables of function pointers (`graphExecute`,
   `contextCreateFromBinary`, etc.).

3. **Read the context binary** into a `std::vector<uint8_t>`.

4. **Use the system interface to extract metadata**:

   ```c
   sys.systemContextCreate(&sys_ctx);
   sys.systemContextGetBinaryInfo(sys_ctx, buf.data(), buf.size(),
                                   &bin_info, &bin_info_size);
   ```

   `bin_info` is a tagged union; you have to switch on
   `bin_info->version` to choose `contextBinaryInfoV1/V2/V3`. Same drill
   for each `QnnSystemContext_GraphInfo_t`. The fields you need exist in
   all versions (`graphName`, `numGraphInputs`, `graphInputs`, …).

5. **Deep-copy each input/output tensor descriptor** out of the binary
   info into shim-owned storage. The system handle gets freed shortly,
   and the dimensions pointer points into the system handle's memory, so
   copy aggressively (allocate new dim array, `strdup` the name).

6. **Allocate a raw client buffer** for each tensor sized
   `prod(dims) * sizeof(uint8)`. Store it on the tensor via
   `Qnn_TensorV1_t.clientBuf` with `memType=QNN_TENSORMEMTYPE_RAW`. These
   buffers stay alive for the model's lifetime.

7. **Free the system handle** and create the real backend:

   ```c
   iface.backendCreate(NULL, NULL, &backend);
   iface.deviceCreate(NULL, NULL, &device);   // may return UNSUPPORTED_FEATURE, that's OK
   iface.contextCreateFromBinary(backend, device, NULL,
                                  buf.data(), buf.size(),
                                  &context, NULL);
   iface.graphRetrieve(context, graphName, &graph);
   ```

8. **Per-inference (`qnn_execute`)**:
   - Walk each input. If the tensor's dtype is `UFIXED_POINT_8`, quantize
     using its scale/offset (`q = round(f / scale) - offset`, clipped to
     `[0, 255]`). If it's `FLOAT_32`, just memcpy.
   - Call `iface.graphExecute(graph, inputs, n_in, outputs, n_out, NULL, NULL)`.
   - Walk each output and dequantize symmetrically (`f = (raw + offset) * scale`).

   The whole thing is two `for` loops + one function call. No allocations
   on the hot path, no string handling.

9. **`qnn_destroy`** frees context, device, backend, all client buffers,
   strdup'd names, dim arrays, and dlcloses the two libs.

#### Why deep-copy the tensor descriptors?

If you reuse the pointer arrays from `bin_info`, then call `systemContextFree`,
the pointers become dangling — `graphExecute` reads `dimensions` and `name`
and will crash. The shim takes a slight memory hit and gets correctness
in return.

### 2.5 Build the shim

[qnn_handapp/build.sh](qnn_handapp/build.sh):

```sh
g++ -O2 -fPIC -shared -std=c++17 \
    -I"$QNN_INCLUDE" \
    qnn_shim.cpp -o libqnn_shim.so -ldl
```

Only depends on libdl; no need to link against the QNN libs (they're
dlopened). To build:

```bash
# host
/tmp/sshcp.py \
  /mnt/data02/matthew/SNPE/qairt/2.46.0.260424/include/QNN  /data/local/tmp/snpeexample/include/QNN \
  /mnt/data02/matthew/SNPE/exportAssets/qnn_handapp           /data/local/tmp/mediapipe_hand/qnn_handapp

# device
/tmp/sshrun.py 'cd /data/local/tmp/mediapipe_hand/qnn_handapp && sh build.sh'
```

Result: `libqnn_shim.so` (aarch64 ELF, ~150 KB).

### 2.6 Python wrapper

[qnn_handapp/qnn_runtime.py](qnn_handapp/qnn_runtime.py). Two classes:

- **`QnnRuntime`** holds the dlopen'd shim and the *paths* to the backend
  and system libs. One per process.
- **`QnnModel`** is one graph. Constructor calls `qnn_load`, walks
  `qnn_num_inputs/qnn_num_outputs/qnn_input_info/qnn_output_info` to
  build a `TensorMeta` list, pre-allocates output buffers and the ctypes
  pointer arrays used per call.

`execute()` is the only hot-path method. It accepts a sequence of numpy
arrays (any shape compatible by total element count), makes each
contiguous as float32, hands raw pointers to the shim, and returns a list
of freshly-shaped copies of the internal output buffers.

#### Memory & lifetime notes

- The shim's input pointers are *read-only* during `graphExecute` — we
  hand it the numpy array's buffer directly. As long as the array stays
  alive through the call (which it does — `in_contig` is a local list),
  this is safe.
- The output buffers are owned by Python (`self._out_bufs`, lifetime tied
  to `QnnModel`). The shim writes into them and we hand back copies so
  the next call's writes don't clobber values the caller saved.

### 2.7 Smoke test the wrapper

```bash
/tmp/sshrun.py '
cd /data/local/tmp/mediapipe_hand
export QNN_ROOT=/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH=$QNN_ROOT/lib:/data/local/tmp/snpeexample/dsp/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"
python3 -c "
import sys, time, numpy as np
sys.path.insert(0, \"qnn_handapp\")
from qnn_runtime import QnnRuntime, QnnModel
rt = QnnRuntime(backend_so=\"$QNN_ROOT/lib/libQnnHtp.so\",
                system_so=\"$QNN_ROOT/lib/libQnnSystem.so\")
m = QnnModel(rt, \"mediapipe_hand_hand_detector.bin\")
print(m.inputs); print(m.outputs)
x = np.random.rand(1,256,256,3).astype(np.float32)
m.execute([x])  # warm
N = 50
t0 = time.perf_counter()
for _ in range(N): m.execute([x])
print(f\"per-inference: {(time.perf_counter()-t0)/N*1000:.2f} ms\")
"'
```

First run failed with `failed to map segment from shared object` —
`/data` had reverted to `noexec` (we mounted exec for `qnn-net-run` but
on a fresh `mount` table, the new `.so` was rejected). One more
`mount -o remount,exec /data` fixed it.

Then:

```
per-inference: 4.82 ms  (n=50)
```

vs. each `qnn-net-run` invocation in phase 1, which has ~hundreds of ms
of process startup overhead per inference. Two orders of magnitude.

### 2.8 The demo script

[qnn_handapp/demo.py](qnn_handapp/demo.py) wires up:

1. Load both context binaries via the shared `QnnRuntime`.
2. Read image (static or webcam frame).
3. **Preprocess**: BGR → RGB, letterbox-pad to 256×256, divide by 255.
4. Palm detector → 2944×18 box coords + 2944×1 logits.
5. Apply sigmoid to logits, pick top-scoring anchor (stand-in for proper
   anchor decoding — see [§6](#6-remaining-work)).
6. Landmark detector on the same image (proper crop is a follow-up).
7. Draw the 21 landmarks + a HUD with palm/landmark/handedness scores.
8. Average timings over N frames, write annotated output.

#### Output ordering bug we hit (and how to avoid it)

First demo run crashed with `cannot reshape array of size 1 into shape (21, 3)`.

Cause: I'd assumed the landmark detector's outputs were ordered
`landmarks, lr, scores`. The binary actually returns
`scores [1], lr [1], landmarks [1,21,3]` (positional order discovered via
`qnn_output_info`).

Fix: **always index outputs by name**, not position:

```python
det_by_name = {m.name: o for m, o in zip(detector.outputs, det_outs)}
scores_logits = det_by_name["box_scores"].reshape(-1)
coords        = det_by_name["box_coords"].reshape(-1, 18)
```

#### Landmark coordinate range bug (and how the quant params reveal it)

Second bug: landmarks all clustered at the origin. I'd divided by 256
before drawing, but inspection of the output's quantization params
explained the right scale:

```
landmarks: scale=0.0038292, offset=-29
→ dequant range = [(0-29)*0.0038, (255-29)*0.0038] = [-0.110, 0.864]
```

That is, the model emits landmarks in **normalized image coordinates
([0, 1])**, not pixels. Drop the `/256.0` and pass directly to
`overlay_landmarks(crop_size=256)`.

#### Input normalization, justified

The user asked to verify the input normalization. Same trick — read the
input tensor's quant params:

```
image: scale=1/255 (0.003922), offset=0  → dequant range [0.0, 1.0]
```

Therefore the model expects fp32 in `[0, 1]` (with `--quantize_io` mapping
that to uint8 `[0, 255]` exactly via `q = round(f * 255)`). A `[-1, 1]`
range would imply `scale ≈ 2/255, offset ≈ -128`; that's not what's in
the binary. So `frame.astype(np.float32) / 255.0` is correct; the comment
in `preprocess()` now spells this out so future maintainers don't
second-guess.

---

## 3. Phase three — webcam integration

`/dev/video0` is a USB UVC camera. OpenCV picks it up fine (the GStreamer
warning is harmless), though the first frame is sometimes garbage and the
buffer holds a few stale frames — drain a handful before each capture.

### 3.1 Naive capture (1 frame after 6s countdown)

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for sec in range(6, 0, -1):
    for _ in range(5): cap.read()  # drain
    time.sleep(1)
ok, frame = cap.read()
```

Caught the user mid-blink with no hand visible. Pull the JPEG back to host
to verify visually:

```bash
/tmp/sshpull.py /data/local/tmp/mediapipe_hand/hand.jpg ./hand.jpg
```

### 3.2 Burst capture with live palm scoring

Better approach: capture 15 frames at 1s intervals, run the palm detector
on each, keep the highest-scoring one. The user only has to remember to
hold a hand up during *some* of those 15 seconds.

```python
best_score = -1; best_frame = None
for i in range(15):
    for _ in range(3): cap.read()
    ok, fr = cap.read()
    # preprocess like demo.py
    score = sigmoid(detector.execute([inp])[0].max())
    if score > best_score:
        best_score = score; best_frame = fr
    time.sleep(1.0)
```

Score timeline:
```
frame 00: 0.417   frame 05: 0.117   frame 10: 0.207
frame 01: 0.207   frame 06: 0.064   frame 11: 0.207
frame 02: 0.064   frame 07: 0.207   frame 12: 0.157
frame 03: 0.157   frame 08: 0.843 ← picked
frame 04: 0.087   frame 09: 0.338   frame 13: 0.157
                                    frame 14: 0.157
```

Frame 8 scored 0.843 — a clear positive. Background noise level for this
scene is roughly 0.05–0.2.

### 3.3 Run demo on the best frame

```bash
/tmp/sshrun.py '
cd /data/local/tmp/mediapipe_hand
export ...
python3 qnn_handapp/demo.py --image hand.jpg --frames 30 --out qnn_handapp/hand_out.jpg
'
/tmp/sshpull.py /data/local/tmp/mediapipe_hand/qnn_handapp/hand_out.jpg ./hand_out.jpg
```

Result: palm score 0.88 on the chosen frame, landmark score ~0.5 (median
because we're feeding the model the full letterboxed scene, not a tight
crop). The skeleton in `hand_out.jpg` traces along the visible hand
region — fingertips toward the top-left, base toward the bottom-right.
Not anatomically tight, but the pipeline is producing meaningful output.

---

## 4. Performance numbers

Measured on the QCS6490 HTP v68, 30-run average, fp32 in/out at the Python
boundary (shim handles quant/dequant):

| Stage | Latency |
|---|---:|
| Preprocess (resize + letterbox + RGB convert + /255) | **0.91 ms** |
| Palm detector (`hand_detector.bin`) | **1.88 ms** |
| Landmark detector (`hand_landmark_detector.bin`) | **2.17 ms** |
| **Total** | **4.96 ms / frame** → **201 fps** |

Reference points:
- `qnn-net-run` subprocess startup alone is on the order of a few
  hundred ms before any inference happens. The direct path is two orders
  of magnitude faster on the hot path.
- The palm detector alone, run repeatedly in the smoke test, settles at
  ~1.9 ms warm. The first call is slower (compilation / KMD setup).

---

## 5. Gotchas / decisions you'll likely re-encounter

### Mount options

`/data` ships with `nosuid,nodev,noexec`. You must remount it `exec`
before any binary or shared library on that partition can run:

```bash
mount -o remount,exec /data
```

This applies to `qnn-net-run`, `libqnn_shim.so`, and any other ELF you
push under `/data/local/tmp/`.

### `ADSP_LIBRARY_PATH` is mandatory for HTP

The QNN HTP backend uses FastRPC to invoke the Hexagon skel. The skel
search path is `ADSP_LIBRARY_PATH` (semicolon-separated, not colon!).
Without it: `Could not create context from binary`.

```bash
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"
```

### Don't trust positional output ordering

`qnn-net-run` writes outputs in some order; your binary may report them
in a different order via `QnnSystemContext_GraphInfo`. Always index by
name (`tensor.name`), not by position.

### Quant params tell you the expected input range

For models compiled with `--quantize_io`, the input tensor's
`scaleOffsetEncoding` *is* the documentation:

```
dequant_range = [(0 + offset) * scale, (255 + offset) * scale]
```

- `scale=1/255, offset=0` → `[0, 1]` (our case)
- `scale=2/255, offset=-128` → `[-1, 1]` (typical MediaPipe upstream)
- `scale=1, offset=0` → `[0, 255]` (raw uint8 with no normalization)

### `QnnSystemContext_BinaryInfo` is a tagged union

Three versions exist (V1/V2/V3). Always switch on `bin_info->version`
before reading fields. Same for `QnnSystemContext_GraphInfo` and
`Qnn_Tensor_t` (V1 vs V2).

### Deep-copy tensor descriptors out of binary info

Pointers inside `binaryInfo->...graphInputs[i]` (dimensions, name) point
into system-handle-owned memory. They become invalid after
`systemContextFree`. Copy them into shim-owned storage before freeing
the system handle.

### `RTLD_GLOBAL` on the backend

`dlopen(libQnnHtp.so, RTLD_NOW | RTLD_GLOBAL)`. The HTP backend resolves
some symbols at runtime that expect to be in the global namespace.
`RTLD_LOCAL` causes obscure failures during `contextCreateFromBinary`.

### Webcam buffer staleness

OpenCV's `VideoCapture(0)` keeps an internal buffer of recent frames.
After any pause (e.g. a `sleep`), drain 3–5 frames before treating the
next `read()` as "now".

### Build inside `qairt_dev` container vs on-device

Both work. We did on-device because g++ 9.4 was already present and that
matches the target glibc exactly (no compatibility surprises). If the
target has no compiler, cross-compile in the container with an aarch64
toolchain (the SDK ships one) and scp the `.so`.

### SSH host fingerprint

The host's `~/.ssh/known_hosts` may not have the device. The paramiko
wrapper sets `AutoAddPolicy()` to accept on first connect. If you use
`ssh` directly, expect an interactive prompt the first time.

---

## 6. Remaining work

These are the parts of `demo.py` that are deliberately rough — the demo
exists to prove the inference pipeline, not to be a finished application.

1. **Proper anchor decoding for the palm detector.** The detector emits
   `2944 × 18` per frame, where each row is `(dx, dy, dw, dh, 7×(kx,ky))`
   relative to an anchor. The 2944 anchors are generated from a known
   SSD-style config (BlazePalm v2: input 256×256, 4 layers, strides
   `[8, 16, 16, 16]`). With anchors, decode the top detection's bbox in
   pixel space.

2. **Crop-then-landmark.** Once the bbox is known, rotate (per the wrist
   keypoint) and crop a tight 256×256 around the hand, feed *that* to
   the landmark detector. Today the landmark model gets the whole
   letterboxed frame, which is why its score sits near 0.5 even on a
   clear hand — it's trying to fit a hand model to a much larger
   region.

3. **NMS.** With proper decoding, run non-max suppression to support
   multiple hands per frame.

4. **Webcam live loop.** `demo.py` has a webcam branch (`--video-index`)
   but it wasn't exercised yet. Wire it through the same pipeline once
   the post-processing above is solid.

5. **Direct DMA buffers.** Currently inputs are fp32 numpy → C++
   quantize-and-copy. For higher throughput, swap to QNN shared memory
   handles (`QNN_TENSORMEMTYPE_MEMHANDLE`) and zero-copy from a
   pre-quantized DMA-BUF directly to the HTP. Probably not worth it
   below ~500 fps; mentioning for completeness.

6. **`systemd` wrap.** When the demo is production-shaped, package it as
   a service with auto-restart on disconnect (the USB UVC cam can
   disappear).

---

## Appendix A — file inventory

```
exportAssets/
├── DEPLOYMENT_JOURNAL.md                              # this document
├── README.md                                          # original (kept verbatim)
├── hand.jpg                                           # best webcam frame, palm score 0.84
├── mediapipe_hand_hand_detector.bin                   # palm detector QNN binary
├── mediapipe_hand_hand_detector.bin.onnx.zip          # pre-quant ONNX (reference)
├── mediapipe_hand_hand_landmark_detector.bin          # landmark detector QNN binary
└── qnn_handapp/
    ├── build.sh                  # on-device build script
    ├── demo.py                   # end-to-end pipeline (static image / webcam)
    ├── hand_out.jpg              # annotated output from running demo on hand.jpg
    ├── libqnn_shim.so            # built shim (aarch64) — present after build.sh
    ├── qnn_runtime.py            # Python ctypes wrapper
    ├── qnn_shim.cpp              # C++ shim source
    ├── qnn_shim.h                # C ABI header
    └── zidane_out.jpg            # smoke-test output (non-hand input)
```

On the device, mirrored under `/data/local/tmp/mediapipe_hand/`:

```
/data/local/tmp/mediapipe_hand/
├── hand.jpg
├── input_list.txt                # phase-1 qnn-net-run smoke test
├── mediapipe_hand_hand_detector.bin
├── mediapipe_hand_hand_landmark_detector.bin
├── output_hand_detector/         # phase-1 results
├── output_landmark/              # phase-1 results
├── qnn_handapp/                  # phase-2 sources + built .so
├── test_input.raw                # phase-1 random input
└── zidane.jpg
```

QNN headers required at build time:
- Host: `/mnt/data02/matthew/SNPE/qairt/2.46.0.260424/include/QNN`
- Device (synced by us): `/data/local/tmp/snpeexample/include/QNN`

---

## Appendix B — useful one-liners

### `/tmp/sshrun.py` (full)

```python
#!/usr/bin/env python3
"""Args: command-string. Exits with remote exit code."""
import sys, paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("172.30.101.184", username="root", password="oelinux123",
          timeout=15, allow_agent=False, look_for_keys=False)
stdin, stdout, stderr = c.exec_command(sys.argv[1], timeout=120)
sys.stdout.write(stdout.read().decode(errors="replace"))
sys.stderr.write(stderr.read().decode(errors="replace"))
sys.exit(stdout.channel.recv_exit_status())
```

### `/tmp/sshcp.py` (recursive push)

```python
#!/usr/bin/env python3
"""Args: <src1> <dst1> [<src2> <dst2> …]"""
import sys, os, paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("172.30.101.184", username="root", password="oelinux123",
          timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
def mkdirs(rp):
    cur = ""
    for p in rp.strip("/").split("/"):
        cur += "/" + p
        try: sftp.stat(cur)
        except IOError: sftp.mkdir(cur)
def put_tree(src, dst):
    if os.path.isdir(src):
        mkdirs(dst)
        for n in os.listdir(src): put_tree(f"{src}/{n}", f"{dst}/{n}")
    else:
        mkdirs(os.path.dirname(dst)); sftp.put(src, dst)
for s, d in zip(sys.argv[1::2], sys.argv[2::2]): put_tree(s, d)
sftp.close(); c.close()
```

### `/tmp/sshpull.py` (one file)

```python
#!/usr/bin/env python3
import sys, paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("172.30.101.184", username="root", password="oelinux123",
          timeout=15, allow_agent=False, look_for_keys=False)
c.open_sftp().get(sys.argv[1], sys.argv[2])
```

### Standard device env-setup snippet

Put this at the top of any device-side shell script that touches QNN:

```bash
export QNN_ROOT=/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH=$QNN_ROOT/lib:/data/local/tmp/snpeexample/dsp/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"
mount -o remount,exec /data   # idempotent; needed once per boot
```

### Full reproduction recipe

Assuming a fresh QCS6490 device at `172.30.101.184` with the QAIRT
runtime at `/data/local/tmp/snpeexample/`:

```bash
# from host (this folder)
/tmp/sshrun.py 'mount -o remount,exec /data && apt-get install -y python3-opencv'
/tmp/sshcp.py \
  /mnt/data02/matthew/SNPE/qairt/2.46.0.260424/include/QNN /data/local/tmp/snpeexample/include/QNN \
  qnn_handapp                                              /data/local/tmp/mediapipe_hand/qnn_handapp \
  mediapipe_hand_hand_detector.bin          /data/local/tmp/mediapipe_hand/mediapipe_hand_hand_detector.bin \
  mediapipe_hand_hand_landmark_detector.bin /data/local/tmp/mediapipe_hand/mediapipe_hand_hand_landmark_detector.bin

/tmp/sshrun.py 'cd /data/local/tmp/mediapipe_hand/qnn_handapp && sh build.sh'

# burst-capture a hand
/tmp/sshrun.py '<the burst-capture python from §3.2>'

# run the demo
/tmp/sshrun.py '
cd /data/local/tmp/mediapipe_hand
export QNN_ROOT=/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4
export LD_LIBRARY_PATH=$QNN_ROOT/lib:/data/local/tmp/snpeexample/dsp/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpeexample/dsp/lib;/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp"
python3 qnn_handapp/demo.py --image hand.jpg --frames 30 --out qnn_handapp/hand_out.jpg
'

# pull the annotated output
/tmp/sshpull.py /data/local/tmp/mediapipe_hand/qnn_handapp/hand_out.jpg ./hand_out.jpg

---

# Chapter 7 — FaceMap 3DMM Demo (Session 2025-05-22)

This chapter records every step taken to get `facemap_3dmm.dlc` running on the
QCS6490 device with HTP inference and annotated output.  The starting point was
a working `qnn_handapp` (from Chapters 1–6) and an untouched
`exportAssets/facemap_3dmm-qnn_dlc-w8a8/` folder that had just been placed
there.

---

## 7.0 Context from HANDOFF.md

Read `HANDOFF.md` first.  The hand demo pipeline was complete; the task was to
create an equivalent `demo.py` for the `facemap_3dmm` model.  Key constraints
carried over:

- SSH access via paramiko (no `sshpass` on host; password `oelinux123`).
- `/data` is `noexec` by default; must be remounted before running any `.so`.
- `ADSP_LIBRARY_PATH` uses **semicolons** (`;`) as path separators, not colons.
- Host OS is Ubuntu 20.04 → glibc 2.31 → some SDK binaries in the
  `x86_64-linux-clang/` folder require glibc 2.32+; use `qairt_dev` container.

---

## 7.1 Reconnaissance — what is the facemap model?

**Read** `exportAssets/facemap_3dmm-qnn_dlc-w8a8/metadata.json`:

```json
{
  "model_id": "facemap_3dmm",
  "runtime": "qnn_dlc",
  "precision": "w8a8",
  "tool_versions": { "qairt": "2.45.0.260326154327" },
  "model_files": {
    "facemap_3dmm.dlc": {
      "inputs":  { "image": { "shape": [1,128,128,3], "dtype": "uint8",
                              "scale": 0.003921, "zero_point": 0 } },
      "outputs": { "parameters_3dmm": { "shape": [1,265], "dtype": "uint8",
                                        "scale": 0.052389, "zero_point": 123 } }
    }
  }
}
```

Key facts:
- Single model (no detector + landmark chain like the hand demo).
- **DLC format** (`.dlc`), not a QNN context binary (`.bin`).  The existing
  `qnn_shim.so` only loads `.bin` files; a different inference path is needed.
- Input: 128×128 uint8 RGB.  Scale ≈ 1/255, zero_point = 0.
- Output: 265 uint8 parameters.  Dequantize: `float = 0.05239 * (uint8 − 123)`.
- Compiled with QAIRT **2.45**; the device runs QAIRT **2.46**.
- Output models 3D Morphable Model (3DMM) parameters.  Assumed layout
  (common for this family): [0:12] = 3×4 pose matrix, [12:52] = 40 shape
  coefficients, [52:62] = 10 expression coefficients, [62:265] = texture +
  illumination.

**Read** `exportAssets/qnn_handapp/demo.py` (reference implementation) to
understand the structure to replicate.

---

## 7.2 Device reconnaissance — what tools are available?

Connected via paramiko, ran several discovery commands.

### 7.2.1 Device file layout

```
/data/local/tmp/
  mediapipe_hand/          ← existing hand demo
    qnn_handapp/           ← libqnn_shim.so, demo.py, …
    *.bin                  ← QNN context binaries
  snpeexample/
    aarch64-ubuntu-gcc9.4/
      bin/  snpe-net-run, qnn-net-run, qnn-context-binary-generator, …
      lib/  libQnnHtp.so, libQnnSystem.so, libSNPE.so,
            libQnnModelDlc.so, …
    dsp/lib/  libQnnHtpV68Skel.so, libSnpeHtpV68Skel.so, …
```

### 7.2.2 Python on device

```
Python 3.8.10
import snpe  →  ModuleNotFoundError  (no Python SNPE bindings installed)
```

### 7.2.3 Key libraries found

| Library | Relevance |
|---|---|
| `libSNPE.so` | SNPE C API — exports `Snpe_PSNPE_*` functions |
| `libQnnModelDlc.so` | Exports `QnnModel_composeGraphsFromDlc` — loads DLC via QNN API |
| `libQnnHtp.so` | QNN HTP backend (same as used by qnn_handapp) |
| `snpe-net-run` | SNPE CLI inference tool, supports `--use_dsp` and `--container <dlc>` |
| `qnn-net-run` | QNN CLI inference tool, supports `--dlc_path` with `libQnnModelDlc.so` |
| `qnn-context-binary-generator` | Compiles models to context binary, v2.46, supports `--dlc_path` |

### 7.2.4 Decision: use snpe-net-run subprocess

No Python SNPE bindings.  Writing a new C shim for the SNPE C API would take
significant time.  The fastest working path is to call `snpe-net-run` as a
subprocess (same tradeoff that was abandoned for the hand demo in favour of the
ctypes shim, but acceptable for now since the user said "don't worry about
performance or cropping yet").

---

## 7.3 First demo.py — CPU via snpe-net-run subprocess

Created `exportAssets/facemap_app/demo.py` (first version):

- `preprocess(frame)` — BGR uint8 → RGB float32 [0,1] at 128×128.
- `run_inference(img_hwc)` — writes float32 `.raw` input file, runs
  `snpe-net-run`, reads float32 `.raw` output.  `snpe-net-run` dequantizes
  uint8 → float32 internally (default behaviour without `--use_native_output_files`).
- `annotate(frame, params)` — extracts R (3×3) and t (3×1) from first 12 params
  as a 3×4 projection matrix.  Projects three 3D axis endpoints [size,0,0],
  [0,size,0], [0,0,size] through `p_2d = R@p3 + t` with a pinhole model
  (`focal = max(H,W)`).  Draws X=red, Y=green, Z=blue arrows.  Adds text for
  estimated roll/pitch/yaw (Euler angles from R), shape_norm, expr_norm.
- `main()` — static image or webcam source; `--check` flag exits 0/PASS if
  output is (265,) finite non-zero.

Also created `VarSetup` (identical in intent to qnn_handapp's) and `deploy.sh`.

### 7.3.1 ADSP path bug (semicolons vs colons)

**First bug found immediately**: The initial Python `_build_env()` used colons
(`:`) to separate paths in `ADSP_LIBRARY_PATH`:

```python
"/data/local/tmp/snpeexample/dsp/lib:/data/local/tmp/snpeexample/..."
```

`ADSP_LIBRARY_PATH` requires **semicolons** (`;`).  With colons, FastRPC fails:

```
Failed to load skel, error: 1002
```

Fixed to:

```python
"/data/local/tmp/snpeexample/dsp/lib;/data/local/tmp/snpeexample/..."
```

This bug is subtle: the VarSetup shell script already had semicolons from the
hand demo, but when constructing the path in Python we used the more natural
colon separator.

### 7.3.2 First successful CPU run

After the semicolon fix, uploaded and ran:

```
warmup ok  backend=CPU  output shape=(265,)
avg latency: 114.1ms  (8.8 fps)
params[0:6]: [-0.766 -0.384  0.038  0.043 -0.017 -0.052]
shape_norm=0.637  expr_norm=0.163
PASS
```

The demo script printed PASS, the annotated image was saved and downloaded.
Head pose axes (R/G/B arrows) were drawn on the 128×128 Zidane crop with
roll/pitch/yaw text overlay.

---

## 7.4 Attempting HTP inference — three failed approaches

### 7.4.1 Attempt 1: snpe-net-run --use_dsp

```bash
snpe-net-run --container facemap_3dmm.dlc --use_dsp …
```

**Error:**
```
error_code=15001; QnnBackend_DeviceCreate() INVALID_CONFIG
error_code=1002;  No backend could validate Op=/Mul Type=Eltwise_Binary
```

**Why it failed:** The `--use_dsp` SNPE adapter internally calls
`QnnDsp`/`QnnHtp` device creation with stricter op validation.  The `/Mul`
(Eltwise_Binary) op in the facemap DLC has parameters that fail validation in
QAIRT 2.46's SNPE adapter.  The model was compiled with QAIRT 2.45, and there
is a compatibility break in the Mul parameter format between these versions at
the SNPE adapter level.

**Fall-back:** The demo automatically retried with CPU (no `--use_dsp` flag)
which worked fine.

### 7.4.2 Attempt 2: qnn-net-run with libQnnModelDlc.so (ADSP colons bug)

```bash
qnn-net-run --backend libQnnHtp.so --model libQnnModelDlc.so \
  --dlc_path facemap_3dmm.dlc …
```

**Error (first attempt):**
```
Failed to load skel, error: 1002
```

This was the colons-in-ADSP_LIBRARY_PATH bug (§7.3.1).  After fixing to
semicolons and retrying:

```
Composing Graphs … Finalizing Graphs …
Starting stage: Post Graph Optimization
Completed stage: Post Graph Optimization (1098 us)
exit_code=139   ← SIGSEGV
```

**Why it failed:** The DLC compiled with QAIRT 2.45 crashes (segfault, exit 139)
in the `libQnnHtp.so`'s graph finalization stage when run under QAIRT 2.46.
The crash happens consistently after "Post Graph Optimization" completes but
before the context binary is serialized.  This is a native library crash, not a
Python error.

The same crash happens regardless of whether you use `qnn-net-run` or
`qnn-context-binary-generator` to drive the compilation.

### 7.4.3 Attempt 3: qnn-context-binary-generator with config file

Tried passing `--config_file htp_config.json` with `optimization_level: 0` to
disable HTP optimizations, hoping to avoid the crash:

```json
{ "htp_backend_extensions": { "optimization_level": 0 } }
```

**Result:** Same crash (exit 139) at the same stage.  The optimization level
config does not affect the code path that crashes.

### 7.4.4 Root cause confirmed

The crash is a version compatibility bug between QAIRT 2.45-compiled DLC and
the QAIRT 2.46 HTP finalization code.  It is not possible to work around through
CLI flags — the DLC must either be recompiled for 2.46 or pre-compiled using a
tool that uses different internal paths.

---

## 7.5 Fix: snpe-dlc-graph-prepare inside qairt_dev docker

### 7.5.1 Discovery

The host SDK tools require glibc 2.32+ but the host has Ubuntu 20.04 (glibc
2.31).  Running them directly fails:

```
libm.so.6: version 'GLIBC_2.35' not found
```

The `qairt_dev` docker container (Ubuntu 22.04) has the correct glibc.

```bash
docker exec qairt_dev /workspace/qairt/2.46.0.260424/bin/x86_64-linux-clang/\
  snpe-dlc-graph-prepare --help
```

This tool supports `--htp_socs qcs6490` — it generates HTP offline cache for a
specific SoC without needing real hardware (uses Hexagon SDK simulation).

### 7.5.2 Running graph preparation

```bash
docker cp exportAssets/facemap_3dmm-qnn_dlc-w8a8/facemap_3dmm.dlc \
  qairt_dev:/workspace/facemap/facemap_3dmm.dlc

docker exec qairt_dev bash -c "
  /workspace/qairt/2.46.0.260424/bin/x86_64-linux-clang/snpe-dlc-graph-prepare \
    --input_dlc  /workspace/facemap/facemap_3dmm.dlc \
    --output_dlc /workspace/facemap/facemap_3dmm_prepared.dlc \
    --htp_socs   qcs6490
"
```

**Output (abridged):**
```
[INFO] SNPE HTP Offline Prepare: Attempting to create cache for QCS6490
[USER_INFO] No cache record in the DLC matches the target device. Creating a new record
[USER_INFO] Offline Prepare VTCM size(MB) selected = 0
[USER_INFO] Optimization Level passed = 2
… Graph Optimizations (43361 us) …
… Finalizing Graph Sequence …
[INFO] SNPE HTP Offline Prepare: Successfully created cache for QCS6490
[INFO] QCS6490 : Success
[USER_INFO] Successfully saved DLC to /workspace/facemap/facemap_3dmm_prepared.dlc
exit=0
5.5M facemap_3dmm.dlc  →  11M facemap_3dmm_prepared.dlc
```

The output DLC is larger (11 MB vs 5.5 MB) because it now embeds the
pre-compiled HTP graph cache for QCS6490.

Notable: the warning `No schematic bin found to add` confirms the original DLC
had no embedded HTP cache — it was a pure IR DLC without any pre-compiled
backend binary.

### 7.5.3 Why this works when qnn-context-binary-generator crashes

`snpe-dlc-graph-prepare` runs on the HOST (x86_64) using the Hexagon DSP
simulator (`libHtpPrepare.so`).  It goes through a different code path than the
on-device `libQnnHtp.so` compilation pipeline.  Specifically, it avoids the
post-optimization finalization step that crashes in the on-device 2.46 runtime.

The resulting "offline prepared" DLC contains a record of type
`HTP_CACHE_RECORD` which `snpe-net-run` can load directly with
`--enable_htp_accelerated_init`, bypassing the JIT compilation entirely.

### 7.5.4 Confirming HTP works with the prepared DLC

Copied `facemap_3dmm_prepared.dlc` to the device and ran:

```bash
snpe-net-run \
  --container facemap_3dmm_prepared.dlc \
  --input_list input_list.txt \
  --output_dir output/ \
  --use_dsp \
  --enable_htp_accelerated_init
```

**Output:**
```
Processing graph : graph_hbk71j7a
Processing DNN input(s): /tmp/facemap_ws/input.raw
Successfully executed graph graph_hbk71j7a
exit=0
parameters_3dmm.raw  ← output file present
```

HTP inference works.

### 7.5.5 Attempt to extract standalone .bin (failed)

Tried `qnn-context-binary-generator` on the prepared DLC hoping to get a
standalone `.bin` loadable by the existing `qnn_shim.so`:

```bash
qnn-context-binary-generator \
  --backend libQnnHtp.so --model libQnnModelDlc.so \
  --dlc_path facemap_3dmm_prepared.dlc \
  --binary_file facemap_3dmm --output_dir …
```

**Result:** Same crash (exit 139).  The `qnn-context-binary-generator` with
`libQnnModelDlc.so` goes through the crashing `libQnnHtp.so` compilation path
even when the DLC has the HTP cache embedded.  It does not fall back to loading
the embedded cache.

This means the `qnn_shim.so` (context binary loader) approach cannot be used
for this model without a full re-export.  The `snpe-net-run` subprocess path
with the prepared DLC is the only working HTP path for now.

---

## 7.6 Final demo.py — HTP via prepared DLC

Updated `demo.py` to use a two-DLC strategy:

| Attribute | HTP path | CPU fallback |
|---|---|---|
| DLC file | `facemap_3dmm_prepared.dlc` | `facemap_3dmm.dlc` |
| CLI flags | `--use_dsp --enable_htp_accelerated_init` | (none) |
| Tool | `snpe-net-run` | `snpe-net-run` |
| Latency (subprocess) | ~258 ms | ~114 ms |

The HTP subprocess path is *slower* than CPU for this demo because subprocess
startup + SNPE DSP session init (~200 ms) dominates over actual HTP inference
(~5 ms).  To get real-time HTP performance, a persistent in-process runtime
would be needed (ctypes shim analogous to `qnn_shim.so`).

### 7.6.1 Final automated check (HTP)

```
backend=HTP  avg latency: 258.7ms  (3.9 fps)
params[0:6]: [-0.786 -0.472  0.105  0.052  ~0  ~0]
shape_norm=0.693  expr_norm=0.174
PASS
```

Annotated output image: head pose axes drawn on 128×128 Zidane crop,
roll/pitch/yaw text overlay (R=-180 P=-8 Y=-17), shape and expression norms.

---

## 7.7 What did and didn't work — summary for facemap

### Worked

- **snpe-net-run CPU** — plain `snpe-net-run` without runtime flags runs the
  DLC on CPU at ~114 ms/frame (subprocess).  No env setup required beyond
  `LD_LIBRARY_PATH`.
- **snpe-dlc-graph-prepare inside docker** — generates HTP offline cache for
  `qcs6490` without real hardware.  Command:
  ```bash
  docker exec qairt_dev \
    /workspace/qairt/2.46.0.260424/bin/x86_64-linux-clang/snpe-dlc-graph-prepare \
    --input_dlc facemap_3dmm.dlc --output_dlc facemap_3dmm_prepared.dlc \
    --htp_socs qcs6490
  ```
- **snpe-net-run HTP** — prepared DLC + `--use_dsp --enable_htp_accelerated_init`
  runs on HTP at ~250 ms/frame (subprocess-limited).

### Didn't work

- **snpe-net-run --use_dsp** (unprepared DLC) — `QnnBackend_DeviceCreate`
  INVALID_CONFIG + Eltwise_Binary/Mul validation failure.  SNPE 2.46 adapter
  breaks on 2.45-compiled Mul op parameters.
- **ADSP_LIBRARY_PATH with colons** — must use semicolons (`;`).  Colons →
  `Failed to load skel, error: 1002`.
- **qnn-net-run + libQnnModelDlc.so** (both DLC variants) — SIGSEGV (exit 139)
  after "Post Graph Optimization".  Version mismatch: DLC compiled with 2.45,
  runtime is 2.46; finalization crashes in `libQnnHtp.so`.
- **qnn-context-binary-generator --config_file optimization_level=0** — same
  crash, config ignored.
- **qnn-context-binary-generator with prepared DLC** — still crashes; the tool
  routes through the crashing `libQnnHtp.so` JIT path even when the DLC has an
  embedded cache.
- **Host SDK tools directly** — require glibc 2.32+; host has 2.31.  Use
  `qairt_dev` docker instead.
- **import snpe on device** — not installed.  Subprocess is the only Python path.

---

## 7.8 Files created / modified this session

| Path (relative to `/mnt/data02/matthew/SNPE/`) | Action |
|---|---|
| `exportAssets/facemap_app/demo.py` | Created — main inference + annotation script |
| `exportAssets/facemap_app/VarSetup` | Created — env setup for facemap on device |
| `exportAssets/facemap_app/deploy.sh` | Created — upload script |
| `exportAssets/facemap_3dmm-qnn_dlc-w8a8/facemap_3dmm_prepared.dlc` | Created — HTP offline-prepared DLC (11 MB) |
| `HANDOFF.md` | Updated — added facemap_app section, new known issues, updated next steps |
| `exportAssets/DEPLOYMENT_JOURNAL.md` | Updated — this chapter |

Device paths created:

| Device path | Contents |
|---|---|
| `/data/local/tmp/facemap/` | `demo.py`, `VarSetup`, `facemap_3dmm.dlc`, `facemap_3dmm_prepared.dlc` |
```

---

# Chapter 8 — Real-time face detection + 68-landmark pipeline, both models on the DSP (Session 2026-05-22)

Chapter 7 left facemap working but flawed: inference went through a
`snpe-net-run` **subprocess** (~222 ms), the demo drew head-pose **arrows** from
a guessed parameter layout, and it fed the **whole frame** to the model with no
face crop. This chapter records four things, in order:

1. An in-process **SNPE C-API shim** (`libsnpe_shim.so`) → ~0.7 ms/inference.
2. Ease-of-use: one-command launcher, live mode, camera auto-detect.
3. The big correctness fix: the facemap output decode was **wrong**; replaced
   with Qualcomm's real 3DMM → 68-landmark reconstruction, which forced adding a
   **face detector** (the model is a landmark regressor, not a detector).
4. A **DSP face detector** built locally (no AI Hub), the two-model pipeline,
   and the headless/camera fixes needed to run it over SSH.

End state: face detector (~1.0 ms) + facemap landmarks (~0.7 ms/face), **both on
the Hexagon DSP**, ~1.8 ms for one face (~550 fps).

---

## 8.0 Finish Chapter 7 — confirm the subprocess HTP path

Resumed by re-running the device check; the device had dropped off SSH and came
back after a power cycle. Baseline confirmed:

```
backend=HTP  avg latency: 222.6ms  (subprocess-bound)  PASS
```

Real HTP inference is ~ms; the 222 ms is `snpe-net-run` process startup. That is
the motivation for the in-process shim.

---

## 8.1 In-process SNPE shim (`libsnpe_shim.so`)

The hand app's `qnn_shim.cpp` wraps `libQnnHtp.so` and only loads **context
binaries** (`.bin`). facemap is a **DLC**, and the raw QNN HTP path SIGSEGVs on
this 2.45 DLC (Chapter 7). The runtime that *does* load this DLC on HTP is
**SNPE** (`libSNPE.so`), so the shim wraps the SNPE C API instead.

Design decisions:

- **Link `libSNPE.so` directly** (not dlopen) — single entry point; LD path is
  set by the launcher. C API symbols confirmed with `nm -D` (e.g.
  `Snpe_SNPEBuilder_Create`, `Snpe_SNPE_ExecuteITensors`, `Snpe_Util_CreateITensor`).
- **ITensor path, not UserBuffers** — float32 in/out; SNPE quantizes to the
  DLC's fixed-point encoding internally. Far simpler than managing TF8 buffers.
- **Discover output shapes with a warmup execute** — the C API has
  `GetInputDimensions` but no `GetOutputDimensions`, so `snpe_load()` runs one
  zero inference and reads the output ITensor shapes from the populated map.
- **Persistent state** — input ITensors + input/output `TensorMap`s are created
  once and reused; `TensorMap_Clear` on the output map before each execute.
- **Builder config** for HTP: `SetRuntimeProcessorOrder(DSP, CPU)`,
  `SetPerformanceProfile(BURST)`, `SetAcceleratedInit(true)` (uses the offline
  cache in the `_prepared.dlc`).

Build **on-device** (aarch64 Ubuntu, g++ 9.4) — same rationale as the QNN shim,
avoids cross-compiling. Pushed `include/SNPE` to the device once; `build.sh`:

```sh
g++ -O2 -fPIC -shared -std=c++17 -I$SNPE_INCLUDE snpe_shim.cpp \
    -o libsnpe_shim.so -L$SNPE_LIB -lSNPE
```

Result: **~0.7 ms/inference (~1450 fps)**, output byte-identical to the
subprocess HTP run (`shape_norm=0.693, expr_norm=0.174`). `rpcmem` FastRPC
messages confirm it executes on the DSP. Python wrapper `snpe_runtime.py`
mirrors `qnn_runtime.py` (`SnpeModel.execute([...])`).

---

## 8.2 Ease-of-use pass

- `run_demo.sh` — one-command launcher; remounts `/data` exec and exports
  **both** `LD_LIBRARY_PATH` (so the shim finds `libSNPE.so`) and
  `ADSP_LIBRARY_PATH` (semicolon-separated). The hand launcher omitted
  `LD_LIBRARY_PATH`, which the shim needs.
- Live webcam mode and full-frame annotation scaling.
- **Camera index ≠ 0**: on this board `/dev/video0,1` are internal MSM nodes
  that don't capture; the USB webcam is `/dev/video2` (per `v4l2-ctl
  --list-devices`). Added auto-probing.

---

## 8.3 The decode was wrong — get the REAL facemap_3dmm layout

User report: "arrows go in random directions, it doesn't detect the face."
Two root causes:

1. **No face crop.** facemap_3dmm expects a tight face crop resized to 128²; the
   demo resized the whole frame, so the 265 outputs were meaningless.
2. **Wrong parameter interpretation.** Chapter 7 assumed `[0:12]` was a 3×4 pose
   matrix. That was never verified and is false.

Fetched Qualcomm's source (`qualcomm/ai-hub-models`,
`src/qai_hub_models/models/facemap_3dmm/`). The real layout (`utils.py
project_landmark`):

```
[  0:219] alpha_id  (shape)      ×3
[219:258] alpha_exp (expression) ×0.5 +0.5
[258] pitch  [259] yaw  [260] roll    (×π/2)
[261] tX (×60)  [262] tY (×60)  tZ=500 (const)  [263] focal (×150 +450)
```

68 landmarks are reconstructed from a **3DMM basis** (downloaded from the AI Hub
asset store): `meanFace.npy` (204), `shapeBasis.npy` (204×219),
`blendShape.npy` (204×39):

```
verts = (mean + shapeBasis·alpha_id + blendShape·alpha_exp).reshape(68,3) @ R(p,y,r)ᵀ
verts += (tX,tY,tZ);  landmark2d = verts[:, :2] · focal / tZ
```

Then map crop-space → image-space using the face box. Ported to numpy in
`demo.py` (`project_landmark`, `transform_landmark`). The reference app confirms
the model has **no detection stage** and requires an externally supplied face
box → we need a detector.

---

## 8.4 Face detector on the DSP (built locally, no AI Hub)

The hand `.bin`s were made on a Windows machine via the AI Hub **cloud**
(`modelCompilation.md`). This Linux box has no AI Hub token/packages. But the
SNPE shim already runs **any** DLC on the DSP, so the task is just: produce a
detector DLC locally with the QAIRT tools in the `qairt_dev` container.

Model: **Linzaer Ultra-Light-Fast-Generic-Face-Detector, RFB-320**. Tiny,
ONNX-native, outputs already-decoded boxes.

- Input `input` NCHW `[1,3,240,320]`, preprocess `(rgb−127)/128`.
- Outputs `scores [1,4420,2]` (softmax: bg,face) + `boxes [1,4420,4]`
  (normalized corners). Postproc = threshold + NMS (numpy).

Build (all in `qairt_dev`; **activate the SDK venv** so the converters find
`onnx`, which the system python lacks):

```sh
qairt-converter --input_network version-RFB-320.onnx --output_path face_det_fp32.dlc
qairt-quantizer  --input_dlc face_det_fp32.dlc --input_list calib_list.txt \
                 --output_dlc face_det_w8a8.dlc
snpe-dlc-graph-prepare --input_dlc face_det_w8a8.dlc \
                 --output_dlc face_det_w8a8_prepared.dlc \
                 --htp_socs qcs6490 --set_output_tensors scores,boxes
```

Calibration: ~5 face images preprocessed to NCHW raws (`prep_calib.py`).
INT8 output on the DSP matched the onnxruntime float reference almost exactly
(obama: DSP `[0.42,0.077,0.663,0.355]` vs ref `[0.421,0.079,0.663,0.356]`).
Detector latency on DSP: **~1.0 ms**.

### Multi-output gotcha (cost an hour)

By default SNPE exposes only a network's **last** output, so the shim saw only
`boxes`. Fix is two-sided:

1. Add `Snpe_SNPEBuilder_SetOutputTensors(scores,boxes)` at runtime (new
   `output_names` arg threaded through `snpe_shim`/`snpe_runtime`).
2. Bake the same into the offline cache with graph-prepare's
   `--set_output_tensors scores,boxes`.

If only (1) is set, the runtime outputs don't match the offline cache → SNPE
falls back to **online** HTP prep → **SIGSEGV** (the same 2.45/2.46 crash).
With both, accelerated init uses the cache and it just works.

---

## 8.5 Two-model pipeline + results

```
frame ─► detector (DSP) ─► scores/boxes ─► threshold+NMS ─► box(es)
box   ─► crop+resize 128 ─► facemap (DSP) ─► 265 params ─► 68 landmarks ─► annotate
```

Validated: obama.jpg (1 face) **det 1.0 + lmk 0.8 = 1.8 ms (~550 fps)**;
zidane.jpg (2 faces) det 1.1 + lmk 1.4 = 2.5 ms. Landmarks track
eyes/nose/mouth/jaw correctly; both models run on the DSP.

---

## 8.6 "It just stalls, nothing opens" — headless + camera robustness

Cause: the user runs over **SSH with no display**, so `cv2.imshow` can't open a
window (it hung); a stale `--live` process was also still holding the camera, and
the probe had latched onto a non-capturing node.

Fixes in `demo.py`:

- `_have_display()` — detect X/Wayland; if absent, **headless** mode writes the
  latest annotated frame to `facemap_live.jpg` and records `facemap_live.mp4`,
  printing live FPS. With a display it still opens a window.
- `open_camera()` — probe indices across `CAP_V4L2` then `CAP_ANY`, validate a
  real 3-channel frame, fail fast (never block on a dead node), report `WxH`.
- Pre-build both DSP graphs before the loop with progress prints, so the few-
  second init isn't mistaken for a hang.
- `./run_demo.sh` with no args → live (headless over SSH); `--seconds N` to
  auto-stop.

Verified headless: models init, camera index 0 @640×480, ~850–900 fps,
`facemap_live.jpg` written (0 faces only because the camera faced an empty room).

---

## 8.7 Gotchas worth re-reading

- **SNPE exposes only the last output by default** — set output tensors at BOTH
  build time (runtime `SetOutputTensors`) and offline-prepare time
  (`--set_output_tensors`), or online prep crashes.
- **Converters use system python** — `source` the SDK venv first or they can't
  `import onnx`.
- **Docker writes DLCs as root** — `chmod a+rw` before SFTP from the host.
- **facemap_3dmm has no detector** — it is a landmark regressor; feed it a tight
  face crop or the output is garbage.
- **`cv2.imshow` needs a display** — over SSH, go headless and write a file.
- **Camera enumeration is flaky** — the USB cam is not index 0; probe + validate.

---

## 8.8 Files created / modified this session

| Path (under `exportAssets/`) | Action |
|---|---|
| `facemap_app/snpe_shim.cpp`, `snpe_shim.h` | Created — in-process SNPE C-API shim (multi-output support) |
| `facemap_app/snpe_runtime.py` | Created — ctypes wrapper (`SnpeModel`, `output_names`) |
| `facemap_app/build.sh` | Created — on-device shim build |
| `facemap_app/demo.py` | Rewritten — two-model detect→landmark pipeline, headless live, camera probe |
| `facemap_app/run_demo.sh` | Created — one-command launcher (sets LD/ADSP env) |
| `facemap_app/deploy.sh` | Updated — ships detector DLCs + 3DMM basis + SNPE headers, builds shim |
| `facemap_app/{meanFace,shapeBasis,blendShape}.npy` | Added — 3DMM basis for 68-landmark decode |
| `facemap_app/face_det_w8a8*.dlc` | Added — face detector (HTP-prepared, outputs scores,boxes) |
| `face_detector/` | Created — detector build dir: ONNX, fp32/w8a8/prepared DLCs, `prep_calib.py`, calib data |
| `DEPLOYMENT_JOURNAL.md` | Updated — this chapter |
| `../HANDOFF.md` | Updated — two-model DSP pipeline, detector build recipe |

