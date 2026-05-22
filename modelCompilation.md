# Technical Deep Dive: MediaPipe Hand on QCS6490 HTP

A complete record of every decision, problem, and fix encountered when compiling and deploying the MediaPipe Hand model to a QCS6490-based Qualcomm Linux device.

---

## Table of Contents

1. [Goal](#goal)
2. [Environment](#environment)
3. [The Model](#the-model)
4. [Running the Local Demo](#running-the-local-demo)
5. [Why the Standard Export Path Fails](#why-the-standard-export-path-fails)
6. [The Solution: Low-Level qai_hub API](#the-solution-low-level-qai_hub-api)
7. [Step-by-Step Pipeline with Every Error](#step-by-step-pipeline-with-every-error)
8. [Final Working Pipeline](#final-working-pipeline)
9. [Deployment](#deployment)
10. [Key Decisions Summary](#key-decisions-summary)

---

## Goal

Deploy the [MediaPipe Hand model](https://github.com/qualcomm/ai-hub-models/tree/v0.54.0/src/qai_hub_models/models/mediapipe_hand) from Qualcomm's AI Hub Models repository onto a QCS6490-based device running Qualcomm Linux, targeting the on-chip **Hexagon v68 HTP (DSP)** for inference — not the CPU or GPU.

The model detects hands and outputs bounding boxes, palm keypoints, and 21 hand landmarks. It is composed of two sequential sub-networks:

- **HandDetector** — a BlazePalm-based palm detector
- **HandLandmarkDetector** — a BlazeHand-based 21-point landmark regressor

---

## Environment

| Component | Version / Detail |
|---|---|
| Host OS | Windows 11 Pro |
| Python | 3.11.9 |
| Package | `qai-hub-models[mediapipe_hand]` v0.54.0 |
| Target chip | QCS6490 (Hexagon v68 HTP) |
| Target OS | Qualcomm Linux 1.6 |
| AI Hub device string | `"Dragonwing RB3 Gen 2 Vision Kit"` (proxy device used for cloud compilation — user's actual board shares the same QCS6490 chipset) |

The `qai-hub-models` package was already installed. Python 3.11.9 satisfies the package requirement of `3.10 <= PYTHON_VERSION < 3.14`.

---

## The Model

### Architecture

MediaPipe Hand is a `CollectionModel` — a container for two independent `BaseModel` subclasses that run in sequence:

```
Input frame
    │
    ▼
HandDetector          (BlazePalm)
  input:  (1, 3, 256, 256) float32
  output: box_coords (1, N, 18), box_scores (1, N, 1)
    │
    │  (crop & rotate hand region from bounding box)
    ▼
HandLandmarkDetector  (BlazeHandLandmark)
  input:  (1, 3, 256, 256) float32
  output: scores, lr (left/right hand), landmarks (21 × x,y,z)
```

### Memory layout

Both models declare `get_channel_last_inputs() → ["image"]`, meaning they prefer NHWC (height × width × channels) memory layout on hardware even though the PyTorch model operates in NCHW. The AI Hub compile step handles this transpose when `--force_channel_last_input image` is passed.

### Third-party dependency

The model weights and architecture come from [zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch). On first load, `qai_hub_models` prompts interactively to clone this repo to `~/.qaihm/qai-hub-models/models/mediapipe_pytorch/v1/zmurez_MediaPipePyTorch_git/`. This clone only happens once; subsequent loads reuse the cached copy.

---

## Running the Local Demo

### First attempt (non-interactive shell)

```
python -m qai_hub_models.models.mediapipe_hand.demo --use-default-image
```

**Error:**
```
EOFError: EOF when reading a line
```

**Why:** The demo loads `MediaPipePyTorch` on first run, which calls `input()` to ask permission to clone the repo. Running through a non-interactive shell (Claude Code's bash tool) means stdin is closed — `input()` immediately raises `EOFError`.

**Fix:** Clone the repo manually first:
```bash
git clone https://github.com/zmurez/MediaPipePyTorch
```
Then run the demo interactively in a real terminal. The repo gets cloned to the internal cache path and the prompt never appears again.

### Second attempt (webcam)

Running without `--use-default-image` opens the default camera (`/dev/video0` equivalent, camera index 0) and runs inference in a live loop. The terminal printed:

> *"Note: This demo is running through torch, and not meant to be real-time without dedicated ML hardware."*

This is expected — on a standard x86 CPU running PyTorch, each inference pass takes hundreds of milliseconds. The model is designed for Qualcomm NPU hardware where the same inference takes single-digit milliseconds.

---

## Why the Standard Export Path Fails

The normal export command is:
```bash
python -m qai_hub_models.models.mediapipe_hand.export \
  --device "QCS6490 (Proxy)" \
  --target-runtime qnn_context_binary
```

**Error:**
```
ValueError: The selected precision (float) requires FP16 support,
but the selected device does not support FP16.
Please try a different precision or target device.
```

### Root cause

The `qai_hub_models` export script performs a client-side hardware capability check before submitting any cloud job. The check is in `qai_hub_models/utils/qai_hub_helpers.py → raise_if_fp_is_unsupported()`.

The QCS6490's Hexagon v68 HTP **does not support FP16**. Newer Snapdragon chips (Hexagon v73/v75, found in Snapdragon 8 Gen 2/3) added FP16 support to the HTP. The v68 is INT8/INT16 only on the DSP.

The `mediapipe_hand` export script only exposes `--precision {float}` (no INT8 option). So:
- Float precision requires FP16 → device doesn't have it → blocked
- INT8 precision isn't offered → can't select it

This validation happens entirely on the client before any job is submitted. The `qai_hub` cloud API itself would accept the job.

### Why not just patch the check?

Even if we monkey-patched `raise_if_fp_is_unsupported` to a no-op, the export script would still compile a float32 model targeting QNN context binary. The QNN compiler on the cloud would then reject it because the Hexagon v68 HTP requires quantized (INT8) graphs. The validation is correct — it just doesn't offer a path forward.

### Solution

Use the low-level `qai_hub` Python API directly, bypassing `qai_hub_models` entirely. This allows us to:
1. Compile the model to ONNX first (intermediate format for quantization)
2. Apply INT8 post-training quantization
3. Recompile the quantized model to QNN context binary

---

## Step-by-Step Pipeline with Every Error

### Overview of the 3-step pipeline

```
TorchScript (.pt)
        │
        │  hub.submit_compile_job(options="--target_runtime onnx")
        ▼
ONNX model
        │
        │  hub.submit_quantize_job(weights=INT8, activations=INT8)
        ▼
INT8 QDQ ONNX (Quantize-Dequantize format)
        │
        │  hub.submit_compile_job(options="--target_runtime qnn_context_binary ...")
        ▼
QNN context binary (.bin)  ←  runs on Hexagon v68 HTP
```

---

### Attempt 1: GPU compute unit with QNN context binary

**What we tried:**
```python
options="--target_runtime qnn_context_binary --compute_unit gpu"
```

**Rationale:** The Adreno 643 GPU on QCS6490 supports FP16. If we route computation to the GPU instead of the HTP, float models should work.

**Error (from cloud compile job):**
```
QNN context binaries can be compiled only for compute units: ['all', 'npu', 'npu,cpu']
```

**Why:** QNN context binary is a format specifically for the Hexagon NPU/HTP. The QNN compiler simply doesn't support compiling context binaries for GPU — that's a different runtime path (Adreno uses a different backend). GPU acceleration in QNN uses `libQnnGpu.so` with a different model format.

**Fix:** Drop the GPU path entirely. Use ONNX runtime for a CPU-based fallback, or do proper INT8 quantization for the HTP.

---

### Attempt 2: ONNX runtime (intermediate success)

**What we tried:**
```python
options="--target_runtime onnx"
```

**Why:** ONNX Runtime on Qualcomm Linux runs on the CPU in FP32, which is fully supported. No FP16 restriction. This proved the low-level API path worked end-to-end.

**Result:** ✅ Compile succeeded. Downloaded `mediapipe_hand_hand_detector.bin.onnx.zip`.

However this does **not** use the HTP. It runs on the Arm CPU — functional but not the goal.

---

### Attempt 3: Unicode crash in PowerShell while waiting

**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '⏳' in position 4
```

**Why:** The `qai_hub` SDK prints a live status line using emoji — specifically the hourglass ⏳ (`U+23F3`). PowerShell on Windows defaults to code page 1252 (Western European), which cannot encode that character. The crash happened mid-wait, after the compile job was already submitted to the cloud.

**Fix:** Set `PYTHONUTF8=1` environment variable before running Python, which forces all I/O to UTF-8:
```powershell
$env:PYTHONUTF8 = "1"; python script.py
```

Also added a programmatic fix at the top of the script as a fallback:
```python
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
```

The cloud job (`jp3qlxrx5`) was already running — we resumed it by job ID rather than resubmitting.

---

### Attempt 4: INT8 quantization — wrong calibration data shape

**What we tried:**
```python
sample_inputs = model.sample_inputs()  # returns {"image": [np.ndarray]}
hub.submit_quantize_job(model=onnx_model, calibration_data=sample_inputs, ...)
```

**Error (from cloud quantize job):**
```
Input 'image' in calibration data set has shape (1, 256, 256, 3) but expected (1, 3, 256, 256).
```

**Why:** `model.sample_inputs()` returns arrays in **NHWC** format `(1, 256, 256, 3)` because that's the model's preferred channel-last layout. But the ONNX model compiled in Step 1 (without `--force_channel_last_input`) uses standard PyTorch **NCHW** format `(1, 3, 256, 256)`. The shapes don't match.

**Fix:** Transpose calibration data from NHWC → NCHW before uploading:
```python
raw_inputs = model.sample_inputs()
sample_inputs = {
    k: [arr.transpose(0, 3, 1, 2) if arr.ndim == 4 else arr for arr in v]
    for k, v in raw_inputs.items()
}
```

---

### Attempt 5: --quantize_io is not a quantize job option

**What we tried:**
```python
hub.submit_quantize_job(..., options="--quantize_io")
```

**Rationale:** The compiled QNN context binary later complained that the input tensor was still float. We assumed `--quantize_io` would force quantization of the I/O tensors in the quantize step.

**Error:**
```
qai_hub.client.UserError: unrecognized arguments: --quantize_io
```

**Why:** `--quantize_io` is a **compile job** option, not a quantize job option. We discovered this by reading `qai_hub_models/utils/base_model.py → get_hub_compile_options()`:

```python
if precision.activations_type is not None:
    compile_options += " --quantize_io"
```

The quantize job produces a QDQ ONNX model with quantized weights and internal activations, but the graph's input/output tensors remain float32. The `--quantize_io` flag in the *compile* step tells the QNN compiler to add explicit quantize/dequantize ops at the graph boundary, making the I/O INT8-compatible.

**Fix:** Remove `--quantize_io` from the quantize job. Add it to the QNN compile job options instead (see next section).

---

### Attempt 6: Float I/O rejected by QNN compiler

**Error (from cloud QNN compile job):**
```
Tensor 'image' has a floating-point type which is not supported by the targeted device.
Please quantize the model including its I/O and try again.
```

**Why:** After fixing the quantize step, the resulting QDQ ONNX model has INT8 weights and activations internally, but the graph's input tensor `image` is still declared as `float32`. The QNN HTP compiler for a device without FP16 support rejects any float tensor at the I/O boundary.

**Fix:** Add `--quantize_io` to the QNN compile job options. This instructs the compiler to insert a QuantizeLinear node at the input (converting incoming float → INT8 using the calibrated scale/zero-point) and a DequantizeLinear node at the output (converting INT8 → float). The user of the model provides float input; the conversion happens inside the compiled graph.

---

### Attempt 7: NPU-only compute unit also rejects float I/O

**What we tried:**
```python
options="--target_runtime qnn_context_binary --compute_unit npu --quantize_io ..."
```

**Error:**
```
Tensor 'image' has a floating-point type which is not supported by the targeted device.
```

**Why:** With `--compute_unit npu`, every operation — including the I/O quantize/dequantize nodes added by `--quantize_io` — must run on the HTP. The HTP cannot run float32. Even the boundary conversion nodes need to be on a compute unit that supports float.

**Fix:** Use `--compute_unit all`, which allows the compiler to assign the float-boundary conversion nodes to the CPU while keeping all quantized ops on the HTP. In practice this means a tiny amount of work runs on CPU (the input scale/zero-point conversion), while the entire model body runs on the HTP.

---

### Attempt 8: Missing compile options — output names and context graph name

Reading `base_model.py → get_hub_compile_options()` in full revealed two more required options:

```python
compile_options += f" --output_names {','.join(self.get_output_names())}"
# ...
compile_options += f" --force_channel_last_input {channel_last_inputs}"
# ...
compile_options += f" --qnn_options context_enable_graphs={context_graph_name}"
```

**`--output_names`**: tells the compiler which output tensors to expose. Without this, the QNN compiler may expose intermediary tensors or name outputs differently, breaking downstream parsing.

**`--force_channel_last_input image`**: tells the compiler to accept NCHW input from the caller and internally reorder to NHWC before the first layer. This matches what the HTP prefers for convolutional workloads. Without it, the model would accept NCHW input and the HTP would work on a sub-optimal memory layout.

**`--qnn_options context_enable_graphs=<name>`**: required for ahead-of-time (AoT) compiled QNN context binaries. The context binary format bundles one or more named graphs; this option tells the compiler which graph to activate. Without it, the runtime cannot load the binary correctly.

**Final working compile options for HandDetector:**
```
--target_runtime qnn_context_binary
--compute_unit all
--quantize_io
--output_names box_coords,box_scores
--force_channel_last_input image
--qnn_options context_enable_graphs=hand_detector
```

---

### Attempt 9: Status check API mismatch

**Error:**
```python
if onnx_compile_job.get_status().code.name != "SUCCESS":
AttributeError: 'str' object has no attribute 'name'
```

**Why:** `get_status()` returns a `JobStatus` dataclass. Its `.code` field is a `State` enum, not a string. Calling `.name` on it (Python enum attribute) would have worked, but `.code` was being compared as a string from a previous misread of the API.

**Fix:** Use the provided boolean properties instead:
```python
if not job.get_status().success:
    raise RuntimeError(...)
```

`JobStatus` exposes `.success`, `.failure`, `.pending`, and `.running` as direct boolean properties.

---

## Final Working Pipeline

### Complete flow

```python
import torch
import qai_hub as hub
from qai_hub_models.models.mediapipe_hand.model import HandDetector, HandLandmarkDetector

device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit", "1.6")

# Load PyTorch models
model = HandDetector.from_pretrained()   # or HandLandmarkDetector

# STEP 1 — Compile TorchScript → ONNX
traced = torch.jit.trace(model.eval(), torch.zeros(1, 3, 256, 256))
onnx_job = hub.submit_compile_job(
    model=traced,
    device=device,
    input_specs={"image": (1, 3, 256, 256)},
    options="--target_runtime onnx",
)
onnx_job.wait()
onnx_model = onnx_job.get_target_model()

# STEP 2 — Quantize ONNX → INT8 QDQ ONNX
# Calibration data must be NCHW to match the ONNX model's input layout.
raw = model.sample_inputs()
calibration = {
    k: [arr.transpose(0, 3, 1, 2) if arr.ndim == 4 else arr for arr in v]
    for k, v in raw.items()
}
quant_job = hub.submit_quantize_job(
    model=onnx_model,
    calibration_data=calibration,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
quant_job.wait()
quantized_model = quant_job.get_target_model()

# STEP 3 — Compile INT8 QDQ ONNX → QNN context binary
output_names = ",".join(model.get_output_names())
channel_last = ",".join(model.get_channel_last_inputs())
qnn_job = hub.submit_compile_job(
    model=quantized_model,
    device=device,
    input_specs={"image": (1, 3, 256, 256)},
    options=(
        "--target_runtime qnn_context_binary"
        " --compute_unit all"
        " --quantize_io"
        f" --output_names {output_names}"
        f" --force_channel_last_input {channel_last}"
        f" --qnn_options context_enable_graphs={component_name}"
    ),
)
qnn_job.wait()
qnn_model = qnn_job.get_target_model()
qnn_model.download("output.bin")
```

### AI Hub job reference

| Component | Step 1 (ONNX) | Step 2 (Quantize) | Step 3 (QNN bin) | Profile |
|---|---|---|---|---|
| hand_detector | jp3qlxrx5 | jpr1momvg | jpe408075 | jp8wd6nzp |
| hand_landmark_detector | jgkrwo1y5 | jpvzye87g | jgzvq86zp | jp3qlxyx5 |

---

## Deployment

### Runtime requirements on the QCS6490 device

The QNN runtime libraries are part of the Qualcomm AI Stack, typically pre-installed at `/opt/qcom/aistack/qairt/<version>/`:

| Library | Purpose |
|---|---|
| `libQnnHtp.so` | Hexagon HTP (DSP) execution backend |
| `libQnnHtpPrepared.so` | Required companion for pre-compiled context binaries |
| `qnn-net-run` | CLI tool for running inference from raw binary inputs |

### Running with qnn-net-run

```bash
# Input: float32 raw binary, NHWC layout (1 × 256 × 256 × 3)
# The --quantize_io compile option added a float→INT8 node at the graph input,
# so qnn-net-run receives float and the conversion happens inside the binary.

qnn-net-run \
  --model mediapipe_hand_hand_detector.bin \
  --backend /opt/qcom/aistack/qairt/*/lib/aarch64-oe-linux-gcc11/libQnnHtp.so \
  --input_list input_list.txt \
  --output_dir ./output
```

Input preparation:
```python
import cv2, numpy as np
frame = cv2.imread("hand.jpg")
img = cv2.resize(frame, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img.tofile("input.raw")   # shape: (256, 256, 3) — HWC, no batch dim for raw file
```

---

## Key Decisions Summary

| Decision | Why |
|---|---|
| Use low-level `qai_hub` API instead of `qai_hub_models` export script | The export script has a client-side FP16 validation that blocks QCS6490; bypassing it requires the lower-level API |
| INT8 quantization (not float) | Hexagon v68 HTP is INT8/INT16 only. Float inference would require CPU fallback, defeating the purpose of the HTP |
| 3-step pipeline (compile → quantize → compile) instead of 1-step | `qai_hub` does not support direct quantization from TorchScript; it requires an intermediate ONNX representation for the quantize job |
| `--compute_unit all` instead of `npu` | `--compute_unit npu` rejects float32 tensors even at the I/O boundary. `all` allows the float→INT8 boundary conversion nodes to run on CPU while the model body runs on HTP |
| `--quantize_io` on the compile step, not the quantize step | `--quantize_io` is a QNN compiler directive that inserts quantize/dequantize boundary nodes during compilation. It is not a quantize job option |
| Transpose calibration data NHWC → NCHW | `model.sample_inputs()` returns channel-last arrays matching the model's preferred layout, but the intermediate ONNX model (compiled without `--force_channel_last_input`) expects channel-first |
| `--force_channel_last_input image` on the QNN compile step | The HTP executes convolutional layers most efficiently in NHWC. This option inserts a layout-reorder at the input so callers provide NCHW and the hardware uses NHWC internally |
| Profile jobs submitted asynchronously | Profile jobs require physical device availability in AI Hub's device cloud and can queue; they don't block the download of the compiled binary |
| Used existing job IDs on re-runs | Cloud compile and quantize jobs are idempotent once complete. Resuming by job ID avoids re-uploading multi-MB model files and re-running minute-long cloud jobs |
