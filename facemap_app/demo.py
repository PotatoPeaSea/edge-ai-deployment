"""Face landmarking demo on QCS6490 — two models, both on the Hexagon DSP.

Pipeline (all inference on HTP via the in-process SNPE shim):

  frame ──► face detector (Ultra-Light RFB-320, w8a8 DLC)
            └─ scores + boxes ─► threshold + NMS ─► face box
  face box ─► crop+resize 128 ─► facemap_3dmm (w8a8 DLC)
            └─ 265 params ─► 3DMM reconstruction ─► 68 landmarks

The facemap_3dmm output decode follows Qualcomm's reference
(qai_hub_models/models/facemap_3dmm/utils.py):
  [  0:219] alpha_id  (shape)        [219:258] alpha_exp (expression)
  [258] pitch [259] yaw [260] roll   [261] tX [262] tY [263] focal
68 landmarks are reconstructed from meanFace + shapeBasis + blendShape.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# ---- models -----------------------------------------------------------------
FACE_DLC_PREP = HERE / "facemap_3dmm_prepared.dlc"
FACE_DLC      = HERE / "facemap_3dmm.dlc"
DET_DLC_PREP  = HERE / "face_det_w8a8_prepared.dlc"
DET_DLC       = HERE / "face_det_w8a8.dlc"
SHIM_SO       = HERE / "libsnpe_shim.so"

# ---- 3DMM basis assets ------------------------------------------------------
MEAN_FACE   = HERE / "meanFace.npy"
SHAPE_BASIS = HERE / "shapeBasis.npy"
BLEND_SHAPE = HERE / "blendShape.npy"

FACE_IN = 128                       # facemap input size
DET_W, DET_H = 320, 240            # detector input (WxH)
VERTEX_NUM, ALPHA_ID, ALPHA_EXP = 68, 219, 39

from snpe_runtime import SnpeModel  # noqa: E402

_DET = None
_FACE = None
_BASIS = None


def _get_detector() -> SnpeModel:
    global _DET
    if _DET is None:
        use_prep = DET_DLC_PREP.exists()
        dlc = str(DET_DLC_PREP) if use_prep else str(DET_DLC)
        _DET = SnpeModel(dlc, use_dsp=True, accelerated_init=use_prep,
                         output_names=["scores", "boxes"])
    return _DET


def _get_facemap() -> SnpeModel:
    global _FACE
    if _FACE is None:
        use_prep = FACE_DLC_PREP.exists()
        dlc = str(FACE_DLC_PREP) if use_prep else str(FACE_DLC)
        _FACE = SnpeModel(dlc, use_dsp=True, accelerated_init=use_prep)
    return _FACE


def _get_basis():
    global _BASIS
    if _BASIS is None:
        face = np.load(MEAN_FACE).reshape(3 * VERTEX_NUM, 1).astype(np.float64)
        bid = np.load(SHAPE_BASIS).reshape(3 * VERTEX_NUM, ALPHA_ID).astype(np.float64)
        bexp = np.load(BLEND_SHAPE).reshape(3 * VERTEX_NUM, ALPHA_EXP).astype(np.float64)
        _BASIS = (face, bid, bexp)
    return _BASIS


# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------

def preprocess_det(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> NCHW float32 (1,3,240,320), RGB, (x-127)/128."""
    img = cv2.resize(frame_bgr, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.0) / 128.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Plain NMS on pixel boxes (x1,y1,x2,y2). Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def detect_faces(frame_bgr: np.ndarray, score_thresh: float = 0.7,
                 iou_thresh: float = 0.3, max_faces: int = 5
                 ) -> tuple[list[tuple[int, int, int, int, float]], float]:
    """Return ([(x0,y0,x1,y1,score), ...], latency_ms) in image pixel coords."""
    h, w = frame_bgr.shape[:2]
    det = _get_detector()
    x = preprocess_det(frame_bgr)
    t0 = time.perf_counter()
    outs = det.execute([x])
    ms = (time.perf_counter() - t0) * 1000.0
    od = {t.name: o for t, o in zip(det.outputs, outs)}
    face_p = od["scores"].reshape(-1, 2)[:, 1]
    boxes = od["boxes"].reshape(-1, 4)            # normalized x1,y1,x2,y2
    keep = face_p > score_thresh
    face_p, boxes = face_p[keep], boxes[keep]
    if len(boxes) == 0:
        return [], ms
    px = boxes * np.array([w, h, w, h], dtype=np.float32)
    idx = _nms(px, face_p, iou_thresh)[:max_faces]
    out = []
    for i in idx:
        x0, y0, x1, y1 = px[i]
        x0 = int(np.clip(x0, 0, w - 1)); x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1)); y1 = int(np.clip(y1, 0, h - 1))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1, float(face_p[i])))
    return out, ms


# ---------------------------------------------------------------------------
# FaceMap 3DMM landmarks
# ---------------------------------------------------------------------------

def preprocess_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the face box and produce facemap input: RGB float32 [0,1] 128x128 HWC."""
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1 + 1, x0:x1 + 1]
    img = cv2.resize(crop, (FACE_IN, FACE_IN), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)


def _rot(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Rotation matrix matching the reference project_landmark()."""
    p = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # flip Y,Z (about X by pi)
    cz, sz = np.cos(-roll), np.sin(-roll)
    roll_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    yaw_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cx, sx = np.cos(-pitch), np.sin(-pitch)
    pitch_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return yaw_m @ (pitch_m @ (p @ roll_m))


def project_landmark(params: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """265 params -> (68,2) landmarks in crop-centered coords + (pitch,yaw,roll) deg."""
    face, basis_id, basis_exp = _get_basis()
    p = params.astype(np.float64)
    alpha_id = p[0:219] * 3.0
    alpha_exp = p[219:258] * 0.5 + 0.5
    pitch = p[258] * np.pi / 2
    yaw = p[259] * np.pi / 2
    roll = p[260] * np.pi / 2
    tX = p[261] * 60.0
    tY = p[262] * 60.0
    tZ = 500.0
    f = p[263] * 150.0 + 450.0

    r = _rot(pitch, yaw, roll)
    shp = (face + basis_id @ alpha_id[:, None] + basis_exp @ alpha_exp[:, None])
    verts = shp.reshape(VERTEX_NUM, 3) @ r.T
    verts[:, 0] += tX
    verts[:, 1] += tY
    verts[:, 2] += tZ
    lmk = verts[:, 0:2] * np.array([f, f]) / tZ
    deg = (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))
    return lmk, deg


def transform_landmark(lmk: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Map crop-centered landmarks back to original image pixel coords."""
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    out = lmk.copy()
    out[:, 0] = (lmk[:, 0] + FACE_IN / 2) * width / FACE_IN + x0
    out[:, 1] = (lmk[:, 1] + FACE_IN / 2) * height / FACE_IN + y0
    return out


def landmarks_for_box(frame_bgr, box):
    """Run facemap on one face box -> (image-space (68,2) landmarks, pyr deg, ms)."""
    face = _get_facemap()
    inp = preprocess_face(frame_bgr, box)
    t0 = time.perf_counter()
    params = face.execute([inp])[0].reshape(-1)
    ms = (time.perf_counter() - t0) * 1000.0
    lmk, pyr = project_landmark(params)
    return transform_landmark(lmk, box), pyr, ms


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(frame_bgr, faces, scale_ref=720.0):
    """Draw face boxes + 68 landmarks per face on a copy of the frame."""
    out = frame_bgr.copy()
    s = max(1, int(round(out.shape[0] / scale_ref * 2)))
    r = max(1, int(round(out.shape[0] / scale_ref * 2)))
    for (box, lmk, pyr, score) in faces:
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), s)
        for (px, py) in lmk.astype(int):
            cv2.circle(out, (int(px), int(py)), r, (0, 0, 255), -1)
        cv2.putText(out, f"{score:.2f} P{pyr[0]:.0f} Y{pyr[1]:.0f} R{pyr[2]:.0f}",
                    (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (0, 255, 0), max(1, s // 2))
    return out


def process_frame(frame_bgr, score_thresh=0.7):
    """Detect faces then landmark each. Returns (faces, det_ms, face_ms_total)."""
    dets, det_ms = detect_faces(frame_bgr, score_thresh=score_thresh)
    faces = []
    face_ms = 0.0
    for (x0, y0, x1, y1, score) in dets:
        lmk, pyr, ms = landmarks_for_box(frame_bgr, (x0, y0, x1, y1))
        face_ms += ms
        faces.append(((x0, y0, x1, y1), lmk, pyr, score))
    return faces, det_ms, face_ms


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _try_open(idx: int):
    """Try a camera index across backends; return (cap, h, w) if it yields a
    real frame, else None. Fails fast — never blocks on a dead node."""
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            frame = None
            for _ in range(3):
                ok, f = cap.read()
                if ok and f is not None and f.ndim == 3:
                    frame = f
                    break
            if frame is not None and frame.shape[0] >= 64 and frame.shape[1] >= 64:
                return cap, frame.shape[0], frame.shape[1]
        cap.release()
    return None


def open_camera(preferred: int | None = None) -> cv2.VideoCapture:
    """Open a working camera that actually delivers frames.

    Camera enumeration on the QCS6490 is flaky (internal MSM nodes + the USB
    webcam share /dev/video*), so we probe several indices/backends and keep the
    first that returns a valid frame."""
    candidates = [preferred] if preferred is not None else [2, 0, 1, 3, 4]
    for idx in candidates:
        res = _try_open(idx)
        if res is not None:
            cap, h, w = res
            print(f"camera: using index {idx} ({w}x{h})")
            return cap
    tried = ", ".join(str(c) for c in candidates)
    print(f"ERROR: no working camera (tried index {tried}).", file=sys.stderr)
    print("       List devices:  v4l2-ctl --list-devices ; then pass --video-index N",
          file=sys.stderr)
    sys.exit(1)


def _have_display() -> bool:
    """True only if a GUI window can actually be created (X/Wayland present)."""
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.namedWindow("FaceMap landmarks", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("FaceMap landmarks")
        return True
    except cv2.error:
        return False


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------

def run_live(video_index, score_thresh, seconds=None, live_jpg="facemap_live.jpg"):
    # Build both DSP graphs up front so the first-frame init isn't mistaken for
    # a hang (graph init takes a few seconds per model).
    print("initializing detector + facemap on the DSP (builds graphs, ~a few s) …",
          flush=True)
    _get_detector()
    _get_facemap()
    print("models ready", flush=True)

    cap = open_camera(video_index)
    gui = _have_display()
    writer = None
    if gui:
        print("live mode — press 'q' in the window (or Ctrl-C) to quit", flush=True)
    else:
        print("no display detected -> HEADLESS mode:", flush=True)
        print(f"  • latest annotated frame written to {live_jpg} (re-open to refresh)",
              flush=True)
        print("  • recording to facemap_live.mp4 ; press Ctrl-C to stop", flush=True)

    ema = None
    t_start = time.time()
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            faces, det_ms, face_ms = process_frame(frame, score_thresh)
            total = det_ms + face_ms
            ema = total if ema is None else 0.9 * ema + 0.1 * total
            fps = 1000.0 / ema if ema else 0.0
            out = annotate(frame, faces)
            cv2.putText(out, f"{len(faces)} face  det {det_ms:.1f} + lmk {face_ms:.1f} ms"
                        f"  {fps:.0f}fps", (8, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            n += 1
            if gui:
                cv2.imshow("FaceMap landmarks", out)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                cv2.imwrite(live_jpg, out)
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter("facemap_live.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             15.0, (w, h))
                writer.write(out)
                if n % 15 == 0:
                    print(f"  frame {n}: {len(faces)} face(s)  {total:.1f}ms  {fps:.0f}fps",
                          flush=True)
            if seconds is not None and (time.time() - t_start) >= seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if not gui:
            print(f"\nstopped after {n} frames. View {live_jpg} or facemap_live.mp4 "
                  "(scp them off the device).", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DSP face detection + 68-landmark demo")
    ap.add_argument("--image", type=str, default=None,
                    help="Input image path. If omitted, uses the webcam")
    ap.add_argument("--video-index", type=int, default=None,
                    help="Camera index. Omit to auto-detect (USB cam is usually 2)")
    ap.add_argument("--live", action="store_true",
                    help="Force continuous webcam mode (default when no --image)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Headless live: stop after N seconds (default: until Ctrl-C)")
    ap.add_argument("--out", type=str, default="facemap_out.jpg")
    ap.add_argument("--score-thresh", type=float, default=0.7)
    ap.add_argument("--frames", type=int, default=5,
                    help="Frames to average latency over (image mode)")
    ap.add_argument("--check", action="store_true",
                    help="Automated check mode: print PASS/FAIL and exit")
    args = ap.parse_args()

    for p in (FACE_DLC, MEAN_FACE, SHAPE_BASIS, BLEND_SHAPE):
        if not p.exists():
            print(f"ERROR: missing required asset {p}", file=sys.stderr)
            sys.exit(1)

    # Webcam (no --image) -> continuous live mode. Over SSH/no-display this runs
    # headless and writes facemap_live.jpg + .mp4 instead of opening a window.
    if args.live or not args.image:
        run_live(args.video_index, args.score_thresh, seconds=args.seconds)
        return

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)

    print("warmup …")
    faces, det_ms, face_ms = process_frame(frame, args.score_thresh)
    print(f"  detected {len(faces)} face(s)  det={det_ms:.1f}ms  lmk={face_ms:.1f}ms")

    det_times, lmk_times = [], []
    for i in range(args.frames):
        faces, det_ms, face_ms = process_frame(frame, args.score_thresh)
        det_times.append(det_ms)
        lmk_times.append(face_ms)
        print(f"  [{i+1}/{args.frames}] det={det_ms:.1f}ms  lmk={face_ms:.1f}ms")

    det_avg = float(np.mean(det_times))
    lmk_avg = float(np.mean(lmk_times))
    total = det_avg + lmk_avg
    print(f"\nfaces={len(faces)}  detector={det_avg:.1f}ms  landmarks={lmk_avg:.1f}ms"
          f"  total={total:.1f}ms  ({1000.0/total:.1f} fps)" if total else "")
    if faces:
        (box, lmk, pyr, score) = faces[0]
        print(f"face0 box={box} score={score:.3f} PYR={tuple(round(v,1) for v in pyr)}")
        print(f"landmarks[0:3]=\n{lmk[:3].round(1)}")

    out_img = annotate(frame, faces)
    cv2.imwrite(args.out, out_img)
    print(f"wrote {args.out}")

    if args.check:
        ok = (len(faces) >= 1
              and faces[0][1].shape == (VERTEX_NUM, 2)
              and np.isfinite(faces[0][1]).all())
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
