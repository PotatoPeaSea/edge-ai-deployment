"""Eye Aspect Ratio (EAR) demo — blink / eye-close detector on QCS6490 HTP.

Same pipeline as facemap_app:
  frame ──► Ultra-Light RFB-320 face detector (w8a8 DLC, 320×240 NCHW)
            └─ boxes + scores ─► NMS ─► face box
  face box ─► crop 128×128 ─► FaceMap 3DMM (w8a8 DLC)
            └─ 265 params ─► 3DMM reconstruction ─► 68 landmarks (ibug order)

EAR added on top using ibug 68-point eye indices:
  Right eye: 36-41   Left eye: 42-47
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Displays per-eye EAR values and a "EYE CLOSE DETECTED" banner when either
eye's EAR falls below --ear-threshold (default 0.20).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from snpe_runtime import SnpeModel  # noqa: E402

# ── model / asset paths (all flat in the same directory after deploy) ─────────
FACE_DLC_PREP = HERE / "facemap_3dmm_prepared.dlc"
FACE_DLC      = HERE / "facemap_3dmm.dlc"
DET_DLC_PREP  = HERE / "face_det_w8a8_prepared.dlc"
DET_DLC       = HERE / "face_det_w8a8.dlc"
MEAN_FACE     = HERE / "meanFace.npy"
SHAPE_BASIS   = HERE / "shapeBasis.npy"
BLEND_SHAPE   = HERE / "blendShape.npy"

# ── model constants ────────────────────────────────────────────────────────────
FACE_IN   = 128           # facemap crop input (H == W)
DET_W     = 320           # detector input width
DET_H     = 240           # detector input height
VERTEX_NUM, ALPHA_ID, ALPHA_EXP = 68, 219, 39

# ── EAR ───────────────────────────────────────────────────────────────────────
# ibug/dlib 68-point eye landmark indices: [p1, p2, p3, p4, p5, p6]
#   p1/p4 = horizontal corners,  p2/p3 = upper lid,  p5/p6 = lower lid
_RIGHT_EYE_IDX = [36, 37, 38, 39, 40, 41]
_LEFT_EYE_IDX  = [42, 43, 44, 45, 46, 47]
EAR_CLOSE_THRESH = 0.20

# ── lazy-loaded singletons ────────────────────────────────────────────────────
_DET   = None
_FACE  = None
_BASIS = None


def _get_detector() -> SnpeModel:
    global _DET
    if _DET is None:
        use_prep = DET_DLC_PREP.exists()
        _DET = SnpeModel(str(DET_DLC_PREP if use_prep else DET_DLC),
                         use_dsp=True, accelerated_init=use_prep,
                         output_names=["scores", "boxes"])
    return _DET


def _get_facemap() -> SnpeModel:
    global _FACE
    if _FACE is None:
        use_prep = FACE_DLC_PREP.exists()
        _FACE = SnpeModel(str(FACE_DLC_PREP if use_prep else FACE_DLC),
                          use_dsp=True, accelerated_init=use_prep)
    return _FACE


def _get_basis():
    global _BASIS
    if _BASIS is None:
        face = np.load(MEAN_FACE).reshape(3 * VERTEX_NUM, 1).astype(np.float64)
        bid  = np.load(SHAPE_BASIS).reshape(3 * VERTEX_NUM, ALPHA_ID).astype(np.float64)
        bexp = np.load(BLEND_SHAPE).reshape(3 * VERTEX_NUM, ALPHA_EXP).astype(np.float64)
        _BASIS = (face, bid, bexp)
    return _BASIS


# ── face detector ──────────────────────────────────────────────────────────────

def preprocess_det(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → NCHW float32 (1,3,240,320), RGB, (x-127)/128."""
    img = cv2.resize(frame_bgr, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.0) / 128.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area  = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def detect_faces(frame_bgr: np.ndarray, score_thresh=0.7, iou_thresh=0.3,
                 max_faces=5) -> tuple[list, float]:
    """Return ([(x0,y0,x1,y1,score),...], latency_ms) in image pixel coords."""
    h, w = frame_bgr.shape[:2]
    det  = _get_detector()
    x    = preprocess_det(frame_bgr)
    t0   = time.perf_counter()
    outs = det.execute([x])
    ms   = (time.perf_counter() - t0) * 1000.0
    od   = {t.name: o for t, o in zip(det.outputs, outs)}
    face_p = od["scores"].reshape(-1, 2)[:, 1]
    boxes  = od["boxes"].reshape(-1, 4)
    keep   = face_p > score_thresh
    face_p, boxes = face_p[keep], boxes[keep]
    if len(boxes) == 0:
        return [], ms
    px  = boxes * np.array([w, h, w, h], dtype=np.float32)
    idx = _nms(px, face_p, iou_thresh)[:max_faces]
    out = []
    for i in idx:
        x0, y0, x1, y1 = px[i]
        x0 = int(np.clip(x0, 0, w - 1)); x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1)); y1 = int(np.clip(y1, 0, h - 1))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1, float(face_p[i])))
    return out, ms


# ── FaceMap 3DMM landmarks ─────────────────────────────────────────────────────

def preprocess_face(frame_bgr: np.ndarray,
                    box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop face box → RGB float32 [0,1] 128×128 HWC."""
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1 + 1, x0:x1 + 1]
    img  = cv2.resize(crop, (FACE_IN, FACE_IN), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)


def _rot(pitch: float, yaw: float, roll: float) -> np.ndarray:
    p  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    cz, sz = np.cos(-roll),  np.sin(-roll)
    roll_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    cy, sy = np.cos(-yaw),   np.sin(-yaw)
    yaw_m  = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cx, sx = np.cos(-pitch), np.sin(-pitch)
    pit_m  = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return yaw_m @ (pit_m @ (p @ roll_m))


def project_landmark(params: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """265 params → (68,2) crop-centred landmarks + (pitch,yaw,roll) degrees."""
    face, bid, bexp = _get_basis()
    p         = params.astype(np.float64)
    alpha_id  = p[0:219] * 3.0
    alpha_exp = p[219:258] * 0.5 + 0.5
    pitch = p[258] * np.pi / 2;  yaw = p[259] * np.pi / 2;  roll = p[260] * np.pi / 2
    tX = p[261] * 60.0;  tY = p[262] * 60.0;  tZ = 500.0
    f  = p[263] * 150.0 + 450.0
    r   = _rot(pitch, yaw, roll)
    shp = face + bid @ alpha_id[:, None] + bexp @ alpha_exp[:, None]
    v   = shp.reshape(VERTEX_NUM, 3) @ r.T
    v[:, 0] += tX;  v[:, 1] += tY;  v[:, 2] += tZ
    lmk = v[:, 0:2] * np.array([f, f]) / tZ
    return lmk, (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))


def transform_landmark(lmk: np.ndarray,
                       box: tuple[int, int, int, int]) -> np.ndarray:
    """Map crop-centred landmarks to original image pixel coords."""
    x0, y0, x1, y1 = box
    w = x1 - x0 + 1;  h = y1 - y0 + 1
    out = lmk.copy()
    out[:, 0] = (lmk[:, 0] + FACE_IN / 2) * w / FACE_IN + x0
    out[:, 1] = (lmk[:, 1] + FACE_IN / 2) * h / FACE_IN + y0
    return out


def landmarks_for_box(frame_bgr, box):
    """Run facemap on one box → (image-space (68,2) landmarks, pyr_deg, ms)."""
    face  = _get_facemap()
    inp   = preprocess_face(frame_bgr, box)
    t0    = time.perf_counter()
    params = face.execute([inp])[0].reshape(-1)
    ms    = (time.perf_counter() - t0) * 1000.0
    lmk, pyr = project_landmark(params)
    return transform_landmark(lmk, box), pyr, ms


# ── EAR ───────────────────────────────────────────────────────────────────────

def _ear(lmk: np.ndarray, indices: list[int]) -> float:
    """Eye Aspect Ratio from 6 2-D landmark points [p1..p6].

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p  = lmk[indices, :2]
    v1 = float(np.linalg.norm(p[1] - p[5]))
    v2 = float(np.linalg.norm(p[2] - p[4]))
    h  = float(np.linalg.norm(p[0] - p[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


# ── annotation ────────────────────────────────────────────────────────────────

def annotate(frame_bgr, faces, ear_thresh=EAR_CLOSE_THRESH, scale_ref=720.0):
    """Draw face boxes, 68 landmarks, per-eye EAR, and eye-close banner."""
    out = frame_bgr.copy()
    s   = max(1, int(round(out.shape[0] / scale_ref * 2)))
    r   = max(1, int(round(out.shape[0] / scale_ref * 2)))
    H, W = out.shape[:2]

    for (box, lmk, pyr, score) in faces:
        x0, y0, x1, y1 = box

        # Face bounding box.
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), s)

        # 68 landmark dots.
        for (px, py) in lmk.astype(int):
            cv2.circle(out, (int(px), int(py)), r, (0, 0, 255), -1)

        # EAR values.
        ear_r    = _ear(lmk, _RIGHT_EYE_IDX)
        ear_l    = _ear(lmk, _LEFT_EYE_IDX)
        r_closed = ear_r < ear_thresh
        l_closed = ear_l < ear_thresh

        # Highlight closed-eye landmark dots in orange.
        for idx in _RIGHT_EYE_IDX:
            if r_closed:
                cv2.circle(out, (int(lmk[idx, 0]), int(lmk[idx, 1])),
                           r + 1, (0, 140, 255), -1)
        for idx in _LEFT_EYE_IDX:
            if l_closed:
                cv2.circle(out, (int(lmk[idx, 0]), int(lmk[idx, 1])),
                           r + 1, (0, 140, 255), -1)

        # HUD: score + pose + EAR per eye (above the face box).
        r_col = (0, 60, 255) if r_closed else (180, 255, 180)
        l_col = (0, 60, 255) if l_closed else (180, 255, 180)
        hud = (f"P{pyr[0]:.0f} Y{pyr[1]:.0f} R{pyr[2]:.0f}  "
               f"score={score:.2f}")
        cv2.putText(out, hud, (x0, max(0, y0 - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 255, 0), max(1, s // 2))
        cv2.putText(out, f"R-EAR:{ear_r:.3f}", (x0, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, r_col, max(1, s // 2))
        cv2.putText(out, f"L-EAR:{ear_l:.3f}", (x0 + 90, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, l_col, max(1, s // 2))

        # Centred banner when an eye is closed.
        if r_closed and l_closed:
            msg = "BOTH EYES CLOSED"
        elif r_closed:
            msg = "RIGHT EYE CLOSED"
        elif l_closed:
            msg = "LEFT EYE CLOSED"
        else:
            msg = None

        if msg:
            font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, 2
            (tw, th), _ = cv2.getTextSize(msg, font, fscale, thick)
            tx = (W - tw) // 2;  ty = H // 2
            cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                          (0, 0, 180), -1)
            cv2.putText(out, msg, (tx, ty), font, fscale, (255, 255, 255), thick)

    return out


# ── frame pipeline ─────────────────────────────────────────────────────────────

def process_frame(frame_bgr, score_thresh=0.7):
    """Detect + landmark all faces. Returns (faces, det_ms, lmk_ms_total)."""
    dets, det_ms = detect_faces(frame_bgr, score_thresh=score_thresh)
    faces = [];  lmk_ms = 0.0
    for (x0, y0, x1, y1, score) in dets:
        lmk, pyr, ms = landmarks_for_box(frame_bgr, (x0, y0, x1, y1))
        lmk_ms += ms
        faces.append(((x0, y0, x1, y1), lmk, pyr, score))
    return faces, det_ms, lmk_ms


# ── camera helpers (from facemap_app) ─────────────────────────────────────────

def _try_open(idx: int):
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            for _ in range(3):
                ok, f = cap.read()
                if ok and f is not None and f.ndim == 3 and f.shape[0] >= 64:
                    print(f"camera: using index {idx} ({f.shape[1]}x{f.shape[0]})")
                    return cap
        cap.release()
    return None


def open_camera(preferred=None):
    candidates = [preferred] if preferred is not None else [2, 0, 1, 3, 4]
    for idx in candidates:
        cap = _try_open(idx)
        if cap is not None:
            return cap
    print(f"ERROR: no working camera (tried {candidates})", file=sys.stderr)
    print("       List devices:  v4l2-ctl --list-devices ; pass --video-index N",
          file=sys.stderr)
    sys.exit(1)


def _have_display() -> bool:
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.namedWindow("EAR", cv2.WINDOW_NORMAL); cv2.destroyWindow("EAR")
        return True
    except cv2.error:
        return False


# ── live loop ─────────────────────────────────────────────────────────────────

def run_live(video_index, score_thresh, ear_thresh, seconds=None,
             live_jpg="ear_live.jpg"):
    print("initializing detector + facemap on the DSP …", flush=True)
    _get_detector(); _get_facemap()
    print("models ready", flush=True)

    cap = open_camera(video_index)
    gui = _have_display()
    writer = None
    if not gui:
        print("headless mode: writing ear_live.jpg + ear_live.mp4", flush=True)

    ema = None;  t_start = time.time();  n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            faces, det_ms, lmk_ms = process_frame(frame, score_thresh)
            total = det_ms + lmk_ms
            ema   = total if ema is None else 0.9 * ema + 0.1 * total
            fps   = 1000.0 / ema if ema else 0.0
            out   = annotate(frame, faces, ear_thresh=ear_thresh)
            cv2.putText(out,
                        f"{len(faces)} face  det {det_ms:.1f}+lmk {lmk_ms:.1f}ms  {fps:.0f}fps",
                        (8, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            n += 1
            if gui:
                cv2.imshow("EAR demo", out)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                cv2.imwrite(live_jpg, out)
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter("ear_live.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             15.0, (w, h))
                writer.write(out)
                if n % 15 == 0:
                    ears = [(f"R{_ear(lmk,_RIGHT_EYE_IDX):.2f}"
                             f"/L{_ear(lmk,_LEFT_EYE_IDX):.2f}")
                            for _, lmk, _, _ in faces]
                    print(f"  frame {n}: {len(faces)} face(s) "
                          f"{' '.join(ears)}  {total:.1f}ms  {fps:.0f}fps",
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Face EAR blink detector on QCS6490 HTP")
    ap.add_argument("--image",         type=str,   default=None)
    ap.add_argument("--video-index",   type=int,   default=None)
    ap.add_argument("--live",          action="store_true")
    ap.add_argument("--seconds",       type=float, default=None)
    ap.add_argument("--out",           type=str,   default="ear_out.jpg")
    ap.add_argument("--score-thresh",  type=float, default=0.7)
    ap.add_argument("--ear-threshold", type=float, default=EAR_CLOSE_THRESH,
                    help="EAR below this → eye closed (default 0.20)")
    ap.add_argument("--frames",        type=int,   default=5)
    args = ap.parse_args()

    for p in (FACE_DLC, MEAN_FACE, SHAPE_BASIS, BLEND_SHAPE):
        if not p.exists():
            print(f"ERROR: missing required asset {p}", file=sys.stderr)
            sys.exit(1)

    if args.live or not args.image:
        run_live(args.video_index, args.score_thresh,
                 args.ear_threshold, seconds=args.seconds)
        return

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr); sys.exit(1)

    print("warmup …")
    faces, det_ms, lmk_ms = process_frame(frame, args.score_thresh)
    print(f"  {len(faces)} face(s)  det={det_ms:.1f}ms  lmk={lmk_ms:.1f}ms")

    det_times, lmk_times = [], []
    for i in range(args.frames):
        faces, det_ms, lmk_ms = process_frame(frame, args.score_thresh)
        det_times.append(det_ms); lmk_times.append(lmk_ms)

    det_avg = float(np.mean(det_times)); lmk_avg = float(np.mean(lmk_times))
    total   = det_avg + lmk_avg
    print(f"det={det_avg:.1f}ms  lmk={lmk_avg:.1f}ms  "
          f"total={total:.1f}ms  ({1000.0/total:.1f} fps)")

    for i, (box, lmk, pyr, score) in enumerate(faces):
        ear_r = _ear(lmk, _RIGHT_EYE_IDX)
        ear_l = _ear(lmk, _LEFT_EYE_IDX)
        print(f"face{i}  score={score:.3f}  R-EAR={ear_r:.3f}  L-EAR={ear_l:.3f}  "
              f"PYR=({pyr[0]:.1f},{pyr[1]:.1f},{pyr[2]:.1f})")

    out_img = annotate(frame, faces, ear_thresh=args.ear_threshold)
    cv2.imwrite(args.out, out_img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
