"""End-to-end mediapipe-hand demo using the direct-QNN ctypes shim.

Runs the palm detector + landmark detector back-to-back on a webcam stream
(or a static image), prints per-stage latency, and writes an annotated
output frame.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qnn_runtime import QnnModel, QnnRuntime


DEFAULT_BACKEND = "/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib/libQnnHtp.so"
DEFAULT_SYSTEM  = "/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib/libQnnSystem.so"
HERE = Path(__file__).resolve().parent
DETECTOR_BIN  = HERE.parent / "mediapipe_hand_hand_detector.bin"
LANDMARK_BIN  = HERE.parent / "mediapipe_hand_hand_landmark_detector.bin"

DET_INPUT_SIZE    = 256   # palm detector H == W
PALM_SCORE_THRESH = 0.5   # minimum sigmoid score to keep a detection
IOU_THRESH        = 0.3   # NMS overlap threshold
PALM_MARGIN       = 0.4   # fractional margin grown around the detected box

_ANCHORS: np.ndarray | None = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_anchors() -> np.ndarray:
    """Generate BlazePalm SSD anchors for a 256×256 input.

    Standard config: strides [8,16,16,16], fixed_anchor_size=True,
    interpolated_scale_aspect_ratio=1.0 → 2 anchors per grid cell → 3584 total.

    If the exported model outputs a different anchor count (e.g. 2944) the
    caller falls back to _decode_boxes_no_anchor which works without the
    exact anchor priors.
    """
    global _ANCHORS
    if _ANCHORS is not None:
        return _ANCHORS
    strides = [8, 16, 16, 16]
    min_scale, max_scale = 0.1484375, 0.75
    num_layers = len(strides)
    rows: list[list[float]] = []
    for i, stride in enumerate(strides):
        fmap = DET_INPUT_SIZE // stride
        scale = min_scale + (max_scale - min_scale) * i / (num_layers - 1)
        next_scale = (min_scale + (max_scale - min_scale) * (i + 1) / (num_layers - 1)
                      if i + 1 < num_layers else 1.0)
        for y in range(fmap):
            for x in range(fmap):
                cx = (x + 0.5) / fmap
                cy = (y + 0.5) / fmap
                rows.append([cx, cy, 1.0, 1.0])                          # base anchor
                rows.append([cx, cy, float(np.sqrt(scale * next_scale)),  # interp anchor
                              float(np.sqrt(scale * next_scale))])
    _ANCHORS = np.array(rows, dtype=np.float32)
    return _ANCHORS


def _decode_boxes(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """SSD decode: (N,18) raw + (N,4) anchors → (N,4) [x1,y1,x2,y2] in [0,1].

    raw[:,0] = dy offset, raw[:,1] = dx offset (pixel units for DET_INPUT_SIZE).
    raw[:,2] = height,    raw[:,3] = width      (pixel units).
    anchors[:,0] = cx, anchors[:,1] = cy in normalised [0,1].
    """
    s = float(DET_INPUT_SIZE)
    cx = anchors[:, 0] + raw[:, 1] / s
    cy = anchors[:, 1] + raw[:, 0] / s
    w  = raw[:, 3] / s
    h  = raw[:, 2] / s
    return np.stack([cx - w * 0.5, cy - h * 0.5,
                     cx + w * 0.5, cy + h * 0.5], axis=1)


def _decode_boxes_no_anchor(raw: np.ndarray) -> np.ndarray:
    """Fallback decoder used when the model anchor count doesn't match _build_anchors.

    Heuristic: if the top-anchor's raw values are already in [0,1] (model bakes
    in decoding), use them directly.  Otherwise they're pixel-unit offsets —
    divide by DET_INPUT_SIZE to normalise, centring on 0.5.
    """
    top = int(np.argmax(np.abs(raw[:, 2])))  # pick anchor with largest predicted height
    scale = 1.0 if np.all(np.abs(raw[top, :4]) < 2.0) else 1.0 / float(DET_INPUT_SIZE)
    cy = raw[:, 0] * scale
    cx = raw[:, 1] * scale
    h  = np.abs(raw[:, 2] * scale)
    w  = np.abs(raw[:, 3] * scale)
    # When treating as already-decoded, cy/cx are already centres; when treating
    # as pixel offsets from a unit anchor at 0.5, add the base.
    if scale < 1.0:
        cx = cx + 0.5
        cy = cy + 0.5
    return np.stack([cx - w * 0.5, cy - h * 0.5,
                     cx + w * 0.5, cy + h * 0.5], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Simple greedy NMS.  Returns indices into boxes/scores of kept detections."""
    order = np.argsort(scores)[::-1].tolist()
    kept: list[int] = []
    while order:
        i = order[0]
        kept.append(i)
        a = boxes[i]
        surviving = []
        for j in order[1:]:
            b = boxes[j]
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            iou = inter / (area_a + area_b - inter + 1e-6)
            if iou < IOU_THRESH:
                surviving.append(j)
        order = surviving
    return kept


def _palm_crop(canvas: np.ndarray, box_norm: np.ndarray,
               out_size: int = 224) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the palm region from the letterboxed canvas and resize for the landmark model.

    box_norm: [x1,y1,x2,y2] in [0,1] within the 256×256 canvas.
    Returns (NHWC float32 RGB, (px1,py1,px2,py2) pixel coords inside canvas).
    """
    H, W = canvas.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_norm]
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    bw = x2 - x1; bh = y2 - y1
    if bw < 0.01 or bh < 0.01:
        # Degenerate box: fall back to full canvas.
        rgb = cv2.cvtColor(
            cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        return rgb[None, ...], (0, 0, W, H)
    side = max(bw, bh) * (1.0 + PALM_MARGIN)
    cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
    x1n = max(0.0, cx - side * 0.5); y1n = max(0.0, cy - side * 0.5)
    x2n = min(1.0, cx + side * 0.5); y2n = min(1.0, cy + side * 0.5)
    px1 = int(x1n * W); py1 = int(y1n * H)
    px2 = int(x2n * W); py2 = int(y2n * H)
    crop = canvas[py1:py2, px1:px2]
    if crop.size == 0:
        crop = canvas; px1, py1, px2, py2 = 0, 0, W, H
    rgb = cv2.cvtColor(
        cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    return rgb[None, ...], (px1, py1, px2, py2)


def preprocess(frame: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 -> NHWC float32 [0,1] with letterbox padding.

    The QNN binary has quant params scale=1/255, offset=0, so feeding RGB
    floats in [0,1] maps cleanly to uint8 [0,255].
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    return rgb[None, ...], canvas  # (1,256,256,3) and letterboxed BGR view


def overlay_landmarks(img: np.ndarray, landmarks: np.ndarray,
                      crop_px: tuple[int, int, int, int]) -> None:
    """Draw 21 landmarks on `img` mapping from landmark-space [0,1] to canvas pixels.

    crop_px: (px1, py1, px2, py2) — the crop rectangle inside the canvas.
    landmarks shape: (21, 3) with xy in [0,1] relative to the crop.
    """
    px1, py1, px2, py2 = crop_px
    cw = px2 - px1; ch = py2 - py1
    pts = []
    for i in range(landmarks.shape[0]):
        x = int(px1 + landmarks[i, 0] * cw)
        y = int(py1 + landmarks[i, 1] * ch)
        pts.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20), # pinky
        (5, 9), (9, 13), (13, 17),             # palm arch
    ]
    for a, b in edges:
        cv2.line(img, pts[a], pts[b], (255, 200, 0), 1)


def run_once(detector: QnnModel, landmark: QnnModel, frame: np.ndarray) -> dict:
    """Single-frame pipeline. Returns timing dict and annotated BGR image."""
    t0 = time.perf_counter()
    det_input, canvas = preprocess(frame, size=DET_INPUT_SIZE)
    t1 = time.perf_counter()

    det_outs = detector.execute([det_input])
    det_by_name = {m.name: o for m, o in zip(detector.outputs, det_outs)}
    t2 = time.perf_counter()

    scores_logits = det_by_name["box_scores"].reshape(-1)       # (N,)
    coords        = det_by_name["box_coords"].reshape(-1, 18)   # (N, 18)
    scores = sigmoid(scores_logits)

    # Decode SSD anchors → normalised [x1,y1,x2,y2] boxes.
    anchors = _build_anchors()
    boxes = (_decode_boxes(coords, anchors)
             if len(anchors) == len(scores)
             else _decode_boxes_no_anchor(coords))

    # Filter by confidence then apply NMS.
    mask = scores > PALM_SCORE_THRESH
    if not np.any(mask):
        top = int(np.argmax(scores))
        mask = np.zeros(len(scores), dtype=bool)
        mask[top] = True
    full_indices = np.where(mask)[0]
    kept = _nms(boxes[mask], scores[mask])
    best_idx = full_indices[kept[0]]
    top_score = float(scores[best_idx])
    best_box  = boxes[best_idx]   # [x1,y1,x2,y2] in [0,1] canvas space

    # Crop detected palm region and feed to landmark model.
    lm_size = landmark.inputs[0].shape[1]   # H of (1,H,W,3) — typically 224 or 256
    lm_input, crop_px = _palm_crop(canvas, best_box, out_size=lm_size)

    lm_outs = landmark.execute([lm_input])
    lm_by_name = {m.name: o for m, o in zip(landmark.outputs, lm_outs)}
    t3 = time.perf_counter()

    landmarks_xyz = lm_by_name["landmarks"].reshape(21, 3)
    handedness    = float(lm_by_name["lr"].reshape(-1)[0])
    lm_score      = float(lm_by_name["scores"].reshape(-1)[0])

    # Annotate: draw palm box + skeleton.
    out = canvas.copy()
    H, W = canvas.shape[:2]
    bx1 = int(np.clip(best_box[0], 0.0, 1.0) * W)
    by1 = int(np.clip(best_box[1], 0.0, 1.0) * H)
    bx2 = int(np.clip(best_box[2], 0.0, 1.0) * W)
    by2 = int(np.clip(best_box[3], 0.0, 1.0) * H)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 128, 255), 1)
    overlay_landmarks(out, landmarks_xyz, crop_px)
    label = (f"palm={top_score:.2f} lm={lm_score:.2f} "
             f"hand={'R' if handedness > 0.5 else 'L'}")
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {
        "preprocess_ms":  (t1 - t0) * 1000,
        "detector_ms":    (t2 - t1) * 1000,
        "landmark_ms":    (t3 - t2) * 1000,
        "top_palm_score": top_score,
        "landmark_score": lm_score,
        "annotated":      out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",        type=str, default=None,
                    help="Static image path; if omitted, opens /dev/video0")
    ap.add_argument("--video-index",  type=int, default=0)
    ap.add_argument("--frames",       type=int, default=30)
    ap.add_argument("--out",          type=str, default="hand_demo_out.jpg")
    ap.add_argument("--backend",      type=str, default=DEFAULT_BACKEND)
    ap.add_argument("--system",       type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--detector-bin", type=str, default=str(DETECTOR_BIN))
    ap.add_argument("--landmark-bin", type=str, default=str(LANDMARK_BIN))
    args = ap.parse_args()

    print("loading shim + models …")
    rt = QnnRuntime(backend_so=args.backend, system_so=args.system)
    detector = QnnModel(rt, args.detector_bin)
    landmark = QnnModel(rt, args.landmark_bin)
    print("  detector inputs:",  detector.inputs)
    print("  detector outputs:", detector.outputs)
    print("  landmark inputs:",  landmark.inputs)
    print("  landmark outputs:", landmark.outputs)

    anchors = _build_anchors()
    n_model = detector.outputs[0].num_elements  # rough check on first output
    print(f"  anchor count (generated)={len(anchors)}  "
          f"landmark input size={landmark.inputs[0].shape[1]}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"could not read {args.image}", file=sys.stderr); sys.exit(1)
        run_once(detector, landmark, frame)   # warmup
        ts = [run_once(detector, landmark, frame) for _ in range(args.frames)]
        cv2.imwrite(args.out, ts[-1]["annotated"])
        det_ms = np.mean([t["detector_ms"]  for t in ts])
        lm_ms  = np.mean([t["landmark_ms"]  for t in ts])
        pp_ms  = np.mean([t["preprocess_ms"] for t in ts])
        total  = pp_ms + det_ms + lm_ms
        print(f"avg over {args.frames} runs:  pre={pp_ms:.2f}ms  det={det_ms:.2f}ms  "
              f"lm={lm_ms:.2f}ms  total={total:.2f}ms  ({1000.0/total:.1f} fps)")
        print(f"wrote {args.out}")
        return

    cap = cv2.VideoCapture(args.video_index)
    if not cap.isOpened():
        print(f"could not open /dev/video{args.video_index}", file=sys.stderr); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        print("failed to grab warmup frame", file=sys.stderr); sys.exit(1)
    run_once(detector, landmark, frame)   # warmup

    ts = []
    last_annot = None
    i = 0
    print("press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = run_once(detector, landmark, frame)
        ts.append(r)
        last_annot = r["annotated"]
        print(f"[{i:03d}] det={r['detector_ms']:.1f}ms lm={r['landmark_ms']:.1f}ms "
              f"palm={r['top_palm_score']:.2f} lm_score={r['landmark_score']:.2f}")
        cv2.imshow("hand", last_annot)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    if last_annot is not None:
        cv2.imwrite(args.out, last_annot)
        print(f"wrote {args.out}")
    if ts:
        det_ms = np.mean([t["detector_ms"]  for t in ts])
        lm_ms  = np.mean([t["landmark_ms"]  for t in ts])
        pp_ms  = np.mean([t["preprocess_ms"] for t in ts])
        total  = pp_ms + det_ms + lm_ms
        print(f"avg over {len(ts)} frames:  pre={pp_ms:.2f}ms  det={det_ms:.2f}ms  "
              f"lm={lm_ms:.2f}ms  total={total:.2f}ms  ({1000.0/total:.1f} fps)")


if __name__ == "__main__":
    main()
