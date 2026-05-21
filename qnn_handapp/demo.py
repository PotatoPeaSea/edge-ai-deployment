"""End-to-end mediapipe-hand demo using the direct-QNN ctypes shim.

Runs the palm detector + landmark detector back-to-back on a webcam stream
(or a static image), prints per-stage latency, and writes an annotated
output frame.

The box-decoding step (turning 2944 anchors into pixel boxes) is left as a
placeholder per the README's "Tune output parsing" follow-up — we just take
the highest-scoring anchor and crop a centred 256x256 region. The point of
this demo is the direct-QNN inference path, not the SSD post-processing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from qnn_runtime import QnnModel, QnnRuntime


DEFAULT_BACKEND = "/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib/libQnnHtp.so"
DEFAULT_SYSTEM  = "/data/local/tmp/snpeexample/aarch64-ubuntu-gcc9.4/lib/libQnnSystem.so"
HERE = Path(__file__).resolve().parent
DETECTOR_BIN  = HERE.parent / "mediapipe_hand_hand_detector.bin"
LANDMARK_BIN  = HERE.parent / "mediapipe_hand_hand_landmark_detector.bin"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(frame: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 -> NHWC float32 normalized to [0,1], with letterbox padding.

    The exported QNN binary has input quant params scale=1/255, offset=0
    (UFIXED_POINT_8), i.e. the dequantized range is [0.0, 1.0]. Feeding RGB
    floats in that range maps to uint8 [0, 255] cleanly with no clipping.
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
    return rgb[None, ...], canvas  # (1,256,256,3) and the letterboxed BGR view


def overlay_landmarks(img: np.ndarray, landmarks: np.ndarray,
                      origin_xy: tuple[int, int] = (0, 0),
                      crop_size: int = 256) -> None:
    """Draw 21 landmarks on `img`. landmarks shape (21, 3) in [0,1]."""
    pts = []
    ox, oy = origin_xy
    for i in range(landmarks.shape[0]):
        x = int(ox + landmarks[i, 0] * crop_size)
        y = int(oy + landmarks[i, 1] * crop_size)
        pts.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    edges = [
        (0,1),(1,2),(2,3),(3,4),       # thumb
        (0,5),(5,6),(6,7),(7,8),       # index
        (0,9),(9,10),(10,11),(11,12),  # middle
        (0,13),(13,14),(14,15),(15,16),# ring
        (0,17),(17,18),(18,19),(19,20),# pinky
        (5,9),(9,13),(13,17),          # palm
    ]
    for a, b in edges:
        cv2.line(img, pts[a], pts[b], (255, 200, 0), 1)


def run_once(detector: QnnModel, landmark: QnnModel, frame: np.ndarray):
    """Single-frame pipeline. Returns timing dict and annotated BGR image."""
    t0 = time.perf_counter()
    det_input, canvas = preprocess(frame, size=256)
    t1 = time.perf_counter()

    det_outs = detector.execute([det_input])
    det_by_name = {m.name: o for m, o in zip(detector.outputs, det_outs)}
    t2 = time.perf_counter()

    scores_logits = det_by_name["box_scores"].reshape(-1)
    coords        = det_by_name["box_coords"].reshape(-1, 18)
    scores = sigmoid(scores_logits)
    top = int(np.argmax(scores))
    top_score = float(scores[top])

    # Stub: skip anchor decode, just crop the centred 256x256 region the
    # detector already saw.  See README "Next Steps" for proper decoding.
    lm_input = det_input

    lm_outs = landmark.execute([lm_input])
    lm_by_name = {m.name: o for m, o in zip(landmark.outputs, lm_outs)}
    t3 = time.perf_counter()

    landmarks_xyz = lm_by_name["landmarks"].reshape(21, 3)
    handedness    = float(lm_by_name["lr"].reshape(-1)[0])
    lm_score      = float(lm_by_name["scores"].reshape(-1)[0])

    out = canvas.copy()
    overlay_landmarks(out, landmarks_xyz, origin_xy=(0, 0), crop_size=256)
    label = f"palm={top_score:.2f} lm={lm_score:.2f} hand={'R' if handedness > 0.5 else 'L'}"
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {
        "preprocess_ms": (t1 - t0) * 1000,
        "detector_ms":   (t2 - t1) * 1000,
        "landmark_ms":   (t3 - t2) * 1000,
        "top_palm_score": top_score,
        "landmark_score": lm_score,
        "annotated": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None,
                    help="Static image path; if omitted, opens /dev/video0")
    ap.add_argument("--video-index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", type=str, default="hand_demo_out.jpg")
    ap.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    ap.add_argument("--system",  type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--detector-bin", type=str, default=str(DETECTOR_BIN))
    ap.add_argument("--landmark-bin", type=str, default=str(LANDMARK_BIN))
    args = ap.parse_args()

    print(f"loading shim + models …")
    rt = QnnRuntime(backend_so=args.backend, system_so=args.system)
    detector = QnnModel(rt, args.detector_bin)
    landmark = QnnModel(rt, args.landmark_bin)
    print("  detector inputs:",  detector.inputs)
    print("  detector outputs:", detector.outputs)
    print("  landmark inputs:",  landmark.inputs)
    print("  landmark outputs:", landmark.outputs)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"could not read {args.image}", file=sys.stderr)
            sys.exit(1)
        # Warmup
        run_once(detector, landmark, frame)
        # Timed runs
        ts = []
        for _ in range(args.frames):
            r = run_once(detector, landmark, frame)
            ts.append(r)
        last = ts[-1]
        cv2.imwrite(args.out, last["annotated"])
        det_ms = np.mean([t["detector_ms"] for t in ts])
        lm_ms  = np.mean([t["landmark_ms"]  for t in ts])
        pp_ms  = np.mean([t["preprocess_ms"] for t in ts])
        total  = pp_ms + det_ms + lm_ms
        print(f"avg over {args.frames} runs:  pre={pp_ms:.2f}ms  det={det_ms:.2f}ms  "
              f"lm={lm_ms:.2f}ms  total={total:.2f}ms  ({1000.0/total:.1f} fps)")
        print(f"wrote {args.out}")
        return

    # Webcam mode
    cap = cv2.VideoCapture(args.video_index)
    if not cap.isOpened():
        print(f"could not open /dev/video{args.video_index}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Warmup
    ok, frame = cap.read()
    if not ok:
        print("failed to grab warmup frame", file=sys.stderr); sys.exit(1)
    run_once(detector, landmark, frame)

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
              f"palm_score={r['top_palm_score']:.2f}")
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
        det_ms = np.mean([t["detector_ms"] for t in ts])
        lm_ms  = np.mean([t["landmark_ms"]  for t in ts])
        pp_ms  = np.mean([t["preprocess_ms"] for t in ts])
        total  = pp_ms + det_ms + lm_ms
        print(f"avg over {len(ts)} frames:  pre={pp_ms:.2f}ms  det={det_ms:.2f}ms  "
              f"lm={lm_ms:.2f}ms  total={total:.2f}ms  ({1000.0/total:.1f} fps)")


if __name__ == "__main__":
    main()
