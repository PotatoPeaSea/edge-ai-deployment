"""Build INT8 calibration raws for the Ultra-Light face detector, and (if
onnxruntime is present) print a reference detection so we can validate the
preprocessing/decoding before trusting the quantized DLC.

Detector preprocessing (MUST match runtime in demo.py):
    resize to 320x240 (WxH) -> RGB -> (x-127)/128 -> NCHW float32
"""
import glob
import os
import sys

import cv2
import numpy as np

W, H = 320, 240
HERE = os.path.dirname(os.path.abspath(__file__))


def preprocess_det(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.0) / 128.0
    chw = np.transpose(rgb, (2, 0, 1))          # 3,240,320
    return np.ascontiguousarray(chw, dtype=np.float32)


def build_calib():
    raw_dir = os.path.join(HERE, "calib_raw")
    os.makedirs(raw_dir, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(HERE, "calib_src", "*.jpg")))
    lines = []
    for s in srcs:
        bgr = cv2.imread(s)
        if bgr is None:
            continue
        arr = preprocess_det(bgr)
        out = os.path.join(raw_dir, os.path.splitext(os.path.basename(s))[0] + ".raw")
        arr.tofile(out)
        lines.append(out)
    list_path = os.path.join(HERE, "calib_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} calibration raws -> {list_path}")
    return list_path


def reference(image_path: str):
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[skip] onnxruntime not available ({e}); device validation only")
        return
    sess = ort.InferenceSession(os.path.join(HERE, "version-RFB-320.onnx"),
                                providers=["CPUExecutionProvider"])
    bgr = cv2.imread(image_path)
    inp = preprocess_det(bgr)[None]             # 1,3,240,320
    scores, boxes = sess.run(["scores", "boxes"], {"input": inp})
    scores, boxes = scores[0], boxes[0]         # (4420,2),(4420,4)
    face_p = scores[:, 1]
    keep = face_p > 0.7
    print(f"reference {os.path.basename(image_path)}: {keep.sum()} dets > 0.7")
    idx = np.argsort(-face_p)[:5]
    for i in idx:
        print(f"  p={face_p[i]:.3f} box(norm)={boxes[i].round(3)}")


if __name__ == "__main__":
    build_calib()
    ref_img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "calib_src", "obama.jpg")
    if os.path.exists(ref_img):
        reference(ref_img)
