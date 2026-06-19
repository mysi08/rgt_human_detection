#!/usr/bin/env python3
import openvino as ov
import torch, numpy, cv2

# Optional: Ultralytics YOLO
try:
    import ultralytics
    print("Ultralytics version:", ultralytics.__version__)
except ImportError:
    print("Ultralytics not installed")

# Optional: NNCF
try:
    import nncf
    print("NNCF version:", nncf.__version__)
except ImportError:
    print("NNCF not installed")

print("OpenVINO version:", ov.__version__)
print("PyTorch version:", torch.__version__)
print("NumPy version:", numpy.__version__)
print("OpenCV version:", cv2.__version__)


# import os
# import sys
# import time

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import numpy as np
# np.int = int
# np.float = float
# np.bool = bool
# np.complex = complex
# np.object = object
# np.str = str

# from model import Detector


# WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
# CKPT_PATH   = os.path.join(WEIGHTS_DIR, "lidar_human.pth")


# def make_fake_scan(n_pts=450, min_r=0.8, max_r=8.0):
#     scan = np.full((n_pts,), max_r, dtype=np.float32)

#     centers = [120, 230, 340]
#     widths = [10, 14, 12]
#     depths = [2.2, 3.8, 2.8]

#     for c, w, d in zip(centers, widths, depths):
#         left = max(0, c - w)
#         right = min(n_pts, c + w + 1)
#         xs = np.arange(left, right)
#         bump = np.exp(-0.5 * ((xs - c) / (0.35 * w + 1e-6)) ** 2)
#         local = max_r - (max_r - d) * bump
#         scan[left:right] = np.minimum(scan[left:right], local.astype(np.float32))

#     noise = np.random.normal(0.0, 0.01, size=n_pts).astype(np.float32)
#     scan = np.clip(scan + noise, min_r, max_r)

#     scan[20] = np.nan
#     scan[21] = np.inf
#     scan[22] = -np.inf
#     np.nan_to_num(scan, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
#     return scan


# def main():
#     print(f"Checkpoint: {CKPT_PATH}")
#     if not os.path.isfile(CKPT_PATH):
#         print("ERROR: chk.pth not found")
#         sys.exit(1)

#     try:
#         t0 = time.perf_counter()
#         detector = Detector(
#             CKPT_PATH,
#             gpu=False,
#             stride=4,
#             panoramic_scan=True,
#         )
#         t1 = time.perf_counter()
#         print(f"Detector loaded in {(t1 - t0) * 1000:.1f} ms")
#     except Exception as e:
#         print(f"ERROR: detector load failed: {e}")
#         sys.exit(2)

#     try:
#         detector.set_laser_fov(225.0)
#         print("Laser FOV set to 225.0 deg")
#     except Exception as e:
#         print(f"ERROR: set_laser_fov failed: {e}")
#         sys.exit(3)

#     scan = make_fake_scan(n_pts=450)

#     try:
#         t2 = time.perf_counter()
#         dets_xy, dets_cls, instance_mask = detector(scan)
#         t3 = time.perf_counter()
#     except Exception as e:
#         print(f"ERROR: detector inference failed: {e}")
#         sys.exit(4)

#     print(f"Inference time: {(t3 - t2) * 1000:.1f} ms")
#     print(f"dets_xy type: {type(dets_xy)} shape: {getattr(dets_xy, 'shape', None)}")
#     print(f"dets_cls type: {type(dets_cls)} shape: {getattr(dets_cls, 'shape', None)}")
#     print(f"instance_mask shape: {getattr(instance_mask, 'shape', None)}")

#     n_show = min(10, len(dets_cls))
#     print(f"Detections found: {len(dets_cls)}")
#     for i in range(n_show):
#         x, y = dets_xy[i]
#         s = dets_cls[i]
#         print(f"[{i:02d}] x={x:+.3f} y={y:+.3f} score={s:.4f}")

#     print("Verification finished.")


# if __name__ == "__main__":
#     main()