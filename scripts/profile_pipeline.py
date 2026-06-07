"""
PRISM — Pipeline Profiler
Reads 20 frames from the bag and times each component individually.
Run this to find the bottleneck.
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BAG = sys.argv[1] if len(sys.argv) > 1 else "/home/koushik-test/Downloads/test1-003.db3"
N_FRAMES = 20

print(f"\nProfiling {N_FRAMES} frames from: {BAG}\n")

# ── 1. Bag reading speed ───────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Bag reading speed")
from prism.sensory_core.ros2_bag_loader import ROS2BagLoader
loader = ROS2BagLoader(BAG)
frames = []
t0 = time.perf_counter()
for f in loader.iter_frames(max_frames=N_FRAMES):
    if f["image"] is not None:
        frames.append(f)
read_ms = (time.perf_counter() - t0) * 1000
print(f"  Read {len(frames)} frames in {read_ms:.0f}ms  → {read_ms/len(frames):.0f}ms/frame")

if not frames:
    print("No frames found!"); sys.exit(1)

img   = frames[0]["image"]
cloud = frames[0]["point_cloud"]
print(f"  Image shape: {img.shape}  |  Cloud points: {len(cloud) if cloud is not None else 0}")

# ── 2. YOLO detection ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2: YOLO detection (yolov8n, device=auto)")
from prism.utils.common import get_device
device = get_device("mps")
print(f"  Device: {device}")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Warmup
model(img, device=device, verbose=False)

times = []
for f in frames[:N_FRAMES]:
    t = time.perf_counter()
    model(f["image"], conf=0.35, iou=0.45, imgsz=640,
          classes=[0,1,2,3,5,7,9,11], device=device, verbose=False)
    times.append((time.perf_counter() - t) * 1000)

print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")

# ── 3. Optical flow ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3a: Raw Farneback at FULL resolution (1920x1080)")
gray_prev = cv2.cvtColor(frames[0]["image"], cv2.COLOR_BGR2GRAY)
times = []
for f in frames[1:N_FRAMES]:
    gray = cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY)
    t = time.perf_counter()
    cv2.calcOpticalFlowFarneback(gray_prev, gray, None,
                                  0.5, 3, 15, 3, 5, 1.2, 0)
    times.append((time.perf_counter() - t) * 1000)
    gray_prev = gray
print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")

print("\nTEST 3b: OpticalFlowEstimator (fixed — runs at 320x180, upsamples)")
from prism.sensory_core.core import OpticalFlowEstimator
flow_est = OpticalFlowEstimator()
flow_est.compute(frames[0]["image"])  # init prev frame
times = []
for f in frames[1:N_FRAMES]:
    t = time.perf_counter()
    flow_est.compute(f["image"])
    times.append((time.perf_counter() - t) * 1000)
print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")

# ── 4. LiDAR fast clustering ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4: FastLiDARProcessor")
from prism.sensory_core.lidar_processor import FastLiDARProcessor
lidar = FastLiDARProcessor()
lidar_frames = [f for f in frames if f["point_cloud"] is not None]
if lidar_frames:
    times = []
    for f in lidar_frames[:N_FRAMES]:
        t = time.perf_counter()
        lidar.process(f["point_cloud"])
        times.append((time.perf_counter() - t) * 1000)
    print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")
else:
    print("  No LiDAR frames found")

# ── 5. Depth Anything (relative) on CUDA ──────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 5: Depth Anything v2 Small on CUDA")
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image as PILImage

proc  = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
dmodel = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf").to(device)
dmodel.eval()

small = cv2.resize(img, (320, 180))
rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
inp   = proc(images=PILImage.fromarray(rgb), return_tensors="pt")
inp   = {k: v.to(device) for k, v in inp.items()}

# Warmup
with torch.no_grad():
    dmodel(**inp)

times = []
for _ in range(10):
    t = time.perf_counter()
    with torch.no_grad():
        dmodel(**inp)
    times.append((time.perf_counter() - t) * 1000)
print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")

# ── 6. Dashboard render ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 6: Dashboard render")
from prism.viz.dashboard import render_dashboard
from prism.utils.common import SensoryFrame

sf = SensoryFrame(frame_idx=1, timestamp=0.0, camera_name="CAM_FRONT", image=img)
fr = {"decision":"CRUISE","risk":0.1,"speed_mps":8.0,"n_cam_dets":2,
      "n_lidar_dets":1,"latency_ms":50,"frame_idx":1,
      "throttle":0.3,"brake":0.0,"sensory_frame":sf}
times = []
for _ in range(20):
    t = time.perf_counter()
    render_dashboard(img, fr, [])
    times.append((time.perf_counter() - t) * 1000)
print(f"  avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY — fix the slowest component first")
print("=" * 50 + "\n")
