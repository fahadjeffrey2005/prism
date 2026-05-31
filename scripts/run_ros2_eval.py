"""
PRISM — Real-World Evaluation on ROS2 Bag Data
===============================================
Runs the full PRISM 6-layer pipeline on real sensor data captured in
ROS2 bag files (Velodyne LiDAR + USB camera).

No nuScenes. No simulation. Real data.

What this does:
    1. Reads camera frames + LiDAR scans from .db3 bag files
    2. Extracts USB camera intrinsics from /usb_cam/camera_info
    3. Runs SensoryCore (YOLO + Depth Anything) on each camera frame
    4. Runs LiDARProcessor (ground removal + DBSCAN) on each LiDAR scan
    5. Fuses camera and LiDAR detections
    6. Runs WorldModel → PredictiveEngine → ArbitrationCore → Planner
    7. Saves annotated video + metrics JSON

Bags available (auto-detected in prism/new data/):
    test1/test1.db3      — ~237s  camera + LiDAR + IMU
    test2/test12.db3     — ~71s   camera + LiDAR + IMU
    ros2_bag/ros2_bag.db3 — ~208s LiDAR + IMU only (no camera)

Usage:
    cd ~/prism
    python3 scripts/run_ros2_eval.py                     # runs test1 by default
    python3 scripts/run_ros2_eval.py --bag test2
    python3 scripts/run_ros2_eval.py --bag ros2_bag      # LiDAR-only mode
    python3 scripts/run_ros2_eval.py --bag all           # all bags sequentially
    python3 scripts/run_ros2_eval.py --no-vlm            # VLM ablation
    python3 scripts/run_ros2_eval.py --max-frames 200    # quick test run
    python3 scripts/run_ros2_eval.py --save-video        # write annotated mp4

Output:
    ~/prism_data/experiments/ros2_eval_<bag>_<timestamp>.json
    ~/prism_data/experiments/ros2_eval_<bag>_<timestamp>.mp4  (if --save-video)
"""

import sys
import json
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.sensory_core.ros2_bag_loader import ROS2BagLoader, USBCamIntrinsics
from prism.sensory_core.lidar_processor import LiDARProcessor, LiDARDetection

logger = get_logger("ROS2Eval")

# ── Bag registry ──────────────────────────────────────────────────────────────

BAGS_ROOT = Path(__file__).parent.parent / "prism" / "new data"

BAG_REGISTRY = {
    "test1":    Path.home() / "Downloads" / "test1-003.db3",
    "test2":    Path.home() / "Downloads" / "test12-002.db3",
    "ros2_bag": BAGS_ROOT  / "ros2_bag"  / "ros2_bag.db3",
}

# ── USB camera defaults (fallback if camera_info not in bag) ─────────────────
# Approximate values for a typical USB webcam at 640x480
USB_CAM_DEFAULT_K = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)
USB_CAM_DEFAULT_W = 640
USB_CAM_DEFAULT_H = 480

# ── Safety thresholds ─────────────────────────────────────────────────────────
COLLISION_THRESH_M = 2.0
NEAR_MISS_THRESH_M = 4.0


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(cfg: dict, vlm_enabled: bool = True) -> dict:
    """Construct and return PRISM pipeline dict — mirrors run_closedloop_eval."""
    from prism.sensory_core.core import SensoryCore
    from prism.sensory_core.metric_depth import MetricDepthEngine
    from prism.world_model.world_model import WorldModel
    from prism.predictive_engine.engine import PredictiveEngine
    from prism.predictive_engine.decision import SmartDecisionEngine
    from prism.semantic_reasoner.reasoner import SemanticReasoner
    from prism.arbitration.core import ArbitrationCore
    from prism.planner.planner import AdaptivePlanner

    if not vlm_enabled:
        cfg = dict(cfg)
        cfg["vlm"] = dict(cfg.get("vlm", {}))
        cfg["vlm"]["enabled"] = False

    return {
        "core":       SensoryCore(cfg),
        "depth":      MetricDepthEngine(cfg),
        "world":      WorldModel(cfg),
        "predictor":  PredictiveEngine(cfg),
        "decision":   SmartDecisionEngine(),
        "reasoner":   SemanticReasoner(cfg),
        "arbitrator": ArbitrationCore(cfg),
        "planner":    AdaptivePlanner(cfg),
    }


# ── Per-frame pipeline ────────────────────────────────────────────────────────

def run_frame(
    image: Optional[np.ndarray],
    lidar_dets: List[LiDARDetection],
    timestamp: float,
    frame_idx: int,
    pipeline: dict,
    calib: dict,
    camera_name: str = "usb_cam",
) -> dict:
    """
    Run one frame through the full 6-layer PRISM pipeline.
    Mirrors run_closedloop_eval.run_frame() exactly.
    """
    t0 = time.perf_counter()

    # ── Layer 1: SensoryCore ─────────────────────────────────────────────────
    sensory_frame = None
    metric_dets   = []

    if image is not None:
        sensory_frame = pipeline["core"].process(
            image, camera_name=camera_name, timestamp=timestamp
        )
        pipeline["depth"].update_intrinsics(calib)
        metric_dets, _ = pipeline["depth"].process_frame(
            image, sensory_frame.detections,
            run_model=sensory_frame.depth_map is not None,
        )

        # Inject LiDAR-only blobs that don't overlap camera detections
        for lidar_det in lidar_dets:
            if lidar_det.distance_m <= 0:
                continue
            overlaps = any(
                abs(cd.distance_m - lidar_det.distance_m) < 3.0 and
                abs(cd.lateral_m - lidar_det.lateral_m) < 2.0
                for cd in metric_dets
            )
            if not overlaps:
                sensory_frame.detections.append(
                    _lidar_to_detection(lidar_det, timestamp, frame_idx)
                )

    # ── Layer 2: WorldModel ───────────────────────────────────────────────────
    world_state = None
    if sensory_frame is not None:
        world_state = pipeline["world"].update(sensory_frame, calibration=calib)

    # ── Layer 3: PredictiveEngine ─────────────────────────────────────────────
    pred_state = None
    if world_state is not None:
        pred_state = pipeline["predictor"].update(world_state, metric_dets)

    # ── Layer 3b: SmartDecisionEngine ────────────────────────────────────────
    scene = None
    if world_state is not None and pred_state is not None:
        scene = pipeline["decision"].assess(world_state, pred_state, metric_dets)

    # ── Layer 4: SemanticReasoner (VLM) ───────────────────────────────────────
    vlm_out = None
    vlm_summary = ""
    vlm_intents = {}
    if image is not None and world_state is not None and pred_state is not None:
        vlm_out = pipeline["reasoner"].update(image, world_state, pred_state)
        if vlm_out:
            vlm_summary = getattr(vlm_out, "scene_summary", "")
            vlm_intents = getattr(vlm_out, "actor_intents", {})
            if vlm_out.actor_intents:
                pipeline["predictor"].update_vlm_intents(vlm_out.actor_intents)
            pipeline["arbitrator"].update_vlm(vlm_out.to_arb_dict())
            logger.info(
                f"[VLM fired @ frame {frame_idx}] "
                f"summary='{vlm_summary}' | "
                f"intents={vlm_intents}"
            )

    # ── Layer 5: ArbitrationCore ──────────────────────────────────────────────
    arb = None
    if pred_state is not None and scene is not None:
        arb = pipeline["arbitrator"].arbitrate(
            world_state, pred_state, scene, timestamp=timestamp
        )

    # ── Layer 6: AdaptivePlanner ──────────────────────────────────────────────
    control = None
    if arb is not None:
        control = pipeline["planner"].plan(arb, metric_dets=metric_dets, timestamp=timestamp)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "frame_idx":    frame_idx,
        "timestamp":    timestamp,
        "latency_ms":   latency_ms,
        "n_cam_dets":   len(metric_dets),
        "n_lidar_dets": len(lidar_dets),
        "n_fused":      len(metric_dets) + len(lidar_dets),
        "decision":     arb.action if arb else "UNKNOWN",
        "risk":         float(1.0 - arb.speed_factor) if arb else 0.0,
        "speed_mps":    float(control.target_speed_mps) if control else 0.0,
        "vlm_fired":    vlm_out is not None,
        "vlm_summary":  vlm_summary,
        "vlm_intents":  {str(k): v for k, v in vlm_intents.items()},
        "sensory_frame": sensory_frame,
        "control":      control,
        "arb":          arb,
        "world_state":  world_state,
        "image_bgr":    image,
    }


def _lidar_to_detection(lidar_det: LiDARDetection, timestamp: float, frame_idx: int):
    """Wrap a LiDARDetection as a Detection so SensoryFrame / WorldModel can use it."""
    from prism.utils.common import Detection, BBox2D

    # Synthetic bbox — centred in image, width proportional to lateral size
    dummy_bbox = BBox2D(x1=300, y1=200, x2=340, y2=280,
                        confidence=0.85, class_id=2, class_name="car")
    det = Detection(bbox=dummy_bbox)
    det.depth_estimate = lidar_det.distance_m   # real metric distance from LiDAR
    det.camera_name    = "lidar"
    det.frame_idx      = frame_idx
    det.timestamp      = timestamp
    return det


def run_frame_lidar_only(
    lidar_dets: List[LiDARDetection],
    timestamp: float,
    frame_idx: int,
    pipeline: dict,
) -> dict:
    """
    Run one LiDAR-only frame through the pipeline (no camera, no VLM).
    Builds a synthetic SensoryFrame from LiDAR detections and runs
    WorldModel → PredictiveEngine → SmartDecisionEngine → Arbitration → Planner.
    """
    from prism.utils.common import SensoryFrame

    t0 = time.perf_counter()

    # Build synthetic SensoryFrame from LiDAR detections
    sensory_frame = SensoryFrame(
        frame_idx  = frame_idx,
        timestamp  = timestamp,
        camera_name = "lidar_only",
    )
    sensory_frame.detections = [
        _lidar_to_detection(d, timestamp, frame_idx) for d in lidar_dets
    ]

    # WorldModel
    world_state = pipeline["world"].update(sensory_frame, calibration={})

    # PredictiveEngine
    pred_state = pipeline["predictor"].update(world_state, []) if world_state else None

    # SmartDecisionEngine
    scene = None
    if world_state and pred_state:
        scene = pipeline["decision"].assess(world_state, pred_state, [])

    # Arbitration
    arb = None
    if pred_state and scene:
        arb = pipeline["arbitrator"].arbitrate(
            world_state, pred_state, scene, timestamp=timestamp
        )

    # Planner
    control = None
    if arb:
        control = pipeline["planner"].plan(arb, metric_dets=[], timestamp=timestamp)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "frame_idx":    frame_idx,
        "timestamp":    timestamp,
        "latency_ms":   latency_ms,
        "n_cam_dets":   0,
        "n_lidar_dets": len(lidar_dets),
        "n_fused":      len(lidar_dets),
        "decision":     arb.action if arb else "UNKNOWN",
        "risk":         float(1.0 - arb.speed_factor) if arb else 0.0,
        "speed_mps":    float(control.target_speed_mps) if control else 0.0,
        "vlm_fired":    False,
        "vlm_summary":  "",
        "vlm_intents":  {},
        "sensory_frame": sensory_frame,
        "control":      control,
        "arb":          arb,
        "world_state":  world_state,
        "image_bgr":    None,
    }


# ── Annotation helper ─────────────────────────────────────────────────────────

# VLM text state — persists on screen for several frames after firing
_vlm_text_buffer: str = ""
_vlm_text_ttl: int = 0          # frames remaining to show VLM text
VLM_TEXT_HOLD_FRAMES = 60       # ~5 seconds at 12 fps


def _blend_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    """Draw a semi-transparent filled rectangle."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def _draw_risk_bar(img, x, y, w, h, risk, label="RISK"):
    """Draw a vertical risk bar on the right edge."""
    # Background
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    # Fill
    fill_h = int(h * risk)
    if fill_h > 0:
        r = int(255 * risk)
        g = int(255 * (1 - risk))
        cv2.rectangle(img, (x, y + h - fill_h), (x + w, y + h), (0, g, r), -1)
    # Label
    cv2.putText(img, label, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"{int(risk*100)}%", (x, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def annotate_frame(
    image: np.ndarray,
    frame_result: dict,
    lidar_dets: List[LiDARDetection],
) -> np.ndarray:
    """Draw impressive PRISM outputs onto the camera frame."""
    global _vlm_text_buffer, _vlm_text_ttl

    out = image.copy()
    h, w = out.shape[:2]

    decision = frame_result.get("decision", "UNKNOWN")
    risk     = frame_result.get("risk", 0.0)
    speed    = frame_result.get("speed_mps", 0.0)
    vlm_fired = frame_result.get("vlm_fired", False)
    vlm_summary = frame_result.get("vlm_summary", "")

    # ── Decision color map ────────────────────────────────────────────────────
    DEC_COLORS = {
        "CLEAR":     (30,  200, 30),
        "EASE":      (60,  220, 100),
        "MONITOR":   (180, 180, 30),
        "SLOW":      (30,  180, 220),
        "YIELD":     (30,  160, 255),
        "CAUTION":   (30,  120, 255),
        "STOP":      (30,   30, 220),
        "EMERGENCY": (0,    0,  255),
        "UNKNOWN":   (100, 100, 100),
    }
    dec_color = DEC_COLORS.get(decision, (100, 100, 100))

    # ── Colored border strip at top ───────────────────────────────────────────
    _blend_rect(out, 0, 0, w, 52, dec_color, alpha=0.75)

    # Decision label (large)
    cv2.putText(out, decision, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Speed target
    spd_kmh = speed * 3.6
    cv2.putText(out, f"{spd_kmh:.0f} km/h", (w - 130, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Camera detections (bounding boxes + labels) ───────────────────────────
    sf = frame_result.get("sensory_frame")
    if sf is not None and hasattr(sf, "detections"):
        for det in sf.detections:
            if det.camera_name == "lidar":
                continue   # skip synthetic LiDAR-only blobs
            b = det.bbox
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            cls = b.class_name or "obj"
            dist = det.depth_estimate
            dist_str = f"{dist:.1f}m" if dist is not None else ""
            label = f"{cls} {dist_str}".strip()
            conf  = b.confidence

            # Box color: red closer, green further
            if dist is not None and dist < 5.0:
                box_col = (0, 60, 255)
            elif dist is not None and dist < 15.0:
                box_col = (0, 165, 255)
            else:
                box_col = (50, 220, 50)

            cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            _blend_rect(out, x1, max(0, y1 - th - 8), x1 + tw + 6, y1, (20, 20, 20), 0.7)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ── LiDAR cluster dots along bottom ──────────────────────────────────────
    _blend_rect(out, 0, h - 36, w, h, (10, 10, 10), 0.6)
    shown = sorted(lidar_dets, key=lambda d: d.distance_m)[:8]
    for i, ld in enumerate(shown):
        x_pos = 12 + i * (w // 9)
        dist_ratio = min(ld.distance_m / 40.0, 1.0)
        dot_r = int(255 * (1 - dist_ratio))
        dot_g = int(255 * dist_ratio)
        dot_col = (0, dot_g, dot_r)
        cv2.circle(out, (x_pos + 16, h - 18), 5, dot_col, -1)
        cv2.putText(out, f"{ld.distance_m:.0f}m", (x_pos, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Risk bar (right edge) ─────────────────────────────────────────────────
    bar_w = 18
    bar_h = h - 100
    _draw_risk_bar(out, w - bar_w - 10, 60, bar_w, bar_h, risk)

    # ── HUD (bottom-left) ─────────────────────────────────────────────────────
    hud_lines = [
        f"Frame  {frame_result['frame_idx']:05d}",
        f"Cam    {frame_result['n_cam_dets']}  LiDAR  {frame_result['n_lidar_dets']}",
        f"Risk   {risk:.0%}",
        f"Lat    {frame_result['latency_ms']:.0f} ms",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(out, line, (10, h - 42 - (len(hud_lines) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # ── VLM text overlay ──────────────────────────────────────────────────────
    if vlm_fired and vlm_summary:
        _vlm_text_buffer = vlm_summary
        _vlm_text_ttl    = VLM_TEXT_HOLD_FRAMES

    if _vlm_text_ttl > 0:
        _vlm_text_ttl -= 1
        alpha = min(1.0, _vlm_text_ttl / 15.0)   # fade out last 15 frames

        box_y1 = 58
        box_y2 = 110
        _blend_rect(out, 0, box_y1, w, box_y2, (10, 10, 60), 0.75)

        # PRISM badge
        cv2.putText(out, "PRISM AI", (10, box_y1 + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

        # Word-wrap the VLM summary to fit the frame width
        words   = _vlm_text_buffer.split()
        lines   = []
        cur     = ""
        for word in words:
            test = (cur + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            if tw > w - 110:
                lines.append(cur)
                cur = word
            else:
                cur = test
        if cur:
            lines.append(cur)

        for li, line in enumerate(lines[:2]):
            y_pos = box_y1 + 18 + (li + 1) * 20
            if y_pos < box_y2 - 4:
                cv2.putText(out, line, (90, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (255, 255, 200), 1, cv2.LINE_AA)

    # ── PRISM watermark ───────────────────────────────────────────────────────
    cv2.putText(out, "PRISM", (w // 2 - 28, h - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    return out


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_bag(
    bag_path: Path,
    cfg: dict,
    vlm_enabled: bool    = True,
    max_frames: int      = 0,
    save_video: bool     = False,
    live_display: bool   = False,
    output_dir: Path     = None,
) -> dict:
    """Run PRISM pipeline on a single bag. Returns results dict."""

    bag_name = bag_path.stem
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating bag: {bag_path.name}")
    logger.info(f"VLM: {'enabled' if vlm_enabled else 'DISABLED (ablation)'}")
    logger.info(f"{'='*60}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    logger.info("Building PRISM pipeline...")
    pipeline   = build_pipeline(cfg, vlm_enabled=vlm_enabled)
    lidar_proc = LiDARProcessor()

    # ── Bag loader ────────────────────────────────────────────────────────────
    loader = ROS2BagLoader(bag_path)
    info = loader.get_bag_info()
    logger.info(f"Bag duration: {info['duration_s']:.1f}s")
    logger.info(f"Topics: {info['topics']}")

    # ── Detect LiDAR-only bag ─────────────────────────────────────────────────
    has_camera = any(
        t in info["topics"]
        for t in ["/usb_cam/image_raw", "/camera/image_raw"]
    )
    lidar_only_mode = not has_camera
    if lidar_only_mode:
        logger.info("No camera topic detected — running in LiDAR-only mode (no VLM)")

    # ── Update camera intrinsics once we get them ─────────────────────────────
    intrinsics_set = False

    # ── Video writer ──────────────────────────────────────────────────────────
    video_writer = None

    # ── Metrics accumulators ──────────────────────────────────────────────────
    per_frame   = []
    decisions   = defaultdict(int)
    latencies   = []
    vlm_fires   = 0
    vlm_events  = []   # list of {frame_idx, timestamp, summary, intents, decision}
    frame_count = 0
    start_wall  = time.time()

    try:
        if lidar_only_mode:
            # ── LiDAR-only iteration ──────────────────────────────────────────
            for frame_data in loader.iter_lidar_only(
                max_frames=max_frames if max_frames > 0 else None,
            ):
                ts    = frame_data["timestamp"]
                cloud = frame_data["point_cloud"]

                lidar_dets = lidar_proc.process(cloud) if cloud is not None else []

                result = run_frame_lidar_only(
                    lidar_dets = lidar_dets,
                    timestamp  = ts,
                    frame_idx  = frame_count,
                    pipeline   = pipeline,
                )

                decisions[result["decision"]] += 1
                latencies.append(result["latency_ms"])

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_wall
                    fps     = frame_count / elapsed
                    logger.info(
                        f"  Frame {frame_count} | {fps:.1f}fps | "
                        f"lat={result['latency_ms']:.0f}ms | "
                        f"lidar={result['n_lidar_dets']} | "
                        f"dec={result['decision']}"
                    )
        else:
            # ── Camera + LiDAR iteration ──────────────────────────────────────
            for frame_data in loader.iter_frames(
                sync_tolerance_s=0.08,
                max_frames=max_frames if max_frames > 0 else None,
            ):
                ts         = frame_data["timestamp"]
                image      = frame_data["image"]      # BGR or None
                cloud      = frame_data["point_cloud"] # (N,4) or None
                intrinsics = frame_data["intrinsics"]

                # Build calib dict from live intrinsics (or empty fallback)
                calib = intrinsics.to_calibration_dict() if intrinsics is not None else {}
                if intrinsics is not None and not intrinsics_set:
                    intrinsics_set = True
                    logger.info(f"Intrinsics from bag: {intrinsics}")

                # Process LiDAR
                lidar_dets = lidar_proc.process(cloud) if cloud is not None else []

                # Run pipeline
                result = run_frame(
                    image      = image,
                    lidar_dets = lidar_dets,
                    timestamp  = ts,
                    frame_idx  = frame_count,
                    pipeline   = pipeline,
                    calib      = calib,
                )

                # Accumulate metrics
                decisions[result["decision"]] += 1
                latencies.append(result["latency_ms"])
                if result["vlm_fired"]:
                    vlm_fires += 1
                    vlm_events.append({
                        "frame_idx": frame_count,
                        "timestamp": ts,
                        "summary":   result.get("vlm_summary", ""),
                        "intents":   result.get("vlm_intents", {}),
                        "decision":  result["decision"],
                        "risk":      result["risk"],
                    })

                # Video / live display output
                if (save_video or live_display) and image is not None:
                    annotated = annotate_frame(image, result, lidar_dets)

                    if save_video:
                        if video_writer is None:
                            fh, fw = annotated.shape[:2]
                            if output_dir:
                                vid_path = output_dir / f"ros2_eval_{bag_name}.mp4"
                            else:
                                vid_path = Path(f"/tmp/ros2_eval_{bag_name}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(vid_path), fourcc, 12, (fw, fh))
                            logger.info(f"Video writer opened: {vid_path}")
                        video_writer.write(annotated)

                    if live_display:
                        cv2.imshow("PRISM — Real-Time Perception", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("Live display: user pressed Q — stopping")
                            break

                frame_count += 1

                # Progress log
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_wall
                    fps     = frame_count / elapsed
                    logger.info(
                        f"  Frame {frame_count} | {fps:.1f}fps | "
                        f"lat={result['latency_ms']:.0f}ms | "
                        f"cam={result['n_cam_dets']} lidar={result['n_lidar_dets']} | "
                        f"dec={result['decision']}"
                    )

    finally:
        if video_writer is not None:
            video_writer.release()
        if live_display:
            cv2.destroyAllWindows()

    if frame_count == 0:
        logger.error("No frames processed — check bag path and topic names")
        return {}

    # ── Compute summary metrics ───────────────────────────────────────────────
    total_elapsed = time.time() - start_wall
    total_s       = frame_count / max(len(latencies), 1)   # rough

    results = {
        "bag":              bag_name,
        "bag_path":         str(bag_path),
        "vlm_enabled":      vlm_enabled,
        "total_frames":     frame_count,
        "wall_time_s":      round(total_elapsed, 2),
        "avg_fps":          round(frame_count / total_elapsed, 2),
        "avg_latency_ms":   round(float(np.mean(latencies)), 1),
        "p95_latency_ms":   round(float(np.percentile(latencies, 95)), 1),
        "max_latency_ms":   round(float(np.max(latencies)), 1),
        "vlm_trigger_rate_pct": round(100 * vlm_fires / frame_count, 2),
        "decision_distribution": dict(decisions),
        "intrinsics_from_bag":   intrinsics_set,
        "lidar_only_mode":       lidar_only_mode,
        "vlm_events":            vlm_events,
    }

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Bag:             {bag_name}")
    logger.info(f"Frames:          {frame_count}")
    logger.info(f"Wall time:       {total_elapsed:.1f}s  ({results['avg_fps']:.1f} fps)")
    logger.info(f"Avg latency:     {results['avg_latency_ms']}ms")
    logger.info(f"P95 latency:     {results['p95_latency_ms']}ms")
    logger.info(f"VLM trigger:     {results['vlm_trigger_rate_pct']}%  ({vlm_fires} fires)")
    if vlm_events:
        logger.info("VLM events:")
        for ev in vlm_events:
            logger.info(
                f"  [frame {ev['frame_idx']:04d} t={ev['timestamp']:.1f}s] "
                f"dec={ev['decision']}  risk={ev['risk']:.2f} | "
                f"\"{ev['summary']}\"  intents={ev['intents']}"
            )
    logger.info("Decision distribution:")
    for dec, cnt in sorted(decisions.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / frame_count
        logger.info(f"  {dec:<22} {cnt:4d} ({pct:.1f}%)")
    logger.info(f"{'='*60}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM ROS2 Bag Evaluation")
    parser.add_argument(
        "--bag", default="test1",
        help="Bag name: test1 | test2 | ros2_bag | all  (default: test1)"
    )
    parser.add_argument("--no-vlm",      action="store_true", help="Disable VLM (ablation)")
    parser.add_argument("--max-frames",  type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--save-video",  action="store_true", help="Write annotated video")
    parser.add_argument("--live",        action="store_true", help="Show live OpenCV display window")
    parser.add_argument("--output-dir",  type=str, default=None, help="Override output dir")
    args = parser.parse_args()

    cfg = load_config()

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path(cfg.get("data", {}).get("output_dir",
                      "~/prism_data/experiments")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select bags to run
    if args.bag == "all":
        bag_names = list(BAG_REGISTRY.keys())
    else:
        if args.bag not in BAG_REGISTRY:
            logger.error(f"Unknown bag '{args.bag}'. Choose from: {list(BAG_REGISTRY.keys())}")
            sys.exit(1)
        bag_names = [args.bag]

    all_results = {}

    for bag_name in bag_names:
        bag_path = BAG_REGISTRY[bag_name]
        if not bag_path.exists():
            logger.warning(f"Bag file not found: {bag_path} — skipping")
            continue

        results = evaluate_bag(
            bag_path     = bag_path,
            cfg          = cfg,
            vlm_enabled  = not args.no_vlm,
            max_frames   = args.max_frames,
            save_video   = args.save_video,
            live_display = args.live,
            output_dir   = output_dir,
        )
        all_results[bag_name] = results

        # Save per-bag JSON
        from datetime import datetime
        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        vlm_tag  = "novlm" if args.no_vlm else "vlm"
        out_path = output_dir / f"ros2_eval_{bag_name}_{vlm_tag}_{ts_str}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {out_path}")

    # If multiple bags, save combined
    if len(all_results) > 1:
        ts_str   = time.strftime("%Y%m%d_%H%M%S")
        combined = output_dir / f"ros2_eval_all_{ts_str}.json"
        with open(combined, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Combined results: {combined}")


if __name__ == "__main__":
    main()
