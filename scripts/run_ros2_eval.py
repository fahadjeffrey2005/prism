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

def annotate_frame(
    image: np.ndarray,
    frame_result: dict,
    lidar_dets: List[LiDARDetection],
) -> np.ndarray:
    """Draw PRISM outputs onto the camera frame."""
    out = image.copy()
    h, w = out.shape[:2]

    # ── Camera detections ─────────────────────────────────────────────────────
    for det in frame_result.get("sensory_frame", {}) and [] or []:
        pass  # handled by SensoryVisualizer below

    if frame_result["sensory_frame"] is not None:
        from prism.sensory_core.core import SensoryVisualizer
        out = SensoryVisualizer.draw_detections(out, frame_result["sensory_frame"])

    # ── LiDAR cluster indicators (distance labels at bottom) ─────────────────
    for i, det in enumerate(lidar_dets[:5]):
        x_pos = int(w * (0.15 + i * 0.15))
        label = f"L:{det.distance_m:.1f}m"
        cv2.putText(out, label, (x_pos, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

    # ── Decision banner ───────────────────────────────────────────────────────
    decision = frame_result.get("decision", "UNKNOWN")
    risk     = frame_result.get("risk", 0.0)
    colors   = {
        "PROCEED":         (0, 200, 0),
        "DECELERATE":      (0, 165, 255),
        "STOP":            (0, 0, 220),
        "EMERGENCY_BRAKE": (0, 0, 255),
        "YIELD":           (0, 200, 200),
        "UNKNOWN":         (128, 128, 128),
    }
    color = colors.get(decision, (128, 128, 128))
    cv2.rectangle(out, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.putText(out, f"{decision}  risk={risk:.2f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    hud = [
        f"Frame: {frame_result['frame_idx']}",
        f"Cam: {frame_result['n_cam_dets']}  LiDAR: {frame_result['n_lidar_dets']}",
        f"Latency: {frame_result['latency_ms']:.0f}ms",
        f"VLM: {'ON' if frame_result['vlm_fired'] else 'off'}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 55 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_bag(
    bag_path: Path,
    cfg: dict,
    vlm_enabled: bool    = True,
    max_frames: int      = 0,
    save_video: bool     = False,
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

                # Video output
                if save_video and image is not None:
                    annotated = annotate_frame(image, result, lidar_dets)
                    if video_writer is None:
                        h, w = annotated.shape[:2]
                        if output_dir:
                            vid_path = output_dir / f"ros2_eval_{bag_name}.mp4"
                        else:
                            vid_path = Path(f"/tmp/ros2_eval_{bag_name}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(vid_path), fourcc, 10, (w, h))
                        logger.info(f"Video writer opened: {vid_path}")
                    video_writer.write(annotated)

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
            bag_path    = bag_path,
            cfg         = cfg,
            vlm_enabled = not args.no_vlm,
            max_frames  = args.max_frames,
            save_video  = args.save_video,
            output_dir  = output_dir,
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
