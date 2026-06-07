"""
PRISM — Run Dashboard on Real ROS2 Bag Data
=============================================
Streams camera + LiDAR from a .db3 bag, runs the full perception +
decision pipeline, and renders the live dashboard.

Usage:
    # test1 bag (237s, camera + LiDAR)
    python scripts/run_bag_dashboard.py "prism/new data/test1/test1.db3"

    # test2 bag, save output video
    python scripts/run_bag_dashboard.py "prism/new data/test2/test12.db3" --save

    # limit frames, start at 10s
    python scripts/run_bag_dashboard.py "prism/new data/test1/test1.db3" --max-frames 200 --start 10

Controls (when window is open):
    q / ESC  → quit
    SPACE    → pause / resume
"""

import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

# ── project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from prism.utils.common import load_config, get_logger, SensoryFrame, BBox2D, Detection
from prism.sensory_core.ros2_bag_loader import ROS2BagLoader
from prism.sensory_core.lidar_processor import LiDARProcessor, LiDARDetection
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.metric_depth import (
    MetricDepthEngine, MetricDetection, CameraIntrinsics, CameraExtrinsics
)
from prism.world_model.world_model import WorldModel
from prism.predictive_engine.engine import PredictiveEngine
from prism.predictive_engine.decision import SmartDecisionEngine
from prism.planner.planner import AdaptivePlanner
from prism.arbitration.core import ArbitrationCore
from prism.viz.dashboard import render_dashboard

logger = get_logger("RunBagDashboard")


# ── helper: convert LiDARDetection → MetricDetection-compatible object ─────────
class _LiDARMetricProxy:
    """Thin wrapper so SmartDecisionEngine can consume LiDAR detections."""
    def __init__(self, lidar_det: LiDARDetection):
        self.distance_m  = lidar_det.distance_m
        self.lateral_m   = lidar_det.lateral_m
        self.threat_zone = lidar_det.threat_zone.lower()   # "critical"|"close"|"medium"|"far"
        # Fake BBox so the engine's track-id lookup returns None gracefully
        self.bbox        = _FakeBBox(lidar_det)

    @property
    def is_in_corridor(self):
        return abs(self.lateral_m) < 2.5


class _FakeBBox:
    def __init__(self, det: LiDARDetection):
        self.track_id   = None
        self.class_name = "unknown"
        self.class_id   = -1
        self.confidence = 0.9
        # Approximate image-space bbox from bearing + distance (for display only)
        w = 640
        fx = 600.0
        u  = w / 2 - det.lateral_m * fx / max(det.distance_m, 1.0)
        self.x1 = max(0, int(u - 20))
        self.y1 = max(0, 200)
        self.x2 = min(w, int(u + 20))
        self.y2 = 300


# ── decision level escalation ──────────────────────────────────────────────────
_LEVEL = {
    "CLEAR":0,"MONITOR":1,"EASE":2,"SLOW":3,
    "CAUTION":4,"YIELD":5,"STOP":6,"EMERGENCY":7
}
_LEVEL_INV = {v: k for k, v in _LEVEL.items()}

def _escalate(decision: str, lidar_dets: list) -> str:
    """Override decision upward if LiDAR sees a close corridor obstacle."""
    if not lidar_dets:
        return decision
    corridor = [d for d in lidar_dets if abs(d.lateral_m) < 2.5]
    if not corridor:
        return decision
    nearest = min(corridor, key=lambda d: d.distance_m)
    lidar_decision = (
        "EMERGENCY" if nearest.distance_m < 3.0 else
        "STOP"      if nearest.distance_m < 5.0 else
        "CAUTION"   if nearest.distance_m < 10.0 else
        "SLOW"      if nearest.distance_m < 20.0 else
        decision
    )
    # Take the more cautious of camera-based and LiDAR-based
    return _LEVEL_INV[max(_LEVEL.get(decision, 0), _LEVEL.get(lidar_decision, 0))]


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRISM bag dashboard runner")
    p.add_argument("bag", help="Path to .db3 bag file")
    p.add_argument("--config",      default=str(ROOT / "configs/config.yaml"))
    p.add_argument("--max-frames",  type=int,   default=None)
    p.add_argument("--start",       type=float, default=None,  help="Start time in seconds")
    p.add_argument("--end",         type=float, default=None,  help="End time in seconds")
    p.add_argument("--save",        action="store_true",        help="Save output as MP4")
    p.add_argument("--no-show",     action="store_true",        help="Disable live window")
    p.add_argument("--no-lidar",    action="store_true",        help="Skip LiDAR processing")
    p.add_argument("--no-depth",    action="store_true",        help="Skip metric depth model")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    bag_path = Path(args.bag)
    if not bag_path.is_absolute():
        bag_path = ROOT / bag_path

    # ── Init pipeline components ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PRISM  —  Bag Dashboard Runner")
    logger.info("=" * 60)
    logger.info(f"Bag: {bag_path}")

    loader        = ROS2BagLoader(str(bag_path))
    bag_info      = loader.get_bag_info()
    logger.info(f"Bag topics : {bag_info['topics']}")
    logger.info(f"Duration   : {bag_info['duration_s']:.1f}s")

    sensory       = SensoryCore(cfg)
    lidar_proc    = LiDARProcessor(cfg.get("lidar", {}))
    metric_engine = MetricDepthEngine(cfg)
    world         = WorldModel(cfg)
    predictor     = PredictiveEngine(cfg)
    decision_eng  = SmartDecisionEngine()
    arbitration   = ArbitrationCore(cfg)
    planner       = AdaptivePlanner(cfg)

    logger.info("All components initialised")
    logger.info("-" * 60)

    # ── Video writer setup (deferred until first frame) ────────────────────────
    writer    = None
    save_path = None

    # ── Stats ──────────────────────────────────────────────────────────────────
    frame_count  = 0
    paused       = False
    t_run_start  = time.time()

    # ── Main loop ──────────────────────────────────────────────────────────────
    for bag_frame in loader.iter_frames(
        max_frames        = args.max_frames,
        start_s           = args.start,
        end_s             = args.end,
    ):
        t0         = time.perf_counter()
        image      = bag_frame["image"]          # (H, W, 3) BGR — may be None
        cloud      = bag_frame["point_cloud"]    # (N, 4) float32 — may be None
        intrinsics = bag_frame["intrinsics"]     # USBCamIntrinsics or None
        timestamp  = bag_frame["timestamp"]

        if image is None:
            continue   # skip lidar-only frames

        frame_count += 1
        cam_h, cam_w = image.shape[:2]

        # ── Update metric engine intrinsics ────────────────────────────────────
        if intrinsics is not None:
            metric_engine.update_intrinsics(intrinsics.to_calibration_dict())

        # ── Camera: Sensory Core ───────────────────────────────────────────────
        sensory_frame = sensory.process(image, camera_name="CAM_FRONT",
                                         timestamp=timestamp)

        # ── Camera: Metric depth ───────────────────────────────────────────────
        metric_dets, _ = metric_engine.process_frame(
            image,
            sensory_frame.detections,
            run_model=(not args.no_depth),
        )

        # ── LiDAR processing ───────────────────────────────────────────────────
        lidar_dets = []
        if cloud is not None and not args.no_lidar:
            lidar_dets = lidar_proc.process(cloud)

        # ── World Model update ─────────────────────────────────────────────────
        calibration = intrinsics.to_calibration_dict() if intrinsics else None
        world_state = world.update(sensory_frame, calibration)

        # ── Predictive Engine ──────────────────────────────────────────────────
        pred_state = predictor.update(world_state, metric_dets)

        # ── Smart Decision ─────────────────────────────────────────────────────
        all_metric = list(metric_dets) + [_LiDARMetricProxy(d) for d in lidar_dets]
        scene      = decision_eng.assess(world_state, pred_state, all_metric)

        # ── Arbitration ────────────────────────────────────────────────────────
        arb = arbitration.arbitrate(world_state, pred_state, scene, timestamp)

        # LiDAR corridor escalation (hard safety override)
        final_decision = _escalate(arb.action, lidar_dets)

        # ── Planner ────────────────────────────────────────────────────────────
        plan = planner.plan(arb, all_metric, timestamp)

        # ── Timing ────────────────────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Build frame_result for dashboard ──────────────────────────────────
        frame_result = {
            "decision":     final_decision,
            "risk":         world_state.risk_score,
            "speed_mps":    plan.current_speed_mps,
            "n_cam_dets":   len([d for d in sensory_frame.detections
                                 if d.bbox.class_name not in ("unknown",)]),
            "n_lidar_dets": len(lidar_dets),
            "latency_ms":   latency_ms,
            "frame_idx":    frame_count,
            "throttle":     plan.control.throttle,
            "brake":        plan.control.brake,
            "sensory_frame": sensory_frame,
        }

        # ── Render dashboard ───────────────────────────────────────────────────
        out = render_dashboard(image, frame_result, lidar_dets)

        # ── Logging ───────────────────────────────────────────────────────────
        if frame_count % 10 == 0 or frame_count == 1:
            logger.info(
                f"Frame {frame_count:4d} | {latency_ms:5.1f}ms | "
                f"{final_decision:<10} | "
                f"risk={world_state.risk_score:.2f} | "
                f"cam={len(sensory_frame.detections)} lidar={len(lidar_dets)} | "
                f"v={plan.current_speed_mps*3.6:.1f}km/h"
            )

        # ── Video writer ───────────────────────────────────────────────────────
        if args.save:
            if writer is None:
                out_dir = Path(cfg["logging"]["viz_output_dir"]).expanduser()
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / f"dashboard_{bag_path.stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(save_path), fourcc, 10,
                                         (out.shape[1], out.shape[0]))
                logger.info(f"Saving to: {save_path}")
            writer.write(out)

        # ── Live display ───────────────────────────────────────────────────────
        if not args.no_show:
            cv2.imshow("PRISM Dashboard", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or ESC
                logger.info("Quit by user")
                break
            if key == ord(" "):
                paused = not paused
            while paused:
                key2 = cv2.waitKey(50) & 0xFF
                if key2 == ord(" "):
                    paused = False
                elif key2 in (ord("q"), 27):
                    paused = False
                    frame_count = args.max_frames or frame_count  # force exit

    # ── Cleanup ────────────────────────────────────────────────────────────────
    if writer:
        writer.release()
        logger.info(f"Video saved: {save_path}")
    if not args.no_show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_run_start
    logger.info("=" * 60)
    logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s "
                f"({frame_count/elapsed:.1f} fps)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
