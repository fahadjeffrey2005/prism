"""
PRISM — Run Dashboard on Real ROS2 Bag Data  (optimised for real-time)
========================================================================
Camera + LiDAR are processed in PARALLEL threads.
LiDAR uses FastLiDARProcessor (numpy-only, no sklearn) — <5ms per scan.
Targets 30-50fps on Jetson Thor.

Usage:
    python scripts/run_bag_dashboard.py ~/Downloads/test1-003.db3
    python scripts/run_bag_dashboard.py ~/Downloads/test1-003.db3 --save
    python scripts/run_bag_dashboard.py ~/Downloads/test1-003.db3 --profile

Controls:
    q / ESC  → quit      SPACE → pause/resume
"""

import sys
import time
import argparse
import threading
import cv2
import numpy as np
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from prism.utils.common import load_config, get_logger, SensoryFrame
from prism.sensory_core.ros2_bag_loader import ROS2BagLoader
from prism.sensory_core.lidar_processor import FastLiDARProcessor, LiDARDetection
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.metric_depth import MetricDepthEngine
from prism.world_model.world_model import WorldModel
from prism.predictive_engine.engine import PredictiveEngine
from prism.predictive_engine.decision import SmartDecisionEngine
from prism.planner.planner import AdaptivePlanner
from prism.arbitration.core import ArbitrationCore
from prism.viz.dashboard import render_dashboard, TrajectoryTracker

logger = get_logger("RunBagDashboard")


# ── LiDAR metric proxy (same as before) ───────────────────────────────────────
class _LiDARMetricProxy:
    def __init__(self, d: LiDARDetection):
        self.distance_m  = d.distance_m
        self.lateral_m   = d.lateral_m
        self.threat_zone = d.threat_zone.lower()
        self.bbox        = type("B", (), {"track_id": None, "class_name": "unknown",
                                          "class_id": -1, "confidence": 0.9})()
    @property
    def is_in_corridor(self):
        return abs(self.lateral_m) < 2.5


# ── Decision escalation from LiDAR ────────────────────────────────────────────
_LEVEL     = {"CLEAR":0,"MONITOR":1,"EASE":2,"SLOW":3,
               "CAUTION":4,"YIELD":5,"STOP":6,"EMERGENCY":7}
_LEVEL_INV = {v: k for k, v in _LEVEL.items()}

def _escalate(decision: str, lidar_dets: list) -> str:
    corridor = [d for d in lidar_dets if abs(d.lateral_m) < 2.5]
    if not corridor:
        return decision
    nearest = min(corridor, key=lambda d: d.distance_m)
    lidar_dec = ("EMERGENCY" if nearest.distance_m <  3 else
                 "STOP"      if nearest.distance_m <  5 else
                 "CAUTION"   if nearest.distance_m < 10 else
                 "SLOW"      if nearest.distance_m < 20 else decision)
    return _LEVEL_INV[max(_LEVEL.get(decision,0), _LEVEL.get(lidar_dec,0))]


# ── FPS tracker ────────────────────────────────────────────────────────────────
class FPSTracker:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
    def tick(self):
        self._times.append(time.perf_counter())
    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("bag")
    p.add_argument("--config",     default=str(ROOT / "configs/config.yaml"))
    p.add_argument("--max-frames", type=int,   default=None)
    p.add_argument("--start",      type=float, default=None)
    p.add_argument("--end",        type=float, default=None)
    p.add_argument("--save",       action="store_true")
    p.add_argument("--no-show",    action="store_true")
    p.add_argument("--no-lidar",   action="store_true")
    p.add_argument("--no-depth",   action="store_true")
    p.add_argument("--profile",    action="store_true", help="Print per-component ms")
    p.add_argument("--width",      type=int, default=960,
                   help="Resize camera to this width before processing (default 960)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    bag_path = Path(args.bag).resolve()   # always resolves from actual cwd

    # Kill SensoryCore depth when --no-depth is set
    if args.no_depth:
        cfg.setdefault("sensory_core", {}).setdefault("sampling", {})["depth_fps"] = 0

    logger.info("=" * 60)
    logger.info("PRISM  —  Bag Dashboard Runner  (real-time optimised)")
    logger.info("=" * 60)

    # ── Init ──────────────────────────────────────────────────────────────────
    loader     = ROS2BagLoader(str(bag_path))
    info       = loader.get_bag_info()
    logger.info(f"Bag: {bag_path.name}  |  {info['duration_s']:.0f}s  |  topics: {info['topics']}")

    sensory    = SensoryCore(cfg)
    lidar_proc = FastLiDARProcessor()          # numpy-only, <5ms
    metric_eng = MetricDepthEngine(cfg)
    world      = WorldModel(cfg)
    predictor  = PredictiveEngine(cfg)
    dec_eng    = SmartDecisionEngine()
    arb        = ArbitrationCore(cfg)
    planner    = AdaptivePlanner(cfg)

    logger.info("All components ready")
    logger.info("-" * 60)

    fps_tracker = FPSTracker()
    trajectory  = TrajectoryTracker(max_points=1000)
    writer      = None
    save_path   = None
    frame_count = 0
    paused      = False
    last_intrinsics = None

    # ── LiDAR background thread state ─────────────────────────────────────────
    _lidar_result = [None]   # shared between main and lidar thread
    _lidar_lock   = threading.Lock()

    def _run_lidar(cloud):
        dets = lidar_proc.process(cloud)
        with _lidar_lock:
            _lidar_result[0] = dets

    # ── Main loop ─────────────────────────────────────────────────────────────
    for bag_frame in loader.iter_frames(
        max_frames=args.max_frames,
        start_s=args.start,
        end_s=args.end,
    ):
        image      = bag_frame["image"]
        cloud      = bag_frame["point_cloud"]
        intrinsics = bag_frame["intrinsics"]
        timestamp  = bag_frame["timestamp"]

        if image is None:
            continue

        # ── Resize to target width (keeps aspect ratio) ───────────────────────
        orig_h, orig_w = image.shape[:2]
        if orig_w != args.width:
            scale = args.width / orig_w
            new_h = int(orig_h * scale)
            image = cv2.resize(image, (args.width, new_h), interpolation=cv2.INTER_LINEAR)

        frame_count += 1
        t0 = time.perf_counter()
        timings = {}

        # ── LiDAR: kick off in background thread ──────────────────────────────
        lidar_thread = None
        if cloud is not None and not args.no_lidar:
            lidar_thread = threading.Thread(target=_run_lidar, args=(cloud.copy(),),
                                            daemon=True)
            lidar_thread.start()

        # ── Update intrinsics ─────────────────────────────────────────────────
        if intrinsics is not None:
            metric_eng.update_intrinsics(intrinsics.to_calibration_dict())
            last_intrinsics = intrinsics

        # ── Camera: SensoryCore ───────────────────────────────────────────────
        t1 = time.perf_counter()
        sensory_frame = sensory.process(image, camera_name="CAM_FRONT",
                                         timestamp=timestamp)
        timings["sensory_ms"] = (time.perf_counter() - t1) * 1000

        # ── Camera: Metric depth ──────────────────────────────────────────────
        t2 = time.perf_counter()
        metric_dets, _ = metric_eng.process_frame(
            image, sensory_frame.detections, run_model=(not args.no_depth)
        )
        timings["metric_ms"] = (time.perf_counter() - t2) * 1000

        # ── Wait for LiDAR thread ─────────────────────────────────────────────
        t3 = time.perf_counter()
        if lidar_thread is not None:
            lidar_thread.join()
        with _lidar_lock:
            lidar_dets = _lidar_result[0] or []
        timings["lidar_ms"] = (time.perf_counter() - t3) * 1000

        # ── World Model ───────────────────────────────────────────────────────
        t4 = time.perf_counter()
        calib       = intrinsics.to_calibration_dict() if intrinsics else None
        world_state = world.update(sensory_frame, calib)
        timings["world_ms"] = (time.perf_counter() - t4) * 1000

        # ── Predictive Engine ─────────────────────────────────────────────────
        t5 = time.perf_counter()
        pred_state  = predictor.update(world_state, metric_dets)
        timings["pred_ms"] = (time.perf_counter() - t5) * 1000

        # ── Decision + Arbitration + Planner ──────────────────────────────────
        t6 = time.perf_counter()
        all_metric     = list(metric_dets) + [_LiDARMetricProxy(d) for d in lidar_dets]
        scene          = dec_eng.assess(world_state, pred_state, all_metric)
        arb_decision   = arb.arbitrate(world_state, pred_state, scene, timestamp)
        final_decision = _escalate(arb_decision.action, lidar_dets)
        plan           = planner.plan(arb_decision, all_metric, timestamp)
        timings["decision_ms"] = (time.perf_counter() - t6) * 1000

        # ── Render (trajectory.update called inside using lane_steer) ───────────
        t7 = time.perf_counter()
        frame_result = {
            "decision":      final_decision,
            "risk":          world_state.risk_score,
            "speed_mps":     plan.current_speed_mps,
            "n_cam_dets":    len(sensory_frame.detections),
            "n_lidar_dets":  len(lidar_dets),
            "latency_ms":    (time.perf_counter() - t0) * 1000,
            "frame_idx":     frame_count,
            "throttle":      plan.control.throttle,
            "brake":         plan.control.brake,
            "sensory_frame": sensory_frame,
            "timestamp":     timestamp,
        }
        out = render_dashboard(image, frame_result, lidar_dets,
                               trajectory=trajectory,
                               intrinsics=last_intrinsics,
                               optical_flow=sensory_frame.optical_flow,
                               metric_dets=metric_dets)
        timings["render_ms"] = (time.perf_counter() - t7) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        fps_tracker.tick()

        # ── Logging ───────────────────────────────────────────────────────────
        if frame_count % 5 == 0 or frame_count == 1:
            fps = fps_tracker.fps
            logger.info(
                f"Frame {frame_count:4d} | {total_ms:5.1f}ms | {fps:4.1f}fps | "
                f"{final_decision:<10} | risk={world_state.risk_score:.2f} | "
                f"cam={len(sensory_frame.detections)} lidar={len(lidar_dets)}"
            )
            if args.profile:
                logger.info(
                    f"  [PROFILE] sensory={timings['sensory_ms']:.0f}ms "
                    f"metric={timings['metric_ms']:.0f}ms "
                    f"lidar={timings['lidar_ms']:.0f}ms "
                    f"world={timings['world_ms']:.0f}ms "
                    f"pred={timings['pred_ms']:.0f}ms "
                    f"decision={timings['decision_ms']:.0f}ms "
                    f"render={timings['render_ms']:.0f}ms "
                    f"TOTAL={total_ms:.0f}ms"
                )

        # ── Save ──────────────────────────────────────────────────────────────
        if args.save:
            if writer is None:
                out_dir   = Path(cfg["logging"]["viz_output_dir"]).expanduser()
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / f"dashboard_{bag_path.stem}.mp4"
                fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
                writer    = cv2.VideoWriter(str(save_path), fourcc, 15,
                                             (out.shape[1], out.shape[0]))
                logger.info(f"Saving to: {save_path}")
            writer.write(out)

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_show:
            cv2.imshow("PRISM Dashboard", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("Quit")
                break
            if key == ord(" "):
                paused = not paused
            while paused:
                key2 = cv2.waitKey(50) & 0xFF
                if key2 == ord(" "):
                    paused = False
                elif key2 in (ord("q"), 27):
                    paused = False
                    frame_count = args.max_frames or frame_count + 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if writer:
        writer.release()
        logger.info(f"Saved: {save_path}")
    if not args.no_show:
        cv2.destroyAllWindows()

    elapsed = time.time() - time.time()   # just for formatting
    logger.info(f"Done — {frame_count} frames processed | avg {fps_tracker.fps:.1f} fps")


if __name__ == "__main__":
    main()
