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
from prism.semantic_reasoner.reasoner import SemanticReasoner
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

def _escalate(decision: str, lidar_dets: list, ego_speed_mps: float = 5.0) -> str:
    """
    Escalate decision based on LiDAR corridor obstacles.
    Thresholds are physics-based (stopping distance) so they scale with speed.
    """
    corridor = [d for d in lidar_dets if abs(d.lateral_m) < 2.5]
    if not corridor:
        return decision
    nearest = min(corridor, key=lambda d: d.distance_m)
    v = max(float(ego_speed_mps), 1.4)          # min 5 km/h assumed speed
    d_stop = v * v / (2 * 4.0)                  # comfortable stopping distance
    d_emrg = v * v / (2 * 8.0)                  # panic stopping distance
    d = nearest.distance_m
    lidar_dec = ("EMERGENCY" if d < d_emrg + 1.0  else
                 "STOP"      if d < d_stop + 1.5  else
                 "CAUTION"   if d < d_stop * 3.0  else
                 "SLOW"      if d < d_stop * 5.0  else decision)
    return _LEVEL_INV[max(_LEVEL.get(decision, 0), _LEVEL.get(lidar_dec, 0))]


# ── Ego speed estimator (optical flow based) ──────────────────────────────────
class EgoSpeedEstimator:
    """
    Estimates actual forward speed from optical flow on the road surface.

    Physics (pinhole camera, ground plane):
        flow_y (road at distance d) ≈ fy * cam_h * v / (fps * d²)
        → v ≈ flow_y * fps * d² / (fy * cam_h)

    Uses lower-centre image patch (road ~8m ahead) as reference.
    EMA-smoothed to suppress frame-to-frame jitter.
    Much better than plan.current_speed_mps which ramps to cruise
    regardless of actual vehicle state.
    """
    REF_DIST_M = 8.0    # reference ground distance for calibration
    CAM_H_M    = 1.2    # camera height above ground (metres)
    EMA_ALPHA  = 0.30   # smoothing — higher = more responsive, more noise
    MAX_MPS    = 15.0   # clamp at 54 km/h; anything above is noise

    def __init__(self):
        self._speed = 0.0

    def update(self, flow: "np.ndarray | None",
               intrinsics=None, fps: float = 12.0) -> float:
        if flow is None:
            self._speed *= (1.0 - self.EMA_ALPHA)
            return self._speed

        h, w = flow.shape[:2]
        # Lower-centre patch — road surface at ~REF_DIST_M ahead
        y1, y2 = int(h * 0.68), int(h * 0.88)
        x1, x2 = int(w * 0.30), int(w * 0.70)
        road_vy = flow[y1:y2, x1:x2, 1]   # vertical component

        # Forward motion → road moves DOWN in image (positive v)
        fwd_flow = float(np.clip(road_vy, 0, None).mean())

        # Calibration constant — use actual fy when available
        fy = float(intrinsics.fy) * (w / intrinsics.width) if intrinsics else 600.0
        k  = (fps * self.REF_DIST_M ** 2) / (fy * self.CAM_H_M)

        raw = float(np.clip(fwd_flow * k, 0.0, self.MAX_MPS))
        self._speed = (1.0 - self.EMA_ALPHA) * self._speed + self.EMA_ALPHA * raw
        return self._speed

    @property
    def kmh(self) -> float:
        return self._speed * 3.6


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


# ── Bag finder ────────────────────────────────────────────────────────────────
def _find_bag(arg: str) -> Path:
    """
    Locate a .db3 bag file robustly.
    Handles spaces in paths, subdirectory layouts, and filesystem encoding issues.

    Resolution order:
      1. Exact path as given
      2. Path resolved from cwd
      3. os.walk search in ~/Downloads for a file with the same name
         (bypasses any Unicode/encoding issue with the path string itself)
    """
    import os as _os

    # Try the path directly in multiple forms
    for candidate in (Path(arg), Path(arg).resolve(), Path.cwd() / arg):
        try:
            if candidate.exists() and candidate.suffix == ".db3":
                return candidate
        except (OSError, ValueError):
            pass

    # Walk ~/Downloads to find by filename (handles encoding issues in path)
    target_name = Path(arg).name
    if not target_name.endswith(".db3"):
        logger.error(f"Argument does not end in .db3: {arg}")
        sys.exit(1)

    search_root = str(Path.home() / "Downloads")
    logger.info(f"Direct path failed — searching {search_root} for {target_name}")
    for root, dirs, files in _os.walk(search_root):
        if target_name in files:
            found = Path(root) / target_name
            logger.info(f"Found: {found}")
            return found

    # Give up with a useful error showing what's actually on disk
    logger.error(f"Cannot find bag: {arg!r}")
    logger.error(f"Searched ~/Downloads for filename: {target_name!r}")
    logger.error("Run:  find ~/Downloads -name '*.db3'  to see available bags")
    sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    bag_path = _find_bag(args.bag)

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
    # Only instantiate SemanticReasoner if VLM is explicitly enabled in config —
    # it starts a background thread even in mock mode which adds scheduling overhead.
    _vlm_enabled = cfg.get("vlm", {}).get("enabled", False)
    reasoner     = SemanticReasoner(cfg) if _vlm_enabled else None

    logger.info("All components ready")
    logger.info("-" * 60)

    fps_tracker     = FPSTracker()
    trajectory      = TrajectoryTracker(max_points=1000)
    speed_est       = EgoSpeedEstimator()   # optical-flow based actual speed
    writer          = None
    save_path       = None
    frame_count     = 0
    paused          = False
    last_intrinsics = None
    ego_speed_mps   = 0.0    # estimated from optical flow, used for decisions

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
        img_scale = 1.0
        if orig_w != args.width:
            img_scale = args.width / orig_w
            new_h = int(orig_h * img_scale)
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

        # ── Update intrinsics — scale K matrix to match the resized image ─────
        # YOLO bboxes are in resized-image pixel space; intrinsics from camera_info
        # are at original resolution. Scaling fx, fy, cx, cy by img_scale ensures
        # lateral_m calculations in MetricDepthEngine are correct.
        if intrinsics is not None:
            calib = intrinsics.to_calibration_dict()
            if img_scale != 1.0:
                K = np.array(calib["camera_intrinsic"], dtype=np.float64)
                K[0, 0] *= img_scale   # fx
                K[0, 2] *= img_scale   # cx
                K[1, 1] *= img_scale   # fy
                K[1, 2] *= img_scale   # cy
                calib["camera_intrinsic"] = K.tolist()
            metric_eng.update_intrinsics(calib)
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
        scene          = dec_eng.assess(world_state, pred_state, all_metric,
                                        ego_speed_mps=ego_speed_mps)
        arb_decision   = arb.arbitrate(world_state, pred_state, scene, timestamp)
        final_decision = _escalate(arb_decision.action, lidar_dets,
                                   ego_speed_mps=ego_speed_mps)
        plan           = planner.plan(arb_decision, all_metric, timestamp)
        vlm_output     = (reasoner.update(image, world_state, pred_state)
                          if reasoner is not None else None)
        timings["decision_ms"] = (time.perf_counter() - t6) * 1000

        # ── Estimate actual ego speed from optical flow ────────────────────────
        # Use this for both display and next frame's decision thresholds.
        # plan.current_speed_mps is a commanded ramp — not the actual vehicle speed.
        ego_speed_mps = speed_est.update(
            sensory_frame.optical_flow, last_intrinsics, fps=12.0
        )

        # ── Render ────────────────────────────────────────────────────────────
        t7 = time.perf_counter()
        frame_result = {
            "decision":        final_decision,
            "risk":            world_state.risk_score,
            "speed_mps":       ego_speed_mps,          # actual estimated speed
            "n_cam_dets":      len(sensory_frame.detections),
            "n_lidar_dets":    len(lidar_dets),
            "latency_ms":      (time.perf_counter() - t0) * 1000,
            "frame_idx":       frame_count,
            "throttle":        plan.control.throttle,
            "brake":           plan.control.brake,
            "sensory_frame":   sensory_frame,
            "timestamp":       timestamp,
            "scene_assessment": scene,
            "vlm_output":       (vlm_output or
                                 (reasoner.get_current_semantic_state()
                                  if reasoner is not None else None)),
            "vlm_is_real":      (reasoner is not None and
                                 getattr(reasoner.vlm, "_available", False)),
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
