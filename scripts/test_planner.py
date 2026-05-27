"""
PRISM — Adaptive Planner Live Test
====================================
Runs the full pipeline through to the Adaptive Planner and prints the
velocity profile, waypoints, and control command per frame.

Shows:
  - Arbitration decision from all 4 signals
  - Planner velocity ramp (jerk-limited S-curve)
  - Control command: throttle / brake
  - Spatial override flag if obstacle forces earlier braking
  - Emergency brake flag

Run on Jetson:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/test_planner.py

    # Specific scene, more frames
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/test_planner.py \
        --scene 5 --max-frames 30

    # Only print summary
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/test_planner.py --quiet
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.sensory_core.data_loader import NuScenesLoader
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.metric_depth import MetricDepthEngine
from prism.world_model.world_model import WorldModel
from prism.predictive_engine.engine import PredictiveEngine
from prism.predictive_engine.decision import SmartDecisionEngine, DECISIONS
from prism.arbitration.core import ArbitrationCore, LEVEL_ORDER
from prism.planner.planner import AdaptivePlanner

logger = get_logger("test_planner")

CONFIG_PATH   = "config/default.yaml"
NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"

# Unicode bar chars for velocity profile
VBAR_FULL  = "█"
VBAR_EMPTY = "░"


# ── Display helpers ───────────────────────────────────────────────────────────

def speed_bar(v: float, v_max: float = 10.0, width: int = 20) -> str:
    filled = int(round(width * min(v, v_max) / max(v_max, 0.01)))
    return VBAR_FULL * filled + VBAR_EMPTY * (width - filled)


def profile_sparkline(profile, v_max: float = 10.0, width: int = 40) -> str:
    """One-line ASCII sparkline of the velocity profile."""
    _BLOCKS = " ▁▂▃▄▅▆▇█"
    step = max(1, len(profile) // width)
    sampled = profile[::step][:width]
    chars = []
    for v in sampled:
        idx = int(round(8 * min(v, v_max) / max(v_max, 0.01)))
        chars.append(_BLOCKS[idx])
    return "".join(chars)


def print_frame(
    frame_idx:  int,
    arb,                    # ArbitrationDecision
    plan,                   # PlannerOutput
    timestamp:  float,
):
    """Print the Arbitration + Planner audit for one frame."""
    c = plan.control

    print(f"\n{'═'*76}")
    print(
        f"  Frame {frame_idx:>3}  │  t={timestamp:.2f}s  │  "
        f"ARB: {arb.action:<10}conf={arb.confidence*100:.0f}%  │  "
        f"PLN: {plan.current_speed_mps:.2f}→{plan.target_speed_mps:.2f} m/s"
    )
    print(f"{'═'*76}")

    # Velocity ramp bar
    v_now = plan.current_speed_mps
    v_tgt = plan.target_speed_mps
    print(
        f"  Speed now   : {speed_bar(v_now):20s}  {v_now:.2f} m/s  "
        f"({v_now*3.6:.1f} km/h)"
    )
    print(
        f"  Speed target: {speed_bar(v_tgt):20s}  {v_tgt:.2f} m/s  "
        f"({v_tgt*3.6:.1f} km/h)"
    )

    # Velocity profile sparkline over horizon
    if plan.velocity_profile:
        spark = profile_sparkline(plan.velocity_profile)
        print(f"  Profile [{plan.plan_horizon_s:.0f}s]: {spark}")

    print()

    # Control command
    thr_bar = speed_bar(c.throttle, v_max=1.0, width=15)
    brk_bar = speed_bar(c.brake,    v_max=1.0, width=15)
    print(f"  Throttle : {thr_bar}  {c.throttle:.3f}")
    print(f"  Brake    : {brk_bar}  {c.brake:.3f}")
    print(f"  Steer    : {c.steer:+.3f}   Gear: {c.gear}")

    print()

    # Safety flags
    flags = []
    if plan.emergency:
        flags.append("⚡ EMERGENCY BRAKE")
    if plan.spatial_override:
        flags.append("🔀 SPATIAL OVERRIDE")
    if arb.hard_override:
        flags.append(f"🛑 ARB HARD OVERRIDE: {arb.override_reason[:50]}")
    if flags:
        print(f"  {'  │  '.join(flags)}")
        print()

    # Stopping distance
    print(
        f"  Stop dist   : {plan.stopping_distance_m:.1f}m  "
        f"  Nearest obs: "
        + (f"{plan.reason.split('at')[1].split('m')[0].strip()}m" if "at" in plan.reason and "m" in plan.reason.split("at")[-1] else "—")
    )

    # One-line reason
    print(f"\n  ▶ {plan.reason[:70]}")

    # Dominant signal
    print(f"  Dominant: {arb.dominant_signal}  │  {arb.short_reason[:60]}")


def print_summary(
    frame_count:       int,
    arb_times_ms:      list,
    plan_times_ms:     list,
    decision_counts:   dict,
    emergency_count:   int,
    spatial_count:     int,
    final_speed_mps:   float,
):
    print(f"\n{'═'*76}")
    print("  PLANNER SUMMARY")
    print(f"{'═'*76}")
    print(f"  Frames processed  : {frame_count}")
    print(
        f"  Avg arb latency   : {np.mean(arb_times_ms):.2f}ms  "
        f"(max={max(arb_times_ms):.2f}ms)"
    )
    print(
        f"  Avg plan latency  : {np.mean(plan_times_ms):.3f}ms  "
        f"(max={max(plan_times_ms):.3f}ms)"
    )
    print(f"  Final speed       : {final_speed_mps:.2f} m/s  ({final_speed_mps*3.6:.1f} km/h)")
    print(f"  Emergency frames  : {emergency_count}")
    print(f"  Spatial overrides : {spatial_count}")
    print()
    print("  Decision distribution:")
    for decision in LEVEL_ORDER:
        count = decision_counts.get(decision, 0)
        if count == 0:
            continue
        pct = 100.0 * count / frame_count
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"    {decision:<12} {bar}  {count:3d} ({pct:.1f}%)")
    print(f"{'═'*76}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",      type=int, default=5)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--config",     default=CONFIG_PATH)
    p.add_argument("--quiet",      action="store_true",
                   help="Only print summary, not per-frame output")
    args = p.parse_args()

    print("=" * 76)
    print("PRISM — Adaptive Planner Live Test")
    print("=" * 76)

    # Config
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = {}
        cfg.setdefault("data", {})["nuscenes_root"] = NUSCENES_ROOT

    # Components
    loader         = NuScenesLoader(cfg)
    core           = SensoryCore(cfg)
    metric_engine  = MetricDepthEngine(cfg)
    world          = WorldModel(cfg)
    predictor      = PredictiveEngine(cfg)
    decision_engine= SmartDecisionEngine()
    arbitrator     = ArbitrationCore(cfg)
    planner        = AdaptivePlanner(cfg)

    scene_info = loader.get_scene_info(args.scene)
    print(f"\n  Scene {args.scene}: {scene_info['description']}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Cruise speed: {planner.cruise_speed_mps:.1f} m/s  "
          f"({planner.cruise_speed_mps*3.6:.0f} km/h)\n")

    decision_counts: dict = {}
    arb_times_ms:   list  = []
    plan_times_ms:  list  = []
    emergency_count:  int = 0
    spatial_count:    int = 0
    frame_idx               = 0
    last_plan               = None

    for frame_data in loader.iter_primary_camera(scene_idx=args.scene):
        if frame_idx >= args.max_frames:
            break

        image     = frame_data["image"]
        calib     = frame_data.get("calibration", {})
        timestamp = frame_data["timestamp"]

        # Full pipeline
        metric_engine.update_intrinsics(calib)
        sensory_frame = core.process(
            image,
            camera_name=frame_data["camera_name"],
            timestamp=timestamp,
        )
        metric_dets, _ = metric_engine.process_frame(image, sensory_frame.detections)
        world_state  = world.update(sensory_frame, calibration=calib)
        pred_state   = predictor.update(world_state, metric_dets)
        scene        = decision_engine.assess(world_state, pred_state, metric_dets)

        # Layer 5 — Arbitration
        t_arb = time.perf_counter()
        arb   = arbitrator.arbitrate(world_state, pred_state, scene, timestamp=timestamp)
        arb_ms = (time.perf_counter() - t_arb) * 1000

        # Layer 6 — Planner
        t_pln = time.perf_counter()
        plan  = planner.plan(arb, metric_dets=metric_dets, timestamp=timestamp)
        pln_ms = (time.perf_counter() - t_pln) * 1000

        arb_times_ms.append(arb_ms)
        plan_times_ms.append(pln_ms)
        decision_counts[arb.action] = decision_counts.get(arb.action, 0) + 1
        if plan.emergency:
            emergency_count += 1
        if plan.spatial_override:
            spatial_count += 1

        last_plan = plan

        if not args.quiet:
            print_frame(frame_idx + 1, arb, plan, timestamp)

        frame_idx += 1

    print_summary(
        frame_count      = frame_idx,
        arb_times_ms     = arb_times_ms,
        plan_times_ms    = plan_times_ms,
        decision_counts  = decision_counts,
        emergency_count  = emergency_count,
        spatial_count    = spatial_count,
        final_speed_mps  = last_plan.current_speed_mps if last_plan else 0.0,
    )


if __name__ == "__main__":
    main()
