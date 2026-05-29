"""
PRISM — CARLA Closed-Loop Evaluation
======================================
Runs PRISM's full 6-layer pipeline inside CARLA simulator and collects
quantitative metrics for the paper.

Metrics collected:
    collision_rate        — collisions per km driven
    route_completion      — % of waypoints reached
    mean_speed_kmh        — average driving speed
    rms_jerk              — ride comfort (lower = smoother)
    emergency_brake_rate  — % of frames with emergency brake
    vlm_trigger_rate      — % of frames that triggered VLM
    avg_latency_ms        — mean per-frame pipeline latency
    decision_distribution — MONITOR/EASE/SLOW/CAUTION/YIELD/STOP/EMERGENCY %

Requirements (Ubuntu / Parrot OS x86_64):
    1. Download CARLA 0.9.15:
       mkdir -p ~/carla && cd ~/carla
       wget https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz
       tar -xf CARLA_0.9.15.tar.gz

    2. Install Python client:
       pip install carla==0.9.15

    3. Start CARLA server (in a separate terminal):
       ~/carla/CarlaUE4.sh -RenderOffScreen   # headless (no display needed)
       # OR with display:
       ~/carla/CarlaUE4.sh

    4. Run this script:
       python scripts/run_carla_eval.py

    5. Results saved to:
       ~/prism_data/experiments/carla_eval_results.json

Arguments:
    --host       CARLA server host (default: localhost)
    --port       CARLA server port (default: 2000)
    --town       CARLA map (default: Town03 — urban intersections)
    --steps      Simulation steps at 10fps (default: 500 = 50 seconds)
    --config     PRISM config (default: configs/config.yaml)
    --no-vlm     Disable VLM for comparison run
    --save-video Save CARLA frames as video

Usage examples:
    # Standard eval (50 seconds, Town03):
    python scripts/run_carla_eval.py

    # Longer run (2 minutes, Town05 — complex intersections):
    python scripts/run_carla_eval.py --town Town05 --steps 1200

    # Compare: with VLM vs without VLM:
    python scripts/run_carla_eval.py --steps 500 --tag with_vlm
    python scripts/run_carla_eval.py --steps 500 --no-vlm --tag no_vlm
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.carla_bridge.bridge import CARLABridge

logger = get_logger("CARLAEval")


def parse_args():
    p = argparse.ArgumentParser(description="PRISM CARLA closed-loop evaluation")
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int, default=2000)
    p.add_argument("--town",       default="Town03")
    p.add_argument("--weather",    default="ClearNoon",
                   choices=["ClearNoon", "CloudyNoon", "WetNoon",
                            "ClearSunset", "HardRainNoon", "SoftRainSunset"])
    p.add_argument("--steps",      type=int, default=500,
                   help="Sim steps at 10fps. 500=50s, 1200=2min")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--no-vlm",     action="store_true",
                   help="Disable VLM for ablation comparison")
    p.add_argument("--tag",        default="",
                   help="Optional label appended to results filename")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--out",        type=Path,
                   default=Path("~/prism_data/experiments/carla_eval_results.json"))
    return p.parse_args()


def build_prism_pipeline(cfg: dict, no_vlm: bool):
    """Initialise all PRISM components."""
    from prism.sensory_core.core import SensoryCore
    from prism.sensory_core.metric_depth import MetricDepthEngine
    from prism.world_model.world_model import WorldModel
    from prism.predictive_engine.engine import PredictiveEngine
    from prism.predictive_engine.decision import SmartDecisionEngine
    from prism.semantic_reasoner.reasoner import SemanticReasoner
    from prism.arbitration.core import ArbitrationCore
    from prism.planner.planner import AdaptivePlanner

    if no_vlm:
        cfg = dict(cfg)
        cfg["vlm"] = dict(cfg.get("vlm", {}))
        cfg["vlm"]["enabled"] = False
        logger.info("VLM disabled for this run (ablation mode)")

    return dict(
        core      = SensoryCore(cfg),
        depth     = MetricDepthEngine(cfg),
        world     = WorldModel(cfg),
        predictor = PredictiveEngine(cfg),
        decision  = SmartDecisionEngine(),
        reasoner  = SemanticReasoner(cfg),
        arbitrator= ArbitrationCore(cfg),
        planner   = AdaptivePlanner(cfg),
    )


def run_prism_frame(pipeline: dict, image: np.ndarray, speed_mps: float,
                    timestamp: float) -> tuple:
    """
    Run one frame through all 6 PRISM layers.
    Returns (control, arb_decision, scene, vlm_output, latency_ms).
    """
    t0 = time.time()

    # Fake calibration (CARLA camera, approximate nuScenes FOV match)
    calib = {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 225.0}

    # Layer 1 — Sensory Core
    sensory = pipeline["core"].process(image, camera_name="CAM_FRONT", timestamp=timestamp)

    # Layer 1b — Metric Depth
    pipeline["depth"].update_intrinsics(calib)
    metric_dets, _ = pipeline["depth"].process_frame(
        image, sensory.detections, run_model=sensory.depth_map is not None
    )

    # Layer 2 — World Model
    world_state = pipeline["world"].update(sensory, calibration=calib)

    # Layer 3 — Predictive Engine
    pred_state = pipeline["predictor"].update(world_state, metric_dets)

    # Layer 3c — Smart Decision
    scene = pipeline["decision"].assess(world_state, pred_state, metric_dets)

    # Layer 4 — VLM (event-driven, non-blocking)
    vlm_output = pipeline["reasoner"].update(image, world_state, pred_state)
    if vlm_output and vlm_output.actor_intents:
        pipeline["predictor"].update_vlm_intents(vlm_output.actor_intents)

    # Layer 5 — Arbitration
    if vlm_output:
        pipeline["arbitrator"].update_vlm(vlm_output.to_arb_dict())
    arb = pipeline["arbitrator"].arbitrate(world_state, pred_state, scene, timestamp=timestamp)

    # Layer 6 — Planner
    control = pipeline["planner"].plan(arb, metric_dets=metric_dets, timestamp=timestamp)

    latency_ms = (time.time() - t0) * 1000
    return control, arb, scene, vlm_output, latency_ms


def main():
    args    = parse_args()
    cfg     = load_config(args.config)
    out_dir = Path(args.out).expanduser().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 68)
    logger.info("PRISM — CARLA Closed-Loop Evaluation")
    logger.info("=" * 68)
    logger.info(f"  Town    : {args.town}  Weather: {args.weather}")
    logger.info(f"  Steps   : {args.steps} ({args.steps/10:.0f}s @ 10fps)")
    logger.info(f"  VLM     : {'DISABLED (ablation)' if args.no_vlm else 'ENABLED'}")

    # ── Build PRISM pipeline ──────────────────────────────────────────────────
    logger.info("\nInitialising PRISM pipeline...")
    pipeline = build_prism_pipeline(cfg, no_vlm=args.no_vlm)
    logger.info("PRISM ready.")

    # ── Connect to CARLA ──────────────────────────────────────────────────────
    bridge = CARLABridge(host=args.host, port=args.port)
    try:
        bridge.connect()
        bridge.setup(world_name=args.town, weather=args.weather)
    except Exception as e:
        logger.error(f"CARLA connection failed: {e}")
        logger.error(
            "Make sure CARLA server is running:\n"
            "  ~/carla/CarlaUE4.sh -RenderOffScreen\n"
            "And carla Python package is installed:\n"
            "  pip install carla==0.9.15"
        )
        sys.exit(1)

    # ── Eval loop ─────────────────────────────────────────────────────────────
    metrics = bridge.metrics
    decision_counts = defaultdict(int)
    vlm_call_count  = 0

    video_writer = None
    if args.save_video:
        import cv2
        vpath = str(out_dir / f"carla_eval_{args.tag or 'run'}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(vpath, fourcc, 10.0, (800, 450))

    try:
        for frame in bridge.run(max_steps=args.steps):
            control, arb, scene, vlm_out, latency_ms = run_prism_frame(
                pipeline, frame.image, frame.speed_mps, frame.timestamp
            )

            # Apply control to CARLA
            bridge.apply_control(
                throttle=control.control.throttle,
                brake=control.control.brake,
                steer=control.control.steer,
            )

            # Track metrics
            metrics.frame_latencies_ms.append(latency_ms)
            decision_counts[arb.action] += 1
            if control.emergency:
                metrics.emergency_brakes += 1
            if vlm_out is not None:
                vlm_call_count += 1
                metrics.vlm_triggers += 1

            # Log every 50 frames
            if metrics.total_frames % 50 == 0:
                logger.info(
                    f"Step {metrics.total_frames:4d}/{args.steps} | "
                    f"spd={frame.speed_mps*3.6:.1f}km/h | "
                    f"action={arb.action} | "
                    f"col={metrics.total_collisions} | "
                    f"lat={latency_ms:.0f}ms"
                )

            if video_writer is not None:
                video_writer.write(frame.image)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Wait for in-flight VLM inference
        if pipeline["reasoner"].worker.is_busy:
            logger.info("Waiting for VLM inference to complete...")
            deadline = time.time() + 60
            while pipeline["reasoner"].worker.is_busy and time.time() < deadline:
                time.sleep(1.0)

        bridge.cleanup()
        if video_writer:
            video_writer.release()

    # ── Compute final metrics ─────────────────────────────────────────────────
    vlm_stats = pipeline["reasoner"].stats

    results = {
        "run_config": {
            "town":    args.town,
            "weather": args.weather,
            "steps":   args.steps,
            "vlm_enabled": not args.no_vlm,
            "tag":     args.tag,
        },
        "collision_rate_per_km":  round(metrics.collision_rate, 4),
        "total_collisions":       metrics.total_collisions,
        "distance_km":            round(metrics.distance_km, 3),
        "route_completion_pct":   round(metrics.route_completion * 100, 1),
        "mean_speed_kmh":         round(metrics.mean_speed_kmh, 2),
        "rms_jerk_ms3":           round(metrics.rms_jerk, 4),
        "emergency_brake_rate_pct": round(
            metrics.emergency_brakes / max(1, metrics.total_frames) * 100, 2
        ),
        "vlm_trigger_rate_pct":   round(metrics.vlm_trigger_rate * 100, 2),
        "vlm_total_calls":        vlm_stats["vlm_calls"],
        "vlm_avg_inference_ms":   round(vlm_stats["avg_inference_ms"], 1),
        "avg_latency_ms":         round(metrics.avg_latency_ms, 1),
        "total_frames":           metrics.total_frames,
        "decision_distribution":  {
            k: round(v / max(1, metrics.total_frames) * 100, 1)
            for k, v in decision_counts.items()
        },
        "collision_events": [
            {
                "frame":   e.frame_id,
                "other":   e.other_actor,
                "impulse": round(e.impulse_norm, 2),
            }
            for e in metrics.collision_events
        ],
    }

    # Save JSON
    tag_str = f"_{args.tag}" if args.tag else ""
    out_file = Path(args.out).expanduser().parent / f"carla_eval{tag_str}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 68)
    logger.info("CARLA EVALUATION RESULTS")
    logger.info("=" * 68)
    logger.info(f"  Town/Weather        : {args.town} / {args.weather}")
    logger.info(f"  Total frames        : {results['total_frames']}")
    logger.info(f"  Distance driven     : {results['distance_km']:.3f} km")
    logger.info(f"  Mean speed          : {results['mean_speed_kmh']:.1f} km/h")
    logger.info(f"")
    logger.info(f"  Collision rate      : {results['collision_rate_per_km']:.3f} /km")
    logger.info(f"  Total collisions    : {results['total_collisions']}")
    logger.info(f"  Route completion    : {results['route_completion_pct']:.1f}%")
    logger.info(f"")
    logger.info(f"  RMS jerk            : {results['rms_jerk_ms3']:.4f} m/s³")
    logger.info(f"  Emergency brakes    : {results['emergency_brake_rate_pct']:.1f}% of frames")
    logger.info(f"")
    logger.info(f"  VLM trigger rate    : {results['vlm_trigger_rate_pct']:.1f}%")
    logger.info(f"  VLM total calls     : {results['vlm_total_calls']}")
    logger.info(f"  VLM avg inference   : {results['vlm_avg_inference_ms']:.0f}ms")
    logger.info(f"")
    logger.info(f"  Avg pipeline latency: {results['avg_latency_ms']:.0f}ms")
    logger.info(f"")
    logger.info("  Decision distribution:")
    _ORDER = ["CLEAR","MONITOR","EASE","SLOW","CAUTION","YIELD","STOP","EMERGENCY"]
    for action in _ORDER:
        pct = results["decision_distribution"].get(action, 0)
        if pct == 0:
            continue
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        logger.info(f"    {action:<12} {bar}  {pct:.0f}%")
    logger.info(f"")
    logger.info(f"  Results saved: {out_file}")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
