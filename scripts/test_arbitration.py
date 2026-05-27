"""
PRISM — Arbitration Core Live Test
====================================
Runs the full pipeline and prints the Arbitration Core audit trail every frame.
This shows all four signals, their weights, and the final fused decision.

Run on Jetson:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/test_arbitration.py

    # Specific scene, more frames
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/test_arbitration.py \
        --scene 5 --max-frames 30
"""

import sys
import time
import argparse
import yaml
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

logger = get_logger("test_arbitration")

CONFIG_PATH   = "config/default.yaml"
NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"

# Decision level bar characters
BAR_CHARS = "░▒▓█"

def level_bar(level: int, max_level: int = 7) -> str:
    filled = int(round(20 * level / max(max_level, 1)))
    return "█" * filled + "░" * (20 - filled)


def signal_row(name: str, decision: str, level: int, weight_pct: float,
               conf: float, reason: str, age_s: float = 0.0) -> str:
    age = f" age={age_s:.1f}s" if age_s > 0.1 else ""
    bar = level_bar(level)
    return (
        f"  {name:<16} {bar}  {decision:<10} "
        f"w={weight_pct:4.1f}%  conf={conf:.2f}{age}\n"
        f"  {'':16}   └─ {reason[:70]}"
    )


def print_frame(frame_idx: int, arb, timestamp: float):
    """Print the full arbitration audit for one frame."""
    print(f"\n{'═'*72}")
    print(f"  Frame {frame_idx:>3}  │  t={timestamp:.2f}s  │  "
          f"Decision: {arb.action:<10} {DECISIONS[arb.action]['speed']*100:.0f}% speed  │  "
          f"Conf: {arb.confidence*100:.0f}%")
    print(f"{'═'*72}")

    total_w = sum(s.weight for s in arb.signals)
    for sig in arb.signals:
        pct = 100.0 * sig.weight / max(total_w, 1e-6)
        print(signal_row(
            sig.name, sig.decision, sig.level,
            pct, sig.confidence, sig.reason, sig.age_s
        ))
        print()

    # Agreement bar
    agree_pct = int(arb.signal_agreement * 100)
    agree_bar = level_bar(agree_pct, 100)
    print(f"  Signal agreement: {agree_bar}  {agree_pct}%", end="")
    if arb.is_conservative:
        print("  ⚠ Safety buffer applied", end="")
    print()

    # Override
    if arb.hard_override:
        print(f"\n  ⚡ HARD OVERRIDE: {arb.override_reason}")

    # Dominant signal
    print(f"\n  Dominant: {arb.dominant_signal}  │  {arb.short_reason[:65]}")

    # Decision trend
    print(f"\n  ▶ {arb.action}  —  {DECISIONS[arb.action]['speed']*100:.0f}% speed", end="")
    if arb.action in ("STOP", "EMERGENCY"):
        print("  🔴", end="")
    elif arb.action in ("CAUTION", "YIELD"):
        print("  🟡", end="")
    else:
        print("  🟢", end="")
    print()


def print_summary(decision_counts: dict, total_frames: int, arb_times_ms: list):
    print(f"\n{'═'*72}")
    print("  ARBITRATION SUMMARY")
    print(f"{'═'*72}")
    print(f"  Frames processed : {total_frames}")
    print(f"  Avg arb latency  : {np.mean(arb_times_ms):.2f}ms  "
          f"(max={max(arb_times_ms):.2f}ms)")
    print()
    print("  Decision distribution:")
    for decision in LEVEL_ORDER:
        count = decision_counts.get(decision, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total_frames
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"    {decision:<12} {bar}  {count:3d} ({pct:.1f}%)")
    print(f"{'═'*72}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",      type=int, default=5)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--config",     default=CONFIG_PATH)
    p.add_argument("--quiet",      action="store_true",
                   help="Only print summary, not per-frame audit")
    args = p.parse_args()

    print("=" * 72)
    print("PRISM — Arbitration Core Live Test")
    print("=" * 72)

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

    scene_info = loader.get_scene_info(args.scene)
    print(f"\n  Scene {args.scene}: {scene_info['description']}")
    print(f"  Max frames: {args.max_frames}\n")

    decision_counts: dict = {}
    arb_times_ms:   list = []
    frame_idx = 0

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
        metric_dets  = metric_engine.process_frame(image, sensory_frame.detections)
        world_state  = world.update(sensory_frame, calibration=calib)
        pred_state   = predictor.update(world_state, metric_dets)
        scene        = decision_engine.assess(world_state, pred_state, metric_dets)

        # Arbitration
        t0 = time.perf_counter()
        arb = arbitrator.arbitrate(world_state, pred_state, scene, timestamp=timestamp)
        arb_ms = (time.perf_counter() - t0) * 1000
        arb_times_ms.append(arb_ms)

        decision_counts[arb.action] = decision_counts.get(arb.action, 0) + 1

        if not args.quiet:
            print_frame(frame_idx + 1, arb, timestamp)

        frame_idx += 1

    print_summary(decision_counts, frame_idx, arb_times_ms)


if __name__ == "__main__":
    main()
