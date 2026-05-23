"""
PRISM — World Model Validation Script
======================================
Runs Sensory Core + World Model together.
Produces the first real PRISM composite output:
    - Camera view with tracked actors + velocity arrows
    - BEV grid showing occupancy
    - Risk HUD with live level indicator

Usage:
    python scripts/run_world_model.py
    python scripts/run_world_model.py --scene 2 --max-frames 40
"""

import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.data_loader import NuScenesLoader
from prism.world_model.world_model import WorldModel, WorldModelVisualizer

logger = get_logger("run_world_model")


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM World Model Validation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["logging"]["viz_output_dir"]).expanduser() / "world_model"
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to: {output_dir}")

    logger.info("=" * 60)
    logger.info("PRISM — World Model Validation")
    logger.info("=" * 60)
    video_writer = None

    # Init components
    loader = NuScenesLoader(cfg)
    core = SensoryCore(cfg)
    world = WorldModel(cfg)
    viz = WorldModelVisualizer()

    scene_info = loader.get_scene_info(args.scene)
    logger.info(f"Scene: {scene_info['name']} — {scene_info['description']}")

    # Stats
    stats = {
        "frames": 0,
        "total_actors": 0,
        "risk_levels": [0, 0, 0, 0],
        "max_actors": 0,
        "times": []
    }

    for frame_data in loader.iter_primary_camera(scene_idx=args.scene):
        if stats["frames"] >= args.max_frames:
            break

        image = frame_data["image"]
        t0 = time.time()

        # Layer 1 — Sensory Core
        sensory_frame = core.process(
            image,
            camera_name=frame_data["camera_name"],
            timestamp=frame_data["timestamp"]
        )

        # Layer 2 — World Model
        state = world.update(sensory_frame, calibration=frame_data.get("calibration"))

        elapsed = (time.time() - t0) * 1000
        stats["frames"] += 1
        stats["total_actors"] += state.actor_count
        stats["max_actors"] = max(stats["max_actors"], state.actor_count)
        stats["risk_levels"][state.risk_level] += 1
        stats["times"].append(elapsed)

        # Log
        actor_str = ", ".join([
            f"#{a.track_id}{a.class_name[0].upper()}"
            for a in state.actors if a.is_confirmed
        ])
        logger.info(
            f"Frame {stats['frames']:3d} | "
            f"{elapsed:5.0f}ms | "
            f"{state.level_name:<6} ({state.risk_score:.2f}) | "
            f"{state.actor_count} actors [{actor_str}] | "
            f"conf={state.perception_confidence:.2f}"
        )

        # Visualize
        bev_img = world.bev.visualize(size=450)
        composite = viz.make_composite(image, bev_img, state)

        if args.save:
            if video_writer is None:
                h, w = composite.shape[:2]
                video_path = str(output_dir / "world_model_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w, h))
                logger.info(f"Video writer initialised: {video_path}")
            video_writer.write(composite)

        if args.show:
            cv2.imshow("PRISM — World Model", composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Summary
    logger.info("=" * 60)
    logger.info("WORLD MODEL VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Frames:           {stats['frames']}")
    logger.info(f"Avg FPS:          {1000/np.mean(stats['times']):.1f}")
    logger.info(f"Avg actors/frame: {stats['total_actors']/stats['frames']:.1f}")
    logger.info(f"Max actors:       {stats['max_actors']}")
    logger.info(f"Risk distribution:")
    for i, name in enumerate(["GREEN", "YELLOW", "ORANGE", "RED"]):
        pct = stats["risk_levels"][i] / stats["frames"] * 100
        logger.info(f"  {name:<8} {stats['risk_levels'][i]:3d} frames ({pct:.0f}%)")
    if video_writer is not None:
        video_writer.release()
        logger.info(f"Video saved: {output_dir}/world_model_output.mp4")
    logger.info(f"\nOutput: {output_dir}")
    logger.info("=" * 60)
    logger.info("Next: python scripts/run_predictive_engine.py")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
