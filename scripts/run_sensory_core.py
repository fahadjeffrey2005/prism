"""
PRISM — Sensory Core Validation Script
=======================================
Run this to validate your full Sensory Core pipeline.

What this does:
    1. Loads a nuScenes scene
    2. Streams frames through the Sensory Core
    3. Runs detection + depth on each frame
    4. Saves annotated output frames
    5. Prints a performance summary

Usage:
    python scripts/run_sensory_core.py
    python scripts/run_sensory_core.py --scene 1 --camera CAM_FRONT_LEFT
    python scripts/run_sensory_core.py --no-save --show   # display live (needs display)
"""

import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.sensory_core.core import SensoryCore, SensoryVisualizer
from prism.sensory_core.data_loader import NuScenesLoader

logger = get_logger("run_sensory_core")


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM Sensory Core Validation")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--scene", type=int, default=0, help="nuScenes scene index")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera to use")
    parser.add_argument("--max-frames", type=int, default=40, help="Max frames to process")
    parser.add_argument("--save", action="store_true", default=True, help="Save output frames")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--show", action="store_true", help="Display frames live (needs display)")
    parser.add_argument("--depth-alpha", type=float, default=0.35, help="Depth overlay opacity")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg["cameras"]["primary"] = args.camera

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = Path(cfg["logging"]["viz_output_dir"]).expanduser() / "sensory_core"
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving output to: {output_dir}")

    # ── Init components ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PRISM — Sensory Core Validation")
    logger.info("=" * 60)

    loader = NuScenesLoader(cfg)
    core = SensoryCore(cfg)
    viz = SensoryVisualizer()

    scene_info = loader.get_scene_info(args.scene)
    logger.info(f"Scene: {scene_info['name']}")
    logger.info(f"Description: {scene_info['description']}")
    logger.info(f"Samples: {scene_info['num_samples']}")
    logger.info(f"Camera: {args.camera}")
    logger.info("-" * 60)

    # ── Stats tracking ────────────────────────────────────────────────────────
    stats = {
        "frames": 0,
        "total_detections": 0,
        "detection_times": [],
        "depth_times": [],
        "flow_times": [],
        "class_counts": {}
    }

    # ── Main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()
    frames_buffer = []  # for making a summary grid at the end

    for frame_data in loader.iter_primary_camera(scene_idx=args.scene):
        if stats["frames"] >= args.max_frames:
            break

        image = frame_data["image"]
        camera_name = frame_data["camera_name"]
        timestamp = frame_data["timestamp"]

        # ── Process through Sensory Core ──────────────────────────────────────
        t0 = time.time()
        sensory_frame = core.process(image, camera_name=camera_name, timestamp=timestamp)
        total_time = (time.time() - t0) * 1000

        # ── Update stats ──────────────────────────────────────────────────────
        stats["frames"] += 1
        stats["total_detections"] += len(sensory_frame.detections)

        if "detection_ms" in sensory_frame.processing_times:
            stats["detection_times"].append(sensory_frame.processing_times["detection_ms"])
        if "depth_ms" in sensory_frame.processing_times:
            stats["depth_times"].append(sensory_frame.processing_times["depth_ms"])
        if "flow_ms" in sensory_frame.processing_times:
            stats["flow_times"].append(sensory_frame.processing_times["flow_ms"])

        for det in sensory_frame.detections:
            cls = det.bbox.class_name
            stats["class_counts"][cls] = stats["class_counts"].get(cls, 0) + 1

        # ── Logging ───────────────────────────────────────────────────────────
        det_str = ", ".join([f"{d.bbox.class_name}({d.bbox.confidence:.2f})"
                             for d in sensory_frame.detections[:4]])
        if len(sensory_frame.detections) > 4:
            det_str += f" +{len(sensory_frame.detections)-4} more"

        logger.info(
            f"Frame {stats['frames']:3d} | "
            f"{total_time:5.1f}ms | "
            f"{len(sensory_frame.detections)} objects | "
            f"{det_str}"
        )

        # ── Visualize ─────────────────────────────────────────────────────────
        vis = viz.make_overlay(
            image, sensory_frame,
            show_depth=True,
            alpha=args.depth_alpha
        )

        if args.save:
            out_path = output_dir / f"frame_{stats['frames']:04d}_{camera_name}.jpg"
            cv2.imwrite(str(out_path), vis)

        if args.show:
            cv2.imshow("PRISM — Sensory Core", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Save a sample frame every 10 frames for the summary grid
        if stats["frames"] % 10 == 0:
            frames_buffer.append(cv2.resize(vis, (480, 270)))

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("SENSORY CORE VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Frames processed:     {stats['frames']}")
    logger.info(f"Total time:           {total_time:.1f}s")
    logger.info(f"Avg FPS:              {stats['frames']/total_time:.1f}")
    logger.info(f"Total detections:     {stats['total_detections']}")
    logger.info(f"Avg detections/frame: {stats['total_detections']/max(stats['frames'],1):.1f}")

    if stats["detection_times"]:
        logger.info(f"Detection latency:    {np.mean(stats['detection_times']):.1f}ms avg "
                    f"| {np.max(stats['detection_times']):.1f}ms max")
    if stats["depth_times"]:
        logger.info(f"Depth latency:        {np.mean(stats['depth_times']):.1f}ms avg "
                    f"| {np.max(stats['depth_times']):.1f}ms max")

    logger.info("\nDetected classes:")
    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        logger.info(f"  {cls:<20} {count}")

    # ── Save summary grid ─────────────────────────────────────────────────────
    if args.save and frames_buffer:
        grid = make_grid(frames_buffer, cols=2)
        grid_path = output_dir / "summary_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        logger.info(f"\nSummary grid saved: {grid_path}")
        logger.info(f"All frames saved to: {output_dir}")

    if args.show:
        cv2.destroyAllWindows()

    logger.info("=" * 60)
    logger.info("Sensory Core validation complete.")
    logger.info("Next step: build the World Model → python scripts/run_world_model.py")
    logger.info("=" * 60)


def make_grid(frames: list, cols: int = 2) -> np.ndarray:
    """Arrange frames into a grid image."""
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    return grid


if __name__ == "__main__":
    main()
