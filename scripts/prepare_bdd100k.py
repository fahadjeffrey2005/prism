"""
PRISM — BDD100K Trajectory Extractor (Phase 2 Training Data)
==============================================================
Extracts actor trajectories from BDD100K Multi-Object Tracking annotations
and converts them to the same (T, 6) normalised feature format used by
the LSTM intent predictor.

Why BDD100K?
    nuScenes mini has ~500 training trajectories → 67.9% accuracy.
    BDD100K MOT has 2000+ diverse video sequences (100K frames total)
    with tens of thousands of actor tracks — 40-50× more data.
    This is critical for rare classes: turn_left, braking, lane_change_*.

Input format (BDD100K MOT 2020):
    labels/box_track_20/train/<video_name>.json  — per-video annotations
    Each file: list of frames, each frame has list of labelled objects
    with category and 2D bbox. We track each object_id across frames.

Coordinate approach:
    BDD100K has no per-video camera calibration, so we work in pixel space.
    Pixel trajectories are normalised with the same translation+rotation
    invariant scheme as nuScenes — the direction pattern is preserved.
    We flip the y-axis (pixels go top→down, real-world goes bottom→up).

Output:
    ~/prism_data/checkpoints/lstm_intent/bdd100k_trajectories.npz
    Arrays:
        features  — (N, T, 6) float32  normalised trajectory windows
        labels    — (N,)      int64    maneuver class indices
        sources   — (N,)      str      video name for debugging

Download BDD100K MOT labels (~80 MB, labels only — no images needed):
    https://dl.cv.ethz.ch/bdd100k/data/bdd100k_mot_labels_coco_format.zip
    OR from official site: https://bdd-data.berkeley.edu/

    Structure expected:
        ~/prism_data/datasets/bdd100k/labels/box_track_20/train/*.json
        OR any path passed via --bdd-root

Usage:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/prepare_bdd100k.py

    # Custom paths
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/prepare_bdd100k.py \
        --bdd-root ~/prism_data/datasets/bdd100k \
        --out ~/prism_data/checkpoints/lstm_intent/bdd100k_trajectories.npz
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.predictive_engine.lstm_intent import (
    MANEUVERS, SEQUENCE_LEN, _normalise_trajectory,
)
from prism.utils.common import get_logger

logger = get_logger("prepare_bdd100k")

DEFAULT_BDD_ROOT = Path("/home/koushik-test/prism_data/datasets/bdd100k")
DEFAULT_OUT      = Path("/home/koushik-test/prism_data/checkpoints/lstm_intent/bdd100k_trajectories.npz")

# BDD100K category → simplified class (matching nuScenes DYNAMIC_CATEGORIES style)
BDD_CATEGORY_MAP = {
    "car":          "car",
    "truck":        "truck",
    "bus":          "bus",
    "motorcycle":   "motorcycle",
    "bicycle":      "bicycle",
    "pedestrian":   "pedestrian",
    "rider":        "pedestrian",   # treat rider same as pedestrian
}

# Sliding window config — must match nuScenes training
FUTURE_LEN   = 5
MIN_TRAJ_LEN = SEQUENCE_LEN + FUTURE_LEN
BDD_FPS      = 5.0   # BDD100K MOT is annotated at 5fps
DT           = 1.0 / BDD_FPS


# ── Label derivation (pixel space) ───────────────────────────────────────────

def derive_label_bdd(future_pos_px: np.ndarray) -> int:
    """
    Derive maneuver label from future pixel positions.

    Pixel coordinates: origin top-left, y increases downward.
    We flip y before computing so the coordinate system matches nuScenes
    (y increasing forward = away from ego vehicle = downward in image).

    Note: pixel-space magnitudes aren't metric, but direction patterns
    (turning, lane change, stopping) are preserved after normalisation.
    """
    # Flip y so increasing y = moving away from ego (forward)
    pos = future_pos_px.copy()
    pos[:, 1] = -pos[:, 1]

    if len(pos) < 2:
        return MANEUVERS.index("constant_velocity")

    vels  = np.diff(pos, axis=0) / DT   # px/s
    speeds = np.linalg.norm(vels, axis=1)
    final_speed = speeds[-1]

    # Actor heading from first velocity step
    first_speed = float(np.linalg.norm(vels[0]))
    if first_speed > 0.5:
        heading_fwd = vels[0] / first_speed
    else:
        heading_fwd = np.array([0.0, 1.0], dtype=np.float32)
    heading_lat = np.array([-heading_fwd[1], heading_fwd[0]])

    net_disp    = pos[-1] - pos[0]
    net_forward = float(np.dot(net_disp, heading_fwd))
    net_lateral = float(np.dot(net_disp, heading_lat))

    # Heading changes
    headings = np.arctan2(vels[:, 0], vels[:, 1])
    if len(headings) > 1:
        diffs = [(headings[i] - headings[i-1] + np.pi) % (2*np.pi) - np.pi
                 for i in range(1, len(headings))]
        mean_turn   = float(np.mean(np.abs(diffs)))
        signed_turn = float(np.mean(diffs))
    else:
        mean_turn = signed_turn = 0.0

    if len(speeds) > 1:
        mean_accel = float(np.diff(speeds).mean() / DT)
    else:
        mean_accel = 0.0

    # BDD pixel-space thresholds (scaled to 5fps, ~1280px wide images)
    # A car moving at ~30km/h ≈ 8.3 m/s ≈ 40-80 px/frame at typical ranges
    if net_forward < -5.0 and speeds.mean() > 1.0:
        return MANEUVERS.index("reversing")
    if final_speed < 2.0 and speeds.mean() < 20.0:
        return MANEUVERS.index("stopping")
    if mean_accel < -5.0 and final_speed > 2.0:
        return MANEUVERS.index("braking")
    if mean_accel > 5.0:
        return MANEUVERS.index("accelerating")
    if mean_turn > np.radians(8.0):
        return MANEUVERS.index("turn_left" if signed_turn > 0 else "turn_right")
    if abs(net_lateral) > 8.0 and net_forward > 5.0:
        # net_lateral > 0 = moved left (heading_lat points left)
        return MANEUVERS.index("lane_change_left" if net_lateral > 0 else "lane_change_right")
    return MANEUVERS.index("constant_velocity")


# ── BDD100K annotation loader ─────────────────────────────────────────────────

def load_bdd_annotations(label_dir: Path) -> List[Path]:
    """Return all per-video JSON annotation files."""
    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        # Try subdirectory structure
        json_files = sorted(label_dir.glob("**/*.json"))
    return json_files


def extract_tracks_from_video(json_path: Path) -> Dict[str, List[dict]]:
    """
    Parse one BDD100K tracking JSON and return per-track lists of
    {frame_idx, cx, cy, w, h, category}.
    """
    with open(json_path) as f:
        data = json.load(f)

    # BDD100K MOT format: list of frames, each frame has "labels"
    tracks: Dict[str, List[dict]] = defaultdict(list)

    # Handle both flat list (old format) and dict with "frames" key (new)
    frames = data if isinstance(data, list) else data.get("frames", [])

    for frame in frames:
        frame_idx = frame.get("index", frame.get("frameIndex", 0))
        labels    = frame.get("labels", [])
        for label in labels:
            cat = label.get("category", "")
            if BDD_CATEGORY_MAP.get(cat) is None:
                continue
            track_id = str(label.get("id", label.get("trackId", "")))
            if not track_id:
                continue
            box2d = label.get("box2d", {})
            if not box2d:
                continue
            x1, y1 = box2d.get("x1", 0), box2d.get("y1", 0)
            x2, y2 = box2d.get("x2", 0), box2d.get("y2", 0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tracks[track_id].append({
                "frame":    frame_idx,
                "cx":       cx,
                "cy":       cy,
                "category": BDD_CATEGORY_MAP[cat],
            })

    return dict(tracks)


def extract_windows_from_track(
    observations: List[dict],
) -> List[Tuple[np.ndarray, int]]:
    """
    Given sorted observations for one track, extract all (features, label) pairs.
    """
    if len(observations) < MIN_TRAJ_LEN:
        return []

    # Sort by frame index and deduplicate
    obs = sorted(observations, key=lambda x: x["frame"])
    positions = np.array([[o["cx"], o["cy"]] for o in obs], dtype=np.float32)

    pairs = []
    for start in range(len(positions) - MIN_TRAJ_LEN + 1):
        input_pos  = positions[start : start + SEQUENCE_LEN]
        future_pos = positions[start + SEQUENCE_LEN : start + SEQUENCE_LEN + FUTURE_LEN]

        feat  = _normalise_trajectory(input_pos)
        if feat is None or feat.shape != (SEQUENCE_LEN, 6):
            continue

        label = derive_label_bdd(future_pos)
        pairs.append((feat, label))

    return pairs


# ── Main extraction ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="BDD100K trajectory extractor")
    p.add_argument("--bdd-root", type=Path, default=DEFAULT_BDD_ROOT)
    p.add_argument("--out",      type=Path, default=DEFAULT_OUT)
    p.add_argument("--split",    default="train", choices=["train", "val"])
    p.add_argument("--max-videos", type=int, default=0,
                   help="Limit to N videos (0 = all). Use for quick test.")
    args = p.parse_args()

    # Find annotation directory — try both common structures
    label_dir = args.bdd_root / "labels" / "box_track_20" / args.split
    if not label_dir.exists():
        label_dir = args.bdd_root / "bdd100k" / "labels" / "box_track_20" / args.split
    if not label_dir.exists():
        label_dir = args.bdd_root / args.split
    if not label_dir.exists():
        logger.error(
            f"BDD100K label directory not found under {args.bdd_root}.\n"
            f"Expected: {args.bdd_root}/labels/box_track_20/{args.split}/*.json\n"
            f"Download labels from https://bdd-data.berkeley.edu/ "
            f"(bdd100k_mot_labels_coco_format.zip, ~80MB, labels only — no images needed)"
        )
        sys.exit(1)

    json_files = load_bdd_annotations(label_dir)
    if not json_files:
        logger.error(f"No JSON files found in {label_dir}")
        sys.exit(1)

    if args.max_videos > 0:
        json_files = json_files[:args.max_videos]

    logger.info(f"Found {len(json_files)} video annotation files in {label_dir}")

    all_features: List[np.ndarray] = []
    all_labels:   List[int]        = []
    all_sources:  List[str]        = []

    label_counts = defaultdict(int)
    skipped      = 0
    total_tracks = 0

    for i, jf in enumerate(json_files):
        try:
            tracks = extract_tracks_from_video(jf)
        except Exception as e:
            logger.warning(f"Failed to parse {jf.name}: {e}")
            skipped += 1
            continue

        total_tracks += len(tracks)
        video_name = jf.stem

        for tid, observations in tracks.items():
            pairs = extract_windows_from_track(observations)
            for feat, label in pairs:
                all_features.append(feat)
                all_labels.append(label)
                all_sources.append(video_name)
                label_counts[label] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(json_files):
            logger.info(
                f"  [{i+1}/{len(json_files)}] "
                f"tracks={total_tracks}  windows={len(all_features)}  skipped={skipped}"
            )

    if not all_features:
        logger.error("No trajectory windows extracted. Check label format.")
        sys.exit(1)

    features = np.stack(all_features, axis=0).astype(np.float32)   # (N, T, 6)
    labels   = np.array(all_labels,   dtype=np.int64)               # (N,)
    sources  = np.array(all_sources,  dtype=object)                 # (N,)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=features, labels=labels, sources=sources)

    # Report
    print("\n" + "="*60)
    print("BDD100K TRAJECTORY EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Videos processed : {len(json_files) - skipped}  (skipped: {skipped})")
    print(f"  Total tracks     : {total_tracks}")
    print(f"  Total windows    : {len(all_features)}")
    print(f"  Output           : {args.out}")
    print()
    print(f"  {'Class':<25} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*42}")
    total = len(all_features)
    for ci, name in enumerate(MANEUVERS):
        count = label_counts[ci]
        pct   = count / total * 100 if total > 0 else 0
        print(f"  {name:<25} {count:>8d}  {pct:>5.1f}%")
    print("="*60)
    print("\nNext step: retrain with combined nuScenes + BDD100K data:")
    print("  python scripts/train_lstm_intent.py --bdd-npz", args.out)
    print()


if __name__ == "__main__":
    main()
