"""
PRISM — LSTM Intent Evaluator
===============================
Shows the LSTM intent model making real predictions on nuScenes scenes.
Prints rich terminal output per actor so you can see the system thinking.

Run BEFORE training (Bayesian baseline):
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/eval_lstm_intent.py --no-model

Run AFTER training (LSTM model):
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/eval_lstm_intent.py \
        --checkpoint /home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt

Options:
    --scenes 5 6 8 9      (default: 2 5 8 9 — mix of train and val)
    --max-frames 30       (default: 20)
    --verbose             show full probability distribution per actor
    --no-model            run Bayesian baseline for comparison

What you will see:
    Each frame prints every tracked actor with:
      - Class, distance, speed, heading direction
      - LSTM intent bars (top-3 maneuvers with probability)
      - TTC warning if collision risk
      - Whether the prediction matched what actually happened (GT lookahead)
      - Final scene-level decision
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from prism.predictive_engine.lstm_intent import (
    LSTMIntentNet, LSTMIntentPredictor, TrajectoryBuffer,
    MANEUVERS, SEQUENCE_LEN, load_lstm_model,
    format_intent_bar, top_maneuver, DISPLAY_OVERRIDE,
    _normalise_trajectory,
)
from scripts.train_lstm_intent import derive_label, ANNOTATED_HZ

NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"
CHECKPOINT    = "/home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt"

# ── Direction helpers ─────────────────────────────────────────────────────────

def heading_arrow(dx: float, dy: float) -> str:
    """Convert (dx, dy) velocity to compass arrow."""
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return "·"
    angle = np.degrees(np.arctan2(dx, dy))
    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    idx = int((angle + 202.5) / 45.0) % 8
    return arrows[idx]


# ── Per-instance trajectory tracker (for eval only) ──────────────────────────

class InstanceTracker:
    """
    Follows nuScenes annotation chains to get ground-truth trajectories.
    Used for verifying LSTM predictions (did the actor really do what LSTM said?).
    """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        # instance_token → list of (sample_token, translation)
        self._chains: Dict[str, List] = defaultdict(list)
        self._build_chains()

    def _build_chains(self):
        for ann in self.nusc.sample_annotation:
            self._chains[ann["instance_token"]].append(
                (ann["sample_token"], ann["translation"], ann["token"])
            )
        # Sort by sample order is tricky; instead we'll use token chains below

    def get_future_positions(self, ann_token: str, n: int = 5) -> Optional[np.ndarray]:
        """Follow annotation chain `n` steps forward. Returns (n, 2) or None."""
        positions = []
        cur = ann_token
        for _ in range(n):
            a = self.nusc.get("sample_annotation", cur)
            positions.append(a["translation"][:2])
            if not a["next"]:
                break
            cur = a["next"]
        if len(positions) < 2:
            return None
        return np.array(positions, dtype=np.float32)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_scene(
    nusc:         NuScenes,
    scene_idx:    int,
    model:        Optional[LSTMIntentNet],
    device:       str,
    max_frames:   int,
    verbose:      bool,
    inst_tracker: InstanceTracker,
):
    scene     = nusc.scene[scene_idx]
    split     = "TRAIN" if scene_idx < 8 else "VAL"

    print(f"\n{'═'*68}")
    print(f"  Scene {scene_idx} [{split}] — {scene['description']}")
    print(f"{'═'*68}")

    # Per-instance LSTM predictor (keyed by instance_token)
    lstm_predictors: Dict[str, LSTMIntentPredictor] = {}

    # Track positions per instance for heading estimation
    last_positions: Dict[str, np.ndarray] = {}

    sample_token = scene["first_sample_token"]
    frame_idx    = 0

    scene_stats = {
        "total_predictions": 0,
        "lstm_ready":        0,
        "correct":           0,
        "risky_actors":      0,
    }

    while sample_token and frame_idx < max_frames:
        sample   = nusc.get("sample", sample_token)
        t        = sample["timestamp"] / 1e6
        anns     = sample["anns"]

        # Filter to dynamic actors
        actors = []
        for ann_token in anns:
            ann  = nusc.get("sample_annotation", ann_token)
            cat  = ann["category_name"]
            if not any(
                cat.startswith(p)
                for p in ["vehicle.car", "vehicle.truck", "vehicle.bus",
                          "vehicle.motorcycle", "vehicle.bicycle",
                          "human.pedestrian"]
            ):
                continue
            actors.append((ann_token, ann))

        if not actors:
            sample_token = sample["next"]
            frame_idx   += 1
            continue

        print(f"\n  ── Frame {frame_idx+1:>2}/{min(scene['nbr_samples'], max_frames)} "
              f"│ t={t:.1f}s │ {len(actors)} actors ──")

        frame_decisions = []

        for ann_token, ann in actors:
            itoken   = ann["instance_token"]
            cat      = ann["category_name"]
            pos_xy   = np.array(ann["translation"][:2], dtype=np.float32)

            # Simplified class
            if "pedestrian" in cat:
                cls = "pedestrian"
            elif "bicycle" in cat:
                cls = "bicycle"
            elif "motorcycle" in cat:
                cls = "motorcycle"
            elif "truck" in cat or "bus" in cat:
                cls = "truck"
            else:
                cls = "car"

            # Ego-relative distance (approximate using global position for now)
            # In full pipeline this comes from metric_depth; here use GT directly
            dist_m = np.linalg.norm(pos_xy)   # rough — GT global pos from origin

            # Speed from position delta
            prev_pos = last_positions.get(itoken)
            if prev_pos is not None:
                dp    = pos_xy - prev_pos
                speed = np.linalg.norm(dp) * ANNOTATED_HZ   # m/s
                arrow = heading_arrow(dp[0], dp[1])
            else:
                speed = 0.0
                arrow = "?"
            last_positions[itoken] = pos_xy

            # ── LSTM prediction ──────────────────────────────

            if model is not None:
                if itoken not in lstm_predictors:
                    lstm_predictors[itoken] = LSTMIntentPredictor(model, device)

                predictor = lstm_predictors[itoken]
                probs     = predictor.update(pos_xy, t)
            else:
                probs = None

            scene_stats["total_predictions"] += 1

            # ── Ground truth verification ─────────────────────

            future_pos = inst_tracker.get_future_positions(ann_token, n=5)
            if future_pos is not None and len(future_pos) >= 2:
                gt_label_idx = derive_label(future_pos)
                gt_label     = MANEUVERS[gt_label_idx]
            else:
                gt_label_idx = None
                gt_label     = "?"

            # ── Print actor block ──────────────────────────────

            bar = "─" * 66
            print(f"\n    {bar}")
            dist_str  = f"{dist_m:5.1f}m" if dist_m < 200 else "  far"
            speed_kph = speed * 3.6
            print(f"    Actor [{cls:<10}] dist={dist_str}  speed={speed_kph:5.1f}km/h  heading {arrow}")

            if probs is not None and predictor.ready:
                scene_stats["lstm_ready"] += 1
                top_label = top_maneuver(probs, cls)
                top_prob  = float(probs[np.argmax(probs)])

                lines = format_intent_bar(probs, cls, top_k=3 if verbose else 2)
                print(f"    [LSTM intent]")
                for line in lines:
                    print(f"    {line}")

                # TTC estimate — simple: pedestrian within 5m crossing = high risk
                is_risky = False
                if cls == "pedestrian" and dist_m < 6.0 and "crossing" in top_label:
                    ttc_est = dist_m / max(speed, 0.5)
                    is_risky = True
                    scene_stats["risky_actors"] += 1
                    print(f"    ⚠️  PEDESTRIAN CROSSING RISK  TTC≈{ttc_est:.1f}s")
                elif dist_m < 4.0 and speed > 2.0:
                    is_risky = True
                    scene_stats["risky_actors"] += 1
                    print(f"    ⚠️  CLOSE RANGE HIGH SPEED")

                # Verification
                if gt_label_idx is not None:
                    predicted_idx = int(np.argmax(probs))
                    correct = (predicted_idx == gt_label_idx)
                    scene_stats["correct"] += 1 if correct else 0
                    override = DISPLAY_OVERRIDE.get(cls, {})
                    gt_display = override.get(gt_label, gt_label)
                    sym = "✓" if correct else "✗"
                    print(f"    {sym} GT: {gt_display:<22}  LSTM: {top_label:<22}  ({top_prob*100:.0f}%)")

                frame_decisions.append(("lstm", top_label, top_prob, is_risky))

            else:
                # Not enough history yet — show Bayesian placeholder
                frames_needed = SEQUENCE_LEN - len(
                    lstm_predictors.get(itoken, LSTMIntentPredictor(model or LSTMIntentNet(), device)).buffer._positions
                ) if model else SEQUENCE_LEN
                print(f"    [intent: building history — need {frames_needed} more frames]")

                if gt_label_idx is not None:
                    override   = DISPLAY_OVERRIDE.get(cls, {})
                    gt_display = override.get(gt_label, gt_label)
                    print(f"    GT next: {gt_display}")

        # ── Frame-level decision ──────────────────────────────

        risky = [d for d in frame_decisions if d[3]]
        if risky:
            decision = "CAUTION"
            dec_sym  = "🔴"
        elif any("braking" in d[1] or "stopping" in d[1] for d in frame_decisions):
            decision = "MONITOR"
            dec_sym  = "🟡"
        else:
            decision = "CLEAR"
            dec_sym  = "🟢"

        print(f"\n    {dec_sym} Frame decision: {decision}  "
              f"({len(frame_decisions)} actors tracked, "
              f"{len(risky)} risky)")

        sample_token = sample["next"]
        frame_idx   += 1

    return scene_stats


def print_summary(all_stats: List[dict], scene_indices: List[int]):
    print(f"\n{'═'*68}")
    print("  EVALUATION SUMMARY")
    print(f"{'═'*68}")

    total_preds  = sum(s["total_predictions"] for s in all_stats)
    lstm_ready   = sum(s["lstm_ready"] for s in all_stats)
    total_correct = sum(s["correct"] for s in all_stats)
    total_risky  = sum(s["risky_actors"] for s in all_stats)

    ready_pct   = 100.0 * lstm_ready / max(total_preds, 1)
    correct_pct = 100.0 * total_correct / max(lstm_ready, 1)

    print(f"\n  Scenes evaluated     : {len(scene_indices)}  {scene_indices}")
    print(f"  Total actor-frames   : {total_preds}")
    print(f"  LSTM active          : {lstm_ready}  ({ready_pct:.1f}% of frames)")
    print(f"  Intent accuracy      : {total_correct}/{lstm_ready}  ({correct_pct:.1f}%)")
    print(f"  Risky actor-frames   : {total_risky}")

    if correct_pct >= 70:
        print("\n  VERDICT: LSTM is learning meaningful intent patterns ✓")
    elif correct_pct >= 50:
        print("\n  VERDICT: LSTM shows partial learning — more data or epochs needed")
    else:
        print("\n  VERDICT: LSTM not yet reliable — check training loss / data quality")
    print(f"{'═'*68}\n")


def main():
    p = argparse.ArgumentParser(description="PRISM LSTM intent evaluator")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--data-root",  default=NUSCENES_ROOT)
    p.add_argument("--scenes",     type=int, nargs="+", default=[2, 5, 8, 9])
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--no-model",   action="store_true",
                   help="Run without LSTM (shows GT labels only, useful pre-training)")
    args = p.parse_args()

    print("=" * 68)
    print("PRISM — LSTM Intent Predictor Evaluation")
    print("=" * 68)

    # Device
    device = "cuda" if torch.cuda.is_available() else (
             "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"  Device     : {device}")
    print(f"  Scenes     : {args.scenes}")
    print(f"  Max frames : {args.max_frames}")

    # Load model
    model = None
    if not args.no_model:
        model = load_lstm_model(args.checkpoint, device)
        if model is None:
            print("  Mode: GT verification only (no model checkpoint found)")
        else:
            model.eval()
            print(f"  Model: loaded from {args.checkpoint}")
    else:
        print("  Mode: GT labels only (--no-model)")

    # nuScenes
    print(f"\nLoading nuScenes ...")
    nusc         = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)
    inst_tracker = InstanceTracker(nusc)

    all_stats = []
    for si in args.scenes:
        if si >= len(nusc.scene):
            print(f"  Scene {si} out of range — skipping")
            continue
        stats = run_scene(nusc, si, model, device, args.max_frames, args.verbose, inst_tracker)
        all_stats.append(stats)

    print_summary(all_stats, args.scenes)


if __name__ == "__main__":
    main()
