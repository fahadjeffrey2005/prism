"""
PRISM — Synthetic Trajectory Generator
========================================
Generates physically plausible trajectories for underrepresented maneuver classes
so the LSTM can be trained to 80%+ without downloading external datasets.

Why this works:
    After _normalise_trajectory, the model learns NORMALISED kinematic patterns:
      - Turns      → curved path with angular velocity in (vx, vy)
      - Braking    → decreasing speed magnitude, near-zero final velocity
      - Lane change → lateral displacement with maintained forward speed
    Synthetic trajectories with correct physics produce identical normalised
    features to real trajectories of the same class.  Each generated trajectory
    is validated by running the SAME derive_label() used during training, so
    only samples that match the intended class are kept.

Classes generated (underrepresented in nuScenes mini):
    turn_left / turn_right  — circular arc with random radius and speed
    braking                 — linear deceleration, final_speed > 0.25 m/s
    accelerating            — linear acceleration from low initial speed
    lane_change_left / right — sigmoid lateral displacement

Output:
    ~/prism_data/checkpoints/lstm_intent/synthetic_trajectories.npz
    Same format as bdd100k_trajectories.npz:
        features (N, SEQUENCE_LEN, INPUT_DIM)   float32
        labels   (N,)                           int64

Usage:
    cd ~/prism
    python scripts/generate_synthetic.py

Then retrain with synthetic data mixed in:
    python scripts/train_lstm_intent.py \
        --bdd-npz ~/prism_data/checkpoints/lstm_intent/synthetic_trajectories.npz
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.predictive_engine.lstm_intent import (
    MANEUVERS, SEQUENCE_LEN, INPUT_DIM, _normalise_trajectory,
)
from prism.utils.common import get_logger

logger = get_logger("generate_synthetic")

# ── Constants ─────────────────────────────────────────────────────────────────

DT         = 0.5            # nuScenes 2Hz — frame interval (seconds)
FUTURE_LEN = 5              # must match train_lstm_intent.py
N_TOTAL    = SEQUENCE_LEN + FUTURE_LEN

DEFAULT_OUT = Path("/home/koushik-test/prism_data/checkpoints/lstm_intent/synthetic_trajectories.npz")

# Samples to generate per class.
# nuScenes mini has ~40-500 raw samples per class (before 5× augmentation).
# Synthetic gets 1.5× augmentation in TrajectoryDataset (50% noise copy).
# These counts roughly equalise training exposure across classes.
N_PER_CLASS = {
    "turn_left":          1500,
    "turn_right":         1500,
    "braking":            2000,
    "accelerating":       2000,
    "lane_change_left":   2000,
    "lane_change_right":  2000,
}

# ── Label validation ───────────────────────────────────────────────────────────

def derive_label_local(future_pos: np.ndarray) -> int:
    """
    Exact copy of derive_label() from train_lstm_intent.py.
    Kept inline to avoid circular imports.
    Must stay in sync with the training script.
    """
    if len(future_pos) < 2:
        return MANEUVERS.index("constant_velocity")

    dt    = DT
    vels  = np.diff(future_pos, axis=0) / dt
    speeds      = np.linalg.norm(vels, axis=1)
    final_speed = speeds[-1]
    mean_speed  = speeds.mean()

    first_speed = float(np.linalg.norm(vels[0]))
    if first_speed > 0.05:
        heading_fwd = vels[0] / first_speed
    else:
        heading_fwd = np.array([0.0, 1.0], dtype=np.float32)
    heading_lat = np.array([-heading_fwd[1], heading_fwd[0]])

    net_disp    = future_pos[-1] - future_pos[0]
    net_forward = float(np.dot(net_disp, heading_fwd))
    net_lateral = float(np.dot(net_disp, heading_lat))

    headings = np.arctan2(vels[:, 0], vels[:, 1])
    if len(headings) > 1:
        heading_diffs = []
        for i in range(1, len(headings)):
            diff = (headings[i] - headings[i-1] + np.pi) % (2 * np.pi) - np.pi
            heading_diffs.append(diff)
        mean_turn   = float(np.mean(np.abs(heading_diffs)))
        signed_turn = float(np.mean(heading_diffs))
    else:
        mean_turn = signed_turn = 0.0

    if len(speeds) > 1:
        mean_accel = float(np.diff(speeds).mean() / dt)
    else:
        mean_accel = 0.0

    # ── Same cascade order as train_lstm_intent.py ──────────────────────────
    if net_forward < -0.5 and mean_speed > 0.1:
        return MANEUVERS.index("reversing")
    if mean_turn > np.radians(7.0):
        return MANEUVERS.index("turn_left" if signed_turn < 0 else "turn_right")
    if abs(net_lateral) > 0.8 and net_forward > 0.3:
        return MANEUVERS.index("lane_change_left" if net_lateral > 0 else "lane_change_right")
    if final_speed < 0.25 and mean_speed < 0.7:
        return MANEUVERS.index("stopping")
    if mean_accel < -0.5 and final_speed > 0.25:
        return MANEUVERS.index("braking")
    if mean_accel > 0.5:
        return MANEUVERS.index("accelerating")
    return MANEUVERS.index("constant_velocity")


def build_trajectory(headings_per_step: np.ndarray,
                     speeds_per_step:   np.ndarray,
                     noise_std: float = 0.02,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Build (N_TOTAL, 2) position array from per-step heading and speed arrays
    (both length N_TOTAL-1).  Adds small position noise.
    """
    pos = np.zeros((N_TOTAL, 2), dtype=np.float32)
    for t in range(1, N_TOTAL):
        pos[t, 0] = pos[t-1, 0] + speeds_per_step[t-1] * np.sin(headings_per_step[t-1]) * DT
        pos[t, 1] = pos[t-1, 1] + speeds_per_step[t-1] * np.cos(headings_per_step[t-1]) * DT
    if rng is not None and noise_std > 0:
        pos += rng.normal(0, noise_std, pos.shape).astype(np.float32)
    return pos


# ── Generators ────────────────────────────────────────────────────────────────

def gen_turns(direction: str, n: int, rng: np.random.Generator):
    """
    Circular arc turns.

    Bearing convention: arctan2(vx, vy) increases clockwise.
    Left turn  → heading decreases → omega < 0 → signed_turn < 0 → turn_left.
    Right turn → heading increases → omega > 0 → signed_turn > 0 → turn_right.
    """
    label  = MANEUVERS.index(direction)
    sign   = -1 if direction == "turn_left" else +1
    samples = []

    # Ensure mean_turn (heading change per step) is above the 7° threshold.
    # mean_turn_per_step = |omega| * DT, so |omega| > radians(7)/DT ≈ 0.244 rad/s
    omega_min = np.radians(8.0) / DT    # just above 7° threshold (rad/s)
    omega_max = np.radians(40.0) / DT   # max realistic turn rate (rad/s)

    attempts = 0
    while len(samples) < n:
        attempts += 1
        if attempts > n * 20:
            logger.warning(f"  {direction}: only generated {len(samples)}/{n} valid samples")
            break
        try:
            speed = rng.uniform(2.0, 14.0)        # m/s
            omega = sign * rng.uniform(omega_min, omega_max)  # rad/s
            h0    = rng.uniform(0, 2 * np.pi)     # random global heading

            # Turn starts somewhere in the input window so future is fully turning
            turn_start = int(rng.integers(0, SEQUENCE_LEN - 2))

            headings = np.empty(N_TOTAL - 1, dtype=np.float32)
            h = h0
            for t in range(N_TOTAL - 1):
                headings[t] = h
                if t >= turn_start:
                    h += omega * DT

            speeds = np.full(N_TOTAL - 1, speed, dtype=np.float32)
            # Slight speed variation during turn
            speeds[turn_start:] *= rng.uniform(0.85, 1.0)

            pos = build_trajectory(headings, speeds, noise_std=0.03, rng=rng)

            if derive_label_local(pos[SEQUENCE_LEN:]) != label:
                continue

            feat = _normalise_trajectory(pos[:SEQUENCE_LEN])
            samples.append((feat, label))

        except Exception:
            continue

    return samples


def gen_braking(n: int, rng: np.random.Generator):
    """
    Deceleration with final_speed > 0.25 m/s (so derive_label picks braking,
    not stopping).
    """
    label   = MANEUVERS.index("braking")
    samples = []
    attempts = 0

    while len(samples) < n:
        attempts += 1
        if attempts > n * 10:
            logger.warning(f"  braking: only {len(samples)}/{n}")
            break
        try:
            v0    = rng.uniform(4.0, 18.0)        # initial speed (m/s)
            v1    = rng.uniform(0.4, v0 * 0.6)   # final speed (still moving)
            h0    = rng.uniform(0, 2 * np.pi)
            slight_curve = rng.uniform(-0.02, 0.02)  # small heading drift

            # Linear deceleration profile over full N_TOTAL frames
            speed_profile = np.linspace(v0, v1, N_TOTAL - 1).astype(np.float32)
            headings      = (h0 + slight_curve * np.arange(N_TOTAL - 1)).astype(np.float32)

            pos = build_trajectory(headings, speed_profile, noise_std=0.02, rng=rng)

            if derive_label_local(pos[SEQUENCE_LEN:]) != label:
                continue

            feat = _normalise_trajectory(pos[:SEQUENCE_LEN])
            samples.append((feat, label))

        except Exception:
            continue

    return samples


def gen_accelerating(n: int, rng: np.random.Generator):
    """
    Acceleration from near-zero to higher speed (mean_accel > 0.5 m/s²).
    """
    label   = MANEUVERS.index("accelerating")
    samples = []
    attempts = 0

    while len(samples) < n:
        attempts += 1
        if attempts > n * 10:
            logger.warning(f"  accelerating: only {len(samples)}/{n}")
            break
        try:
            v0    = rng.uniform(0.0, 3.0)         # start nearly stationary
            v1    = rng.uniform(v0 + 2.0, 18.0)  # end speed
            h0    = rng.uniform(0, 2 * np.pi)
            slight_curve = rng.uniform(-0.01, 0.01)

            speed_profile = np.linspace(v0, v1, N_TOTAL - 1).astype(np.float32)
            headings      = (h0 + slight_curve * np.arange(N_TOTAL - 1)).astype(np.float32)

            pos = build_trajectory(headings, speed_profile, noise_std=0.02, rng=rng)

            if derive_label_local(pos[SEQUENCE_LEN:]) != label:
                continue

            feat = _normalise_trajectory(pos[:SEQUENCE_LEN])
            samples.append((feat, label))

        except Exception:
            continue

    return samples


def gen_lane_change(direction: str, n: int, rng: np.random.Generator):
    """
    Sigmoid lateral displacement (2-4m) at maintained forward speed.

    Coordinate check:
        heading_lat = [-fwd_y, fwd_x]  points LEFT of heading direction.
        net_lateral = dot(net_disp, heading_lat) > 0  →  moved LEFT → lane_change_left.
        Moving in +heading_lat direction = -fwd_y component → for north-heading
        actor (fwd=(0,1)): heading_lat=(-1,0), so LEFT means decreasing x.
    """
    label    = MANEUVERS.index(direction)
    # lat_sign = +1 → move in heading_lat direction (LEFT) → lane_change_left
    # lat_sign = -1 → move in -heading_lat direction (RIGHT) → lane_change_right
    lat_sign = +1 if direction == "lane_change_left" else -1
    samples  = []
    attempts = 0

    while len(samples) < n:
        attempts += 1
        if attempts > n * 10:
            logger.warning(f"  {direction}: only {len(samples)}/{n}")
            break
        try:
            speed     = rng.uniform(5.0, 20.0)       # forward speed (m/s)
            lat_total = rng.uniform(1.5, 4.0)         # total lateral displacement (m)
            h0        = rng.uniform(0, 2 * np.pi)

            # Forward direction
            fwd = np.array([np.sin(h0), np.cos(h0)], dtype=np.float32)
            # Left perpendicular: heading_lat = [-fwd_y, fwd_x]
            lat = np.array([-fwd[1], fwd[0]], dtype=np.float32)

            # Sigmoid lateral profile — most of the lane change in the future window
            t_vals   = np.linspace(-2.5, 2.5, N_TOTAL)
            sig      = 1.0 / (1.0 + np.exp(-t_vals))
            lat_disp = lat_sign * lat_total * (sig - sig[0])   # (N_TOTAL,) in metres

            # Build positions
            pos = np.zeros((N_TOTAL, 2), dtype=np.float32)
            for t in range(N_TOTAL):
                fwd_dist = speed * t * DT
                pos[t]   = fwd * fwd_dist + lat * lat_disp[t]

            noise = rng.normal(0, 0.03, pos.shape).astype(np.float32)
            pos  += noise

            if derive_label_local(pos[SEQUENCE_LEN:]) != label:
                continue

            feat = _normalise_trajectory(pos[:SEQUENCE_LEN])
            samples.append((feat, label))

        except Exception:
            continue

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="PRISM synthetic trajectory generator")
    p.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int,  default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("PRISM — Synthetic Trajectory Generation")
    print("=" * 60)

    generators = [
        ("turn_left",          lambda n: gen_turns("turn_left",          n, rng)),
        ("turn_right",         lambda n: gen_turns("turn_right",         n, rng)),
        ("braking",            lambda n: gen_braking(n, rng)),
        ("accelerating",       lambda n: gen_accelerating(n, rng)),
        ("lane_change_left",   lambda n: gen_lane_change("lane_change_left",  n, rng)),
        ("lane_change_right",  lambda n: gen_lane_change("lane_change_right", n, rng)),
    ]

    all_features: list = []
    all_labels:   list = []
    label_counts       = defaultdict(int)

    for cls_name, gen_fn in generators:
        n       = N_PER_CLASS[cls_name]
        samples = gen_fn(n)
        valid   = [(f, l) for f, l in samples
                   if f is not None and f.shape == (SEQUENCE_LEN, INPUT_DIM)]
        for feat, lbl in valid:
            all_features.append(feat)
            all_labels.append(lbl)
            label_counts[cls_name] += 1
        print(f"  {cls_name:<22}: {len(valid):>5} samples  "
              f"({'OK' if len(valid) >= n * 0.9 else 'LOW — check thresholds'})")

    if not all_features:
        print("ERROR: No samples generated.")
        return

    features = np.stack(all_features).astype(np.float32)   # (N, T, D)
    labels   = np.array(all_labels,  dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=features, labels=labels)

    print(f"\n  Total synthetic samples : {len(all_features)}")
    print(f"  Feature shape           : {features.shape}")
    print(f"  Saved to                : {args.out}")
    print()
    print("Next — retrain with synthetic data:")
    print(f"  python scripts/train_lstm_intent.py --bdd-npz {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
