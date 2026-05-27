"""
PRISM — LSTM Intent Model Training
====================================
Extracts actor trajectories from nuScenes, derives maneuver labels from
future motion kinematics, augments the data, and trains a 2-layer LSTM.

Run on Jetson overnight:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/train_lstm_intent.py

    # Custom output path
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/train_lstm_intent.py \
        --out /home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt

Data pipeline:
    1. Follow per-instance annotation chains across all scenes
    2. Transform each trajectory into ego-relative, heading-normalised features
    3. Label the INPUT window from the FUTURE trajectory (what actually happened next)
    4. Augment: Gaussian noise + speed scaling + random rotation
    5. Train with class-weighted cross-entropy (rare maneuvers upweighted)

Label derivation (from 5-frame future window at 2Hz = 2.5s lookahead):
    stopping          — final speed < 0.3 m/s
    braking           — mean deceleration < -0.4 m/s²
    accelerating      — mean acceleration > +0.4 m/s²
    turn_left/right   — mean heading change > 15°
    lane_change_*     — significant lateral displacement (>0.8m) with forward motion
    reversing         — net forward displacement < -0.3m
    constant_velocity — none of the above
"""

import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from prism.predictive_engine.lstm_intent import (
    LSTMIntentNet, TrajectoryBuffer, MANEUVERS,
    SEQUENCE_LEN, INPUT_DIM, _normalise_trajectory,
)

NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"
DEFAULT_OUT   = "/home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt"

FUTURE_LEN    = 5      # frames to look ahead for labelling
MIN_TRAJ_LEN  = SEQUENCE_LEN + FUTURE_LEN
ANNOTATED_HZ  = 2.0   # nuScenes annotations at 2Hz

# Classes to include (ignore static objects)
DYNAMIC_CATEGORIES = {
    "vehicle.car", "vehicle.truck", "vehicle.bus",
    "vehicle.motorcycle", "vehicle.bicycle",
    "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.police_officer", "human.pedestrian.construction_worker",
}

VEHICLE_CATS = {c for c in DYNAMIC_CATEGORIES if c.startswith("vehicle")}
PED_CATS     = {c for c in DYNAMIC_CATEGORIES if c.startswith("human")}


# ── Trajectory extraction ─────────────────────────────────────────────────────

def extract_trajectories(nusc: NuScenes) -> List[dict]:
    """
    Follow every annotation chain and return full trajectories in global frame.
    Returns list of dicts:
        {
            "category": str,
            "class":    str   (simplified: car/pedestrian/bicycle/etc),
            "positions": np.ndarray  (N, 3)  xyz in global frame,
            "timestamps": list[float]
        }
    """
    trajectories = []
    seen_instances = set()

    for ann in nusc.sample_annotation:
        itoken = ann["instance_token"]
        if itoken in seen_instances:
            continue
        if ann["prev"] != "":
            continue   # start from the first annotation of each instance

        cat = ann["category_name"]
        if not any(cat.startswith(c.split(".")[0] + "." + c.split(".")[1])
                   for c in DYNAMIC_CATEGORIES):
            continue

        seen_instances.add(itoken)

        # Follow annotation chain forward
        positions  = []
        timestamps = []
        cur_token  = ann["token"]

        while cur_token:
            cur = nusc.get("sample_annotation", cur_token)
            sample = nusc.get("sample", cur["sample_token"])
            positions.append(cur["translation"])    # [x, y, z] global
            timestamps.append(sample["timestamp"] / 1e6)
            cur_token = cur["next"] if cur["next"] else None

        if len(positions) < MIN_TRAJ_LEN:
            continue

        # Determine simplified class name
        if cat.startswith("vehicle.car"):
            cls = "car"
        elif cat.startswith("vehicle.truck") or cat.startswith("vehicle.bus"):
            cls = "truck"
        elif cat.startswith("vehicle.motorcycle"):
            cls = "motorcycle"
        elif cat.startswith("vehicle.bicycle"):
            cls = "bicycle"
        elif cat.startswith("human.pedestrian"):
            cls = "pedestrian"
        else:
            cls = "other"

        trajectories.append({
            "category":   cat,
            "class":      cls,
            "positions":  np.array(positions, dtype=np.float32),   # (N, 3)
            "timestamps": timestamps,
        })

    print(f"  Extracted {len(trajectories)} valid trajectories")
    counter = Counter(t["class"] for t in trajectories)
    for cls, count in sorted(counter.items()):
        print(f"    {cls:<15}: {count}")
    return trajectories


# ── Label derivation ──────────────────────────────────────────────────────────

def derive_label(future_pos: np.ndarray) -> int:
    """
    Given future positions (M, 2) in xy global plane, return maneuver index.
    Uses velocity and heading change over the future window.

    IMPORTANT: forward/lateral are defined relative to the actor's OWN initial
    heading, NOT the global y-axis. This prevents cars heading SW from being
    mislabelled as 'reversing' just because their y-displacement is negative.
    """
    if len(future_pos) < 2:
        return 0   # constant_velocity fallback

    # Velocities at each step (finite diff, m/s assuming 2Hz)
    dt = 1.0 / ANNOTATED_HZ
    vels = np.diff(future_pos, axis=0) / dt    # (M-1, 2)

    speeds      = np.linalg.norm(vels, axis=1)  # (M-1,)
    final_speed = speeds[-1]
    mean_speed  = speeds.mean()

    # Actor's initial heading unit vector (from first motion step)
    # Falls back to [0,1] (global y) if actor is stationary
    first_vel = vels[0]
    first_speed = float(np.linalg.norm(first_vel))
    if first_speed > 0.05:
        heading_fwd  = first_vel / first_speed          # forward unit vector
    else:
        heading_fwd  = np.array([0.0, 1.0], dtype=np.float32)
    heading_lat = np.array([-heading_fwd[1], heading_fwd[0]])  # left perpendicular

    # Net displacement projected onto actor-relative axes
    net_disp    = future_pos[-1] - future_pos[0]
    net_forward = float(np.dot(net_disp, heading_fwd))   # +ve = actor moved forward
    net_lateral = float(np.dot(net_disp, heading_lat))   # +ve = actor moved left

    # Heading at each step (in radians) — for turn detection
    headings = np.arctan2(vels[:, 0], vels[:, 1])

    # Mean heading change per step (turn rate)
    if len(headings) > 1:
        heading_diffs = []
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i - 1]
            # Wrap to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            heading_diffs.append(diff)
        mean_turn   = float(np.mean(np.abs(heading_diffs)))
        signed_turn = float(np.mean(heading_diffs))
    else:
        mean_turn   = 0.0
        signed_turn = 0.0

    # Acceleration over future window
    if len(speeds) > 1:
        accels     = np.diff(speeds) / dt   # m/s²
        mean_accel = float(accels.mean())
    else:
        mean_accel = 0.0

    # ── Decision tree ──────────────────────────────

    # Reversing: actor moved backward along its own heading axis
    if net_forward < -0.5 and mean_speed > 0.1:
        return MANEUVERS.index("reversing")

    # Stopping: final speed near zero.
    # Threshold raised to 3.0 m/s so hard-braking-to-stop (mean≈2.5 m/s)
    # is captured. Was 1.0 m/s — too tight, caused braking-to-stop to fall
    # through to constant_velocity.
    if final_speed < 0.3 and mean_speed < 3.0:
        return MANEUVERS.index("stopping")

    # Strong deceleration
    if mean_accel < -0.5 and final_speed > 0.3:
        return MANEUVERS.index("braking")

    # Strong acceleration
    if mean_accel > 0.5:
        return MANEUVERS.index("accelerating")

    # Turning — significant heading change (> 10 deg / step)
    if mean_turn > np.radians(10.0):
        if signed_turn > 0:
            return MANEUVERS.index("turn_left")
        else:
            return MANEUVERS.index("turn_right")

    # Lane change — significant lateral displacement with forward motion.
    # heading_lat points LEFT, so net_lateral > 0 means actor moved LEFT.
    # BUG-FIX: was net_lateral < 0 → lane_change_left (backwards — negative
    # means rightward). Matches run_benchmark._derive_label convention.
    if abs(net_lateral) > 0.8 and net_forward > 0.3:
        if net_lateral > 0:
            return MANEUVERS.index("lane_change_left")
        else:
            return MANEUVERS.index("lane_change_right")

    return MANEUVERS.index("constant_velocity")


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Sliding window dataset over nuScenes trajectories + optional BDD100K npz.
    Each sample: (features (T, 6), label int)
    """

    def __init__(
        self,
        trajectories: List[dict],
        augment:      bool = True,
        stride:       int  = 1,
        bdd_npz_path: Optional[str] = None,
    ):
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, int]] = []

        # ── nuScenes trajectories ──────────────────────────────────────────
        for traj in trajectories:
            positions  = traj["positions"][:, :2]  # (N, 2) xy only
            n          = len(positions)

            for start in range(0, n - MIN_TRAJ_LEN + 1, stride):
                window     = positions[start : start + SEQUENCE_LEN]
                future     = positions[start + SEQUENCE_LEN : start + SEQUENCE_LEN + FUTURE_LEN]
                label      = derive_label(future)
                features   = _normalise_trajectory(window)
                self.samples.append((features, label))

                # Augmentation: add noise copies
                if augment:
                    for _ in range(2):
                        noisy  = window + np.random.randn(*window.shape).astype(np.float32) * 0.05
                        aug_ft = _normalise_trajectory(noisy)
                        self.samples.append((aug_ft, label))

                    # Speed scaling: compress/stretch trajectory
                    for scale in [0.7, 1.3]:
                        scaled = window * scale
                        s_ft   = _normalise_trajectory(scaled)
                        self.samples.append((s_ft, label))

        nuscenes_count = len(self.samples)

        # ── BDD100K pre-extracted windows (optional) ───────────────────────
        bdd_count = 0
        if bdd_npz_path:
            bdd_path = Path(bdd_npz_path)
            if bdd_path.exists():
                data = np.load(bdd_path, allow_pickle=True)
                bdd_feats  = data["features"]   # (N, T, 6)
                bdd_labels = data["labels"]      # (N,)
                for feat, label in zip(bdd_feats, bdd_labels):
                    if feat.shape == (SEQUENCE_LEN, 6):
                        self.samples.append((feat.astype(np.float32), int(label)))
                        bdd_count += 1
                        # Light augmentation on BDD too
                        if augment and random.random() < 0.5:
                            noisy = feat + np.random.randn(*feat.shape).astype(np.float32) * 0.03
                            self.samples.append((noisy, int(label)))
                            bdd_count += 1
                print(f"  BDD100K windows loaded: {bdd_count}")
            else:
                print(f"  ⚠  BDD100K npz not found: {bdd_path} — training on nuScenes only")

        print(f"  Dataset: {len(self.samples)} samples  "
              f"(nuScenes={nuscenes_count}  BDD100K={bdd_count})")
        label_counts = Counter(s[1] for s in self.samples)
        for idx, cnt in sorted(label_counts.items()):
            pct = 100 * cnt / len(self.samples)
            print(f"    {MANEUVERS[idx]:<22}: {cnt:4d} ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for balanced training."""
        counts = Counter(s[1] for s in self.samples)
        total  = len(self.samples)
        weights = torch.zeros(NUM_CLASSES := len(MANEUVERS))
        for i in range(NUM_CLASSES):
            weights[i] = total / max(counts.get(i, 1), 1)
        return weights / weights.sum() * NUM_CLASSES


# ── Focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.
    FL(p) = -alpha * (1 - p)^gamma * log(p)

    gamma=0 → standard cross-entropy
    gamma=2 → strongly down-weights easy/frequent examples,
              forcing the model to focus on rare hard classes
              like turn_left (13 samples) and braking (8 samples).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # per-class weights (inverse frequency)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)           # (B, C)
        probs     = log_probs.exp()                             # (B, C)
        gathered_log  = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        gathered_prob = probs.gather(1,     targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_factor  = (1.0 - gathered_prob) ** self.gamma
        loss          = -focal_factor * gathered_log            # (B,)

        if self.weight is not None:
            w    = self.weight[targets]
            loss = loss * w

        return loss.mean()


# ── Confusion-penalty loss ────────────────────────────────────────────────────

class ConfusionPenaltyCELoss(nn.Module):
    """
    Cross-entropy with label smoothing + explicit confusion penalties.

    For each (gt, wrong_pred) pair, when gt is the true class and the model
    assigns high probability to wrong_pred, the loss is multiplied by
    penalty_multiplier. This directly discourages the known confusion pairs:
        stopping → reversing  (×3)
        lane_change_left ↔ lane_change_right (×3)

    Label smoothing 0.05 prevents overconfidence on dominant classes
    (constant_velocity dominated Phase 1 training at 36.7%).
    """

    def __init__(
        self,
        weight:          Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        penalty_pairs:   list  = None,
        device:          str   = "cpu",
    ):
        super().__init__()
        self.weight          = weight
        self.label_smoothing = label_smoothing
        self.penalty_pairs   = penalty_pairs or []
        self.n_classes       = len(MANEUVERS)

        # Build penalty matrix: penalty_mat[gt, wrong] = extra_multiplier
        pm = torch.ones(self.n_classes, self.n_classes, device=device)
        for gt, wrong, mult in self.penalty_pairs:
            pm[gt, wrong] = mult
        self.register_buffer = None   # no buffers needed
        self.penalty_mat = pm

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy with label smoothing
        base_loss = nn.functional.cross_entropy(
            logits, targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = "none",
        )   # (B,)

        # Confusion penalty: for each sample, scale loss by the penalty
        # associated with (true_class, predicted_class)
        with torch.no_grad():
            preds    = logits.argmax(dim=-1)         # (B,)
            penalties = self.penalty_mat[targets, preds]  # (B,)

        return (base_loss * penalties).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 65)
    print("PRISM — LSTM Intent Model Training")
    print("=" * 65)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device : {device}")

    # Load nuScenes
    print(f"\nLoading nuScenes from {args.data_root} ...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    # Extract trajectories
    print("\nExtracting trajectories ...")
    all_trajs = extract_trajectories(nusc)

    if not all_trajs:
        print("ERROR: No valid trajectories extracted. Check dataset path.")
        return

    # Train / val split by instance (not random sample split)
    random.seed(42)
    random.shuffle(all_trajs)
    split = int(0.85 * len(all_trajs))
    train_trajs = all_trajs[:split]
    val_trajs   = all_trajs[split:]
    print(f"\n  Train trajectories: {len(train_trajs)}")
    print(f"  Val   trajectories: {len(val_trajs)}")

    # Datasets
    bdd_npz = args.bdd_npz if args.bdd_npz else None
    print("\nBuilding training dataset ...")
    train_ds = TrajectoryDataset(train_trajs, augment=True,  stride=1, bdd_npz_path=bdd_npz)
    print("\nBuilding val dataset ...")
    val_ds   = TrajectoryDataset(val_trajs,   augment=False, stride=2)

    # Weighted sampler — oversample rare maneuvers
    sample_weights = [train_ds.class_weights()[label].item()
                      for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = LSTMIntentNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Loss — class-weighted cross entropy with label smoothing.
    # Label smoothing 0.05 reduces overconfidence on dominant classes.
    # confusion_penalty extra-penalises stopping→reversing and
    # lane_change_left↔right swaps (the two confirmed confusion pairs).
    class_w   = train_ds.class_weights().to(device)
    criterion = ConfusionPenaltyCELoss(
        weight          = class_w,
        label_smoothing = 0.05,
        penalty_pairs   = [
            # (gt_class, wrong_pred_class, extra_penalty_multiplier)
            (MANEUVERS.index("stopping"),         MANEUVERS.index("reversing"),        3.0),
            (MANEUVERS.index("lane_change_left"),  MANEUVERS.index("lane_change_right"),3.0),
            (MANEUVERS.index("lane_change_right"), MANEUVERS.index("lane_change_left"), 3.0),
        ],
        device = device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Output dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history      = []

    print(f"\nTraining for {args.epochs} epochs ...")
    print(f"{'Epoch':<7} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * x.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += x.size(0)

        scheduler.step()
        t_loss = train_loss / train_total
        t_acc  = 100.0 * train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits     = model(x)
                loss       = criterion(logits, y)
                val_loss  += loss.item() * x.size(0)
                preds      = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += x.size(0)

        v_loss = val_loss / val_total if val_total > 0 else float("nan")
        v_acc  = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        history.append({
            "epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
            "val_loss": v_loss, "val_acc": v_acc
        })

        marker = " ← best" if v_acc > best_val_acc else ""
        print(f"{epoch:<7} {t_loss:<12.4f} {t_acc:<12.1f} {v_loss:<12.4f} {v_acc:<12.1f}{marker}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": v_acc,
                "val_loss": v_loss,
                "train_acc": t_acc,
                "train_loss": t_loss,
                "maneuvers": MANEUVERS,
                "input_dim": INPUT_DIM,
                "seq_len": SEQUENCE_LEN,
            }, out_path)

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {out_path}")

    # Per-class breakdown on full val set
    print("\nPer-class accuracy on val set:")
    model.eval()
    class_correct = Counter()
    class_total   = Counter()
    with torch.no_grad():
        for x, y in val_loader:
            x, y   = x.to(device), y.to(device)
            preds  = model(x).argmax(dim=1)
            for gt, pred in zip(y.cpu().numpy(), preds.cpu().numpy()):
                class_total[int(gt)] += 1
                if gt == pred:
                    class_correct[int(gt)] += 1

    for idx, name in enumerate(MANEUVERS):
        total = class_total[idx]
        if total > 0:
            acc = 100.0 * class_correct[idx] / total
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {name:<22} {bar}  {acc:5.1f}%  ({class_correct[idx]}/{total})")

    # Save training history
    hist_path = out_path.parent / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history: {hist_path}")
    print("=" * 65)


def parse_args():
    p = argparse.ArgumentParser(description="PRISM LSTM intent model training")
    p.add_argument("--data-root",  default=NUSCENES_ROOT)
    p.add_argument("--version",    default="v1.0-mini")
    p.add_argument("--out",        default=DEFAULT_OUT)
    p.add_argument("--epochs",     type=int,   default=120)
    p.add_argument("--lr",         type=float, default=3e-3)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--bdd-npz",    default=None,
                   help="Path to BDD100K pre-extracted trajectories .npz "
                        "(from scripts/prepare_bdd100k.py). Optional.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
