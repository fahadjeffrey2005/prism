"""
PRISM — Benchmark Suite
=========================
Full quantitative evaluation of the PRISM pipeline.
Designed to produce paper-ready numbers for CVPR/IROS submission.

Five evaluation sections:

  1. DEPTH ACCURACY
     Compares metric depth estimates against nuScenes LiDAR ground truth.
     Metrics: MAE, RMSE, MedAE, MRE, δ<1.25, δ<1.5625, δ<2.0, lateral MAE

  2. LSTM INTENT ACCURACY
     Runs the LSTM predictor on val scenes (8-9), compares predictions against
     GT maneuver labels derived from future annotations.
     Metrics: overall accuracy, per-class accuracy, macro F1, confusion matrix

  3. PIPELINE LATENCY PROFILE
     Times every layer independently.
     Metrics: mean, p50, p95, p99 per layer, total pipeline throughput (FPS)

  4. SYSTEM ABLATION
     Runs 4 configurations: Full | No-VLM | No-LSTM | No-Arbitration
     Metrics: mean decision level, % frames ≥ CAUTION, latency per config

  5. VLM EFFICIENCY
     Measures event-driven trigger rate vs hypothetical fixed-rate baseline.
     Metrics: trigger rate, avg inference ms, % frames skipped, latency savings

Outputs:
  Terminal — formatted ASCII tables (copy-paste into paper appendix)
  JSON     — ~/prism_data/experiments/benchmark_results.json

Usage:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/run_benchmark.py

    # Val scenes only (faster)
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/run_benchmark.py \
        --scenes 8 9 --max-frames 40

    # Full dataset
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/run_benchmark.py \
        --scenes 0 1 2 3 4 5 6 7 8 9 --max-frames 999
"""

import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger
from prism.sensory_core.data_loader import NuScenesLoader
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.metric_depth import MetricDepthEngine, DepthValidator
from prism.world_model.world_model import WorldModel
from prism.predictive_engine.engine import PredictiveEngine
from prism.predictive_engine.decision import SmartDecisionEngine, DECISIONS
from prism.predictive_engine.lstm_intent import (
    MANEUVERS, SEQUENCE_LEN, _normalise_trajectory,
)
from prism.arbitration.core import ArbitrationCore, LEVEL_ORDER
from prism.planner.planner import AdaptivePlanner

logger = get_logger("benchmark")

NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"
CONFIG_PATH   = "config/default.yaml"
OUTPUT_PATH   = Path("/home/koushik-test/prism_data/experiments/benchmark_results.json")

# Val scenes in nuScenes mini
DEFAULT_SCENES = [8, 9]

LEVEL_TO_INT = {d: i for i, d in enumerate(LEVEL_ORDER)}


# ── LiDAR GT helper (from run_predictive.py) ─────────────────────────────────

def get_lidar_distance(nusc, sample_token: str, track_bbox, camera_name: str) -> Optional[float]:
    """Project LiDAR points into camera frame, return median depth inside bbox."""
    try:
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        import pyquaternion

        sample      = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = nusc.get("sample_data", lidar_token)
        lidar_path  = Path(nusc.dataroot) / lidar_data["filename"]
        pc          = LidarPointCloud.from_file(str(lidar_path))

        cam_token   = sample["data"][camera_name]
        cam_data    = nusc.get("sample_data", cam_token)
        cam_calib   = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        lidar_to_ego = pyquaternion.Quaternion(lidar_calib["rotation"]).rotation_matrix
        pc.rotate(lidar_to_ego)
        pc.translate(np.array(lidar_calib["translation"]))

        ego_to_cam = pyquaternion.Quaternion(cam_calib["rotation"]).rotation_matrix.T
        pc.translate(-np.array(cam_calib["translation"]))
        pc.rotate(ego_to_cam)

        depths   = pc.points[2, :]
        mask     = depths > 0.5
        pts_2d   = view_points(pc.points[:3, mask],
                               np.array(cam_calib["camera_intrinsic"]), True)

        x1, y1, x2, y2 = track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2
        in_box = (
            (pts_2d[0] >= x1) & (pts_2d[0] <= x2) &
            (pts_2d[1] >= y1) & (pts_2d[1] <= y2)
        )
        if in_box.sum() < 3:
            return None
        return float(np.median(depths[mask][in_box]))

    except Exception:
        return None


# ── LSTM GT label extraction ──────────────────────────────────────────────────

ANNOTATED_HZ = 2.0
FUTURE_LEN   = 5
MIN_TRAJ_LEN = SEQUENCE_LEN + FUTURE_LEN

DYNAMIC_CATEGORIES = {
    "vehicle.car", "vehicle.truck", "vehicle.bus",
    "vehicle.motorcycle", "vehicle.bicycle",
    "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.police_officer", "human.pedestrian.construction_worker",
}


def _derive_label(future_pos: np.ndarray) -> int:
    """GT maneuver label from future positions — actor-relative heading."""
    if len(future_pos) < 2:
        return 0
    dt    = 1.0 / ANNOTATED_HZ
    vels  = np.diff(future_pos, axis=0) / dt
    speeds = np.linalg.norm(vels, axis=1)
    final_speed = speeds[-1]

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
        diffs = [(headings[i] - headings[i-1] + np.pi) % (2*np.pi) - np.pi
                 for i in range(1, len(headings))]
        mean_turn   = float(np.mean(np.abs(diffs)))
        signed_turn = float(np.mean(diffs))
    else:
        mean_turn = signed_turn = 0.0

    if len(speeds) > 1:
        mean_accel = float(np.diff(speeds).mean() / dt)
    else:
        mean_accel = 0.0

    if net_forward < -0.5 and speeds.mean() > 0.1:
        return MANEUVERS.index("reversing")
    if final_speed < 0.3 and speeds.mean() < 1.0:
        return MANEUVERS.index("stopping")
    if mean_accel < -0.5 and final_speed > 0.3:
        return MANEUVERS.index("braking")
    if mean_accel > 0.5:
        return MANEUVERS.index("accelerating")
    if mean_turn > np.radians(10):
        return MANEUVERS.index("turn_left" if signed_turn > 0 else "turn_right")
    if abs(net_lateral) > 0.8 and net_forward > 1.0:
        return MANEUVERS.index("lane_change_left" if net_lateral > 0 else "lane_change_right")
    return MANEUVERS.index("constant_velocity")


def extract_lstm_gt_pairs(nusc, scene_indices: List[int]) -> List[Tuple[np.ndarray, int]]:
    """
    Extract (input_features, gt_label) pairs from nuScenes annotations.
    Returns list of (normalised_trajectory, label_index) for LSTM eval.
    """
    from nuscenes.nuscenes import NuScenes
    pairs = []
    scenes = nusc.scene

    for si in scene_indices:
        if si >= len(scenes):
            continue
        scene     = scenes[si]
        sample_tk = scene["first_sample_token"]
        ann_chains: Dict[str, List] = {}   # instance_token → list of ann

        # Collect all annotation chains for this scene
        while sample_tk:
            sample = nusc.get("sample", sample_tk)
            for ann_tk in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_tk)
                cat = ann["category_name"]
                if cat not in DYNAMIC_CATEGORIES:
                    continue
                inst = ann["instance_token"]
                if inst not in ann_chains:
                    ann_chains[inst] = []
                ann_chains[inst].append(ann)
            sample_tk = sample["next"]

        # Build trajectories and extract sliding windows
        for inst, anns in ann_chains.items():
            if len(anns) < MIN_TRAJ_LEN:
                continue
            positions = np.array([a["translation"][:2] for a in anns], dtype=np.float32)
            for start in range(len(positions) - MIN_TRAJ_LEN + 1):
                input_pos  = positions[start : start + SEQUENCE_LEN]
                future_pos = positions[start + SEQUENCE_LEN : start + SEQUENCE_LEN + FUTURE_LEN]
                feat = _normalise_trajectory(input_pos)
                if feat is None or feat.shape != (SEQUENCE_LEN, 6):
                    continue
                label = _derive_label(future_pos)
                pairs.append((feat, label))
    return pairs


# ── Table printing helpers ────────────────────────────────────────────────────

def hline(width: int = 72) -> str:
    return "─" * width


def section(title: str, width: int = 72) -> str:
    pad = width - len(title) - 4
    return f"{'═'*2}  {title}  {'═'*pad}"


def kv(label: str, value: str, width: int = 32) -> str:
    return f"  {label:<{width}} {value}"


def bar(value: float, total: float = 100.0, width: int = 30) -> str:
    filled = int(round(width * min(value, total) / max(total, 1e-6)))
    return "█" * filled + "░" * (width - filled)


# ── Section 1: Depth Accuracy ─────────────────────────────────────────────────

def run_depth_benchmark(
    cfg:         dict,
    scene_indices: List[int],
    max_frames:  int,
) -> dict:
    print(f"\n{section('1 / 5 — DEPTH ACCURACY')}")
    print(f"  Scenes: {scene_indices}  │  Max frames per scene: {max_frames}")

    loader        = NuScenesLoader(cfg)
    core          = SensoryCore(cfg)
    metric_engine = MetricDepthEngine(cfg)
    validator     = DepthValidator()
    nusc          = loader.nusc

    # Collect sample tokens per scene for LiDAR lookup
    sample_tokens_per_scene: Dict[int, List[str]] = {}
    for si in scene_indices:
        toks = []
        scene = nusc.scene[si]
        tk = scene["first_sample_token"]
        while tk:
            toks.append(tk)
            tk = nusc.get("sample", tk)["next"]
        sample_tokens_per_scene[si] = toks

    total_frames = 0
    for si in scene_indices:
        frame_idx = 0
        sample_tokens = sample_tokens_per_scene[si]
        for frame_data in loader.iter_primary_camera(scene_idx=si):
            if frame_idx >= max_frames:
                break
            image     = frame_data["image"]
            calib     = frame_data.get("calibration", {})
            cam_name  = frame_data["camera_name"]

            metric_engine.update_intrinsics(calib)
            sensory_frame = core.process(image, camera_name=cam_name,
                                         timestamp=frame_data["timestamp"])
            metric_dets, _ = metric_engine.process_frame(image, sensory_frame.detections)

            if frame_idx < len(sample_tokens):
                for md in metric_dets:
                    gt = get_lidar_distance(nusc, sample_tokens[frame_idx], md.bbox, cam_name)
                    if gt:
                        validator.add_sample(md.distance_m, gt,
                                             pred_lateral_m=md.lateral_m,
                                             method=md.method if hasattr(md, "method") else "unknown")

            frame_idx += 1
            total_frames += 1

        print(f"  Scene {si}: {frame_idx} frames, "
              f"{sum(1 for _ in range(1))} samples so far… "
              f"(total validator samples: {len(validator.dist_errors)})")

    metrics = validator.compute_metrics()
    if not metrics:
        print("  ⚠  No LiDAR GT matches found — check nuScenes root path.")
        return {}

    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {hline(40)}")
    print(f"  {'Samples (actor-frames)':<28} {metrics['n_samples']:>10d}")
    print(f"  {'MAE (m)':<28} {metrics['dist_MAE_m']:>10.3f}")
    print(f"  {'RMSE (m)':<28} {metrics['dist_RMSE_m']:>10.3f}")
    print(f"  {'Median AE (m)':<28} {metrics['dist_MedAE_m']:>10.3f}")
    print(f"  {'Mean Rel Error':<28} {metrics['dist_MRE']*100:>9.1f}%")
    print(f"  {'δ < 1.25':<28} {metrics['delta_1_25']*100:>9.1f}%")
    print(f"  {'δ < 1.5625':<28} {metrics['delta_1_5']*100:>9.1f}%")
    print(f"  {'δ < 2.00':<28} {metrics['delta_2']*100:>9.1f}%")
    print(f"  {'Lateral MAE (m)':<28} {metrics['lateral_MAE_m']:>10.3f}")
    print(f"  {'Lateral Median AE (m)':<28} {metrics['lateral_MedAE_m']:>10.3f}")
    if metrics.get("method_mae"):
        print(f"\n  {'By depth method'}")
        for meth, mae in metrics["method_mae"].items():
            print(f"    {meth:<22} MAE={mae:.3f}m")

    return metrics


# ── Section 2: LSTM Intent Accuracy ──────────────────────────────────────────

def run_lstm_benchmark(cfg: dict, scene_indices: List[int]) -> dict:
    print(f"\n{section('2 / 5 — LSTM INTENT ACCURACY')}")
    print(f"  Scenes: {scene_indices}")

    import torch
    from prism.predictive_engine.lstm_intent import load_lstm_model, LSTMIntentNet

    loader = NuScenesLoader(cfg)
    nusc   = loader.nusc

    # Load model
    ckpt_path = cfg.get("lstm_intent_checkpoint",
                        "/home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt")
    ckpt_path = str(Path(ckpt_path).expanduser())
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = load_lstm_model(ckpt_path, device)

    if model is None:
        print("  ⚠  LSTM checkpoint not found — skipping intent benchmark.")
        return {}

    model.eval()

    # Extract GT pairs
    print("  Extracting GT trajectories…")
    pairs = extract_lstm_gt_pairs(nusc, scene_indices)
    print(f"  {len(pairs)} trajectory windows extracted")

    if not pairs:
        print("  ⚠  No trajectory pairs found.")
        return {}

    # Run inference
    correct = 0
    per_class_correct  = defaultdict(int)
    per_class_total    = defaultdict(int)
    confusion = np.zeros((len(MANEUVERS), len(MANEUVERS)), dtype=int)
    all_preds, all_gts = [], []

    with torch.no_grad():
        batch_size = 64
        for start in range(0, len(pairs), batch_size):
            batch  = pairs[start : start + batch_size]
            feats  = np.stack([p[0] for p in batch])             # (B, T, 6)
            labels = np.array([p[1] for p in batch], dtype=int)
            x   = torch.tensor(feats, dtype=torch.float32).to(device)
            out = model(x)                                        # (B, 9)
            preds = out.argmax(dim=-1).cpu().numpy()

            for pred, gt in zip(preds, labels):
                if pred == gt:
                    correct += 1
                per_class_correct[gt]  += int(pred == gt)
                per_class_total[gt]    += 1
                confusion[gt, pred]    += 1
                all_preds.append(int(pred))
                all_gts.append(int(gt))

    overall_acc = correct / len(pairs)

    # Per-class accuracy + F1
    per_class_acc = {}
    per_class_f1  = {}
    for ci, name in enumerate(MANEUVERS):
        total = per_class_total[ci]
        if total == 0:
            per_class_acc[name] = None
            per_class_f1[name]  = None
            continue
        per_class_acc[name] = per_class_correct[ci] / total
        # Precision, recall, F1
        tp = confusion[ci, ci]
        fp = confusion[:, ci].sum() - tp
        fn = confusion[ci, :].sum() - tp
        prec   = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1     = 2 * prec * recall / max(prec + recall, 1e-6)
        per_class_f1[name] = float(f1)

    valid_f1  = [v for v in per_class_f1.values() if v is not None]
    macro_f1  = float(np.mean(valid_f1)) if valid_f1 else 0.0

    print(f"\n  Overall accuracy : {overall_acc*100:.1f}%   "
          f"Macro F1: {macro_f1*100:.1f}%")
    print(f"  Samples          : {len(pairs)}")
    print()
    print(f"  {'Class':<22} {'Acc':>7}  {'F1':>7}  {'Count':>6}  Bar")
    print(f"  {hline(60)}")
    for ci, name in enumerate(MANEUVERS):
        acc   = per_class_acc.get(name)
        f1    = per_class_f1.get(name)
        total = per_class_total[ci]
        if total == 0:
            print(f"  {name:<22}    —        —      0")
            continue
        acc_str = f"{acc*100:.1f}%" if acc is not None else "  —  "
        f1_str  = f"{f1*100:.1f}%" if f1  is not None else "  —  "
        b       = bar(acc * 100 if acc else 0, 100, 20)
        print(f"  {name:<22} {acc_str:>7}  {f1_str:>7}  {total:>6}  {b}")

    print(f"\n  Confusion matrix (rows=GT, cols=pred):")
    header = "  " + "".join(f"{n[:4]:>6}" for n in MANEUVERS)
    print(header)
    for ci, name in enumerate(MANEUVERS):
        row = "  " + f"{name[:6]:<6}" + "".join(f"{confusion[ci,cj]:>6}" for cj in range(len(MANEUVERS)))
        print(row)

    return {
        "overall_accuracy":  overall_acc,
        "macro_f1":          macro_f1,
        "per_class_accuracy": {k: v for k, v in per_class_acc.items() if v is not None},
        "per_class_f1":      {k: v for k, v in per_class_f1.items()  if v is not None},
        "n_samples":         len(pairs),
        "confusion_matrix":  confusion.tolist(),
    }


# ── Section 3: Latency Profile ────────────────────────────────────────────────

def run_latency_benchmark(
    cfg:          dict,
    scene_indices: List[int],
    max_frames:   int,
) -> dict:
    print(f"\n{section('3 / 5 — PIPELINE LATENCY PROFILE')}")
    print(f"  Scenes: {scene_indices}  │  Max frames per scene: {max_frames}")

    loader         = NuScenesLoader(cfg)
    core           = SensoryCore(cfg)
    metric_engine  = MetricDepthEngine(cfg)
    world          = WorldModel(cfg)
    predictor      = PredictiveEngine(cfg)
    decision_engine= SmartDecisionEngine()
    arbitrator     = ArbitrationCore(cfg)
    planner        = AdaptivePlanner(cfg)

    layer_times: Dict[str, List[float]] = {
        "sensory":    [],
        "metric":     [],
        "world":      [],
        "predictive": [],
        "decision":   [],
        "arbitration":[],
        "planner":    [],
        "total":      [],
    }

    for si in scene_indices:
        frame_idx = 0
        for frame_data in loader.iter_primary_camera(scene_idx=si):
            if frame_idx >= max_frames:
                break

            image     = frame_data["image"]
            calib     = frame_data.get("calibration", {})
            timestamp = frame_data["timestamp"]

            t_total = time.perf_counter()

            t0 = time.perf_counter()
            metric_engine.update_intrinsics(calib)
            sensory_frame = core.process(image, camera_name=frame_data["camera_name"],
                                         timestamp=timestamp)
            layer_times["sensory"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            metric_dets, _ = metric_engine.process_frame(image, sensory_frame.detections)
            layer_times["metric"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            world_state = world.update(sensory_frame, calibration=calib)
            layer_times["world"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            pred_state = predictor.update(world_state, metric_dets)
            layer_times["predictive"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            scene = decision_engine.assess(world_state, pred_state, metric_dets)
            layer_times["decision"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            arb = arbitrator.arbitrate(world_state, pred_state, scene, timestamp=timestamp)
            layer_times["arbitration"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            planner.plan(arb, metric_dets=metric_dets, timestamp=timestamp)
            layer_times["planner"].append((time.perf_counter() - t0) * 1000)

            layer_times["total"].append((time.perf_counter() - t_total) * 1000)
            frame_idx += 1

    print(f"\n  {'Layer':<16} {'Mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'Max':>8}  ms")
    print(f"  {hline(65)}")
    layer_order = ["sensory","metric","world","predictive","decision","arbitration","planner","total"]
    results = {}
    for layer in layer_order:
        times = np.array(layer_times[layer])
        if len(times) == 0:
            continue
        mean = np.mean(times)
        p50  = np.percentile(times, 50)
        p95  = np.percentile(times, 95)
        p99  = np.percentile(times, 99)
        mx   = np.max(times)
        sep  = "─" * 65 if layer == "total" else ""
        if sep:
            print(f"  {sep}")
        print(f"  {layer:<16} {mean:>8.2f}  {p50:>8.2f}  {p95:>8.2f}  {p99:>8.2f}  {mx:>8.2f}")
        results[layer] = {"mean": mean, "p50": p50, "p95": p95, "p99": p99, "max": mx}

    total_mean = results.get("total", {}).get("mean", 0)
    fps = 1000.0 / total_mean if total_mean > 0 else 0
    n = len(layer_times["total"])
    print(f"\n  Frames measured : {n}")
    print(f"  Throughput      : {fps:.1f} FPS  (non-VLM layers only)")
    results["fps"] = fps
    results["n_frames"] = n
    return results


# ── Section 4: System Ablation ────────────────────────────────────────────────

def run_ablation_benchmark(
    cfg:          dict,
    scene_indices: List[int],
    max_frames:   int,
) -> dict:
    print(f"\n{section('4 / 5 — SYSTEM ABLATION')}")
    print(f"  Scenes: {scene_indices}  │  Max frames per scene: {max_frames}")
    print(f"  Configurations: Full | No-VLM | No-LSTM | No-Arbitration\n")

    configs_to_run = [
        ("Full",            {"use_lstm": True,  "use_vlm": False, "use_arb": True}),
        ("No-VLM",          {"use_lstm": True,  "use_vlm": False, "use_arb": True}),
        ("No-LSTM",         {"use_lstm": False, "use_vlm": False, "use_arb": True}),
        ("No-Arbitration",  {"use_lstm": True,  "use_vlm": False, "use_arb": False}),
    ]
    # Note: VLM is always disabled in ablation (it requires GPU inference per frame;
    # the event-driven efficiency benchmark in Section 5 covers VLM separately)

    ablation_results = {}

    for config_name, flags in configs_to_run:
        print(f"  Running: {config_name}…")

        loader         = NuScenesLoader(cfg)
        core           = SensoryCore(cfg)
        metric_engine  = MetricDepthEngine(cfg)
        world          = WorldModel(cfg)
        predictor      = PredictiveEngine(cfg)
        decision_engine= SmartDecisionEngine()
        arbitrator     = ArbitrationCore(cfg) if flags["use_arb"] else None

        # No-LSTM: disable after construction so engine initialises cleanly
        if not flags["use_lstm"]:
            predictor._lstm_model  = None
            predictor._lstm_active = False

        decision_counts: Dict[str, int] = defaultdict(int)
        level_sum   = 0
        frame_count = 0
        times_ms    = []

        for si in scene_indices:
            frame_idx = 0
            for frame_data in loader.iter_primary_camera(scene_idx=si):
                if frame_idx >= max_frames:
                    break

                image     = frame_data["image"]
                calib     = frame_data.get("calibration", {})
                timestamp = frame_data["timestamp"]

                t0 = time.perf_counter()
                metric_engine.update_intrinsics(calib)
                sensory_frame = core.process(image, camera_name=frame_data["camera_name"],
                                             timestamp=timestamp)
                metric_dets, _ = metric_engine.process_frame(image, sensory_frame.detections)
                world_state  = world.update(sensory_frame, calibration=calib)
                pred_state   = predictor.update(world_state, metric_dets)
                scene_obj    = decision_engine.assess(world_state, pred_state, metric_dets)

                if arbitrator:
                    decision = arbitrator.arbitrate(
                        world_state, pred_state, scene_obj, timestamp=timestamp
                    ).action
                else:
                    # No arbitration: use SmartDecision directly
                    decision = scene_obj.decision

                times_ms.append((time.perf_counter() - t0) * 1000)
                decision_counts[decision] += 1
                level_sum += LEVEL_TO_INT.get(decision, 0)
                frame_count += 1
                frame_idx += 1

        mean_level  = level_sum / max(frame_count, 1)
        caution_pct = sum(v for k, v in decision_counts.items()
                          if LEVEL_TO_INT.get(k, 0) >= LEVEL_TO_INT["CAUTION"]) / max(frame_count, 1)
        liberal_pct = sum(v for k, v in decision_counts.items()
                          if LEVEL_TO_INT.get(k, 0) <= LEVEL_TO_INT["MONITOR"]) / max(frame_count, 1)
        mean_ms     = float(np.mean(times_ms)) if times_ms else 0.0

        ablation_results[config_name] = {
            "mean_decision_level": mean_level,
            "pct_caution_or_above": caution_pct,
            "pct_monitor_or_below": liberal_pct,
            "decision_counts": dict(decision_counts),
            "mean_latency_ms": mean_ms,
            "n_frames": frame_count,
        }

        print(f"    Mean level: {mean_level:.2f}/7  │  "
              f"≥CAUTION: {caution_pct*100:.1f}%  │  "
              f"≤MONITOR: {liberal_pct*100:.1f}%  │  "
              f"{mean_ms:.1f}ms/frame")

    # Print comparison table
    print(f"\n  {'Config':<20} {'MeanLvl':>9}  {'≥CAUTION':>9}  {'≤MONITOR':>9}  {'ms/frame':>9}")
    print(f"  {hline(62)}")
    for config_name, r in ablation_results.items():
        print(
            f"  {config_name:<20} "
            f"{r['mean_decision_level']:>9.2f}  "
            f"{r['pct_caution_or_above']*100:>8.1f}%  "
            f"{r['pct_monitor_or_below']*100:>8.1f}%  "
            f"{r['mean_latency_ms']:>8.1f}"
        )

    print(f"\n  Decision distribution per config:")
    for config_name, r in ablation_results.items():
        dist = r["decision_counts"]
        total = r["n_frames"]
        parts = [f"{k}:{v/total*100:.0f}%" for k, v in dist.items() if v > 0]
        print(f"    {config_name:<20} {' | '.join(parts)}")

    return ablation_results


# ── Section 5: VLM Efficiency ─────────────────────────────────────────────────

def run_vlm_efficiency_benchmark(cfg: dict, scene_indices: List[int], max_frames: int) -> dict:
    print(f"\n{section('5 / 5 — VLM EFFICIENCY')}")
    print(f"  Measures event-driven VLM trigger rate vs fixed-rate baseline.")
    print(f"  VLM not invoked — using trigger decision logic only.\n")

    loader         = NuScenesLoader(cfg)
    core           = SensoryCore(cfg)
    metric_engine  = MetricDepthEngine(cfg)
    world          = WorldModel(cfg)
    predictor      = PredictiveEngine(cfg)
    decision_engine= SmartDecisionEngine()
    arbitrator     = ArbitrationCore(cfg)

    # VLM trigger config
    vlm_cfg             = cfg.get("vlm", {})
    divergence_threshold = vlm_cfg.get("divergence_threshold", 0.35)
    max_silence_s        = vlm_cfg.get("max_silence_s",        4.0)
    cooldown_s           = vlm_cfg.get("cooldown_s",           0.4)
    camera_fps           = cfg.get("sensory_core", {}).get("sampling", {}).get("camera_fps", 12)

    total_frames    = 0
    vlm_triggers    = 0
    last_vlm_time   = 0.0
    last_vlm_forced = 0.0

    divergence_events   = 0
    silence_events      = 0
    cooldown_skips      = 0

    level_history: List[int] = []

    for si in scene_indices:
        frame_idx = 0
        for frame_data in loader.iter_primary_camera(scene_idx=si):
            if frame_idx >= max_frames:
                break

            image     = frame_data["image"]
            calib     = frame_data.get("calibration", {})
            timestamp = frame_data["timestamp"]

            metric_engine.update_intrinsics(calib)
            sensory_frame = core.process(image, camera_name=frame_data["camera_name"],
                                         timestamp=timestamp)
            metric_dets, _ = metric_engine.process_frame(image, sensory_frame.detections)
            world_state   = world.update(sensory_frame, calibration=calib)
            pred_state    = predictor.update(world_state, metric_dets)
            scene_obj     = decision_engine.assess(world_state, pred_state, metric_dets)
            arb           = arbitrator.arbitrate(world_state, pred_state, scene_obj,
                                                  timestamp=timestamp)

            level_history.append(LEVEL_TO_INT.get(arb.action, 0))

            # Simulate VLM trigger logic
            age_since_vlm = timestamp - last_vlm_time
            in_cooldown   = age_since_vlm < cooldown_s

            # Divergence: large swing in decision level vs recent history
            should_trigger_divergence = False
            if len(level_history) >= 3:
                recent_std = float(np.std(level_history[-3:]))
                if recent_std > divergence_threshold * 7:  # normalise to 0-7 scale
                    should_trigger_divergence = True

            # Silence: too long without VLM
            should_trigger_silence = (age_since_vlm > max_silence_s)

            would_trigger = (should_trigger_divergence or should_trigger_silence)

            if would_trigger:
                if in_cooldown:
                    cooldown_skips += 1
                else:
                    vlm_triggers += 1
                    if should_trigger_divergence:
                        divergence_events += 1
                    if should_trigger_silence:
                        silence_events += 1
                    last_vlm_time = timestamp

            total_frames += 1
            frame_idx += 1

    trigger_rate    = vlm_triggers / max(total_frames, 1)
    frames_skipped  = 1.0 - trigger_rate
    # Fixed-rate baseline: VLM every N frames at camera_fps
    fixed_rate_1hz  = 1.0 / camera_fps       # every frame
    fixed_rate_2hz  = 2.0 / camera_fps       # every other frame
    saving_vs_1fps  = 1.0 - trigger_rate / max(fixed_rate_1hz, 1e-6)

    # Hypothetical latency saved (assuming ~800ms VLM inference)
    assumed_vlm_ms  = 800.0
    saved_ms_per_frame = (fixed_rate_1hz - trigger_rate) * assumed_vlm_ms

    print(f"  {'Metric':<40} {'Value':>12}")
    print(f"  {hline(55)}")
    print(f"  {'Total frames':<40} {total_frames:>12d}")
    print(f"  {'VLM trigger count':<40} {vlm_triggers:>12d}")
    print(f"  {'Trigger rate':<40} {trigger_rate*100:>11.1f}%")
    print(f"  {'Frames skipped (no VLM call)':<40} {frames_skipped*100:>11.1f}%")
    print(f"  {'Divergence events':<40} {divergence_events:>12d}")
    print(f"  {'Silence events (forced)':<40} {silence_events:>12d}")
    print(f"  {'Cooldown skips':<40} {cooldown_skips:>12d}")
    print(f"  {'Baseline: VLM every frame (1× fps)':<40} {fixed_rate_1hz*100:>11.1f}%")
    print(f"  {'Latency saved vs 1×fps baseline':<40} {saving_vs_1fps*100:>11.1f}%")
    print(f"  {'Est. saved time (800ms VLM, /frame)':<40} {saved_ms_per_frame:>10.1f}ms")

    return {
        "total_frames":        total_frames,
        "vlm_triggers":        vlm_triggers,
        "trigger_rate":        trigger_rate,
        "frames_skipped_pct":  frames_skipped,
        "divergence_events":   divergence_events,
        "silence_events":      silence_events,
        "cooldown_skips":      cooldown_skips,
        "saving_vs_1fps_pct":  saving_vs_1fps,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="PRISM Benchmark Suite")
    p.add_argument("--scenes",     type=int, nargs="+", default=DEFAULT_SCENES,
                   help="Scene indices to evaluate (default: 8 9)")
    p.add_argument("--max-frames", type=int, default=40,
                   help="Max frames per scene (default: 40)")
    p.add_argument("--config",     default=CONFIG_PATH)
    p.add_argument("--skip-depth",    action="store_true")
    p.add_argument("--skip-lstm",     action="store_true")
    p.add_argument("--skip-latency",  action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-vlm",      action="store_true")
    p.add_argument("--output", default=str(OUTPUT_PATH))
    args = p.parse_args()

    print("=" * 72)
    print("PRISM — BENCHMARK SUITE")
    print("=" * 72)
    print(f"  Scenes     : {args.scenes}")
    print(f"  Max frames : {args.max_frames} per scene")

    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = {}
        cfg.setdefault("data", {})["nuscenes_root"] = NUSCENES_ROOT

    results = {
        "scenes":     args.scenes,
        "max_frames": args.max_frames,
    }

    t_bench_start = time.time()

    if not args.skip_depth:
        results["depth"] = run_depth_benchmark(cfg, args.scenes, args.max_frames)

    if not args.skip_lstm:
        results["lstm"] = run_lstm_benchmark(cfg, args.scenes)

    if not args.skip_latency:
        results["latency"] = run_latency_benchmark(cfg, args.scenes, args.max_frames)

    if not args.skip_ablation:
        results["ablation"] = run_ablation_benchmark(cfg, args.scenes, args.max_frames)

    if not args.skip_vlm:
        results["vlm_efficiency"] = run_vlm_efficiency_benchmark(
            cfg, args.scenes, args.max_frames)

    total_time = time.time() - t_bench_start
    results["benchmark_time_s"] = total_time

    # Save JSON
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'═'*72}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Total time : {total_time:.1f}s")
    print(f"  Results    : {out_path}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
