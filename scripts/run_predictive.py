"""
PRISM — Full Pipeline Validation
==================================
Runs all layers together:
    Layer 1 — Sensory Core
    Layer 1b— Metric Depth Engine
    Layer 2 — World Model
    Layer 3 — Predictive Engine

Output per frame:
    LEFT:   Camera + tracked actors with METRIC distances
    CENTER: BEV grid with predicted trajectories overlaid
    RIGHT:  Prediction panel — maneuver probs + recommended action

Also runs depth validation against nuScenes LiDAR ground truth
and prints accuracy metrics at the end.

Usage:
    python scripts/run_predictive.py
    python scripts/run_predictive.py --scene 2 --max-frames 40
    python scripts/run_predictive.py --validate  # full GT comparison
"""

import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger, CLASS_COLORS, ARBITRATION_COLORS
from prism.sensory_core.core import SensoryCore
from prism.sensory_core.data_loader import NuScenesLoader
from prism.sensory_core.metric_depth import (
    MetricDepthEngine, DepthValidator, CameraIntrinsics
)
from prism.world_model.world_model import WorldModel, WorldModelVisualizer
from prism.predictive_engine.engine import PredictiveEngine, MANEUVERS
from prism.predictive_engine.decision import SmartDecisionEngine, DECISIONS, PED_STATES
from prism.semantic_reasoner.reasoner import SemanticReasoner
from prism.arbitration.core import ArbitrationCore
from prism.planner.planner import AdaptivePlanner

logger = get_logger("run_predictive")


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM Full Pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="Run depth validation against LiDAR GT")
    return parser.parse_args()


# ── Visualisation helpers ─────────────────────────────────────────────────────

def draw_metric_detections(image: np.ndarray, metric_dets: list) -> np.ndarray:
    """Clean camera view — box colour = threat zone, single label line."""
    out = image.copy()
    ZONE_COLORS = {
        "CRITICAL": (0,   0,   255),
        "CLOSE":    (0,   140, 255),
        "MEDIUM":   (0,   220, 220),
        "FAR":      (80,  200, 80),
    }
    for md in metric_dets:
        b = md.bbox
        color = ZONE_COLORS.get(md.threat_zone, (180, 180, 180))
        # Box — thicker for closer objects
        thickness = 3 if md.threat_zone in ("CRITICAL", "CLOSE") else 1
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, thickness)
        # Single clean label: class + distance
        label = f"{md.class_name}  {md.distance_m:.1f}m"
        lx, ly = int(b.x1), max(int(b.y1) - 7, 12)
        # Dark background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return out


def draw_bev(pred_state, world_state, size: int = 485) -> np.ndarray:
    """
    Clean BEV grid.
    - Dark background
    - Distance rings at 10m / 20m / 30m
    - Actors as coloured dots with track ID
    - Short trajectory as a single clean line
    - Ego vehicle as white arrow at centre
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 28)   # dark navy background
    cx, cy = size // 2, size // 2
    scale = size / 100.0        # 100m total range → pixels

    # Distance rings
    for dist_m, label in [(10, "10m"), (20, "20m"), (30, "30m"), (50, "50m")]:
        r = int(dist_m * scale / 2)  # /2 because scale covers ±50m
        alpha_val = 60 if dist_m < 50 else 35
        ring_color = (alpha_val, alpha_val, alpha_val + 20)
        cv2.circle(canvas, (cx, cy), r, ring_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx + r + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 100), 1, cv2.LINE_AA)

    # Cardinal labels
    cv2.putText(canvas, "FRONT", (cx - 22, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "L", (6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 110), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R", (size - 16, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 110), 1, cv2.LINE_AA)

    # Centre crosshair (faint)
    cv2.line(canvas, (cx, 0), (cx, size), (30, 30, 45), 1)
    cv2.line(canvas, (0, cy), (size, cy), (30, 30, 45), 1)

    ZONE_COLORS = {
        "CRITICAL": (0,   0,   255),
        "CLOSE":    (0,   140, 255),
        "MEDIUM":   (0,   220, 220),
        "FAR":      (80,  200, 80),
    }

    def w2p(x_m, z_m):
        px = int(cx + x_m * scale / 2)
        py = int(cy - z_m * scale / 2)
        return int(np.clip(px, 0, size-1)), int(np.clip(py, 0, size-1))

    # Draw actors
    for pred in pred_state.predictions:
        if len(pred.current_pos) < 3:
            continue
        x_m = float(pred.current_pos[0])
        z_m = float(pred.current_pos[2])

        # Threat zone colour
        dist = float(np.linalg.norm([x_m, z_m]))
        zone = "CRITICAL" if dist < 5 else "CLOSE" if dist < 15 else "MEDIUM" if dist < 30 else "FAR"
        color = ZONE_COLORS[zone]

        # Short trajectory — clean single line
        if len(pred.short_trajectory) > 2:
            traj_pts = []
            for pos in pred.short_trajectory[::2]:
                traj_pts.append(w2p(float(pos[0]), float(pos[2])))
            for i in range(len(traj_pts) - 1):
                # Fade out over distance
                alpha = max(0.3, 1.0 - i / len(traj_pts))
                c = tuple(int(v * alpha) for v in color)
                cv2.line(canvas, traj_pts[i], traj_pts[i+1], c, 1, cv2.LINE_AA)

        # Actor dot
        ax, ay = w2p(x_m, z_m)
        dot_r = 7 if zone in ("CRITICAL", "CLOSE") else 5
        cv2.circle(canvas, (ax, ay), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ax, ay), dot_r + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Track ID + class initial — compact
        class_initial = pred.class_name[0].upper()
        lbl = f"{class_initial}{pred.track_id}"
        cv2.putText(canvas, lbl, (ax + dot_r + 2, ay + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    # Ego vehicle — white arrow pointing up
    ego_pts = np.array([
        [cx,      cy - 14],
        [cx - 7,  cy + 8],
        [cx,      cy + 3],
        [cx + 7,  cy + 8],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [ego_pts], (255, 255, 255))

    return canvas


def draw_prediction_panel(pred_state, world_state, scene=None, size=(300, 485)) -> np.ndarray:
    """
    Clean right panel.
    Big ACTION. Risk bar. Simple actor list — nothing else.
    """
    W, H = size
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)

    ACTION_COLORS = {
        "continue": (50,  200, 80),
        "slow":     (220, 200, 50),
        "brake":    (220, 120, 40),
        "stop":     (220, 50,  50),
    }
    ICON = {
        "car": "CAR", "truck": "TRK", "bus": "BUS",
        "person": "PED", "motorcycle": "MCY",
        "bicycle": "BCY", "unknown": "UNK",
    }

    # Use smart decision if available, else fall back to pred_state
    if scene is not None:
        action = scene.decision
        a_color = tuple(scene.decision_color)
        level_color = tuple(scene.decision_color)
    else:
        action = pred_state.recommended_action.upper()
        a_color = ACTION_COLORS.get(pred_state.recommended_action, (200, 200, 200))
        level_color = ARBITRATION_COLORS[world_state.risk_level]

    # ── ACTION block ──────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (W, 78), (28, 28, 40), -1)
    cv2.putText(panel, action, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, a_color, 3, cv2.LINE_AA)
    speed_pct = int(scene.speed_factor * 100) if scene else int(pred_state.recommended_speed_factor * 100)
    cv2.putText(panel, f"{speed_pct}% speed", (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, a_color, 1, cv2.LINE_AA)
    # Primary reason
    if scene:
        reason_short = scene.primary_reason[:35]
        cv2.putText(panel, reason_short, (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 160), 1, cv2.LINE_AA)

    # ── Risk bar ──────────────────────────────────────────────────────────────
    y = 78
    cv2.putText(panel, "RISK", (12, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1, cv2.LINE_AA)
    bar_x, bar_w = 50, W - 60
    cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 14), (40, 40, 55), -1)
    filled = int(pred_state.scene_risk_predicted * bar_w)
    cv2.rectangle(panel, (bar_x, y), (bar_x + filled, y + 14), level_color, -1)
    cv2.putText(panel, world_state.level_name, (bar_x + bar_w + 4, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, level_color, 1, cv2.LINE_AA)

    # ── Secondary reasons ────────────────────────────────────────────────────
    if scene and scene.secondary_reasons:
        y += 6
        for sec in scene.secondary_reasons[:2]:
            cv2.putText(panel, f"• {sec[:36]}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 140, 120), 1, cv2.LINE_AA)
            y += 13

    # ── Scene density summary ─────────────────────────────────────────────────
    if scene:
        y += 4
        density_str = f"CRIT:{scene.critical_zone_count}  CLOSE:{scene.close_zone_count}  MED:{scene.medium_zone_count}"
        cv2.putText(panel, density_str, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 140), 1, cv2.LINE_AA)
        trend_color = (50,200,80) if scene.trend == "improving" else (200,50,50) if scene.trend == "worsening" else (140,140,140)
        cv2.putText(panel, f"trend: {scene.trend}", (W-90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, trend_color, 1, cv2.LINE_AA)

    # ── Divider ───────────────────────────────────────────────────────────────
    y += 14
    cv2.line(panel, (10, y), (W - 10, y), (45, 45, 65), 1)
    y += 14

    cv2.putText(panel, "ACTORS", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 140), 1, cv2.LINE_AA)
    y += 16

    ZONE_COLORS = {
        "CRITICAL": (0,   0,   255),
        "CLOSE":    (0,   140, 255),
        "MEDIUM":   (0,   220, 220),
        "FAR":      (80,  200, 80),
    }

    # ── Actor rows ────────────────────────────────────────────────────────────
    for pred in pred_state.predictions[:7]:
        dist = float(np.linalg.norm(pred.current_pos[:3])) if len(pred.current_pos) >= 3 else 0
        zone = "CRITICAL" if dist < 5 else "CLOSE" if dist < 15 else "MEDIUM" if dist < 30 else "FAR"
        z_color = ZONE_COLORS[zone]
        icon = ICON.get(pred.class_name, "OBJ")
        maneuver = pred.most_likely_maneuver.replace("_", " ")
        prob = float(pred.maneuver_probs.max()) * 100

        # Coloured zone pill
        cv2.rectangle(panel, (10, y - 11), (46, y + 3), z_color, -1)
        cv2.putText(panel, icon, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

        # Distance
        cv2.putText(panel, f"{dist:.1f}m", (52, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, z_color, 1, cv2.LINE_AA)

        # Maneuver — truncated
        short_m = maneuver[:14]
        cv2.putText(panel, f"→ {short_m}", (105, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 180, 160), 1, cv2.LINE_AA)

        # TTC warning
        if pred.ttc and pred.ttc < 4.0:
            cv2.putText(panel, f"!{pred.ttc:.1f}s", (W - 38, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        y += 20
        if y > H - 20:
            break

    # ── Frame info ────────────────────────────────────────────────────────────
    cv2.line(panel, (10, H - 28), (W - 10, H - 28), (45, 45, 65), 1)
    cv2.putText(panel, f"PRISM  |  {world_state.actor_count} actors  |  frame {world_state.frame_idx}",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 110), 1, cv2.LINE_AA)

    return panel



def draw_pedestrian_panel(pred_state, world_state, metric_dets: list, scene=None, size=(300, 485)) -> np.ndarray:
    """
    Dedicated pedestrian awareness panel.
    Focuses entirely on people — are they a threat? Is it safe to proceed?

    Shows:
    - GO / NO-GO status
    - Mini spatial map (top-down, pedestrians only)
    - Per-pedestrian: distance, position, crossing probability
    - Overall recommendation
    """
    W, H = size
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    panel[:] = (18, 22, 28)   # slightly different tint from prediction panel

    # ── Extract pedestrian predictions ────────────────────────────────────────
    ped_preds = [p for p in pred_state.predictions if p.class_name == "person"]
    ped_dets  = [md for md in metric_dets if md.class_name == "person"]

    # ── Compute GO / NO-GO ────────────────────────────────────────────────────
    # NO-GO conditions:
    #   - Any pedestrian within 8m in front
    #   - Any pedestrian with crossing probability > 50%
    #   - Any pedestrian on collision course

    danger_peds = []
    for p in ped_preds:
        dist = float(np.linalg.norm(p.current_pos[:3])) if len(p.current_pos) >= 3 else 99
        z    = float(p.current_pos[2]) if len(p.current_pos) > 2 else 99
        x    = float(p.current_pos[0]) if len(p.current_pos) > 0 else 0

        # Crossing probability — high lateral velocity + close = crossing
        vx = float(p.current_vel[0]) if len(p.current_vel) > 0 else 0
        vz = float(p.current_vel[2]) if len(p.current_vel) > 2 else 0
        speed = float(np.linalg.norm([vx, vz]))
        lateral_ratio = abs(vx) / (speed + 0.01)
        crossing_prob = float(np.clip(lateral_ratio * 1.4, 0, 1.0))

        # Adjust by maneuver belief
        stop_prob  = float(p.maneuver_probs[5])   # stopping
        cross_prob = float(p.maneuver_probs[6] + p.maneuver_probs[7])  # lane changes as proxy
        crossing_prob = float(np.clip(crossing_prob + cross_prob * 0.5, 0, 1.0))

        is_danger = (
            (dist < 8.0 and z > 0) or
            (crossing_prob > 0.50 and dist < 20.0) or
            p.is_on_collision_course
        )
        danger_peds.append({
            "pred":         p,
            "dist":         dist,
            "x":            x,
            "z":            z,
            "crossing_prob":crossing_prob,
            "is_danger":    is_danger,
            "speed":        speed,
        })

    no_go = any(d["is_danger"] for d in danger_peds)
    status_color = (0, 60, 220) if no_go else (40, 180, 60)
    status_text  = "NO-GO" if no_go else "GO"

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (W, 28), (24, 28, 38), -1)
    cv2.putText(panel, "PEDESTRIAN AWARENESS", (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 160, 200), 1, cv2.LINE_AA)

    # ── GO / NO-GO block ──────────────────────────────────────────────────────
    y = 36
    cv2.rectangle(panel, (8, y), (W - 8, y + 38), status_color, -1)
    cv2.rectangle(panel, (8, y), (W - 8, y + 38), (255, 255, 255), 1)
    # Status dot
    cv2.circle(panel, (28, y + 19), 8, (255, 255, 255), -1)
    cv2.putText(panel, status_text, (44, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    ped_count = len(danger_peds)
    cv2.putText(panel, f"{ped_count} pedestrian{'s' if ped_count != 1 else ''} detected",
                (44, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)
    y += 52

    # ── Mini spatial map ──────────────────────────────────────────────────────
    MAP_SIZE = 130
    map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    map_img[:] = (22, 26, 36)
    mc = MAP_SIZE // 2

    # Danger zone — red semicircle in front
    cv2.ellipse(map_img, (mc, mc), (40, 40), 0, 200, 340, (60, 0, 0), -1)
    # Safe corridor
    cv2.ellipse(map_img, (mc, mc), (25, 25), 0, 200, 340, (0, 40, 0), -1)
    # Range rings
    for r in [25, 45]:
        cv2.circle(map_img, (mc, mc), r, (40, 40, 60), 1)

    # Ego arrow
    ego_pts = np.array([[mc, mc-10],[mc-5, mc+6],[mc, mc+2],[mc+5, mc+6]], np.int32)
    cv2.fillPoly(map_img, [ego_pts], (255, 255, 255))

    def map_w2p(x_m, z_m):
        # 50m range → MAP_SIZE pixels
        px = int(mc + x_m * MAP_SIZE / 60.0)
        py = int(mc - z_m * MAP_SIZE / 60.0)
        return int(np.clip(px, 2, MAP_SIZE-3)), int(np.clip(py, 2, MAP_SIZE-3))

    for d in danger_peds:
        px, py = map_w2p(d["x"], d["z"])
        dot_color = (0, 60, 220) if d["is_danger"] else (40, 180, 100)
        cv2.circle(map_img, (px, py), 5, dot_color, -1, cv2.LINE_AA)
        cv2.circle(map_img, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Place map centered in panel
    map_x = (W - MAP_SIZE) // 2
    panel[y:y+MAP_SIZE, map_x:map_x+MAP_SIZE] = map_img
    # Border
    cv2.rectangle(panel, (map_x-1, y-1), (map_x+MAP_SIZE+1, y+MAP_SIZE+1), (60, 70, 90), 1)
    y += MAP_SIZE + 10

    # ── Divider ───────────────────────────────────────────────────────────────
    cv2.line(panel, (10, y), (W-10, y), (45, 50, 70), 1)
    y += 12

    # ── Per pedestrian rows ───────────────────────────────────────────────────
    if not danger_peds:
        cv2.putText(panel, "No pedestrians in scene", (12, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 90, 110), 1, cv2.LINE_AA)
    else:
        for d in sorted(danger_peds, key=lambda x: x["dist"])[:5]:
            p = d["pred"]
            dist = d["dist"]
            x_m  = d["x"]
            cross = d["crossing_prob"]
            is_danger = d["is_danger"]

            # Position label
            if abs(x_m) < 2.0:  pos_label = "FRONT"
            elif x_m < 0:        pos_label = "LEFT"
            else:                pos_label = "RIGHT"

            row_color = (0, 80, 220) if is_danger else (40, 160, 80)

            # Status pill
            pill_text = "DANGER" if is_danger else "CLEAR"
            cv2.rectangle(panel, (8, y-11), (62, y+3), row_color, -1)
            cv2.putText(panel, pill_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

            # Distance + position
            cv2.putText(panel, f"{dist:.1f}m  {pos_label}", (68, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 220), 1, cv2.LINE_AA)

            y += 16

            # Crossing probability bar
            cv2.putText(panel, "cross", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (100, 110, 130), 1, cv2.LINE_AA)
            bar_x, bar_w_full = 46, W - 56
            cv2.rectangle(panel, (bar_x, y-8), (bar_x+bar_w_full, y), (35, 38, 50), -1)
            filled = int(cross * bar_w_full)
            bar_fill_color = (0, 60, 220) if cross > 0.5 else (40, 140, 220) if cross > 0.25 else (40, 100, 80)
            cv2.rectangle(panel, (bar_x, y-8), (bar_x+filled, y), bar_fill_color, -1)
            cv2.putText(panel, f"{int(cross*100)}%", (bar_x+bar_w_full+3, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 160, 180), 1, cv2.LINE_AA)

            y += 14
            if y > H - 40:
                break

    # ── Recommendation ────────────────────────────────────────────────────────
    cv2.line(panel, (10, H-32), (W-10, H-32), (45, 50, 70), 1)
    if no_go:
        rec = "YIELD — pedestrian priority"
        rec_color = (0, 80, 220)
    elif ped_count > 0:
        rec = "CAUTION — monitor closely"
        rec_color = (180, 160, 40)
    else:
        rec = "CLEAR — no pedestrians"
        rec_color = (40, 160, 80)
    cv2.putText(panel, rec, (10, H-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, rec_color, 1, cv2.LINE_AA)

    return panel


def draw_pedestrian_panel(pred_state, world_state, metric_dets: list, scene=None, size=(300, 485)) -> np.ndarray:
    """
    Dedicated pedestrian spatial awareness panel.

    Layout:
    ┌─────────────────────────┐
    │  GO / NO-GO  (big)      │
    ├─────────────────────────┤
    │  Spatial radar map      │
    │  (top-down ped view)    │
    ├─────────────────────────┤
    │  Per-ped status rows    │
    └─────────────────────────┘

    GO:    no pedestrians within 15m OR all stationary and not on path
    CAUTION: pedestrians 8-15m, intent unclear
    NO-GO: pedestrian within 8m OR on predicted collision path
    """
    W, H = size
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)

    def txt(msg, x, y, color=(200,200,200), scale=0.40, thick=1):
        cv2.putText(panel, msg, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # ── Gather pedestrian data ────────────────────────────────────────────────
    ped_predictions = [p for p in pred_state.predictions
                       if p.class_name in ("person", "bicycle", "motorcycle")]

    ped_metric = [md for md in metric_dets
                  if md.class_name in ("person", "bicycle", "motorcycle")]

    # Build unified ped list
    peds = []
    for pred in ped_predictions:
        dist = float(np.linalg.norm(pred.current_pos[:3])) if len(pred.current_pos) >= 3 else 99
        lateral = float(pred.current_pos[0]) if len(pred.current_pos) >= 1 else 0
        moving = pred.current_vel is not None and float(np.linalg.norm(pred.current_vel[:2])) > 0.05
        maneuver = pred.most_likely_maneuver

        # Is this ped on a collision path?
        # Check if any predicted position crosses ego corridor (|lateral| < 2m)
        on_path = False
        if len(pred.short_trajectory) > 0:
            for pos in pred.short_trajectory:
                if abs(float(pos[0])) < 2.0 and float(pos[2]) > 0:
                    on_path = True
                    break

        peds.append({
            "track_id":  pred.track_id,
            "class":     pred.class_name,
            "dist":      dist,
            "lateral":   lateral,
            "moving":    moving,
            "on_path":   on_path,
            "maneuver":  maneuver,
            "ttc":       pred.ttc,
            "trajectory": pred.short_trajectory,
            "pos":       pred.current_pos,
        })

    # Sort by distance
    peds.sort(key=lambda p: p["dist"])

    # ── GO / NO-GO decision ───────────────────────────────────────────────────
    no_go = any(
        p["dist"] < 5.0 or
        (p["dist"] < 8.0 and p["on_path"]) or
        (p["ttc"] is not None and p["ttc"] < 2.0)
        for p in peds
    )
    caution = not no_go and any(
        (p["dist"] < 15.0 and p["moving"]) or
        (p["dist"] < 12.0 and p["on_path"])
        for p in peds
    )
    go = not no_go and not caution

    # Use smart decision engine if available for richer context
    if scene is not None and scene.ped_count > 0:
        lvl = scene.decision_level
        if lvl >= 6 or any(p["on_path"] and p["dist"] < 6 for p in peds):
            no_go = True
            caution = False
        elif lvl >= 4 or any(p["on_path"] for p in peds) or any(p["dist"] < 8 and p["moving"] for p in peds):
            no_go = False
            caution = True

    if no_go:
        decision = "NO-GO"
        d_color  = (50, 50, 220)
        d_bg     = (30, 15, 40)
    elif caution:
        decision = "CAUTION"
        d_color  = (50, 180, 220)
        d_bg     = (20, 28, 35)
    else:
        decision = "GO"
        d_color  = (50, 200, 80)
        d_bg     = (15, 28, 18)

    # ── Decision block ────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (W, 64), d_bg, -1)
    cv2.putText(panel, decision, (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, d_color, 3, cv2.LINE_AA)
    ped_count = len(peds)
    on_path_count = sum(1 for p in peds if p["on_path"])
    moving_count  = sum(1 for p in peds if p["moving"])
    txt(f"{ped_count} detected  |  {moving_count} moving  |  {on_path_count} on path",
        12, 62, (80, 80, 110), 0.30)

    # ── Spatial radar ─────────────────────────────────────────────────────────
    radar_size = 180
    radar_y0 = 72
    radar_cx  = W // 2
    radar_cy  = radar_y0 + radar_size // 2
    radar_range = 25.0   # metres shown

    # Radar background
    cv2.circle(panel, (radar_cx, radar_cy), radar_size//2, (28, 28, 45), -1)
    cv2.circle(panel, (radar_cx, radar_cy), radar_size//2, (45, 45, 70), 1)

    # Range rings
    for r_m, label in [(5, "5m"), (15, "15m"), (25, "25m")]:
        r_px = int(r_m / radar_range * radar_size / 2)
        cv2.circle(panel, (radar_cx, radar_cy), r_px, (40, 40, 65), 1, cv2.LINE_AA)
        txt(label, radar_cx + r_px + 2, radar_cy - 2, (55, 55, 85), 0.25)

    # Safe corridor — green band down centre
    corridor_w = int(2.0 / radar_range * radar_size / 2)
    cv2.rectangle(panel,
                  (radar_cx - corridor_w, radar_y0),
                  (radar_cx + corridor_w, radar_y0 + radar_size),
                  (20, 50, 20), -1)

    # Centre lines
    cv2.line(panel, (radar_cx, radar_y0), (radar_cx, radar_y0 + radar_size), (35, 35, 55), 1)
    cv2.line(panel, (radar_cx - radar_size//2, radar_cy),
             (radar_cx + radar_size//2, radar_cy), (35, 35, 55), 1)

    # Ego marker
    ego_pts = np.array([
        [radar_cx,     radar_cy + 10],
        [radar_cx - 5, radar_cy + 18],
        [radar_cx,     radar_cy + 15],
        [radar_cx + 5, radar_cy + 18],
    ], dtype=np.int32)
    cv2.fillPoly(panel, [ego_pts], (255, 255, 255))

    # Plot ALL pedestrians on radar — every blip shows regardless of threat
    for ped in peds:
        dist = ped["dist"]
        lat  = ped["lateral"]
        # Still show blip even beyond radar range — clamp to edge
        clamped = dist > radar_range

        # Convert to radar pixels — clamp to edge ring if beyond range
        dist_clamped = min(dist, radar_range * 0.97)
        rx = int(radar_cx + (lat / radar_range) * radar_size / 2)
        ry = int(radar_cy - (dist_clamped / radar_range) * radar_size / 2)
        rx = int(np.clip(rx, radar_cx - radar_size//2 + 4, radar_cx + radar_size//2 - 4))
        ry = int(np.clip(ry, radar_y0 + 4, radar_y0 + radar_size - 4))

        # Colour by status
        if ped["on_path"] or dist < 5:
            dot_color = (50,  50,  220)   # red — danger
        elif dist < 12 and ped["moving"]:
            dot_color = (50,  180, 220)   # amber — watch
        else:
            dot_color = (80,  200, 80)    # green — safe / peripheral

        # All peds get a blip — size varies by proximity
        dot_r = 7 if dist < 5 else 5 if dist < 15 else 4
        cv2.circle(panel, (rx, ry), dot_r, dot_color, -1, cv2.LINE_AA)
        cv2.circle(panel, (rx, ry), dot_r + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Extra ring for on-path peds
        if ped["on_path"]:
            cv2.circle(panel, (rx, ry), dot_r + 5, dot_color, 1, cv2.LINE_AA)

        # Beyond-range indicator — small arrow on edge
        if clamped:
            cv2.putText(panel, "►" if lat > 0 else "◄",
                        (rx - 4, ry + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, dot_color, 1, cv2.LINE_AA)

        # Trajectory only for close/moving peds — keeps radar clean
        if dist < radar_range and len(ped["trajectory"]) > 2 and ped["moving"]:
            prev = (rx, ry)
            for pos in ped["trajectory"][::3]:
                nx = int(radar_cx + (float(pos[0]) / radar_range) * radar_size / 2)
                ny = int(radar_cy - (float(pos[2]) / radar_range) * radar_size / 2)
                nx = int(np.clip(nx, radar_cx - radar_size//2 + 4, radar_cx + radar_size//2 - 4))
                ny = int(np.clip(ny, radar_y0 + 4, radar_y0 + radar_size - 4))
                cv2.line(panel, prev, (nx, ny), dot_color, 1, cv2.LINE_AA)
                prev = (nx, ny)

        # ID label — always shown
        txt(f"P{ped['track_id']}", rx + dot_r + 2, ry + 4, (200, 200, 200), 0.28)
        # Distance label for close peds
        if dist < 20:
            txt(f"{dist:.0f}m", rx + dot_r + 2, ry + 14, dot_color, 0.25)

    # Radar labels
    txt("FRONT", radar_cx - 16, radar_y0 + 11, (60, 60, 90), 0.28)
    txt("L", radar_cx - radar_size//2 + 2, radar_cy + 4, (60, 60, 90), 0.28)
    txt("R", radar_cx + radar_size//2 - 10, radar_cy + 4, (60, 60, 90), 0.28)

    # ── Per-ped status rows ───────────────────────────────────────────────────
    y = radar_y0 + radar_size + 16
    cv2.line(panel, (10, y - 8), (W - 10, y - 8), (45, 45, 65), 1)
    txt("PEDESTRIAN STATUS", 12, y, (100, 100, 140), 0.35)
    y += 16

    if not peds:
        txt("No pedestrians detected", 12, y, (60, 60, 90), 0.38)
        y += 18
        txt("Path is clear", 12, y, (50, 200, 80), 0.38)
    else:
        for ped in peds[:5]:
            dist  = ped["dist"]
            lat   = ped["lateral"]
            cls   = ped["class"]

            # Status indicator
            if ped["on_path"] or dist < 5:
                status_c = (50, 50, 220)
                status   = "DANGER"
            elif dist < 12 and ped["moving"]:
                status_c = (50, 180, 220)
                status   = "WATCH"
            else:
                status_c = (50, 180, 80)
                status   = "CLEAR"

            # Pill
            cv2.rectangle(panel, (10, y - 11), (62, y + 3), status_c, -1)
            txt(status, 12, y, (0, 0, 0), 0.28, 1)

            # Info
            lat_str = f"L{abs(lat):.1f}m" if lat < 0 else f"R{abs(lat):.1f}m"
            move_str = "MOV" if ped["moving"] else "STA"
            info = f"{cls[:3].upper()}  {dist:.1f}m  {lat_str}  {move_str}"
            txt(info, 68, y, (180, 180, 200), 0.33)

            # On path warning
            if ped["on_path"]:
                txt("ON PATH", W - 58, y, (50, 50, 220), 0.30, 1)

            y += 18
            if y > H - 24:
                break

    # ── Footer ────────────────────────────────────────────────────────────────
    cv2.line(panel, (10, H - 22), (W - 10, H - 22), (45, 45, 65), 1)
    txt("PEDESTRIAN AWARENESS MODULE", 10, H - 9, (55, 55, 85), 0.28)

    return panel


def draw_vlm_overlay(image: np.ndarray, vlm_output, age_s: float) -> np.ndarray:
    """
    Draws VLM semantic output as a clean overlay on the camera image.
    Shows: scene summary, risk flags, caution level.
    Fades out as VLM output gets older.
    """
    if vlm_output is None:
        return image
    out = image.copy()
    h, w = out.shape[:2]

    # Fade alpha based on age — VLM output is fresh for 2s
    alpha = max(0.3, 1.0 - age_s / 3.0)

    CAUTION_COLORS = {
        "normal":   (50, 200, 80),
        "elevated": (50, 200, 220),
        "high":     (50, 140, 220),
        "critical": (50, 50, 220),
    }
    c_color = CAUTION_COLORS.get(vlm_output.recommended_caution, (200, 200, 200))
    c_color = tuple(int(v * alpha) for v in c_color)

    # Bottom strip — scene summary
    strip_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (10, 10, 20), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)

    # Caution badge
    badge_label = f"VLM: {vlm_output.recommended_caution.upper()}"
    cv2.putText(out, badge_label, (10, h - strip_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_color, 1, cv2.LINE_AA)

    # Scene summary — truncated
    summary = vlm_output.scene_summary[:80] if vlm_output.scene_summary else vlm_output.scene_context[:80]
    cv2.putText(out, summary, (10, h - strip_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)

    # Risk flags — top right corner
    for i, flag in enumerate(vlm_output.risk_flags[:3]):
        flag_y = 20 + i * 16
        flag_text = f"⚑ {flag}"
        (fw, _), _ = cv2.getTextSize(flag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(out, flag_text, (w - fw - 10, flag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_color, 1, cv2.LINE_AA)

    # Trigger info — very small, bottom right
    trigger_str = f"trigger: {vlm_output.trigger_reason}  {vlm_output.inference_time_ms:.0f}ms"
    cv2.putText(out, trigger_str, (w - 280, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60, 60, 90), 1, cv2.LINE_AA)

    return out


def get_lidar_distance(nusc, sample_token: str, track_bbox, camera_name: str) -> Optional[float]:
    """
    Get LiDAR ground truth distance for validation.
    Projects LiDAR points into camera frame and finds points within bbox.
    """
    try:
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        import pyquaternion

        sample = nusc.get("sample", sample_token)

        # Get LiDAR data
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        lidar_path = Path(nusc.dataroot) / lidar_data["filename"]
        pc = LidarPointCloud.from_file(str(lidar_path))

        # Get camera data and calibration
        cam_token = sample["data"][camera_name]
        cam_data = nusc.get("sample_data", cam_token)
        cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # Transform LiDAR points to camera frame
        # LiDAR → ego
        lidar_to_ego = pyquaternion.Quaternion(lidar_calib["rotation"]).rotation_matrix
        pc.rotate(lidar_to_ego)
        pc.translate(np.array(lidar_calib["translation"]))

        # ego → camera
        ego_to_cam = pyquaternion.Quaternion(cam_calib["rotation"]).rotation_matrix.T
        pc.translate(-np.array(cam_calib["translation"]))
        pc.rotate(ego_to_cam)

        # Keep points in front of camera
        depths = pc.points[2, :]
        mask = depths > 0.5
        points_2d = view_points(pc.points[:3, mask], np.array(cam_calib["camera_intrinsic"]), True)

        # Find points inside the bounding box
        x1, y1, x2, y2 = track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2
        in_box = (
            (points_2d[0] >= x1) & (points_2d[0] <= x2) &
            (points_2d[1] >= y1) & (points_2d[1] <= y2)
        )

        if in_box.sum() < 3:
            return None

        # Median depth of points inside bbox
        box_depths = depths[mask][in_box]
        return float(np.median(box_depths))

    except Exception:
        return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["logging"]["viz_output_dir"]).expanduser() / "predictive"
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PRISM — Full Pipeline (Metric Depth + Prediction)")
    logger.info("=" * 60)
    video_writer = None  # initialised on first frame

    # Init all components
    loader = NuScenesLoader(cfg)
    core = SensoryCore(cfg)
    metric_engine = MetricDepthEngine(cfg)
    world = WorldModel(cfg)
    predictor = PredictiveEngine(cfg)
    validator = DepthValidator() if args.validate else None
    viz = WorldModelVisualizer()

    scene_info = loader.get_scene_info(args.scene)
    logger.info(f"Scene: {scene_info['name']} — {scene_info['description']}")

    decision_engine = SmartDecisionEngine()
    arbitrator      = ArbitrationCore(cfg)
    planner         = AdaptivePlanner(cfg)
    reasoner        = SemanticReasoner(cfg)
    logger.info(f"VLM available: {reasoner.vlm.available}")

    stats = {
        "frames": 0,
        "times": [],
        "actions": {},
        "threat_zones": {"CRITICAL": 0, "CLOSE": 0, "MEDIUM": 0, "FAR": 0}
    }

    sample_tokens = []
    if args.validate and loader.nusc:
        scene = loader.nusc.scene[args.scene]
        tok = scene["first_sample_token"]
        while tok:
            sample_tokens.append(tok)
            tok = loader.nusc.get("sample", tok)["next"]

    for frame_data in loader.iter_primary_camera(scene_idx=args.scene):
        if stats["frames"] >= args.max_frames:
            break

        image = frame_data["image"]
        calib = frame_data.get("calibration", {})
        t0 = time.time()

        # Update camera intrinsics from calibration
        metric_engine.update_intrinsics(calib)

        # Layer 1 — Sensory Core
        sensory_frame = core.process(
            image,
            camera_name=frame_data["camera_name"],
            timestamp=frame_data["timestamp"]
        )

        # Layer 1b — Metric Depth
        run_model = sensory_frame.depth_map is not None
        metric_dets, metric_depth_map = metric_engine.process_frame(
            image,
            sensory_frame.detections,
            run_model=run_model
        )

        # Attach track IDs from world model to metric detections
        # (we'll use detection index as proxy before full tracker integration)
        for i, md in enumerate(metric_dets):
            md.bbox.track_id = i  # placeholder — replaced by tracker in world model

        # Layer 2 — World Model
        world_state = world.update(sensory_frame, calibration=calib)

        # Sync track IDs
        for actor in world_state.actors:
            # Match by class and approximate bbox overlap
            for md in metric_dets:
                if md.class_name == actor.class_name:
                    md.bbox.track_id = actor.track_id
                    break

        # Layer 3 — Predictive Engine
        pred_state = predictor.update(world_state, metric_dets)

        # Layer 3c — Smart Decision Engine
        scene = decision_engine.assess(world_state, pred_state, metric_dets)

        # Layer 4 — VLM Semantic Reasoner (non-blocking)
        vlm_output = reasoner.update(image, world_state, pred_state)
        if vlm_output and vlm_output.actor_intents:
            predictor.update_vlm_intents(vlm_output.actor_intents)

        # Layer 5 — Arbitration Core (fuses all signals, produces audit trail)
        if vlm_output is not None:
            arbitrator.update_vlm(vlm_output.to_arb_dict())
        arb_decision = arbitrator.arbitrate(
            world_state, pred_state, scene, timestamp=world_state.timestamp
        )
        logger.debug(arb_decision.short_reason)

        # Layer 6 — Adaptive Planner (jerk-limited velocity profile + control)
        plan = planner.plan(arb_decision, metric_dets=metric_dets,
                            timestamp=world_state.timestamp)

        elapsed = (time.time() - t0) * 1000
        stats["frames"] += 1
        stats["times"].append(elapsed)

        action = arb_decision.action
        stats["actions"][action] = stats["actions"].get(action, 0) + 1
        if plan.emergency:
            stats.setdefault("emergency_frames", 0)
            stats["emergency_frames"] += 1
        if plan.spatial_override:
            stats.setdefault("spatial_overrides", 0)
            stats["spatial_overrides"] += 1

        for md in metric_dets:
            stats["threat_zones"][md.threat_zone] += 1

        # Depth validation against LiDAR GT
        if validator and stats["frames"] <= len(sample_tokens):
            sample_tok = sample_tokens[stats["frames"] - 1]
            for md in metric_dets:
                gt_dist = get_lidar_distance(
                    loader.nusc, sample_tok, md.bbox,
                    frame_data["camera_name"]
                )
                if gt_dist:
                    validator.add_sample(md.distance_m, gt_dist)

        # Log — Arbitration Core + Planner output
        _emrg_tag = " ⚡EMRG" if plan.emergency else (" 🔀SPA" if plan.spatial_override else "")
        logger.info(
            f"Frame {stats['frames']:3d} | {elapsed:.0f}ms | "
            f"[ARB] {arb_decision.action:<9} conf={arb_decision.confidence*100:.0f}% | "
            f"[PLN] v={plan.current_speed_mps:.1f}→{plan.target_speed_mps:.1f}m/s "
            f"thr={plan.control.throttle:.2f} brk={plan.control.brake:.2f}{_emrg_tag} | "
            f"closest:{scene.closest_threat_m:.1f}m peds:{scene.ped_count} veh:{scene.vehicle_count}"
        )

        # Visualise
        cam_annotated = draw_metric_detections(image, metric_dets)
        vlm_state = reasoner.get_current_semantic_state()
        vlm_age = time.time() - vlm_state.timestamp if vlm_state else 99
        cam_annotated = draw_vlm_overlay(cam_annotated, vlm_state, vlm_age)

        target_h = 485
        bev_img      = draw_bev(pred_state, world_state, size=target_h)
        pred_panel   = draw_prediction_panel(pred_state, world_state, scene, size=(280, target_h))
        ped_panel    = draw_pedestrian_panel(pred_state, world_state, metric_dets, scene, size=(280, target_h))

        # Resize camera to match height
        cam_w = int(cam_annotated.shape[1] * target_h / cam_annotated.shape[0])
        cam_resized = cv2.resize(cam_annotated, (cam_w, target_h))

        # Composite: camera | BEV | vehicle panel | ped panel
        composite = np.concatenate([cam_resized, bev_img, pred_panel, ped_panel], axis=1)

        # Clean title bar
        bar = np.zeros((28, composite.shape[1], 3), dtype=np.uint8)
        bar[:] = (18, 18, 28)
        level_color = ARBITRATION_COLORS[world_state.risk_level]
        cv2.putText(bar,
            f"PRISM  —  {scene.decision}  {int(scene.speed_factor*100)}%  |  "
            f"{scene.actor_count} actors  {scene.ped_count} peds  |  "
            f"{scene.primary_reason[:50]}  |  {elapsed:.0f}ms",
            (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tuple(scene.decision_color), 1, cv2.LINE_AA
        )
        composite = np.vstack([bar, composite])

        if args.save:
            # Init video writer on first frame once we know dimensions
            if video_writer is None:
                h, w = composite.shape[:2]
                video_path = str(output_dir / "prism_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w, h))
                logger.info(f"Video writer initialised: {video_path}")
            video_writer.write(composite)

        if args.show:
            cv2.imshow("PRISM", composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Summary
    logger.info("=" * 60)
    logger.info("FULL PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Frames:       {stats['frames']}")
    logger.info(f"Avg FPS:      {1000/np.mean(stats['times']):.1f}")
    logger.info(f"Avg latency:  {np.mean(stats['times']):.0f}ms")
    logger.info(f"\nArbitration decisions:")
    _ARB_ORDER = ["CLEAR", "MONITOR", "EASE", "SLOW", "CAUTION", "YIELD", "STOP", "EMERGENCY"]
    for action in _ARB_ORDER:
        count = stats["actions"].get(action, 0)
        if count == 0:
            continue
        pct = count / stats["frames"] * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        logger.info(f"  {action:<12} {bar}  {count:3d} ({pct:.0f}%)")
    logger.info(f"\nThreat zones:")
    for zone, count in stats["threat_zones"].items():
        logger.info(f"  {zone:<10} {count:3d}")

    pln_stats = planner.stats
    logger.info(f"\nPlanner:")
    logger.info(f"  Final speed   : {pln_stats['v_current']:.2f} m/s  ({pln_stats['v_current']*3.6:.1f} km/h)")
    logger.info(f"  Emergency brk : {stats.get('emergency_frames', 0)} frames")
    logger.info(f"  Spatial ovrd  : {stats.get('spatial_overrides', 0)} frames")

    if validator:
        validator.print_report()

    if args.show:
        cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
        logger.info(f"Video saved: {output_dir}/prism_output.mp4")
    vlm_stats = reasoner.stats
    logger.info(f"\nVLM Statistics:")
    logger.info(f"  Total calls:    {vlm_stats['vlm_calls']}")
    logger.info(f"  Trigger rate:   {vlm_stats['trigger_rate']*100:.1f}% of frames")
    logger.info(f"  Avg inference:  {vlm_stats['avg_inference_ms']:.0f}ms")
    logger.info(f"  Efficiency:     {vlm_stats['efficiency']*100:.1f}% frames skipped")
    logger.info(f"\nOutput: {output_dir}")
    logger.info("=" * 60)




if __name__ == "__main__":
    main()
