"""
PRISM — World Model
====================
The single shared truth of the scene.
Every PRISM component reads from here. Nothing talks to raw sensors except SensoryCore.

The World Model maintains 4 layers:
    Static Layer   — road geometry, infrastructure (slow-changing)
    Dynamic Layer  — tracked actors with velocity and intent
    Occupancy Layer— probabilistic BEV grid (free vs occupied)
    Risk Layer     — danger heatmap derived from all other layers

This is the difference between:
    "I see objects in this image frame"        ← pipeline thinking
    "I maintain a model of what the world is"  ← world model thinking
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from prism.utils.common import get_logger, SensoryFrame, CLASS_COLORS, ARBITRATION_COLORS
from prism.world_model.tracker import Sort, TrackedActor
from prism.world_model.bev_grid import BEVOccupancyGrid, BEVConfig

logger = get_logger("WorldModel")


# ── Risk Layer ────────────────────────────────────────────────────────────────

class RiskLayer:
    """
    Computes a scalar risk score and heatmap for the current scene.

    Risk is a function of:
    - Actor proximity (closer = more risk)
    - Actor velocity (faster = more risk)
    - Actor class (pedestrian > car for lateral risk, truck > car for frontal)
    - Occupancy confidence (uncertain = elevated risk)
    - Prediction divergence (unexpected behavior = spike in risk)

    This is the "gut feeling" in data form.
    The Arbitration Core reads this every frame.
    """

    # Risk weights per class
    CLASS_RISK = {
        "person":        1.8,
        "bicycle":       1.4,
        "motorcycle":    1.3,
        "car":           1.0,
        "truck":         1.2,
        "bus":           1.1,
        "traffic light": 0.3,
        "stop sign":     0.2,
        "unknown":       1.0,
    }

    def __init__(self):
        self.current_score = 0.0
        self.history = []
        self.max_history = 30

    def compute(self, actors: list, occupancy: np.ndarray) -> dict:
        """
        Compute risk from current actor list and occupancy grid.
        Returns dict with score (0-1) and per-actor risk contributions.
        """
        if not actors:
            self.current_score = 0.0
            return {"score": 0.0, "level": 0, "actors": [], "trend": "stable"}

        actor_risks = []
        for actor in actors:
            if not actor.is_confirmed:
                continue

            risk = self._actor_risk(actor)
            actor_risks.append({
                "track_id": actor.track_id,
                "class": actor.class_name,
                "risk": risk,
                "moving": actor.is_moving,
                "depth": actor.depth
            })

        # Scene-level risk = weighted combination
        if actor_risks:
            risks = [a["risk"] for a in actor_risks]
            # Max risk dominates, but average matters too
            scene_risk = 0.6 * max(risks) + 0.4 * np.mean(risks)
        else:
            scene_risk = 0.0

        # Occupancy density also contributes
        occ_density = float(np.mean(occupancy > 0.5))
        scene_risk = np.clip(scene_risk + occ_density * 0.1, 0.0, 1.0)

        self.current_score = float(scene_risk)
        self.history.append(self.current_score)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return {
            "score": self.current_score,
            "level": self._score_to_level(self.current_score),
            "actors": actor_risks,
            "trend": self._trend(),
        }

    def _actor_risk(self, actor: TrackedActor) -> float:
        """Compute risk contribution of a single actor."""
        # Base risk from class
        base = self.CLASS_RISK.get(actor.class_name, 1.0)

        # Proximity risk — closer = more dangerous
        if actor.depth is not None:
            # depth 0=very close, 1=far
            # Risk spikes below 0.2 (roughly <10m)
            proximity = max(0.0, 1.0 - actor.depth * 2.0)
            proximity = np.clip(proximity, 0.0, 1.0)
        else:
            proximity = 0.3  # unknown depth = moderate risk

        # Motion risk — moving actors are more dangerous
        speed_norm = np.clip(actor.estimated_speed_px / 20.0, 0.0, 1.0)
        motion = speed_norm * 0.4 if actor.is_moving else 0.0

        # Combine
        risk = base * (0.5 * proximity + 0.3 * motion + 0.2)
        return float(np.clip(risk, 0.0, 1.0))

    def _score_to_level(self, score: float) -> int:
        """Map continuous risk score to discrete level 0-3."""
        if score < 0.25:   return 0  # GREEN
        if score < 0.50:   return 1  # YELLOW
        if score < 0.75:   return 2  # ORANGE
        return 3                      # RED

    def _trend(self) -> str:
        """Is risk increasing, decreasing or stable?"""
        if len(self.history) < 5:
            return "stable"
        recent = np.mean(self.history[-3:])
        older = np.mean(self.history[-6:-3])
        delta = recent - older
        if delta > 0.05:   return "increasing"
        if delta < -0.05:  return "decreasing"
        return "stable"


# ── World Model State ─────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    Snapshot of the World Model at a single timestep.
    This is what the Predictive Engine and Arbitration Core read.
    """
    timestamp: float
    frame_idx: int

    # Dynamic layer
    actors: list = field(default_factory=list)        # List[TrackedActor]
    actor_count: int = 0

    # Occupancy layer
    occupancy: Optional[np.ndarray] = None            # (H, W) float32

    # Risk layer
    risk_score: float = 0.0
    risk_level: int = 0                               # 0=GREEN 1=YELLOW 2=ORANGE 3=RED
    risk_trend: str = "stable"
    risk_actors: list = field(default_factory=list)

    # Confidence
    perception_confidence: float = 1.0               # drops when detections are sparse

    # Last sensory frame metadata
    camera_name: str = ""
    has_depth: bool = False
    has_flow: bool = False

    @property
    def level_name(self) -> str:
        return ["GREEN", "YELLOW", "ORANGE", "RED"][self.risk_level]

    @property
    def level_color(self) -> tuple:
        return ARBITRATION_COLORS[self.risk_level]


# ── World Model ───────────────────────────────────────────────────────────────

class WorldModel:
    """
    The heart of PRISM.

    Ingests SensoryFrames and maintains a living 4D model of the world.
    All downstream components (Predictive Engine, VLM, Arbitration) read from here.

    Usage:
        world = WorldModel(cfg)
        for sensory_frame in sensory_core:
            state = world.update(sensory_frame)
            planner.act(state)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        wm_cfg = cfg.get("world_model", {})
        bev_cfg_dict = wm_cfg.get("bev", {})
        tracker_cfg = wm_cfg.get("actor_tracker", {})

        # BEV config
        bev_config = BEVConfig(
            x_range=tuple(bev_cfg_dict.get("x_range", [-50, 50])),
            y_range=tuple(bev_cfg_dict.get("y_range", [-50, 50])),
            resolution=bev_cfg_dict.get("resolution", 0.5),
        )

        # Init layers
        self.tracker = Sort(
            max_age=tracker_cfg.get("max_age", 10),
            min_hits=tracker_cfg.get("min_hits", 2),
            iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
        )
        self.bev = BEVOccupancyGrid(bev_config)
        self.risk = RiskLayer()

        # State history — Predictive Engine reads this
        self.state_history: list[WorldState] = []
        self.max_history = 60  # ~5 seconds at 12fps

        # Last depth map — reused when depth doesn't run every frame
        self._last_depth_map: Optional[np.ndarray] = None
        self._last_calibration: Optional[dict] = None

        self._frame_idx = 0
        logger.info("World Model ready")

    def update(self, sensory_frame: SensoryFrame, calibration: Optional[dict] = None) -> WorldState:
        """
        Core update loop — called every frame.
        Ingests a SensoryFrame, updates all layers, returns current WorldState.
        """
        self._frame_idx += 1

        # Cache depth map — reuse on frames where depth didn't run
        if sensory_frame.depth_map is not None:
            self._last_depth_map = sensory_frame.depth_map
        if calibration:
            self._last_calibration = calibration

        # ── Dynamic Layer: update tracker ────────────────────────────────────
        actors = self.tracker.update(sensory_frame.detections)

        # Enrich actors with smoothed depth from last known depth map
        if self._last_depth_map is not None:
            for actor in actors:
                if actor.depth is None and actor.depth_history:
                    actor.depth = float(np.median(actor.depth_history[-3:]))

        # ── Occupancy Layer: update BEV grid ─────────────────────────────────
        self.bev.update_from_actors(actors, self._last_calibration)
        occupancy = self.bev.get_occupancy()

        # ── Risk Layer: compute danger ────────────────────────────────────────
        risk_result = self.risk.compute(actors, occupancy)

        # ── Perception confidence ─────────────────────────────────────────────
        # Drop confidence if we have very few detections unexpectedly
        confidence = self._compute_confidence(sensory_frame, actors)

        # ── Build WorldState ──────────────────────────────────────────────────
        state = WorldState(
            timestamp=sensory_frame.timestamp,
            frame_idx=self._frame_idx,
            actors=actors,
            actor_count=len(actors),
            occupancy=occupancy,
            risk_score=risk_result["score"],
            risk_level=risk_result["level"],
            risk_trend=risk_result["trend"],
            risk_actors=risk_result["actors"],
            perception_confidence=confidence,
            camera_name=sensory_frame.camera_name,
            has_depth=sensory_frame.depth_map is not None,
            has_flow=sensory_frame.optical_flow is not None,
        )

        # Add to history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        return state

    def _compute_confidence(self, frame: SensoryFrame, actors: list) -> float:
        """
        How confident are we in our current world model?
        Drops when: few detections, no depth, high occlusion.
        """
        confidence = 1.0

        # Penalize if no depth available
        if self._last_depth_map is None:
            confidence *= 0.7

        # Penalize if suddenly very few actors (possible occlusion or sensor issue)
        if len(self.state_history) > 5:
            recent_counts = [s.actor_count for s in self.state_history[-5:]]
            avg_count = np.mean(recent_counts)
            if avg_count > 2 and len(actors) == 0:
                confidence *= 0.5  # went from busy scene to nothing — suspicious

        return float(np.clip(confidence, 0.0, 1.0))

    def get_recent_states(self, n: int = 5) -> list:
        """Returns last N world states — used by Predictive Engine."""
        return self.state_history[-n:]

    def reset(self):
        self.tracker.reset()
        self.bev.reset()
        self.state_history = []
        self._last_depth_map = None
        self._frame_idx = 0


# ── World Model Visualizer ────────────────────────────────────────────────────

class WorldModelVisualizer:
    """
    Renders the World Model state as a composite display.
    This is the main competition demo visualization.

    Layout:
    ┌─────────────────┬──────────────┐
    │  Camera + Tracks│   BEV Grid   │
    ├─────────────────┴──────────────┤
    │         Risk HUD               │
    └────────────────────────────────┘
    """

    @staticmethod
    def draw_tracks(image: np.ndarray, state: WorldState) -> np.ndarray:
        """Draw tracked actors with IDs and velocity arrows."""
        out = image.copy()
        for actor in state.actors:
            if not actor.is_confirmed:
                continue
            b = actor.bbox
            color = CLASS_COLORS.get(actor.class_name, CLASS_COLORS["unknown"])

            # Box
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)

            # Track ID + class label
            depth_str = f" {actor.depth:.2f}d" if actor.depth else ""
            label = f"#{actor.track_id} {actor.class_name}{depth_str}"
            cv2.putText(out, label, (int(b.x1), int(b.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            # Velocity arrow for moving actors
            if actor.is_moving:
                cx = int((b.x1 + b.x2) / 2)
                cy = int((b.y1 + b.y2) / 2)
                vx, vy = actor.velocity_px
                scale = 3.0
                ex = int(cx + vx * scale)
                ey = int(cy + vy * scale)
                cv2.arrowedLine(out, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

        return out

    @staticmethod
    def draw_risk_hud(image: np.ndarray, state: WorldState) -> np.ndarray:
        """Draw risk level overlay on the image."""
        out = image.copy()
        h, w = out.shape[:2]

        # Risk level bar — bottom of image
        level_color = state.level_color
        bar_h = 6
        cv2.rectangle(out, (0, h - bar_h), (w, h), level_color, -1)

        # Risk info panel — top right
        panel_x = w - 220
        lines = [
            f"Risk: {state.level_name} ({state.risk_score:.2f})",
            f"Trend: {state.risk_trend}",
            f"Actors: {state.actor_count}",
            f"Confidence: {state.perception_confidence:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(out, line, (panel_x, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        level_color, 1, cv2.LINE_AA)

        return out

    @staticmethod
    def make_composite(
        camera_image: np.ndarray,
        bev_image: np.ndarray,
        state: WorldState,
        target_width: int = 1200
    ) -> np.ndarray:
        """
        Build the full composite demo display.
        Camera view (left) + BEV grid (right) + risk HUD.
        """
        # Annotate camera view
        cam = WorldModelVisualizer.draw_tracks(camera_image, state)
        cam = WorldModelVisualizer.draw_risk_hud(cam, state)

        # Resize both to same height
        target_h = 450
        cam_w = int(cam.shape[1] * target_h / cam.shape[0])
        cam_resized = cv2.resize(cam, (cam_w, target_h))

        bev_resized = cv2.resize(bev_image, (target_h, target_h))

        # Concatenate side by side
        composite = np.concatenate([cam_resized, bev_resized], axis=1)

        # Title bar
        title_bar = np.zeros((35, composite.shape[1], 3), dtype=np.uint8)
        level_color = state.level_color
        cv2.rectangle(title_bar, (0, 0), (composite.shape[1], 35), (20, 20, 20), -1)
        cv2.putText(title_bar,
                    f"PRISM  |  Frame {state.frame_idx}  |  {state.level_name}  |  "
                    f"{state.actor_count} actors  |  Risk {state.risk_score:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, level_color, 1, cv2.LINE_AA)

        composite = np.vstack([title_bar, composite])
        return composite
