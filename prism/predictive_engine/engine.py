"""
PRISM — Predictive Engine
==========================
Predicts where every actor will be in the next 0-8 seconds.
Runs continuously during VLM latency gaps — the system never stops thinking.

Three prediction horizons:

SHORT (0-1.5s) — Physics only
    Constant velocity + Kalman smoothing
    Always accurate, always running
    This is the "reflex" layer

MEDIUM (1.5-6s) — Intent weighted
    Maintains probability distribution over possible maneuvers
    Updates when VLM provides semantic labels
    This is the "prediction" layer

LONG (6-30s) — Scene context
    Informed by VLM scene understanding
    Used for proactive speed adjustment
    This is the "anticipation" layer

Key design principle:
    The predictive engine NEVER waits for the VLM.
    It makes the best prediction it can with what it has.
    When VLM updates arrive, it smoothly incorporates them.
    This eliminates jerk and latency-induced surprises.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from prism.utils.common import get_logger

logger = get_logger("PredictiveEngine")

# LSTM intent model — imported lazily so engine works without it
try:
    from prism.predictive_engine.lstm_intent import (
        LSTMIntentNet, LSTMIntentPredictor, load_lstm_model,
        top_maneuver, format_intent_bar,
    )
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False
    logger.warning("lstm_intent module not found — Bayesian-only mode")


# ── Maneuver definitions ──────────────────────────────────────────────────────

MANEUVERS = [
    "constant_velocity",   # 0 — keep going same speed and direction
    "turn_left",           # 1 — turning left
    "turn_right",          # 2 — turning right
    "braking",             # 3 — slowing down
    "accelerating",        # 4 — speeding up
    "stopping",            # 5 — coming to a stop
    "lane_change_left",    # 6 — lateral move left
    "lane_change_right",   # 7 — lateral move right
    "reversing",           # 8 — moving backward
]

# Default prior probabilities for each maneuver
# Most actors on a road are going straight at constant speed
DEFAULT_PRIORS = np.array([
    0.60,  # constant_velocity
    0.06,  # turn_left
    0.06,  # turn_right
    0.10,  # braking
    0.05,  # accelerating
    0.05,  # stopping
    0.04,  # lane_change_left
    0.04,  # lane_change_right
    0.00,  # reversing
], dtype=np.float32)


# ── Single actor predictor ────────────────────────────────────────────────────

@dataclass
class ActorPrediction:
    """
    Predicted future states for a single tracked actor.
    Contains a distribution over possible futures.
    """
    track_id: int
    class_name: str
    current_pos: np.ndarray          # (x_m, y_m, z_m) current position
    current_vel: np.ndarray          # (vx, vy) velocity in m/s

    # Short horizon — single deterministic trajectory (physics)
    short_trajectory: np.ndarray     # (N_short, 3) positions at t+dt intervals
    short_times: np.ndarray          # timestamps for short trajectory

    # Medium horizon — multiple weighted trajectories
    medium_trajectories: list        # list of (trajectory, probability, maneuver)
    medium_times: np.ndarray         # timestamps for medium trajectory

    # Maneuver probabilities
    maneuver_probs: np.ndarray       # shape (9,) — prob for each maneuver

    # Risk contribution from this actor's predicted trajectories
    predicted_risk: float = 0.0

    # Ego collision time estimate
    ttc: Optional[float] = None      # Time To Collision in seconds

    @property
    def most_likely_maneuver(self) -> str:
        return MANEUVERS[int(np.argmax(self.maneuver_probs))]

    @property
    def is_on_collision_course(self) -> bool:
        return self.ttc is not None and self.ttc < 3.0

    @property
    def expected_position_1s(self) -> np.ndarray:
        """Best estimate of where this actor will be in 1 second."""
        if len(self.short_trajectory) > 0:
            # Find index closest to 1.0s
            idx = np.argmin(np.abs(self.short_times - 1.0))
            return self.short_trajectory[idx]
        return self.current_pos

    @property
    def expected_position_3s(self) -> np.ndarray:
        """Weighted average position across all medium trajectories at 3s."""
        if not self.medium_trajectories:
            return self.current_pos + self.current_vel * 3.0
        positions = []
        weights = []
        for traj, prob, _ in self.medium_trajectories:
            if len(traj) > 0:
                idx = np.argmin(np.abs(self.medium_times - 3.0))
                if idx < len(traj):
                    positions.append(traj[idx])
                    weights.append(prob)
        if not positions:
            return self.current_pos
        weights = np.array(weights)
        weights /= weights.sum()
        return np.average(positions, axis=0, weights=weights)


class ActorPredictor:
    """
    Predicts future trajectory of a single actor.
    Maintains a running belief over its maneuver intent.
    Updated every frame — deterministic and fast.
    """

    # Time steps
    SHORT_DT = 0.1    # 100ms steps for short horizon
    SHORT_HORIZON = 1.5  # seconds
    MEDIUM_DT = 0.25  # 250ms steps for medium horizon
    MEDIUM_HORIZON = 6.0  # seconds

    def __init__(self, track_id: int, class_name: str):
        self.track_id = track_id
        self.class_name = class_name
        self.maneuver_probs = DEFAULT_PRIORS.copy()

        # History for intent inference
        self.pos_history = []     # list of (x, y, z)
        self.vel_history = []     # list of (vx, vy)
        self.time_history = []

        # Pixel-to-metric velocity scale (calibrated externally)
        self.px_to_m_scale = 0.05  # rough: 1 pixel/frame ≈ 0.05 m/s at 10m

    def update(
        self,
        pos_m: np.ndarray,
        vel_px: tuple,
        timestamp: float,
        depth_m: float,
        vlm_intent: Optional[str] = None
    ) -> ActorPrediction:
        """
        Update predictor with new observation and return prediction.

        Args:
            pos_m:      (x, y, z) position in metres
            vel_px:     (vx, vy) velocity in pixels/frame
            timestamp:  current time
            depth_m:    distance from ego in metres
            vlm_intent: optional VLM-provided intent label
        """
        # Convert pixel velocity to metric
        # Scale by depth: objects far away have smaller pixel displacement per m/s
        scale = self.px_to_m_scale * (depth_m / 10.0)
        vel_m = np.array([vel_px[0] * scale, vel_px[1] * scale, 0.0])

        # Update history
        self.pos_history.append(pos_m.copy())
        self.vel_history.append(vel_m.copy())
        self.time_history.append(timestamp)
        if len(self.pos_history) > 30:
            self.pos_history.pop(0)
            self.vel_history.pop(0)
            self.time_history.pop(0)

        # Update maneuver beliefs from observed behavior
        self._update_beliefs(pos_m, vel_m, vlm_intent)

        # Generate predictions
        short_traj, short_times = self._predict_short(pos_m, vel_m)
        medium_trajs, medium_times = self._predict_medium(pos_m, vel_m)

        # Compute TTC
        ttc = self._estimate_ttc(short_traj, short_times, depth_m)

        # Predicted risk
        pred_risk = self._compute_predicted_risk(depth_m, ttc, vel_m)

        return ActorPrediction(
            track_id=self.track_id,
            class_name=self.class_name,
            current_pos=pos_m,
            current_vel=vel_m,
            short_trajectory=short_traj,
            short_times=short_times,
            medium_trajectories=medium_trajs,
            medium_times=medium_times,
            maneuver_probs=self.maneuver_probs.copy(),
            predicted_risk=pred_risk,
            ttc=ttc,
        )

    def _update_beliefs(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        vlm_intent: Optional[str]
    ):
        """
        Update maneuver probability distribution from observations.
        Uses Bayesian-style update: prior × likelihood → posterior.
        """
        if len(self.vel_history) < 2:
            return

        speed = float(np.linalg.norm(vel[:2]))
        prev_vel = self.vel_history[-2] if len(self.vel_history) >= 2 else vel

        # Compute acceleration
        accel = vel - prev_vel
        accel_mag = float(np.linalg.norm(accel[:2]))

        # Compute turn rate (change in direction)
        prev_speed = float(np.linalg.norm(prev_vel[:2]))
        turn_rate = 0.0
        if speed > 0.1 and prev_speed > 0.1:
            cos_angle = np.clip(
                np.dot(vel[:2], prev_vel[:2]) / (speed * prev_speed), -1, 1
            )
            turn_rate = float(1.0 - cos_angle)

        # Lateral vs forward velocity ratio
        lateral_ratio = abs(vel[0]) / (abs(vel[2]) + 0.01) if abs(vel[2]) > 0.1 else 0.0

        # Build likelihood vector — how consistent is each maneuver with observation?
        likelihood = np.ones(len(MANEUVERS), dtype=np.float32)

        # constant_velocity: low acceleration, low turn
        likelihood[0] = np.exp(-accel_mag * 2.0) * np.exp(-turn_rate * 3.0)

        # turning: high turn rate
        if vel[0] < -0.1:  # moving left
            likelihood[1] = 1.0 + turn_rate * 5.0  # turn_left
            likelihood[2] = max(0.1, 1.0 - turn_rate * 3.0)
        else:
            likelihood[2] = 1.0 + turn_rate * 5.0  # turn_right
            likelihood[1] = max(0.1, 1.0 - turn_rate * 3.0)

        # braking: negative acceleration in forward direction
        forward_accel = accel[2] if len(accel) > 2 else 0.0
        if forward_accel < -0.05:
            likelihood[3] = 1.0 + abs(forward_accel) * 5.0
        elif forward_accel > 0.05:
            likelihood[4] = 1.0 + forward_accel * 3.0

        # stopping: very low speed
        if speed < 0.15:
            likelihood[5] = 3.0
            likelihood[0] = 0.2

        # lane changes: high lateral ratio
        if lateral_ratio > 0.3:
            if vel[0] < 0:
                likelihood[6] = 1.0 + lateral_ratio * 3.0
            else:
                likelihood[7] = 1.0 + lateral_ratio * 3.0

        # Bayesian update
        posterior = self.maneuver_probs * likelihood
        total = posterior.sum()
        if total > 0:
            self.maneuver_probs = posterior / total
        else:
            self.maneuver_probs = DEFAULT_PRIORS.copy()

        # VLM override — if VLM says something, strongly weight it
        if vlm_intent is not None:
            vlm_map = {
                "braking": 3, "stopping": 5, "turning_left": 1,
                "turning_right": 2, "accelerating": 4,
                "lane_change_left": 6, "lane_change_right": 7,
                "constant": 0
            }
            if vlm_intent in vlm_map:
                idx = vlm_map[vlm_intent]
                boost = np.ones(len(MANEUVERS), dtype=np.float32) * 0.5
                boost[idx] = 3.0
                self.maneuver_probs = self.maneuver_probs * boost
                self.maneuver_probs /= self.maneuver_probs.sum()

    def _predict_short(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """
        Short horizon prediction — pure constant velocity physics.
        Fast, deterministic, always accurate in the short term.
        """
        times = np.arange(self.SHORT_DT, self.SHORT_HORIZON + self.SHORT_DT, self.SHORT_DT)
        trajectory = np.array([pos + vel * t for t in times])
        return trajectory, times

    def _predict_medium(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """
        Medium horizon — generate one trajectory per plausible maneuver.
        Each weighted by current maneuver probability.
        Returns only maneuvers with probability > 5%.
        """
        times = np.arange(self.MEDIUM_DT, self.MEDIUM_HORIZON + self.MEDIUM_DT, self.MEDIUM_DT)
        trajectories = []
        speed = float(np.linalg.norm(vel[:2]))

        for i, maneuver in enumerate(MANEUVERS):
            prob = float(self.maneuver_probs[i])
            if prob < 0.05:
                continue

            traj = self._generate_maneuver_trajectory(pos, vel, times, maneuver, speed)
            trajectories.append((traj, prob, maneuver))

        return trajectories, times

    def _generate_maneuver_trajectory(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        times: np.ndarray,
        maneuver: str,
        speed: float
    ) -> np.ndarray:
        """Generate a trajectory for a specific maneuver."""
        trajectory = []
        current_pos = pos.copy()
        current_vel = vel.copy()
        dt = float(times[1] - times[0]) if len(times) > 1 else self.MEDIUM_DT

        for _ in times:
            if maneuver == "constant_velocity":
                pass  # velocity unchanged

            elif maneuver == "braking":
                decel = 0.15  # m/s² deceleration
                speed_now = np.linalg.norm(current_vel[:2])
                if speed_now > 0.01:
                    direction = current_vel[:2] / speed_now
                    new_speed = max(0, speed_now - decel * dt)
                    current_vel[:2] = direction * new_speed

            elif maneuver == "stopping":
                decel = 0.4
                speed_now = np.linalg.norm(current_vel[:2])
                if speed_now > 0.01:
                    direction = current_vel[:2] / speed_now
                    new_speed = max(0, speed_now - decel * dt)
                    current_vel[:2] = direction * new_speed

            elif maneuver == "accelerating":
                accel = 0.1
                speed_now = np.linalg.norm(current_vel[:2])
                if speed_now > 0.01:
                    direction = current_vel[:2] / speed_now
                    current_vel[:2] = direction * min(speed_now + accel * dt, 2.0)

            elif maneuver == "turn_left":
                turn_rate = 0.08  # rad/s
                angle = turn_rate * dt
                c, s = np.cos(angle), np.sin(angle)
                vx = c * current_vel[0] - s * current_vel[2]
                vz = s * current_vel[0] + c * current_vel[2]
                current_vel[0], current_vel[2] = vx, vz

            elif maneuver == "turn_right":
                turn_rate = -0.08
                angle = turn_rate * dt
                c, s = np.cos(angle), np.sin(angle)
                vx = c * current_vel[0] - s * current_vel[2]
                vz = s * current_vel[0] + c * current_vel[2]
                current_vel[0], current_vel[2] = vx, vz

            elif maneuver in ("lane_change_left", "lane_change_right"):
                lateral_push = -0.05 if maneuver == "lane_change_left" else 0.05
                current_vel[0] += lateral_push * dt

            current_pos = current_pos + current_vel * dt
            trajectory.append(current_pos.copy())

        return np.array(trajectory)

    def _estimate_ttc(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
        current_depth: float
    ) -> Optional[float]:
        """
        Estimate Time To Collision.
        Simple: find when predicted distance drops below collision threshold.
        """
        COLLISION_THRESHOLD = 3.0  # metres

        for i, t in enumerate(times):
            if i >= len(trajectory):
                break
            # Forward distance decreasing — actor coming toward ego
            predicted_dist = trajectory[i][2] if len(trajectory[i]) > 2 else current_depth
            if predicted_dist < COLLISION_THRESHOLD:
                return float(t)

        return None

    def _compute_predicted_risk(
        self,
        depth_m: float,
        ttc: Optional[float],
        vel: np.ndarray
    ) -> float:
        """Compute risk score based on predicted behavior."""
        risk = 0.0

        # Proximity risk
        proximity = np.clip(1.0 - depth_m / 30.0, 0.0, 1.0)
        risk += proximity * 0.4

        # TTC risk — collision imminent
        if ttc is not None:
            ttc_risk = np.clip(1.0 - ttc / 5.0, 0.0, 1.0)
            risk += ttc_risk * 0.5

        # Uncertainty risk — multiple plausible maneuvers = unpredictable actor
        entropy = -np.sum(
            self.maneuver_probs * np.log(self.maneuver_probs + 1e-8)
        )
        max_entropy = np.log(len(MANEUVERS))
        uncertainty = entropy / max_entropy
        risk += uncertainty * 0.1

        return float(np.clip(risk, 0.0, 1.0))


# ── Predictive Engine ─────────────────────────────────────────────────────────

@dataclass
class PredictiveState:
    """
    Output of the Predictive Engine for one frame.
    Read by the Arbitration Core to make decisions.
    """
    timestamp: float
    frame_idx: int
    predictions: list                    # List[ActorPrediction]
    scene_risk_predicted: float          # predicted risk in next 3s
    divergence_score: float              # how much reality diverged from last prediction
    actors_on_collision_course: list     # track_ids with TTC < 3s
    recommended_action: str              # "continue" | "slow" | "brake" | "stop"
    recommended_speed_factor: float      # 1.0 = normal, 0.5 = half speed, 0.0 = stop

    @property
    def has_collision_risk(self) -> bool:
        return len(self.actors_on_collision_course) > 0


class PredictiveEngine:
    """
    Maintains per-actor predictors and produces scene-level predictions.

    Runs every frame — pure Python/numpy, no neural network.
    Latency: < 5ms regardless of scene complexity.

    Intent classification hierarchy:
        1. LSTM (if checkpoint loaded) — trained on nuScenes, 10-frame history
        2. Bayesian fallback — hand-tuned kinematic likelihoods, always available

    The VLM updates this engine asynchronously via update_vlm_intents().
    The engine never waits for VLM — it always produces predictions.
    """

    DEFAULT_CHECKPOINT = "/home/koushik-test/prism_data/checkpoints/lstm_intent/model.pt"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.actor_predictors: dict  = {}   # track_id → ActorPredictor
        self._frame_idx = 0
        self._last_predictions: dict = {}   # track_id → ActorPrediction
        self._vlm_intents: dict      = {}   # track_id → intent_str from VLM

        # ── LSTM intent model ──────────────────────────────────────────────
        self._lstm_model    = None
        self._lstm_device   = "cpu"
        self._lstm_predictors: dict = {}    # track_id → LSTMIntentPredictor
        self._lstm_active   = False

        if _LSTM_AVAILABLE:
            checkpoint = cfg.get("lstm_intent_checkpoint", self.DEFAULT_CHECKPOINT)
            import torch
            if torch.cuda.is_available():
                self._lstm_device = "cuda"
            elif torch.backends.mps.is_available():
                self._lstm_device = "mps"

            self._lstm_model = load_lstm_model(checkpoint, self._lstm_device)
            if self._lstm_model is not None:
                self._lstm_active = True
                logger.info(
                    f"LSTM intent model active on {self._lstm_device} — "
                    f"Bayesian fallback disabled for actors with ≥10 frames"
                )
            else:
                logger.info("LSTM checkpoint not found — using Bayesian intent classifier")
        # ──────────────────────────────────────────────────────────────────

        logger.info("Predictive Engine ready")

    def update(self, world_state, metric_detections: list = None) -> PredictiveState:
        """
        Main update — called every frame after World Model update.

        Args:
            world_state: current WorldState from World Model
            metric_detections: MetricDetection list (with real distances)
        """
        self._frame_idx += 1

        # Build metric distance lookup
        metric_lookup = {}
        if metric_detections:
            for md in metric_detections:
                if md.bbox.track_id is not None:
                    metric_lookup[md.bbox.track_id] = md

        # Update / create predictors for each tracked actor
        predictions = []
        for actor in world_state.actors:
            if not actor.is_confirmed:
                continue

            # Get or create predictor
            if actor.track_id not in self.actor_predictors:
                self.actor_predictors[actor.track_id] = ActorPredictor(
                    actor.track_id, actor.class_name
                )
            predictor = self.actor_predictors[actor.track_id]

            # Get metric distance
            md = metric_lookup.get(actor.track_id)
            if md:
                distance_m = md.distance_m
                lateral_m = md.lateral_m
                pos_m = np.array([lateral_m, 0.0, distance_m])
            elif actor.depth is not None:
                distance_m = actor.depth * 50.0  # rough fallback
                pos_m = np.array([0.0, 0.0, distance_m])
            else:
                distance_m = 30.0
                pos_m = np.array([0.0, 0.0, distance_m])

            # Get VLM intent if available
            vlm_intent = self._vlm_intents.get(actor.track_id)

            # ── LSTM intent override ───────────────────────────────────────
            lstm_probs = None
            if self._lstm_active and self._lstm_model is not None:
                if actor.track_id not in self._lstm_predictors:
                    self._lstm_predictors[actor.track_id] = LSTMIntentPredictor(
                        self._lstm_model, self._lstm_device
                    )
                lstm_pred  = self._lstm_predictors[actor.track_id]
                lstm_probs = lstm_pred.update(pos_m, world_state.timestamp)

                if lstm_probs is not None:
                    # Replace maneuver probs in the Bayesian predictor with LSTM output
                    predictor.maneuver_probs = lstm_probs.copy()
                    top = top_maneuver(lstm_probs, actor.class_name)
                    top_p = float(lstm_probs.max())
                    logger.debug(
                        f"[LSTM] actor#{actor.track_id} ({actor.class_name}) "
                        f"@ {distance_m:.1f}m → {top} ({top_p*100:.0f}%)"
                    )
            # ──────────────────────────────────────────────────────────────

            # Run predictor (uses maneuver_probs — updated by LSTM above if active)
            pred = predictor.update(
                pos_m=pos_m,
                vel_px=actor.velocity_px,
                timestamp=world_state.timestamp,
                depth_m=distance_m,
                vlm_intent=vlm_intent,
            )
            predictions.append(pred)
            self._last_predictions[actor.track_id] = pred

        # Clean up stale predictors (Bayesian + LSTM)
        active_ids = {a.track_id for a in world_state.actors}
        stale = [tid for tid in self.actor_predictors if tid not in active_ids]
        for tid in stale:
            del self.actor_predictors[tid]
            self._last_predictions.pop(tid, None)
            self._lstm_predictors.pop(tid, None)

        # Scene-level assessment
        collision_actors = [
            p.track_id for p in predictions if p.is_on_collision_course
        ]
        divergence = self._compute_divergence(world_state)
        scene_risk = self._compute_scene_predicted_risk(predictions)
        action, speed_factor = self._recommend_action(
            predictions, scene_risk, divergence, world_state
        )

        return PredictiveState(
            timestamp=world_state.timestamp,
            frame_idx=self._frame_idx,
            predictions=predictions,
            scene_risk_predicted=scene_risk,
            divergence_score=divergence,
            actors_on_collision_course=collision_actors,
            recommended_action=action,
            recommended_speed_factor=speed_factor,
        )

    def update_vlm_intents(self, intents: dict):
        """
        Called by VLM Semantic Reasoner when it produces intent labels.
        Non-blocking — engine uses these on next frame.
        intents: {track_id: intent_string}
        """
        self._vlm_intents.update(intents)

    def _compute_divergence(self, world_state) -> float:
        """
        How much did reality diverge from our last prediction?
        High divergence = something unexpected happened = spike attention.
        """
        if not self._last_predictions:
            return 0.0

        divergences = []
        for actor in world_state.actors:
            if actor.track_id not in self._last_predictions:
                continue
            last_pred = self._last_predictions[actor.track_id]
            if len(last_pred.short_trajectory) == 0:
                continue

            # Compare predicted position at t+1 with actual current position
            if actor.depth is not None:
                actual_z = actor.depth * 50.0
                predicted_z = float(last_pred.short_trajectory[0][2])
                div = abs(actual_z - predicted_z) / max(actual_z, 1.0)
                divergences.append(div)

        return float(np.mean(divergences)) if divergences else 0.0

    def _compute_scene_predicted_risk(self, predictions: list) -> float:
        """Aggregate predicted risk across all actors."""
        if not predictions:
            return 0.0
        risks = [p.predicted_risk for p in predictions]
        return float(0.6 * max(risks) + 0.4 * np.mean(risks))

    def _recommend_action(
        self,
        predictions: list,
        scene_risk: float,
        divergence: float,
        world_state
    ) -> tuple:
        """
        Recommend an action based on predicted state.
        This feeds into the Adaptive Planner.
        Returns (action_string, speed_factor 0-1)
        """
        # Check for collision risk first
        collision_preds = [p for p in predictions if p.is_on_collision_course]
        if collision_preds:
            min_ttc = min(p.ttc for p in collision_preds if p.ttc)
            if min_ttc < 1.0:
                return "stop", 0.0
            elif min_ttc < 2.0:
                return "brake", 0.2
            else:
                return "slow", 0.5

        # High divergence — something unexpected, be cautious
        if divergence > 0.4:
            return "slow", 0.6

        # Risk-based
        if scene_risk > 0.75:
            return "slow", 0.5
        elif scene_risk > 0.50:
            return "slow", 0.7
        elif scene_risk > 0.25:
            return "continue", 0.85

        return "continue", 1.0

    def reset(self):
        self.actor_predictors   = {}
        self._last_predictions  = {}
        self._vlm_intents       = {}
        self._lstm_predictors   = {}
        self._frame_idx         = 0
