"""
PRISM — Smart Decision Engine
==============================
Replaces the simple distance-based action recommender.

Makes decisions using full scene context:
    - Actor proximity AND trajectory direction
    - Scene density and clustering
    - Pedestrian behavioral intent
    - Multi-threat convergence
    - Temporal trend (is situation improving or worsening?)

Decision scale (8 levels):
    CLEAR     → 100% speed — nothing of concern
    MONITOR   → 100% speed — tracking something, no action needed
    EASE      →  85% speed — gentle proactive decel
    SLOW      →  65% speed — meaningful reduction
    CAUTION   →  40% speed — significant reduction, preparing to stop
    YIELD     →  15% speed — near stop, waiting for clear
    STOP      →   0% speed — full stop
    EMERGENCY →   0% speed — maximum braking, collision imminent
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from prism.utils.common import get_logger

logger = get_logger("DecisionEngine")


# ── Decision levels ───────────────────────────────────────────────────────────

DECISIONS = {
    "CLEAR":     {"speed": 1.00, "level": 0, "color": (50,  210, 80)},
    "MONITOR":   {"speed": 1.00, "level": 1, "color": (80,  210, 160)},
    "EASE":      {"speed": 0.85, "level": 2, "color": (50,  200, 220)},
    "SLOW":      {"speed": 0.65, "level": 3, "color": (50,  160, 220)},
    "CAUTION":   {"speed": 0.40, "level": 4, "color": (220, 160, 50)},
    "YIELD":     {"speed": 0.15, "level": 5, "color": (220, 100, 50)},
    "STOP":      {"speed": 0.00, "level": 6, "color": (220, 50,  50)},
    "EMERGENCY": {"speed": 0.00, "level": 7, "color": (180, 0,   220)},
}

# Pedestrian behavioral states
PED_STATES = {
    "stationary_safe":    0,   # on sidewalk, not facing road
    "stationary_watch":   1,   # on sidewalk, facing road
    "moving_parallel":    2,   # moving along road, not crossing
    "approaching_road":   3,   # moving toward road edge
    "crossing":           4,   # actively crossing
    "on_path":            5,   # in ego vehicle path
    "running_toward":     6,   # high speed toward path
}


@dataclass
class ActorAssessment:
    """Full assessment of a single actor."""
    track_id: int
    class_name: str
    distance_m: float
    lateral_m: float
    is_moving: bool
    speed_ms: float                    # estimated m/s
    is_approaching: bool               # moving toward ego
    approach_rate: float               # m/s closing speed (positive = closing)
    time_to_reach: Optional[float]     # seconds until at ego position
    is_in_corridor: bool               # within ±2m of ego centerline
    threat_score: float                # 0-1 composite threat
    ped_state: Optional[int]           # pedestrian behavioral state
    decision_contribution: str         # what decision this actor drives


@dataclass
class SceneAssessment:
    """Full assessment of the current scene."""
    timestamp: float
    frame_idx: int

    # Individual actor assessments
    actor_assessments: list            # List[ActorAssessment]

    # Scene-level metrics
    actor_count: int
    ped_count: int
    vehicle_count: int
    actors_in_corridor: int
    actors_approaching: int
    closest_threat_m: float
    min_time_to_reach: Optional[float]

    # Density zones
    critical_zone_count: int           # < 5m
    close_zone_count: int              # 5-15m
    medium_zone_count: int             # 15-30m

    # Final decision
    decision: str                      # one of DECISIONS keys
    speed_factor: float
    decision_color: tuple
    primary_reason: str                # human readable reason
    secondary_reasons: list            # additional context

    # Trend
    trend: str                         # "improving" | "stable" | "worsening"
    confidence: float                  # how confident is this assessment

    @property
    def decision_level(self) -> int:
        return DECISIONS[self.decision]["level"]

    @property
    def is_safe_to_proceed(self) -> bool:
        return self.decision in ("CLEAR", "MONITOR", "EASE")


class SmartDecisionEngine:
    """
    Context-aware decision engine.
    Replaces simple distance thresholds with full scene understanding.
    """

    # Corridor width — space directly in front of ego
    CORRIDOR_WIDTH_M = 2.5

    # Distance zones
    CRITICAL_M = 5.0
    CLOSE_M = 15.0
    MEDIUM_M = 30.0

    # Approach rate thresholds
    FAST_APPROACH_MS = 3.0     # m/s — closing fast
    SLOW_APPROACH_MS = 0.5     # m/s — barely moving toward

    def __init__(self):
        self._history = []          # recent decisions for trend
        self._max_history = 10
        self._frame_idx = 0
        logger.info("Smart Decision Engine ready")

    def assess(
        self,
        world_state,
        pred_state,
        metric_detections: list,
        ego_speed_mps: float = 5.0,   # actual / commanded vehicle speed
    ) -> SceneAssessment:
        """
        Full scene assessment — called every frame.
        Returns SceneAssessment with decision and full context.
        """
        self._frame_idx += 1

        # Build metric lookup by track_id
        metric_by_track = {}
        for md in metric_detections:
            if md.bbox.track_id is not None:
                metric_by_track[md.bbox.track_id] = md

        # Physics-based distance thresholds, scaled by current speed.
        # Use a minimum of 1.4 m/s (5 km/h) so thresholds never collapse to
        # near-zero even when the vehicle is commanded to stop.
        v_eff = max(float(ego_speed_mps), 1.4)
        stop_dist = v_eff ** 2 / (2 * 4.0)     # comfortable stop at 4 m/s² decel
        emrg_dist = v_eff ** 2 / (2 * 8.0)     # panic stop at 8 m/s² decel

        # ── Assess each actor ─────────────────────────────────────────────────
        assessments = []
        for actor in world_state.actors:
            if not actor.is_confirmed:
                continue
            md = metric_by_track.get(actor.track_id)
            assessment = self._assess_actor(actor, md, pred_state,
                                            stop_dist=stop_dist,
                                            emrg_dist=emrg_dist)
            assessments.append(assessment)

        # ── Scene-level metrics ───────────────────────────────────────────────
        ped_count = sum(1 for a in assessments if a.class_name in
                        ("person", "bicycle", "motorcycle"))
        vehicle_count = sum(1 for a in assessments if a.class_name in
                            ("car", "truck", "bus"))
        in_corridor = sum(1 for a in assessments if a.is_in_corridor)
        approaching = sum(1 for a in assessments if a.is_approaching)

        distances = [a.distance_m for a in assessments]
        closest = min(distances) if distances else 999.0

        times_to_reach = [a.time_to_reach for a in assessments
                          if a.time_to_reach is not None]
        min_ttr = min(times_to_reach) if times_to_reach else None

        # Speed-scaled zone boundaries
        critical_m = max(self.CRITICAL_M,  emrg_dist + 1.5)
        close_m    = max(self.CLOSE_M,     stop_dist * 3.0)
        medium_m   = max(self.MEDIUM_M,    stop_dist * 6.0)
        critical_count = sum(1 for d in distances if d < critical_m)
        close_count = sum(1 for d in distances if critical_m <= d < close_m)
        medium_count = sum(1 for d in distances if close_m <= d < medium_m)

        # ── Make decision ─────────────────────────────────────────────────────
        decision, reason, secondary = self._decide(
            assessments, ped_count, vehicle_count,
            in_corridor, approaching, closest, min_ttr,
            world_state, pred_state
        )

        # ── Trend ─────────────────────────────────────────────────────────────
        self._history.append(DECISIONS[decision]["level"])
        if len(self._history) > self._max_history:
            self._history.pop(0)
        trend = self._compute_trend()

        # ── Confidence ───────────────────────────────────────────────────────
        confidence = world_state.perception_confidence
        if len(assessments) == 0:
            confidence *= 0.6   # no actors = uncertain

        scene = SceneAssessment(
            timestamp=world_state.timestamp,
            frame_idx=self._frame_idx,
            actor_assessments=assessments,
            actor_count=len(assessments),
            ped_count=ped_count,
            vehicle_count=vehicle_count,
            actors_in_corridor=in_corridor,
            actors_approaching=approaching,
            closest_threat_m=closest,
            min_time_to_reach=min_ttr,
            critical_zone_count=critical_count,
            close_zone_count=close_count,
            medium_zone_count=medium_count,
            decision=decision,
            speed_factor=DECISIONS[decision]["speed"],
            decision_color=DECISIONS[decision]["color"],
            primary_reason=reason,
            secondary_reasons=secondary,
            trend=trend,
            confidence=confidence,
        )
        return scene

    def _assess_actor(self, actor, md, pred_state,
                      stop_dist=5.0, emrg_dist=2.5) -> ActorAssessment:
        """Full assessment of a single actor."""

        # Distance and position
        if md:
            dist = md.distance_m
            lateral = md.lateral_m
        elif actor.depth:
            dist = actor.depth * 50.0
            lateral = 0.0
        else:
            dist = 30.0
            lateral = 0.0

        # Velocity
        vx, vy = actor.velocity_px
        speed_px = float(np.sqrt(vx**2 + vy**2))
        # Convert pixel velocity to m/s (rough: 1px/frame ≈ 0.05 m/s at 10m)
        scale = 0.05 * (dist / 10.0)
        speed_ms = speed_px * scale

        # Approach rate — is actor closing distance?
        # vy in image = forward/back movement
        # Positive vy in image = moving down = coming toward camera
        approach_rate = float(vy) * scale if vy > 0 else 0.0

        is_approaching = approach_rate > self.SLOW_APPROACH_MS
        is_in_corridor = abs(lateral) < self.CORRIDOR_WIDTH_M

        # Time to reach ego
        ttr = None
        if is_approaching and approach_rate > 0.1:
            ttr = dist / approach_rate

        # Pedestrian behavioral state
        ped_state = None
        if actor.class_name == "person":
            ped_state = self._classify_ped_state(
                dist, lateral, speed_ms, approach_rate,
                is_in_corridor, is_approaching
            )

        # Threat score — composite
        threat = self._compute_threat(
            dist, lateral, speed_ms, approach_rate,
            is_in_corridor, is_approaching, actor.class_name, ped_state
        )

        # Decision contribution
        contribution = self._actor_contribution(
            dist, is_in_corridor, is_approaching,
            approach_rate, actor.class_name, ped_state, ttr,
            stop_dist=stop_dist, emrg_dist=emrg_dist,
        )

        return ActorAssessment(
            track_id=actor.track_id,
            class_name=actor.class_name,
            distance_m=dist,
            lateral_m=lateral,
            is_moving=actor.is_moving,
            speed_ms=speed_ms,
            is_approaching=is_approaching,
            approach_rate=approach_rate,
            time_to_reach=ttr,
            is_in_corridor=is_in_corridor,
            threat_score=threat,
            ped_state=ped_state,
            decision_contribution=contribution,
        )

    def _classify_ped_state(
        self, dist, lateral, speed_ms,
        approach_rate, in_corridor, approaching
    ) -> int:
        """Classify pedestrian behavioral state."""
        if in_corridor:
            if speed_ms > 2.0 and approaching:
                return PED_STATES["running_toward"]
            elif approaching:
                return PED_STATES["on_path"]
            else:
                return PED_STATES["crossing"]
        else:
            if approaching and approach_rate > 0.5:
                return PED_STATES["approaching_road"]
            elif speed_ms > 0.3 and abs(lateral) > 3:
                return PED_STATES["moving_parallel"]
            elif speed_ms < 0.1:
                if dist < 10:
                    return PED_STATES["stationary_watch"]
                return PED_STATES["stationary_safe"]
            else:
                return PED_STATES["approaching_road"]

    def _compute_threat(
        self, dist, lateral, speed_ms, approach_rate,
        in_corridor, approaching, class_name, ped_state
    ) -> float:
        """Compute 0-1 threat score for a single actor."""
        threat = 0.0

        # Proximity component
        if dist < self.CRITICAL_M:
            threat += 0.5
        elif dist < self.CLOSE_M:
            threat += 0.3 * (1 - (dist - self.CRITICAL_M) / (self.CLOSE_M - self.CRITICAL_M))
        elif dist < self.MEDIUM_M:
            threat += 0.1

        # Corridor component — in path is more dangerous
        if in_corridor:
            threat += 0.25

        # Approach component
        if approaching:
            approach_norm = min(approach_rate / self.FAST_APPROACH_MS, 1.0)
            threat += 0.2 * approach_norm

        # Class-specific weights
        class_weights = {
            "person":     1.4,
            "bicycle":    1.2,
            "motorcycle": 1.1,
            "car":        1.0,
            "truck":      1.1,
            "bus":        1.0,
        }
        threat *= class_weights.get(class_name, 1.0)

        # Pedestrian state modifier
        if ped_state is not None:
            ped_multipliers = {
                PED_STATES["stationary_safe"]:  0.3,
                PED_STATES["stationary_watch"]: 0.6,
                PED_STATES["moving_parallel"]:  0.4,
                PED_STATES["approaching_road"]: 0.9,
                PED_STATES["crossing"]:         1.1,
                PED_STATES["on_path"]:          1.3,
                PED_STATES["running_toward"]:   1.5,
            }
            threat *= ped_multipliers.get(ped_state, 1.0)

        return float(np.clip(threat, 0.0, 1.0))

    def _actor_contribution(
        self, dist, in_corridor, approaching,
        approach_rate, class_name, ped_state, ttr,
        stop_dist: float = 5.0,
        emrg_dist: float = 2.5,
    ) -> str:
        """
        What decision level does this actor drive?

        All thresholds are derived from physics (stopping distance) so they
        automatically scale with vehicle speed — at 5 km/h the thresholds
        are much tighter than at 30 km/h.

        stop_dist = v² / (2 * 4 m/s²)  — comfortable braking distance
        emrg_dist = v² / (2 * 8 m/s²)  — panic braking distance
        """
        # TTC for actors closing toward ego
        ttc = dist / max(approach_rate, 0.01) if approaching else float("inf")

        # ── EMERGENCY ─────────────────────────────────────────────────────────
        # Can't stop even with panic braking, or imminent collision (<1.5s)
        if (dist < emrg_dist + 1.0 and in_corridor) or (ttc < 1.5 and in_corridor):
            return "EMERGENCY"

        # ── STOP ──────────────────────────────────────────────────────────────
        # Need to stop: obstacle is within comfortable stopping distance in path
        if dist < stop_dist + 1.5 and in_corridor and approaching:
            return "STOP"
        if ped_state in (PED_STATES["on_path"], PED_STATES["running_toward"]):
            return "STOP"
        if ttc < 2.5 and in_corridor:
            return "STOP"

        # ── YIELD ─────────────────────────────────────────────────────────────
        ped_yield_m = max(6.0, stop_dist * 2.0)
        if ped_state == PED_STATES["crossing"] and dist < ped_yield_m:
            return "YIELD"
        if dist < stop_dist + 1.5 and in_corridor:
            return "YIELD"   # stationary obstacle inside stopping distance

        # ── CAUTION ───────────────────────────────────────────────────────────
        ped_caution_m = max(5.0, stop_dist * 1.5 + 2.0)
        if ped_state == PED_STATES["approaching_road"] and dist < ped_caution_m:
            return "CAUTION"
        if dist < stop_dist * 3.0 and in_corridor and approaching:
            return "CAUTION"
        if ttc < 5.0 and in_corridor:
            return "CAUTION"

        # ── SLOW ──────────────────────────────────────────────────────────────
        slow_m = max(10.0, stop_dist * 4.0)
        if dist < slow_m and approaching and approach_rate > self.SLOW_APPROACH_MS:
            return "SLOW"
        if ped_state == PED_STATES["stationary_watch"] and dist < max(6.0, stop_dist * 2.0):
            return "SLOW"

        # ── EASE ──────────────────────────────────────────────────────────────
        ease_m = max(15.0, stop_dist * 5.0)
        if dist < ease_m and in_corridor:
            return "EASE"
        if ped_state == PED_STATES["approaching_road"]:
            return "EASE"

        # ── MONITOR ───────────────────────────────────────────────────────────
        monitor_m = max(25.0, stop_dist * 8.0)
        if dist < monitor_m:
            return "MONITOR"

        return "CLEAR"

    def _decide(
        self, assessments, ped_count, vehicle_count,
        in_corridor, approaching, closest, min_ttr,
        world_state, pred_state
    ) -> tuple:
        """
        Scene-level decision.
        Takes the worst individual contribution and adjusts for density.
        Returns (decision, primary_reason, secondary_reasons)
        """
        if not assessments:
            return "CLEAR", "No actors detected", []

        # Find worst individual contribution
        level_order = list(DECISIONS.keys())
        contributions = [a.decision_contribution for a in assessments]
        worst_level = max(contributions,
                          key=lambda d: DECISIONS[d]["level"])

        # Find actor driving worst decision
        primary_actor = next(
            (a for a in assessments if a.decision_contribution == worst_level),
            assessments[0]
        )

        # Build primary reason
        reason = self._build_reason(primary_actor, worst_level)

        # Secondary reasons
        secondary = []

        # Density modifier — many actors = increase caution one level
        total = len(assessments)
        if total >= 5 and DECISIONS[worst_level]["level"] < 4:
            current_idx = level_order.index(worst_level)
            worst_level = level_order[min(current_idx + 1, len(level_order)-1)]
            secondary.append(f"{total} actors in scene — elevated caution")

        # Many peds modifier
        if ped_count >= 3:
            current_idx = level_order.index(worst_level)
            if current_idx < 4:
                worst_level = level_order[min(current_idx + 1, len(level_order)-1)]
            secondary.append(f"{ped_count} pedestrians detected")

        # Convergence — multiple threats approaching simultaneously
        if approaching >= 2 and in_corridor >= 1:
            current_idx = level_order.index(worst_level)
            worst_level = level_order[min(current_idx + 1, len(level_order)-1)]
            secondary.append(f"{approaching} actors converging")

        # Divergence from prediction — uncertainty = caution
        if pred_state.divergence_score > 0.5:
            secondary.append(f"High prediction uncertainty ({pred_state.divergence_score:.2f})")
            if DECISIONS[worst_level]["level"] < 3:
                current_idx = level_order.index(worst_level)
                worst_level = level_order[min(current_idx + 1, len(level_order)-1)]

        # VLM caution override
        if hasattr(world_state, 'vlm_caution'):
            vlm_map = {"elevated": 2, "high": 3, "critical": 5}
            vlm_min = vlm_map.get(world_state.vlm_caution, 0)
            if vlm_min > DECISIONS[worst_level]["level"]:
                worst_level = level_order[vlm_min]
                secondary.append(f"VLM: {world_state.vlm_caution} caution")

        # Improving trend — allow relaxing one level if stable
        if (self._compute_trend() == "improving" and
                DECISIONS[worst_level]["level"] > 2 and
                closest > self.CLOSE_M):
            current_idx = level_order.index(worst_level)
            worst_level = level_order[max(current_idx - 1, 0)]
            secondary.append("Situation improving")

        return worst_level, reason, secondary[:3]

    def _build_reason(self, actor: ActorAssessment, decision: str) -> str:
        """Human-readable reason for the decision."""
        cls = actor.class_name
        dist = actor.distance_m
        lat_str = f"{'L' if actor.lateral_m < 0 else 'R'}{abs(actor.lateral_m):.1f}m"

        if decision == "EMERGENCY":
            return f"{cls} at {dist:.1f}m — imminent collision"
        elif decision == "STOP":
            if actor.ped_state in (PED_STATES["on_path"], PED_STATES["running_toward"]):
                return f"Pedestrian in path at {dist:.1f}m"
            return f"{cls} blocking path at {dist:.1f}m"
        elif decision == "YIELD":
            if actor.ped_state == PED_STATES["crossing"]:
                return f"Pedestrian crossing at {dist:.1f}m — yielding"
            return f"{cls} at {dist:.1f}m in corridor — yielding"
        elif decision == "CAUTION":
            if actor.ped_state == PED_STATES["approaching_road"]:
                return f"Pedestrian approaching road at {dist:.1f}m {lat_str}"
            return f"{cls} closing at {actor.approach_rate:.1f}m/s — {dist:.1f}m ahead"
        elif decision == "SLOW":
            return f"{cls} at {dist:.1f}m {lat_str} — reducing speed"
        elif decision == "EASE":
            return f"{cls} at {dist:.1f}m — easing proactively"
        elif decision == "MONITOR":
            return f"Tracking {cls} at {dist:.1f}m {lat_str}"
        else:
            return "Path clear"

    def _compute_trend(self) -> str:
        """Is situation improving, stable, or worsening?"""
        if len(self._history) < 4:
            return "stable"
        recent = np.mean(self._history[-3:])
        older = np.mean(self._history[-6:-3]) if len(self._history) >= 6 else recent
        delta = recent - older
        if delta > 0.5:   return "worsening"
        if delta < -0.5:  return "improving"
        return "stable"
