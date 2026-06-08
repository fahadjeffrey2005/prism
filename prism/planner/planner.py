"""
PRISM — Adaptive Planner
=========================
Layer 6 in the PRISM pipeline.

Converts an ArbitrationDecision into a concrete, physically consistent plan:
    - Jerk-limited velocity profile (smooth transitions, no abrupt speed changes)
    - Spatial braking guarantee (stops before nearest corridor obstacle)
    - Emergency override (bypasses jerk limit for collision-imminent stops)
    - Longitudinal waypoints projected along ego heading
    - Control command: throttle ∈ [0,1], brake ∈ [0,1], steer ∈ [-1,1]

Design philosophy:
    The Arbitration Core decides WHAT to do (SLOW, CAUTION, STOP, …).
    The Adaptive Planner decides HOW to do it — the smooth trajectory that
    realises that decision within the vehicle's physical limits.

    Two safety layers run every frame:
        1. Jerk limit  — prevents passenger discomfort and actuator wear
        2. Spatial stop — guarantees we halt before any detected corridor obstacle

    EMERGENCY always bypasses (1) — comfort sacrificed for safety.

Input:
    ArbitrationDecision  — action, speed_factor, confidence, level
    metric_dets          — list[MetricDetection] from MetricDepthEngine
    timestamp            — float (seconds, monotonic)

Output:
    PlannerOutput:
        target_speed_mps    — immediate speed setpoint (m/s)
        current_speed_mps   — ramp-tracked current speed (m/s)
        velocity_profile    — [v0, v1, …] at plan_dt intervals over horizon
        waypoints           — [(y_fwd_m, x_lat_m), …] in ego frame
        control             — ControlCommand(throttle, brake, steer, gear)
        stopping_distance_m — physics-based stop distance from current speed
        plan_horizon_s      — seconds the profile covers
        emergency           — True if emergency brake active
        reason              — one-line explanation
        audit               — full decision audit string
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from prism.utils.common import get_logger

logger = get_logger("AdaptivePlanner")


# ── Vehicle dynamics constants ────────────────────────────────────────────────

# nuScenes urban scenes; override via config
DEFAULT_CRUISE_SPEED_MPS = 4.17   # 15 km/h — conservative campus/urban default
DEFAULT_MAX_ACCEL        = 2.0    # m/s²  — gentle urban acceleration
DEFAULT_MAX_DECEL        = 4.0    # m/s²  — firm braking (0.4 g)
DEFAULT_EMERGENCY_DECEL  = 8.0    # m/s²  — panic brake (0.8 g)
DEFAULT_MAX_JERK         = 2.0    # m/s³  — ISO 2631 comfort limit
DEFAULT_PLAN_HORIZON_S   = 3.0    # seconds ahead
DEFAULT_PLAN_DT_S        = 0.1    # 10 Hz planning rate
DEFAULT_SAFETY_MARGIN_M  = 3.0    # clearance to keep ahead of nearest obstacle

# Minimum speed below which we clamp to zero (prevents crawl-forever edge case)
SPEED_ZERO_THRESHOLD_MPS = 0.05

# Arbitration level → target speed factor (mirrors DECISIONS in decision.py)
# Used for spatial braking logic (we need the factor as float)
_SPEED_FACTOR = {
    "CLEAR":     1.00,
    "MONITOR":   1.00,
    "EASE":      0.85,
    "SLOW":      0.65,
    "CAUTION":   0.40,
    "YIELD":     0.15,
    "STOP":      0.00,
    "EMERGENCY": 0.00,
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ControlCommand:
    """
    Normalised actuator command.
    Throttle and brake are mutually exclusive — never both > 0 simultaneously.
    """
    throttle: float    # [0.0, 1.0]  — 0 = coast, 1 = full throttle
    brake:    float    # [0.0, 1.0]  — 0 = no brake, 1 = maximum brake
    steer:    float    # [-1.0, 1.0] — negative = left, positive = right
    gear:     str      # "D" | "N" | "R"

    def __post_init__(self):
        # Clamp to valid ranges
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.brake    = float(np.clip(self.brake,    0.0, 1.0))
        self.steer    = float(np.clip(self.steer,   -1.0, 1.0))
        # Mutual exclusion — brake takes priority
        if self.brake > 0.01:
            self.throttle = 0.0


@dataclass
class PlannerOutput:
    """
    Full output of one AdaptivePlanner.plan() call.
    Every field is populated — nothing optional downstream.
    """
    timestamp:    float
    frame_idx:    int

    # Speed
    target_speed_mps:  float          # setpoint the ramp is converging to
    current_speed_mps: float          # ramp's tracked current speed

    # Trajectory
    velocity_profile:  List[float]    # v(t) at each plan_dt step over horizon
    waypoints:         List[Tuple[float, float]]  # (y_fwd_m, x_lat_m) ego-frame

    # Control
    control: ControlCommand

    # Meta
    plan_horizon_s:      float
    stopping_distance_m: float        # d = v² / (2 * max_decel)
    emergency:           bool         # True if emergency decel applied
    spatial_override:    bool         # True if spatial braking overrode arb target
    reason:              str          # one-line
    audit:               str          # full multi-line explanation


# ── Velocity Ramp ─────────────────────────────────────────────────────────────

class VelocityRamp:
    """
    Jerk-limited velocity ramp.

    Models the physical constraint that acceleration cannot jump instantaneously.
    At each step:
        desired_accel = clip(dv / dt, -max_decel, max_accel)
        da = clip(desired_accel - current_accel, -max_jerk*dt, +max_jerk*dt)
        current_accel += da
        velocity = max(0, velocity + current_accel * dt)

    This guarantees a smooth S-curve approach to any speed target.
    Emergency stop bypasses jerk limit to maximise deceleration rate.
    """

    def __init__(
        self,
        max_accel:       float = DEFAULT_MAX_ACCEL,
        max_decel:       float = DEFAULT_MAX_DECEL,
        emergency_decel: float = DEFAULT_EMERGENCY_DECEL,
        max_jerk:        float = DEFAULT_MAX_JERK,
        dt:              float = DEFAULT_PLAN_DT_S,
    ):
        self.max_accel       = max_accel
        self.max_decel       = max_decel
        self.emergency_decel = emergency_decel
        self.max_jerk        = max_jerk
        self.dt              = dt

        self.v: float = 0.0   # current speed
        self.a: float = 0.0   # current acceleration

    def reset(self, v: float = 0.0):
        self.v = v
        self.a = 0.0

    # ── Single-step advance ───────────────────────────────────────────────────

    def step(self, v_target: float) -> float:
        """Advance one dt toward v_target with jerk limiting. Returns new speed."""
        dv         = v_target - self.v
        a_desired  = np.clip(dv / self.dt, -self.max_decel, self.max_accel)
        da         = np.clip(
            a_desired - self.a,
            -self.max_jerk * self.dt,
            +self.max_jerk * self.dt,
        )
        self.a     = np.clip(self.a + da, -self.max_decel, self.max_accel)
        self.v     = max(0.0, self.v + self.a * self.dt)
        if self.v < SPEED_ZERO_THRESHOLD_MPS and v_target < SPEED_ZERO_THRESHOLD_MPS:
            self.v = 0.0
            self.a = 0.0
        return self.v

    def emergency_step(self) -> float:
        """One dt at maximum deceleration — no jerk limit."""
        self.a = -self.emergency_decel
        self.v = max(0.0, self.v + self.a * self.dt)
        if self.v < SPEED_ZERO_THRESHOLD_MPS:
            self.v = 0.0
            self.a = 0.0
        return self.v

    # ── Look-ahead planning ───────────────────────────────────────────────────

    def plan(self, v_target: float, n_steps: int, emergency: bool = False) -> List[float]:
        """
        Return n_steps future velocities WITHOUT advancing the ramp state.
        Used for look-ahead; the real step happens once per frame in plan().
        """
        v_saved = self.v
        a_saved = self.a
        profile = []
        for _ in range(n_steps):
            if emergency:
                v = self.emergency_step()
            else:
                v = self.step(v_target)
            profile.append(v)
        self.v = v_saved
        self.a = a_saved
        return profile

    # ── Physics helpers ───────────────────────────────────────────────────────

    def stopping_distance(self, v: Optional[float] = None, decel: Optional[float] = None) -> float:
        """Minimum distance to stop from speed v using given decel (m/s²)."""
        v     = self.v if v is None else v
        decel = self.max_decel if decel is None else decel
        return (v ** 2) / (2.0 * max(decel, 0.01))

    def time_to_stop(self, v: Optional[float] = None, decel: Optional[float] = None) -> float:
        """Seconds to stop from speed v."""
        v     = self.v if v is None else v
        decel = self.max_decel if decel is None else decel
        return v / max(decel, 0.01)


# ── Adaptive Planner ─────────────────────────────────────────────────────────

class AdaptivePlanner:
    """
    Converts ArbitrationDecision → jerk-limited velocity profile + control.

    Call once per frame:
        plan_out = planner.plan(arb_decision, metric_dets, timestamp)

    The planner maintains an internal VelocityRamp that tracks the vehicle's
    physical speed history across frames, ensuring continuity between frames.
    """

    def __init__(self, cfg: dict):
        plan_cfg = cfg.get("planner", {})

        self.cruise_speed_mps  = float(plan_cfg.get("cruise_speed_mps",  DEFAULT_CRUISE_SPEED_MPS))
        self.max_accel         = float(plan_cfg.get("max_accel_mps2",    DEFAULT_MAX_ACCEL))
        self.max_decel         = float(plan_cfg.get("max_decel_mps2",    DEFAULT_MAX_DECEL))
        self.emergency_decel   = float(plan_cfg.get("emergency_decel_mps2", DEFAULT_EMERGENCY_DECEL))
        self.max_jerk          = float(plan_cfg.get("max_jerk_mps3",     DEFAULT_MAX_JERK))
        self.plan_horizon_s    = float(plan_cfg.get("plan_horizon_s",    DEFAULT_PLAN_HORIZON_S))
        self.plan_dt           = float(plan_cfg.get("plan_dt_s",         DEFAULT_PLAN_DT_S))
        self.safety_margin_m   = float(plan_cfg.get("safety_margin_m",   DEFAULT_SAFETY_MARGIN_M))

        self.n_steps = max(1, int(self.plan_horizon_s / self.plan_dt))

        self._ramp = VelocityRamp(
            max_accel       = self.max_accel,
            max_decel       = self.max_decel,
            emergency_decel = self.emergency_decel,
            max_jerk        = self.max_jerk,
            dt              = self.plan_dt,
        )
        self._frame_idx = 0

        logger.info(
            f"AdaptivePlanner: cruise={self.cruise_speed_mps:.1f}m/s "
            f"accel={self.max_accel}m/s² decel={self.max_decel}m/s² "
            f"jerk={self.max_jerk}m/s³ horizon={self.plan_horizon_s}s"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        arb_decision,                      # ArbitrationDecision
        metric_dets:  Optional[list] = None,
        timestamp:    float = 0.0,
    ) -> PlannerOutput:
        """
        Main planning call. Called once per frame.

        Args:
            arb_decision: ArbitrationDecision from ArbitrationCore
            metric_dets:  list of MetricDetection — used for spatial braking
            timestamp:    current frame time (monotonic seconds)

        Returns:
            PlannerOutput — velocity profile, waypoints, control command
        """
        self._frame_idx += 1
        metric_dets = metric_dets or []

        action       = arb_decision.action
        speed_factor = arb_decision.speed_factor   # 0.0 – 1.0
        arb_conf     = arb_decision.confidence

        # ── 1. Compute nominal target speed from arbitration ─────────────────
        v_arb_target = speed_factor * self.cruise_speed_mps

        # ── 2. Spatial braking check ─────────────────────────────────────────
        #    Find the nearest obstacle in the driving corridor.
        #    If our stopping distance (with margin) exceeds the gap, clamp the
        #    target speed down so we can physically stop in time.
        nearest_m, spatial_override, v_spatial_target = self._spatial_brake_check(
            v_current    = self._ramp.v,
            metric_dets  = metric_dets,
        )

        # Take the more conservative of arb target and spatial target
        v_target  = min(v_arb_target, v_spatial_target)
        emergency = (action == "EMERGENCY") or (
            nearest_m is not None and nearest_m < self.safety_margin_m * 0.5
        )

        # ── 3. Advance the velocity ramp one real step ───────────────────────
        if emergency:
            v_now = self._ramp.emergency_step()
        else:
            v_now = self._ramp.step(v_target)

        # ── 4. Look-ahead profile (read-only — does NOT advance ramp) ────────
        velocity_profile = self._ramp.plan(v_target, self.n_steps, emergency)

        # ── 5. Generate longitudinal waypoints ───────────────────────────────
        waypoints = self._project_waypoints(velocity_profile)

        # ── 6. Compute control command ───────────────────────────────────────
        control = self._derive_control(
            v_current  = v_now,
            v_next     = velocity_profile[0] if velocity_profile else 0.0,
            emergency  = emergency,
        )

        # ── 7. Stopping distance at current speed ────────────────────────────
        d_stop = self._ramp.stopping_distance(
            v     = v_now,
            decel = self.emergency_decel if emergency else self.max_decel,
        )

        # ── 8. Build reason and audit strings ────────────────────────────────
        reason, audit = self._build_audit(
            action           = action,
            v_arb_target     = v_arb_target,
            v_target         = v_target,
            v_now            = v_now,
            speed_factor     = speed_factor,
            nearest_m        = nearest_m,
            spatial_override = spatial_override,
            emergency        = emergency,
            arb_conf         = arb_conf,
            d_stop           = d_stop,
            control          = control,
        )

        logger.debug(
            f"[PLN] Frame {self._frame_idx:>3}  {action:<10} "
            f"v={v_now:.1f}m/s → {v_target:.1f}m/s  "
            f"thr={control.throttle:.2f}  brk={control.brake:.2f}  "
            f"{'[EMRG]' if emergency else ''}"
            f"{'[SPA]' if spatial_override else ''}"
        )

        return PlannerOutput(
            timestamp           = timestamp,
            frame_idx           = self._frame_idx,
            target_speed_mps    = v_target,
            current_speed_mps   = v_now,
            velocity_profile    = velocity_profile,
            waypoints           = waypoints,
            control             = control,
            plan_horizon_s      = self.plan_horizon_s,
            stopping_distance_m = d_stop,
            emergency           = emergency,
            spatial_override    = spatial_override,
            reason              = reason,
            audit               = audit,
        )

    def reset(self, v: float = 0.0):
        """Reset ramp to a given speed (call when resuming after a stop)."""
        self._ramp.reset(v)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _spatial_brake_check(
        self,
        v_current:   float,
        metric_dets: list,
    ) -> Tuple[Optional[float], bool, float]:
        """
        Scan metric detections for the nearest obstacle in the driving corridor.

        Returns:
            nearest_m        — distance to nearest corridor obstacle (None if clear)
            spatial_override — True if we had to clamp the speed target
            v_spatial_target — maximum safe speed given the gap
        """
        # Filter to corridor obstacles only (threat_zone == "corridor")
        corridor_dets = [
            d for d in metric_dets
            if getattr(d, "threat_zone", "none") == "corridor"
        ]

        if not corridor_dets:
            return None, False, self.cruise_speed_mps   # nothing in corridor → unconstrained

        nearest_m = min(d.distance_m for d in corridor_dets)

        # Gap available = obstacle distance minus safety margin
        gap_m = max(0.0, nearest_m - self.safety_margin_m)

        # Maximum speed such that we can stop within gap_m:
        #   d_stop = v² / (2 * max_decel)  →  v = sqrt(2 * max_decel * gap_m)
        v_max_safe = np.sqrt(2.0 * self.max_decel * gap_m) if gap_m > 0 else 0.0

        override = v_max_safe < v_current - 0.1   # only flag if it's actually constraining
        return nearest_m, override, v_max_safe

    def _project_waypoints(self, velocity_profile: List[float]) -> List[Tuple[float, float]]:
        """
        Project longitudinal waypoints along ego heading (straight ahead).

        Returns list of (y_fwd_m, x_lat_m) in ego frame.
        Lateral offset is 0.0 — pure longitudinal planning (no lane-change yet).
        """
        waypoints: List[Tuple[float, float]] = []
        y_fwd = 0.0
        for v in velocity_profile:
            y_fwd += v * self.plan_dt
            waypoints.append((round(y_fwd, 3), 0.0))
        return waypoints

    def _derive_control(
        self,
        v_current: float,
        v_next:    float,
        emergency: bool,
    ) -> ControlCommand:
        """
        Derive throttle / brake from speed delta.

        Throttle = how hard to push to reach v_next
        Brake    = how hard to slow to reach v_next
        These are normalised to [0, 1] using vehicle limits.
        """
        dv = v_next - v_current

        if emergency:
            # Full brake, no throttle
            return ControlCommand(throttle=0.0, brake=1.0, steer=0.0, gear="D")

        if v_next < SPEED_ZERO_THRESHOLD_MPS and v_current < SPEED_ZERO_THRESHOLD_MPS:
            # Stationary → neutral
            return ControlCommand(throttle=0.0, brake=0.0, steer=0.0, gear="N")

        if dv >= 0:
            # Accelerating
            throttle = float(np.clip(dv / (self.max_accel * self.plan_dt), 0.0, 1.0))
            return ControlCommand(throttle=throttle, brake=0.0, steer=0.0, gear="D")
        else:
            # Decelerating
            brake = float(np.clip(-dv / (self.max_decel * self.plan_dt), 0.0, 1.0))
            return ControlCommand(throttle=0.0, brake=brake, steer=0.0, gear="D")

    def _build_audit(
        self,
        action:           str,
        v_arb_target:     float,
        v_target:         float,
        v_now:            float,
        speed_factor:     float,
        nearest_m:        Optional[float],
        spatial_override: bool,
        emergency:        bool,
        arb_conf:         float,
        d_stop:           float,
        control:          ControlCommand,
    ) -> Tuple[str, str]:
        """Build a one-line reason and multi-line audit string."""

        # One-liner
        if emergency:
            reason = f"EMERGENCY BRAKE — v={v_now:.1f}m/s → 0, max decel {self.emergency_decel}m/s²"
        elif spatial_override:
            reason = (
                f"Spatial brake: obstacle at {nearest_m:.1f}m "
                f"→ clamped to {v_target:.1f}m/s (arb wanted {v_arb_target:.1f}m/s)"
            )
        elif v_target < 0.05:
            reason = f"{action}: full stop — v={v_now:.2f}m/s"
        else:
            reason = (
                f"{action} ({speed_factor*100:.0f}% cruise) → "
                f"v_target={v_target:.1f}m/s  v_now={v_now:.1f}m/s"
            )

        # Full audit
        lines = [
            f"AdaptivePlanner — Frame {self._frame_idx}",
            f"  Arbitration : {action}  (speed_factor={speed_factor:.2f}  conf={arb_conf:.2f})",
            f"  Arb target  : {v_arb_target:.2f} m/s  ({v_arb_target*3.6:.1f} km/h)",
        ]

        if nearest_m is not None:
            lines.append(
                f"  Nearest obs : {nearest_m:.1f}m in corridor"
                + (f"  [SPATIAL OVERRIDE → {v_target:.1f}m/s]" if spatial_override else "  [within safe gap]")
            )
        else:
            lines.append(f"  Nearest obs : clear corridor")

        lines += [
            f"  Effective   : {v_target:.2f} m/s  ({v_target*3.6:.1f} km/h)",
            f"  Current     : {v_now:.2f} m/s  ({v_now*3.6:.1f} km/h)",
            f"  Stop dist   : {d_stop:.1f}m  (at {'emrg' if emergency else 'std'} decel)",
            f"  Control     : throttle={control.throttle:.2f}  brake={control.brake:.2f}  steer={control.steer:.2f}  gear={control.gear}",
        ]

        if emergency:
            lines.append(f"  ⚡ EMERGENCY DECEL {self.emergency_decel}m/s² applied")

        audit = "\n".join(lines)
        return reason, audit

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "frames":     self._frame_idx,
            "v_current":  round(self._ramp.v, 3),
            "a_current":  round(self._ramp.a, 3),
        }
