"""
PRISM — Arbitration Core
=========================
The final decision layer. Reads four independent signals simultaneously and
fuses them into one unified decision with a full natural language audit trail.

This is the key architectural contribution of PRISM for the paper:
    - Four signals never share state — they fail and recover independently
    - Weights adapt per frame based on signal freshness and confidence
    - Every decision is fully explainable — the audit trail is machine-readable
    - VLM staleness is tracked to the millisecond and penalised in the weights

Signal stack (all run in parallel, arbitration reads all four each frame):
    ┌─────────────────────────────────────────────────────────────┐
    │  WorldModel      — scene risk, actor states, occupancy      │  weight 0.15
    │  PredictiveEngine— LSTM intent + trajectory forecasts       │  weight 0.35
    │  SmartDecision   — physics + behavioural assessment         │  weight 0.35
    │  VLM             — semantic understanding (async, staleable)│  weight 0.15*
    └─────────────────────────────────────────────────────────────┘
    * VLM weight decays exponentially with staleness (tau=2s)

Decision fusion:
    1. Each signal maps to a numeric level 0–7 (CLEAR → EMERGENCY)
    2. Weighted average → nearest integer → decision string
    3. If signal std > 1.5 levels (disagreement), add safety buffer of +1
    4. Hard overrides always apply (e.g. actor in corridor < 3m → EMERGENCY)

Output:
    ArbitrationDecision with:
        - action string (8-level PRISM scale)
        - speed factor (0.0 – 1.0)
        - confidence (0.0 – 1.0)
        - audit_trail: full natural language explanation
        - signal_breakdown: per-signal contribution for visualisation
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from prism.utils.common import get_logger
from prism.predictive_engine.decision import DECISIONS

logger = get_logger("ArbitrationCore")

# ── Decision level map ────────────────────────────────────────────────────────

LEVEL_ORDER = ["CLEAR", "MONITOR", "EASE", "SLOW", "CAUTION", "YIELD", "STOP", "EMERGENCY"]
LEVEL_TO_INT = {d: i for i, d in enumerate(LEVEL_ORDER)}

# Human-readable speed descriptions for audit trail
SPEED_LABELS = {
    "CLEAR":     "full speed",
    "MONITOR":   "full speed, watching",
    "EASE":      "easing to 85%",
    "SLOW":      "slowing to 65%",
    "CAUTION":   "reducing to 40%",
    "YIELD":     "near stop at 15%",
    "STOP":      "full stop",
    "EMERGENCY": "emergency brake",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SignalSnapshot:
    """One signal's contribution to the arbitration decision."""
    name:        str          # "WorldModel" | "Predictive" | "SmartDecision" | "VLM"
    decision:    str          # raw decision from this signal
    level:       int          # numeric 0-7
    weight:      float        # effective weight after freshness penalty
    confidence:  float        # signal's own confidence estimate
    reason:      str          # one-line summary of why this signal decided what it did
    age_s:       float = 0.0  # seconds since last update (0 for real-time signals)


@dataclass
class ArbitrationDecision:
    """
    The unified output of the Arbitration Core.
    Every field is populated every frame — nothing is optional for downstream.
    """
    timestamp:    float
    frame_idx:    int

    # ── Core decision ──────────────────────────────────────────────────────
    action:       str         # one of LEVEL_ORDER
    speed_factor: float       # 0.0 – 1.0
    confidence:   float       # 0.0 – 1.0
    level:        int         # numeric 0-7

    # ── Signal breakdown ──────────────────────────────────────────────────
    signals:      List[SignalSnapshot] = field(default_factory=list)
    dominant_signal: str = ""    # which signal drove the decision
    signal_agreement: float = 0.0  # 0 = total disagreement, 1 = unanimous

    # ── Audit trail ───────────────────────────────────────────────────────
    audit_trail:  str = ""    # full natural language explanation
    short_reason: str = ""    # one-line summary for HUD overlay

    # ── Override flags ────────────────────────────────────────────────────
    hard_override: bool = False   # True if a hard safety rule fired
    override_reason: str = ""

    @property
    def is_conservative(self) -> bool:
        """True if a safety buffer was added due to signal disagreement."""
        return self.signal_agreement < 0.5

    @property
    def color(self) -> tuple:
        return DECISIONS[self.action]["color"]


# ── VLM signal cache ──────────────────────────────────────────────────────────

@dataclass
class VLMSignal:
    """Latest VLM output, held between async updates."""
    decision:        str   = "MONITOR"
    pedestrian_count: int  = 0
    vehicle_count:   int   = 0
    primary_risk:    str   = ""
    reasoning:       str   = ""
    updated_at:      float = 0.0   # wall-clock time of last VLM update

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.updated_at if self.updated_at > 0 else 999.0

    @property
    def is_fresh(self) -> bool:
        return self.age_s < 3.0


# ── Arbitration Core ──────────────────────────────────────────────────────────

class ArbitrationCore:
    """
    Reads WorldModel + PredictiveEngine + SmartDecision + VLM every frame.
    Produces a unified ArbitrationDecision with full natural language audit trail.

    Designed to be called every frame synchronously.
    VLM updates arrive asynchronously via update_vlm().
    """

    # Base signal weights (before freshness adjustment)
    BASE_WEIGHTS = {
        "SmartDecision": 0.35,
        "Predictive":    0.35,
        "WorldModel":    0.15,
        "VLM":           0.15,
    }

    # VLM weight decay: w(t) = w0 * exp(-t / tau)
    VLM_DECAY_TAU = 2.0     # seconds — at 2s stale, VLM weight halved
    VLM_WEIGHT_MIN = 0.02   # floor — never fully ignore VLM

    # Safety buffer: added to weighted level when signals disagree
    DISAGREEMENT_BUFFER = 1.0   # levels

    def __init__(self, cfg: dict = None):
        self.cfg          = cfg or {}
        self._frame_idx   = 0
        self._vlm         = VLMSignal()
        self._last_action = "CLEAR"
        self._history: List[int] = []   # recent decision levels for smoothing
        logger.info("Arbitration Core ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def update_vlm(self, vlm_output: dict):
        """
        Called asynchronously by the VLM Semantic Reasoner when a new
        inference result arrives. Thread-safe (GIL protected, single write).

        vlm_output keys: total_pedestrians, total_vehicles, decision,
                         primary_risk, reasoning
        """
        decision = vlm_output.get("decision", "MONITOR").upper()
        if decision not in LEVEL_TO_INT:
            decision = "MONITOR"

        self._vlm = VLMSignal(
            decision         = decision,
            pedestrian_count = int(vlm_output.get("total_pedestrians", 0)),
            vehicle_count    = int(vlm_output.get("total_vehicles", 0)),
            primary_risk     = str(vlm_output.get("primary_risk", "")),
            reasoning        = str(vlm_output.get("reasoning", "")),
            updated_at       = time.monotonic(),
        )
        logger.debug(
            f"VLM update: {decision}  peds={self._vlm.pedestrian_count}  "
            f"vehs={self._vlm.vehicle_count}  risk='{self._vlm.primary_risk[:60]}'"
        )

    def arbitrate(
        self,
        world_state,        # WorldState from WorldModel
        pred_state,         # PredictiveState from PredictiveEngine
        scene_assessment,   # SceneAssessment from SmartDecisionEngine
        timestamp: float = 0.0,
    ) -> ArbitrationDecision:
        """
        Main arbitration call — runs every frame.
        Fuses all four signals into one decision with audit trail.
        """
        self._frame_idx += 1

        # ── 1. Collect signal snapshots ────────────────────────────────────
        signals = self._collect_signals(world_state, pred_state, scene_assessment)

        # ── 2. Check hard overrides ────────────────────────────────────────
        override, override_reason = self._check_hard_overrides(
            world_state, pred_state, scene_assessment
        )

        # ── 3. Weighted fusion ─────────────────────────────────────────────
        if override is not None:
            fused_level  = LEVEL_TO_INT[override]
            agreement    = 1.0
            hard_override = True
        else:
            fused_level, agreement = self._fuse(signals)
            hard_override = False

        action = LEVEL_ORDER[int(np.clip(fused_level, 0, 7))]

        # ── 4. Confidence ──────────────────────────────────────────────────
        confidence = self._compute_confidence(
            signals, agreement, scene_assessment, world_state
        )

        # ── 5. Identify dominant signal ────────────────────────────────────
        dominant = max(signals, key=lambda s: s.weight * s.level)

        # ── 6. Audit trail ─────────────────────────────────────────────────
        audit, short = self._build_audit_trail(
            signals, action, fused_level, agreement,
            confidence, hard_override, override_reason, dominant
        )

        # ── 7. History ─────────────────────────────────────────────────────
        self._history.append(fused_level)
        if len(self._history) > 20:
            self._history.pop(0)
        self._last_action = action

        return ArbitrationDecision(
            timestamp        = timestamp or time.monotonic(),
            frame_idx        = self._frame_idx,
            action           = action,
            speed_factor     = DECISIONS[action]["speed"],
            confidence       = confidence,
            level            = fused_level,
            signals          = signals,
            dominant_signal  = dominant.name,
            signal_agreement = agreement,
            audit_trail      = audit,
            short_reason     = short,
            hard_override    = hard_override,
            override_reason  = override_reason,
        )

    # ── Signal collection ─────────────────────────────────────────────────────

    def _collect_signals(self, world_state, pred_state, scene_assessment) -> List[SignalSnapshot]:
        signals = []

        # ── SmartDecision ──────────────────────────────────────────────────
        sd_level = LEVEL_TO_INT.get(scene_assessment.decision, 1)
        sd_conf  = float(scene_assessment.confidence)
        signals.append(SignalSnapshot(
            name       = "SmartDecision",
            decision   = scene_assessment.decision,
            level      = sd_level,
            weight     = self.BASE_WEIGHTS["SmartDecision"] * sd_conf,
            confidence = sd_conf,
            reason     = scene_assessment.primary_reason,
        ))

        # ── Predictive Engine ──────────────────────────────────────────────
        pred_decision = pred_state.recommended_action.upper()
        # Map engine's 4-level output to our 8-level scale
        pred_map = {"CONTINUE": "MONITOR", "SLOW": "SLOW",
                    "BRAKE": "CAUTION", "STOP": "STOP"}
        pred_decision = pred_map.get(pred_decision, pred_decision)
        if pred_decision not in LEVEL_TO_INT:
            pred_decision = "MONITOR"
        pred_level = LEVEL_TO_INT[pred_decision]

        # Build a reason from the most dangerous predicted actor
        if pred_state.actors_on_collision_course:
            pred_reason = (
                f"{len(pred_state.actors_on_collision_course)} actor(s) on collision course"
            )
        elif pred_state.predictions:
            worst = max(pred_state.predictions, key=lambda p: p.predicted_risk)
            pred_reason = (
                f"{worst.class_name} #{worst.track_id} → "
                f"{worst.most_likely_maneuver} (risk={worst.predicted_risk:.2f})"
            )
        else:
            pred_reason = "No confirmed actors"

        pred_conf = float(np.clip(1.0 - pred_state.divergence_score, 0.3, 1.0))
        signals.append(SignalSnapshot(
            name       = "Predictive",
            decision   = pred_decision,
            level      = pred_level,
            weight     = self.BASE_WEIGHTS["Predictive"] * pred_conf,
            confidence = pred_conf,
            reason     = pred_reason,
        ))

        # ── WorldModel ─────────────────────────────────────────────────────
        risk_to_level = [0, 2, 4, 6]   # GREEN=CLEAR, YELLOW=EASE, ORANGE=CAUTION, RED=STOP
        wm_level  = risk_to_level[world_state.risk_level]
        wm_action = LEVEL_ORDER[wm_level]
        wm_conf   = float(world_state.perception_confidence)
        wm_reason = (
            f"risk={world_state.risk_score:.2f} ({world_state.level_name}), "
            f"trend={world_state.risk_trend}, "
            f"{world_state.actor_count} actors"
        )
        signals.append(SignalSnapshot(
            name       = "WorldModel",
            decision   = wm_action,
            level      = wm_level,
            weight     = self.BASE_WEIGHTS["WorldModel"] * wm_conf,
            confidence = wm_conf,
            reason     = wm_reason,
        ))

        # ── VLM ────────────────────────────────────────────────────────────
        vlm_age  = self._vlm.age_s
        vlm_w    = max(
            self.BASE_WEIGHTS["VLM"] * np.exp(-vlm_age / self.VLM_DECAY_TAU),
            self.VLM_WEIGHT_MIN
        )
        vlm_level = LEVEL_TO_INT.get(self._vlm.decision, 1)
        if self._vlm.updated_at > 0:
            vlm_reason = (
                f"{self._vlm.decision} — peds={self._vlm.pedestrian_count} "
                f"vehs={self._vlm.vehicle_count}"
                + (f" — '{self._vlm.primary_risk[:50]}'" if self._vlm.primary_risk else "")
                + f" [age={vlm_age:.1f}s]"
            )
            vlm_conf = float(np.exp(-vlm_age / (self.VLM_DECAY_TAU * 2)))
        else:
            vlm_reason = "No VLM update received yet"
            vlm_conf   = 0.0
        signals.append(SignalSnapshot(
            name       = "VLM",
            decision   = self._vlm.decision,
            level      = vlm_level,
            weight     = vlm_w,
            confidence = vlm_conf,
            reason     = vlm_reason,
            age_s      = vlm_age,
        ))

        return signals

    # ── Hard overrides ────────────────────────────────────────────────────────

    def _check_hard_overrides(
        self, world_state, pred_state, scene_assessment
    ):
        """
        Safety-critical rules that bypass weighted fusion entirely.
        Returns (decision_str, reason) or (None, "").
        """
        # Actor literally inside ego corridor < 3m
        for aa in scene_assessment.actor_assessments:
            if aa.distance_m < 3.0 and aa.is_in_corridor:
                return "EMERGENCY", f"{aa.class_name} at {aa.distance_m:.1f}m in corridor"

        # Any predicted TTC < 1s
        for p in pred_state.predictions:
            if p.ttc is not None and p.ttc < 1.0:
                return "EMERGENCY", (
                    f"TTC={p.ttc:.1f}s for {p.class_name} #{p.track_id}"
                )

        # SmartDecision already called EMERGENCY
        if scene_assessment.decision == "EMERGENCY":
            return "EMERGENCY", scene_assessment.primary_reason

        # Risk RED + worsening + close actor = don't wait for consensus
        if (world_state.risk_level == 3 and
                world_state.risk_trend == "increasing" and
                scene_assessment.closest_threat_m < 8.0):
            return "STOP", (
                f"Risk RED+increasing, closest threat {scene_assessment.closest_threat_m:.1f}m"
            )

        return None, ""

    # ── Weighted fusion ───────────────────────────────────────────────────────

    def _fuse(self, signals: List[SignalSnapshot]):
        """
        Weighted average of signal levels → fused level.
        Returns (fused_level int, agreement float 0-1).
        """
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 1, 0.5

        weighted_level = sum(s.weight * s.level for s in signals) / total_weight

        # Agreement: 1 when all signals identical, 0 when maximally spread
        levels = np.array([s.level for s in signals], dtype=float)
        std    = float(np.std(levels))
        agreement = float(np.clip(1.0 - std / 3.5, 0.0, 1.0))

        # Safety buffer — add headroom when signals disagree significantly
        if std > 1.5:
            weighted_level += self.DISAGREEMENT_BUFFER

        fused = int(np.round(np.clip(weighted_level, 0, 7)))
        return fused, agreement

    # ── Confidence ────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        signals:          List[SignalSnapshot],
        agreement:        float,
        scene_assessment,
        world_state,
    ) -> float:
        """
        Confidence = f(signal agreement, perception quality, VLM freshness).
        """
        # Base: mean signal confidence weighted by their weights
        total_w  = sum(s.weight for s in signals)
        mean_conf = sum(s.weight * s.confidence for s in signals) / max(total_w, 1e-6)

        # Agreement bonus/penalty
        conf = mean_conf * (0.6 + 0.4 * agreement)

        # Perception quality
        conf *= world_state.perception_confidence

        # VLM freshness bonus (small — 5% max)
        vlm_sig = next((s for s in signals if s.name == "VLM"), None)
        if vlm_sig and vlm_sig.age_s < 1.0:
            conf = min(conf + 0.05, 1.0)

        return float(np.clip(conf, 0.05, 0.99))

    # ── Audit trail ───────────────────────────────────────────────────────────

    def _build_audit_trail(
        self,
        signals:          List[SignalSnapshot],
        action:           str,
        fused_level:      int,
        agreement:        float,
        confidence:       float,
        hard_override:    bool,
        override_reason:  str,
        dominant:         SignalSnapshot,
    ):
        """
        Builds the full natural language audit trail and a one-line summary.
        The audit trail is the key paper contribution — every decision is
        fully explainable from signal inputs.
        """
        lines = []
        total_w = sum(s.weight for s in signals)

        for s in signals:
            pct      = 100.0 * s.weight / max(total_w, 1e-6)
            age_note = f", age={s.age_s:.1f}s" if s.age_s > 0.1 else ""
            conf_note = f", conf={s.confidence:.2f}"
            lines.append(
                f"  [{s.name:<14} {pct:4.1f}%{age_note}{conf_note}]  "
                f"{s.decision:<10}  {s.reason}"
            )

        # Agreement note
        if agreement >= 0.85:
            agree_note = "Signals unanimous."
        elif agreement >= 0.6:
            agree_note = "Signals broadly agree."
        elif agreement >= 0.4:
            agree_note = f"Signals disagree (std={1.0 - agreement:.2f}) — safety buffer applied."
        else:
            agree_note = f"Signals strongly disagree — safety buffer applied, confidence penalised."

        # Override note
        if hard_override:
            override_note = f"\n  ⚡ HARD OVERRIDE: {override_reason}"
        else:
            override_note = ""

        # Trend
        if len(self._history) >= 4:
            recent = np.mean(self._history[-3:])
            older  = np.mean(self._history[-6:-3]) if len(self._history) >= 6 else recent
            delta  = recent - older
            if   delta >  0.5:  trend_note = "Situation WORSENING."
            elif delta < -0.5:  trend_note = "Situation IMPROVING."
            else:               trend_note = "Situation stable."
        else:
            trend_note = ""

        signal_block = "\n".join(lines)
        audit = (
            f"=== PRISM Arbitration — frame {self._frame_idx} ===\n"
            f"{signal_block}\n"
            f"  {agree_note} {trend_note}"
            f"{override_note}\n"
            f"  Dominant signal: {dominant.name}  "
            f"Final: {action} ({SPEED_LABELS[action]})  "
            f"Confidence: {confidence*100:.0f}%"
        )

        short = (
            f"{dominant.name}→{action} | conf={confidence*100:.0f}% | "
            f"{dominant.reason[:60]}"
        )

        return audit, short

    def reset(self):
        self._frame_idx = 0
        self._vlm       = VLMSignal()
        self._history   = []
        self._last_action = "CLEAR"
