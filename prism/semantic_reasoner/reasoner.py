"""
PRISM — VLM Semantic Reasoner
==============================
The "slow brain" of PRISM.

Runs Qwen2.5-VL asynchronously — never blocks the fast loop.
Triggered by prediction divergence, not a fixed timer.

This is the core novel contribution:
    Traditional: VLM every N ms (wastes compute, misses moments)
    PRISM:       VLM when prediction ≠ reality (fires when it matters)

The fast loop (Predictive Engine) runs at 5-7fps continuously.
The VLM fires 1-3x per second on divergence events.
Between VLM calls, the fast loop uses the last known semantic state.

Output structure:
{
    "scene_context": "busy urban intersection, night, wet road",
    "actors": [
        {
            "track_id": 3,
            "intent": "braking",
            "confidence": 0.82,
            "reasoning": "brake lights visible, speed reducing"
        }
    ],
    "risk_flags": ["pedestrian crossing", "obscured traffic light"],
    "pedestrian_status": "caution",
    "recommended_caution": "high",
    "scene_summary": "one sentence describing the critical situation"
}
"""

import time
import threading
import queue
import json
import re
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from prism.utils.common import get_logger, get_device

logger = get_logger("VLMReasoner")


# ── VLM Output Schema ─────────────────────────────────────────────────────────

@dataclass
class VLMOutput:
    """Structured output from one VLM inference call."""
    timestamp: float
    trigger_reason: str              # what caused this VLM call
    divergence_score: float          # divergence that triggered it

    scene_context: str = ""
    scene_summary: str = ""
    risk_flags: list = field(default_factory=list)
    actor_intents: dict = field(default_factory=dict)  # track_id → intent str
    pedestrian_status: str = "clear"
    recommended_caution: str = "normal"  # normal | elevated | high | critical
    raw_response: str = ""
    inference_time_ms: float = 0.0
    success: bool = False

    # Caution → PRISM decision level mapping
    _CAUTION_TO_DECISION = {
        "normal":   "MONITOR",
        "elevated": "EASE",
        "high":     "CAUTION",
        "critical": "STOP",
    }

    @property
    def caution_level(self) -> int:
        mapping = {"normal": 0, "elevated": 1, "high": 2, "critical": 3}
        return mapping.get(self.recommended_caution, 0)

    def to_arb_dict(self) -> dict:
        """
        Convert VLMOutput to the dict format expected by ArbitrationCore.update_vlm().
        Maps recommended_caution → PRISM 8-level decision string.
        """
        return {
            "decision":          self._CAUTION_TO_DECISION.get(self.recommended_caution, "MONITOR"),
            "total_pedestrians": sum(1 for v in self.actor_intents.values() if "cross" in v or "ped" in v),
            "total_vehicles":    sum(1 for v in self.actor_intents.values() if v not in ("crossing", "unknown")),
            "primary_risk":      self.risk_flags[0] if self.risk_flags else "",
            "reasoning":         self.scene_summary,
        }


# ── Divergence Monitor ────────────────────────────────────────────────────────

class DivergenceMonitor:
    """
    Monitors prediction accuracy every frame.
    Fires a trigger event when reality diverges from prediction.

    This is the gating mechanism — the VLM only runs when this fires.

    Trigger conditions:
        1. Prediction divergence > threshold (main trigger)
        2. New actor entered scene
        3. Actor disappeared unexpectedly
        4. Risk level jumped (e.g. GREEN → RED in one frame)
        5. Time-based fallback (max N seconds without VLM call)
    """

    def __init__(
        self,
        divergence_threshold: float = 0.40,
        max_silence_s: float = 8.0,      # force VLM after this many seconds
        cooldown_s: float = 2.0,         # min time between VLM calls — relaxed for M4
    ):
        self.divergence_threshold = divergence_threshold
        self.max_silence_s = max_silence_s
        self.cooldown_s = cooldown_s

        self._last_trigger_time = 0.0
        self._last_actor_count = 0
        self._last_risk_level = 0
        self._trigger_count = 0
        self._skipped_count = 0

    def check(self, pred_state, world_state) -> Optional[str]:
        """
        Check if VLM should be triggered this frame.
        Returns trigger reason string, or None if no trigger.
        """
        now = time.time()

        # Cooldown — don't spam VLM
        if now - self._last_trigger_time < self.cooldown_s:
            self._skipped_count += 1
            return None

        reason = None

        # Trigger 1 — prediction divergence
        if pred_state.divergence_score > self.divergence_threshold:
            reason = f"divergence={pred_state.divergence_score:.2f}"

        # Trigger 2 — new actor entered scene
        elif world_state.actor_count > self._last_actor_count:
            new_count = world_state.actor_count - self._last_actor_count
            reason = f"new_actor+{new_count}"

        # Trigger 3 — risk level jumped by 2+ levels
        elif world_state.risk_level >= self._last_risk_level + 2:
            reason = f"risk_jump:{self._last_risk_level}→{world_state.risk_level}"

        # Trigger 4 — actor on collision course appeared
        elif len(pred_state.actors_on_collision_course) > 0:
            ttcs = [p.ttc for p in pred_state.predictions
                    if p.ttc is not None and p.ttc < 3.0]
            if ttcs:
                reason = f"collision_risk:ttc={min(ttcs):.1f}s"

        # Trigger 5 — max silence fallback
        elif now - self._last_trigger_time > self.max_silence_s:
            reason = "periodic_update"

        # Update state
        self._last_actor_count = world_state.actor_count
        self._last_risk_level = world_state.risk_level

        if reason:
            self._last_trigger_time = now
            self._trigger_count += 1
            logger.debug(f"VLM triggered: {reason}")

        return reason

    @property
    def stats(self) -> dict:
        return {
            "triggers": self._trigger_count,
            "skipped": self._skipped_count,
            "efficiency": self._skipped_count / max(1, self._trigger_count + self._skipped_count)
        }


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds structured prompts for the VLM.
    Injects world model state so VLM has full context.
    """

    @staticmethod
    def build(world_state, pred_state, trigger_reason: str) -> str:
        # Summarise actors for context
        actor_lines = []
        for actor in world_state.actors[:6]:
            depth_str = f"{actor.depth*50:.1f}m" if actor.depth else "?"
            move_str = "moving" if actor.is_moving else "stationary"
            actor_lines.append(
                f"- #{actor.track_id} {actor.class_name} at ~{depth_str}, {move_str}"
            )
        actors_str = "\n".join(actor_lines) if actor_lines else "- No confirmed actors"

        # Collision risks
        collision_str = ""
        if pred_state.actors_on_collision_course:
            ttc_info = []
            for p in pred_state.predictions:
                if p.track_id in pred_state.actors_on_collision_course and p.ttc:
                    ttc_info.append(f"#{p.track_id} ({p.class_name}) TTC={p.ttc:.1f}s")
            collision_str = f"\nCOLLISION RISK: {', '.join(ttc_info)}"

        prompt = f"""You are an autonomous vehicle perception system analyzing a driving scene.

CURRENT WORLD STATE:
- Risk level: {world_state.level_name} (score: {world_state.risk_score:.2f})
- Tracked actors:
{actors_str}
- Prediction divergence: {pred_state.divergence_score:.2f}
- Trigger reason: {trigger_reason}{collision_str}

Analyze this image carefully and respond with ONLY a JSON object in this exact format:
{{
  "scene_context": "brief description of road type, conditions, time of day",
  "scene_summary": "one sentence describing the most critical aspect right now",
  "risk_flags": ["list", "of", "specific", "risks"],
  "actors": [
    {{
      "track_id": <number or null if unknown>,
      "class": "car/truck/person/bicycle/etc",
      "intent": "constant_velocity/braking/turning_left/turning_right/stopping/crossing/unknown",
      "confidence": <0.0-1.0>,
      "reasoning": "brief reason for intent estimate"
    }}
  ],
  "pedestrian_status": "clear/caution/danger",
  "recommended_caution": "normal/elevated/high/critical"
}}

Focus on: pedestrian intent, vehicle trajectories, any unexpected behaviors, road conditions."""

        return prompt


# ── VLM Model ─────────────────────────────────────────────────────────────────

class VLMModel:
    """
    Qwen2.5-VL-7B-Instruct wrapper.
    Loads from a local model directory (downloaded by scripts/download_vlm.py).
    Falls back gracefully if model unavailable — pipeline continues in mock mode.

    Local-first loading:
        If model_dir exists on disk, it is used directly (no internet needed).
        If model_dir is absent, falls back to downloading from HuggingFace.
        On Jetson Thor, always prefer local dir — no Wi-Fi during inference.
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        device:    str  = "cuda",
        enabled:   bool = False,
        model_dir: str  = "",
    ):
        self.device    = device
        self.model_dir = model_dir
        self.model     = None
        self.processor = None
        self._available = False
        self._enabled   = enabled
        self._load()

    def _load(self):
        if not self._enabled:
            logger.info("VLM disabled — running in mock mode (set vlm.enabled=true to enable)")
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            # Prefer local model dir; fall back to HuggingFace hub
            from pathlib import Path as _Path
            local = _Path(self.model_dir).expanduser() if self.model_dir else None
            source = str(local) if (local and local.exists()) else self.MODEL_ID

            logger.info(f"Loading Qwen2.5-VL-7B from: {source}")

            self.processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
            )

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            self._available = True
            logger.info(f"Qwen2.5-VL-7B ready on {self.device}")

        except ImportError as e:
            logger.warning(f"transformers missing or too old for Qwen2.5-VL: {e}")
            logger.warning(
                "Running in MOCK mode.\n"
                "Install: /home/koushik-test/cad_pipeline_env/bin/pip install "
                "transformers accelerate qwen-vl-utils Pillow"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"VLM load failed: {e}")
            logger.warning("Running in MOCK mode")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def infer(self, image: np.ndarray, prompt: str) -> tuple:
        """
        Run VLM inference.
        Returns (response_text, inference_time_ms)
        """
        if not self._available:
            return self._mock_response(), 0.0

        try:
            import torch
            from PIL import Image as PILImage

            # Convert BGR → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Qwen2.5-VL processor accepts images directly.
            # Use BatchFeature.to() — handles pixel_values as list-of-tensors
            # correctly (plain dict comprehension breaks for list values).
            batch  = self.processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
            )
            inputs = batch.to(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            inference_ms = (time.perf_counter() - t0) * 1000

            input_len = inputs["input_ids"].shape[1]
            response  = self.processor.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            return response, inference_ms

        except Exception as e:
            logger.warning(f"VLM inference failed: {e}")
            return self._mock_response(), 0.0

    def _mock_response(self) -> str:
        """
        Mock response for when VLM is unavailable.
        Returns realistic-looking output for testing the pipeline.
        """
        import random
        contexts = [
            "urban intersection, daytime, clear conditions",
            "busy crosswalk, multiple pedestrians present",
            "construction zone, reduced lanes",
            "night driving, wet road surface",
        ]
        flags = random.sample([
            "vehicles decelerating ahead",
            "pedestrian near crosswalk",
            "lane merge ahead",
            "traffic light visible",
            "parked vehicles reducing lane width"
        ], k=random.randint(1, 3))

        return json.dumps({
            "scene_context": random.choice(contexts),
            "scene_summary": "Scene appears nominal with standard urban traffic conditions.",
            "risk_flags": flags,
            "actors": [],
            "pedestrian_status": random.choice(["clear", "clear", "caution"]),
            "recommended_caution": random.choice(["normal", "normal", "elevated"])
        })


# ── Response Parser ───────────────────────────────────────────────────────────

class ResponseParser:
    """Parses VLM text output into structured VLMOutput."""

    @staticmethod
    def parse(
        raw: str,
        timestamp: float,
        trigger_reason: str,
        divergence: float,
        inference_ms: float
    ) -> VLMOutput:
        output = VLMOutput(
            timestamp=timestamp,
            trigger_reason=trigger_reason,
            divergence_score=divergence,
            raw_response=raw,
            inference_time_ms=inference_ms,
        )

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in VLM response")
                return output

            data = json.loads(json_match.group())

            output.scene_context      = data.get("scene_context", "")
            output.scene_summary      = data.get("scene_summary", "")
            output.risk_flags         = data.get("risk_flags", [])
            output.pedestrian_status  = data.get("pedestrian_status", "clear")
            output.recommended_caution = data.get("recommended_caution", "normal")

            # Parse actor intents
            for actor_data in data.get("actors", []):
                tid = actor_data.get("track_id")
                intent = actor_data.get("intent", "unknown")
                if tid is not None:
                    output.actor_intents[int(tid)] = intent

            output.success = True

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
        except Exception as e:
            logger.warning(f"Response parse error: {e}")

        return output


# ── Async VLM Worker ──────────────────────────────────────────────────────────

class AsyncVLMWorker:
    """
    Runs VLM inference in a background thread.
    The fast loop NEVER waits for this — it posts a job and moves on.
    Results are picked up on the next frame via get_latest_output().

    This is the key to eliminating latency:
        Fast loop: post job → continue immediately
        Background: VLM runs → result stored
        Fast loop: picks up result when convenient
    """

    def __init__(self, vlm_model: VLMModel):
        self.model = vlm_model
        self._job_queue = queue.Queue(maxsize=1)  # only keep latest job
        self._result_queue = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._total_calls = 0
        self._total_ms = 0.0
        logger.info("Async VLM worker started")

    def post_job(
        self,
        image: np.ndarray,
        prompt: str,
        trigger_reason: str,
        divergence: float
    ):
        """Post a VLM job. Non-blocking — drops old job if queue full."""
        try:
            # Drop old pending job if any — only care about latest
            try:
                self._job_queue.get_nowait()
            except queue.Empty:
                pass
            self._job_queue.put_nowait({
                "image": image.copy(),
                "prompt": prompt,
                "trigger_reason": trigger_reason,
                "divergence": divergence,
                "posted_at": time.time()
            })
        except queue.Full:
            pass  # already have a pending job

    def get_latest_output(self) -> Optional[VLMOutput]:
        """Get the latest VLM result if available. Non-blocking."""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker(self):
        """Background thread — processes VLM jobs continuously."""
        while True:
            try:
                job = self._job_queue.get(timeout=1.0)
                t_start = time.time()
                logger.info(f"VLM worker: picked up job trigger={job['trigger_reason']} (queued {(t_start - job['posted_at'])*1000:.0f}ms ago)")

                response, inference_ms = self.model.infer(
                    job["image"], job["prompt"]
                )

                output = ResponseParser.parse(
                    raw=response,
                    timestamp=time.time(),
                    trigger_reason=job["trigger_reason"],
                    divergence=job["divergence"],
                    inference_ms=inference_ms
                )

                self._total_calls += 1
                self._total_ms += inference_ms

                # Store result — drop oldest if queue full
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._result_queue.put_nowait(output)

                logger.info(
                    f"VLM complete | {inference_ms:.0f}ms | "
                    f"trigger={output.trigger_reason} | "
                    f"caution={output.recommended_caution} | "
                    f"flags={output.risk_flags[:2]}"
                )

            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                logger.error(f"VLM worker error: {e}\n{traceback.format_exc()}")

    @property
    def avg_inference_ms(self) -> float:
        return self._total_ms / max(1, self._total_calls)

    @property
    def total_calls(self) -> int:
        return self._total_calls


# ── Main Semantic Reasoner ────────────────────────────────────────────────────

class SemanticReasoner:
    """
    Top-level VLM Semantic Reasoner.
    Called every frame — internally decides whether to actually run VLM.

    Usage:
        reasoner = SemanticReasoner(cfg)
        for frame in scene:
            sensory_frame = core.process(image)
            world_state = world.update(sensory_frame)
            pred_state = predictor.update(world_state)

            # This is non-blocking — returns immediately
            vlm_output = reasoner.update(image, world_state, pred_state)

            # vlm_output may be None (no new VLM result this frame)
            # or VLMOutput (fresh result from background thread)
    """

    def __init__(self, cfg: dict):
        vlm_cfg = cfg.get("vlm", {})
        device  = vlm_cfg.get("device", "cuda")

        self.monitor = DivergenceMonitor(
            divergence_threshold=vlm_cfg.get("divergence_threshold", 0.35),
            max_silence_s=vlm_cfg.get("max_silence_s", 4.0),
            cooldown_s=vlm_cfg.get("cooldown_s", 0.4),
        )
        self.prompt_builder = PromptBuilder()
        vlm_enabled = vlm_cfg.get("enabled", False)
        VLMModel.MODEL_ID = vlm_cfg.get("model_id", VLMModel.MODEL_ID)
        model_dir   = vlm_cfg.get(
            "model_dir",
            "/home/koushik-test/prism_data/models/qwen2_5_vl_7b",
        )
        self.vlm = VLMModel(device=device, enabled=vlm_enabled, model_dir=model_dir)
        self.worker = AsyncVLMWorker(self.vlm)

        # Latest known semantic state — persists between VLM calls
        self.latest_output: Optional[VLMOutput] = None
        self._frame_count = 0

        logger.info(f"Semantic Reasoner ready | VLM available: {self.vlm.available}")

    def update(
        self,
        image: np.ndarray,
        world_state,
        pred_state
    ) -> Optional[VLMOutput]:
        """
        Called every frame. Non-blocking.

        1. Check if VLM should be triggered
        2. If yes, post job to background worker (returns immediately)
        3. Check if background worker has a new result
        4. Return new result if available, else None
        """
        self._frame_count += 1

        # Check trigger
        trigger_reason = self.monitor.check(pred_state, world_state)
        if trigger_reason:
            logger.info(f"VLM trigger fired: {trigger_reason} (frame {self._frame_count})")
            prompt = self.prompt_builder.build(world_state, pred_state, trigger_reason)
            self.worker.post_job(
                image=image,
                prompt=prompt,
                trigger_reason=trigger_reason,
                divergence=pred_state.divergence_score
            )

        # Pick up any completed VLM result
        new_output = self.worker.get_latest_output()
        if new_output:
            self.latest_output = new_output
            return new_output

        return None

    def get_current_semantic_state(self) -> Optional[VLMOutput]:
        """Returns the most recent VLM output — may be several frames old."""
        return self.latest_output

    def get_actor_intent(self, track_id: int) -> Optional[str]:
        """Get VLM-provided intent for a specific actor."""
        if self.latest_output:
            return self.latest_output.actor_intents.get(track_id)
        return None

    @property
    def stats(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "vlm_calls": self.worker.total_calls,
            "trigger_rate": self.worker.total_calls / max(1, self._frame_count),
            "avg_inference_ms": self.worker.avg_inference_ms,
            **self.monitor.stats,
        }
