"""
PRISM — VLM Scene Monitor (Qwen2.5-VL-7B)
==========================================
Wraps Qwen2.5-VL-7B-Instruct as the scene-change detector inside the
Predictive Engine.  Replaces the mock VLM used during benchmarking with
real inference.

Design:
    - Event-driven: called only when the trigger condition fires (12.5% of
      frames on average — one key frame per ~0.5s at 12fps).
    - Outputs a structured SceneDescription with hazard flags and a
      natural-language caption for logging.
    - Re-uses a single model instance (loaded once at startup) across all
      calls — no per-frame model loading.

Trigger policy (mirrors vlm_trigger.py):
    VLM is invoked when ANY of:
      1. Optical-flow magnitude exceeds threshold (sudden scene change)
      2. Time since last VLM call exceeds max_gap_s (keepalive)
      3. External caller requests immediate evaluation (e.g., new actor appears)

Inference latency:
    Qwen2.5-VL-7B fp16 on Jetson Thor: ~200-400ms per call.
    Because calls are rare (12.5% of frames), average per-frame overhead
    is < 50ms — well within the 83ms frame budget at 12fps.

Usage:
    from prism.predictive_engine.vlm_scene_monitor import VLMSceneMonitor

    monitor = VLMSceneMonitor(model_dir="/home/koushik-test/prism_data/models/qwen2_5_vl_7b")
    desc = monitor.analyze(frame_bgr)  # numpy (H, W, 3) BGR or RGB
    if desc.has_hazard:
        ...
"""

from __future__ import annotations

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger("VLMSceneMonitor")


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class SceneDescription:
    """
    Structured output from one VLM inference call.
    """
    caption:          str            = ""
    has_hazard:       bool           = False
    hazard_types:     List[str]      = field(default_factory=list)
    # Flags derived from VLM output text
    pedestrian_cross: bool           = False
    emergency_vehicle:bool           = False
    construction_zone:bool           = False
    traffic_signal:   str            = "unknown"   # "red" | "yellow" | "green" | "unknown"
    confidence:       float          = 0.0
    latency_ms:       float          = 0.0
    timestamp:        float          = field(default_factory=time.time)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SCENE_PROMPT = (
    "You are an automotive perception system. Analyze this front-facing camera "
    "frame from a moving vehicle and respond in this EXACT format:\n\n"
    "CAPTION: <one sentence describing the scene>\n"
    "HAZARD: <YES or NO>\n"
    "HAZARD_TYPES: <comma-separated list, or NONE>\n"
    "PEDESTRIAN_CROSSING: <YES or NO>\n"
    "EMERGENCY_VEHICLE: <YES or NO>\n"
    "CONSTRUCTION: <YES or NO>\n"
    "TRAFFIC_LIGHT: <RED, YELLOW, GREEN, or NONE>\n\n"
    "Hazard types include: pedestrian_in_path, cyclist_in_path, vehicle_cutting_in, "
    "sudden_braking, debris, wet_road, poor_visibility, red_light_running.\n"
    "Be concise and accurate. No extra text."
)


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse_vlm_response(text: str) -> SceneDescription:
    """
    Parse the structured VLM response into a SceneDescription.
    Tolerant of minor formatting variations.
    """
    desc = SceneDescription()
    lines = {
        k.strip().upper(): v.strip()
        for line in text.strip().splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }

    desc.caption = lines.get("CAPTION", text[:120])

    hazard_str = lines.get("HAZARD", "NO").upper()
    desc.has_hazard = hazard_str.startswith("Y")

    raw_types = lines.get("HAZARD_TYPES", "NONE")
    if raw_types.upper() != "NONE":
        desc.hazard_types = [t.strip().lower() for t in raw_types.split(",") if t.strip()]

    desc.pedestrian_cross  = lines.get("PEDESTRIAN_CROSSING", "NO").upper().startswith("Y")
    desc.emergency_vehicle = lines.get("EMERGENCY_VEHICLE",   "NO").upper().startswith("Y")
    desc.construction_zone = lines.get("CONSTRUCTION",        "NO").upper().startswith("Y")

    tl = lines.get("TRAFFIC_LIGHT", "NONE").upper()
    if tl in ("RED", "YELLOW", "GREEN"):
        desc.traffic_signal = tl.lower()
    else:
        desc.traffic_signal = "unknown"

    # Confidence heuristic: penalise if required keys are missing
    n_keys = sum(1 for k in ("CAPTION", "HAZARD", "HAZARD_TYPES") if k in lines)
    desc.confidence = n_keys / 3.0

    return desc


# ── Main class ────────────────────────────────────────────────────────────────

class VLMSceneMonitor:
    """
    Loads Qwen2.5-VL-7B once; call analyze(frame) as needed.

    Args:
        model_dir   : Path to saved model (from download_vlm.py).
        device      : "cuda" (Jetson) or "cpu" (debug).
        max_new_tokens: Max tokens in the VLM response (keep low for speed).
    """

    def __init__(
        self,
        model_dir:      str  = "/home/koushik-test/prism_data/models/qwen2_5_vl_7b",
        device:         str  = "cuda",
        max_new_tokens: int  = 128,
    ):
        self.model_dir      = Path(model_dir)
        self.device         = device
        self.max_new_tokens = max_new_tokens

        self._model     = None
        self._processor = None
        self._loaded    = False

        self._load_model()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_model(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"VLM model directory not found: {self.model_dir}\n"
                f"Run: python scripts/download_vlm.py"
            )

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            t0 = time.time()
            logger.info(f"Loading Qwen2.5-VL-7B from {self.model_dir} ...")

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
            )

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._model.eval()
            self._loaded = True

            elapsed = time.time() - t0
            logger.info(f"Qwen2.5-VL-7B loaded in {elapsed:.1f}s on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                f"Run: /home/koushik-test/cad_pipeline_env/bin/pip install "
                f"transformers accelerate qwen-vl-utils Pillow"
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Run VLM on a single camera frame.

        Args:
            frame: (H, W, 3) numpy array, BGR or RGB uint8.
        Returns:
            SceneDescription with parsed hazard flags and caption.
        """
        if not self._loaded:
            logger.warning("VLM not loaded — returning empty SceneDescription")
            return SceneDescription()

        try:
            return self._run_inference(frame)
        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return SceneDescription(caption=f"VLM error: {e}", confidence=0.0)

    def _run_inference(self, frame: np.ndarray) -> SceneDescription:
        import torch
        from PIL import Image

        t0 = time.time()

        # Convert BGR → RGB if needed (OpenCV default is BGR)
        if frame.shape[2] == 3:
            rgb = frame[:, :, ::-1].copy()
        else:
            rgb = frame

        pil_image = Image.fromarray(rgb.astype(np.uint8))

        # Build Qwen2.5-VL chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": SCENE_PROMPT},
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode — strip the input tokens
        input_len     = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        latency_ms = (time.time() - t0) * 1000.0
        logger.debug(f"VLM latency: {latency_ms:.0f}ms")
        logger.debug(f"VLM response:\n{response_text}")

        desc             = _parse_vlm_response(response_text)
        desc.latency_ms  = latency_ms
        desc.timestamp   = t0

        return desc

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"VLMSceneMonitor(model_dir={self.model_dir}, device={self.device}, {status})"
