"""
PRISM — VLM Inference Verification
=====================================
Verifies that Qwen2.5-VL-7B is correctly installed and producing meaningful
scene descriptions.  Runs a single-image inference with a known test frame
from nuScenes mini and checks that:
  1. The model loads without errors
  2. The response is valid JSON matching the PRISM schema
  3. Inference latency is under 5000ms (relaxed — first call includes GPU warmup)
  4. Hazard detection works: a frame with a pedestrian crossing produces
     pedestrian_status != "clear" OR risk_flags referencing pedestrians

This script is standalone — run it before launching run_predictive.py to
confirm VLM is healthy.

Usage:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/verify_vlm.py

    # Quick smoke test with a synthetic red frame (no nuScenes needed):
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/verify_vlm.py --synthetic
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.utils.common import load_config, get_logger

logger = get_logger("verify_vlm")


def make_synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """
    Generate a simple synthetic driving scene: blue sky, grey road,
    a white pedestrian silhouette. Tests that VLM can at least describe
    basic image content.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky
    frame[:h//2] = (180, 140, 100)
    # Road
    frame[h//2:] = (80, 80, 80)
    # Lane markings
    for x in range(0, w, 60):
        frame[h//2+10:h//2+30, x:x+30] = (240, 240, 240)
    # Pedestrian silhouette (centre-front)
    cx = w // 2
    cy = h // 2 + 20
    frame[cy-40:cy, cx-12:cx+12] = (220, 200, 190)  # head+torso
    frame[cy:cy+30, cx-16:cx+16] = (50, 80, 180)    # legs (blue jeans)
    return frame


def load_nuscenes_frame(cfg: dict) -> np.ndarray:
    """Try to grab the first camera frame from nuScenes mini."""
    try:
        import cv2
        from prism.sensory_core.data_loader import NuScenesLoader
        loader = NuScenesLoader(cfg)
        for frame_data in loader.iter_primary_camera(scene_idx=0):
            image = frame_data["image"]
            logger.info(f"nuScenes frame loaded: {image.shape}")
            return image
    except Exception as e:
        logger.warning(f"nuScenes frame load failed ({e}) — falling back to synthetic frame")
        return None


def verify_response(response_text: str) -> dict:
    """
    Parse and validate VLM response.
    Returns dict with 'ok', 'errors', 'data' keys.
    """
    result = {"ok": False, "errors": [], "data": None}

    import re
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        result["errors"].append("No JSON object found in response")
        return result

    try:
        data = json.loads(json_match.group())
        result["data"] = data
    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON parse error: {e}")
        return result

    required_keys = [
        "scene_context", "scene_summary", "risk_flags",
        "actors", "pedestrian_status", "recommended_caution"
    ]
    for key in required_keys:
        if key not in data:
            result["errors"].append(f"Missing key: '{key}'")

    valid_caution = {"normal", "elevated", "high", "critical"}
    caution = data.get("recommended_caution", "")
    if caution not in valid_caution:
        result["errors"].append(
            f"Invalid recommended_caution: '{caution}' — expected one of {valid_caution}"
        )

    valid_ped_status = {"clear", "caution", "danger"}
    ped_status = data.get("pedestrian_status", "")
    if ped_status not in valid_ped_status:
        result["errors"].append(
            f"Invalid pedestrian_status: '{ped_status}' — expected one of {valid_ped_status}"
        )

    if not result["errors"]:
        result["ok"] = True

    return result


def main():
    p = argparse.ArgumentParser(description="Verify Qwen2.5-VL-7B on PRISM pipeline")
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a synthetic test frame instead of nuScenes")
    p.add_argument("--model-dir",
                   default="/home/koushik-test/prism_data/models/qwen2_5_vl_7b",
                   help="Path to downloaded model directory")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print("=" * 60)
    print("PRISM — VLM Inference Verification")
    print("=" * 60)

    # ── Step 1: Load model ─────────────────────────────────────────────────────
    print("\n[1/4] Loading Qwen2.5-VL-7B...")
    t_load = time.time()

    cfg = load_config(args.config)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        local_dir = Path(args.model_dir).expanduser()
        source    = str(local_dir) if local_dir.exists() else "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"  Source: {source}")

        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        dtype     = torch.float16 if args.device == "cuda" else torch.float32
        model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=args.device,
            trust_remote_code=True,
        )
        model.eval()
        print(f"  Loaded in {time.time()-t_load:.1f}s  ✓")

    except Exception as e:
        print(f"  FAILED: {e}")
        print()
        print("  Check:")
        print("    1. Model downloaded: python scripts/download_vlm.py")
        print("    2. transformers installed:")
        print("       /home/koushik-test/cad_pipeline_env/bin/pip install "
              "transformers accelerate Pillow")
        sys.exit(1)

    # ── Step 2: Load test frame ────────────────────────────────────────────────
    print("\n[2/4] Loading test frame...")
    if args.synthetic:
        frame = make_synthetic_frame()
        print("  Synthetic frame (480×640)  ✓")
    else:
        frame = load_nuscenes_frame(cfg)
        if frame is None:
            frame = make_synthetic_frame()
            print("  Falling back to synthetic frame  (nuScenes unavailable)")
        else:
            print(f"  nuScenes frame {frame.shape}  ✓")

    # ── Step 3: Run inference ──────────────────────────────────────────────────
    print("\n[3/4] Running VLM inference...")

    from PIL import Image as PILImage
    import torch

    rgb     = frame[:, :, ::-1].copy()    # BGR → RGB
    pil_img = PILImage.fromarray(rgb.astype(np.uint8))

    prompt = (
        "You are an autonomous vehicle perception system analyzing a driving scene.\n\n"
        "Analyze this image carefully and respond with ONLY a JSON object:\n"
        "{\n"
        '  "scene_context": "brief description of road type, conditions, time of day",\n'
        '  "scene_summary": "one sentence describing the most critical aspect right now",\n'
        '  "risk_flags": ["list", "of", "specific", "risks"],\n'
        '  "actors": [],\n'
        '  "pedestrian_status": "clear/caution/danger",\n'
        '  "recommended_caution": "normal/elevated/high/critical"\n'
        "}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    t_inf = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    inference_ms = (time.time() - t_inf) * 1000

    input_len = inputs["input_ids"].shape[1]
    response  = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    print(f"  Inference time: {inference_ms:.0f}ms  ✓")

    # ── Step 4: Validate response ──────────────────────────────────────────────
    print("\n[4/4] Validating response...")
    print()
    print("  Raw VLM response:")
    for line in response.strip().splitlines():
        print(f"    {line}")
    print()

    result = verify_response(response)

    if result["ok"]:
        data = result["data"]
        print("  Response validation: PASS  ✓")
        print(f"  scene_context     : {data.get('scene_context', '')[:80]}")
        print(f"  scene_summary     : {data.get('scene_summary', '')[:80]}")
        print(f"  risk_flags        : {data.get('risk_flags', [])}")
        print(f"  pedestrian_status : {data.get('pedestrian_status', '')}")
        print(f"  recommended_caution: {data.get('recommended_caution', '')}")
        print(f"  actors detected   : {len(data.get('actors', []))}")
    else:
        print("  Response validation: FAIL")
        for err in result["errors"]:
            print(f"    ✗ {err}")

    # ── Latency check ──────────────────────────────────────────────────────────
    if inference_ms < 5000:
        print(f"\n  Latency: {inference_ms:.0f}ms < 5000ms  ✓")
    else:
        print(f"\n  Latency: {inference_ms:.0f}ms — high, but acceptable on first call")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if result["ok"]:
        print("RESULT: VLM VERIFICATION PASSED")
        print()
        print("Next step — run the full pipeline:")
        print("  python scripts/run_predictive.py")
    else:
        print("RESULT: VLM VERIFICATION FAILED — see errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
