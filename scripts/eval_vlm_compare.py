"""
PRISM — VLM Memorization vs Learning Eval
==========================================
Compares base model vs fine-tuned model across train and val scenes
to determine if fine-tuning caused memorization or genuine learning.

Split (same as finetune_vlm.py):
    TRAIN scenes: 0-7
    VAL   scenes: 8-9

Memorization: fine-tuned >> base on TRAIN, fine-tuned ~= base on VAL
Genuine learning: fine-tuned >> base on BOTH train and val

Also fixes the GT comparison bug in previous eval scripts:
    OLD: compared VLM count against ALL 3D annotations across all 6 cameras
    NEW: filters GT to objects visible in CAM_FRONT using ego-frame geometry

Usage:
    # Base model only
    python scripts/eval_vlm_compare.py

    # Compare base vs fine-tuned
    python scripts/eval_vlm_compare.py --checkpoint /home/koushik-test/prism_data/checkpoints/vlm_phase1/latest.pt

    # Specific scenes
    python scripts/eval_vlm_compare.py --scenes 2 3 4 5 8 9
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.finetune_vlm import apply_lora


NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"

COT_PROMPT = """You are an autonomous vehicle perception system. Analyze this driving scene carefully using the following steps:

STEP 1 - SCAN: Describe what you see from left to right across the entire image. Include background, foreground, near and far objects.

STEP 2 - IDENTIFY ACTORS: List every person, vehicle, cyclist you can see. For each one state: type, approximate distance (near/mid/far), and what they appear to be doing.

STEP 3 - ASSESS INTENT: For each moving or potentially moving actor, what are they likely to do in the next 3 seconds?

STEP 4 - IDENTIFY RISKS: What are the top 3 risks in this scene right now?

STEP 5 - DECIDE: Based on your analysis, what should the autonomous vehicle do?
Choose from: CLEAR (full speed) / MONITOR (full speed, watch) / EASE (gentle decel) / SLOW (meaningful decel) / CAUTION (prepare to stop) / YIELD (near stop) / STOP (full stop)

STEP 6 - OUTPUT: Summarize in JSON:
{"total_pedestrians": N, "total_vehicles": N, "decision": "X", "primary_risk": "Y", "reasoning": "Z"}"""


# ── Ground truth — front-camera filtered ─────────────────────────────────────

def get_front_camera_gt(nusc, sample):
    """
    Returns (ped_count, veh_count) for objects roughly visible in CAM_FRONT.
    Filters by ego-frame position: forward, within ~36 degree half-FOV.
    """
    ped_count = veh_count = 0
    ego_pose  = nusc.get("ego_pose",
        nusc.get("sample_data", sample["data"]["CAM_FRONT"])["ego_pose_token"])
    ego_trans = ego_pose["translation"]
    w, x, y, z = ego_pose["rotation"]
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)

    for ann_token in sample["anns"]:
        ann  = nusc.get("sample_annotation", ann_token)
        cat  = nusc.get("category", ann["category_token"])["name"]
        trans = ann["translation"]
        dx = trans[0] - ego_trans[0]
        dy = trans[1] - ego_trans[1]
        ego_x =  cos_y * dx + sin_y * dy
        ego_y = -sin_y * dx + cos_y * dy
        dist  = np.sqrt(ego_x**2 + ego_y**2)
        if ego_x > 0 and dist < 50 and abs(ego_y) / max(ego_x, 0.1) < 0.75:
            if "pedestrian" in cat or "human" in cat:
                ped_count += 1
            elif "vehicle" in cat:
                veh_count += 1

    return ped_count, veh_count


def get_total_gt(nusc, sample):
    peds = sum(1 for a in sample["anns"]
               if "pedestrian" in nusc.get("sample_annotation", a)["category_name"])
    vehs = sum(1 for a in sample["anns"]
               if "vehicle"    in nusc.get("sample_annotation", a)["category_name"])
    return peds, vehs


# ── Inference ─────────────────────────────────────────────────────────────────

def run_scene(model, processor, nusc, scene_idx, model_label="BASE"):
    scene  = nusc.scene[scene_idx]
    sample = nusc.get("sample", scene["first_sample_token"])
    sd     = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image  = Image.open(NUSCENES_ROOT + "/" + sd["filename"]).convert("RGB")

    gt_peds_front, gt_vehs_front = get_front_camera_gt(nusc, sample)
    gt_peds_total, gt_vehs_total = get_total_gt(nusc, sample)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  COT_PROMPT}
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    pred_peds, pred_vehs, decision, primary_risk, reasoning = None, None, "?", "", ""
    try:
        s = response.rfind("{")
        e = response.rfind("}") + 1
        if s >= 0 and e > s:
            p = json.loads(response[s:e])
            pred_peds    = p.get("total_pedestrians")
            pred_vehs    = p.get("total_vehicles")
            decision     = p.get("decision", "?")
            primary_risk = p.get("primary_risk", "")
            reasoning    = p.get("reasoning", "")
    except Exception:
        pass

    ped_error = abs(pred_peds - gt_peds_front) if pred_peds is not None else None
    veh_error = abs(pred_vehs - gt_vehs_front) if pred_vehs is not None else None
    split     = "TRAIN" if scene_idx < 8 else "VAL"

    print(f"\n  [{model_label}] Scene {scene_idx} ({split}): {scene['description'][:60]}")
    print(f"  GT (front-cam): {gt_peds_front} peds, {gt_vehs_front} vehs  "
          f"(all-cam total: {gt_peds_total} peds, {gt_vehs_total} vehs)")
    print(f"  VLM predicted:  {pred_peds} peds, {pred_vehs} vehs")
    if ped_error is not None:
        print(f"  Ped error: {ped_error}  |  Veh error: {veh_error}")
    print(f"  Decision: {decision}  |  Risk: {primary_risk}")
    print(f"  Reasoning: {reasoning[:120]}")
    print(f"\n  Full CoT:")
    for line in response.split("\n")[:25]:
        print(f"    {line}")

    return {
        "scene_idx":      scene_idx,
        "split":          split,
        "in_train":       scene_idx < 8,
        "description":    scene["description"],
        "gt_peds_front":  gt_peds_front,
        "gt_vehs_front":  gt_vehs_front,
        "pred_peds":      pred_peds,
        "pred_vehs":      pred_vehs,
        "ped_error":      ped_error,
        "veh_error":      veh_error,
        "decision":       decision,
        "model_label":    model_label,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results, label):
    train_r = [r for r in results if r["in_train"]]
    val_r   = [r for r in results if not r["in_train"]]
    print(f"\n{'='*65}")
    print(f"SUMMARY — {label}")
    print(f"{'='*65}")
    for name, group in [("TRAIN scenes (0-7)", train_r), ("VAL scenes (8-9)", val_r)]:
        if not group:
            continue
        pe = [r["ped_error"] for r in group if r["ped_error"] is not None]
        ve = [r["veh_error"] for r in group if r["veh_error"] is not None]
        print(f"\n  {name}:")
        if pe: print(f"    Ped MAE: {np.mean(pe):.2f}  (errors: {[round(e,1) for e in pe]})")
        if ve: print(f"    Veh MAE: {np.mean(ve):.2f}  (errors: {[round(e,1) for e in ve]})")
        print(f"    Decisions: {[r['decision'] for r in group]}")


def print_comparison(base_results, ft_results):
    print(f"\n{'='*65}")
    print("MEMORIZATION DIAGNOSIS")
    print(f"{'='*65}")
    print(f"\n  {'Scene':<7} {'Split':<6} {'Base Err':<10} {'FT Err':<10} {'Delta':<10} {'Base Dec':<10} {'FT Dec'}")
    print(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    train_deltas, val_deltas = [], []
    for b, f in zip(base_results, ft_results):
        b_err  = b["ped_error"] if b["ped_error"] is not None else float("nan")
        f_err  = f["ped_error"] if f["ped_error"] is not None else float("nan")
        delta  = b_err - f_err
        sym    = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {b['scene_idx']:<7} {b['split']:<6} {b_err:<10.1f} {f_err:<10.1f} "
              f"{delta:+.1f}{sym:<7} {b['decision']:<10} {f['decision']}")
        (train_deltas if b["in_train"] else val_deltas).append(delta)

    print()
    avg_train = np.mean(train_deltas) if train_deltas else float("nan")
    avg_val   = np.mean(val_deltas)   if val_deltas   else float("nan")
    if train_deltas: print(f"  Avg improvement on TRAIN: {avg_train:+.2f} ped MAE")
    if val_deltas:   print(f"  Avg improvement on VAL:   {avg_val:+.2f} ped MAE")

    if train_deltas and val_deltas:
        gap = avg_train - avg_val
        print(f"  Train-Val gap: {gap:.2f}\n")
        if gap > 1.5:
            print("  VERDICT: MEMORIZATION LIKELY")
            print("  Fine-tuning improved training scenes far more than val.")
            print("  -> Proceed to fine-tuning v2: IDD dataset, 1 epoch.")
        elif avg_val > 0.3:
            print("  VERDICT: GENUINE LEARNING")
            print("  Val improvement confirms generalization.")
            print("  -> Fine-tuning v2 will reinforce this with IDD.")
        else:
            print("  VERDICT: INCONCLUSIVE")
            print("  Base model already strong. Fine-tuning gave marginal gain.")
            print("  -> Focus on IDD fine-tuning for India-specific improvement.")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model():
    print("Loading Qwen2.5-VL-7B-Instruct (base)...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    ).cuda()
    model.eval()
    return model, processor


def load_finetuned_model(checkpoint_path: str):
    model, processor = load_base_model()
    print(f"Loading LoRA checkpoint: {checkpoint_path}")
    _, lora_layers = apply_lora(model, rank=16, alpha=32)
    state = torch.load(checkpoint_path, map_location="cuda")
    loaded = sum(1 for name, layer in lora_layers.items()
                 if name in state and not layer.load_state_dict(state[name], strict=False))
    print(f"  LoRA: {len(lora_layers)} layers patched")
    model.eval()
    return model, processor


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA .pt checkpoint. If given, runs base + fine-tuned.")
    parser.add_argument("--scenes", type=int, nargs="+", default=[2, 3, 4, 5, 8, 9])
    parser.add_argument("--finetuned-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("PRISM — VLM Memorization vs Learning Eval")
    print("=" * 65)
    print(f"Scenes: {args.scenes}")
    print(f"Train: {[s for s in args.scenes if s < 8]}  |  "
          f"Val: {[s for s in args.scenes if s >= 8]}")
    print()

    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_ROOT, verbose=False)

    base_results = []
    if not args.finetuned_only:
        model, processor = load_base_model()
        print(f"\n{'='*65}\nPHASE 1 — BASE MODEL\n{'='*65}")
        for si in args.scenes:
            base_results.append(run_scene(model, processor, nusc, si, "BASE"))
        print_summary(base_results, "BASE MODEL")
        del model
        torch.cuda.empty_cache()

    ft_results = []
    if args.checkpoint:
        model, processor = load_finetuned_model(args.checkpoint)
        print(f"\n{'='*65}\nPHASE 2 — FINE-TUNED MODEL\n{'='*65}")
        for si in args.scenes:
            ft_results.append(run_scene(model, processor, nusc, si, "FINETUNED"))
        print_summary(ft_results, "FINE-TUNED MODEL")

    if base_results and ft_results:
        print_comparison(base_results, ft_results)


if __name__ == "__main__":
    main()
