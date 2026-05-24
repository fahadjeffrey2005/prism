"""
PRISM — VLM Fine-tuning Script
================================
Fine-tunes Qwen2.5-VL-7B on driving scene understanding using LoRA.

Phase 1: nuScenes scenes — general driving understanding
Phase 2: IDD (India Driving Dataset) — Indian road scenarios

LoRA config:
    rank=16, alpha=32
    Target: q_proj, v_proj, k_proj, o_proj
    Trainable params: ~0.5% of total — memory efficient

Training data format:
    {
        "image": <camera frame>,
        "question": "Describe this driving scene and identify risks",
        "answer": <structured scene description>
    }

Usage:
    python scripts/finetune_vlm.py --phase 1
    python scripts/finetune_vlm.py --phase 2 --checkpoint checkpoints/phase1
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from prism.utils.common import load_config, get_logger
from prism.sensory_core.data_loader import NuScenesLoader

logger = get_logger("VLMFinetune")


# ── LoRA implementation ───────────────────────────────────────────────────────

class LoRALayer(torch.nn.Module):
    """
    Low-Rank Adaptation layer.
    Adds trainable rank decomposition to frozen weight matrix.
    W' = W + (B @ A) * scale
    """
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.scale = alpha / rank

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices
        in_features = original_layer.weight.shape[1]
        out_features = original_layer.weight.shape[0]

        self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_features, bias=False)

        # Initialize: A=random, B=zero (so initial delta=0)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


def apply_lora(model, rank=16, alpha=32, target_modules=None):
    """Apply LoRA to target modules in the model."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_layers = {}
    total_params = 0
    lora_params = 0

    for name, module in model.named_modules():
        for param in module.parameters():
            total_params += param.numel()

    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, torch.nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_params += lora_layer.lora_A.weight.numel()
                lora_params += lora_layer.lora_B.weight.numel()
                lora_layers[name] = lora_layer

    logger.info(f"LoRA applied to {len(lora_layers)} layers")
    logger.info(f"Trainable params: {lora_params:,} / {total_params:,} ({lora_params/total_params*100:.2f}%)")
    return model, lora_layers


# ── Dataset ───────────────────────────────────────────────────────────────────

SCENE_QUESTION = """Analyze this driving scene carefully. Provide a structured assessment:
1. Scene context (road type, conditions, time of day)
2. All actors present (vehicles, pedestrians, cyclists) with their positions and states
3. Immediate risks or hazards
4. Recommended action (CLEAR/MONITOR/EASE/SLOW/CAUTION/YIELD/STOP)
5. Brief reasoning

Respond in JSON format."""


def build_answer_from_annotations(nusc, sample_token: str, camera_name: str) -> str:
    """Build structured ground truth answer from nuScenes annotations."""
    sample = nusc.get("sample", sample_token)
    annotations = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cat = ann["category_name"]
        trans = ann["translation"]

        # Compute distance from ego
        dist = float(np.sqrt(trans[0]**2 + trans[1]**2))

        # Simplified category
        if "vehicle.car" in cat:
            cls = "car"
        elif "vehicle.truck" in cat:
            cls = "truck"
        elif "pedestrian" in cat:
            cls = "pedestrian"
        elif "vehicle.bicycle" in cat:
            cls = "bicycle"
        elif "vehicle.motorcycle" in cat:
            cls = "motorcycle"
        elif "vehicle.bus" in cat:
            cls = "bus"
        else:
            continue

        if dist < 50:  # only nearby actors
            annotations.append({
                "class": cls,
                "distance_m": round(dist, 1),
                "lateral_m": round(trans[1], 1),
            })

    # Sort by distance
    annotations.sort(key=lambda x: x["distance_m"])

    # Determine recommended action based on closest actor
    closest = annotations[0]["distance_m"] if annotations else 999
    peds = [a for a in annotations if a["class"] == "pedestrian"]

    if closest < 5:
        action = "STOP"
    elif closest < 10 and peds:
        action = "CAUTION"
    elif closest < 15:
        action = "SLOW"
    elif closest < 25:
        action = "EASE"
    elif annotations:
        action = "MONITOR"
    else:
        action = "CLEAR"

    answer = {
        "scene_context": "urban driving scene",
        "actors": annotations[:8],
        "immediate_risks": [f"{a['class']} at {a['distance_m']}m" for a in annotations[:3] if a['distance_m'] < 20],
        "recommended_action": action,
        "reasoning": f"{'Pedestrians detected nearby. ' if peds else ''}{len(annotations)} actors in scene. Closest at {closest:.1f}m."
    }
    return json.dumps(answer, indent=2)


class NuScenesVLMDataset(Dataset):
    """
    nuScenes dataset formatted for VLM fine-tuning.
    Each sample: (image, question, answer) triple.
    """

    def __init__(self, cfg: dict, processor, split: str = "train"):
        self.cfg = cfg
        self.processor = processor
        self.split = split
        self.samples = []
        self._build_samples(cfg)
        logger.info(f"Dataset: {len(self.samples)} samples ({split})")

    def _build_samples(self, cfg):
        from nuscenes.nuscenes import NuScenes
        data_root = cfg["data"]["nuscenes_root"]
        version = cfg["data"]["nuscenes_version"]
        nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

        # Use 8 scenes for train, 2 for val
        n_scenes = len(nusc.scene)
        if self.split == "train":
            scene_indices = list(range(min(8, n_scenes)))
        else:
            scene_indices = list(range(8, n_scenes))

        camera = "CAM_FRONT"
        for scene_idx in scene_indices:
            scene = nusc.scene[scene_idx]
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                if camera not in sample["data"]:
                    sample_token = sample["next"]
                    continue
                sd_token = sample["data"][camera]
                sd = nusc.get("sample_data", sd_token)
                img_path = str(Path(data_root) / sd["filename"])
                answer = build_answer_from_annotations(nusc, sample_token, camera)
                self.samples.append({
                    "image_path": img_path,
                    "question": SCENE_QUESTION,
                    "answer": answer,
                    "sample_token": sample_token,
                })
                sample_token = sample["next"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        from PIL import Image as PILImage

        image = PILImage.open(sample["image_path"]).convert("RGB")
        image = image.resize((448, 252))  # resize for efficiency

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": sample["answer"],
            }
        ]

        return {
            "messages": messages,
            "answer": sample["answer"],
        }


def collate_fn(batch, processor):
    """Custom collate for VLM training."""
    from qwen_vl_utils import process_vision_info

    all_inputs = []

    for item in batch:
        messages = item["messages"]
        # Full conversation including assistant response
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Process each sample individually to avoid token mismatch
        user_msgs = [m for m in messages if m["role"] == "user"]
        image_inputs, video_inputs = process_vision_info(user_msgs)

        inp = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        all_inputs.append(inp)

    # Batch together
    from torch.nn.utils.rnn import pad_sequence
    import torch

    input_ids = pad_sequence(
        [x["input_ids"][0] for x in all_inputs],
        batch_first=True, padding_value=processor.tokenizer.pad_token_id or 0
    )
    attention_mask = pad_sequence(
        [x["attention_mask"][0] for x in all_inputs],
        batch_first=True, padding_value=0
    )

    result = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Handle pixel values
    if "pixel_values" in all_inputs[0]:
        result["pixel_values"] = torch.cat([x["pixel_values"] for x in all_inputs], dim=0)
    if "image_grid_thw" in all_inputs[0]:
        result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in all_inputs], dim=0)

    # Labels = input_ids shifted (causal LM)
    result["labels"] = input_ids.clone()
    return result


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    cfg = load_config(args.config)

    # Checkpoint directory
    ckpt_dir = Path(cfg["data"]["root"]) / "checkpoints" / f"vlm_phase{args.phase}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"PRISM VLM Fine-tuning — Phase {args.phase}")
    logger.info("=" * 60)
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load model
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    ).cuda()

    # Load from checkpoint if continuing
    if args.checkpoint:
        logger.info(f"Loading LoRA checkpoint: {args.checkpoint}")
        lora_state = torch.load(args.checkpoint)
        # Apply LoRA first then load weights
        model, lora_layers = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        # Load saved LoRA weights
        for name, layer in lora_layers.items():
            if name in lora_state:
                layer.load_state_dict(lora_state[name])
    else:
        model, lora_layers = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Dataset
    logger.info("Building dataset...")
    train_dataset = NuScenesVLMDataset(cfg, processor, split="train")
    val_dataset = NuScenesVLMDataset(cfg, processor, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info(f"Starting training for {args.epochs} epochs...")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        t_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            train_losses.append(loss.item())

            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                val_losses.append(outputs.loss.item())

        avg_val_loss = np.mean(val_losses)
        epoch_time = time.time() - t_start

        logger.info(
            f"Epoch {epoch+1} complete | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = ckpt_dir / f"best_epoch{epoch+1}_val{avg_val_loss:.4f}.pt"
            torch.save(
                {name: layer.state_dict() for name, layer in lora_layers.items()},
                ckpt_path
            )
            logger.info(f"Saved best checkpoint: {ckpt_path}")

        # Save latest
        torch.save(
            {name: layer.state_dict() for name, layer in lora_layers.items()},
            ckpt_dir / "latest.pt"
        )

    logger.info("=" * 60)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM VLM Fine-tuning")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1=nuScenes, 2=India-specific")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA checkpoint to continue from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
