"""
PRISM — Qwen2.5-VL-7B Model Downloader
========================================
Downloads Qwen2.5-VL-7B-Instruct to ~/prism_data/models/qwen2_5_vl_7b
Saves in fp16 for direct loading on Jetson Thor (no quantization needed —
45GB available RAM is well above the ~14GB model footprint).

Usage:
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/download_vlm.py

Or with a custom output path:
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/download_vlm.py \
        --out /home/koushik-test/prism_data/models/qwen2_5_vl_7b
"""

import sys
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Download Qwen2.5-VL-7B for PRISM")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/koushik-test/prism_data/models/qwen2_5_vl_7b"),
        help="Directory to save the model",
    )
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID",
    )
    args = p.parse_args()

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("ERROR: transformers not installed.")
        print("Run: /home/koushik-test/cad_pipeline_env/bin/pip install transformers accelerate qwen-vl-utils Pillow")
        sys.exit(1)

    print("=" * 60)
    print("PRISM — Downloading Qwen2.5-VL-7B-Instruct")
    print("=" * 60)
    print(f"  Model  : {args.model_id}")
    print(f"  Output : {args.out}")
    print()
    print("This will download ~15GB. Progress is shown below.")
    print()

    args.out.mkdir(parents=True, exist_ok=True)

    print("  [1/2] Downloading model weights...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="cpu",          # stay on CPU — caller loads to device
    )
    model.save_pretrained(str(args.out))
    print("  [1/2] Model weights saved.")

    print("  [2/2] Downloading processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.save_pretrained(str(args.out))
    print("  [2/2] Processor saved.")

    print()
    print(f"  Done. Model saved to: {args.out}")
    print()
    print("Next step:")
    print("  python scripts/run_vlm_inference.py --model-dir", args.out)
    print("=" * 60)


if __name__ == "__main__":
    main()
