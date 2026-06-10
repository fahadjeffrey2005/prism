"""
PRISM — VLM Inference Latency Benchmark
=========================================
Measures steady-state Qwen2.5-VL-7B inference latency on the current hardware.
Run this on Jetson to get the real per-inference latency for the paper.

Usage:
    /home/koushik-test/cad_pipeline_env/bin/python scripts/benchmark_vlm_latency.py

Output saved to: ~/prism_data/logs/vlm_latency_benchmark.txt
"""

import torch
import time
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_DIR = "/home/koushik-test/prism_data/models/qwen2_5_vl_7b"
N_RUNS = 5
MAX_NEW_TOKENS = 50  # short response — matches paper's structured prompt

print("=" * 60)
print("PRISM — VLM Latency Benchmark")
print("=" * 60)
print(f"Model:  {MODEL_DIR}")
print(f"Runs:   {N_RUNS}")
print(f"Tokens: {MAX_NEW_TOKENS} max new tokens")
print()

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print("Loading processor...")
proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("Loading model (fp16, cuda)...")
t_load = time.perf_counter()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()
load_ms = (time.perf_counter() - t_load) * 1000
print(f"Model loaded in {load_ms/1000:.1f}s")
print(f"Device: {next(model.parameters()).device}")
print(f"dtype:  {next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print()

# Dummy driving image
img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

msgs = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": "Describe this driving scene briefly. What are the main hazards?"},
    ],
}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

print("Running inference benchmark...")
latencies = []
for i in range(N_RUNS):
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       temperature=None, top_p=None)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    latencies.append(ms)
    print(f"  Run {i+1}/{N_RUNS}: {ms:.0f} ms")

print()
mean_ms  = float(np.mean(latencies))
min_ms   = float(np.min(latencies))
max_ms   = float(np.max(latencies))
p50_ms   = float(np.percentile(latencies, 50))

print("=" * 60)
print("Results:")
print(f"  Mean:  {mean_ms:.0f} ms")
print(f"  Min:   {min_ms:.0f} ms")
print(f"  Max:   {max_ms:.0f} ms")
print(f"  p50:   {p50_ms:.0f} ms")
print()
print(f"  At cooldown_s=0.4s, max VLM rate = 2.5 Hz")
print(f"  At {mean_ms:.0f}ms per inference, actual rate = {1000/mean_ms:.2f} Hz")
print("=" * 60)

# Save for paper
log_dir = Path.home() / "prism_data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
out = log_dir / "vlm_latency_benchmark.txt"
lines = [
    "PRISM — VLM Latency Benchmark",
    f"Model: Qwen2.5-VL-7B-Instruct fp16",
    f"Hardware: {torch.cuda.get_device_name(0)}",
    f"Runs: {N_RUNS} x {MAX_NEW_TOKENS} max new tokens",
    "",
    f"Mean:  {mean_ms:.0f} ms",
    f"Min:   {min_ms:.0f} ms",
    f"Max:   {max_ms:.0f} ms",
    f"p50:   {p50_ms:.0f} ms",
    "",
    f"All runs: {[f'{x:.0f}ms' for x in latencies]}",
]
out.write_text("\n".join(lines))
print(f"\nSaved to: {out}")
