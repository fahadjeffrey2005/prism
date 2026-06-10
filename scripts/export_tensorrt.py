"""
PRISM — TensorRT Export + Latency Benchmark for YOLOv8n
=========================================================
Run this on the Jetson to convert YOLOv8n from CUDA fp16 to TensorRT engine.
TensorRT typically gives 3-4x speedup over raw CUDA inference.

Expected result:
    YOLOv8n CUDA fp16:   ~51ms  (current sensory bottleneck)
    YOLOv8n TensorRT:    ~13-17ms  (projected)
    → Pipeline drops from 86ms → ~50ms, theoretical FPS: ~20fps

Usage:
    # On Jetson — export once, then benchmark:
    python scripts/export_tensorrt.py

    # Benchmark only (if .engine file already exists):
    python scripts/export_tensorrt.py --benchmark-only

    # Specify a different model:
    python scripts/export_tensorrt.py --model yolov8s.pt

Output:
    yolov8n.engine          — TensorRT engine file
    ~/prism_data/logs/tensorrt_benchmark.txt — timing results for paper
"""

import argparse
import time
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="yolov8n.pt")
    p.add_argument("--benchmark-only", action="store_true",
                   help="Skip export, only benchmark existing .engine file")
    p.add_argument("--n-warmup",       type=int, default=20,
                   help="Warmup frames before timing")
    p.add_argument("--n-bench",        type=int, default=200,
                   help="Frames to time for benchmark")
    p.add_argument("--img-size",       type=int, default=640)
    return p.parse_args()


def export_tensorrt(model_name: str, img_size: int) -> str:
    """
    Export YOLOv8 model to TensorRT engine.
    Returns path to the .engine file.
    """
    from ultralytics import YOLO

    engine_name = model_name.replace(".pt", ".engine")

    print(f"\n{'='*60}")
    print(f"Exporting {model_name} → {engine_name}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)

    t0 = time.time()
    model.export(
        format="engine",
        device=0,              # GPU 0
        half=True,             # fp16
        imgsz=img_size,
        workspace=4,           # GB — reduce if Jetson runs OOM
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nExport complete in {elapsed:.0f}s")
    print(f"Engine saved: {engine_name}")
    return engine_name


def benchmark(model_path: str, img_size: int, n_warmup: int, n_bench: int) -> dict:
    """
    Benchmark detection latency: both CUDA fp16 pt and TensorRT engine.
    Returns dict with timing results.
    """
    import torch
    from ultralytics import YOLO

    results = {}
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # ── Benchmark original .pt (fp16 CUDA) ───────────────────────────────────
    pt_path = model_path.replace(".engine", ".pt")
    if Path(pt_path).exists():
        print(f"\nBenchmarking original: {pt_path}")
        model_pt = YOLO(pt_path)
        model_pt.to("cuda")

        # Warmup
        for _ in range(n_warmup):
            model_pt(dummy, device="cuda", half=True, verbose=False)

        # Time
        latencies = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            model_pt(dummy, device="cuda", half=True, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)

        results["cuda_fp16_mean_ms"] = float(np.mean(latencies))
        results["cuda_fp16_p50_ms"]  = float(np.percentile(latencies, 50))
        results["cuda_fp16_p95_ms"]  = float(np.percentile(latencies, 95))
        print(f"  CUDA fp16:  mean={results['cuda_fp16_mean_ms']:.1f}ms  "
              f"p50={results['cuda_fp16_p50_ms']:.1f}ms  "
              f"p95={results['cuda_fp16_p95_ms']:.1f}ms")

    # ── Benchmark TensorRT engine ─────────────────────────────────────────────
    engine_path = model_path if model_path.endswith(".engine") else model_path.replace(".pt", ".engine")
    if not Path(engine_path).exists():
        print(f"\nEngine not found: {engine_path}")
        print("Run without --benchmark-only to export first.")
        return results

    print(f"\nBenchmarking TensorRT: {engine_path}")
    model_trt = YOLO(engine_path)

    # Warmup
    for _ in range(n_warmup):
        model_trt(dummy, verbose=False)

    # Time
    latencies_trt = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        model_trt(dummy, verbose=False)
        latencies_trt.append((time.perf_counter() - t0) * 1000)

    results["trt_mean_ms"] = float(np.mean(latencies_trt))
    results["trt_p50_ms"]  = float(np.percentile(latencies_trt, 50))
    results["trt_p95_ms"]  = float(np.percentile(latencies_trt, 95))

    print(f"  TensorRT:   mean={results['trt_mean_ms']:.1f}ms  "
          f"p50={results['trt_p50_ms']:.1f}ms  "
          f"p95={results['trt_p95_ms']:.1f}ms")

    if "cuda_fp16_mean_ms" in results:
        speedup = results["cuda_fp16_mean_ms"] / results["trt_mean_ms"]
        results["speedup"] = round(speedup, 2)
        print(f"\n  Speedup:    {speedup:.2f}x")

    return results


def write_report(results: dict, model_name: str, img_size: int):
    """Write benchmark results to a text file for the paper."""
    log_dir = Path.home() / "prism_data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "tensorrt_benchmark.txt"

    lines = [
        "=" * 60,
        "PRISM — TensorRT Benchmark Results",
        "=" * 60,
        f"Model:       {model_name}",
        f"Image size:  {img_size}x{img_size}",
        f"Device:      Jetson (CUDA)",
        "",
    ]

    if "cuda_fp16_mean_ms" in results:
        lines += [
            "YOLOv8n CUDA fp16 (baseline):",
            f"  Mean:   {results['cuda_fp16_mean_ms']:.1f} ms",
            f"  p50:    {results['cuda_fp16_p50_ms']:.1f} ms",
            f"  p95:    {results['cuda_fp16_p95_ms']:.1f} ms",
            "",
        ]

    if "trt_mean_ms" in results:
        lines += [
            "YOLOv8n TensorRT fp16:",
            f"  Mean:   {results['trt_mean_ms']:.1f} ms",
            f"  p50:    {results['trt_p50_ms']:.1f} ms",
            f"  p95:    {results['trt_p95_ms']:.1f} ms",
            "",
        ]

    if "speedup" in results:
        proj_sensory = results.get("cuda_fp16_mean_ms", 51.3) / results["speedup"]
        # Current total pipeline is ~86.2ms; sensory is 51.3ms of that
        other_ms = 86.2 - 51.3
        proj_total = proj_sensory + other_ms
        proj_fps = 1000.0 / proj_total

        lines += [
            f"Speedup:     {results['speedup']:.2f}x",
            "",
            "Projected pipeline impact:",
            f"  Sensory latency:  {results['cuda_fp16_mean_ms']:.1f}ms → {proj_sensory:.1f}ms",
            f"  Total pipeline:   86.2ms → {proj_total:.1f}ms",
            f"  Theoretical FPS:  11.6 → {proj_fps:.1f}",
            "",
        ]

    lines.append("=" * 60)
    text = "\n".join(lines)
    print("\n" + text)
    out.write_text(text)
    print(f"\nSaved: {out}")


def main():
    args = parse_args()

    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Run this on the Jetson.")
            sys.exit(1)
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu}")
    except ImportError:
        print("ERROR: torch not installed.")
        sys.exit(1)

    engine_path = args.model.replace(".pt", ".engine")

    if not args.benchmark_only:
        engine_path = export_tensorrt(args.model, args.img_size)

    results = benchmark(engine_path, args.img_size, args.n_warmup, args.n_bench)
    write_report(results, args.model, args.img_size)

    print("\nDone. Update config.yaml to use the engine:")
    print(f'  sensory_core:\n    model: {engine_path}')


if __name__ == "__main__":
    main()
