"""
PRISM — Build TensorRT Engine from ONNX
=========================================
Converts yolov8n.onnx → yolov8n.engine using TensorRT Python API directly.
Works with TensorRT 10+ (avoids the deprecated EXPLICIT_BATCH flag).

Usage:
    /home/koushik-test/cad_pipeline_env/bin/python scripts/build_engine.py

Requires:
    - yolov8n.onnx in current directory (run ONNX export first)
    - tensorrt Python package installed
"""

import time
import sys
from pathlib import Path

ONNX_PATH   = "yolov8n.onnx"
ENGINE_PATH = "yolov8n.engine"
WORKSPACE_GB = 4

print("=" * 60)
print("PRISM — TensorRT Engine Builder")
print("=" * 60)

# ── Check inputs ──────────────────────────────────────────────
if not Path(ONNX_PATH).exists():
    print(f"ERROR: {ONNX_PATH} not found.")
    print("Run first:  yolo export model=yolov8n.pt format=onnx imgsz=640")
    sys.exit(1)

try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("ERROR: tensorrt not installed in this environment.")
    sys.exit(1)

# ── Build engine ──────────────────────────────────────────────
logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# TRT 10+: no flags needed (EXPLICIT_BATCH is default)
network = builder.create_network()
parser  = trt.OnnxParser(network, logger)

print(f"\nParsing ONNX: {ONNX_PATH}")
with open(ONNX_PATH, "rb") as f:
    ok = parser.parse(f.read())

if not ok:
    for i in range(parser.num_errors):
        print(f"  Parse error {i}: {parser.get_error(i)}")
    sys.exit(1)

print(f"  Input:  {network.get_input(0).shape}")
print(f"  Output: {network.get_output(0).shape}")

config = builder.create_builder_config()
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    WORKSPACE_GB * (1 << 30)
)
config.set_flag(trt.BuilderFlag.FP16)

print(f"\nBuilding FP16 engine (workspace={WORKSPACE_GB}GB)...")
print("This takes 3-10 minutes on first run.")
t0 = time.time()

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("ERROR: Engine build failed.")
    sys.exit(1)

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")
print(f"Engine saved: {ENGINE_PATH}  ({Path(ENGINE_PATH).stat().st_size / 1e6:.1f} MB)")
print()
print("Next steps:")
print(f"  1. Update configs/config.yaml:  model: '{ENGINE_PATH}'")
print(f"  2. Re-run profiling to measure FPS improvement")
print(f"  3. python scripts/export_tensorrt.py --benchmark-only")
