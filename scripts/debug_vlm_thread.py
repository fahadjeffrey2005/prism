"""
PRISM — VLM Thread Diagnostic
Tests whether VLMModel.infer() works correctly inside a background thread.
If it fails in a thread but works on the main thread, it's a CUDA context issue.

Usage:
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/debug_vlm_thread.py
"""
import sys
import numpy as np
import threading
import queue
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.semantic_reasoner.reasoner import VLMModel

MODEL_DIR = "/home/koushik-test/prism_data/models/qwen2_5_vl_7b"

print("=" * 60)
print("PRISM — VLM Thread Diagnostic")
print("=" * 60)

vlm = VLMModel(device="cuda", enabled=True, model_dir=MODEL_DIR)
print(f"available: {vlm.available}")

frame = np.zeros((240, 320, 3), dtype=np.uint8)
result_q = queue.Queue()


def worker():
    print("thread: starting infer()...")
    try:
        resp, ms = vlm.infer(frame, "Describe this image.")
        result_q.put(("ok", resp[:80], ms))
        print(f"thread: done — {ms:.0f}ms")
    except Exception as e:
        tb = traceback.format_exc()
        result_q.put(("error", tb, 0))
        print(f"thread ERROR:\n{tb}")


print("\n[1] Synchronous infer (main thread)...")
resp, ms = vlm.infer(frame, "Describe this image.")
print(f"  main thread: {ms:.0f}ms | {resp[:80]}")

print("\n[2] Async infer (background thread)...")
t = threading.Thread(target=worker, daemon=True)
t.start()
t.join(timeout=30)

if not result_q.empty():
    status, resp, ms = result_q.get()
    print(f"\nResult: status={status}  ms={ms:.0f}")
    if status == "ok":
        print(f"  response: {resp}")
        print("\nVLM THREAD TEST: PASS")
    else:
        print(f"  error:\n{resp}")
        print("\nVLM THREAD TEST: FAIL — CUDA context issue in thread")
        print("\nFix: set torch.cuda.set_device(0) at thread start")
else:
    print("\nTIMEOUT — thread did not complete in 30s")
    print("VLM THREAD TEST: FAIL — inference hung in thread")

print("=" * 60)
