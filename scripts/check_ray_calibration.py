"""
PRISM — Ray Unprojection Calibration Check
===========================================
Verifies that real nuScenes calibration data flows correctly through
CameraExtrinsics and that ray_unproject_to_ground() produces sensible
metric positions.

Run on Jetson:
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/check_ray_calibration.py

    # Optionally specify a scene index (default=2)
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/check_ray_calibration.py --scene 2
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from prism.sensory_core.metric_depth import (
    CameraIntrinsics,
    CameraExtrinsics,
    MetricDepthEngine,
    ray_unproject_to_ground,
)

NUSCENES_ROOT = "/home/koushik-test/prism_data/datasets/nuscenes"


def get_real_calibration(nusc: NuScenes, scene_idx: int) -> dict:
    scene  = nusc.scene[scene_idx]
    sample = nusc.get("sample", scene["first_sample_token"])
    sd     = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    calib  = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return {
        "camera_intrinsic": np.array(calib["camera_intrinsic"]),
        "translation":      np.array(calib["translation"]),
        "rotation":         np.array(calib["rotation"]),
    }


def check_ray_unprojection(calib: dict):
    print("\n── Ray Unprojection (real calibration) ──────────────────────")
    K    = calib["camera_intrinsic"]
    intr = CameraIntrinsics.from_matrix(K)
    extr = CameraExtrinsics.from_calibration(calib)

    print(f"  Intrinsics  fx={intr.fx:.1f}  fy={intr.fy:.1f}  "
          f"cx={intr.cx:.1f}  cy={intr.cy:.1f}")
    print(f"  Extrinsics  t={np.round(extr.translation, 3)}  "
          f"q={np.round(extr.rotation_q, 4)}")

    test_pixels = [
        ("centre",     816.0, 650.0),
        ("right",     1150.0, 660.0),
        ("left",       480.0, 660.0),
        ("far-centre", 816.0, 530.0),
    ]

    all_ok = True
    for label, u, v in test_pixels:
        pt = ray_unproject_to_ground(u, v, intr, extr)
        if pt is None:
            print(f"  [{label}] ({u:.0f},{v:.0f}) → FAILED (None returned)")
            all_ok = False
        else:
            fwd = pt[0]
            lat = -pt[1]          # nuScenes y-left → positive=right for display
            ok  = fwd > 0
            sym = "✓" if ok else "✗"
            print(f"  [{sym} {label:<12}] ({u:.0f},{v:.0f}) → "
                  f"forward={fwd:6.2f}m  lateral={lat:+6.2f}m")
            if not ok:
                all_ok = False

    return all_ok


def check_engine_pipeline(calib: dict):
    print("\n── MetricDepthEngine.update_intrinsics() path ───────────────")

    cfg = {}
    engine = MetricDepthEngine(cfg)
    print(f"  Before update: extrinsics = {engine.geometry.extrinsics}")

    engine.update_intrinsics(calib)

    has_extr = engine.geometry.extrinsics is not None
    has_intr = engine.geometry.intrinsics is not None
    print(f"  After  update: intrinsics={'loaded' if has_intr else 'MISSING'}  "
          f"extrinsics={'loaded' if has_extr else 'MISSING'}")

    if has_extr:
        e = engine.geometry.extrinsics
        print(f"  Extrinsics t={np.round(e.translation, 3)}  q={np.round(e.rotation_q, 4)}")

    return has_extr


def main():
    parser = argparse.ArgumentParser(description="PRISM ray unprojection calibration check")
    parser.add_argument("--scene", type=int, default=2,
                        help="nuScenes scene index (default=2)")
    args = parser.parse_args()

    print("=" * 65)
    print("PRISM — Ray Unprojection Calibration Check")
    print("=" * 65)
    print(f"  nuScenes root : {NUSCENES_ROOT}")
    print(f"  Scene index   : {args.scene}")

    print("\nLoading nuScenes...")
    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_ROOT, verbose=False)

    calib = get_real_calibration(nusc, args.scene)
    print(f"\n  camera_intrinsic:\n{np.round(calib['camera_intrinsic'], 2)}")
    print(f"  translation : {np.round(calib['translation'], 4)}")
    print(f"  rotation    : {np.round(calib['rotation'], 4)}")

    ray_ok    = check_ray_unprojection(calib)
    engine_ok = check_engine_pipeline(calib)

    print("\n" + "=" * 65)
    if ray_ok and engine_ok:
        print("RESULT: ALL CHECKS PASSED — ray unprojection is live")
    else:
        print("RESULT: ISSUES DETECTED")
        if not ray_ok:
            print("  ✗ ray_unproject_to_ground() returned bad values")
        if not engine_ok:
            print("  ✗ MetricDepthEngine did not load extrinsics from calib dict")
    print("=" * 65)


if __name__ == "__main__":
    main()
