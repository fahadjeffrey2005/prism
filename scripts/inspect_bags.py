"""
PRISM — Bag Inspector
Scans a directory for .db3 files and prints topics, duration, and pipeline support.

Usage:
    python scripts/inspect_bags.py
    python scripts/inspect_bags.py ~/Downloads/some_other_folder
"""

import sys
import os
import glob
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from prism.sensory_core.ros2_bag_loader import ROS2BagLoader

CAMERA_TOPIC = "/usb_cam/image_raw"
LIDAR_TOPIC  = "/velodyne_points"
IMU_TOPIC    = "/imu/data"

def inspect(bag_dir: str):
    bags = sorted(glob.glob(os.path.join(bag_dir, "**", "*.db3"), recursive=True))
    if not bags:
        print(f"No .db3 files found in: {bag_dir}")
        return

    print(f"\nFound {len(bags)} bag(s) in: {bag_dir}")
    print("=" * 80)
    print(f"{'File':<45} {'Duration':>8}  {'Camera':>6}  {'LiDAR':>5}  {'IMU':>4}  Pipeline")
    print("-" * 80)

    for b in bags:
        name = os.path.basename(b)
        try:
            info    = ROS2BagLoader(b).get_bag_info()
            topics  = info["topics"]
            dur     = info["duration_s"]
            has_cam = CAMERA_TOPIC  in topics
            has_lid = LIDAR_TOPIC   in topics
            has_imu = IMU_TOPIC     in topics
            if has_cam and has_lid:
                pipeline = "FULL  (camera + LiDAR)"
            elif has_cam:
                pipeline = "CAM ONLY"
            elif has_lid:
                pipeline = "LIDAR ONLY"
            else:
                pipeline = "unknown topics"
            print(f"{name:<45} {dur:>7.1f}s  {'YES':>6}  {'YES' if has_lid else 'NO':>5}  {'YES' if has_imu else 'NO':>4}  {pipeline}")
        except Exception as e:
            print(f"{name:<45}  ERROR: {e}")

    print("=" * 80)
    print("\nTo run a bag:")
    print("  python scripts/run_bag_dashboard.py <path/to/file.db3> --no-depth --save --no-show")


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/ROSBAG FILES")
    inspect(bag_dir)
