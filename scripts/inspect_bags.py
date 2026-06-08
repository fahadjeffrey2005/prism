"""
PRISM — Bag Inspector
Scans a directory for .db3 files and prints topics, duration, and pipeline support.
Auto-detects camera/LiDAR topics by message type — works with any topic names.

Usage:
    python scripts/inspect_bags.py
    python scripts/inspect_bags.py ~/Downloads/ROSBAG\ FILES
"""

import sys
import os
import glob
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from prism.sensory_core.ros2_bag_loader import ROS2BagLoader


def inspect(bag_dir: str):
    bags = sorted(glob.glob(os.path.join(bag_dir, "**", "*.db3"), recursive=True))
    if not bags:
        print(f"No .db3 files found in: {bag_dir}")
        return

    print(f"\nFound {len(bags)} bag(s) in: {bag_dir}")
    print("=" * 100)

    for b in bags:
        name = os.path.basename(b)
        try:
            info    = ROS2BagLoader(b).get_bag_info()
            dur     = info["duration_s"]
            has_cam = info["has_camera"]
            has_lid = info["has_lidar"]
            has_imu = info["has_imu"]
            img_t   = info.get("image_topic")   or "—"
            cam_t   = info.get("caminfo_topic") or "—"
            lid_t   = info.get("lidar_topic")   or "—"

            if has_cam and has_lid:   pipeline = "FULL (camera + LiDAR)"
            elif has_cam:             pipeline = "CAM ONLY"
            elif has_lid:             pipeline = "LIDAR ONLY"
            else:                     pipeline = "NO RECOGNISED TOPICS"

            cam_str = f"YES ({img_t})" if has_cam else "NO"
            lid_str = f"YES ({lid_t})" if has_lid else "NO"
            imu_str = f"YES"           if has_imu else "NO"

            print(f"\n{name}  |  {dur:.1f}s  |  {pipeline}")
            print(f"  Camera : {cam_str}")
            print(f"  LiDAR  : {lid_str}")
            print(f"  IMU    : {imu_str}")
            if has_cam and cam_t != "—":
                print(f"  CamInfo: {cam_t}")
            print(f"  All topics: {info['topics']}")
        except Exception as e:
            print(f"\n{name}  ERROR: {e}")

    print("\n" + "=" * 100)
    print("\nTo run a bag:")
    print("  python scripts/run_bag_dashboard.py <path/to/file.db3> --no-depth --save --no-show")
    print("\nTo run on all bags in a folder:")
    print("  for f in ~/Downloads/ROSBAG\\ FILES/*.db3; do")
    print("    python scripts/run_bag_dashboard.py \"$f\" --no-depth --save --no-show")
    print("  done")


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/ROSBAG FILES")
    inspect(bag_dir)
