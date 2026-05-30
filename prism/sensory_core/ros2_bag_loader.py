"""
PRISM — ROS2 Bag Loader
========================
Pure-Python reader for ROS2 SQLite (.db3) bag files.
Reads directly via SQLite — no metadata.yaml, no ROS2 installation required.
Uses rosbags only for CDR message deserialization.

Supported topics:
    /usb_cam/image_raw      → BGR numpy frames
    /usb_cam/camera_info    → CameraIntrinsics (K matrix)
    /velodyne_points        → (N,4) float32 numpy arrays  [x, y, z, intensity]
    /imu/data               → raw IMU dict (optional)

Install dependency (once):
    pip install rosbags --break-system-packages

Usage:
    loader = ROS2BagLoader("/home/user/Downloads/test1-003.db3")
    for frame in loader.iter_frames():
        img   = frame["image"]        # BGR ndarray or None
        cloud = frame["point_cloud"]  # (N,4) float32 or None
        K     = frame["intrinsics"]   # USBCamIntrinsics or None
        ts    = frame["timestamp"]    # float seconds
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from prism.utils.common import get_logger

logger = get_logger("ROS2BagLoader")

TOPIC_IMAGE   = "/usb_cam/image_raw"
TOPIC_CAMINFO = "/usb_cam/camera_info"
TOPIC_LIDAR   = "/velodyne_points"
TOPIC_IMU     = "/imu/data"


# ── Intrinsics ────────────────────────────────────────────────────────────────

class USBCamIntrinsics:
    def __init__(self, K: np.ndarray, width: int, height: int):
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.width  = width
        self.height = height
        self.K      = K

    def to_calibration_dict(self) -> dict:
        return {"camera_intrinsic": self.K.tolist()}

    def __repr__(self):
        return (f"USBCamIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")


# ── Image decoding ────────────────────────────────────────────────────────────

def _decode_image_msg(msg) -> Optional[np.ndarray]:
    import cv2
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    data = bytes(msg.data)
    try:
        if enc == "rgb8":
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
        elif enc in ("rgba8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif enc in ("bgra8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif enc == "mono8":
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif enc in ("yuv422", "yuv422_yuy2"):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 2)
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUYV)
        else:
            logger.warning(f"Unknown encoding '{msg.encoding}' — trying rgb8")
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, -1)
            return cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Image decode failed ({enc} {h}x{w}): {e}")
        return None


# ── PointCloud2 decoding ──────────────────────────────────────────────────────

def _decode_pointcloud2(msg) -> Optional[np.ndarray]:
    fields = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"}.issubset(fields.keys()):
        return None
    fx = fields["x"].offset
    fy = fields["y"].offset
    fz = fields["z"].offset
    fi = fields["intensity"].offset if "intensity" in fields else None
    point_step = int(msg.point_step)
    n_pts = int(msg.height) * int(msg.width)
    try:
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n_pts, point_step)
        endian = ">" if msg.is_bigendian else "<"
        fmt = endian + "f4"
        xs = raw[:, fx:fx+4].copy().view(fmt).reshape(-1)
        ys = raw[:, fy:fy+4].copy().view(fmt).reshape(-1)
        zs = raw[:, fz:fz+4].copy().view(fmt).reshape(-1)
        ins = raw[:, fi:fi+4].copy().view(fmt).reshape(-1) if fi is not None else np.zeros(n_pts, np.float32)
        cloud = np.stack([xs, ys, zs, ins], axis=1).astype(np.float32)
        return cloud[np.isfinite(cloud[:, :3]).all(axis=1)]
    except Exception as e:
        logger.warning(f"PointCloud2 decode failed: {e}")
        return None


# ── CameraInfo decoding ───────────────────────────────────────────────────────

def _decode_camera_info(msg) -> Optional[USBCamIntrinsics]:
    try:
        K = np.array(list(msg.k), dtype=np.float64).reshape(3, 3)
        if K[0, 0] < 1.0:
            return None
        return USBCamIntrinsics(K, int(msg.width), int(msg.height))
    except Exception as e:
        logger.warning(f"CameraInfo decode failed: {e}")
        return None


# ── SQLite bag reader ─────────────────────────────────────────────────────────

class ROS2BagLoader:
    """
    Reads a ROS2 SQLite (.db3) bag file directly via SQLite.
    No metadata.yaml required. Rosbags used only for CDR deserialization.
    """

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {self.bag_path}")
        self._intrinsics: Optional[USBCamIntrinsics] = None
        logger.info(f"ROS2BagLoader: {self.bag_path.name}")

    def _get_deserializer(self):
        try:
            from rosbags.serde import deserialize_cdr
            return deserialize_cdr
        except ImportError:
            raise ImportError(
                "rosbags not installed.\n"
                "Run: pip install rosbags --break-system-packages"
            )

    def _load_topics(self, conn: sqlite3.Connection) -> dict:
        """Return {topic_id: (name, msgtype)} from the bag."""
        rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def get_bag_info(self) -> dict:
        conn = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)
        counts = conn.execute(
            "SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"
        ).fetchall()
        count_map = {tid: cnt for tid, cnt in counts}
        duration_row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM messages"
        ).fetchone()
        conn.close()
        duration_s = (duration_row[1] - duration_row[0]) * 1e-9 if duration_row[0] else 0
        return {
            "path":       str(self.bag_path),
            "duration_s": round(duration_s, 2),
            "topics":     {
                name: {"type": mtype, "count": count_map.get(tid, 0)}
                for tid, (name, mtype) in topics.items()
            },
        }

    def iter_frames(
        self,
        sync_tolerance_s: float = 0.08,
        max_frames: Optional[int] = None,
        start_s: Optional[float] = None,
        end_s:   Optional[float] = None,
    ) -> Iterator[dict]:
        """
        Yield synchronized {image, point_cloud, intrinsics, timestamp} dicts.
        Camera frames drive cadence; nearest LiDAR scan within tolerance is attached.
        """
        deserialize_cdr = self._get_deserializer()
        conn = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)

        # Reverse map: name → (topic_id, msgtype)
        name_to_id = {name: (tid, mtype) for tid, (name, mtype) in topics.items()}

        has_camera = TOPIC_IMAGE   in name_to_id
        has_lidar  = TOPIC_LIDAR   in name_to_id
        has_info   = TOPIC_CAMINFO in name_to_id

        logger.info(f"Topics — camera:{has_camera}  lidar:{has_lidar}  caminfo:{has_info}")

        if not has_camera and not has_lidar:
            logger.error("Bag has neither camera nor LiDAR")
            conn.close()
            return

        # Time bounds
        bounds = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
        bag_start_ns = bounds[0]
        filter_start = bag_start_ns + int(start_s * 1e9) if start_s else 0
        filter_end   = bag_start_ns + int(end_s   * 1e9) if end_s   else 2**63

        # Build topic_id filter for our topics of interest
        wanted_ids = set()
        for t in [TOPIC_IMAGE, TOPIC_CAMINFO, TOPIC_LIDAR]:
            if t in name_to_id:
                wanted_ids.add(name_to_id[t][0])

        placeholders = ",".join("?" * len(wanted_ids))
        query = (
            f"SELECT topic_id, timestamp, data FROM messages "
            f"WHERE topic_id IN ({placeholders}) "
            f"AND timestamp >= ? AND timestamp <= ? "
            f"ORDER BY timestamp ASC"
        )
        params = list(wanted_ids) + [filter_start, filter_end]

        intrinsics_cached = self._intrinsics
        lidar_buffer: list = []   # [(ts_s, cloud), ...]
        frame_count = 0

        for topic_id, ts_ns, raw_data in conn.execute(query, params):
            name, msgtype = topics[topic_id]
            ts = ts_ns * 1e-9

            try:
                msg = deserialize_cdr(bytes(raw_data), msgtype)
            except Exception as e:
                logger.debug(f"CDR error on {name}: {e}")
                continue

            if name == TOPIC_CAMINFO and intrinsics_cached is None:
                intrinsics_cached = _decode_camera_info(msg)
                if intrinsics_cached:
                    self._intrinsics = intrinsics_cached
                    logger.info(f"Intrinsics: {intrinsics_cached}")

            elif name == TOPIC_LIDAR:
                cloud = _decode_pointcloud2(msg)
                if cloud is not None and len(cloud) > 0:
                    lidar_buffer.append((ts, cloud))
                    if len(lidar_buffer) > 10:
                        lidar_buffer.pop(0)

            elif name == TOPIC_IMAGE:
                image = _decode_image_msg(msg)
                if image is None:
                    continue

                # Sync nearest LiDAR
                matched_cloud = None
                if lidar_buffer:
                    diffs = [abs(lts - ts) for lts, _ in lidar_buffer]
                    best  = int(np.argmin(diffs))
                    if diffs[best] <= sync_tolerance_s:
                        matched_cloud = lidar_buffer[best][1]

                yield {
                    "timestamp":   ts,
                    "image":       image,
                    "point_cloud": matched_cloud,
                    "intrinsics":  intrinsics_cached,
                    "imu":         None,
                }

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    conn.close()
                    return

        conn.close()

        # LiDAR-only bag
        if not has_camera and lidar_buffer:
            for ts, cloud in lidar_buffer:
                yield {"timestamp": ts, "image": None,
                       "point_cloud": cloud, "intrinsics": None, "imu": None}

    def iter_lidar_only(self, max_frames: Optional[int] = None) -> Iterator[dict]:
        """Stream LiDAR scans without camera sync."""
        deserialize_cdr = self._get_deserializer()
        conn = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)
        lidar_entry = next(
            ((tid, mtype) for tid, (name, mtype) in topics.items() if name == TOPIC_LIDAR),
            None
        )
        if lidar_entry is None:
            logger.warning("No LiDAR topic in bag")
            conn.close()
            return
        tid, mtype = lidar_entry
        count = 0
        for _, ts_ns, raw_data in conn.execute(
            "SELECT topic_id, timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (tid,)
        ):
            try:
                msg   = deserialize_cdr(bytes(raw_data), mtype)
                cloud = _decode_pointcloud2(msg)
            except Exception:
                continue
            if cloud is not None and len(cloud) > 0:
                yield {"timestamp": ts_ns * 1e-9, "point_cloud": cloud}
                count += 1
                if max_frames and count >= max_frames:
                    break
        conn.close()

    @property
    def intrinsics(self) -> Optional[USBCamIntrinsics]:
        return self._intrinsics
