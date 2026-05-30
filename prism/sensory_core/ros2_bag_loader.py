"""
PRISM — ROS2 Bag Loader
========================
Pure-Python reader for ROS2 SQLite (.db3) bag files.
No ROS2 installation required — uses the `rosbags` library.

Supported topics:
    /usb_cam/image_raw      → BGR numpy frames
    /usb_cam/camera_info    → CameraIntrinsics (K matrix)
    /velodyne_points        → (N,4) float32 numpy arrays  [x, y, z, intensity]
    /imu/data               → raw IMU dict (optional)

Usage:
    loader = ROS2BagLoader(bag_path)
    for frame in loader.iter_frames(sync_tolerance_s=0.05):
        img   = frame["image"]        # BGR ndarray or None
        cloud = frame["point_cloud"]  # (N,4) float32 or None
        K     = frame["intrinsics"]   # CameraIntrinsics or None
        ts    = frame["timestamp"]    # float seconds

Install dependency (once):
    pip install rosbags --break-system-packages
"""

import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from prism.utils.common import get_logger

logger = get_logger("ROS2BagLoader")

# Topics we care about
TOPIC_IMAGE   = "/usb_cam/image_raw"
TOPIC_CAMINFO = "/usb_cam/camera_info"
TOPIC_LIDAR   = "/velodyne_points"
TOPIC_IMU     = "/imu/data"

# Supported raw image encodings
_ENCODING_CHANNELS = {
    "rgb8":    (3, np.uint8),
    "bgr8":    (3, np.uint8),
    "bgra8":   (4, np.uint8),
    "rgba8":   (4, np.uint8),
    "mono8":   (1, np.uint8),
    "mono16":  (1, np.uint16),
    "16UC1":   (1, np.uint16),
    "yuv422":  (2, np.uint8),   # YUYV — needs cv2 conversion
    "yuv422_yuy2": (2, np.uint8),
}


# ── Intrinsics dataclass (mirrors metric_depth.CameraIntrinsics) ──────────────

class USBCamIntrinsics:
    """
    Camera intrinsics extracted from /usb_cam/camera_info.
    Mirrors the interface expected by MetricDepthEngine.
    """

    def __init__(self, K: np.ndarray, width: int, height: int):
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.width  = width
        self.height = height
        self.K      = K

    def to_calibration_dict(self) -> dict:
        """
        Return a dict compatible with MetricDepthEngine.update_intrinsics().
        No extrinsics available from USB cam alone — caller must supply them.
        """
        return {
            "camera_intrinsic": self.K.tolist(),
        }

    def __repr__(self) -> str:
        return (f"USBCamIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")


# ── Image decoding ────────────────────────────────────────────────────────────

def _decode_image_msg(msg) -> Optional[np.ndarray]:
    """
    Decode a sensor_msgs/msg/Image ROS2 message to a BGR numpy array.
    Handles the most common encodings produced by usb_cam driver.
    """
    import cv2

    encoding = msg.encoding.lower()
    h, w     = int(msg.height), int(msg.width)
    step     = int(msg.step)      # bytes per row
    data     = bytes(msg.data)

    try:
        if encoding in ("rgb8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        elif encoding in ("bgr8",):
            return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()

        elif encoding in ("rgba8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        elif encoding in ("bgra8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        elif encoding in ("mono8",):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        elif encoding in ("mono16", "16uc1"):
            arr = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            arr8 = (arr >> 8).astype(np.uint8)
            return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)

        elif encoding in ("yuv422", "yuv422_yuy2"):
            # YUYV packed — 2 bytes per pixel
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 2)
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUYV)

        else:
            logger.warning(f"Unknown image encoding '{msg.encoding}' — attempting rgb8 fallback")
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, -1)
            if arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr[:, :, :3].copy()

    except Exception as e:
        logger.warning(f"Image decode failed ({msg.encoding}, {h}x{w}): {e}")
        return None


# ── PointCloud2 decoding ──────────────────────────────────────────────────────

def _decode_pointcloud2(msg) -> Optional[np.ndarray]:
    """
    Decode sensor_msgs/msg/PointCloud2 to (N, 4) float32 [x, y, z, intensity].
    Handles both big-endian and little-endian, dense and non-dense clouds.
    """
    import struct

    fields = {f.name: f for f in msg.fields}
    required = {"x", "y", "z"}
    if not required.issubset(fields.keys()):
        logger.warning(f"PointCloud2 missing required fields: {required - fields.keys()}")
        return None

    point_step = int(msg.point_step)
    row_step   = int(msg.row_step)
    height     = int(msg.height)
    width      = int(msg.width)
    data       = bytes(msg.data)
    is_bigend  = bool(msg.is_bigendian)

    fx = fields["x"].offset
    fy = fields["y"].offset
    fz = fields["z"].offset
    fi = fields["intensity"].offset if "intensity" in fields else None

    endian = ">" if is_bigend else "<"
    fmt_f  = endian + "f"
    n_pts  = height * width

    # Fast path: if fields are packed as xyzI float32 at start of point_step
    # and point_step divides evenly, use numpy structured array
    try:
        # Build dtype from field offsets
        dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if fi is not None:
            dtype_fields.append(("i", "<f4"))

        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_pts, point_step)
        xs = raw[:, fx:fx+4].view("<f4" if not is_bigend else ">f4").reshape(-1)
        ys = raw[:, fy:fy+4].view("<f4" if not is_bigend else ">f4").reshape(-1)
        zs = raw[:, fz:fz+4].view("<f4" if not is_bigend else ">f4").reshape(-1)

        if fi is not None:
            ins = raw[:, fi:fi+4].view("<f4" if not is_bigend else ">f4").reshape(-1)
        else:
            ins = np.zeros(n_pts, dtype=np.float32)

        cloud = np.stack([xs, ys, zs, ins], axis=1).astype(np.float32)

        # Filter out NaN/Inf points
        valid = np.isfinite(cloud[:, :3]).all(axis=1)
        return cloud[valid]

    except Exception as e:
        logger.warning(f"PointCloud2 fast decode failed: {e} — falling back to slow path")

    # Slow path: parse point by point
    result = []
    for i in range(n_pts):
        offset = i * point_step
        try:
            x = struct.unpack_from(fmt_f, data, offset + fx)[0]
            y = struct.unpack_from(fmt_f, data, offset + fy)[0]
            z = struct.unpack_from(fmt_f, data, offset + fz)[0]
            intensity = struct.unpack_from(fmt_f, data, offset + fi)[0] if fi is not None else 0.0
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                result.append([x, y, z, intensity])
        except Exception:
            continue

    return np.array(result, dtype=np.float32) if result else None


# ── Camera info decoding ──────────────────────────────────────────────────────

def _decode_camera_info(msg) -> Optional[USBCamIntrinsics]:
    """
    Extract intrinsics from sensor_msgs/msg/CameraInfo message.
    The K field is a 9-element row-major 3x3 matrix.
    """
    try:
        K = np.array(list(msg.k), dtype=np.float64).reshape(3, 3)
        if K[0, 0] < 1.0:
            return None  # Invalid / unset calibration
        return USBCamIntrinsics(K, int(msg.width), int(msg.height))
    except Exception as e:
        logger.warning(f"CameraInfo decode failed: {e}")
        return None


# ── Main loader ───────────────────────────────────────────────────────────────

class ROS2BagLoader:
    """
    Reads a ROS2 SQLite (.db3) bag file and streams synchronized frames.

    Each yielded frame dict contains:
        timestamp    : float   — seconds since epoch
        image        : ndarray — BGR (H,W,3) uint8, or None
        point_cloud  : ndarray — (N,4) float32 [x,y,z,intensity], or None
        intrinsics   : USBCamIntrinsics or None (once received, cached)
        imu          : dict or None

    Synchronization:
        Camera frames drive the cadence. Each yielded frame is the camera
        image with the nearest LiDAR scan within sync_tolerance_s.
        If no camera topic exists (e.g. ros2_bag), LiDAR drives the cadence.
    """

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {self.bag_path}")
        self._intrinsics: Optional[USBCamIntrinsics] = None
        logger.info(f"ROS2BagLoader initialised: {self.bag_path.name}")

    def _open_reader(self):
        """Open rosbags reader — raises ImportError with install hint if missing."""
        try:
            from rosbags.rosbag2 import Reader
            from rosbags.serde import deserialize_cdr
            return Reader, deserialize_cdr
        except ImportError:
            raise ImportError(
                "rosbags library not installed.\n"
                "Run: pip install rosbags --break-system-packages"
            )

    def available_topics(self) -> dict:
        """Return {topic_name: message_type} for all topics in the bag."""
        Reader, _ = self._open_reader()
        with Reader(self.bag_path) as reader:
            return {
                conn.topic: conn.msgtype
                for conn in reader.connections
            }

    def iter_frames(
        self,
        sync_tolerance_s: float = 0.05,
        max_frames: Optional[int] = None,
        start_s: Optional[float] = None,
        end_s:   Optional[float] = None,
    ) -> Iterator[dict]:
        """
        Yield synchronized {image, point_cloud, intrinsics, timestamp} dicts.

        Args:
            sync_tolerance_s : max time gap (s) between camera and LiDAR for sync
            max_frames       : stop after this many camera frames
            start_s          : skip messages before this many seconds into the bag
            end_s            : stop after this many seconds into the bag
        """
        Reader, deserialize_cdr = self._open_reader()

        with Reader(self.bag_path) as reader:
            # Map topic → connection objects
            conns_by_topic: dict = {}
            for conn in reader.connections:
                conns_by_topic.setdefault(conn.topic, []).append(conn)

            has_camera = TOPIC_IMAGE   in conns_by_topic
            has_lidar  = TOPIC_LIDAR   in conns_by_topic
            has_info   = TOPIC_CAMINFO in conns_by_topic

            logger.info(
                f"Bag topics: camera={has_camera}  lidar={has_lidar}  "
                f"camera_info={has_info}"
            )

            if not has_camera and not has_lidar:
                logger.error("Bag has neither camera nor LiDAR — nothing to process")
                return

            # Determine bag time bounds
            bag_start_ns = reader.start_time   # nanoseconds
            bag_end_ns   = reader.end_time

            filter_start_ns = (bag_start_ns + int(start_s * 1e9)) if start_s else bag_start_ns
            filter_end_ns   = (bag_start_ns + int(end_s   * 1e9)) if end_s   else bag_end_ns

            # ── Collect all messages into per-topic buffers ──────────────────
            # We need to sync camera + LiDAR, so buffer both first.
            # For large bags this is memory-intensive; we process in a streaming
            # fashion by keeping only a small LiDAR buffer.

            lidar_buffer: list  = []   # [(timestamp_s, cloud), ...]
            cam_buffer:   list  = []   # [(timestamp_s, image), ...]
            info_cached = self._intrinsics

            lidar_conns  = conns_by_topic.get(TOPIC_LIDAR,   [])
            cam_conns    = conns_by_topic.get(TOPIC_IMAGE,   [])
            info_conns   = conns_by_topic.get(TOPIC_CAMINFO, [])

            # Select connections
            active_conns = lidar_conns + cam_conns + info_conns

            frame_count = 0

            for conn, timestamp_ns, rawdata in reader.messages(connections=active_conns):
                if timestamp_ns < filter_start_ns or timestamp_ns > filter_end_ns:
                    continue

                ts = timestamp_ns * 1e-9

                try:
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                except Exception as e:
                    logger.debug(f"Deserialise error on {conn.topic}: {e}")
                    continue

                # ── camera_info → cache intrinsics ──────────────────────────
                if conn.topic == TOPIC_CAMINFO and info_cached is None:
                    info_cached = _decode_camera_info(msg)
                    if info_cached is not None:
                        self._intrinsics = info_cached
                        logger.info(f"Camera intrinsics: {info_cached}")

                # ── LiDAR → buffer ──────────────────────────────────────────
                elif conn.topic == TOPIC_LIDAR:
                    cloud = _decode_pointcloud2(msg)
                    if cloud is not None and len(cloud) > 0:
                        lidar_buffer.append((ts, cloud))
                        # Keep buffer bounded (last 10 scans is enough for sync)
                        if len(lidar_buffer) > 10:
                            lidar_buffer.pop(0)

                # ── Camera image → sync + yield ─────────────────────────────
                elif conn.topic == TOPIC_IMAGE:
                    image = _decode_image_msg(msg)
                    if image is None:
                        continue

                    # Find nearest LiDAR scan within tolerance
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
                        "intrinsics":  info_cached,
                        "imu":         None,
                    }

                    frame_count += 1
                    if max_frames and frame_count >= max_frames:
                        return

            # ── LiDAR-only bag (no camera) ───────────────────────────────────
            if not has_camera and has_lidar and lidar_buffer:
                logger.info("No camera topic — yielding LiDAR-only frames")
                for ts, cloud in lidar_buffer:
                    yield {
                        "timestamp":   ts,
                        "image":       None,
                        "point_cloud": cloud,
                        "intrinsics":  None,
                        "imu":         None,
                    }

    def iter_lidar_only(
        self,
        max_frames: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Stream only LiDAR point clouds (no image sync).
        Useful for testing the LiDAR pipeline independently.
        """
        Reader, deserialize_cdr = self._open_reader()

        with Reader(self.bag_path) as reader:
            lidar_conns = [c for c in reader.connections if c.topic == TOPIC_LIDAR]
            if not lidar_conns:
                logger.warning("No LiDAR topic in bag")
                return

            count = 0
            for conn, ts_ns, rawdata in reader.messages(connections=lidar_conns):
                try:
                    msg   = deserialize_cdr(rawdata, conn.msgtype)
                    cloud = _decode_pointcloud2(msg)
                except Exception:
                    continue
                if cloud is not None and len(cloud) > 0:
                    yield {"timestamp": ts_ns * 1e-9, "point_cloud": cloud}
                    count += 1
                    if max_frames and count >= max_frames:
                        return

    @property
    def intrinsics(self) -> Optional[USBCamIntrinsics]:
        """Return cached intrinsics (available after first camera_info message)."""
        return self._intrinsics

    def get_bag_info(self) -> dict:
        """Return basic stats about the bag."""
        Reader, _ = self._open_reader()
        with Reader(self.bag_path) as reader:
            duration_s = (reader.end_time - reader.start_time) * 1e-9
            topics = {}
            for conn in reader.connections:
                topics[conn.topic] = conn.msgtype
            return {
                "path":       str(self.bag_path),
                "duration_s": round(duration_s, 2),
                "topics":     topics,
            }
