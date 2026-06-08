"""
PRISM — ROS2 Bag Loader
========================
Pure-Python reader for ROS2 SQLite (.db3) bag files.
No ROS2, no rosbags, no metadata.yaml required.
Uses SQLite + hand-written CDR decoder for the three message types we need.

Topic resolution (in priority order):
    1. Preferred names (TOPIC_IMAGE etc.) — exact match
    2. Auto-detection by ROS2 message type — works with any topic naming convention

Detected message types:
    sensor_msgs/msg/Image       → BGR numpy frames
    sensor_msgs/msg/CameraInfo  → CameraIntrinsics (K matrix)
    sensor_msgs/msg/PointCloud2 → (N,4) float32 numpy arrays [x,y,z,intensity]
    sensor_msgs/msg/Imu         → IMU data (not currently used)

Zero extra dependencies beyond numpy, opencv, sqlite3 (all already installed).
"""

import sqlite3
import struct
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
from prism.utils.common import get_logger

logger = get_logger("ROS2BagLoader")

# Preferred topic names (tried first)
TOPIC_IMAGE   = "/usb_cam/image_raw"
TOPIC_CAMINFO = "/usb_cam/camera_info"
TOPIC_LIDAR   = "/velodyne_points"
TOPIC_IMU     = "/imu/data"

# Message type substrings for auto-detection fallback
_MTYPE_IMAGE   = "sensor_msgs/msg/Image"
_MTYPE_CAMINFO = "sensor_msgs/msg/CameraInfo"
_MTYPE_LIDAR   = "sensor_msgs/msg/PointCloud2"
_MTYPE_IMU     = "sensor_msgs/msg/Imu"

# Image topic name fragments that indicate a colour/raw camera (not depth/IR)
_IMAGE_PREFER  = ("color", "rgb", "raw", "image_raw", "usb_cam", "camera", "cv_cam")
_IMAGE_REJECT  = ("depth", "depth_image", "infrared", "ir", "aligned")


def _resolve_topic(
    topics_dict: dict,
    preferred_name: str,
    msg_type_substr: str,
    prefer_hints: tuple = (),
    reject_hints: tuple = (),
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the best matching topic ID + name.

    Resolution order:
      1. Exact match on preferred_name
      2. Type match, ranked by prefer_hints, filtered by reject_hints
      3. First type match with no preference

    topics_dict: {id: (name, type_string)}
    """
    # 1 — exact preferred name
    for tid, (name, typ) in topics_dict.items():
        if name == preferred_name:
            return tid, name

    # 2+3 — type-based fallback
    candidates = [
        (tid, name, typ)
        for tid, (name, typ) in topics_dict.items()
        if msg_type_substr.lower() in typ.lower()
    ]
    if not candidates:
        return None, None

    # Filter rejects
    if reject_hints:
        filtered = [(tid, n, t) for tid, n, t in candidates
                    if not any(h in n.lower() for h in reject_hints)]
        if filtered:
            candidates = filtered

    # Rank by prefer hints (first hint to match wins)
    for hint in prefer_hints:
        for tid, name, typ in candidates:
            if hint in name.lower():
                return tid, name

    # Fall back to first candidate
    tid, name, _ = candidates[0]
    return tid, name


# ── Minimal CDR reader ────────────────────────────────────────────────────────

class CDRReader:
    """
    Reads ROS2 CDR-encoded messages from raw bytes.
    CDR wire format: 4-byte encapsulation header + message payload.
    All primitives aligned to their own size.
    """

    def __init__(self, data: bytes):
        # CDR header: bytes 0-3 (encapsulation kind + padding)
        # byte[1] == 0x01 → little-endian
        self._le   = (data[1] == 0x01) if len(data) > 1 else True
        self._buf  = data
        self._pos  = 4   # skip CDR header

    # ── alignment ────────────────────────────────────────────────────────────

    def _align(self, n: int):
        rem = self._pos % n
        if rem:
            self._pos += n - rem

    # ── primitives ────────────────────────────────────────────────────────────

    def u8(self) -> int:
        v = self._buf[self._pos]
        self._pos += 1
        return v

    def u32(self) -> int:
        self._align(4)
        fmt = "<I" if self._le else ">I"
        v = struct.unpack_from(fmt, self._buf, self._pos)[0]
        self._pos += 4
        return v

    def i32(self) -> int:
        self._align(4)
        fmt = "<i" if self._le else ">i"
        v = struct.unpack_from(fmt, self._buf, self._pos)[0]
        self._pos += 4
        return v

    def f64(self) -> float:
        self._align(8)
        fmt = "<d" if self._le else ">d"
        v = struct.unpack_from(fmt, self._buf, self._pos)[0]
        self._pos += 8
        return v

    def string(self) -> str:
        length = self.u32()
        raw = self._buf[self._pos:self._pos + length]
        self._pos += length
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def bytes_array(self, count: int) -> bytes:
        v = self._buf[self._pos:self._pos + count]
        self._pos += count
        return bytes(v)

    def f64_array(self, count: int) -> list:
        self._align(8)
        fmt = ("<" if self._le else ">") + f"{count}d"
        v = list(struct.unpack_from(fmt, self._buf, self._pos))
        self._pos += 8 * count
        return v

    # ── ROS2 std_msgs/Header ──────────────────────────────────────────────────

    def read_header(self):
        """Read std_msgs/Header → (sec, nanosec, frame_id)"""
        sec       = self.i32()
        nanosec   = self.u32()
        frame_id  = self.string()
        return sec, nanosec, frame_id

    # ── sensor_msgs/PointField ────────────────────────────────────────────────

    def read_point_field(self):
        name     = self.string()
        offset   = self.u32()
        datatype = self.u8()
        self._align(4)
        count    = self.u32()
        return {"name": name, "offset": offset, "datatype": datatype, "count": count}


# ── Message decoders ──────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> Optional[dict]:
    """Decode sensor_msgs/msg/Image → dict with height, width, encoding, data"""
    try:
        import cv2
        r = CDRReader(data)
        r.read_header()                      # stamp + frame_id
        height       = r.u32()
        width        = r.u32()
        encoding     = r.string()
        is_bigendian = r.u8()
        r._align(4)
        step         = r.u32()
        n_bytes      = r.u32()              # sequence length for uint8[]
        raw          = r.bytes_array(n_bytes)

        enc = encoding.lower()
        arr = np.frombuffer(raw, dtype=np.uint8)

        if enc == "rgb8":
            img = arr.reshape(height, width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            return arr.reshape(height, width, 3).copy()
        elif enc in ("rgba8",):
            img = arr.reshape(height, width, 4)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif enc in ("bgra8",):
            img = arr.reshape(height, width, 4)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif enc == "mono8":
            img = arr.reshape(height, width)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif enc in ("yuv422", "yuv422_yuy2"):
            img = arr.reshape(height, width, 2)
            return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        else:
            logger.warning(f"Unknown encoding '{encoding}' — trying rgb8 fallback")
            try:
                img = arr.reshape(height, width, -1)[..., :3]
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
    except Exception as e:
        logger.debug(f"Image decode error: {e}")
        return None


def _decode_camera_info(data: bytes):
    """Decode sensor_msgs/msg/CameraInfo → USBCamIntrinsics or None"""
    try:
        r = CDRReader(data)
        r.read_header()
        height = r.u32()
        width  = r.u32()
        r.string()                 # distortion_model
        d_len = r.u32()
        for _ in range(d_len):
            r.f64()
        k = r.f64_array(9)        # 3×3 intrinsic matrix (row-major)
        K = np.array(k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] < 1.0:
            return None
        return USBCamIntrinsics(K, width, height)
    except Exception as e:
        logger.debug(f"CameraInfo decode error: {e}")
        return None


def _decode_pointcloud2(data: bytes) -> Optional[np.ndarray]:
    """Decode sensor_msgs/msg/PointCloud2 → (N,4) float32 [x,y,z,intensity]"""
    try:
        r = CDRReader(data)
        r.read_header()
        height     = r.u32()
        width      = r.u32()
        n_fields   = r.u32()
        fields     = [r.read_point_field() for _ in range(n_fields)]
        is_bigend  = r.u8()
        r._align(4)
        point_step = r.u32()
        row_step   = r.u32()
        n_bytes    = r.u32()
        raw        = r.bytes_array(n_bytes)

        field_map  = {f["name"]: f["offset"] for f in fields}
        fx = field_map.get("x")
        fy = field_map.get("y")
        fz = field_map.get("z")
        fi = field_map.get("intensity")

        if fx is None or fy is None or fz is None:
            return None

        n_pts  = height * width
        endian = ">" if is_bigend else "<"
        fmt    = endian + "f4"

        raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_pts, point_step)
        xs  = raw_arr[:, fx:fx+4].copy().view(fmt).reshape(-1)
        ys  = raw_arr[:, fy:fy+4].copy().view(fmt).reshape(-1)
        zs  = raw_arr[:, fz:fz+4].copy().view(fmt).reshape(-1)
        ins = (raw_arr[:, fi:fi+4].copy().view(fmt).reshape(-1)
               if fi is not None else np.zeros(n_pts, np.float32))

        cloud = np.stack([xs, ys, zs, ins], axis=1).astype(np.float32)
        return cloud[np.isfinite(cloud[:, :3]).all(axis=1)]

    except Exception as e:
        logger.debug(f"PointCloud2 decode error: {e}")
        return None


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


# ── SQLite bag reader ─────────────────────────────────────────────────────────

class ROS2BagLoader:
    """
    Reads a ROS2 SQLite (.db3) bag file directly.
    No metadata.yaml, no rosbags, no ROS2 installation needed.
    """

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {self.bag_path}")
        self._intrinsics: Optional[USBCamIntrinsics] = None
        logger.info(f"ROS2BagLoader: {self.bag_path.name}")

    def _load_topics(self, conn: sqlite3.Connection) -> dict:
        rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def get_bag_info(self) -> dict:
        conn = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)
        bounds = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM messages"
        ).fetchone()
        conn.close()
        duration_s = (bounds[1] - bounds[0]) * 1e-9 if bounds[0] else 0

        # Resolve which topics we'll actually use
        img_id,  img_name  = _resolve_topic(topics, TOPIC_IMAGE,   _MTYPE_IMAGE,
                                            _IMAGE_PREFER, _IMAGE_REJECT)
        cam_id,  cam_name  = _resolve_topic(topics, TOPIC_CAMINFO, _MTYPE_CAMINFO)
        lid_id,  lid_name  = _resolve_topic(topics, TOPIC_LIDAR,   _MTYPE_LIDAR)
        imu_id,  imu_name  = _resolve_topic(topics, TOPIC_IMU,     _MTYPE_IMU)

        return {
            "path":           str(self.bag_path),
            "duration_s":     round(duration_s, 2),
            "topics":         [name for _, (name, _) in topics.items()],
            "image_topic":    img_name,
            "caminfo_topic":  cam_name,
            "lidar_topic":    lid_name,
            "imu_topic":      imu_name,
            "has_camera":     img_id is not None,
            "has_lidar":      lid_id is not None,
            "has_imu":        imu_id is not None,
        }

    def iter_frames(
        self,
        sync_tolerance_s: float = 0.08,
        max_frames: Optional[int] = None,
        start_s: Optional[float] = None,
        end_s:   Optional[float] = None,
    ) -> Iterator[dict]:
        """
        Yield {image, point_cloud, intrinsics, timestamp} dicts.
        Camera frames drive cadence; nearest LiDAR within tolerance is attached.
        """
        conn   = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)

        # Auto-resolve topics by preferred name, then by message type
        img_id,  img_name  = _resolve_topic(topics, TOPIC_IMAGE,   _MTYPE_IMAGE,
                                            _IMAGE_PREFER, _IMAGE_REJECT)
        cam_id,  cam_name  = _resolve_topic(topics, TOPIC_CAMINFO, _MTYPE_CAMINFO)
        lid_id,  lid_name  = _resolve_topic(topics, TOPIC_LIDAR,   _MTYPE_LIDAR)

        has_camera = img_id is not None
        has_lidar  = lid_id is not None
        has_info   = cam_id is not None

        logger.info(
            f"Topics — camera:{has_camera} ({img_name})  "
            f"lidar:{has_lidar} ({lid_name})  "
            f"caminfo:{has_info} ({cam_name})"
        )

        if not has_camera and not has_lidar:
            logger.error("Bag has neither camera nor LiDAR (checked by message type)")
            logger.error(f"All topics: {[n for _, (n, _) in topics.items()]}")
            conn.close()
            return

        # Time bounds
        bounds = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM messages"
        ).fetchone()
        bag_start_ns  = bounds[0]
        filter_start  = bag_start_ns + int(start_s * 1e9) if start_s else 0
        filter_end    = bag_start_ns + int(end_s   * 1e9) if end_s   else 2**63 - 1

        wanted_ids = set(i for i in [img_id, cam_id, lid_id] if i is not None)

        placeholders = ",".join("?" * len(wanted_ids))
        query = (
            f"SELECT topic_id, timestamp, data FROM messages "
            f"WHERE topic_id IN ({placeholders}) "
            f"AND timestamp >= ? AND timestamp <= ? "
            f"ORDER BY timestamp ASC"
        )
        params = list(wanted_ids) + [filter_start, filter_end]

        intrinsics_cached = self._intrinsics
        lidar_buffer: list = []
        frame_count = 0

        DECODERS = {
            img_id:  ("image",   _decode_image),
            cam_id:  ("caminfo", _decode_camera_info),
            lid_id:  ("lidar",   _decode_pointcloud2),
        }
        # Remove None keys (topics not found)
        DECODERS = {k: v for k, v in DECODERS.items() if k is not None}

        for topic_id, ts_ns, raw_data in conn.execute(query, params):
            ts = ts_ns * 1e-9
            entry = DECODERS.get(topic_id)
            if entry is None:
                continue
            kind, decoder = entry

            result = decoder(bytes(raw_data))
            if result is None:
                continue

            if kind == "caminfo" and intrinsics_cached is None:
                intrinsics_cached = result
                self._intrinsics  = result
                logger.info(f"Intrinsics: {result}")

            elif kind == "lidar":
                if len(result) > 0:
                    lidar_buffer.append((ts, result))
                    if len(lidar_buffer) > 10:
                        lidar_buffer.pop(0)

            elif kind == "image":
                matched_cloud = None
                if lidar_buffer:
                    diffs = [abs(lts - ts) for lts, _ in lidar_buffer]
                    best  = int(np.argmin(diffs))
                    if diffs[best] <= sync_tolerance_s:
                        matched_cloud = lidar_buffer[best][1]

                yield {
                    "timestamp":   ts,
                    "image":       result,
                    "point_cloud": matched_cloud,
                    "intrinsics":  intrinsics_cached,
                    "imu":         None,
                }
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    conn.close()
                    return

        conn.close()

    def iter_lidar_only(self, max_frames: Optional[int] = None) -> Iterator[dict]:
        conn   = sqlite3.connect(str(self.bag_path))
        topics = self._load_topics(conn)
        lidar_id = next(
            (tid for tid, (name, _) in topics.items() if name == TOPIC_LIDAR), None
        )
        if lidar_id is None:
            logger.warning("No LiDAR topic in bag")
            conn.close()
            return
        count = 0
        for _, ts_ns, raw_data in conn.execute(
            "SELECT topic_id, timestamp, data FROM messages "
            "WHERE topic_id=? ORDER BY timestamp", (lidar_id,)
        ):
            cloud = _decode_pointcloud2(bytes(raw_data))
            if cloud is not None and len(cloud) > 0:
                yield {"timestamp": ts_ns * 1e-9, "point_cloud": cloud}
                count += 1
                if max_frames and count >= max_frames:
                    break
        conn.close()

    @property
    def intrinsics(self) -> Optional[USBCamIntrinsics]:
        return self._intrinsics
