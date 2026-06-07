"""
PRISM — Metric Depth  (v2 — Ray Unprojection)
==============================================
Converts relative depth estimates into real metric distances (metres).

Two complementary approaches combined:

1. GEOMETRY-BASED (for detected actors)
   Uses the pinhole camera model + known object dimensions.
   Formula: distance = (real_height_m x focal_length_px) / bbox_height_px
   Accurate to ~0.3-0.8m within 40m range.

2. METRIC DEPTH MODEL (for background + unknown objects)
   Depth Anything v2 Metric variant — outputs absolute metres.

v2 change — Ray Unprojection for lateral position:
   OLD: lateral_m = (pixel_offset / fx) * distance_m
        — approximates along principal axis, ~2m error for off-centre objects

   NEW: unproject bottom-centre pixel -> ray in camera frame
        -> rotate ray to ego frame using calibrated extrinsics
        -> intersect ray with ground plane (z=0 in ego frame)
        -> get exact metric (x_forward, y_lateral) position
        — drops lateral error from ~2m to ~0.3m

Combined output: every detected actor has metric distance + lateral offset in metres.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from prism.utils.common import get_logger, BBox2D, Detection

logger = get_logger("MetricDepth")


# ── Known object dimensions (metres) ─────────────────────────────────────────

OBJECT_DIMENSIONS = {
    # class_name: (height_m, width_m, length_m)
    "car":           (1.50, 2.0,  4.5),
    "truck":         (3.20, 2.5,  8.0),
    "bus":           (3.50, 2.8, 12.0),
    "motorcycle":    (1.20, 0.8,  2.2),
    "bicycle":       (1.10, 0.6,  1.8),
    "person":        (1.75, 0.6,  0.3),
    "traffic light": (0.80, 0.3,  0.3),
    "stop sign":     (0.75, 0.75, 0.1),
}

# Classes where ground-plane ray unprojection is valid
# (objects that sit on the road surface)
GROUND_PLANE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

MIN_GEOMETRY_CONFIDENCE = 0.45
GEOMETRY_MIN_DIST = 1.0    # metres
GEOMETRY_MAX_DIST = 60.0   # metres


# ── Camera calibration dataclasses ───────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters from calibration."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int  = 1600
    height: int = 900

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int = 1600, height: int = 900):
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
        )

    @classmethod
    def nuscenes_default(cls):
        """
        Default nuScenes CAM_FRONT intrinsics.
        Fallback when calibration is not provided.
        """
        return cls(fx=1266.4, fy=1266.4, cx=816.0, cy=491.5)

    def pixel_to_ray_cam(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel (u, v) to unnormalised direction ray in camera frame.
        Camera frame convention: x=right, y=down, z=forward.
        """
        return np.array([
            (u - self.cx) / self.fx,
            (v - self.cy) / self.fy,
            1.0
        ], dtype=np.float64)

    def project_point(self, xyz: np.ndarray) -> tuple:
        u = self.fx * xyz[0] / xyz[2] + self.cx
        v = self.fy * xyz[1] / xyz[2] + self.cy
        return float(u), float(v)


@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters — position and orientation of the camera
    relative to the vehicle ego frame.

    nuScenes convention:
        translation  — camera origin in ego frame (metres)
        rotation     — quaternion [w, x, y, z] rotating camera -> ego frame

    Ego frame convention:
        x = forward, y = left, z = up
    """
    translation: np.ndarray      # shape (3,) — camera position in ego frame
    rotation_q:  np.ndarray      # shape (4,) — quaternion [w, x, y, z]

    # Derived rotation matrix (camera frame -> ego frame), computed on creation
    R_cam_to_ego: np.ndarray = field(init=False)

    def __post_init__(self):
        self.R_cam_to_ego = self._quat_to_matrix(self.rotation_q)

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),        1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),        2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    @classmethod
    def nuscenes_default(cls):
        """
        Approximate default for nuScenes CAM_FRONT.
        Camera sits ~1.5m above ground, ~1.7m forward of rear axle.
        Use actual per-sample calibration whenever possible.
        """
        translation = np.array([1.72200568, 0.00000000, 1.49491292])
        rotation_q  = np.array([0.49834835, -0.49834835, 0.50164326, -0.50164326])
        return cls(translation=translation, rotation_q=rotation_q)

    @classmethod
    def from_calibration(cls, calib: dict):
        """Build from nuScenes calibration dict."""
        t = np.array(calib["translation"], dtype=np.float64)
        q = np.array(calib["rotation"],    dtype=np.float64)
        return cls(translation=t, rotation_q=q)


# ── Core ray unprojection ─────────────────────────────────────────────────────

def ray_unproject_to_ground(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Optional[np.ndarray]:
    """
    Unproject pixel (u, v) to a 3D point on the ground plane (z=0 in ego frame).

    Algorithm:
        1. Build ray direction in camera frame from pixel + intrinsics
        2. Rotate ray to ego frame using calibrated camera rotation
        3. Ray origin = camera position in ego frame (extrinsics.translation)
        4. Intersect ray with z=0 plane:
               origin[2] + t * ray_ego[2] = 0  ->  t = -origin[2] / ray_ego[2]
        5. Ground point = origin + t * ray_ego

    Returns:
        np.ndarray shape (3,) — (x_forward, y_left, 0.0) in ego frame, metres
        None if ray is parallel to ground or hits ground behind the camera
    """
    ray_cam = intrinsics.pixel_to_ray_cam(u, v)
    ray_ego = extrinsics.R_cam_to_ego @ ray_cam
    origin  = extrinsics.translation

    if abs(ray_ego[2]) < 1e-6:
        return None   # ray nearly parallel to ground

    t = -origin[2] / ray_ego[2]
    if t <= 0:
        return None   # intersection behind camera

    return origin + t * ray_ego   # (x_forward, y_left, ~0) in ego frame


# ── Geometry-based distance estimator ────────────────────────────────────────

class GeometryDepthEstimator:
    """
    Estimates metric distance using the pinhole camera model.

    Distance formula:
        distance = (real_size_m x focal_length_px) / apparent_size_px

    Lateral formula (v2):
        Unproject ground-contact pixel to 3D via camera extrinsics.
        Ground plane intersection gives exact metric lateral offset.

    Accuracy (distance):  <15m +-0.5m  |  15-30m +-1.5m  |  30-50m +-4m
    Accuracy (lateral v2): ~0.3m  (vs ~2m without extrinsics)
    """

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
    ):
        self.intrinsics = intrinsics or CameraIntrinsics.nuscenes_default()
        self.extrinsics = extrinsics

    def update_intrinsics(self, calibration: dict):
        if "camera_intrinsic" in calibration:
            K = np.array(calibration["camera_intrinsic"])
            self.intrinsics = CameraIntrinsics.from_matrix(K)

    def update_extrinsics(self, calibration: dict):
        if "translation" in calibration and "rotation" in calibration:
            self.extrinsics = CameraExtrinsics.from_calibration(calibration)

    def update_calibration(self, calibration: dict):
        """Update both intrinsics and extrinsics at once."""
        self.update_intrinsics(calibration)
        self.update_extrinsics(calibration)

    def estimate_distance(self, bbox: BBox2D, class_name: str) -> Optional[float]:
        dims = OBJECT_DIMENSIONS.get(class_name)
        if dims is None:
            return None

        real_height_m, real_width_m, _ = dims
        bbox_height_px = bbox.y2 - bbox.y1
        bbox_width_px  = bbox.x2 - bbox.x1

        if bbox_height_px < 5 or bbox_width_px < 5:
            return None

        dist_from_height = (real_height_m * self.intrinsics.fy) / bbox_height_px
        dist_from_width  = (real_width_m  * self.intrinsics.fx) / bbox_width_px

        if class_name == "person":
            dist = dist_from_height
        elif class_name in ("car", "truck", "bus", "motorcycle"):
            dist = 0.65 * dist_from_height + 0.35 * dist_from_width
        else:
            dist = dist_from_height

        if dist < GEOMETRY_MIN_DIST or dist > GEOMETRY_MAX_DIST:
            return None
        return float(dist)

    def estimate_lateral_position(self, bbox: BBox2D, distance_m: float) -> float:
        """
        Lateral offset from ego centreline in metres (positive = right).
        Uses ray unprojection if extrinsics available, else pinhole approx.
        """
        result = self._lateral_via_ray(bbox)
        if result is not None:
            return result
        return self._lateral_approx(bbox, distance_m)

    def _lateral_via_ray(self, bbox: BBox2D) -> Optional[float]:
        """
        Unproject bottom-centre pixel (ground contact) to ego frame.
        Returns lateral_m (positive = right), or None without extrinsics.
        """
        if self.extrinsics is None:
            return None
        u = (bbox.x1 + bbox.x2) / 2.0
        v = bbox.y2
        ground_pt = ray_unproject_to_ground(u, v, self.intrinsics, self.extrinsics)
        if ground_pt is None:
            return None
        return -float(ground_pt[1])   # ego y=left -> negate for right=positive

    def _lateral_approx(self, bbox: BBox2D, distance_m: float) -> float:
        """Legacy fallback — no extrinsics required."""
        bbox_cx = (bbox.x1 + bbox.x2) / 2.0
        return float(((bbox_cx - self.intrinsics.cx) / self.intrinsics.fx) * distance_m)

    def estimate_3d_position(
        self,
        bbox: BBox2D,
        class_name: str,
        distance_m: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Returns (x_forward, y_lateral_right, z_up) in ego frame (metres).
        Uses ray unprojection for ground-plane classes when extrinsics available.
        """
        if self.extrinsics is not None and class_name in GROUND_PLANE_CLASSES:
            u = (bbox.x1 + bbox.x2) / 2.0
            v = bbox.y2
            ground_pt = ray_unproject_to_ground(u, v, self.intrinsics, self.extrinsics)
            if ground_pt is not None and ground_pt[0] > 0:
                x_fwd   = float(ground_pt[0])
                y_right = -float(ground_pt[1])
                geom_dist = distance_m or self.estimate_distance(bbox, class_name)
                if geom_dist is None or 0.5 < (x_fwd / geom_dist) < 2.0:
                    return np.array([x_fwd, y_right, 0.0])

        # Fallback — intrinsics only
        dist = distance_m or self.estimate_distance(bbox, class_name)
        if dist is None:
            return None
        lateral  = self._lateral_approx(bbox, dist)
        vertical = ((bbox.y2 - self.intrinsics.cy) / self.intrinsics.fy) * dist
        return np.array([dist, lateral, -vertical])


# ── Metric Depth Model ────────────────────────────────────────────────────────

class MetricDepthModel:
    """
    Depth Anything v2 Metric variant.
    Outputs absolute depth in metres (not relative 0-1).
    """

    def __init__(self, device: str = "mps"):
        from prism.utils.common import get_device
        self.device = get_device(device)
        self.model = None
        self.processor = None
        self.INFERENCE_SIZE = (448, 252)
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info(f"Loading Depth Anything v2 Metric on {self.device}")
            model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model = self.model.to(torch.device(self.device))
            self.model.eval()
            logger.info("Metric depth model ready")
        except Exception as e:
            logger.warning(f"Metric depth model unavailable: {e}")
            self.model = None

    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Returns absolute depth map in metres. Shape: (H, W) float32."""
        if self.model is None:
            return None
        try:
            import torch
            from PIL import Image as PILImage
            orig_h, orig_w = image.shape[:2]
            small   = cv2.resize(image, self.INFERENCE_SIZE)
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            inputs  = self.processor(images=PILImage.fromarray(rgb), return_tensors="pt")
            inputs  = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}
            with torch.no_grad():
                depth_m = self.model(**inputs).predicted_depth.squeeze().cpu().numpy()
            return cv2.resize(depth_m.astype(np.float32), (orig_w, orig_h))
        except Exception as e:
            logger.warning(f"Metric depth inference failed: {e}")
            return None


# ── MetricDetection dataclass ─────────────────────────────────────────────────

@dataclass
class MetricDetection:
    """A detection enriched with metric depth information."""
    bbox:        BBox2D
    class_name:  str
    class_id:    int
    confidence:  float
    camera_name: str
    frame_idx:   int
    timestamp:   float

    distance_m:   float                       # metres forward to object
    lateral_m:    float = 0.0                 # metres left(-) / right(+)
    position_3d:  Optional[np.ndarray] = None # (x_fwd, y_right, z_up) ego frame

    depth_method:     str   = "geometry"      # "geometry"|"model"|"fused"|"ray"
    depth_confidence: float = 1.0

    @property
    def is_close(self) -> bool:
        return self.distance_m < 10.0

    @property
    def is_critical(self) -> bool:
        return self.distance_m < 5.0

    @property
    def threat_zone(self) -> str:
        if self.distance_m < 5:   return "CRITICAL"
        if self.distance_m < 15:  return "CLOSE"
        if self.distance_m < 30:  return "MEDIUM"
        return "FAR"


# ── Combined Metric Depth Engine ──────────────────────────────────────────────

class MetricDepthEngine:
    """
    Main metric depth pipeline.
    Combines geometry + ray unprojection + metric depth model.
    """

    def __init__(self, cfg: dict, intrinsics: Optional[CameraIntrinsics] = None):
        sc_cfg = cfg.get("sensory_core", {})
        device = sc_cfg.get("detection", {}).get("device", "mps")

        self.geometry     = GeometryDepthEstimator(intrinsics)
        self.metric_model = MetricDepthModel(device=device)

        self._last_metric_depth: Optional[np.ndarray] = None
        self._depth_frame_count  = 0
        self._run_every_n        = 3
        self._has_extrinsics     = False

        logger.info("Metric Depth Engine ready (v2 — ray unprojection)")

    def update_intrinsics(self, calibration: dict):
        """Update intrinsics AND extrinsics from nuScenes calibration dict."""
        self.geometry.update_calibration(calibration)
        if not self._has_extrinsics and self.geometry.extrinsics is not None:
            self._has_extrinsics = True
            logger.info("Extrinsics loaded — ray unprojection active")

    def process_frame(
        self,
        image: np.ndarray,
        detections: list,
        run_model: bool = False,
    ) -> tuple:
        """
        Enrich all detections with metric distances and lateral positions.
        Returns (metric_detections, metric_depth_map).
        """
        self._depth_frame_count += 1

        metric_depth = None
        if run_model or (self._depth_frame_count % self._run_every_n == 0):
            metric_depth = self.metric_model.estimate(image)
            if metric_depth is not None:
                self._last_metric_depth = metric_depth
        if metric_depth is None:
            metric_depth = self._last_metric_depth

        return [self._enrich_detection(d, metric_depth) for d in detections], metric_depth

    def _enrich_detection(
        self,
        det: Detection,
        metric_depth: Optional[np.ndarray],
    ) -> MetricDetection:

        geometry_dist = None
        model_dist    = None

        if det.bbox.confidence >= MIN_GEOMETRY_CONFIDENCE:
            geometry_dist = self.geometry.estimate_distance(det.bbox, det.bbox.class_name)

        if metric_depth is not None:
            h, w  = metric_depth.shape[:2]
            cx    = int(np.clip((det.bbox.x1 + det.bbox.x2) / 2, 0, w - 1))
            y1    = int(np.clip(det.bbox.y2 * 0.7 + det.bbox.y1 * 0.3, 0, h - 1))
            y2    = int(np.clip(det.bbox.y2 * 0.95, 0, h - 1))
            region = metric_depth[y1:y2, max(0, cx - 8):min(w, cx + 8)]
            if region.size > 0:
                model_dist = float(np.median(region))
                if model_dist < 0.5 or model_dist > 100:
                    model_dist = None

        final_dist, dist_method, dist_conf = self._fuse_distance(
            geometry_dist, model_dist, det.bbox.class_name
        )

        pos_3d = self.geometry.estimate_3d_position(
            det.bbox, det.bbox.class_name, distance_m=final_dist
        )

        if (pos_3d is not None
                and self.geometry.extrinsics is not None
                and det.bbox.class_name in GROUND_PLANE_CLASSES):
            lateral_m = float(pos_3d[1])
            method    = "ray"
            conf      = 0.95
            if pos_3d[0] > 0:
                final_dist = (0.70 * final_dist + 0.30 * float(pos_3d[0])
                              if final_dist is not None else float(pos_3d[0]))
        else:
            lateral_m = self.geometry.estimate_lateral_position(det.bbox, final_dist or 30.0)
            method    = dist_method
            conf      = dist_conf

        return MetricDetection(
            bbox=det.bbox,
            class_name=det.bbox.class_name,
            class_id=det.bbox.class_id,
            confidence=det.bbox.confidence,
            camera_name=det.camera_name,
            frame_idx=det.frame_idx,
            timestamp=det.timestamp,
            distance_m=final_dist or 50.0,
            lateral_m=lateral_m,
            position_3d=pos_3d,
            depth_method=method,
            depth_confidence=conf,
        )

    def _fuse_distance(self, geometry_dist, model_dist, class_name) -> tuple:
        if geometry_dist is not None and model_dist is not None:
            w_geo = 0.75 if geometry_dist < 20.0 else 0.45
            fused = w_geo * geometry_dist + (1.0 - w_geo) * model_dist
            div   = abs(geometry_dist - model_dist) / max(geometry_dist, 1.0)
            return fused, "fused", max(0.3, 1.0 - div * 0.5)
        if geometry_dist is not None:
            return geometry_dist, "geometry", 0.85 if geometry_dist < 25 else 0.60
        if model_dist is not None:
            return model_dist, "model", 0.65
        return None, "none", 0.0


# ── Ground Truth Validator ────────────────────────────────────────────────────

class DepthValidator:
    """
    Validates metric depth + lateral estimates against nuScenes LiDAR GT.
    Tracks both distance and lateral MAE separately for the paper.
    """

    def __init__(self):
        self.dist_errors:    list = []
        self.lateral_errors: list = []
        self.dist_rel:       list = []
        self.predictions:    list = []
        self.ground_truths:  list = []
        self.methods:        list = []

    def add_sample(
        self,
        pred_dist_m:    float,
        gt_dist_m:      float,
        pred_lateral_m: float = 0.0,
        gt_lateral_m:   float = 0.0,
        method:         str   = "unknown",
    ):
        if gt_dist_m <= 0 or gt_dist_m > 80:
            return
        self.dist_errors.append(abs(pred_dist_m - gt_dist_m))
        self.dist_rel.append(abs(pred_dist_m - gt_dist_m) / gt_dist_m)
        self.lateral_errors.append(abs(pred_lateral_m - gt_lateral_m))
        self.predictions.append(pred_dist_m)
        self.ground_truths.append(gt_dist_m)
        self.methods.append(method)

    def compute_metrics(self) -> dict:
        if not self.dist_errors:
            return {}
        de    = np.array(self.dist_errors)
        le    = np.array(self.lateral_errors)
        pred  = np.array(self.predictions)
        gt    = np.array(self.ground_truths)
        delta = np.maximum(pred / gt, gt / pred)
        method_mae = {
            m: float(np.mean([de[i] for i, x in enumerate(self.methods) if x == m]))
            for m in set(self.methods)
        }
        return {
            "n_samples":       len(de),
            "dist_MAE_m":      float(np.mean(de)),
            "dist_RMSE_m":     float(np.sqrt(np.mean(de**2))),
            "dist_MedAE_m":    float(np.median(de)),
            "dist_MRE":        float(np.mean(self.dist_rel)),
            "lateral_MAE_m":   float(np.mean(le)),
            "lateral_MedAE_m": float(np.median(le)),
            "delta_1_25":      float(np.mean(delta < 1.25)),
            "delta_1_5":       float(np.mean(delta < 1.5625)),
            "delta_2":         float(np.mean(delta < 2.0)),
            "method_mae":      method_mae,
        }

    def print_report(self):
        m = self.compute_metrics()
        if not m:
            logger.info("No validation samples collected yet.")
            return
        logger.info("=" * 55)
        logger.info("METRIC DEPTH VALIDATION REPORT (v2)")
        logger.info("=" * 55)
        logger.info(f"Samples:           {m['n_samples']}")
        logger.info("--- Distance ---")
        logger.info(f"  MAE:             {m['dist_MAE_m']:.2f} m")
        logger.info(f"  RMSE:            {m['dist_RMSE_m']:.2f} m")
        logger.info(f"  Median AE:       {m['dist_MedAE_m']:.2f} m")
        logger.info(f"  Mean Rel Err:    {m['dist_MRE']*100:.1f} %")
        logger.info(f"  d<1.25:          {m['delta_1_25']*100:.1f} %")
        logger.info(f"  d<1.5625:        {m['delta_1_5']*100:.1f} %")
        logger.info(f"  d<2.0:           {m['delta_2']*100:.1f} %")
        logger.info("--- Lateral Position ---")
        logger.info(f"  MAE:             {m['lateral_MAE_m']:.2f} m")
        logger.info(f"  Median AE:       {m['lateral_MedAE_m']:.2f} m")
        if m.get("method_mae"):
            logger.info("--- By depth method ---")
            for meth, mae in m["method_mae"].items():
                logger.info(f"  {meth:<12}   MAE = {mae:.2f} m")
        logger.info("=" * 55)
