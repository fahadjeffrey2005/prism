"""
PRISM — Sensory Core
====================
The only component that touches raw pixels.
Everything above this works on abstractions.

Pipeline per frame:
    Raw image
        → Feature-aware detection (YOLOv8)
        → Monocular depth estimation (Depth Anything v2)
        → Lightweight segmentation (FastSAM) [every N frames]
        → Optical flow delta [every frame]
        → SensoryFrame output → World Model
"""

import time
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from prism.utils.common import (
    get_logger, get_device, Timer, FrameSampler,
    BBox2D, Detection, SensoryFrame, CLASS_COLORS
)

logger = get_logger("SensoryCore")


# ── COCO class names (YOLOv8 default) ────────────────────────────────────────

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

DRIVING_CLASS_IDS = {0, 1, 2, 3, 5, 7, 9, 11}  # person, bike, car, moto, bus, truck, light, sign


# ── Detector ─────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLOv8-based object detector.
    Auto-downloads weights on first run.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = get_device(cfg.get("device", "mps"))
        self.conf = cfg.get("confidence_threshold", 0.35)
        self.iou = cfg.get("iou_threshold", 0.45)
        self.img_size = cfg.get("img_size", 640)
        self.classes = list(cfg.get("classes_of_interest", DRIVING_CLASS_IDS))
        self.model = None
        self._load_model(cfg.get("model", "yolov8n.pt"))

    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading detector: {model_name} on {self.device}")
            self.model = YOLO(model_name)
            logger.info("Detector ready")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def detect(self, image: np.ndarray) -> list:
        """
        Run detection on a BGR image (OpenCV format).
        Returns list of Detection objects.
        """
        if self.model is None:
            return []

        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=self.classes,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else "unknown"

                detection = Detection(
                    bbox=BBox2D(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    )
                )
                detections.append(detection)

        return detections


# ── Depth Estimator ───────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Depth Anything v2 — monocular depth estimation.
    Runs directly on MPS at 448x252 — hits 26fps on M4.
    Upsamples back to original resolution after inference.
    """

    INFERENCE_SIZE = (320, 180)   # width, height — reduced for thermal safety

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(get_device(cfg.get("device", "mps")))
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info(f"Loading Depth Anything v2 on {self.device} at {self.INFERENCE_SIZE}")
            self.processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            ).to(self.device)
            self.model.eval()
            logger.info(f"Depth estimator ready on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load Depth Anything: {e}")
            self.model = None

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from BGR image.
        Returns depth map as float32, same HxW as input image.
        """
        if self.model is None:
            h, w = image.shape[:2]
            return np.tile(np.linspace(0.2, 1.0, h).reshape(-1, 1), (1, w)).astype(np.float32)

        try:
            import torch
            from PIL import Image as PILImage
            orig_h, orig_w = image.shape[:2]
            # Resize to fast inference resolution
            small = cv2.resize(image, self.INFERENCE_SIZE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()
            # Normalize to 0-1
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            # Upsample back to original resolution
            return cv2.resize(depth.astype(np.float32), (orig_w, orig_h))
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: BBox2D) -> float:
        """
        Sample depth at the ground contact point of a bounding box.
        Bottom-center is most reliable for distance estimation.
        """
        h, w = depth_map.shape[:2]
        cx = int(np.clip((bbox.x1 + bbox.x2) / 2, 0, w - 1))
        y_bottom = int(np.clip(bbox.y2 * 0.9 + bbox.y1 * 0.1, 0, h - 1))
        y_top = int(np.clip(bbox.y2 * 0.6 + bbox.y1 * 0.4, 0, h - 1))
        region = depth_map[y_top:y_bottom, max(0, cx-5):min(w, cx+5)]
        if region.size == 0:
            return float(depth_map[y_bottom, cx])
        return float(np.median(region))


# ── Optical Flow ──────────────────────────────────────────────────────────────

class OpticalFlowEstimator:
    """
    Dense optical flow using Farneback method.
    Lightweight — runs every frame.
    Catches motion before the detector even fires.
    """

    def __init__(self):
        self._prev_gray = None
        self._params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    def compute(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow between previous and current frame.
        Returns flow array (H, W, 2) or None on first frame.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **self._params
        )
        self._prev_gray = gray
        return flow

    def get_motion_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """Returns scalar motion magnitude per pixel."""
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return magnitude

    def reset(self):
        self._prev_gray = None


# ── Sensory Core ──────────────────────────────────────────────────────────────

class SensoryCore:
    """
    Main Sensory Core.
    Orchestrates all perception models and outputs SensoryFrames.

    Usage:
        core = SensoryCore(cfg)
        for image in camera_stream:
            sensory_frame = core.process(image, camera_name="CAM_FRONT")
            world_model.update(sensory_frame)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sc_cfg = cfg.get("sensory_core", {})
        sampling_cfg = sc_cfg.get("sampling", {})

        # Init subcomponents
        logger.info("Initialising Sensory Core...")
        self.detector = Detector(sc_cfg.get("detection", {}))
        self.depth_estimator = DepthEstimator(sc_cfg.get("depth", {}))
        self.flow_estimator = OpticalFlowEstimator()

        # Frame sampler — controls what runs when
        self.sampler = FrameSampler({
            "detection":    sampling_cfg.get("detection_fps", 12),
            "depth":        sampling_cfg.get("depth_fps", 4),
            "segmentation": sampling_cfg.get("segmentation_fps", 4),
            "flow":         sampling_cfg.get("flow_fps", 12),
        })

        self._frame_idx = 0
        logger.info("Sensory Core ready")

    def process(
        self,
        image: np.ndarray,
        camera_name: str = "CAM_FRONT",
        timestamp: float = 0.0
    ) -> SensoryFrame:
        """
        Process one camera frame.
        Returns a SensoryFrame with all available perception outputs.
        """
        schedule = self.sampler.tick()
        processing_times = {}
        self._frame_idx += 1

        frame = SensoryFrame(
            frame_idx=self._frame_idx,
            timestamp=timestamp or time.time(),
            camera_name=camera_name,
            image=image.copy(),
        )

        # Optical flow — every frame, lightweight
        if schedule["flow"]:
            with Timer("flow") as t:
                flow = self.flow_estimator.compute(image)
                frame.optical_flow = flow
            processing_times["flow_ms"] = round(t.elapsed * 1000, 1)

        # Depth — every N frames
        if schedule["depth"]:
            with Timer("depth") as t:
                frame.depth_map = self.depth_estimator.estimate(image)
            processing_times["depth_ms"] = round(t.elapsed * 1000, 1)

        # Detection — every frame
        if schedule["detection"]:
            with Timer("detection") as t:
                detections = self.detector.detect(image)
                if frame.depth_map is not None:
                    for det in detections:
                        det.depth_estimate = self.depth_estimator.get_depth_at_bbox(
                            frame.depth_map, det.bbox
                        )
                        det.camera_name = camera_name
                        det.frame_idx = self._frame_idx
                        det.timestamp = frame.timestamp
                frame.detections = detections
            processing_times["detection_ms"] = round(t.elapsed * 1000, 1)

        frame.processing_times = processing_times
        return frame

    def reset(self):
        self.sampler.reset()
        self.flow_estimator.reset()
        self._frame_idx = 0


# ── Visualizer ────────────────────────────────────────────────────────────────

class SensoryVisualizer:
    """
    Renders SensoryFrame outputs as annotated images.
    Used for debugging and demo displays.
    """

    @staticmethod
    def draw_detections(image: np.ndarray, frame: SensoryFrame) -> np.ndarray:
        """Draw detection bounding boxes with depth labels."""
        out = image.copy()
        for det in frame.detections:
            b = det.bbox
            color = CLASS_COLORS.get(b.class_name, CLASS_COLORS["unknown"])
            # Box
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            # Label
            depth_str = f" {det.depth_estimate:.2f}d" if det.depth_estimate is not None else ""
            label = f"{b.class_name} {b.confidence:.2f}{depth_str}"
            cv2.putText(out, label, (int(b.x1), int(b.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
        """Convert float depth map to colorized BGR image."""
        if depth_map is None:
            return None
        norm = np.clip(depth_map, 0, 1)
        norm_uint8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_INFERNO)
        return colored

    @staticmethod
    def colorize_flow(flow: np.ndarray) -> np.ndarray:
        """Convert optical flow to HSV visualization."""
        if flow is None:
            return None
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag * 10, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def make_overlay(
        image: np.ndarray,
        frame: SensoryFrame,
        show_depth: bool = True,
        show_flow: bool = False,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Composite visualization — detections + optional depth overlay.
        This is the main output for demos.
        """
        base = SensoryVisualizer.draw_detections(image, frame)

        if show_depth and frame.depth_map is not None:
            depth_colored = SensoryVisualizer.colorize_depth(frame.depth_map)
            depth_resized = cv2.resize(depth_colored, (base.shape[1], base.shape[0]))
            base = cv2.addWeighted(base, 1 - alpha, depth_resized, alpha, 0)
            # Re-draw boxes on top of depth overlay
            base = SensoryVisualizer.draw_detections(base, frame)

        # HUD — top left info
        info_lines = [
            f"Frame: {frame.frame_idx}",
            f"Camera: {frame.camera_name}",
            f"Objects: {len(frame.detections)}",
        ]
        for key, val in frame.processing_times.items():
            info_lines.append(f"{key}: {val}ms")

        for i, line in enumerate(info_lines):
            cv2.putText(base, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return base
