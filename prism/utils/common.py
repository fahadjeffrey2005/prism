"""
PRISM Utilities
Common helpers used across all components.
"""

import os
import time
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand ~ in paths
    _expand_paths(cfg)
    return cfg


def _expand_paths(cfg: dict):
    for key, val in cfg.items():
        if isinstance(val, dict):
            _expand_paths(val)
        elif isinstance(val, str) and val.startswith("~"):
            cfg[key] = str(Path(val).expanduser())


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(preferred: str = "mps") -> str:
    """
    Returns the best available device.
    Priority: preferred → mps → cuda → cpu
    """
    import torch
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for profiling."""
    def __init__(self, name: str = "", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.logger:
            self.logger.debug(f"{self.name}: {self.elapsed*1000:.1f}ms")


# ── Frame Sampler ─────────────────────────────────────────────────────────────

class FrameSampler:
    """
    Controls which components run on which frames.
    Avoids running expensive models every single frame.
    """
    def __init__(self, rates: dict):
        """
        rates: dict of component_name -> fps
        e.g. {"detection": 12, "depth": 4, "segmentation": 4}
        """
        self.rates = rates
        self.frame_count = 0
        self._intervals = {}
        base_fps = max(rates.values())
        for name, fps in rates.items():
            self._intervals[name] = max(1, round(base_fps / fps))

    def tick(self) -> dict:
        """Call every frame. Returns dict of component -> should_run."""
        self.frame_count += 1
        return {
            name: (self.frame_count % interval == 0)
            for name, interval in self._intervals.items()
        }

    def reset(self):
        self.frame_count = 0


# ── BBox Utils ────────────────────────────────────────────────────────────────

@dataclass
class BBox2D:
    """2D bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = -1
    class_name: str = ""
    track_id: Optional[int] = None

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def as_xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def as_xywh(self):
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class Detection:
    """A single detection from the Sensory Core."""
    bbox: BBox2D
    depth_estimate: Optional[float] = None    # meters, from depth model
    camera_name: str = ""
    frame_idx: int = 0
    timestamp: float = 0.0


@dataclass
class SensoryFrame:
    """
    Output of one Sensory Core processing cycle.
    This is what gets fed into the World Model.
    """
    frame_idx: int
    timestamp: float
    camera_name: str
    image: Optional[np.ndarray] = None
    detections: list = field(default_factory=list)       # List[Detection]
    depth_map: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    optical_flow: Optional[np.ndarray] = None
    processing_times: dict = field(default_factory=dict) # component -> ms


# ── Color Palette ─────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "person":        (255, 100, 100),
    "bicycle":       (100, 255, 100),
    "car":           (100, 100, 255),
    "motorcycle":    (255, 200, 50),
    "bus":           (255, 100, 255),
    "truck":         (100, 255, 255),
    "traffic light": (255, 255, 50),
    "stop sign":     (255, 50, 50),
    "unknown":       (180, 180, 180),
}

ARBITRATION_COLORS = {
    0: (50, 200, 50),     # GREEN
    1: (200, 200, 50),    # YELLOW
    2: (200, 120, 50),    # ORANGE
    3: (200, 50, 50),     # RED
}
