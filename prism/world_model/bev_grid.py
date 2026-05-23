"""
PRISM — BEV Occupancy Grid
===========================
Bird's Eye View representation of the world.

This is Tesla's core innovation — instead of thinking in image space
(pixels, bounding boxes), we think in world space (meters, real positions).

The grid is a top-down view centered on the ego vehicle:
    - x axis: forward/backward (meters)
    - y axis: left/right (meters)
    - Each cell: probability of being occupied (0.0 = free, 1.0 = occupied)

Why this matters:
    Image space:  "there's a car in the bottom-right of the frame"
    World space:  "there's a car 12m ahead, 3m to the right"

The second representation is what a planner needs.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional
from prism.utils.common import get_logger, CLASS_COLORS

logger = get_logger("BEVGrid")


@dataclass
class BEVConfig:
    x_range: tuple = (-50, 50)      # meters forward/back
    y_range: tuple = (-50, 50)      # meters left/right
    resolution: float = 0.5         # meters per cell
    decay_rate: float = 0.85        # how fast occupancy fades each frame

    @property
    def grid_h(self) -> int:
        return int((self.x_range[1] - self.x_range[0]) / self.resolution)

    @property
    def grid_w(self) -> int:
        return int((self.y_range[1] - self.y_range[0]) / self.resolution)

    def world_to_grid(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates (meters) to grid indices."""
        gi = int((x_m - self.x_range[0]) / self.resolution)
        gj = int((y_m - self.y_range[0]) / self.resolution)
        gi = np.clip(gi, 0, self.grid_h - 1)
        gj = np.clip(gj, 0, self.grid_w - 1)
        return gi, gj

    def grid_to_world(self, gi: int, gj: int) -> tuple:
        """Convert grid indices back to world coordinates."""
        x_m = gi * self.resolution + self.x_range[0]
        y_m = gj * self.resolution + self.y_range[0]
        return x_m, y_m


class BEVOccupancyGrid:
    """
    Probabilistic BEV occupancy grid.

    Updated every frame from:
    1. Tracked actor positions (from World Model Dynamic Layer)
    2. Depth map projections (from Sensory Core)

    Cells decay toward 0 (free) when not observed — like memory fading.
    This handles occlusion naturally: if we haven't seen something for
    a few frames, we become less certain it's still there.
    """

    def __init__(self, cfg: Optional[BEVConfig] = None):
        self.cfg = cfg or BEVConfig()
        # Main occupancy grid — float32, 0=free, 1=occupied
        self.grid = np.zeros((self.cfg.grid_h, self.cfg.grid_w), dtype=np.float32)
        # Per-class layer — which class occupies each cell
        self.class_grid = np.full((self.cfg.grid_h, self.cfg.grid_w), -1, dtype=np.int8)
        # Track ID grid — which track occupies each cell
        self.track_grid = np.full((self.cfg.grid_h, self.cfg.grid_w), -1, dtype=np.int32)
        self.frame_count = 0

    def update_from_actors(self, actors: list, camera_intrinsic: Optional[np.ndarray] = None):
        """
        Update occupancy from tracked actor list.
        Each actor's footprint is projected onto the BEV grid.
        """
        self.frame_count += 1

        # Decay existing occupancy — memory fades
        self.grid *= self.cfg.decay_rate
        self.class_grid[self.grid < 0.1] = -1
        self.track_grid[self.grid < 0.1] = -1

        for actor in actors:
            if not actor.is_confirmed:
                continue

            # Estimate world position from depth
            # depth is normalized 0-1, we convert to rough meters
            # Typical scene range: 0-50m
            if actor.depth is not None:
                # Invert depth (depth model: 0=far, 1=close in our normalization)
                # Actually our model: 0=close, 1=far (we normalized min->0, max->1)
                # So depth 0.1 = close (~5m), depth 0.9 = far (~45m)
                dist_m = actor.depth * 50.0  # rough metric conversion

                # Estimate lateral offset from bbox center x
                # This is simplified — proper version uses camera intrinsics
                bbox = actor.bbox
                img_w = 1600  # nuScenes image width
                img_cx = (bbox.x1 + bbox.x2) / 2
                # Normalized offset from center (-0.5 to 0.5)
                lateral_norm = (img_cx / img_w) - 0.5
                # Lateral distance in meters (rough estimate)
                lateral_m = lateral_norm * dist_m * 0.8

                # Actor footprint size based on class
                footprint = self._get_footprint(actor.class_name)

                # Mark cells as occupied
                self._mark_footprint(
                    x_m=dist_m,
                    y_m=lateral_m,
                    footprint=footprint,
                    class_id=actor.class_id,
                    track_id=actor.track_id,
                    confidence=min(actor.confidence * 1.2, 1.0)
                )

    def _get_footprint(self, class_name: str) -> tuple:
        """Returns (length_m, width_m) for each class."""
        footprints = {
            "car":          (4.5, 2.0),
            "truck":        (8.0, 2.5),
            "bus":          (12.0, 2.8),
            "motorcycle":   (2.2, 0.8),
            "bicycle":      (1.8, 0.6),
            "person":       (0.5, 0.5),
            "traffic light":(0.3, 0.3),
            "stop sign":    (0.3, 0.3),
        }
        return footprints.get(class_name, (1.0, 1.0))

    def _mark_footprint(
        self, x_m: float, y_m: float,
        footprint: tuple, class_id: int,
        track_id: int, confidence: float
    ):
        """Mark a rectangular footprint as occupied on the grid."""
        length, width = footprint
        x_min = x_m - length / 2
        x_max = x_m + length / 2
        y_min = y_m - width / 2
        y_max = y_m + width / 2

        gi_min, gj_min = self.cfg.world_to_grid(x_min, y_min)
        gi_max, gj_max = self.cfg.world_to_grid(x_max, y_max)

        gi_min, gi_max = sorted([gi_min, gi_max])
        gj_min, gj_max = sorted([gj_min, gj_max])

        self.grid[gi_min:gi_max+1, gj_min:gj_max+1] = np.maximum(
            self.grid[gi_min:gi_max+1, gj_min:gj_max+1],
            confidence
        )
        self.class_grid[gi_min:gi_max+1, gj_min:gj_max+1] = class_id
        self.track_grid[gi_min:gi_max+1, gj_min:gj_max+1] = track_id

    def get_occupancy(self) -> np.ndarray:
        """Returns current occupancy grid (H, W) float32."""
        return self.grid.copy()

    def is_free(self, x_m: float, y_m: float, threshold: float = 0.3) -> bool:
        """Check if a world position is free."""
        gi, gj = self.cfg.world_to_grid(x_m, y_m)
        return self.grid[gi, gj] < threshold

    def get_free_corridor(self, width_m: float = 3.0) -> np.ndarray:
        """
        Returns a mask of the driveable corridor directly ahead.
        Used by the planner to find safe trajectories.
        """
        cfg = self.cfg
        ego_gi, ego_gj = cfg.world_to_grid(0, 0)
        half_w = int(width_m / cfg.resolution / 2)
        corridor = self.grid[
            ego_gi:,
            max(0, ego_gj - half_w):min(cfg.grid_w, ego_gj + half_w)
        ]
        return corridor

    def visualize(self, size: int = 400) -> np.ndarray:
        """
        Render BEV grid as a colored top-down image.
        Ego vehicle shown as white rectangle at center.
        """
        # Resize grid to display size
        vis = cv2.resize(self.grid, (size, size))

        # Colormap: blue=free, red=occupied
        vis_uint8 = (vis * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_HOT)

        # Draw ego vehicle
        cx, cy = size // 2, size // 2
        ego_w, ego_h = int(size * 0.03), int(size * 0.05)
        cv2.rectangle(colored,
                      (cx - ego_w, cy - ego_h),
                      (cx + ego_w, cy + ego_h),
                      (255, 255, 255), -1)

        # Draw grid center lines
        cv2.line(colored, (0, cy), (size, cy), (50, 50, 50), 1)
        cv2.line(colored, (cx, 0), (cx, size), (50, 50, 50), 1)

        # Labels
        cv2.putText(colored, "FRONT", (cx - 20, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(colored, "BEV", (10, size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return colored

    def reset(self):
        self.grid[:] = 0
        self.class_grid[:] = -1
        self.track_grid[:] = -1
        self.frame_count = 0
