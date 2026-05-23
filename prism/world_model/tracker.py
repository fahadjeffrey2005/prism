"""
PRISM — Actor Tracker
======================
Maintains identity of every actor across frames.
Uses SORT (Simple Online Realtime Tracking) — Kalman filter + Hungarian algorithm.

Without tracking:
    Frame 1: "there is a car at position X"
    Frame 2: "there is a car at position Y"
    → Are these the same car? Unknown.

With tracking:
    Frame 1: "Actor #3 (car) is at position X, velocity 0"
    Frame 2: "Actor #3 (car) is at position Y, velocity 12km/h heading north"
    → Same car, now we know its speed and direction.

This is what enables intent prediction in Layer 3.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from filterpy.kalman import KalmanFilter
from prism.utils.common import get_logger, BBox2D

logger = get_logger("Tracker")


# ── Kalman-based single object tracker ───────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman filter.
    State: [x, y, w, h, vx, vy, vw, vh]
    where (x,y) = center, (w,h) = size, v* = velocities
    """
    count = 0

    def __init__(self, bbox: BBox2D):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition — constant velocity model
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement — we observe (x, y, w, h) only
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        # Covariance — high uncertainty on velocities initially
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from first detection
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Metadata from detection
        self.class_name = bbox.class_name
        self.class_id = bbox.class_id
        self.last_confidence = bbox.confidence
        self.depth_history = []

    def update(self, bbox: BBox2D):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.last_confidence = bbox.confidence
        self.class_name = bbox.class_name
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> BBox2D:
        return self._z_to_bbox(self.kf.x)

    @property
    def velocity(self) -> tuple:
        """Returns (vx, vy) in pixel/frame units."""
        return (float(np.squeeze(self.kf.x[4])), float(np.squeeze(self.kf.x[5])))

    @staticmethod
    def _bbox_to_z(bbox: BBox2D) -> np.ndarray:
        """Convert BBox2D to [cx, cy, w, h]."""
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        return np.array([[cx], [cy], [w], [h]], dtype=float)

    @staticmethod
    def _z_to_bbox(x: np.ndarray) -> BBox2D:
        """Convert state vector back to BBox2D."""
        cx = float(np.squeeze(x[0]))
        cy = float(np.squeeze(x[1]))
        w  = float(np.squeeze(x[2]))
        h  = float(np.squeeze(x[3]))
        return BBox2D(
            x1=cx - w/2, y1=cy - h/2,
            x2=cx + w/2, y2=cy + h/2
        )


# ── IoU helper ────────────────────────────────────────────────────────────────

def iou(bb1: BBox2D, bb2: BBox2D) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(bb1.x1, bb2.x1)
    y1 = max(bb1.y1, bb2.y1)
    x2 = min(bb1.x2, bb2.x2)
    y2 = min(bb1.y2, bb2.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bb1.area
    area2 = bb2.area
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(trackers: list, detections: list) -> np.ndarray:
    """Build IoU cost matrix for Hungarian assignment."""
    mat = np.zeros((len(trackers), len(detections)))
    for i, trk in enumerate(trackers):
        for j, det in enumerate(detections):
            mat[i, j] = iou(trk, det)
    return mat


# ── SORT Tracker ──────────────────────────────────────────────────────────────

@dataclass
class TrackedActor:
    """
    A confirmed tracked actor in the scene.
    This is the fundamental unit of the World Model's Dynamic Layer.
    """
    track_id: int
    class_name: str
    class_id: int
    bbox: BBox2D
    velocity_px: tuple           # (vx, vy) in pixels/frame
    confidence: float
    age: int                     # frames since first detection
    frames_since_update: int
    depth: Optional[float] = None
    depth_history: list = field(default_factory=list)

    # Enriched by VLM later
    intent_label: str = "unknown"
    intent_confidence: float = 0.0
    risk_score: float = 0.0

    @property
    def is_confirmed(self) -> bool:
        return self.frames_since_update == 0

    @property
    def estimated_speed_px(self) -> float:
        vx, vy = self.velocity_px
        return float(np.sqrt(vx**2 + vy**2))

    @property
    def is_moving(self) -> bool:
        return self.estimated_speed_px > 2.0  # pixel threshold


class Sort:
    """
    SORT: Simple Online and Realtime Tracking
    Kalman filter per track + Hungarian assignment per frame.
    O(n) per frame, runs at 1000fps on CPU — negligible overhead.
    """

    def __init__(self, max_age: int = 10, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: list) -> list:
        """
        Update trackers with new detections.
        detections: list of Detection objects from SensoryCore
        Returns: list of TrackedActor objects
        """
        from scipy.optimize import linear_sum_assignment
        self.frame_count += 1

        # Predict new locations of existing trackers
        predicted_bboxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan([pred.x1, pred.y1, pred.x2, pred.y2])):
                to_del.append(i)
            else:
                predicted_bboxes.append(pred)
        for i in reversed(to_del):
            self.trackers.pop(i)

        # Hungarian assignment
        matched_indices = set()
        unmatched_dets = list(range(len(detections)))

        if len(predicted_bboxes) > 0 and len(detections) > 0:
            det_bboxes = [d.bbox for d in detections]
            iou_mat = iou_matrix(predicted_bboxes, det_bboxes)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)

            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    self.trackers[r].update(detections[c].bbox)
                    if detections[c].depth_estimate is not None:
                        self.trackers[r].depth_history.append(detections[c].depth_estimate)
                        if len(self.trackers[r].depth_history) > 10:
                            self.trackers[r].depth_history.pop(0)
                    matched_indices.add(r)
                    if c in unmatched_dets:
                        unmatched_dets.remove(c)

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i].bbox)
            if detections[i].depth_estimate is not None:
                trk.depth_history.append(detections[i].depth_estimate)
            self.trackers.append(trk)

        # Build output — only confirmed tracks
        active_actors = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            if trk.time_since_update > self.max_age:
                to_del.append(i)
                continue
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = trk.get_state()
                bbox.class_name = trk.class_name
                bbox.class_id = trk.class_id
                bbox.confidence = trk.last_confidence
                bbox.track_id = trk.id

                depth = None
                if trk.depth_history:
                    depth = float(np.median(trk.depth_history[-3:]))

                actor = TrackedActor(
                    track_id=trk.id,
                    class_name=trk.class_name,
                    class_id=trk.class_id,
                    bbox=bbox,
                    velocity_px=trk.velocity,
                    confidence=trk.last_confidence,
                    age=trk.age,
                    frames_since_update=trk.time_since_update,
                    depth=depth,
                    depth_history=list(trk.depth_history),
                )
                active_actors.append(actor)

        for i in reversed(to_del):
            self.trackers.pop(i)

        return active_actors

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
