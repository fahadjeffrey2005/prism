"""
PRISM — Dashboard Visualizer  (experimentation branch)
=======================================================
Tesla-style side-by-side layout.

Key features:
- Perspective ground projection using camera intrinsics (path sticks to road)
- Real-time trajectory minimap (top-down path trace with curves + obstacles)
- Lane detection overlay (red lines + green fill) with fallback
- Full sidebar: BEV, AI Decision, Steering/Speed gauges, System Status
"""

import cv2
import math
import numpy as np
from collections import deque

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
SIDEBAR_BG    = (10,  12,  10)
TEXT_WHITE    = (235, 235, 235)
TEXT_DIM      = (110, 110, 110)
TEXT_CYAN     = (220, 220,  30)
ACCENT_GREEN  = ( 40, 210,  60)
ACCENT_YELLOW = ( 30, 200, 230)
ACCENT_BLUE   = (220, 130,  40)
ACCENT_ORANGE = ( 30, 140, 230)
LANE_LINE_COL = ( 30,  30, 220)
DRIVE_COL     = ( 30, 180,  30)
WARN_RED      = ( 30,  30, 220)
PATH_FILL     = ( 25, 140,  25)
PATH_LINE     = ( 30,  30, 200)

DECISION_DISPLAY = {
    "CLEAR":     "CRUISE",
    "MONITOR":   "CRUISE",
    "EASE":      "EASE",
    "SLOW":      "SLOW",
    "CAUTION":   "CAUTION",
    "YIELD":     "YIELD",
    "STOP":      "STOP",
    "EMERGENCY": "EMERGENCY",
}

DEC_COLORS = {
    "CRUISE":    ( 30, 200,  50),
    "EASE":      ( 30, 200, 130),
    "SLOW":      ( 30, 130, 210),
    "YIELD":     ( 20, 100, 230),
    "CAUTION":   ( 20,  70, 230),
    "STOP":      ( 20,  20, 210),
    "EMERGENCY": (  0,   0, 220),
    "UNKNOWN":   ( 60,  60,  60),
}

DEC_SUBTITLES = {
    "CRUISE":    "Road clear — cruising",
    "EASE":      "Easing off — caution ahead",
    "SLOW":      "Reducing speed — obstacle ahead",
    "YIELD":     "Yielding to road users",
    "CAUTION":   "Caution — obstacle detected",
    "STOP":      "Coming to a complete stop",
    "EMERGENCY": "EMERGENCY BRAKE",
    "UNKNOWN":   "",
}

_vlm_buf = ""
_vlm_ttl = 0
VLM_HOLD  = 72
SIDEBAR_FRAC = 0.28


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-plane perspective path projection
# ═══════════════════════════════════════════════════════════════════════════════

def _project_ground_point(x_lat, d_fwd, fx, fy, cx, cy, cam_h):
    """
    Project a 3D ground-plane point to image pixel.
    Camera convention: x=right, y=down, z=forward.
    Ground point (x_lat metres right, d_fwd metres ahead, 0m height).
    Camera is cam_h metres above ground.
    Returns (u, v) or None if behind camera / above horizon.
    """
    if d_fwd < 0.1:
        return None
    u = int(fx * x_lat / d_fwd + cx)
    v = int(fy * cam_h  / d_fwd + cy)
    return (u, v)


def draw_ground_path(frame: np.ndarray,
                     fx: float, fy: float,
                     cx: float, cy: float,
                     cam_h: float = 1.0,
                     steer: float = 0.0,
                     corridor_w: float = 1.4):
    """
    Draw a perspective-correct path corridor on the ground plane.
    Uses pinhole camera model so the projection sticks to the road surface.
    steer: -1=full left, 0=straight, +1=full right
    corridor_w: half-width of corridor in metres
    """
    h, w = frame.shape[:2]

    # Distances to sample along the path (metres)
    distances = np.array([1.5, 2.5, 4.0, 6.0, 9.0, 14.0, 22.0, 35.0])

    left_pts, right_pts = [], []

    for d in distances:
        # Steering causes lateral drift that grows with distance squared (arc)
        lat_center = steer * (d ** 1.4) * 0.08

        pt_l = _project_ground_point(lat_center - corridor_w, d, fx, fy, cx, cy, cam_h)
        pt_r = _project_ground_point(lat_center + corridor_w, d, fx, fy, cx, cy, cam_h)

        if pt_l is None or pt_r is None:
            continue
        # Only keep points in the image
        if not (0 <= pt_l[1] < h and 0 <= pt_r[1] < h):
            continue

        left_pts.append(pt_l)
        right_pts.append(pt_r)

    if len(left_pts) < 2:
        return  # not enough valid points

    # Filled polygon
    poly_pts = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [poly_pts], PATH_FILL)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)

    # Left and right boundary lines (red)
    for i in range(len(left_pts) - 1):
        cv2.line(frame, left_pts[i],  left_pts[i+1],  PATH_LINE, 2, cv2.LINE_AA)
        cv2.line(frame, right_pts[i], right_pts[i+1], PATH_LINE, 2, cv2.LINE_AA)

    # Centre dashed path line
    center_pts = []
    for d in distances:
        lat_center = steer * (d ** 1.4) * 0.08
        pt = _project_ground_point(lat_center, d, fx, fy, cx, cy, cam_h)
        if pt and 0 <= pt[1] < h:
            center_pts.append(pt)

    for i in range(0, len(center_pts) - 1, 2):
        cv2.line(frame, center_pts[i], center_pts[min(i+1, len(center_pts)-1)],
                 (180, 180, 180), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory tracker + minimap
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryTracker:
    """
    Dead-reckoning trajectory from speed + steering.
    Maintains a top-down position history for the minimap.
    """
    def __init__(self, max_points: int = 800):
        self.positions: deque = deque(maxlen=max_points)
        self.x        = 0.0    # metres (right = positive)
        self.y        = 0.0    # metres (forward = positive)
        self.heading  = 0.0    # radians (0 = forward)
        self._last_ts = None

    def update(self, speed_mps: float, steer: float, timestamp: float):
        if self._last_ts is None:
            self._last_ts = timestamp
            self.positions.append((self.x, self.y))
            return

        dt = float(np.clip(timestamp - self._last_ts, 0.0, 0.2))
        self._last_ts = timestamp

        if dt > 0 and speed_mps > 0.05:
            # Steering changes heading: scale by speed and steer gain
            self.heading += steer * speed_mps * dt * 0.25
            self.x       += speed_mps * dt * math.sin(self.heading)
            self.y       += speed_mps * dt * math.cos(self.heading)

        self.positions.append((self.x, self.y))

    def draw_minimap(self, lidar_dets: list,
                     size: tuple = (200, 160)) -> np.ndarray:
        """
        Render top-down trajectory minimap.
        Returns a (size[1], size[0], 3) BGR image.
        """
        mw, mh = size
        mini = np.zeros((mh, mw, 3), dtype=np.uint8)
        cv2.rectangle(mini, (0,0), (mw-1, mh-1), (28,38,28), -1)

        if len(self.positions) < 2:
            return mini

        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]

        # Auto-scale: keep last 60m of travel visible
        x_center = xs[-1]
        y_center = ys[-1]
        view_r   = max(15.0, (max(ys) - min(ys) + max(xs) - min(xs)) * 0.5 + 5)

        x_min, x_max = x_center - view_r, x_center + view_r
        y_min, y_max = y_center - view_r, y_center + view_r

        def w2px(x, y):
            px = int((x - x_min) / (x_max - x_min) * (mw - 4) + 2)
            py = int(mh - 2 - (y - y_min) / (y_max - y_min) * (mh - 4))
            return px, py

        # Grid lines
        for gx in np.arange(math.ceil(x_min/10)*10, x_max, 10):
            p1 = w2px(gx, y_min); p2 = w2px(gx, y_max)
            cv2.line(mini, p1, p2, (25,35,25), 1)
        for gy in np.arange(math.ceil(y_min/10)*10, y_max, 10):
            p1 = w2px(x_min, gy); p2 = w2px(x_max, gy)
            cv2.line(mini, p1, p2, (25,35,25), 1)

        # Path history — colour fades from dim to bright
        n = len(self.positions)
        pts_px = [w2px(p[0], p[1]) for p in self.positions]
        for i in range(n - 1):
            frac = i / max(n - 1, 1)
            g    = int(60 + 140 * frac)
            cv2.line(mini, pts_px[i], pts_px[i+1], (20, g, 20), 2)

        # LiDAR obstacles
        ego_x, ego_y = xs[-1], ys[-1]
        for det in lidar_dets:
            # Transform obstacle from ego frame to world frame
            d   = getattr(det, "distance_m", 0)
            lat = getattr(det, "lateral_m",  0)
            # Rotate by current heading
            ox = ego_x + d * math.sin(self.heading) + lat * math.cos(self.heading)
            oy = ego_y + d * math.cos(self.heading) - lat * math.sin(self.heading)
            op = w2px(ox, oy)
            threat = getattr(det, "threat_zone", "FAR").upper()
            col = ((0,0,200) if threat == "CRITICAL" else
                   (0,80,220) if threat == "CLOSE" else (0,140,200))
            cv2.rectangle(mini, (op[0]-3, op[1]-4), (op[0]+3, op[1]+2), col, -1)

        # Ego vehicle + heading arrow
        ep = w2px(ego_x, ego_y)
        cv2.rectangle(mini, (ep[0]-4, ep[1]-7), (ep[0]+4, ep[1]+3),
                      ACCENT_GREEN, -1)
        ax = int(ep[0] + math.sin(self.heading) * 12)
        ay = int(ep[1] - math.cos(self.heading) * 12)
        cv2.arrowedLine(mini, ep, (ax, ay), TEXT_WHITE, 1, tipLength=0.4)

        # Scale bar (10m)
        bar_px = int(10 / (x_max - x_min) * (mw - 4))
        cv2.line(mini, (6, mh-6), (6+bar_px, mh-6), TEXT_DIM, 1)
        cv2.putText(mini, "10m", (6, mh-9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.24, TEXT_DIM, 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(mini, (0,0), (mw-1, mh-1), (50,70,50), 1)
        return mini


# ═══════════════════════════════════════════════════════════════════════════════
# Lane detector (Hough-based, for real road markings)
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self):
        self._left_ema  = None
        self._right_ema = None
        self._alpha     = 0.20

    def detect(self, frame):
        h, w  = frame.shape[:2]
        roi_y = int(h * 0.52)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 130)
        mask  = np.zeros_like(edges)
        roi   = np.array([[0, h], [w, h],
                           [int(w*0.62), roi_y],
                           [int(w*0.38), roi_y]], dtype=np.int32)
        cv2.fillPoly(mask, [roi], 255)
        masked = cv2.bitwise_and(edges, mask)
        lines  = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                                  minLineLength=40, maxLineGap=100)
        if lines is None:
            return None, None, None

        left_s, right_s, cx = [], [], w / 2
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1: continue
            s = (y2 - y1) / (x2 - x1)
            if abs(s) < 0.3: continue
            if s < 0 and x1 < cx and x2 < cx: left_s.append((x1,y1,x2,y2))
            elif s > 0 and x1 > cx and x2 > cx: right_s.append((x1,y1,x2,y2))

        def _fit(segs, yb, yt):
            if not segs: return None
            xs = [x for x1,y1,x2,y2 in segs for x in (x1,x2)]
            ys = [y for x1,y1,x2,y2 in segs for y in (y1,y2)]
            try: m, b = np.polyfit(ys, xs, 1)
            except: return None
            return np.array([[int(m*yb+b), yb],[int(m*yt+b), yt]], dtype=np.int32)

        yb, yt = h - 5, roi_y + 20
        left  = _fit(left_s,  yb, yt)
        right = _fit(right_s, yb, yt)

        def _ema(p, c):
            if c is None: return p
            if p is None: return c.astype(np.float32)
            return p*(1-self._alpha) + c.astype(np.float32)*self._alpha

        self._left_ema  = _ema(self._left_ema,  left)
        self._right_ema = _ema(self._right_ema, right)
        lo = self._left_ema.astype(np.int32)  if self._left_ema  is not None else None
        ro = self._right_ema.astype(np.int32) if self._right_ema is not None else None
        poly = (np.array([lo[0], ro[0], ro[1], lo[1]], dtype=np.int32)
                if lo is not None and ro is not None else None)
        return lo, ro, poly

    def draw(self, frame, left, right, poly):
        if poly is not None:
            ov = frame.copy()
            cv2.fillPoly(ov, [poly], DRIVE_COL)
            cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
        if left  is not None:
            cv2.line(frame, tuple(left[0]),  tuple(left[1]),  LANE_LINE_COL, 2, cv2.LINE_AA)
        if right is not None:
            cv2.line(frame, tuple(right[0]), tuple(right[1]), LANE_LINE_COL, 2, cv2.LINE_AA)

    def steer(self, w):
        if self._left_ema is None or self._right_ema is None: return 0.0
        return float((((self._left_ema[1,0]+self._right_ema[1,0])/2) - w/2) / (w/2))


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _blend(img, x1, y1, x2, y2, color, alpha=0.6):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    ov = roi.copy()
    cv2.rectangle(ov, (0,0), (x2-x1, y2-y1), color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(ov, alpha, roi, 1-alpha, 0)

def _put(img, text, x, y, scale, color, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def _put_center(img, text, cx, y, scale, color, thick=1):
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _put(img, text, cx - tw//2, y, scale, color, thick)


def _draw_steering_gauge(img, cx, cy, r, steer_norm):
    sweep, start = 280, 220
    frac = float(np.clip((steer_norm + 1.0) / 2.0, 0.0, 1.0))
    cv2.ellipse(img, (cx,cy), (r,r), 0, -start, -(start-sweep), (50,50,50), 2, cv2.LINE_AA)
    col = ACCENT_GREEN if abs(steer_norm) < 0.12 else ACCENT_YELLOW
    a   = math.radians(start - frac * sweep)
    cv2.line(img, (cx,cy),
             (int(cx+(r-8)*math.cos(a)), int(cy-(r-8)*math.sin(a))),
             col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 3, col, -1, cv2.LINE_AA)
    _put(img, "L", cx-r-10, cy+4, 0.28, TEXT_DIM)
    _put(img, "R", cx+r+3,  cy+4, 0.28, TEXT_DIM)
    lbl = "LEFT" if steer_norm<-0.12 else "RIGHT" if steer_norm>0.12 else "CENTER"
    _put_center(img, lbl,       cx, cy+r+13, 0.28, TEXT_DIM)
    _put_center(img, "STEERING", cx, cy-r-6,  0.28, TEXT_DIM)


def _draw_speed_gauge(img, cx, cy, r, speed_kmh, max_kmh=80):
    sweep, start = 280, 220
    frac = float(np.clip(speed_kmh/max_kmh, 0.0, 1.0))
    cv2.ellipse(img, (cx,cy), (r,r), 0, -start, -(start-sweep), (50,50,50), 2, cv2.LINE_AA)
    col = (ACCENT_GREEN if speed_kmh<40 else ACCENT_YELLOW if speed_kmh<70 else WARN_RED)
    if frac > 0.01:
        cv2.ellipse(img, (cx,cy), (r-2,r-2), 0, -start, -(start-sweep*frac),
                    col, 2, cv2.LINE_AA)
    a = math.radians(start - frac*sweep)
    cv2.line(img, (cx,cy),
             (int(cx+(r-8)*math.cos(a)), int(cy-(r-8)*math.sin(a))),
             TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 3, TEXT_WHITE, -1, cv2.LINE_AA)
    _put_center(img, f"{speed_kmh:.0f}", cx, cy+5, 0.44, TEXT_WHITE)
    _put_center(img, "km/h",  cx, cy+r+13, 0.26, TEXT_DIM)
    _put_center(img, "SPEED", cx, cy-r-6,  0.28, TEXT_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent state
# ═══════════════════════════════════════════════════════════════════════════════
_lane = LaneDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# Main render
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard(image: np.ndarray,
                     frame_result: dict,
                     lidar_dets: list,
                     trajectory: "TrajectoryTracker" = None,
                     intrinsics=None,
                     cam_h: float = 1.0,
                     panel_frac: float = SIDEBAR_FRAC,
                     vlm_summary: str  = "",
                     vlm_fired: bool   = False) -> np.ndarray:
    global _vlm_buf, _vlm_ttl

    cam_img_h, cam_w = image.shape[:2]
    sb_w  = max(280, int(cam_w * panel_frac / (1.0 - panel_frac)))
    total = cam_w + sb_w

    # ── Unpack ────────────────────────────────────────────────────────────────
    raw_dec   = frame_result.get("decision", "UNKNOWN")
    decision  = DECISION_DISPLAY.get(raw_dec, raw_dec)
    risk      = frame_result.get("risk",        0.0)
    speed_kmh = frame_result.get("speed_mps",   0.0) * 3.6
    n_cam     = frame_result.get("n_cam_dets",  0)
    n_lidar   = frame_result.get("n_lidar_dets",0)
    lat_ms    = frame_result.get("latency_ms",  0.0)
    frame_idx = frame_result.get("frame_idx",   0)
    sf        = frame_result.get("sensory_frame")
    throttle  = frame_result.get("throttle",    0.0)
    brake_val = frame_result.get("brake",       0.0)

    dec_col   = DEC_COLORS.get(decision, DEC_COLORS["UNKNOWN"])
    is_danger = decision in ("STOP", "EMERGENCY")
    fps       = 1000.0 / max(lat_ms, 1.0)

    # ── Camera intrinsics for projection ─────────────────────────────────────
    if intrinsics is not None:
        fx = intrinsics.fx * (cam_w / intrinsics.width)
        fy = intrinsics.fy * (cam_img_h / intrinsics.height)
        cx = intrinsics.cx * (cam_w / intrinsics.width)
        cy = intrinsics.cy * (cam_img_h / intrinsics.height)
    else:
        fx = fy = cam_w * 0.65
        cx = cam_w / 2.0
        cy = cam_img_h * 0.52

    # ── Lane detection ────────────────────────────────────────────────────────
    cam = image.copy()
    left, right, poly = _lane.detect(cam)
    lane_steer = _lane.steer(cam_w)
    lane_detected = poly is not None

    # ── Ground-plane path projection (always on) ──────────────────────────────
    draw_ground_path(cam, fx, fy, cx, cy,
                     cam_h=cam_h,
                     steer=lane_steer,
                     corridor_w=1.4)

    # ── Lane overlay on top of ground path (if detected) ─────────────────────
    _lane.draw(cam, left, right, poly)

    # ── Minimap (trajectory + obstacles) ─────────────────────────────────────
    if trajectory is not None:
        mm_w = max(150, cam_w // 5)
        mm_h = max(120, int(mm_w * 0.75))
        mini = trajectory.draw_minimap(lidar_dets, size=(mm_w, mm_h))
        # Place in top-right corner of camera feed
        mx1 = cam_w - mm_w - 6
        my1 = 6
        _blend(cam, mx1-2, my1-2, cam_w-4, my1+mm_h+2, (0,0,0), 0.55)
        cam[my1:my1+mm_h, mx1:mx1+mm_w] = mini
        _put(cam, "MAP", mx1+4, my1+10, 0.30, TEXT_CYAN)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    if sf is not None and hasattr(sf, "detections"):
        for det in sf.detections:
            if getattr(det, "camera_name", "") == "lidar": continue
            b = det.bbox
            x1,y1,x2,y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            dist = det.depth_estimate
            lbl  = f"{b.class_name}{f' {dist:.1f}m' if dist else ''}"
            bc   = ((0,40,255) if (dist or 99)<6 else
                    (0,140,255) if (dist or 99)<15 else (40,200,40))
            cv2.rectangle(cam, (x1,y1), (x2,y2), bc, 2, cv2.LINE_AA)
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            _blend(cam, x1, max(0,y1-th-6), x1+tw+5, y1, (10,10,10), 0.72)
            _put(cam, lbl, x1+2, y1-3, 0.42, TEXT_WHITE)

    # ── Collision warning ─────────────────────────────────────────────────────
    if is_danger:
        cv2.rectangle(cam, (0,0), (cam_w-1, cam_img_h-1), WARN_RED, 8)
        _blend(cam, 0, 0, cam_w, 44, WARN_RED, 0.85)
        _put_center(cam, "!! COLLISION WARNING !!", cam_w//2, 30, 0.85, TEXT_WHITE, 2)

    # ── Top-left HUD ──────────────────────────────────────────────────────────
    _blend(cam, 0, 0, 320, 48, (0,0,0), 0.52)
    _put(cam, "VISION-ONLY AUTONOMOUS DRIVING", 8, 17, 0.46, TEXT_CYAN)
    _put(cam, f"FPS: {fps:.1f}  |  Speed: {speed_kmh:.0f} km/h", 8, 36, 0.38, ACCENT_GREEN)

    # ── Top-centre counts ─────────────────────────────────────────────────────
    cnt = f"Vehicles: {n_cam}   Persons: {n_lidar}"
    _put_center(cam, cnt, cam_w//2, 18, 0.40, TEXT_WHITE)

    # ── VLM strip ────────────────────────────────────────────────────────────
    if vlm_fired and vlm_summary:
        _vlm_buf = vlm_summary; _vlm_ttl = VLM_HOLD
    if _vlm_ttl > 0:
        _vlm_ttl -= 1
        _blend(cam, 0, cam_img_h-40, cam_w, cam_img_h-18, (4,4,40), 0.82)
        _put(cam, "PRISM AI", 10, cam_img_h-24, 0.36, (220,220,30))
        words, lines_out, cur = _vlm_buf.split(), [], ""
        for ww in words:
            test = (cur+" "+ww).strip()
            (tw2,_),_ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            if tw2 > cam_w-110: lines_out.append(cur); cur=ww
            else: cur=test
        if cur: lines_out.append(cur)
        for li,ln in enumerate(lines_out[:2]):
            _put(cam, ln, 100, cam_img_h-24+li*15, 0.36, (255,240,180))

    # ═══ Sidebar ══════════════════════════════════════════════════════════════
    sb  = np.full((cam_img_h, sb_w, 3), SIDEBAR_BG, dtype=np.uint8)
    pad = 8
    div = (30, 42, 30)

    # ── S1: Bird's Eye View ───────────────────────────────────────────────────
    bev_lh = 20
    bev_h  = int(cam_img_h * 0.32)
    _put_center(sb, "BIRD'S EYE VIEW", sb_w//2, bev_lh-3, 0.32, TEXT_DIM)

    # BEV panel — show trajectory + obstacles in top-down view
    bev_panel = sb[bev_lh:bev_h, pad:sb_w-pad]
    bpw, bph  = bev_panel.shape[1], bev_panel.shape[0]
    bcx, bey  = bpw//2, bph-16

    # Dashed center line with path curvature
    path_dx = int(lane_steer * bpw * 0.15)
    cv2.line(bev_panel, (bcx, bph), (bcx+path_dx, 0), (35,55,35), 1)
    lane_sep = bpw // 5
    cv2.line(bev_panel, (bcx-lane_sep, 0), (bcx-lane_sep, bph), (40,52,40), 1)
    cv2.line(bev_panel, (bcx+lane_sep, 0), (bcx+lane_sep, bph), (40,52,40), 1)

    # Path fill
    bev_path = np.array([
        [bcx-8, bey], [bcx+8, bey],
        [bcx+path_dx+3, 0], [bcx+path_dx-3, 0]
    ], dtype=np.int32)
    bov = bev_panel.copy()
    cv2.fillPoly(bov, [bev_path], (20,70,20))
    cv2.addWeighted(bov, 0.5, bev_panel, 0.5, 0, bev_panel)

    # Obstacles
    for det in lidar_dets:
        d   = min(getattr(det, "distance_m", 40), 40.0)
        lat = getattr(det, "lateral_m", 0.0)
        bx  = int(np.clip(bcx + lat*(bpw*0.3)/15, 2, bpw-2))
        by  = int(np.clip(bey - d*(bph-20)/40, 2, bph-2))
        f   = d/40.0
        cv2.rectangle(bev_panel, (bx-4,by-6),(bx+4,by+2),
                      (0,int(200*f),int(240*(1-f))), -1)

    # Ego
    cv2.rectangle(bev_panel, (bcx-5,bey-14),(bcx+5,bey),
                  ACCENT_GREEN, -1)
    cv2.rectangle(bev_panel, (bcx-5,bey-14),(bcx+5,bey), TEXT_WHITE, 1)

    cv2.line(sb, (0, bev_h), (sb_w, bev_h), div, 1)

    # ── S2: AI Decision ───────────────────────────────────────────────────────
    dec_y    = bev_h + 5
    _put_center(sb, "AI DECISION", sb_w//2, dec_y+16, 0.32, TEXT_DIM)
    ban_y1   = dec_y + 22
    ban_y2   = ban_y1 + 36
    cv2.rectangle(sb, (pad, ban_y1), (sb_w-pad, ban_y2), dec_col, -1)
    _put_center(sb, decision, sb_w//2, ban_y1+(ban_y2-ban_y1)//2+7,
                0.65, TEXT_WHITE, 2)

    sub = DEC_SUBTITLES.get(decision, "")
    if sub:
        _put_center(sb, sub, sb_w//2, ban_y2+14, 0.27, TEXT_WHITE)

    tbs_y = ban_y2 + 28
    cw    = sb_w // 3
    for i,(lbl,col) in enumerate([
        (f"Thr:{throttle:.1f}",       ACCENT_GREEN),
        (f"Brk:{brake_val:.1f}",      ACCENT_BLUE),
        (f"Str:{lane_steer:+.2f}",    ACCENT_ORANGE),
    ]):
        _put(sb, lbl, i*cw+pad//2, tbs_y, 0.30, col)

    div2_y = tbs_y + 14
    cv2.line(sb, (0, div2_y), (sb_w, div2_y), div, 1)

    # ── S3: Gauges ────────────────────────────────────────────────────────────
    gauge_y   = div2_y + 6
    remaining = cam_img_h - gauge_y - 75
    gauge_r   = max(20, min(32, remaining//2 - 16))
    gcx1, gcx2 = sb_w//4, 3*sb_w//4
    gcy       = gauge_y + gauge_r + 16
    _draw_steering_gauge(sb, gcx1, gcy, gauge_r, lane_steer)
    _draw_speed_gauge(sb, gcx2, gcy, gauge_r, speed_kmh)

    div3_y = gcy + gauge_r + 20
    cv2.line(sb, (0, div3_y), (sb_w, div3_y), div, 1)

    # ── S4: System Status ─────────────────────────────────────────────────────
    st_y     = div3_y + 12
    dangers  = 1 if is_danger else 0
    warnings = 1 if decision in ("CAUTION","YIELD","SLOW","EASE") else 0
    _put_center(sb, "SYSTEM STATUS", sb_w//2, st_y, 0.30, TEXT_DIM)
    for i,(txt,col) in enumerate([
        (f"Detections: {n_cam+n_lidar}",         TEXT_WHITE),
        (f"Dangers: {dangers}   Warnings: {warnings}",
         WARN_RED if dangers else TEXT_WHITE),
        (f"Lane Steer: {lane_steer:+.3f}",        ACCENT_YELLOW),
        (f"Frame: {frame_idx}",                   TEXT_DIM),
    ]):
        _put(sb, txt, pad, st_y+14+i*15, 0.28, col)

    # ── Compose ───────────────────────────────────────────────────────────────
    out = np.zeros((cam_img_h, total, 3), dtype=np.uint8)
    out[:, :cam_w] = cam
    out[:, cam_w:] = sb
    cv2.line(out, (cam_w,0), (cam_w,cam_img_h), (45,55,45), 1)
    return out
