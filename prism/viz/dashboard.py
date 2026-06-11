"""
PRISM - Dashboard Visualizer  (experimentation branch)
=======================================================
VLAM (Vision-Language-Action Model) simulation dashboard.
Shows what PRISM *would* command if it were in control of the vehicle.

Camera feed (~72%) | Dark sidebar (~28%)

Camera overlays:
  - Perspective ground path corridor (bends with optical-flow-derived steer)
  - Hough lane lines when detected (single set, no double-drawing)
  - Bounding boxes with class + metric distance
  - Collision warning banner

Sidebar (top to bottom):
  - SPATIAL AWARENESS (unified BEV + trajectory map)
  - AI DECISION banner + commanded Thr/Brk/Str
  - STEERING + SPEED gauges
  - SYSTEM STATUS
"""

import time
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
LANE_LINE_COL = ( 30,  30, 200)
WARN_RED      = ( 30,  30, 220)
PATH_FILL     = ( 20, 120,  20)
PATH_LINE     = ( 30,  30, 200)

DECISION_DISPLAY = {
    "CLEAR": "CRUISE", "MONITOR": "CRUISE",
    "EASE": "EASE", "SLOW": "SLOW",
    "CAUTION": "CAUTION", "YIELD": "YIELD",
    "STOP": "STOP", "EMERGENCY": "EMERGENCY",
}

DEC_COLORS = {
    "CRUISE":    ( 30, 200,  50),
    "EASE":      ( 30, 190, 120),
    "SLOW":      ( 20, 140, 220),
    "YIELD":     ( 20, 100, 230),
    "CAUTION":   ( 20,  70, 230),
    "STOP":      ( 20,  20, 210),
    "EMERGENCY": (  0,   0, 220),
    "UNKNOWN":   ( 60,  60,  60),
}

DEC_SUBTITLES = {
    "CRUISE":    "Road clear - cruising",
    "EASE":      "Easing off - caution ahead",
    "SLOW":      "Reducing speed - obstacle ahead",
    "YIELD":     "Yielding to road users",
    "CAUTION":   "Caution - obstacle detected",
    "STOP":      "Coming to a complete stop",
    "EMERGENCY": "EMERGENCY BRAKE",
    "UNKNOWN":   "",
}

_vlm_buf = ""
_vlm_ttl = 0
VLM_HOLD  = 200   # hold for ~10s at 20fps — matches VLM inference cycle
SIDEBAR_FRAC = 0.30


# ============================================================================
# Optical flow steering estimator
# ============================================================================

class FlowSteerEstimator:
    """
    Estimates vehicle yaw from dense optical flow.
    Much more reliable than Hough lane detection on unmarked roads.

    Physics:
      Right turn -> camera rotates right -> scene shifts LEFT -> u_flow < 0
      Left turn  -> camera rotates left  -> scene shifts RIGHT -> u_flow > 0
      Straight   -> diverging flow, mean ~0
    """
    def __init__(self, alpha: float = 0.25):
        self._ema  = 0.0
        self._alpha = alpha   # EMA smoothing (lower = smoother, more lag)

    def update(self, flow: np.ndarray) -> float:
        """Returns smoothed steer estimate in [-1, 1]."""
        if flow is None:
            self._ema *= (1 - self._alpha)
            return self._ema

        h, w = flow.shape[:2]
        # Sample lower-centre patch (road surface, avoid sky and edges)
        y1, y2 = int(h * 0.45), int(h * 0.82)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        patch_u = flow[y1:y2, x1:x2, 0]

        if patch_u.size == 0:
            return self._ema

        mean_u = float(patch_u.mean())
        # Right turn -> mean_u negative -> positive steer (right)
        raw = float(np.clip(-mean_u / 10.0, -1.0, 1.0))
        self._ema = (1 - self._alpha) * self._ema + self._alpha * raw
        # Dead-band: suppress tiny jitter
        if abs(self._ema) < 0.03:
            self._ema = 0.0
        return self._ema

    @property
    def value(self) -> float:
        return self._ema


# ============================================================================
# Trajectory tracker  (bounded history, no long-term spiral)
# ============================================================================

class TrajectoryTracker:
    """
    Dead-reckoning trajectory using optical-flow-derived steer.
    History is bounded to MAX_SECS seconds to prevent drift accumulation.
    """
    MAX_SECS = 8.0   # only keep last N seconds of path

    def __init__(self, max_points: int = 400):
        self.positions: deque = deque(maxlen=max_points)
        self.timestamps: deque = deque(maxlen=max_points)
        self.x       = 0.0
        self.y       = 0.0
        self.heading = 0.0
        self._last_ts = None

    def update(self, speed_mps: float, flow_steer: float, timestamp: float):
        if self._last_ts is None:
            self._last_ts = timestamp
            self.positions.append((self.x, self.y))
            self.timestamps.append(timestamp)
            return

        dt = float(np.clip(timestamp - self._last_ts, 0.0, 0.12))
        self._last_ts = timestamp

        if dt > 0 and speed_mps > 0.1:
            # Heading rate: flow_steer scaled by speed (faster = tighter turns)
            turn_rate = flow_steer * min(speed_mps, 8.0) * 0.18
            self.heading += turn_rate * dt
            self.x += speed_mps * dt * math.sin(self.heading)
            self.y += speed_mps * dt * math.cos(self.heading)

        self.positions.append((self.x, self.y))
        self.timestamps.append(timestamp)

        # Trim positions older than MAX_SECS
        while (len(self.timestamps) > 2 and
               timestamp - self.timestamps[0] > self.MAX_SECS):
            self.positions.popleft()
            self.timestamps.popleft()


# ============================================================================
# Unified spatial awareness panel (BEV + trajectory merged)
# ============================================================================

def draw_spatial_panel(panel: np.ndarray,
                       lidar_dets: list,
                       yolo_dets: list,
                       flow_steer: float,
                       speed_mps: float,
                       metric_dets: list = None):
    """
    Forward-only top-down spatial view.
    Detection-aware: panel brightens and highlights when obstacles are present,
    shows CLEAR indicator when empty, labels every detection with distance.
    """
    h, w = panel.shape[:2]
    view_fwd = 30.0
    view_lat = 15.0
    ex, ey   = w // 2, h - 22

    PERSON_CLS  = {"person"}
    VEHICLE_CLS = {"car", "truck", "bus"}
    BIKE_CLS    = {"bicycle", "motorcycle"}
    SIGNAL_CLS  = {"traffic light", "stop sign"}
    CORRIDOR_LAT_DISPLAY = 3.5   # lateral limit for spatial panel display

    def w2p(fwd_m, lat_m):
        px = int(np.clip(ex + lat_m / view_lat * (w // 2 - 4), 2, w - 2))
        py = int(np.clip(ey - fwd_m / view_fwd * (h - 30),     2, h - 2))
        return px, py

    # ── Background + grid ─────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (w-1, h-1), (14, 20, 14), -1)
    for fm in range(0, int(view_fwd)+1, 5):
        p1 = w2p(fm, -view_lat); p2 = w2p(fm, view_lat)
        cv2.line(panel, p1, p2, (24, 36, 24), 1)
        if fm > 0 and fm % 10 == 0:
            cv2.putText(panel, f"{fm}m", (p1[0]+2, p1[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, TEXT_DIM, 1)
    for lm in range(-int(view_lat), int(view_lat)+1, 5):
        cv2.line(panel, w2p(0, lm), w2p(view_fwd, lm), (24, 36, 24), 1)

    # ── Pre-scan: validate and collect detections ─────────────────────────
    valid_metric = []
    if metric_dets:
        for md in metric_dets:
            dist = float(md.distance_m)
            lat  = float(md.lateral_m)
            cls  = getattr(md, "class_name", "unknown")
            if dist < 1.0 or dist >= 49.0: continue
            max_d = 25.0 if cls in PERSON_CLS else 30.0
            if dist > max_d:    continue
            if abs(lat) > 8.0:  continue
            if getattr(md, "confidence", 1.0) < 0.38: continue
            valid_metric.append((dist, lat, cls))

    valid_lidar = []
    for det in lidar_dets:
        d   = getattr(det, "distance_m", 0)
        lat = getattr(det, "lateral_m",  0)
        n   = getattr(det, "n_points",   0)
        h_m = getattr(det, "height_m",   1.0)
        if d <= 2.0 or d > view_fwd:        continue
        if abs(lat) > CORRIDOR_LAT_DISPLAY:  continue
        if n < 12:                           continue
        if h_m < 0.15:                       continue
        valid_lidar.append((d, lat, det))

    # Corridor occupancy — drives visual severity
    corridor_dists = [dist for dist, lat, _ in valid_metric if abs(lat) < 2.5]
    corridor_dists += [d   for d, lat, _   in valid_lidar   if abs(lat) < 2.5]
    has_corridor   = len(corridor_dists) > 0
    nearest_m      = min(corridor_dists) if corridor_dists else None
    total_dets     = len(valid_metric) + len(valid_lidar)

    # ── Path corridor — brightness reacts to corridor occupancy ──────────
    dists = np.array([0, 1, 2, 3, 4.5, 6, 8, 10.5, 13, 16, 20, 25])
    lpts, rpts = [], []
    for d in dists:
        lat_c = flow_steer * (d ** 1.5) * 0.07
        lpts.append(w2p(d, lat_c - 1.4))
        rpts.append(w2p(d, lat_c + 1.4))
    poly = np.array(lpts + rpts[::-1], np.int32)
    ov   = panel.copy()

    if nearest_m is not None and nearest_m < 8:
        fill_col   = (10, 60, 140)   # orange tint — threat close
        line_col   = (30, 80, 220)
        fill_alpha = 0.70
    elif has_corridor:
        fill_col   = (10, 80, 60)    # yellow-green — threat present
        line_col   = (30, 50, 180)
        fill_alpha = 0.60
    else:
        fill_col   = (10, 45, 10)    # dim green — all clear
        line_col   = (22, 22, 130)
        fill_alpha = 0.42

    cv2.fillPoly(ov, [poly], fill_col)
    cv2.addWeighted(ov, fill_alpha, panel, 1.0 - fill_alpha, 0, panel)
    for i in range(len(lpts)-1):
        cv2.line(panel, lpts[i], lpts[i+1], line_col, 1)
        cv2.line(panel, rpts[i], rpts[i+1], line_col, 1)
    for i in range(0, len(dists)-1, 2):
        lat_c = flow_steer * (dists[i] ** 1.5) * 0.07
        cv2.line(panel,
                 w2p(dists[i], lat_c),
                 w2p(dists[min(i+1, len(dists)-1)], lat_c),
                 (90, 90, 90), 1)

    # ── Colours ───────────────────────────────────────────────────────────
    COL_PERSON     = (0,   50, 230)
    COL_VEHICLE    = (0,  220, 220)
    COL_STATIONARY = (30, 180,  30)
    COL_BIKE       = (0,  200, 255)

    # ── YOLO / MetricDepth detections ─────────────────────────────────────
    for dist, lat, cls in valid_metric:
        op = w2p(dist, lat)
        if   cls in PERSON_CLS:  col, sym = COL_PERSON,     "P"
        elif cls in VEHICLE_CLS: col, sym = COL_VEHICLE,    "V"
        elif cls in BIKE_CLS:    col, sym = COL_BIKE,       "B"
        elif cls in SIGNAL_CLS:  col, sym = COL_STATIONARY, "S"
        else:                    col, sym = (150, 150, 150), "?"
        cv2.circle(panel, op, 7, col, -1)
        cv2.circle(panel, op, 7, TEXT_WHITE, 1)
        cv2.putText(panel, sym, (op[0]-3, op[1]+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, TEXT_WHITE, 1)
        cv2.putText(panel, f"{dist:.0f}m", (op[0]+9, op[1]+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.20, col, 1)

    # ── LiDAR detections ──────────────────────────────────────────────────
    for d, lat, det in valid_lidar:
        op = w2p(d, lat)
        cv2.rectangle(panel, (op[0]-5, op[1]-6), (op[0]+5, op[1]+2),
                      COL_STATIONARY, -1)
        cv2.rectangle(panel, (op[0]-5, op[1]-6), (op[0]+5, op[1]+2),
                      TEXT_WHITE, 1)
        cv2.putText(panel, f"{d:.0f}m", (op[0]+7, op[1]+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.20, COL_STATIONARY, 1)

    # ── Status overlay ────────────────────────────────────────────────────
    if total_dets == 0:
        cv2.putText(panel, "CLEAR", (ex - 16, ey - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, ACCENT_GREEN, 1)
    else:
        cv2.putText(panel, f"{total_dets} obj", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, TEXT_WHITE, 1)
        if nearest_m is not None:
            warn_col = WARN_RED if nearest_m < 6 else ACCENT_YELLOW
            cv2.putText(panel, f"!{nearest_m:.0f}m", (w - 34, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.24, warn_col, 1)

    # ── Ego vehicle ───────────────────────────────────────────────────────
    cv2.rectangle(panel, (ex-5, ey-12), (ex+5, ey), ACCENT_GREEN, -1)
    cv2.rectangle(panel, (ex-5, ey-12), (ex+5, ey), TEXT_WHITE, 1)
    hdiff = int(flow_steer * 14)
    cv2.arrowedLine(panel, (ex, ey-12), (ex+hdiff, ey-26),
                    TEXT_WHITE, 1, tipLength=0.4)

    # Legend removed — colour coding is self-evident

    cv2.rectangle(panel, (0, 0), (w-1, h-1), (45, 65, 45), 1)


# ============================================================================
# Ground-plane path projection
# ============================================================================

def _project_gnd(x_lat, d_fwd, fx, fy, cx, cy, cam_h):
    if d_fwd < 0.05: return None
    u = fx * x_lat / d_fwd + cx
    v = fy * cam_h  / d_fwd + cy
    return (int(u), int(v))


def draw_ground_path(frame: np.ndarray,
                     fx, fy, cx, cy,
                     cam_h: float = 1.0,
                     flow_steer: float = 0.0,
                     lane_poly=None):
    """
    Perspective-correct ground corridor using optical-flow steer.
    Green fill always drawn. Red boundary lines only when Hough didn't find lanes.
    Anchored to image bottom.
    """
    h, w = frame.shape[:2]
    # Dense near, sparse far — gives smooth close-in curve, stable horizon
    dists = np.array([0.2, 0.4, 0.7, 1.1, 1.6, 2.3, 3.2, 4.5,
                      6.0, 8.0, 10.5, 14.0, 18.0, 24.0, 32.0])
    corridor_w = 1.40

    lpts, rpts = [], []
    for d in dists:
        lat_c = flow_steer * (d ** 1.5) * 0.07
        pl = _project_gnd(lat_c - corridor_w, d, fx, fy, cx, cy, cam_h)
        pr = _project_gnd(lat_c + corridor_w, d, fx, fy, cx, cy, cam_h)
        if pl is None or pr is None: continue
        vl = min(pl[1], h-1)
        vr = min(pr[1], h-1)
        lpts.append((pl[0], vl))
        rpts.append((pr[0], vr))

    if len(lpts) < 2: return

    # Deduplicate bottom-clamped points
    def dedup(pts):
        seen, out = set(), []
        for p in pts:
            if p[1] not in seen:
                seen.add(p[1]); out.append(p)
        return out

    lpts = dedup(lpts)
    rpts = dedup(rpts)
    if len(lpts) < 2: return

    # Green fill
    poly = np.array(lpts + rpts[::-1], np.int32)
    ov   = frame.copy()
    cv2.fillPoly(ov, [poly], PATH_FILL)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)

    # Red boundary lines — only if Hough didn't detect lanes
    if lane_poly is None:
        for i in range(len(lpts)-1):
            cv2.line(frame, lpts[i], lpts[i+1], PATH_LINE, 2, cv2.LINE_AA)
            cv2.line(frame, rpts[i], rpts[i+1], PATH_LINE, 2, cv2.LINE_AA)

    # Centre dashed line
    for d_idx in range(0, len(dists)-1, 2):
        d    = dists[d_idx]
        lat_c = flow_steer * (d ** 1.5) * 0.07
        pc   = _project_gnd(lat_c, d, fx, fy, cx, cy, cam_h)
        pc2  = _project_gnd(flow_steer * (dists[min(d_idx+1, len(dists)-1)] ** 1.5) * 0.07,
                            dists[min(d_idx+1, len(dists)-1)], fx, fy, cx, cy, cam_h)
        if pc and pc2:
            cv2.line(frame,
                     (pc[0],  min(pc[1],  h-1)),
                     (pc2[0], min(pc2[1], h-1)),
                     (160, 160, 160), 1, cv2.LINE_AA)


# ============================================================================
# Lane detector
# ============================================================================

class LaneDetector:
    def __init__(self):
        self._left_ema  = None
        self._right_ema = None
        self._alpha     = 0.18

    def detect(self, frame):
        h, w  = frame.shape[:2]
        roi_y = int(h * 0.52)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 130)
        mask  = np.zeros_like(edges)
        roi   = np.array([[0,h],[w,h],[int(w*0.62),roi_y],[int(w*0.38),roi_y]], np.int32)
        cv2.fillPoly(mask, [roi], 255)
        masked = cv2.bitwise_and(edges, mask)
        lines  = cv2.HoughLinesP(masked, 1, np.pi/180, 30,
                                  minLineLength=40, maxLineGap=100)
        if lines is None: return None, None, None

        left_s, right_s, cxm = [], [], w/2
        for x1,y1,x2,y2 in lines[:,0]:
            if x2==x1: continue
            s = (y2-y1)/(x2-x1)
            if abs(s)<0.3: continue
            if s<0 and x1<cxm and x2<cxm: left_s.append((x1,y1,x2,y2))
            elif s>0 and x1>cxm and x2>cxm: right_s.append((x1,y1,x2,y2))

        def _fit(segs, yb, yt):
            if not segs: return None
            xs=[x for x1,y1,x2,y2 in segs for x in (x1,x2)]
            ys=[y for x1,y1,x2,y2 in segs for y in (y1,y2)]
            try: m,b=np.polyfit(ys,xs,1)
            except: return None
            return np.array([[int(m*yb+b),yb],[int(m*yt+b),yt]],np.int32)

        yb,yt = h-5,roi_y+20
        left  = _fit(left_s,yb,yt)
        right = _fit(right_s,yb,yt)

        def _ema(p,c):
            if c is None: return p
            if p is None: return c.astype(np.float32)
            return p*(1-self._alpha)+c.astype(np.float32)*self._alpha

        self._left_ema  = _ema(self._left_ema, left)
        self._right_ema = _ema(self._right_ema, right)
        lo = self._left_ema.astype(np.int32)  if self._left_ema  is not None else None
        ro = self._right_ema.astype(np.int32) if self._right_ema is not None else None
        poly = (np.array([lo[0],ro[0],ro[1],lo[1]],np.int32)
                if lo is not None and ro is not None else None)
        return lo, ro, poly

    def draw_lines_only(self, frame, left, right):
        if left  is not None:
            cv2.line(frame,tuple(left[0]), tuple(left[1]), LANE_LINE_COL,2,cv2.LINE_AA)
        if right is not None:
            cv2.line(frame,tuple(right[0]),tuple(right[1]),LANE_LINE_COL,2,cv2.LINE_AA)

    def steer(self, w):
        if self._left_ema is None or self._right_ema is None: return 0.0
        return float((((self._left_ema[1,0]+self._right_ema[1,0])/2)-w/2)/(w/2))


# ============================================================================
# Gauge helpers
# ============================================================================

def _blend(img, x1,y1,x2,y2, color, alpha=0.6):
    roi=img[y1:y2,x1:x2]
    if roi.size==0: return
    ov=roi.copy()
    cv2.rectangle(ov,(0,0),(x2-x1,y2-y1),color,-1)
    img[y1:y2,x1:x2]=cv2.addWeighted(ov,alpha,roi,1-alpha,0)

def _put(img,text,x,y,scale,color,thick=1):
    cv2.putText(img,str(text),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def _putc(img,text,cx,y,scale,color,thick=1):
    (tw,_),_=cv2.getTextSize(str(text),cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
    _put(img,text,cx-tw//2,y,scale,color,thick)

def _gauge(img,cx,cy,r,frac,col_arc,label_top,label_bot,val_str):
    sw,st=280,220
    frac=float(np.clip(frac,0,1))
    cv2.ellipse(img,(cx,cy),(r,r),0,-st,-(st-sw),(50,50,50),2,cv2.LINE_AA)
    if frac>0.01:
        cv2.ellipse(img,(cx,cy),(r-2,r-2),0,-st,-(st-sw*frac),col_arc,2,cv2.LINE_AA)
    a=math.radians(st-frac*sw)
    cv2.line(img,(cx,cy),(int(cx+(r-8)*math.cos(a)),int(cy-(r-8)*math.sin(a))),
             TEXT_WHITE,2,cv2.LINE_AA)
    cv2.circle(img,(cx,cy),3,TEXT_WHITE,-1,cv2.LINE_AA)
    if val_str: _putc(img,val_str,cx,cy+5,0.40,TEXT_WHITE)
    _putc(img,label_top,cx,cy-r-7,0.25,TEXT_DIM)
    _putc(img,label_bot,cx,cy+r+12,0.24,TEXT_DIM)

def _steer_gauge(img,cx,cy,r,steer):
    sw,st=280,220
    frac=float(np.clip((steer+1)/2,0,1))
    cv2.ellipse(img,(cx,cy),(r,r),0,-st,-(st-sw),(50,50,50),2,cv2.LINE_AA)
    col=ACCENT_GREEN if abs(steer)<0.12 else ACCENT_YELLOW
    a=math.radians(st-frac*sw)
    cv2.line(img,(cx,cy),(int(cx+(r-8)*math.cos(a)),int(cy-(r-8)*math.sin(a))),
             col,2,cv2.LINE_AA)
    cv2.circle(img,(cx,cy),3,col,-1,cv2.LINE_AA)
    _put(img,"L",cx-r-10,cy+4,0.24,TEXT_DIM)
    _put(img,"R",cx+r+3, cy+4,0.24,TEXT_DIM)
    lbl="LEFT" if steer<-0.12 else "RIGHT" if steer>0.12 else "CENTER"
    _putc(img,lbl,     cx,cy+r+12,0.24,TEXT_DIM)
    _putc(img,"STEER", cx,cy-r-7, 0.24,TEXT_DIM)


# ============================================================================
# Persistent state
# ============================================================================
_lane      = LaneDetector()
_flow_est  = FlowSteerEstimator(alpha=0.25)


# ============================================================================
# Main render
# ============================================================================

def render_dashboard(image: np.ndarray,
                     frame_result: dict,
                     lidar_dets: list,
                     trajectory: TrajectoryTracker = None,
                     intrinsics = None,
                     cam_h: float = 1.0,
                     optical_flow: np.ndarray = None,
                     metric_dets: list = None,
                     panel_frac: float = SIDEBAR_FRAC,
                     vlm_summary: str  = "",
                     vlm_fired: bool   = False) -> np.ndarray:
    global _vlm_buf, _vlm_ttl

    cam_h_px, cam_w = image.shape[:2]
    sb_w  = max(290, int(cam_w * panel_frac / (1.0 - panel_frac)))

    # ── Unpack ────────────────────────────────────────────────────────────────
    raw_dec   = frame_result.get("decision",    "UNKNOWN")
    decision  = DECISION_DISPLAY.get(raw_dec, raw_dec)
    speed_mps = frame_result.get("speed_mps",   0.0)
    speed_kmh = speed_mps * 3.6
    n_cam     = frame_result.get("n_cam_dets",  0)
    n_lidar   = frame_result.get("n_lidar_dets",0)
    lat_ms    = frame_result.get("latency_ms",  0.0)
    frame_idx = frame_result.get("frame_idx",   0)
    sf        = frame_result.get("sensory_frame")
    throttle  = frame_result.get("throttle",    0.0)
    brake_val = frame_result.get("brake",       0.0)
    timestamp = frame_result.get("timestamp",   0.0)

    dec_col   = DEC_COLORS.get(decision, DEC_COLORS["UNKNOWN"])
    is_danger = decision in ("STOP", "EMERGENCY")
    fps       = 1000.0 / max(lat_ms, 1.0)

    # ── Optical flow steer (primary steering signal) ──────────────────────────
    flow_steer = _flow_est.update(optical_flow)

    # ── Update trajectory with flow-based steer ───────────────────────────────
    if trajectory is not None and timestamp > 0:
        trajectory.update(speed_mps, flow_steer, timestamp)

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    if intrinsics is not None:
        fx   = float(intrinsics.fx) * (cam_w / intrinsics.width)
        fy   = float(intrinsics.fy) * (cam_h_px / intrinsics.height)
        cx_k = float(intrinsics.cx) * (cam_w / intrinsics.width)
        cy_k = float(intrinsics.cy) * (cam_h_px / intrinsics.height)
    else:
        fx = fy  = cam_w * 0.65
        cx_k     = cam_w / 2.0
        cy_k     = cam_h_px * 0.52

    # ── Lane detection (geometry only — no Hough lines drawn) ────────────────
    cam  = image.copy()
    _, _, poly = _lane.detect(cam)
    lane_steer = _lane.steer(cam_w)

    # ── Ground path with FLOW steer — smooth projection lines only ───────────
    # Pass lane_poly=None so projection always draws the boundary lines
    # (Hough disabled — it causes choppy/crossing artefacts on unmarked roads)
    draw_ground_path(cam, fx, fy, cx_k, cy_k,
                     cam_h=cam_h, flow_steer=flow_steer, lane_poly=None)

    # ── Bounding boxes — colour-coded by class ────────────────────────────────
    PERSON_CLS  = {"person"}
    VEHICLE_CLS = {"car","truck","bus"}
    BIKE_CLS    = {"bicycle","motorcycle"}
    SIGNAL_CLS  = {"traffic light","stop sign"}

    yolo_dets = []
    if sf is not None and hasattr(sf, "detections"):
        for det in sf.detections:
            if getattr(det,"camera_name","")=="lidar": continue
            b    = det.bbox
            cls  = b.class_name
            dist = det.depth_estimate
            yolo_dets.append(det)

            x1,y1,x2,y2 = int(b.x1),int(b.y1),int(b.x2),int(b.y2)
            lbl = f"{cls}{f' {dist:.1f}m' if dist else ''}"

            # Colour by class type
            if   cls in PERSON_CLS:   bc = (0, 50, 230)   # red
            elif cls in VEHICLE_CLS:  bc = (40,200, 40)   # green
            elif cls in BIKE_CLS:     bc = (20,220,220)   # yellow
            elif cls in SIGNAL_CLS:   bc = (220,180, 20)  # cyan
            else:                     bc = (150,150,150)

            cv2.rectangle(cam,(x1,y1),(x2,y2),bc,2,cv2.LINE_AA)
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            _blend(cam,x1,max(0,y1-th-6),x1+tw+5,y1,(10,10,10),0.72)
            _put(cam,lbl,x1+2,y1-3,0.42,TEXT_WHITE)

    # ── Collision warning ─────────────────────────────────────────────────────
    if is_danger:
        cv2.rectangle(cam,(0,0),(cam_w-1,cam_h_px-1),WARN_RED,8)
        _blend(cam,0,0,cam_w,44,WARN_RED,0.85)
        _putc(cam,"!! COLLISION WARNING !!",cam_w//2,30,0.82,TEXT_WHITE,2)

    # ── Top-left HUD ──────────────────────────────────────────────────────────
    _blend(cam,0,0,310,48,(0,0,0),0.52)
    _put(cam,"PRISM  AUTONOMOUS  PERCEPTION",8,17,0.44,TEXT_CYAN)
    spd_str = f"~{speed_kmh:.0f}" if speed_kmh > 1.0 else "0"
    _put(cam,f"FPS: {fps:.1f}  |  {spd_str} km/h",8,36,0.36,ACCENT_GREEN)

    # ── Top-centre: real YOLO counts ─────────────────────────────────────────
    n_persons  = sum(1 for d in yolo_dets if d.bbox.class_name in PERSON_CLS)
    n_vehicles = sum(1 for d in yolo_dets
                     if d.bbox.class_name in VEHICLE_CLS | BIKE_CLS)
    _putc(cam,f"Vehicles: {n_vehicles}   Persons: {n_persons}",
          cam_w//2,18,0.38,TEXT_WHITE)

    # ── VLM strip ────────────────────────────────────────────────────────────
    if vlm_fired and vlm_summary:
        _vlm_buf=vlm_summary; _vlm_ttl=VLM_HOLD
    if _vlm_ttl>0:
        _vlm_ttl-=1
        _blend(cam,0,cam_h_px-40,cam_w,cam_h_px-18,(4,4,40),0.82)
        _put(cam,"PRISM AI",10,cam_h_px-24,0.34,(220,220,30))
        words,lines_out,cur=_vlm_buf.split(),[],""
        for ww in words:
            test=(cur+" "+ww).strip()
            (tw2,_),_=cv2.getTextSize(test,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
            if tw2>cam_w-110: lines_out.append(cur);cur=ww
            else: cur=test
        if cur: lines_out.append(cur)
        for li,ln in enumerate(lines_out[:2]):
            _put(cam,ln,100,cam_h_px-24+li*14,0.34,(255,240,180))

    # ======== SIDEBAR =========================================================
    sb  = np.full((cam_h_px, sb_w, 3), SIDEBAR_BG, dtype=np.uint8)
    pad = 8
    div = (30, 42, 30)
    y   = 0

    # ── S1: SPATIAL AWARENESS (unified BEV + trajectory) ─────────────────────
    sa_lh = 20
    sa_h  = int(cam_h_px * 0.36)   # reduced from 0.42 to make room for reasoning strip
    _putc(sb,"SPATIAL AWARENESS",sb_w//2,sa_lh-3,0.30,TEXT_DIM)
    sa_panel = sb[sa_lh:sa_h, pad:sb_w-pad].copy()
    draw_spatial_panel(sa_panel, lidar_dets, yolo_dets,
                       flow_steer, speed_mps, metric_dets=metric_dets)
    sb[sa_lh:sa_h, pad:sb_w-pad] = sa_panel
    cv2.line(sb,(0,sa_h),(sb_w,sa_h),div,1)
    y = sa_h

    # ── S2: AI DECISION ───────────────────────────────────────────────────────
    y += 5
    _putc(sb,"AI DECISION",sb_w//2,y+16,0.30,TEXT_DIM)
    by1=y+22; by2=by1+36
    cv2.rectangle(sb,(pad,by1),(sb_w-pad,by2),dec_col,-1)
    _putc(sb,decision,sb_w//2,by1+(by2-by1)//2+7,0.64,TEXT_WHITE,2)

    sub=DEC_SUBTITLES.get(decision,"")
    if sub: _putc(sb,sub,sb_w//2,by2+13,0.26,TEXT_WHITE)

    tbs_y=by2+26; cw=sb_w//3
    for i,(lbl,col) in enumerate([
        (f"Thr:{throttle:.1f}",  ACCENT_GREEN),
        (f"Brk:{brake_val:.1f}", ACCENT_BLUE),
        (f"Str:{flow_steer:+.2f}", ACCENT_ORANGE),
    ]):
        _put(sb,lbl,i*cw+pad//2,tbs_y,0.28,col)

    div2_y=tbs_y+14; cv2.line(sb,(0,div2_y),(sb_w,div2_y),div,1)

    # ── S3: GAUGES ────────────────────────────────────────────────────────────
    gy=div2_y+6
    rem=cam_h_px-gy-68
    gr=max(18,min(30,rem//2-14))
    gcy=gy+gr+14

    _steer_gauge(sb,sb_w//4,gcy,gr,flow_steer)
    speed_col=(ACCENT_GREEN if speed_kmh<40 else
               ACCENT_YELLOW if speed_kmh<70 else WARN_RED)
    _gauge(sb,3*sb_w//4,gcy,gr,
           speed_kmh/80.0,speed_col,
           "SPEED","km/h",f"{speed_kmh:.0f}")

    div3_y=gcy+gr+20; cv2.line(sb,(0,div3_y),(sb_w,div3_y),div,1)

    # ── S4: SCENE INTELLIGENCE ────────────────────────────────────────────────────
    st_y  = div3_y + 10
    scene = frame_result.get("scene_assessment")
    _putc(sb, "SCENE INTELLIGENCE", sb_w // 2, st_y, 0.28, TEXT_DIM)

    si_lines = []   # (text, color) pairs

    if scene is not None:
        # Primary scene understanding — strip non-ASCII (Hershey font is ASCII-only)
        raw_reason = scene.primary_reason or "Scanning scene..."
        reason = ''.join(c if ord(c) < 128 else '-' for c in raw_reason)
        if len(reason) > 38:
            reason = reason[:36] + ".."
        threat_col = (WARN_RED if scene.closest_threat_m < 8
                      else ACCENT_YELLOW if scene.closest_threat_m < 20
                      else TEXT_WHITE)
        si_lines.append((reason, threat_col))

        # Actor breakdown
        parts = []
        if scene.ped_count > 0:
            parts.append(f"{scene.ped_count} person{'s' if scene.ped_count>1 else ''}")
        if scene.vehicle_count > 0:
            parts.append(f"{scene.vehicle_count} vehicle{'s' if scene.vehicle_count>1 else ''}")
        if not parts:
            parts.append("No actors detected")
        actor_str = "  ".join(parts)
        if scene.closest_threat_m < 90:
            actor_str += f"  |  Nearest {scene.closest_threat_m:.1f}m"
        si_lines.append((actor_str, TEXT_WHITE))

        # Corridor status
        if scene.actors_in_corridor > 0:
            si_lines.append((f"! {scene.actors_in_corridor} in corridor", WARN_RED))
        elif scene.actors_approaching > 0:
            si_lines.append((f"{scene.actors_approaching} approaching", ACCENT_YELLOW))
        else:
            si_lines.append(("Corridor clear", ACCENT_GREEN))
    else:
        si_lines.append(("Initialising...", TEXT_DIM))
        si_lines.append(("", TEXT_DIM))
        si_lines.append(("", TEXT_DIM))

    # Telemetry footer
    si_lines.append((
        f"Fr:{frame_idx}  Str:{flow_steer:+.2f}  Cam:{n_cam}  Lid:{n_lidar}",
        TEXT_DIM
    ))

    for i, (txt, col) in enumerate(si_lines[:5]):
        if txt:
            _put(sb, txt, pad, st_y + 14 + i * 14, 0.26, col)

    # ── S5: LIVE REASONING (VLM when fired, SmartDecision secondary otherwise) ──
    si_bottom = st_y + 14 + 5 * 14 + 6
    cv2.line(sb, (0, si_bottom), (sb_w, si_bottom), div, 1)
    lr_y = si_bottom + 8

    vlm_out     = frame_result.get("vlm_output")
    vlm_is_real = frame_result.get("vlm_is_real", False)
    hdr_tag     = "LIVE REASONING" if vlm_is_real else "LIVE REASONING (SIM)"
    _putc(sb, hdr_tag, sb_w // 2, lr_y, 0.27, TEXT_DIM)

    reasoning_lines = []

    if vlm_out is not None and vlm_out.success:
        age     = time.time() - vlm_out.timestamp
        # Fresh < 8s (one keepalive cycle), stale after
        age_col = (TEXT_WHITE if age < 4.0 else
                   ACCENT_YELLOW if age < 8.0 else TEXT_DIM)

        # Scene context — richer than summary
        raw_ctx = vlm_out.scene_context or vlm_out.scene_summary or ""
        ctx = ''.join(c if ord(c) < 128 else '-' for c in raw_ctx)
        if len(ctx) > 40: ctx = ctx[:38] + ".."
        prefix = "VLM" if vlm_is_real else "SIM"
        if ctx:
            reasoning_lines.append((f"{prefix}: {ctx}", age_col))

        # Scene summary (second line if different)
        raw_sum = vlm_out.scene_summary or ""
        summ = ''.join(c if ord(c) < 128 else '-' for c in raw_sum)
        if summ and summ != raw_ctx:
            if len(summ) > 40: summ = summ[:38] + ".."
            reasoning_lines.append((summ, age_col))

        # Risk flags
        for flag in vlm_out.risk_flags[:2]:
            flag_str = ''.join(c if ord(c) < 128 else '-' for c in flag)
            flag_str = flag_str[:38] if len(flag_str) > 38 else flag_str
            reasoning_lines.append((f"! {flag_str}", ACCENT_YELLOW))

        # Caution level + freshness
        caution  = vlm_out.recommended_caution.upper()
        fresh_bar = "*" * max(1, min(5, int(5 * max(0, 1 - age/8.0))))
        reasoning_lines.append((
            f"Caution:{caution}  {fresh_bar}  {age:.0f}s",
            WARN_RED if caution in ("HIGH","CRITICAL") else age_col
        ))
    else:
        # VLM not yet returned — show SmartDecision live data
        if scene is not None and scene.secondary_reasons:
            for r in scene.secondary_reasons[:2]:
                r_str = r[:40] if len(r) > 40 else r
                reasoning_lines.append((r_str, TEXT_WHITE))
        if scene is not None:
            reasoning_lines.append(("Scene nominal — no alerts", ACCENT_GREEN))
        else:
            reasoning_lines.append(("Waiting for scene data...", TEXT_DIM))
        reasoning_lines.append(("VLM computing...", TEXT_DIM))

    for i, (txt, col) in enumerate(reasoning_lines[:4]):
        if txt:
            _put(sb, txt, pad, lr_y + 13 + i * 14, 0.25, col)

    # ── Compose ───────────────────────────────────────────────────────────────
    out=np.zeros((cam_h_px,cam_w+sb_w,3),dtype=np.uint8)
    out[:,:cam_w]=cam
    out[:,cam_w:]=sb
    cv2.line(out,(cam_w,0),(cam_w,cam_h_px),(45,55,45),1)
    return out
