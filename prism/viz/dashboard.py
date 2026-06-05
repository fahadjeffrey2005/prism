"""
PRISM — Dashboard Visualizer  (experimentation branch)
=======================================================
Split-screen HUD modelled on high-end autonomous driving demos.

Layout
------
┌──────────────────────────────┬────────────────┐
│  Camera feed                 │  Dashboard     │
│  • Hough lane lines (red)    │  • BEV minimap │
│  • Drivable-area overlay     │  • AI Decision │
│  • Bounding boxes + distance │  • Gauges      │
│  • VLM text (when fires)     │  • Status      │
│  • Title bar / counts        │                │
└──────────────────────────────┴────────────────┘
"""

import cv2
import math
import numpy as np


# ── colour palette (BGR) ──────────────────────────────────────────────────────
PANEL_BG       = (10,  12,  18)
ACCENT_CYAN    = (0,  220, 220)
ACCENT_GREEN   = (30, 220,  80)
ACCENT_YELLOW  = (30, 200, 255)
TEXT_DIM       = (120, 120, 120)
TEXT_BRIGHT    = (220, 220, 220)
GRID_COL       = (35,  45,  35)
LANE_COL       = (30,  30, 220)   # red lane lines
DRIVE_COL      = (30, 180,  30)   # green drivable area

DEC_COLORS = {
    "CLEAR":     ( 30, 200,  30),
    "EASE":      ( 50, 220, 100),
    "MONITOR":   (160, 160,  30),
    "SLOW":      ( 30, 180, 220),
    "YIELD":     ( 30, 140, 255),
    "CAUTION":   ( 30,  90, 255),
    "STOP":      ( 30,  30, 200),
    "EMERGENCY": (  0,   0, 255),
    "UNKNOWN":   ( 80,  80,  80),
}

# VLM text persistence
_vlm_buf  = ""
_vlm_ttl  = 0
VLM_HOLD  = 72   # frames (~6 s at 12 fps)


# ═══════════════════════════════════════════════════════════════════════════
# Lane detector  (Canny + Hough)
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """Lightweight Hough-based lane detector with EMA smoothing."""

    def __init__(self,
                 roi_top_frac: float = 0.55,
                 canny_lo: int = 50,
                 canny_hi: int = 150,
                 hough_threshold: int = 40,
                 hough_min_len: int = 60,
                 hough_max_gap: int = 80):
        self.roi_top   = roi_top_frac
        self.canny_lo  = canny_lo
        self.canny_hi  = canny_hi
        self.h_thresh  = hough_threshold
        self.h_min_len = hough_min_len
        self.h_max_gap = hough_max_gap
        self._left_ema  = None
        self._right_ema = None
        self._alpha     = 0.25

    def detect(self, frame: np.ndarray):
        h, w   = frame.shape[:2]
        roi_y  = int(h * self.roi_top)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blur, self.canny_lo, self.canny_hi)
        mask   = np.zeros_like(edges)
        roi_pts = np.array([[0, h], [w, h],
                             [int(w*0.6), roi_y],
                             [int(w*0.4), roi_y]], dtype=np.int32)
        cv2.fillPoly(mask, [roi_pts], 255)
        masked = cv2.bitwise_and(edges, mask)
        lines  = cv2.HoughLinesP(masked, 1, np.pi/180,
                                  self.h_thresh,
                                  minLineLength=self.h_min_len,
                                  maxLineGap=self.h_max_gap)
        if lines is None:
            return None, None, None

        left_segs, right_segs = [], []
        cx = w / 2
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4:
                continue
            if slope < 0 and x1 < cx and x2 < cx:
                left_segs.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > cx and x2 > cx:
                right_segs.append((x1, y1, x2, y2))

        def _fit(segs, y_bot, y_top):
            if not segs:
                return None
            xs = [x for x1,y1,x2,y2 in segs for x in (x1,x2)]
            ys = [y for x1,y1,x2,y2 in segs for y in (y1,y2)]
            try:
                m, b = np.polyfit(ys, xs, 1)
            except Exception:
                return None
            return np.array([[int(m*y_bot+b), y_bot],
                              [int(m*y_top+b), y_top]], dtype=np.int32)

        y_bot = h - 5
        y_top = roi_y + 20
        left  = _fit(left_segs,  y_bot, y_top)
        right = _fit(right_segs, y_bot, y_top)

        def _ema(prev, cur):
            if cur  is None: return prev
            if prev is None: return cur.astype(np.float32)
            return prev*(1-self._alpha) + cur.astype(np.float32)*self._alpha

        self._left_ema  = _ema(self._left_ema,  left)
        self._right_ema = _ema(self._right_ema, right)

        lo = self._left_ema.astype(np.int32)  if self._left_ema  is not None else None
        ro = self._right_ema.astype(np.int32) if self._right_ema is not None else None

        poly = None
        if lo is not None and ro is not None:
            poly = np.array([lo[0], ro[0], ro[1], lo[1]], dtype=np.int32)

        return lo, ro, poly

    @staticmethod
    def draw(frame, left, right, poly, alpha=0.38):
        if poly is not None:
            ov = frame.copy()
            cv2.fillPoly(ov, [poly], DRIVE_COL)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        if left  is not None:
            cv2.line(frame, tuple(left[0]),  tuple(left[1]),  LANE_COL, 2, cv2.LINE_AA)
        if right is not None:
            cv2.line(frame, tuple(right[0]), tuple(right[1]), LANE_COL, 2, cv2.LINE_AA)

    def lane_steer(self, frame_w: int) -> float:
        if self._left_ema is None or self._right_ema is None:
            return 0.0
        cx_lane = (self._left_ema[1, 0] + self._right_ema[1, 0]) / 2
        return float((cx_lane - frame_w/2) / (frame_w/2))


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _blend(img, x1, y1, x2, y2, color, alpha=0.55):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    ov = roi.copy()
    cv2.rectangle(ov, (0, 0), (x2-x1, y2-y1), color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(ov, alpha, roi, 1-alpha, 0)


def _speedometer(img, cx, cy, r, speed_kmh, max_kmh=80):
    start, end = 220, -40
    cv2.ellipse(img, (cx,cy), (r,r), 0, -start, -end, (50,50,50), 2, cv2.LINE_AA)
    for i in range(9):
        frac = i/8
        ang  = math.radians(start - frac*(start-end))
        xi1  = int(cx+(r-8)*math.cos(ang)); yi1 = int(cy-(r-8)*math.sin(ang))
        xi2  = int(cx+(r+2)*math.cos(ang)); yi2 = int(cy-(r+2)*math.sin(ang))
        cv2.line(img, (xi1,yi1), (xi2,yi2), TEXT_DIM, 1, cv2.LINE_AA)
    frac = max(0.0, min(1.0, speed_kmh/max_kmh))
    sweep = start-end
    fe    = start - frac*sweep
    r2    = int(frac*255); g2 = int((1-frac)*200)
    cv2.ellipse(img, (cx,cy), (r-2,r-2), 0, -start, -fe, (0,g2,r2), 3, cv2.LINE_AA)
    ang = math.radians(start - frac*sweep)
    nx  = int(cx+(r-10)*math.cos(ang)); ny = int(cy-(r-10)*math.sin(ang))
    cv2.line(img, (cx,cy), (nx,ny), TEXT_BRIGHT, 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 4, TEXT_BRIGHT, -1, cv2.LINE_AA)
    spd_s = f"{speed_kmh:.0f}"
    (tw,_),_ = cv2.getTextSize(spd_s, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(img, spd_s,  (cx-tw//2, cy+7),  cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_BRIGHT, 1, cv2.LINE_AA)
    (tw2,_),_ = cv2.getTextSize("km/h", cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
    cv2.putText(img, "km/h", (cx-tw2//2, cy+19), cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_DIM,    1, cv2.LINE_AA)
    (tw3,_),_ = cv2.getTextSize("SPEED", cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(img, "SPEED",(cx-tw3//2, cy+r+16),cv2.FONT_HERSHEY_SIMPLEX,0.35, TEXT_DIM,   1, cv2.LINE_AA)


def _steering_dial(img, cx, cy, r, steer_norm):
    cv2.circle(img, (cx,cy), r,   (50,50,50), 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), r+1, (25,25,25), 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 3,   TEXT_DIM,  -1, cv2.LINE_AA)
    ang = math.radians(90 - steer_norm*65)
    nx  = int(cx+(r-6)*math.cos(ang)); ny = int(cy-(r-6)*math.sin(ang))
    col = ACCENT_YELLOW if abs(steer_norm) > 0.25 else ACCENT_GREEN
    cv2.line(img, (cx,cy), (nx,ny), col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), 4, col, -1, cv2.LINE_AA)
    cv2.putText(img, "L", (cx-r-14, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "R", (cx+r+4,  cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
    lbl = "LEFT" if steer_norm < -0.1 else "RIGHT" if steer_norm > 0.1 else "CENTER"
    (tw,_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(img, lbl, (cx-tw//2, cy+r+16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_DIM, 1, cv2.LINE_AA)


def _bev_minimap(panel, x, y, w, h, lidar_dets):
    cv2.rectangle(panel, (x,y), (x+w,y+h), (6,10,6), -1)
    cv2.rectangle(panel, (x,y), (x+w,y+h), GRID_COL, 1)
    for i in range(1, 4):
        gx = x+i*w//4; gy = y+i*h//4
        cv2.line(panel, (gx,y), (gx,y+h), GRID_COL, 1)
        cv2.line(panel, (x,gy), (x+w,gy), GRID_COL, 1)
    for dy in range(0, h, 10):
        cv2.line(panel, (x+w//2,y+dy), (x+w//2,y+dy+5), (40,60,40), 1)
    ex, ey = x+w//2, y+h-20
    cv2.rectangle(panel, (ex-6,ey-14),(ex+6,ey+4), ACCENT_GREEN, -1)
    cv2.rectangle(panel, (ex-6,ey-14),(ex+6,ey+4), TEXT_BRIGHT,  1)
    max_r = 40.0
    for det in lidar_dets:
        dist = min(getattr(det,"distance_m",max_r), max_r)
        lat  = getattr(det,"lateral_m", 0.0)
        bx   = int(ex + lat*(w*0.45)/20.0)
        by_  = int(ey - dist*(h-30)/max_r)
        bx   = max(x+2, min(x+w-2, bx))
        by_  = max(y+2, min(y+h-2, by_))
        frac = dist/max_r
        cv2.rectangle(panel, (bx-4,by_-6),(bx+4,by_+2),
                      (0, int(200*frac), int(255*(1-frac))), -1)


# ═══════════════════════════════════════════════════════════════════════════
# Persistent lane detector instance
# ═══════════════════════════════════════════════════════════════════════════
_lane_detector = LaneDetector()


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(image: np.ndarray,
                     frame_result: dict,
                     lidar_dets: list,
                     panel_frac: float = 0.30,
                     vlm_summary: str  = "",
                     vlm_fired: bool   = False) -> np.ndarray:
    """
    Compose a full-width dashboard frame.
    Returns np.ndarray: [camera feed + overlays | right dashboard panel]
    """
    global _vlm_buf, _vlm_ttl

    h, w = image.shape[:2]
    pw   = int(w * panel_frac)

    decision  = frame_result.get("decision",  "UNKNOWN")
    risk      = frame_result.get("risk",       0.0)
    speed_kmh = frame_result.get("speed_mps",  0.0) * 3.6
    n_cam     = frame_result.get("n_cam_dets", 0)
    n_lidar   = frame_result.get("n_lidar_dets", 0)
    lat_ms    = frame_result.get("latency_ms", 0.0)
    frame_idx = frame_result.get("frame_idx",  0)
    sf        = frame_result.get("sensory_frame")
    dec_col   = DEC_COLORS.get(decision, (80,80,80))

    # ── Camera canvas ─────────────────────────────────────────────────────────
    cam = image.copy()

    # Lane lines + drivable overlay
    left, right, poly = _lane_detector.detect(cam)
    LaneDetector.draw(cam, left, right, poly, alpha=0.38)
    lane_steer = _lane_detector.lane_steer(w)

    # Bounding boxes
    if sf is not None and hasattr(sf, "detections"):
        for det in sf.detections:
            if getattr(det, "camera_name", "") == "lidar":
                continue
            b = det.bbox
            x1,y1,x2,y2 = int(b.x1),int(b.y1),int(b.x2),int(b.y2)
            dist = det.depth_estimate
            ds   = f" {dist:.1f}m" if dist is not None else ""
            lbl  = f"{b.class_name}{ds}"
            bcol = ((0,40,255) if (dist or 99)<6 else
                    (0,140,255) if (dist or 99)<15 else (40,220,40))
            cv2.rectangle(cam, (x1,y1),(x2,y2), bcol, 2, cv2.LINE_AA)
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            _blend(cam, x1, max(0,y1-th-8), x1+tw+6, y1, (10,10,10), 0.72)
            cv2.putText(cam, lbl, (x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    # Title bar
    _blend(cam, 0, 0, w, 30, (0,0,0), 0.72)
    cv2.putText(cam, "PRISM  AUTONOMOUS  PERCEPTION",
                (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.55, ACCENT_CYAN, 1, cv2.LINE_AA)
    fps_s = f"FPS: {1000/max(lat_ms,1):.1f}   Speed: {speed_kmh:.0f} km/h"
    cv2.putText(cam, fps_s, (10,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_GREEN, 1, cv2.LINE_AA)
    cnt_s = f"Vehicles: {n_cam}   Persons: {n_lidar}"
    (tw,_),_ = cv2.getTextSize(cnt_s, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(cam, cnt_s, (w-tw-10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_BRIGHT, 1, cv2.LINE_AA)

    # VLM text
    if vlm_fired and vlm_summary:
        _vlm_buf = vlm_summary
        _vlm_ttl = VLM_HOLD
    if _vlm_ttl > 0:
        _vlm_ttl -= 1
        _blend(cam, 0, h-72, w, h, (4,4,40), 0.80)
        cv2.putText(cam, "PRISM AI", (10,h-50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, ACCENT_CYAN, 1, cv2.LINE_AA)
        words, lines_out, cur = _vlm_buf.split(), [], ""
        for word in words:
            test = (cur+" "+word).strip()
            (tw2,_),_ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            if tw2 > w-120: lines_out.append(cur); cur = word
            else: cur = test
        if cur: lines_out.append(cur)
        for li, ln in enumerate(lines_out[:2]):
            cv2.putText(cam, ln, (95,h-50+li*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,240,180), 1, cv2.LINE_AA)

    # ── Dashboard panel ───────────────────────────────────────────────────────
    panel = np.full((h, pw, 3), PANEL_BG, dtype=np.uint8)
    pad   = 10
    yc    = pad

    def _sec(label, y):
        cv2.putText(panel, label, (pad, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, ACCENT_CYAN, 1, cv2.LINE_AA)
        cv2.line(panel, (pad,y+16), (pw-pad,y+16), (30,55,30), 1)
        return y+22

    # BEV
    yc    = _sec("BIRD'S EYE VIEW", yc)
    bev_h = 138
    _bev_minimap(panel, pad, yc, pw-2*pad, bev_h, lidar_dets)
    yc   += bev_h + 8

    # AI Decision
    yc = _sec("AI DECISION", yc)
    _blend(panel, pad, yc, pw-pad, yc+44, dec_col, 0.88)
    (tw,_),_ = cv2.getTextSize(decision, cv2.FONT_HERSHEY_DUPLEX, 0.88, 2)
    cv2.putText(panel, decision, ((pw-tw)//2, yc+31),
                cv2.FONT_HERSHEY_DUPLEX, 0.88, (255,255,255), 2, cv2.LINE_AA)
    yc += 50
    subtitles = {
        "CLEAR":     f"Speeding up to {speed_kmh:.0f} km/h",
        "EASE":      "Ease off — caution ahead",
        "MONITOR":   "Monitoring surroundings",
        "SLOW":      f"Reducing speed to {speed_kmh:.0f} km/h",
        "YIELD":     "Yielding to road users",
        "CAUTION":   "Caution — obstacle detected",
        "STOP":      "Coming to a complete stop",
        "EMERGENCY": "EMERGENCY BRAKE APPLIED",
    }
    sub = subtitles.get(decision, "")
    if sub:
        (tw,_),_ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(panel, sub, ((pw-tw)//2, yc+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    yc += 18

    # Throttle / Brake / Steer readouts
    throttle  = max(0.0, 1.0-risk) if decision not in ("STOP","EMERGENCY") else 0.0
    brake     = risk               if decision in ("STOP","EMERGENCY","CAUTION") else 0.0
    yc += 4
    for lbl, val, col in [
        ("Throttle:", f"{throttle:.1f}", ACCENT_GREEN),
        ("Brake:",    f"{brake:.1f}",    (30,30,220)),
        ("Steer:",    f"{lane_steer:+.3f}", ACCENT_YELLOW),
    ]:
        cv2.putText(panel, lbl, (pad,    yc+12), cv2.FONT_HERSHEY_SIMPLEX, 0.37, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, val, (pad+72, yc+12), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col,      1, cv2.LINE_AA)
        yc += 17
    yc += 8

    # Gauges
    yc  = _sec("STEERING             SPEED", yc)
    gr  = 34
    gy  = yc + gr + 8
    _steering_dial(panel, pw//4,     gy, gr, lane_steer)
    _speedometer(  panel, 3*pw//4,   gy, gr, speed_kmh, max_kmh=80)
    yc += gr*2 + 30

    # System status
    yc = _sec("SYSTEM STATUS", yc)
    dangers  = 1 if decision in ("EMERGENCY","STOP")              else 0
    warnings = 1 if decision in ("CAUTION","YIELD","SLOW","EASE") else 0
    rows = [
        ("Detections:", f"{n_cam+n_lidar}",   TEXT_BRIGHT),
        ("Dangers:",    str(dangers),          (30,30,220)   if dangers  else TEXT_DIM),
        ("Warnings:",   str(warnings),         ACCENT_YELLOW if warnings else TEXT_DIM),
        ("Lane Steer:", f"{lane_steer:+.3f}", ACCENT_YELLOW),
        ("Frame:",      str(frame_idx),        TEXT_DIM),
        ("Latency:",    f"{lat_ms:.0f} ms",    TEXT_DIM),
    ]
    for i, (k, v, col) in enumerate(rows):
        col_x = pad + (i%2)*(pw//2-pad)
        row_y = yc + (i//2)*18 + 14
        cv2.putText(panel, k, (col_x,    row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, v, (col_x+70, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col,      1, cv2.LINE_AA)

    return np.concatenate([cam, panel], axis=1)
