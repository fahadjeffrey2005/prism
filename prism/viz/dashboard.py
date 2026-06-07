"""
PRISM — Dashboard Visualizer  (experimentation branch)
=======================================================
Tesla-style side-by-side layout.

Output layout (target 960x540 camera feed):
  ┌──────────────────────────────────┬─────────────────┐
  │  VISION-ONLY AUTONOMOUS DRIVING  │  BIRD'S EYE VIEW │
  │  FPS  Speed     Vehicles Persons │  (minimap+path)  │
  │                                  │─────────────────│
  │  Camera feed + path corridor     │  AI DECISION    │
  │  + lane overlay (green fill/red) │  [BANNER]       │
  │  + bounding boxes + distance     │  Thr  Brk  Str  │
  │  + collision warning             │─────────────────│
  │                                  │  STEERING  SPEED │
  │                                  │─────────────────│
  │                                  │  SYSTEM STATUS  │
  └──────────────────────────────────┴─────────────────┘
"""

import cv2
import math
import numpy as np

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
SIDEBAR_BG    = (10,  12,  10)
TEXT_WHITE    = (235, 235, 235)
TEXT_DIM      = (110, 110, 110)
TEXT_CYAN     = (220, 220,  30)
ACCENT_GREEN  = ( 40, 210,  60)
ACCENT_YELLOW = ( 30, 200, 230)
ACCENT_BLUE   = (220, 130,  40)
ACCENT_ORANGE = ( 30, 140, 230)
LANE_LINE_COL = ( 30,  30, 220)   # red lane lines
DRIVE_COL     = ( 30, 180,  30)   # green drivable fill
WARN_RED      = ( 30,  30, 220)   # red (BGR)
PATH_COL      = ( 30, 160,  30)   # forward path corridor

# ── Decision display names (maps PRISM 8-level to friendly labels) ─────────────
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
# Lane detector + path corridor
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self):
        self._left_ema  = None
        self._right_ema = None
        self._alpha     = 0.22

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
            cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)
        if left  is not None:
            cv2.line(frame, tuple(left[0]),  tuple(left[1]),  LANE_LINE_COL, 2, cv2.LINE_AA)
        if right is not None:
            cv2.line(frame, tuple(right[0]), tuple(right[1]), LANE_LINE_COL, 2, cv2.LINE_AA)

    def steer(self, w):
        if self._left_ema is None or self._right_ema is None: return 0.0
        return float((((self._left_ema[1,0]+self._right_ema[1,0])/2) - w/2) / (w/2))


def draw_path_corridor(frame: np.ndarray, steer: float = 0.0):
    """
    Always-on forward path corridor.
    Draws a perspective trapezoid showing the planned route.
    Falls back to this when lane detection has no lines.
    Steer shifts the corridor left/right.
    """
    h, w = frame.shape[:2]
    horizon_y   = int(h * 0.46)
    bot_half_w  = int(w * 0.18)
    top_half_w  = int(w * 0.04)
    cx          = w // 2 + int(steer * w * 0.12)   # steer offset

    # Filled drivable corridor
    pts = np.array([
        [cx - bot_half_w, h],
        [cx + bot_half_w, h],
        [cx + top_half_w, horizon_y],
        [cx - top_half_w, horizon_y],
    ], dtype=np.int32)

    ov = frame.copy()
    cv2.fillPoly(ov, [pts], PATH_COL)
    cv2.addWeighted(ov, 0.30, frame, 0.70, 0, frame)

    # Boundary lines (red)
    cv2.line(frame, (cx - bot_half_w, h), (cx - top_half_w, horizon_y),
             LANE_LINE_COL, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + bot_half_w, h), (cx + top_half_w, horizon_y),
             LANE_LINE_COL, 2, cv2.LINE_AA)

    # Centre dashed path line
    n_dashes = 8
    for i in range(n_dashes):
        t0 = i / n_dashes
        t1 = (i + 0.5) / n_dashes
        y0 = int(h + t0 * (horizon_y - h))
        y1 = int(h + t1 * (horizon_y - h))
        x0 = int(cx + t0 * 0)
        x1 = int(cx + t1 * 0)
        cv2.line(frame, (x0, y0), (x1, y1), (200, 200, 200), 1, cv2.LINE_AA)


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
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _put(img, text, cx - tw // 2, y, scale, color, thick)


def _draw_bev(panel, lidar_dets, steer):
    """Bird's Eye View — shows path + obstacles."""
    h, w = panel.shape[:2]
    cx, ey = w // 2, h - 24

    # Grid lines
    lane_w = w // 5
    cv2.line(panel, (cx - lane_w, 0), (cx - lane_w, h), (40, 50, 40), 1)
    cv2.line(panel, (cx + lane_w, 0), (cx + lane_w, h), (40, 50, 40), 1)

    # Path projection in BEV
    path_dx = int(steer * lane_w * 0.8)
    bev_pts = np.array([
        [cx - 10,          ey],
        [cx + 10,          ey],
        [cx + path_dx + 4, h // 3],
        [cx + path_dx - 4, h // 3],
    ], dtype=np.int32)
    ov = panel.copy()
    cv2.fillPoly(ov, [bev_pts], (25, 100, 25))
    cv2.addWeighted(ov, 0.5, panel, 0.5, 0, panel)

    # Centre dashed line
    for dy in range(0, h, 14):
        if (dy // 14) % 2 == 0:
            cv2.line(panel, (cx + path_dx, dy), (cx + path_dx, min(dy+8, h)),
                     (70, 70, 70), 1)

    # Ego vehicle rectangle
    cv2.rectangle(panel, (cx - 7, ey - 18), (cx + 7, ey),
                  ACCENT_GREEN, -1)
    cv2.rectangle(panel, (cx - 7, ey - 18), (cx + 7, ey),
                  TEXT_WHITE, 1)

    # Obstacles from LiDAR
    for det in lidar_dets:
        dist    = min(getattr(det, "distance_m", 40), 40.0)
        lat     = getattr(det, "lateral_m",   0.0)
        bx      = int(np.clip(cx + lat * (w * 0.35) / 15, 2, w - 2))
        by      = int(np.clip(ey - dist * (h - 30) / 40, 2, h - 2))
        frac    = dist / 40.0
        color   = (0, int(200 * frac), int(240 * (1 - frac)))
        cv2.rectangle(panel, (bx - 4, by - 6), (bx + 4, by + 2), color, -1)


def _draw_steering_gauge(img, cx, cy, r, steer_norm):
    sweep, start = 280, 220
    frac = float(np.clip((steer_norm + 1.0) / 2.0, 0.0, 1.0))
    cv2.ellipse(img, (cx, cy), (r, r), 0, -start, -(start-sweep), (50,50,50), 2, cv2.LINE_AA)
    col = ACCENT_GREEN if abs(steer_norm) < 0.12 else ACCENT_YELLOW
    a = math.radians(start - frac * sweep)
    nx = int(cx + (r - 8) * math.cos(a))
    ny = int(cy - (r - 8) * math.sin(a))
    cv2.line(img, (cx, cy), (nx, ny), col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, col, -1, cv2.LINE_AA)
    _put(img, "L", cx - r - 10, cy + 4, 0.28, TEXT_DIM)
    _put(img, "R", cx + r + 3,  cy + 4, 0.28, TEXT_DIM)
    lbl = "LEFT" if steer_norm < -0.12 else "RIGHT" if steer_norm > 0.12 else "CENTER"
    _put_center(img, lbl,      cx, cy + r + 13, 0.28, TEXT_DIM)
    _put_center(img, "STEERING", cx, cy - r - 6, 0.28, TEXT_DIM)


def _draw_speed_gauge(img, cx, cy, r, speed_kmh, max_kmh=80):
    sweep, start = 280, 220
    frac = float(np.clip(speed_kmh / max_kmh, 0.0, 1.0))
    cv2.ellipse(img, (cx, cy), (r, r), 0, -start, -(start-sweep), (50,50,50), 2, cv2.LINE_AA)
    col = (ACCENT_GREEN if speed_kmh < 40 else ACCENT_YELLOW if speed_kmh < 70 else WARN_RED)
    if frac > 0.01:
        cv2.ellipse(img, (cx, cy), (r-2, r-2), 0, -start, -(start-sweep*frac),
                    col, 2, cv2.LINE_AA)
    a = math.radians(start - frac * sweep)
    nx = int(cx + (r - 8) * math.cos(a))
    ny = int(cy - (r - 8) * math.sin(a))
    cv2.line(img, (cx, cy), (nx, ny), TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, TEXT_WHITE, -1, cv2.LINE_AA)
    _put_center(img, f"{speed_kmh:.0f}", cx, cy + 5, 0.44, TEXT_WHITE)
    _put_center(img, "km/h",   cx, cy + r + 13, 0.26, TEXT_DIM)
    _put_center(img, "SPEED",  cx, cy - r - 6,  0.28, TEXT_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent instances
# ═══════════════════════════════════════════════════════════════════════════════
_lane = LaneDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# Main render
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard(image: np.ndarray,
                     frame_result: dict,
                     lidar_dets: list,
                     panel_frac: float = SIDEBAR_FRAC,
                     vlm_summary: str  = "",
                     vlm_fired: bool   = False) -> np.ndarray:
    global _vlm_buf, _vlm_ttl

    cam_h, cam_w = image.shape[:2]
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

    # ── Camera: path corridor (always shown) ───────────────────────────────────
    cam = image.copy()
    left, right, poly = _lane.detect(cam)
    lane_steer = _lane.steer(cam_w)

    # Always draw the forward path corridor first (background)
    draw_path_corridor(cam, steer=lane_steer)

    # Overlay detected lane lines on top if found
    _lane.draw(cam, left, right, poly)

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
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            _blend(cam, x1, max(0,y1-th-8), x1+tw+6, y1, (10,10,10), 0.72)
            _put(cam, lbl, x1+3, y1-4, 0.44, (255,255,255))

    # ── Collision warning ─────────────────────────────────────────────────────
    if is_danger:
        cv2.rectangle(cam, (0,0), (cam_w-1, cam_h-1), WARN_RED, 8)
        _blend(cam, 0, 0, cam_w, 44, WARN_RED, 0.85)
        _put_center(cam, "!! COLLISION WARNING !!", cam_w//2, 30,
                    0.85, TEXT_WHITE, 2)

    # ── Top-left HUD ──────────────────────────────────────────────────────────
    _blend(cam, 0, 0, 330, 48, (0,0,0), 0.50)
    _put(cam, "VISION-ONLY AUTONOMOUS DRIVING", 8, 18, 0.50, TEXT_CYAN)
    _put(cam, f"FPS: {fps:.1f}  |  Speed: {speed_kmh:.0f} km/h", 8, 38, 0.40, ACCENT_GREEN)

    # ── Top-centre: counts ────────────────────────────────────────────────────
    cnt = f"Vehicles: {n_cam}   Persons: {n_lidar}"
    _put_center(cam, cnt, cam_w//2, 20, 0.44, TEXT_WHITE)

    # ── VLM strip ────────────────────────────────────────────────────────────
    if vlm_fired and vlm_summary:
        _vlm_buf = vlm_summary; _vlm_ttl = VLM_HOLD
    if _vlm_ttl > 0:
        _vlm_ttl -= 1
        _blend(cam, 0, cam_h-42, cam_w, cam_h-18, (4,4,40), 0.82)
        _put(cam, "PRISM AI", 10, cam_h-25, 0.38, (220,220,30))
        words, lines_out, cur = _vlm_buf.split(), [], ""
        for w2 in words:
            test = (cur+" "+w2).strip()
            (tw2,_),_ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            if tw2 > cam_w-110: lines_out.append(cur); cur=w2
            else: cur=test
        if cur: lines_out.append(cur)
        for li,ln in enumerate(lines_out[:2]):
            _put(cam, ln, 100, cam_h-25+li*16, 0.38, (255,240,180))

    # ── Build sidebar ─────────────────────────────────────────────────────────
    sb   = np.full((cam_h, sb_w, 3), SIDEBAR_BG, dtype=np.uint8)
    pad  = 8
    div  = (30, 42, 30)

    # ── S1: Bird's Eye View ───────────────────────────────────────────────────
    bev_lh  = 20
    bev_h   = int(cam_h * 0.34)
    _put_center(sb, "BIRD'S EYE VIEW", sb_w//2, bev_lh-3, 0.34, TEXT_DIM)
    bev_pan = sb[bev_lh:bev_h, pad:sb_w-pad]
    _draw_bev(bev_pan, lidar_dets, lane_steer)
    cv2.line(sb, (0, bev_h), (sb_w, bev_h), div, 1)

    # ── S2: AI Decision ───────────────────────────────────────────────────────
    dec_y   = bev_h + 5
    dlabel_h= 18
    _put_center(sb, "AI DECISION", sb_w//2, dec_y+dlabel_h-2, 0.34, TEXT_DIM)
    ban_y1  = dec_y + dlabel_h + 4
    ban_y2  = ban_y1 + 36
    cv2.rectangle(sb, (pad, ban_y1), (sb_w-pad, ban_y2), dec_col, -1)
    _put_center(sb, decision, sb_w//2, ban_y1+(ban_y2-ban_y1)//2+7,
                0.68, TEXT_WHITE, 2)

    sub = DEC_SUBTITLES.get(decision, "")
    if sub:
        _put_center(sb, sub, sb_w//2, ban_y2+15, 0.30, TEXT_WHITE)

    # Throttle / Brake / Steer row
    tbs_y  = ban_y2 + 30
    col_w  = sb_w // 3
    for i,(lbl,col) in enumerate([
        (f"Throttle: {throttle:.1f}", ACCENT_GREEN),
        (f"Brake: {brake_val:.1f}",   ACCENT_BLUE),
        (f"Steer: {lane_steer:+.2f}", ACCENT_ORANGE),
    ]):
        _put(sb, lbl, i*col_w + pad//2, tbs_y, 0.28, col)

    div2_y = tbs_y + 14
    cv2.line(sb, (0, div2_y), (sb_w, div2_y), div, 1)

    # ── S3: Gauges ────────────────────────────────────────────────────────────
    gauge_y  = div2_y + 6
    remaining = cam_h - gauge_y - 80
    gauge_r  = max(22, min(34, remaining // 2 - 18))
    gcx1     = sb_w // 4
    gcx2     = 3 * sb_w // 4
    gcy      = gauge_y + gauge_r + 18

    _draw_steering_gauge(sb, gcx1, gcy, gauge_r, lane_steer)
    _draw_speed_gauge(sb, gcx2, gcy, gauge_r, speed_kmh)

    div3_y = gcy + gauge_r + 22
    cv2.line(sb, (0, div3_y), (sb_w, div3_y), div, 1)

    # ── S4: System Status ─────────────────────────────────────────────────────
    st_y = div3_y + 14
    _put_center(sb, "SYSTEM STATUS", sb_w//2, st_y, 0.32, TEXT_DIM)

    dangers  = 1 if is_danger else 0
    warnings = 1 if decision in ("CAUTION","YIELD","SLOW","EASE") else 0
    for i,(txt, col) in enumerate([
        (f"Detections: {n_cam+n_lidar}",         TEXT_WHITE),
        (f"Dangers: {dangers}   Warnings: {warnings}",
         WARN_RED if dangers else TEXT_WHITE),
        (f"Lane Steer: {lane_steer:+.3f}",        ACCENT_YELLOW),
        (f"Frame: {frame_idx}",                   TEXT_DIM),
    ]):
        _put(sb, txt, pad, st_y+16+i*16, 0.30, col)

    # ── Compose ───────────────────────────────────────────────────────────────
    out = np.zeros((cam_h, total, 3), dtype=np.uint8)
    out[:, :cam_w]   = cam
    out[:, cam_w:]   = sb
    cv2.line(out, (cam_w, 0), (cam_w, cam_h), (45, 55, 45), 1)
    return out
