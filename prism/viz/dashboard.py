"""
PRISM — Dashboard Visualizer  (experimentation branch)
=======================================================
Tesla-style side-by-side layout matching reference video.

Output layout:
  ┌──────────────────────────────────┬─────────────────┐
  │                                  │  BIRD'S EYE VIEW │
  │   Camera feed (~75% width)       │─────────────────│
  │   + lane overlay (green/red)     │  AI DECISION    │
  │   + bounding boxes               │  [BANNER]       │
  │   + top HUD text                 │  Thr  Brk  Str  │
  │   + collision warning overlay    │─────────────────│
  │                                  │ STEER   SPEED   │
  │                                  │─────────────────│
  │                                  │ SYSTEM STATUS   │
  └──────────────────────────────────┴─────────────────┘
"""

import cv2
import math
import numpy as np

# ── Palette (BGR) ──────────────────────────────────────────────────────────────
SIDEBAR_BG    = (12,  14,  12)   # near-black dark green tint
TEXT_WHITE    = (235, 235, 235)
TEXT_DIM      = (120, 120, 120)
TEXT_CYAN     = (220, 220,  30)  # yellow-green for title
ACCENT_GREEN  = ( 40, 210,  60)
ACCENT_YELLOW = ( 30, 200, 230)
ACCENT_BLUE   = (230, 140,  40)  # blue-ish for brake
ACCENT_ORANGE = ( 30, 140, 230)  # orange for steer
LANE_LINE_COL = ( 30,  30, 220)  # RED lane lines
DRIVE_COL     = ( 30, 180,  30)  # green drivable overlay
WARN_RED      = ( 30,  30, 220)  # red in BGR

DEC_COLORS = {
    "CLEAR":     ( 30, 200,  50),
    "MONITOR":   ( 30, 180, 120),
    "EASE":      ( 30, 200, 100),
    "ACCELERATE":( 30, 190,  50),
    "CRUISE":    ( 30, 200,  60),
    "SLOW":      ( 30, 130, 210),
    "YIELD":     ( 20, 100, 230),
    "CAUTION":   ( 20,  80, 230),
    "STOP":      ( 20,  20, 210),
    "STOP ??? RED LIGHT": (20, 20, 200),
    "EMERGENCY": (  0,   0, 220),
    "UNKNOWN":   ( 60,  60,  60),
}

DEC_SUBTITLES = {
    "CLEAR":     "Road clear — maintaining speed",
    "MONITOR":   "Monitoring surroundings",
    "EASE":      "Easing off — caution ahead",
    "ACCELERATE":"Speeding up to 35 km/h",
    "CRUISE":    "Cruising — road clear",
    "SLOW":      "Slowing down — obstacle detected",
    "YIELD":     "Yielding to road users",
    "CAUTION":   "Caution — obstacle detected",
    "STOP":      "Coming to a complete stop",
    "STOP ??? RED LIGHT": "Red traffic light — stopping",
    "EMERGENCY": "EMERGENCY BRAKE APPLIED",
    "UNKNOWN":   "",
}

_vlm_buf = ""
_vlm_ttl = 0
VLM_HOLD  = 72   # frames to keep VLM text visible

# Sidebar fraction of total output width
SIDEBAR_FRAC = 0.28


# ═══════════════════════════════════════════════════════════════════════════════
# Lane detector
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    def __init__(self):
        self._left_ema  = None
        self._right_ema = None
        self._alpha     = 0.22

    def detect(self, frame):
        h, w  = frame.shape[:2]
        roi_y = int(h * 0.54)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        mask  = np.zeros_like(edges)
        roi   = np.array([[0, h], [w, h],
                           [int(w*0.62), roi_y],
                           [int(w*0.38), roi_y]], dtype=np.int32)
        cv2.fillPoly(mask, [roi], 255)
        masked = cv2.bitwise_and(edges, mask)
        lines  = cv2.HoughLinesP(masked, 1, np.pi/180, 40,
                                  minLineLength=55, maxLineGap=80)
        if lines is None:
            return None, None, None

        left_s, right_s, cx = [], [], w / 2
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            s = (y2 - y1) / (x2 - x1)
            if abs(s) < 0.4:
                continue
            if s < 0 and x1 < cx and x2 < cx:
                left_s.append((x1, y1, x2, y2))
            elif s > 0 and x1 > cx and x2 > cx:
                right_s.append((x1, y1, x2, y2))

        def _fit(segs, yb, yt):
            if not segs:
                return None
            xs = [x for x1, y1, x2, y2 in segs for x in (x1, x2)]
            ys = [y for x1, y1, x2, y2 in segs for y in (y1, y2)]
            try:
                m, b = np.polyfit(ys, xs, 1)
            except Exception:
                return None
            return np.array([[int(m*yb+b), yb], [int(m*yt+b), yt]], dtype=np.int32)

        yb, yt = h - 5, roi_y + 20
        left  = _fit(left_s,  yb, yt)
        right = _fit(right_s, yb, yt)

        def _ema(p, c):
            if c is None:
                return p
            if p is None:
                return c.astype(np.float32)
            return p * (1 - self._alpha) + c.astype(np.float32) * self._alpha

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
        if left is not None:
            cv2.line(frame, tuple(left[0]),  tuple(left[1]),  LANE_LINE_COL, 3, cv2.LINE_AA)
        if right is not None:
            cv2.line(frame, tuple(right[0]), tuple(right[1]), LANE_LINE_COL, 3, cv2.LINE_AA)

    def steer(self, w):
        if self._left_ema is None or self._right_ema is None:
            return 0.0
        return float(
            (((self._left_ema[1, 0] + self._right_ema[1, 0]) / 2) - w / 2) / (w / 2)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _blend(img, x1, y1, x2, y2, color, alpha=0.6):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    ov = roi.copy()
    cv2.rectangle(ov, (0, 0), (x2-x1, y2-y1), color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(ov, alpha, roi, 1-alpha, 0)


def _draw_bev(panel, cam_w, cam_h, lidar_dets):
    """
    Bird's Eye View panel section.
    Minimal: two vertical lane lines, center dashed line, ego vehicle rect,
    detected obstacles as small coloured rectangles.
    """
    h, w = panel.shape[:2]

    # Background already dark — draw lane lines
    cx = w // 2
    # Lane boundaries
    lane_w = w // 5
    cv2.line(panel, (cx - lane_w, 0), (cx - lane_w, h), (70, 70, 70), 1)
    cv2.line(panel, (cx + lane_w, 0), (cx + lane_w, h), (70, 70, 70), 1)

    # Center dashed line
    for dy in range(0, h, 16):
        if (dy // 16) % 2 == 0:
            cv2.line(panel, (cx, dy), (cx, min(dy + 10, h)), (80, 80, 80), 1)

    # Ego vehicle rectangle near bottom
    ey = h - 28
    ew, eh = 14, 22
    cv2.rectangle(panel, (cx - ew//2, ey - eh), (cx + ew//2, ey), ACCENT_GREEN, -1)
    cv2.rectangle(panel, (cx - ew//2, ey - eh), (cx + ew//2, ey), TEXT_WHITE, 1)

    # Detected obstacles
    for det in lidar_dets:
        dist = min(getattr(det, "distance_m", 40), 40.0)
        lat  = getattr(det, "lateral_m", 0.0)
        bx   = int(np.clip(cx + lat * (w * 0.3) / 15, 2, w - 2))
        by   = int(np.clip(ey - dist * (h - 40) / 40, 2, h - 2))
        frac = dist / 40.0
        color = (0, int(200 * frac), int(240 * (1 - frac)))
        cv2.rectangle(panel, (bx - 5, by - 8), (bx + 5, by + 2), color, -1)


def _draw_circular_gauge(img, cx, cy, r, value, value_max,
                          label_top, label_bot, color_arc,
                          show_value=True, fmt="{:.0f}"):
    """Generic circular gauge with arc + needle."""
    start_angle = 220   # degrees (CCW from 3 o'clock, OpenCV convention)
    sweep       = 280   # total sweep in degrees

    # Background ring
    cv2.ellipse(img, (cx, cy), (r, r), 0,
                -(start_angle), -(start_angle - sweep),
                (55, 55, 55), 3, cv2.LINE_AA)

    # Colored arc proportional to value
    frac = float(np.clip(value / max(value_max, 1e-3), 0.0, 1.0))
    if frac > 0.01:
        cv2.ellipse(img, (cx, cy), (r - 2, r - 2), 0,
                    -(start_angle), -(start_angle - sweep * frac),
                    color_arc, 3, cv2.LINE_AA)

    # Needle
    needle_angle = math.radians(start_angle - frac * sweep)
    nx = int(cx + (r - 10) * math.cos(needle_angle))
    ny = int(cy - (r - 10) * math.sin(needle_angle))
    cv2.line(img, (cx, cy), (nx, ny), TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, TEXT_WHITE, -1, cv2.LINE_AA)

    # Centre value text
    if show_value:
        val_str = fmt.format(value)
        (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(img, val_str, (cx - tw//2, cy + th//2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_WHITE, 1, cv2.LINE_AA)

    # Labels
    if label_top:
        (tw, _), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(img, label_top, (cx - tw//2, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    if label_bot:
        (tw, _), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(img, label_bot, (cx - tw//2, cy + r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)


def _draw_steering_gauge(img, cx, cy, r, steer_norm):
    """Steering gauge — centred needle, ±1 range."""
    start_angle = 220
    sweep       = 280
    frac = float(np.clip((steer_norm + 1.0) / 2.0, 0.0, 1.0))  # map -1..1 → 0..1

    # Background arc
    cv2.ellipse(img, (cx, cy), (r, r), 0,
                -(start_angle), -(start_angle - sweep), (55, 55, 55), 3, cv2.LINE_AA)

    # Needle colour: green when near centre
    col = ACCENT_GREEN if abs(steer_norm) < 0.15 else ACCENT_YELLOW
    needle_angle = math.radians(start_angle - frac * sweep)
    nx = int(cx + (r - 10) * math.cos(needle_angle))
    ny = int(cy - (r - 10) * math.sin(needle_angle))
    cv2.line(img, (cx, cy), (nx, ny), col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, col, -1, cv2.LINE_AA)

    # L / R labels
    cv2.putText(img, "L", (cx - r - 12, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "R", (cx + r + 4, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    # Direction label below
    lbl = "LEFT" if steer_norm < -0.12 else "RIGHT" if steer_norm > 0.12 else "CENTER"
    (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
    cv2.putText(img, lbl, (cx - tw//2, cy + r + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_DIM, 1, cv2.LINE_AA)

    # "STEERING" above
    (tw, _), _ = cv2.getTextSize("STEERING", cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
    cv2.putText(img, "STEERING", (cx - tw//2, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_DIM, 1, cv2.LINE_AA)


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
    """
    Render the PRISM dashboard.
    Returns a wider image: [camera feed | dark sidebar].
    """
    global _vlm_buf, _vlm_ttl

    cam_h, cam_w = image.shape[:2]
    sidebar_w = int(cam_w * panel_frac / (1.0 - panel_frac))
    total_w   = cam_w + sidebar_w

    # ── Unpack frame result ────────────────────────────────────────────────────
    decision  = frame_result.get("decision",    "UNKNOWN")
    risk      = frame_result.get("risk",         0.0)
    speed_kmh = frame_result.get("speed_mps",   0.0) * 3.6
    n_cam     = frame_result.get("n_cam_dets",  0)
    n_lidar   = frame_result.get("n_lidar_dets",0)
    lat_ms    = frame_result.get("latency_ms",  0.0)
    frame_idx = frame_result.get("frame_idx",   0)
    sf        = frame_result.get("sensory_frame")
    throttle  = frame_result.get("throttle",    max(0.0, 1.0 - risk) if decision not in ("STOP","EMERGENCY") else 0.0)
    brake_val = frame_result.get("brake",       risk if decision in ("STOP","EMERGENCY","CAUTION") else 0.0)

    dec_col   = DEC_COLORS.get(decision, DEC_COLORS["UNKNOWN"])
    is_danger = decision in ("STOP", "EMERGENCY", "STOP ??? RED LIGHT")
    fps       = 1000.0 / max(lat_ms, 1.0)

    # ── Camera feed — copy and draw overlays ──────────────────────────────────
    cam = image.copy()

    # Lane lines + drivable overlay
    left, right, poly = _lane.detect(cam)
    _lane.draw(cam, left, right, poly)
    lane_steer = _lane.steer(cam_w)

    # Bounding boxes from sensory frame
    if sf is not None and hasattr(sf, "detections"):
        for det in sf.detections:
            if getattr(det, "camera_name", "") == "lidar":
                continue
            b    = det.bbox
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            dist = det.depth_estimate
            lbl  = f"{b.class_name}{f' {dist:.1f}m' if dist else ''}"
            bc   = (( 0, 40, 255) if (dist or 99) <  6 else
                    ( 0,140, 255) if (dist or 99) < 15 else
                    (40, 200, 40))
            cv2.rectangle(cam, (x1, y1), (x2, y2), bc, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            _blend(cam, x1, max(0, y1 - th - 8), x1 + tw + 6, y1, (10, 10, 10), 0.72)
            cv2.putText(cam, lbl, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Collision warning ─────────────────────────────────────────────────────
    if is_danger:
        # Thick red border around camera feed
        cv2.rectangle(cam, (0, 0), (cam_w - 1, cam_h - 1), WARN_RED, 8)
        # Banner
        banner_h = 42
        _blend(cam, 0, 0, cam_w, banner_h, WARN_RED, 0.85)
        warn_text = "!! COLLISION WARNING !!"
        (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(cam, warn_text, ((cam_w - tw) // 2, banner_h // 2 + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_WHITE, 2, cv2.LINE_AA)

    # ── Top-left HUD text ─────────────────────────────────────────────────────
    _blend(cam, 0, 0, 340, 46, (0, 0, 0), 0.50)
    cv2.putText(cam, "VISION-ONLY AUTONOMOUS DRIVING",
                (8, 18), cv2.FONT_HERSHEY_DUPLEX, 0.52, TEXT_CYAN, 1, cv2.LINE_AA)
    cv2.putText(cam, f"FPS: {fps:.1f}  |  Speed: {speed_kmh:.0f} km/h",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_GREEN, 1, cv2.LINE_AA)

    # ── Top-centre: Vehicles / Persons ────────────────────────────────────────
    cnt_text = f"Vehicles: {n_cam}   Persons: {n_lidar}"
    (tw, _), _ = cv2.getTextSize(cnt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(cam, cnt_text, ((cam_w - tw) // 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_WHITE, 1, cv2.LINE_AA)

    # ── VLM text strip near bottom ────────────────────────────────────────────
    if vlm_fired and vlm_summary:
        _vlm_buf = vlm_summary
        _vlm_ttl = VLM_HOLD
    if _vlm_ttl > 0:
        _vlm_ttl -= 1
        _blend(cam, 0, cam_h - 42, cam_w, cam_h - 18, (4, 4, 40), 0.82)
        cv2.putText(cam, "PRISM AI", (10, cam_h - 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (220, 220, 30), 1, cv2.LINE_AA)
        words, lines_out, cur = _vlm_buf.split(), [], ""
        for word in words:
            test = (cur + " " + word).strip()
            (tw2, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            if tw2 > cam_w - 110:
                lines_out.append(cur)
                cur = word
            else:
                cur = test
        if cur:
            lines_out.append(cur)
        for li, ln in enumerate(lines_out[:2]):
            cv2.putText(cam, ln, (100, cam_h - 25 + li * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 240, 180), 1, cv2.LINE_AA)

    # ── Build sidebar ─────────────────────────────────────────────────────────
    sidebar = np.full((cam_h, sidebar_w, 3), SIDEBAR_BG, dtype=np.uint8)
    sb_w    = sidebar_w
    pad     = 10

    # Divider colour
    div_col = (35, 45, 35)

    # ── Section 1: Bird's Eye View ─────────────────────────────────────────────
    bev_label_h = 22
    bev_h       = int(cam_h * 0.38)

    # Label
    (tw, _), _ = cv2.getTextSize("BIRD'S EYE VIEW", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(sidebar, "BIRD'S EYE VIEW",
                ((sb_w - tw) // 2, bev_label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    bev_panel = sidebar[bev_label_h:bev_h, pad:sb_w - pad]
    _draw_bev(bev_panel, cam_w, cam_h, lidar_dets)

    # Divider
    cv2.line(sidebar, (0, bev_h), (sb_w, bev_h), div_col, 1)

    # ── Section 2: AI Decision ─────────────────────────────────────────────────
    dec_y       = bev_h + 6
    dec_label_h = 20

    # "AI DECISION" label
    (tw, _), _ = cv2.getTextSize("AI DECISION", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(sidebar, "AI DECISION",
                ((sb_w - tw) // 2, dec_y + dec_label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    # Large decision banner
    banner_y1 = dec_y + dec_label_h + 4
    banner_y2 = banner_y1 + 38
    cv2.rectangle(sidebar, (pad, banner_y1), (sb_w - pad, banner_y2), dec_col, -1)
    (tw, th), _ = cv2.getTextSize(decision, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
    cv2.putText(sidebar, decision,
                ((sb_w - tw) // 2, banner_y1 + (banner_y2 - banner_y1) // 2 + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, TEXT_WHITE, 2, cv2.LINE_AA)

    # Subtitle
    sub = DEC_SUBTITLES.get(decision, "")
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(sidebar, sub,
                    ((sb_w - tw) // 2, banner_y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_WHITE, 1, cv2.LINE_AA)

    # Throttle / Brake / Steer row
    tbs_y = banner_y2 + 34
    col_w = sb_w // 3
    for i, (lbl, val, col) in enumerate([
        (f"Throttle: {throttle:.1f}", throttle, ACCENT_GREEN),
        (f"Brake: {brake_val:.1f}",   brake_val, ACCENT_BLUE),
        (f"Steer: {lane_steer:+.2f}", abs(lane_steer), ACCENT_ORANGE),
    ]):
        x = i * col_w + pad // 2
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        cv2.putText(sidebar, lbl, (x, tbs_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)

    # Divider
    div2_y = tbs_y + 14
    cv2.line(sidebar, (0, div2_y), (sb_w, div2_y), div_col, 1)

    # ── Section 3: Gauges ──────────────────────────────────────────────────────
    gauge_y   = div2_y + 8
    gauge_r   = min(34, (cam_h - gauge_y - 90) // 2)
    gauge_cx1 = sb_w // 4
    gauge_cx2 = 3 * sb_w // 4
    gauge_cy  = gauge_y + gauge_r + 20

    _draw_steering_gauge(sidebar, gauge_cx1, gauge_cy, gauge_r, lane_steer)

    # Speed gauge — green → red arc
    speed_col = (ACCENT_GREEN if speed_kmh < 40
                 else ACCENT_YELLOW if speed_kmh < 70
                 else WARN_RED)
    _draw_circular_gauge(
        sidebar, gauge_cx2, gauge_cy, gauge_r,
        value=speed_kmh, value_max=80,
        label_top="SPEED", label_bot="km/h",
        color_arc=speed_col,
        fmt="{:.0f}"
    )

    # Divider
    div3_y = gauge_cy + gauge_r + 20
    cv2.line(sidebar, (0, div3_y), (sb_w, div3_y), div_col, 1)

    # ── Section 4: System Status ───────────────────────────────────────────────
    status_y = div3_y + 16
    (tw, _), _ = cv2.getTextSize("SYSTEM STATUS", cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.putText(sidebar, "SYSTEM STATUS",
                ((sb_w - tw) // 2, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)

    dangers  = 1 if is_danger else 0
    warnings = 1 if decision in ("CAUTION", "YIELD", "SLOW", "EASE") else 0
    status_lines = [
        (f"Detections: {n_cam + n_lidar}", TEXT_WHITE),
        (f"Dangers: {dangers}   Warnings: {warnings}",
         WARN_RED if dangers else TEXT_WHITE),
        (f"Lane Steer: {lane_steer:+.3f}", ACCENT_YELLOW),
        (f"Frame: {frame_idx}", TEXT_DIM),
    ]
    for i, (txt, col) in enumerate(status_lines):
        cv2.putText(sidebar, txt,
                    (pad, status_y + 18 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)

    # ── Compose final image ────────────────────────────────────────────────────
    out = np.zeros((cam_h, total_w, 3), dtype=np.uint8)
    out[:, :cam_w]         = cam
    out[:, cam_w:total_w]  = sidebar

    # Thin separator line between camera and sidebar
    cv2.line(out, (cam_w, 0), (cam_w, cam_h), (50, 60, 50), 1)

    return out
