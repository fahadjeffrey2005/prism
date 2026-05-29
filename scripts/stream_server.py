"""
PRISM — Live Stream Server
===========================
Runs the PRISM pipeline on nuScenes frames and streams annotated
video + metrics to the web dashboard in real time via WebSocket.

Install dependency (once):
    /home/koushik-test/cad_pipeline_env/bin/pip install websockets --break-system-packages

Start server (on Jetson):
    cd ~/prism
    /home/koushik-test/cad_pipeline_env/bin/python3 scripts/stream_server.py

Then open dashboard_live.html in any browser on your Mac and connect to:
    ws://<jetson-ip>:8765

Find Jetson IP:    hostname -I | awk '{print $1}'
"""

import sys
import json
import time
import base64
import asyncio
import argparse
import threading
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from prism.utils.common import load_config, get_logger

logger = get_logger("StreamServer")

# ── Colour maps ───────────────────────────────────────────────────────────────

DECISION_BGR = {
    "CLEAR":     ( 29, 158, 118),
    "MONITOR":   ( 24,  95, 165),
    "EASE":      ( 29, 158,  29),
    "SLOW":      (128, 128, 128),
    "CAUTION":   ( 39, 159, 239),
    "YIELD":     ( 27, 159, 239),
    "STOP":      ( 29,  71, 153),
    "EMERGENCY": ( 20,  20, 163),
}

DECISION_HEX = {
    "CLEAR":     "#1d9e75",
    "MONITOR":   "#185fa5",
    "EASE":      "#1d9e75",
    "SLOW":      "#888780",
    "CAUTION":   "#ba7517",
    "YIELD":     "#ef9f27",
    "STOP":      "#993c1d",
    "EMERGENCY": "#e24b4a",
}


# ── Frame Annotator ───────────────────────────────────────────────────────────

class FrameAnnotator:
    """Draws bounding boxes, decision banner, VLM caption, and risk bar."""

    @staticmethod
    def _dist_color(d_m: float):
        if d_m < 5:   return ( 20,  20, 163)   # critical — dark red
        if d_m < 15:  return ( 44,  75, 226)   # close    — red
        if d_m < 30:  return ( 39, 159, 239)   # medium   — amber
        return             (118, 200,  29)      # safe     — green

    def annotate(self, frame, metric_dets, arb, world_state, vlm_out, latency_ms):
        out = frame.copy()
        h, w = out.shape[:2]

        decision  = arb.action if arb else "---"
        d_bgr     = DECISION_BGR.get(decision, (80, 80, 80))
        risk      = float(world_state.risk_score) if world_state else 0.0
        n_actors  = len(metric_dets)

        # ── Top decision banner ───────────────────────────────────────────
        cv2.rectangle(out, (0, 0), (w, 36), (15, 15, 15), -1)
        cv2.rectangle(out, (0, 0), (148, 36), d_bgr, -1)
        cv2.putText(out, decision, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        info = f"risk {risk:.2f}   actors {n_actors}   {latency_ms:.0f}ms"
        cv2.putText(out, info, (158, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

        # ── Bounding boxes ────────────────────────────────────────────────
        for det in metric_dets:
            try:
                bb  = det.bbox
                d   = det.distance_m or 50.0
                c   = self._dist_color(d)
                x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
                cls   = (getattr(bb, "class_name", "?") or "?")[:4].upper()
                label = f"{cls} {d:.1f}m"
                lw    = len(label) * 8 + 4
                cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + lw, y1), c, -1)
                cv2.putText(out, label, (x1 + 2, max(13, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                continue

        # ── VLM caption strip ─────────────────────────────────────────────
        if vlm_out and getattr(vlm_out, "scene_summary", None):
            cap = vlm_out.scene_summary[:90]
            cv2.rectangle(out, (0, h - 40), (w, h), (10, 10, 10), -1)
            cv2.rectangle(out, (0, h - 40), (4, h), (139, 212, 159), -1)
            cv2.putText(out, f" VLM  {cap}", (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (139, 212, 159), 1, cv2.LINE_AA)

        # ── Risk bar (right edge) ─────────────────────────────────────────
        bar_h = max(1, int((h - 36) * min(risk, 1.0)))
        cv2.rectangle(out, (w - 6, 36), (w, h), (30, 30, 30), -1)
        cv2.rectangle(out, (w - 6, h - bar_h), (w, h), d_bgr, -1)

        return out


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def build_pipeline(cfg):
    from prism.sensory_core.core import SensoryCore
    from prism.sensory_core.metric_depth import MetricDepthEngine
    from prism.world_model.world_model import WorldModel
    from prism.predictive_engine.engine import PredictiveEngine
    from prism.predictive_engine.decision import SmartDecisionEngine
    from prism.semantic_reasoner.reasoner import SemanticReasoner
    from prism.arbitration.core import ArbitrationCore
    from prism.planner.planner import AdaptivePlanner
    return {
        "core":       SensoryCore(cfg),
        "depth":      MetricDepthEngine(cfg),
        "world":      WorldModel(cfg),
        "predictor":  PredictiveEngine(cfg),
        "decision":   SmartDecisionEngine(),
        "reasoner":   SemanticReasoner(cfg),
        "arbitrator": ArbitrationCore(cfg),
        "planner":    AdaptivePlanner(cfg),
    }


def run_pipeline_frame(pipeline, image, calib, timestamp):
    t0 = time.time()
    sensory  = pipeline["core"].process(image, "CAM_FRONT", timestamp=timestamp)
    pipeline["depth"].update_intrinsics(calib)
    mdets, _ = pipeline["depth"].process_frame(
        image, sensory.detections, run_model=sensory.depth_map is not None)
    ws   = pipeline["world"].update(sensory, calibration=calib)
    ps   = pipeline["predictor"].update(ws, mdets)
    sc   = pipeline["decision"].assess(ws, ps, mdets)
    vlmo = pipeline["reasoner"].update(image, ws, ps)
    if vlmo and vlmo.actor_intents:
        pipeline["predictor"].update_vlm_intents(vlmo.actor_intents)
    if vlmo:
        pipeline["arbitrator"].update_vlm(vlmo.to_arb_dict())
    arb  = pipeline["arbitrator"].arbitrate(ws, ps, sc, timestamp=timestamp)
    ctrl = pipeline["planner"].plan(arb, metric_dets=mdets, timestamp=timestamp)
    lat  = (time.time() - t0) * 1000
    return ctrl, arb, ws, mdets, vlmo, lat


def encode_jpg(frame, quality=75):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ── Pipeline thread ───────────────────────────────────────────────────────────

def pipeline_thread(cfg, state, loop, async_queue):
    """
    Background thread: loads pipeline + nuScenes, processes frames, pushes
    results to the asyncio queue so the WebSocket loop can broadcast them.
    """
    from prism.sensory_core.data_loader import NuScenesLoader

    annotator = FrameAnnotator()

    def _safe_push(msg):
        """Thread-safe put onto asyncio.Queue — drops if full."""
        def _put():
            try:
                async_queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass
        loop.call_soon_threadsafe(_put)

    _safe_push({"type": "status", "state": "loading",
                "message": "Building PRISM pipeline..."})
    logger.info("Building pipeline...")
    pipeline = build_pipeline(cfg)
    logger.info("Pipeline ready. Loading nuScenes...")

    _safe_push({"type": "status", "state": "loading",
                "message": "Loading nuScenes dataset..."})
    loader = NuScenesLoader(cfg)
    logger.info(f"nuScenes loaded — {loader.num_scenes} scenes")

    # Send scene list to dashboard
    _safe_push({"type": "scenes", "scenes": [
        {
            "idx":         i,
            "name":        loader.get_scene_info(i)["name"],
            "description": loader.get_scene_info(i)["description"],
            "frames":      loader.get_scene_info(i)["num_samples"],
        }
        for i in range(loader.num_scenes)
    ]})

    dec_counts   = defaultdict(int)
    total_frames = 0
    vlm_triggers = 0
    latencies    = deque(maxlen=60)
    current_scene = state["scene_idx"]

    while True:
        # Check for scene change request
        if state.get("change_scene_to") is not None:
            current_scene         = state.pop("change_scene_to")
            state["scene_idx"]    = current_scene
            dec_counts   = defaultdict(int)
            total_frames = 0
            vlm_triggers = 0
            latencies    = deque(maxlen=60)

        scene_info = loader.get_scene_info(current_scene)
        _safe_push({"type": "status", "state": "running",
                    "message": f"Scene {current_scene}: {scene_info['name']}"})
        logger.info(f"Streaming scene {current_scene}: {scene_info['name']}")

        for fd in loader.iter_primary_camera(scene_idx=current_scene):
            # Respect pause
            while state.get("paused"):
                time.sleep(0.1)
                if state.get("change_scene_to") is not None:
                    break

            # Scene change mid-run
            if state.get("change_scene_to") is not None:
                break

            # VLM live toggle
            pipeline["reasoner"].vlm._enabled = state.get("vlm_enabled", True)

            image     = fd["image"]
            calib     = fd.get("calibration", {})
            timestamp = fd["timestamp"]
            frame_idx = fd["sample_idx"]

            ctrl, arb, ws, mdets, vlmo, lat = run_pipeline_frame(
                pipeline, image, calib, timestamp)

            total_frames += 1
            latencies.append(lat)
            dec_counts[arb.action] += 1
            if vlmo:
                vlm_triggers += 1

            # Annotate + downscale for streaming
            ann     = annotator.annotate(image, mdets, arb, ws, vlmo, lat)
            sw, sh  = 640, 360
            raw_s   = cv2.resize(image, (sw, sh))
            ann_s   = cv2.resize(ann,   (sw, sh))

            n   = max(1, total_frames)
            msg = {
                "type":           "frame",
                "scene":          scene_info["name"],
                "frame_idx":      frame_idx,
                "total_frames":   scene_info["num_samples"],
                "input":          encode_jpg(raw_s),
                "output":         encode_jpg(ann_s),
                "decision":       arb.action,
                "decision_color": DECISION_HEX.get(arb.action, "#888"),
                "risk_score":     round(float(ws.risk_score) if ws else 0.0, 3),
                "risk_level":     int(ws.risk_level) if ws else 0,
                "actors":         len(mdets),
                "latency_ms":     round(lat, 1),
                "avg_latency_ms": round(float(np.mean(latencies)), 1),
                "vlm_active":     vlmo is not None,
                "vlm_caption":    (getattr(vlmo, "scene_summary", "") or "")[:90] if vlmo else "",
                "vlm_rate_pct":   round(vlm_triggers / n * 100, 1),
                "decisions":      {k: round(v / n * 100, 1) for k, v in dec_counts.items()},
                "total_processed": total_frames,
            }
            _safe_push(msg)

            # Pace to replay speed (default 1× = 0.5s/frame for nuScenes)
            target_dt = 0.5 / max(0.1, state.get("speed", 1.0))
            sleep_s   = max(0.0, target_dt - lat / 1000.0)
            if sleep_s > 0.02:
                time.sleep(sleep_s)

        else:
            # Scene finished — loop after brief pause
            logger.info(f"Scene {current_scene} complete — looping")
            time.sleep(2.0)


# ── WebSocket server ──────────────────────────────────────────────────────────

async def ws_main(args, cfg):
    import websockets

    state = {
        "scene_idx":   args.scene,
        "paused":      False,
        "speed":       1.0,
        "vlm_enabled": not args.no_vlm,
    }

    async_queue = asyncio.Queue(maxsize=3)
    clients     = set()
    loop        = asyncio.get_event_loop()

    # Start pipeline background thread
    t = threading.Thread(
        target=pipeline_thread,
        args=(cfg, state, loop, async_queue),
        daemon=True,
    )
    t.start()

    # ── WebSocket connection handler ──────────────────────────────────────
    async def handler(websocket):
        clients.add(websocket)
        logger.info(f"Client connected  ({len(clients)} active)")

        # Replay latest message so dashboard loads immediately
        # (pipeline may already be running)

        try:
            async for raw in websocket:
                try:
                    cmd = json.loads(raw)
                    c   = cmd.get("cmd")
                    if   c == "pause":
                        state["paused"] = True
                    elif c == "resume":
                        state["paused"] = False
                    elif c == "set_scene":
                        state["change_scene_to"] = int(cmd.get("scene_idx", 0))
                        state["paused"] = False
                    elif c == "set_speed":
                        state["speed"] = float(
                            np.clip(cmd.get("multiplier", 1.0), 0.25, 8.0))
                    elif c == "set_vlm":
                        state["vlm_enabled"] = bool(cmd.get("enabled", True))
                    logger.debug(f"CMD {c} from client")
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            clients.discard(websocket)
            logger.info(f"Client disconnected ({len(clients)} active)")

    # ── Broadcast loop ────────────────────────────────────────────────────
    async def broadcast_loop():
        while True:
            try:
                msg  = await asyncio.wait_for(async_queue.get(), timeout=1.0)
                data = json.dumps(msg)
                dead = set()
                for ws in list(clients):
                    try:
                        await ws.send(data)
                    except Exception:
                        dead.add(ws)
                clients.difference_update(dead)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.warning(f"Broadcast error: {e}")

    # ── Serve ─────────────────────────────────────────────────────────────
    server = await websockets.serve(handler, args.host, args.port)
    logger.info("=" * 60)
    logger.info(f"PRISM Stream Server ready")
    logger.info(f"  WebSocket : ws://{args.host}:{args.port}")
    logger.info(f"  Dashboard : open dashboard_live.html in browser")
    logger.info(f"  Jetson IP : run  hostname -I | awk '{{print $1}}'")
    logger.info("=" * 60)

    await asyncio.gather(server.wait_closed(), broadcast_loop())


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRISM live stream server")
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8765)
    p.add_argument("--scene",  type=int, default=0,
                   help="nuScenes scene index to start on")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--no-vlm", action="store_true",
                   help="Start with VLM disabled")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    if args.no_vlm:
        cfg.setdefault("vlm", {})["enabled"] = False
    try:
        asyncio.run(ws_main(args, cfg))
    except KeyboardInterrupt:
        logger.info("Server stopped")
