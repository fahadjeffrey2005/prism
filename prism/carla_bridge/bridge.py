"""
PRISM — CARLA Bridge
=====================
Connects PRISM's full pipeline to a running CARLA server.

Architecture:
    CARLA server  →  RGB camera sensor  →  PRISM pipeline  →  vehicle control
                  →  collision sensor   →  metrics logger
                  →  GPS / IMU          →  route tracker

The bridge is a thin adapter:
    - CARLA provides camera frames (numpy BGR arrays)
    - PRISM processes them through all 6 layers
    - Planner outputs (throttle, brake, steering) go back to CARLA
    - Collision, TTC, jerk logged for paper metrics

Sensor setup (matches nuScenes CAM_FRONT approximately):
    Resolution:   1600 × 900
    FOV:          70°
    Position:     1.5m forward, 2.1m above ground
"""

from __future__ import annotations

import time
import queue
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from prism.utils.common import get_logger

logger = get_logger("CARLABridge")


# ── Sensor configuration ───────────────────────────────────────────────────────

CAMERA_TRANSFORM_ARGS = dict(x=1.5, z=2.1, pitch=-5.0)
CAMERA_FOV    = 70.0
CAMERA_WIDTH  = 800
CAMERA_HEIGHT = 450     # lower res = faster, still good for detection


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SensorFrame:
    image:       np.ndarray        # (H, W, 3) BGR
    timestamp:   float
    frame_id:    int
    speed_mps:   float = 0.0      # from IMU/velocity


@dataclass
class CollisionEvent:
    timestamp:  float
    frame_id:   int
    other_actor: str
    impulse_norm: float            # N·s


@dataclass
class EvalMetrics:
    """Accumulated metrics for one CARLA run."""
    # Collision
    total_collisions:   int   = 0
    collision_events:   List[CollisionEvent] = field(default_factory=list)

    # Route
    waypoints_total:    int   = 0
    waypoints_reached:  int   = 0

    # Speed / comfort
    speeds_mps:         List[float] = field(default_factory=list)
    jerks:              List[float] = field(default_factory=list)
    throttles:          List[float] = field(default_factory=list)
    brakes:             List[float] = field(default_factory=list)

    # Planner decisions
    decisions:          dict  = field(default_factory=dict)
    emergency_brakes:   int   = 0

    # VLM
    vlm_triggers:       int   = 0
    total_frames:       int   = 0

    # Timing
    frame_latencies_ms: List[float] = field(default_factory=list)

    @property
    def collision_rate(self) -> float:
        """Collisions per km driven."""
        dist_km = self.distance_km
        return self.total_collisions / dist_km if dist_km > 0 else 0.0

    @property
    def distance_km(self) -> float:
        if len(self.speeds_mps) < 2:
            return 0.0
        dt = 0.1   # ~10fps CARLA
        return float(np.array(self.speeds_mps).sum() * dt / 1000.0)

    @property
    def route_completion(self) -> float:
        if self.waypoints_total == 0:
            return 0.0
        return self.waypoints_reached / self.waypoints_total

    @property
    def mean_speed_kmh(self) -> float:
        return float(np.mean(self.speeds_mps) * 3.6) if self.speeds_mps else 0.0

    @property
    def rms_jerk(self) -> float:
        return float(np.sqrt(np.mean(np.array(self.jerks) ** 2))) if self.jerks else 0.0

    @property
    def vlm_trigger_rate(self) -> float:
        return self.vlm_triggers / max(1, self.total_frames)

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.frame_latencies_ms)) if self.frame_latencies_ms else 0.0


# ── CARLA Bridge ──────────────────────────────────────────────────────────────

class CARLABridge:
    """
    Connects PRISM to a running CARLA server.

    Usage:
        bridge = CARLABridge(host="localhost", port=2000)
        bridge.setup(world_name="Town03")
        for frame in bridge.run(max_steps=500):
            # frame is a SensorFrame — feed to PRISM
            control = your_prism_pipeline(frame)
            bridge.apply_control(control.throttle, control.brake, control.steer)
        metrics = bridge.metrics
    """

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 10.0):
        self.host    = host
        self.port    = port
        self.timeout = timeout

        self._client     = None
        self._world      = None
        self._vehicle    = None
        self._camera     = None
        self._collision  = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)

        self.metrics = EvalMetrics()
        self._prev_speed = 0.0
        self._prev_accel = 0.0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect(self):
        """Connect to CARLA server."""
        import carla
        self._carla = carla
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        ver = self._client.get_server_version()
        logger.info(f"Connected to CARLA {ver} at {self.host}:{self.port}")

    def setup(self, world_name: str = "Town03", weather: str = "ClearNoon"):
        """Load world, spawn vehicle and sensors."""
        carla = self._carla
        logger.info(f"Loading world: {world_name}")
        self._world = self._client.load_world(world_name)
        self._world.set_weather(getattr(carla.WeatherParameters, weather))

        # Synchronous mode for deterministic evaluation
        settings = self._world.get_settings()
        settings.synchronous_mode  = True
        settings.fixed_delta_seconds = 0.1   # 10fps
        self._world.apply_settings(settings)

        self._spawn_vehicle()
        self._attach_camera()
        self._attach_collision_sensor()
        logger.info("CARLA setup complete")

    def _spawn_vehicle(self):
        carla = self._carla
        bp_lib   = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_pts  = self._world.get_map().get_spawn_points()
        if not spawn_pts:
            raise RuntimeError("No spawn points in this map")
        self._vehicle = None
        for sp in spawn_pts:
            try:
                self._vehicle = self._world.spawn_actor(vehicle_bp, sp)
                break
            except Exception:
                continue
        if self._vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")
        logger.info(f"Vehicle spawned: {self._vehicle.type_id}")

    def _attach_camera(self):
        carla  = self._carla
        bp_lib = self._world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_bp.set_attribute("fov", str(CAMERA_FOV))
        transform = carla.Transform(
            carla.Location(**CAMERA_TRANSFORM_ARGS),
            carla.Rotation(pitch=CAMERA_TRANSFORM_ARGS.get("pitch", 0.0))
        )
        self._camera = self._world.spawn_actor(cam_bp, transform, attach_to=self._vehicle)
        self._camera.listen(self._on_camera_frame)
        logger.info(f"Camera attached: {CAMERA_WIDTH}x{CAMERA_HEIGHT} FOV={CAMERA_FOV}")

    def _attach_collision_sensor(self):
        carla  = self._carla
        bp_lib = self._world.get_blueprint_library()
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._collision_sensor.listen(self._on_collision)

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _on_camera_frame(self, image):
        """Convert CARLA raw image → numpy BGR."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3][:, :, ::-1].copy()   # RGBA → BGR

        # Get vehicle speed
        vel = self._vehicle.get_velocity()
        speed = float((vel.x**2 + vel.y**2 + vel.z**2)**0.5)

        frame = SensorFrame(
            image=bgr,
            timestamp=image.timestamp,
            frame_id=image.frame,
            speed_mps=speed,
        )
        # Drop old frame if queue full — keep freshest
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
        self._frame_q.put_nowait(frame)

    def _on_collision(self, event):
        impulse = event.normal_impulse
        norm = float((impulse.x**2 + impulse.y**2 + impulse.z**2)**0.5)
        other = event.other_actor.type_id if event.other_actor else "unknown"
        ev = CollisionEvent(
            timestamp=event.timestamp,
            frame_id=event.frame,
            other_actor=other,
            impulse_norm=norm,
        )
        self.metrics.collision_events.append(ev)
        self.metrics.total_collisions += 1
        logger.warning(f"COLLISION with {other}  impulse={norm:.1f} N·s")

    # ── Control ───────────────────────────────────────────────────────────────

    def apply_control(self, throttle: float, brake: float, steer: float = 0.0):
        """Send control commands to CARLA vehicle."""
        carla = self._carla
        ctrl = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake,    0.0, 1.0)),
            steer=float(np.clip(steer,   -1.0, 1.0)),
        )
        self._vehicle.apply_control(ctrl)
        self.metrics.throttles.append(ctrl.throttle)
        self.metrics.brakes.append(ctrl.brake)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 500):
        """
        Generator that yields SensorFrames.
        Caller applies PRISM processing and calls apply_control() each step.
        """
        logger.info(f"Starting CARLA eval loop ({max_steps} steps)")
        step = 0
        while step < max_steps:
            self._world.tick()   # advance simulation one step

            try:
                frame = self._frame_q.get(timeout=2.0)
            except queue.Empty:
                logger.warning("Camera frame timeout")
                continue

            # Track speed for jerk computation
            dt = 0.1
            accel = (frame.speed_mps - self._prev_speed) / dt
            jerk  = (accel - self._prev_accel) / dt
            self.metrics.jerks.append(abs(jerk))
            self.metrics.speeds_mps.append(frame.speed_mps)
            self._prev_speed = frame.speed_mps
            self._prev_accel = accel
            self.metrics.total_frames += 1

            step += 1
            yield frame

    # ── Teardown ──────────────────────────────────────────────────────────────

    def cleanup(self):
        """Destroy all CARLA actors."""
        for actor in [self._camera, getattr(self, '_collision_sensor', None), self._vehicle]:
            if actor and actor.is_alive:
                actor.destroy()
        # Restore async mode
        if self._world:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)
        logger.info("CARLA actors cleaned up")
