"""
PRISM — nuScenes Data Loader
=============================
Feeds nuScenes scenes into the Sensory Core.
Handles all 6 cameras, timestamps, calibration data.

Usage:
    loader = NuScenesLoader(cfg)
    for scene_data in loader.iter_scenes():
        for frame_data in scene_data:
            core.process(frame_data["image"], frame_data["camera"])
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Iterator, Optional
from prism.utils.common import get_logger

logger = get_logger("NuScenesLoader")


class NuScenesLoader:
    """
    Wraps the nuScenes devkit for PRISM.
    Streams camera images scene by scene, sample by sample.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        data_cfg = cfg.get("data", {})
        self.data_root = str(Path(data_cfg.get("nuscenes_root", "~/prism_data/datasets/nuscenes")).expanduser())
        self.version = data_cfg.get("nuscenes_version", "v1.0-mini")
        self.camera_names = cfg.get("cameras", {}).get("names", ["CAM_FRONT"])
        self.primary_camera = cfg.get("cameras", {}).get("primary", "CAM_FRONT")
        self.nusc = None
        self._load()

    def _load(self):
        try:
            from nuscenes.nuscenes import NuScenes
            logger.info(f"Loading nuScenes {self.version} from {self.data_root}")
            self.nusc = NuScenes(
                version=self.version,
                dataroot=self.data_root,
                verbose=False
            )
            logger.info(f"Loaded {len(self.nusc.scene)} scenes, "
                        f"{len(self.nusc.sample)} samples")
        except ImportError:
            logger.error("nuscenes-devkit not installed. Run: pip install nuscenes-devkit")
            raise
        except Exception as e:
            logger.error(f"Failed to load nuScenes: {e}")
            logger.error(f"Make sure dataset is at: {self.data_root}")
            raise

    @property
    def num_scenes(self) -> int:
        return len(self.nusc.scene) if self.nusc else 0

    def get_scene_info(self, scene_idx: int) -> dict:
        scene = self.nusc.scene[scene_idx]
        return {
            "name": scene["name"],
            "description": scene["description"],
            "num_samples": scene["nbr_samples"],
            "token": scene["token"]
        }

    def load_image(self, sample_data_token: str) -> np.ndarray:
        """Load image as BGR numpy array (OpenCV format)."""
        sample_data = self.nusc.get("sample_data", sample_data_token)
        img_path = Path(self.data_root) / sample_data["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return image

    def get_calibration(self, sample_data_token: str) -> dict:
        """Get camera intrinsics and extrinsics."""
        sample_data = self.nusc.get("sample_data", sample_data_token)
        calib = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        return {
            "camera_intrinsic": np.array(calib["camera_intrinsic"]),
            "translation": np.array(calib["translation"]),
            "rotation": np.array(calib["rotation"]),
        }

    def iter_scene(
        self,
        scene_idx: int = 0,
        cameras: Optional[list] = None
    ) -> Iterator[dict]:
        """
        Iterate over all samples in a scene.
        Yields one dict per sample containing images from all cameras.

        Yields:
            {
                "sample_idx": int,
                "timestamp": float,
                "cameras": {
                    "CAM_FRONT": {
                        "image": np.ndarray,   # BGR
                        "token": str,
                        "calibration": dict
                    },
                    ...
                }
            }
        """
        if cameras is None:
            cameras = self.camera_names

        scene = self.nusc.scene[scene_idx]
        sample_token = scene["first_sample_token"]
        sample_idx = 0

        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            frame_data = {
                "sample_idx": sample_idx,
                "timestamp": sample["timestamp"] / 1e6,  # microseconds → seconds
                "cameras": {}
            }

            for cam in cameras:
                if cam not in sample["data"]:
                    continue
                sd_token = sample["data"][cam]
                try:
                    image = self.load_image(sd_token)
                    calib = self.get_calibration(sd_token)
                    frame_data["cameras"][cam] = {
                        "image": image,
                        "token": sd_token,
                        "calibration": calib
                    }
                except Exception as e:
                    logger.warning(f"Could not load {cam} at sample {sample_idx}: {e}")

            yield frame_data

            sample_token = sample["next"]
            sample_idx += 1

    def iter_primary_camera(self, scene_idx: int = 0) -> Iterator[dict]:
        """
        Simplified iterator — yields only the primary camera (CAM_FRONT).
        Good for initial testing before multi-camera setup.
        """
        for frame_data in self.iter_scene(scene_idx, cameras=[self.primary_camera]):
            cam_data = frame_data["cameras"].get(self.primary_camera)
            if cam_data is None:
                continue
            yield {
                "sample_idx": frame_data["sample_idx"],
                "timestamp": frame_data["timestamp"],
                "camera_name": self.primary_camera,
                "image": cam_data["image"],
                "calibration": cam_data["calibration"],
            }

    def get_annotations(self, sample_token: str) -> list:
        """
        Get ground truth annotations for a sample.
        Useful for evaluating detection accuracy.
        """
        sample = self.nusc.get("sample", sample_token)
        annotations = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            annotations.append({
                "category": ann["category_name"],
                "translation": ann["translation"],  # 3D position
                "size": ann["size"],                 # 3D size
                "rotation": ann["rotation"],
                "num_lidar_pts": ann["num_lidar_pts"],
                "token": ann_token
            })
        return annotations
