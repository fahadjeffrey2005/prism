"""
PRISM — Video Frame Extractor
"""

import cv2
import argparse
import sys
from pathlib import Path

DEFAULT_RAW_DATA_DIR = Path(__file__).parent.parent / "prism" / "new data" / "raw_video_frames"

def extract_frames(video_path, output_dir, fps=None, every_n=1, quality=95):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, round(source_fps / fps)) if fps else max(1, every_n)
    folder = output_dir / video_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % skip == 0:
            cv2.imwrite(str(folder / f"frame_{saved+1:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"Done. {saved} frames → {folder}")
    return folder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fps",        type=float, default=None)
    p.add_argument("--every-n",    type=int,   default=1)
    p.add_argument("--quality",    type=int,   default=95)
    args = p.parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_RAW_DATA_DIR
    extract_frames(video_path, output_dir, args.fps, args.every_n, args.quality)

if __name__ == "__main__":
    main()
