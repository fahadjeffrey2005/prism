"""
PRISM — Video Frame Extractor
==============================
Breaks a video file into frames and saves them into a folder named after
the video, inside a designated raw_data directory.

Usage:
    python scripts/extract_frames.py path/to/video.mp4
    python scripts/extract_frames.py path/to/video.mov --fps 2
    python scripts/extract_frames.py path/to/video.mp4 --output-dir /custom/path
    python scripts/extract_frames.py path/to/video.mp4 --every-n 10   # every 10th frame

Output:
    <output_dir>/<video_stem>/frame_0001.jpg
    <output_dir>/<video_stem>/frame_0002.jpg
    ...
"""

import cv2
import argparse
import sys
from pathlib import Path


DEFAULT_RAW_DATA_DIR = Path(__file__).parent.parent / "prism" / "new data" / "raw_video_frames"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = None,
    every_n: int = 1,
    quality: int = 95,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / source_fps

    # Compute skip interval
    if fps is not None:
        # Keep one frame every (source_fps / desired_fps) source frames
        skip = max(1, round(source_fps / fps))
    else:
        skip = max(1, every_n)

    # Output folder: <output_dir>/<video_stem>/
    folder = output_dir / video_path.stem
    folder.mkdir(parents=True, exist_ok=True)

    print(f"Video     : {video_path.name}")
    print(f"Resolution: {width}x{height}  |  Source FPS: {source_fps:.2f}")
    print(f"Duration  : {duration_s:.1f}s  |  Total frames: {total_frames}")
    print(f"Saving 1 frame every {skip} source frames → ~{total_frames // skip} output frames")
    print(f"Output    : {folder}")
    print("-" * 60)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            out_path = folder / f"frame_{saved + 1:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1

            if saved % 20 == 0 or saved == 1:
                print(f"  Saved {saved} frames (source frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    print("-" * 60)
    print(f"Done. {saved} frames saved to: {folder}")
    return folder


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video", help="Path to video file (.mp4, .mov, .avi, …)")
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Root directory for output folders (default: {DEFAULT_RAW_DATA_DIR})"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Desired output frame rate (e.g. 2 = 2 frames/sec). Overrides --every-n."
    )
    parser.add_argument(
        "--every-n", type=int, default=1,
        help="Save every Nth source frame (default: 1 = all frames)."
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)."
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_RAW_DATA_DIR

    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
        every_n=args.every_n,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
