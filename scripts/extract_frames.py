"""Extract frames from videos at specified FPS and resolution."""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 10,
    target_size: tuple[int, int] = (224, 224),
) -> None:
    """
    Extract frames from video.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Target frames per second
        target_size: Target frame size (width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Invalid FPS for video: {video_path}")
        return

    frame_interval = int(video_fps / fps) if fps < video_fps else 1

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_idx % frame_interval == 0:
            # Resize
            frame_resized = cv2.resize(frame, target_size)

            # Save
            output_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame_resized)
            saved_idx += 1

        frame_idx += 1

    cap.release()


def process_directory(
    input_dir: Path,
    output_dir: Path,
    fps: int = 10,
    target_size: tuple[int, int] = (224, 224),
) -> None:
    """
    Process all videos in directory maintaining class structure.

    Args:
        input_dir: Input directory with class subdirectories
        output_dir: Output directory
        fps: Target frames per second
        target_size: Target frame size
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # Find all videos
    videos = []
    for ext in video_extensions:
        videos.extend(input_dir.rglob(f"*{ext}"))

    print(f"Found {len(videos)} videos")

    for video_path in tqdm(videos, desc="Extracting frames"):
        # Maintain directory structure
        rel_path = video_path.relative_to(input_dir)
        output_subdir = output_dir / rel_path.parent / video_path.stem

        extract_frames(video_path, output_subdir, fps, target_size)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target size (width height)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    target_size = tuple(args.size)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    process_directory(input_dir, output_dir, args.fps, target_size)
    print(f"Done! Frames saved to: {output_dir}")


if __name__ == "__main__":
    main()
