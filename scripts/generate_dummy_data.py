"""Generate dummy video data for testing the pipeline."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def generate_dummy_video(
    output_path: Path,
    duration: int = 2,
    fps: int = 10,
    resolution: tuple = (224, 224),
    color: tuple = (128, 128, 128),
) -> None:
    """Generate a dummy video with solid color frames.

    Args:
        output_path: Path to save video
        duration: Video duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        color: BGR color tuple for frames
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        resolution,
    )

    num_frames = duration * fps
    for i in range(num_frames):
        # Create frame with slight variation
        frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8)
        frame[:, :] = color

        # Add frame number text
        cv2.putText(
            frame,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Generate dummy video data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for dummy data",
    )
    parser.add_argument(
        "--videos-per-class",
        type=int,
        default=5,
        help="Number of videos per class",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=2,
        help="Video duration in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Define classes with different colors
    classes = {
        "stable": (100, 150, 100),      # Green-ish
        "critical": (100, 100, 200),    # Red-ish
        "terminal": (50, 50, 50),       # Dark gray
    }

    print(f"Generating dummy data in: {output_dir}")

    total_videos = len(classes) * args.videos_per_class
    with tqdm(total=total_videos, desc="Creating videos") as pbar:
        for class_name, color in classes.items():
            class_dir = output_dir / class_name

            for i in range(args.videos_per_class):
                video_path = class_dir / f"video_{i:03d}.mp4"

                generate_dummy_video(
                    video_path,
                    duration=args.duration,
                    fps=args.fps,
                    color=color,
                )

                pbar.update(1)

    print(f"\n✓ Generated {total_videos} dummy videos")
    print(f"  - {args.videos_per_class} videos per class")
    print(f"  - {args.duration}s duration, {args.fps} fps")
    print("\nNext steps:")
    print(f"  1. python scripts/extract_frames.py --input {output_dir} --output data/interim/frames")
    print("  2. python scripts/split_dataset.py --input data/interim/frames --output data/processed")
    print("  3. python -m src.cli train --config configs/default.yaml")


if __name__ == "__main__":
    main()
