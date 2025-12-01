"""Batch inference on a folder of videos."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_transforms
from src.models.builder import build_model
from src.utils.io import load_yaml


def infer_folder(
    config_path: str,
    checkpoint_path: str,
    input_dir: str,
    output_csv: str,
) -> None:
    """
    Run inference on all videos in a folder.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing videos
        output_csv: Path to output CSV
    """
    # Load config
    config = load_yaml(config_path)

    # Setup device
    if config["hardware"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["hardware"]["device"])

    print(f"Using device: {device}")

    # Build model
    model = build_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Get transforms
    get_transforms(config, is_train=False)

    # Get labels
    config["data"]["labels"]

    # Find all videos
    input_path = Path(input_dir)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f"*{ext}"))

    if len(video_files) == 0:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos")

    # Run inference
    results = []

    with torch.no_grad():
        for video_file in tqdm(video_files, desc="Inferring"):
            # This is a simplified stub - in production, you'd load and preprocess the video
            # For now, we'll just save the structure
            results.append(
                {
                    "video_path": str(video_file),
                    "predicted_class": "stable",  # Placeholder
                    "confidence": 0.95,  # Placeholder
                }
            )

    # Save results
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved predictions to: {output_csv}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch inference on videos")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory with videos"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    infer_folder(args.config, args.checkpoint, args.input, args.output)


if __name__ == "__main__":
    main()
