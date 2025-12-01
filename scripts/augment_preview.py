"""Preview data augmentations."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import VideoDataset
from src.data.transforms import get_transforms
from src.utils.io import load_yaml


def preview_augmentations(config_path: str, num_samples: int = 5) -> None:
    """
    Preview augmentations on sample videos.

    Args:
        config_path: Path to config file
        num_samples: Number of samples to visualize
    """
    # Load config
    config = load_yaml(config_path)

    # Build dataset with and without augmentation
    data_dir = Path(config["data"]["processed_dir"])
    labels = config["data"]["labels"]
    num_frames = config["data"]["preprocessing"]["num_frames"]

    # Dataset with augmentation
    transform_aug = get_transforms(config, is_train=True)
    dataset_aug = VideoDataset(
        data_dir=data_dir,
        labels=labels,
        split="train",
        num_frames=num_frames,
        transform=transform_aug,
    )

    # Dataset without augmentation
    transform_no_aug = get_transforms(config, is_train=False)
    dataset_no_aug = VideoDataset(
        data_dir=data_dir,
        labels=labels,
        split="train",
        num_frames=num_frames,
        transform=transform_no_aug,
    )

    # Visualize
    for i in range(min(num_samples, len(dataset_aug))):
        video_no_aug, label = dataset_no_aug[i]
        video_aug, _ = dataset_aug[i]

        # Select middle frame
        mid_frame = num_frames // 2

        # Denormalize
        mean = torch.tensor(config["data"]["preprocessing"]["normalize"]["mean"]).view(3, 1, 1)
        std = torch.tensor(config["data"]["preprocessing"]["normalize"]["std"]).view(3, 1, 1)

        frame_no_aug = video_no_aug[mid_frame] * std + mean
        frame_aug = video_aug[mid_frame] * std + mean

        frame_no_aug = frame_no_aug.permute(1, 2, 0).numpy()
        frame_aug = frame_aug.permute(1, 2, 0).numpy()

        # Clip to [0, 1]
        frame_no_aug = frame_no_aug.clip(0, 1)
        frame_aug = frame_aug.clip(0, 1)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(frame_no_aug)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(frame_aug)
        axes[1].set_title("Augmented")
        axes[1].axis("off")

        fig.suptitle(f"Sample {i + 1} | Label: {labels[label]}")
        fig.tight_layout()
        plt.show()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preview augmentations")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to visualize"
    )

    args = parser.parse_args()

    preview_augmentations(args.config, args.samples)


if __name__ == "__main__":
    main()
