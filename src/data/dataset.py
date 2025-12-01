"""Video dataset for severity classification."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoDataset(Dataset):
    """
    Video dataset for severity classification.

    Loads preprocessed video frames from disk and applies transforms.
    """

    def __init__(
        self,
        data_dir: str | Path,
        labels: list[str],
        split: str = "train",
        num_frames: int = 16,
        transform: Callable | None = None,
        cache: bool = False,
    ):
        """
        Initialize video dataset.

        Args:
            data_dir: Root directory containing split subdirectories
            labels: List of class labels
            split: Data split ('train', 'val', or 'test')
            num_frames: Number of frames to sample per video
            transform: Optional transform to apply to video
            cache: Whether to cache videos in memory (use for small datasets)
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.cache = cache

        # Label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Build file list
        self.samples = self._build_samples()
        logger.info(
            f"Loaded {len(self.samples)} samples from {split} split "
            f"({len(labels)} classes)"
        )

        # Cache for storing loaded videos
        self._cache: dict[int, torch.Tensor] = {}

        # Preload if caching enabled
        if self.cache:
            logger.info("Caching videos in memory...")
            self._preload_cache()

    def _build_samples(self) -> list[tuple[Path, int]]:
        """
        Build list of (video_path, label_idx) tuples.

        Returns:
            List of samples
        """
        samples = []
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for label in self.labels:
            label_dir = split_dir / label
            if not label_dir.exists():
                logger.warning(f"Label directory not found: {label_dir}, skipping")
                continue

            label_idx = self.label_to_idx[label]

            # Find all video directories (each video is a folder of frames)
            video_dirs = sorted([d for d in label_dir.iterdir() if d.is_dir()])

            for video_dir in video_dirs:
                samples.append((video_dir, label_idx))

        return samples

    def _preload_cache(self) -> None:
        """Preload all videos into cache."""
        for idx in tqdm(range(len(self.samples)), desc="Caching videos"):
            video_path, _ = self.samples[idx]
            video = self._load_video(video_path)
            self._cache[idx] = video

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load video frames from directory.

        Args:
            video_path: Path to directory containing video frames

        Returns:
            Video tensor of shape (T, C, H, W)
        """
        # Get all frame files
        frame_files = sorted(
            [f for f in video_path.glob("*.jpg") if f.is_file()]
            + [f for f in video_path.glob("*.png") if f.is_file()]
        )

        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {video_path}")

        # Sample frames uniformly
        indices = self._sample_frame_indices(len(frame_files), self.num_frames)

        # Load frames
        frames = []
        for idx in indices:
            frame_path = frame_files[idx]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Failed to load frame: {frame_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0

        return video

    def _sample_frame_indices(self, total_frames: int, num_frames: int) -> list[int]:
        """
        Sample frame indices uniformly.

        Args:
            total_frames: Total number of available frames
            num_frames: Number of frames to sample

        Returns:
            List of frame indices
        """
        if total_frames <= num_frames:
            # If fewer frames than needed, repeat last frame
            indices = list(range(total_frames))
            while len(indices) < num_frames:
                indices.append(total_frames - 1)
            return indices

        # Uniform sampling
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]
        return indices

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get video and label by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (video tensor, label index)
        """
        # Check cache first
        if idx in self._cache:
            video = self._cache[idx]
        else:
            video_path, _ = self.samples[idx]
            video = self._load_video(video_path)

        label_idx = self.samples[idx][1]

        # Apply transforms
        if self.transform is not None:
            video = self.transform(video)

        return video, label_idx

    def get_class_counts(self) -> dict[str, int]:
        """
        Get count of samples per class.

        Returns:
            Dictionary mapping class names to counts
        """
        counts = dict.fromkeys(self.labels, 0)
        for _, label_idx in self.samples:
            label = self.idx_to_label[label_idx]
            counts[label] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced loss.

        Returns:
            Tensor of class weights
        """
        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (len(self.labels) * counts[self.idx_to_label[i]])
             for i in range(len(self.labels))]
        )
        return weights


def collate_fn(batch):
    """
    Custom collate function that pads videos to same temporal length.

    Args:
        batch: List of (video, label) tuples

    Returns:
        Tuple of (batched videos, batched labels)
    """
    videos, labels = zip(*batch)

    # Find max temporal length in batch
    max_frames = max(v.shape[0] for v in videos)

    # Pad videos to same length
    padded_videos = []
    for video in videos:
        if video.shape[0] < max_frames:
            # Pad by repeating last frame
            padding = video[-1:].repeat(max_frames - video.shape[0], 1, 1, 1)
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)

    return torch.stack(padded_videos), torch.tensor(labels)


def build_dataloaders(
    config: dict[str, Any],
    transforms: dict[str, Callable],
) -> dict[str, torch.utils.data.DataLoader]:
    """
    Build train, val, and test dataloaders.

    Args:
        config: Configuration dictionary
        transforms: Dictionary of transforms for each split

    Returns:
        Dictionary of dataloaders
    """
    data_dir = Path(config["data"]["processed_dir"])
    labels = config["data"]["labels"]
    num_frames = config["data"]["preprocessing"]["num_frames"]
    batch_size = config["data"]["dataloader"]["batch_size"]
    num_workers = config["data"]["dataloader"]["num_workers"]
    pin_memory = config["data"]["dataloader"]["pin_memory"]

    dataloaders = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping")
            continue

        dataset = VideoDataset(
            data_dir=data_dir,
            labels=labels,
            split=split,
            num_frames=num_frames,
            transform=transforms.get(split),
            cache=False,  # Set to True for small datasets
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            collate_fn=collate_fn,
        )

        dataloaders[split] = dataloader
        logger.info(
            f"{split} dataloader: {len(dataset)} samples, "
            f"{len(dataloader)} batches"
        )

    return dataloaders
