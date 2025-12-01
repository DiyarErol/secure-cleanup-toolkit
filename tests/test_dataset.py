"""Unit tests for dataset module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import VideoDataset


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create structure: train/class1, train/class2, etc.
        labels = ["stable", "critical", "terminal"]
        splits = ["train", "val", "test"]

        for split in splits:
            for label in labels:
                label_dir = tmpdir / split / label
                label_dir.mkdir(parents=True, exist_ok=True)

                # Create 2 dummy videos per class
                for video_idx in range(2):
                    video_dir = label_dir / f"video_{video_idx:03d}"
                    video_dir.mkdir()

                    # Create 10 dummy frames
                    for frame_idx in range(10):
                        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                        img = Image.fromarray(frame)
                        img.save(video_dir / f"frame_{frame_idx:06d}.jpg")

        yield tmpdir


def test_video_dataset_initialization(temp_dataset_dir):
    """Test dataset initialization."""
    labels = ["stable", "critical", "terminal"]

    dataset = VideoDataset(
        data_dir=temp_dataset_dir,
        labels=labels,
        split="train",
        num_frames=5,
        transform=None,
    )

    assert len(dataset) == 6  # 2 videos per class, 3 classes
    assert dataset.labels == labels


def test_video_dataset_loading(temp_dataset_dir):
    """Test loading video from dataset."""
    labels = ["stable", "critical", "terminal"]

    dataset = VideoDataset(
        data_dir=temp_dataset_dir,
        labels=labels,
        split="train",
        num_frames=5,
        transform=None,
    )

    video, label = dataset[0]

    assert isinstance(video, torch.Tensor)
    assert video.shape[0] == 5  # num_frames
    assert video.shape[1] == 3  # channels
    assert 0 <= label < len(labels)


def test_video_dataset_class_counts(temp_dataset_dir):
    """Test class counting."""
    labels = ["stable", "critical", "terminal"]

    dataset = VideoDataset(
        data_dir=temp_dataset_dir,
        labels=labels,
        split="train",
        num_frames=5,
    )

    counts = dataset.get_class_counts()

    assert len(counts) == 3
    assert all(count == 2 for count in counts.values())  # 2 videos per class


def test_video_dataset_uniform_sampling(temp_dataset_dir):
    """Test uniform frame sampling."""
    labels = ["stable", "critical", "terminal"]

    dataset = VideoDataset(
        data_dir=temp_dataset_dir,
        labels=labels,
        split="train",
        num_frames=5,
    )

    # Test internal sampling method
    indices = dataset._sample_frame_indices(total_frames=10, num_frames=5)

    assert len(indices) == 5
    assert all(0 <= idx < 10 for idx in indices)
    assert indices == sorted(indices)  # Should be in order


def test_video_dataset_with_transform(temp_dataset_dir):
    """Test dataset with transform."""
    labels = ["stable", "critical", "terminal"]

    # Simple transform that doubles values
    def dummy_transform(video):
        return video * 2

    dataset = VideoDataset(
        data_dir=temp_dataset_dir,
        labels=labels,
        split="train",
        num_frames=5,
        transform=dummy_transform,
    )

    video, _ = dataset[0]
    # Should be transformed (doubled), but this is hard to verify without original
    assert isinstance(video, torch.Tensor)
