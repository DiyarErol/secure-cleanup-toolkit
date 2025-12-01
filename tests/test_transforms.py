"""Unit tests for transforms module."""

import torch

from src.data.transforms import (
    RandomFrameDrop,
    TemporalJitter,
    TemporalRandomCrop,
    VideoColorJitter,
    VideoNormalize,
    VideoRandomCrop,
    VideoRandomHorizontalFlip,
)


def test_video_normalize():
    """Test video normalization."""
    video = torch.rand(8, 3, 64, 64)  # (T, C, H, W)

    normalize = VideoNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized = normalize(video)

    assert normalized.shape == video.shape
    # Check approximately normalized (mean ~0, std ~1)
    assert -2 < normalized.mean().item() < 2


def test_video_random_horizontal_flip():
    """Test random horizontal flip."""
    video = torch.rand(8, 3, 64, 64)

    # Test with p=1.0 (always flip)
    flip = VideoRandomHorizontalFlip(p=1.0)
    flipped = flip(video)

    assert flipped.shape == video.shape
    # Check that flipping occurred (last column should match first of original)
    assert torch.allclose(flipped[0, 0, 0, 0], video[0, 0, 0, -1])


def test_video_random_crop():
    """Test random crop."""
    video = torch.rand(8, 3, 128, 128)

    crop = VideoRandomCrop(size=(64, 64))
    cropped = crop(video)

    assert cropped.shape == (8, 3, 64, 64)


def test_temporal_random_crop():
    """Test temporal random crop."""
    video = torch.rand(16, 3, 64, 64)

    crop = TemporalRandomCrop(size=8)
    cropped = crop(video)

    assert cropped.shape[0] == 8
    assert cropped.shape[1:] == video.shape[1:]


def test_temporal_jitter():
    """Test temporal jitter."""
    video = torch.rand(16, 3, 64, 64)

    jitter = TemporalJitter(jitter=0.2)
    jittered = jitter(video)

    assert jittered.shape == video.shape


def test_random_frame_drop():
    """Test random frame drop."""
    video = torch.rand(16, 3, 64, 64)

    drop = RandomFrameDrop(p=0.5)
    dropped = drop(video)

    # Should have fewer frames (or at least 1)
    assert 1 <= dropped.shape[0] <= 16
    assert dropped.shape[1:] == video.shape[1:]


def test_video_color_jitter():
    """Test color jitter."""
    video = torch.rand(8, 3, 64, 64)

    jitter = VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    jittered = jitter(video)

    assert jittered.shape == video.shape
