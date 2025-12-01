"""Video transforms for spatial and temporal augmentation."""

from typing import Any

import torch
import torchvision.transforms as T


class VideoNormalize:
    """Normalize video tensor with mean and std."""

    def __init__(self, mean: list[float], std: list[float]):
        """
        Initialize video normalizer.

        Args:
            mean: Mean values for each channel [R, G, B]
            std: Standard deviation values for each channel [R, G, B]
        """
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Normalize video tensor.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Normalized video tensor
        """
        return (video - self.mean) / self.std


class VideoRandomHorizontalFlip:
    """Randomly flip video horizontally."""

    def __init__(self, p: float = 0.5):
        """
        Initialize random horizontal flip.

        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply random horizontal flip to video.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Flipped video tensor
        """
        if torch.rand(1).item() < self.p:
            return torch.flip(video, dims=[-1])  # Flip width dimension
        return video


class VideoRandomCrop:
    """Randomly crop video to given size."""

    def __init__(self, size: tuple[int, int]):
        """
        Initialize random crop.

        Args:
            size: Target size (height, width)
        """
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply random crop to video.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Cropped video tensor
        """
        t, c, h, w = video.shape
        th, tw = self.size

        if h == th and w == tw:
            return video

        # Random crop position
        top = torch.randint(0, h - th + 1, (1,)).item()
        left = torch.randint(0, w - tw + 1, (1,)).item()

        return video[:, :, top : top + th, left : left + tw]


class VideoColorJitter:
    """Apply color jitter to video frames."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        """
        Initialize color jitter.

        Args:
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
        """
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply color jitter to video.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Jittered video tensor
        """
        # Apply same jitter to all frames
        t, c, h, w = video.shape
        jittered = []
        for i in range(t):
            frame = video[i]
            jittered.append(self.jitter(frame))
        return torch.stack(jittered, dim=0)


class TemporalRandomCrop:
    """Randomly crop video in temporal dimension."""

    def __init__(self, size: int):
        """
        Initialize temporal random crop.

        Args:
            size: Number of frames to sample
        """
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Randomly crop video temporally.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Temporally cropped video
        """
        t = video.shape[0]
        if t <= self.size:
            return video

        start = torch.randint(0, t - self.size + 1, (1,)).item()
        return video[start : start + self.size]


class TemporalJitter:
    """Apply temporal jitter by randomly shifting frame selection."""

    def __init__(self, jitter: float = 0.1):
        """
        Initialize temporal jitter.

        Args:
            jitter: Maximum jitter as fraction of video length
        """
        self.jitter = jitter

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal jitter to video.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Jittered video tensor
        """
        t = video.shape[0]
        max_shift = int(t * self.jitter)
        if max_shift == 0:
            return video

        # Random shift
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        indices = torch.arange(t) + shift
        indices = torch.clamp(indices, 0, t - 1)

        return video[indices]


class RandomFrameDrop:
    """Randomly drop frames from video."""

    def __init__(self, p: float = 0.1):
        """
        Initialize random frame drop.

        Args:
            p: Probability of dropping each frame
        """
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop frames from video.

        Args:
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Video with some frames dropped
        """
        t = video.shape[0]
        mask = torch.rand(t) > self.p
        if mask.sum() == 0:  # Keep at least one frame
            mask[0] = True
        return video[mask]


def get_transforms(
    config: dict[str, Any], is_train: bool = True
) -> T.Compose:
    """
    Get video transforms based on config.

    Args:
        config: Configuration dictionary
        is_train: Whether transforms are for training (includes augmentation)

    Returns:
        Composed transforms
    """
    transforms_list: list[Any] = []

    # Extract config
    resolution = config["data"]["preprocessing"]["resolution"]
    mean = config["data"]["preprocessing"]["normalize"]["mean"]
    std = config["data"]["preprocessing"]["normalize"]["std"]

    if is_train and config["data"]["augmentation"]["enabled"]:
        # Training augmentations
        aug_cfg = config["data"]["augmentation"]

        # Spatial augmentations
        if aug_cfg.get("random_crop", False):
            transforms_list.append(VideoRandomCrop(size=tuple(resolution)))

        if aug_cfg.get("horizontal_flip", 0.0) > 0:
            transforms_list.append(VideoRandomHorizontalFlip(p=aug_cfg["horizontal_flip"]))

        if "color_jitter" in aug_cfg:
            cj = aug_cfg["color_jitter"]
            transforms_list.append(
                VideoColorJitter(
                    brightness=cj.get("brightness", 0.0),
                    contrast=cj.get("contrast", 0.0),
                    saturation=cj.get("saturation", 0.0),
                    hue=cj.get("hue", 0.0),
                )
            )

        # Temporal augmentations
        if aug_cfg.get("temporal_jitter", 0.0) > 0:
            transforms_list.append(TemporalJitter(jitter=aug_cfg["temporal_jitter"]))

        if aug_cfg.get("random_frame_drop", 0.0) > 0:
            transforms_list.append(RandomFrameDrop(p=aug_cfg["random_frame_drop"]))

    # Normalization (always applied)
    transforms_list.append(VideoNormalize(mean=mean, std=std))

    return T.Compose(transforms_list)
