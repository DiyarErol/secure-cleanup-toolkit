"""Video model backbones (3D-ResNet, SlowFast, TimeSformer stubs)."""


import torch
import torch.nn as nn
import torchvision.models.video as video_models

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_backbone(
    name: str, pretrained: bool = True, num_classes: int = 3
) -> tuple[nn.Module, int]:
    """
    Get video backbone model.

    Args:
        name: Backbone name ('resnet3d_18', 'resnet3d_34', 'slowfast', 'timesformer')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes

    Returns:
        Tuple of (backbone model, feature dimension)

    Raises:
        ValueError: If backbone name is not recognized
    """
    name = name.lower()

    if name == "resnet3d_18" or name == "r3d_18":
        model = video_models.r3d_18(
            weights=video_models.R3D_18_Weights.DEFAULT if pretrained else None
        )
        # Replace final FC layer
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore[assignment]
        logger.info(f"Loaded 3D ResNet-18 (feature_dim={feature_dim})")
        return model, feature_dim

    elif name == "resnet3d_34":
        # Note: torchvision doesn't have r3d_34, using r3d_18 as placeholder
        logger.warning("3D ResNet-34 not available, using 3D ResNet-18")
        return get_backbone("resnet3d_18", pretrained, num_classes)

    elif name == "slowfast":
        # SlowFast stub - in production, use PyTorchVideo or similar
        logger.warning("SlowFast not implemented, using 3D ResNet-18 as fallback")
        return get_backbone("resnet3d_18", pretrained, num_classes)

    elif name == "timesformer":
        # TimeSformer stub - in production, use timm or similar
        logger.warning("TimeSformer not implemented, using 3D ResNet-18 as fallback")
        return get_backbone("resnet3d_18", pretrained, num_classes)

    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Supported: resnet3d_18, resnet3d_34, slowfast, timesformer"
        )


class VideoClassifier(nn.Module):
    """Video classifier with configurable backbone and head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        """
        Initialize video classifier.

        Args:
            backbone: Backbone model
            feature_dim: Feature dimension from backbone
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout probability
        """
        super().__init__()
        self.backbone = backbone

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input video tensor of shape (B, T, C, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        # Rearrange to (B, C, T, H, W) for 3D convolution
        x = x.permute(0, 2, 1, 3, 4)

        # Extract features
        features = self.backbone(x)

        # Classification
        logits = self.head(features)

        return logits
