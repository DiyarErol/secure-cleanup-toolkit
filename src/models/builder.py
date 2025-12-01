"""Model builder and loss functions."""

from typing import Any

import torch
import torch.nn as nn

from src.models.backbones import VideoClassifier, get_backbone
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_model(config: dict[str, Any]) -> nn.Module:
    """
    Build video classification model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Video classification model
    """
    # Extract model config
    model_cfg = config["model"]
    backbone_name = model_cfg["backbone"]
    pretrained = model_cfg["pretrained"]
    num_classes = config["data"]["num_classes"]
    hidden_dim = model_cfg["hidden_dim"]
    dropout = model_cfg["dropout"]

    # Build backbone
    backbone, feature_dim = get_backbone(
        name=backbone_name, pretrained=pretrained, num_classes=num_classes
    )

    # Build classifier
    model = VideoClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Built model: {backbone_name} | "
        f"Total params: {total_params:,} | "
        f"Trainable: {trainable_params:,}"
    )

    return model


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter for hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (B, C)
            targets: Ground truth labels of shape (B,)

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1))
        targets_one_hot = targets_one_hot.float()

        # Compute focal loss
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_loss(config: dict[str, Any], class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Build loss function from config.

    Args:
        config: Configuration dictionary
        class_weights: Optional class weights for imbalanced data

    Returns:
        Loss function module
    """
    loss_cfg = config["model"]["loss"]
    loss_type = loss_cfg["type"]

    if loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using CrossEntropyLoss")

    elif loss_type == "focal_loss":
        alpha = loss_cfg.get("focal_alpha", 0.25)
        gamma = loss_cfg.get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion
