"""Explainability via Grad-CAM and saliency maps."""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Grad-CAM for visual explanations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: Model to explain
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """Save forward activations."""
        self.activations = output.detach()

    def save_gradient(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """Save backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input video tensor
            target_class: Target class index

        Returns:
            Grad-CAM heatmap
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients  # (B, C, T, H, W)
        activations = self.activations  # (B, C, T, H, W)

        # Global average pooling over spatial dimensions
        weights = torch.mean(gradients, dim=[3, 4], keepdim=True)  # type: ignore[arg-type,call-overload]

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def overlay_heatmap(
    frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap on frame.

    Args:
        frame: Original frame (H, W, 3)
        heatmap: Heatmap (H, W)
        alpha: Blending factor

    Returns:
        Overlaid image
    """
    # Resize heatmap to frame size
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)  # type: ignore[assignment]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlaid = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    return overlaid


def explain_video(
    model: nn.Module,
    video_tensor: torch.Tensor,
    true_label: int,
    pred_label: int,
    frames_to_visualize: list[int],
    save_dir: Path,
    sample_id: int,
    class_names: list[str],
) -> None:
    """
    Generate Grad-CAM explanations for video.

    Args:
        model: Model to explain
        video_tensor: Video tensor (1, T, C, H, W)
        true_label: True class label
        pred_label: Predicted class label
        frames_to_visualize: Indices of frames to visualize
        save_dir: Directory to save visualizations
        sample_id: Sample identifier
        class_names: List of class names
    """
    # Find target layer (last conv layer in backbone)
    target_layer = None
    for _name, module in model.backbone.named_modules():  # type: ignore[attr-defined]
        if isinstance(module, nn.Conv3d):
            target_layer = module

    if target_layer is None:
        logger.warning("No Conv3d layer found, skipping Grad-CAM")
        return

    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap
    cam = grad_cam.generate_cam(video_tensor, pred_label)  # (T, H, W)

    # Visualize selected frames
    fig, axes = plt.subplots(2, len(frames_to_visualize), figsize=(4 * len(frames_to_visualize), 8))

    video_np = video_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    video_np = (video_np * 255).astype(np.uint8)

    for i, frame_idx in enumerate(frames_to_visualize):
        if frame_idx >= video_np.shape[0]:
            continue

        frame = video_np[frame_idx]
        heatmap = cam[frame_idx]

        # Original frame
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame {frame_idx}")
        axes[0, i].axis("off")

        # Overlaid frame
        overlaid = overlay_heatmap(frame, heatmap)
        axes[1, i].imshow(overlaid)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    fig.suptitle(
        f"Sample {sample_id} | True: {class_names[true_label]} | "
        f"Pred: {class_names[pred_label]}",
        fontsize=14,
    )
    fig.tight_layout()

    save_path = save_dir / f"sample_{sample_id}_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain(config: dict[str, Any], checkpoint_path: str) -> None:
    """
    Generate explainability visualizations.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    from src.data.dataset import build_dataloaders
    from src.data.transforms import get_transforms
    from src.models.builder import build_model

    # Setup device
    if config["hardware"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["hardware"]["device"])

    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    # Build test dataloader
    transforms = {"test": get_transforms(config, is_train=False)}
    dataloaders = build_dataloaders(config, transforms)

    if "test" not in dataloaders:
        raise ValueError("Test split not found")

    # Setup output directory
    output_dir = Path(config["explainability"]["output_dir"])
    ensure_dir(output_dir)

    # Get config
    num_frames_to_viz = config["explainability"]["num_frames_to_visualize"]
    labels = config["data"]["labels"]
    num_frames = config["data"]["preprocessing"]["num_frames"]

    # Generate explanations for a few samples
    max_samples = 10
    sample_count = 0

    with torch.no_grad():
        for videos, batch_labels in tqdm(dataloaders["test"], desc="Generating explanations"):
            if sample_count >= max_samples:
                break

            videos = videos.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(videos)
            _, preds = torch.max(outputs, 1)

            # Process each video in batch
            for i in range(videos.size(0)):
                if sample_count >= max_samples:
                    break

                video = videos[i: i + 1]  # (1, T, C, H, W)
                true_label = batch_labels[i].item()
                pred_label = preds[i].item()

                # Select frames to visualize (evenly spaced)
                frame_indices = np.linspace(0, num_frames - 1, num_frames_to_viz, dtype=int).tolist()

                # Generate explanation
                explain_video(
                    model=model,
                    video_tensor=video,
                    true_label=true_label,
                    pred_label=int(pred_label),
                    frames_to_visualize=frame_indices,
                    save_dir=output_dir,
                    sample_id=sample_count,
                    class_names=labels,
                )

                sample_count += 1

    logger.info(f"Generated explanations for {sample_count} samples in {output_dir}")
