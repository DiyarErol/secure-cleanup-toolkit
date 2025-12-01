"""Evaluation logic with comprehensive metrics."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    labels: list[str],
) -> dict[str, Any]:
    """
    Evaluate model and compute metrics.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on
        labels: List of class labels

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for videos, batch_labels in tqdm(dataloader, desc="Evaluating"):
        videos = videos.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    preds_array = np.array(all_preds)
    labels_array = np.array(all_labels)
    probs_array = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(labels_array, preds_array)

    precision_recall_fscore_support(labels_array, preds_array, average=None)
    # Per-class and macro metrics
    report = classification_report(
        labels_array, preds_array, target_names=labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels_array, preds_array)

    metrics = {
        "accuracy": accuracy,
        "per_class": {},
        "macro_avg": report["macro avg"],  # type: ignore[index]
        "weighted_avg": report["weighted avg"],  # type: ignore[index]
        "confusion_matrix": cm.tolist(),
        "predictions": preds_array.tolist(),
        "labels": labels_array.tolist(),
        "probabilities": probs_array.tolist(),
    }

    # Per-class metrics
    for _idx, label in enumerate(labels):
        if label in report:
            metrics["per_class"][label] = report[label]  # type: ignore[index]

    return metrics


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: Path) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        labels: Class labels
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                # Save classification report CSV
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix: {save_path}")


def plot_pr_curves(
    labels_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """
    Plot precision-recall curves for each class.

    Args:
        labels_true: True labels
        probs: Predicted probabilities
        class_names: Class names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 5))

    if len(class_names) == 1:
        axes = [axes]

    for idx, class_name in enumerate(class_names):
        # Binary labels for this class
        binary_labels = (labels_true == idx).astype(int)
        class_probs = probs[:, idx]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(binary_labels, class_probs)

        # Plot
        axes[idx].plot(recall, precision, linewidth=2)
        axes[idx].set_xlabel("Recall")
        axes[idx].set_ylabel("Precision")
        axes[idx].set_title(f"PR Curve: {class_name}")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved PR curves: {save_path}")


def generate_html_report(metrics: dict[str, Any], labels: list[str], save_path: Path) -> None:
    """Generate HTML evaluation report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            h1, h2 {{ color: #333; }}
            .metric {{ font-size: 18px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>

        <h2>Overall Metrics</h2>
        <div class="metric"><strong>Accuracy:</strong> {metrics['accuracy']:.4f}</div>
        <div class="metric"><strong>Macro Avg Precision:</strong> {metrics['macro_avg']['precision']:.4f}</div>
        <div class="metric"><strong>Macro Avg Recall:</strong> {metrics['macro_avg']['recall']:.4f}</div>
        <div class="metric"><strong>Macro Avg F1-Score:</strong> {metrics['macro_avg']['f1-score']:.4f}</div>

        <h2>Per-Class Metrics</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
    """

    for label in labels:
        if label in metrics["per_class"]:
            pc = metrics["per_class"][label]
            html += f"""
            <tr>
                <td>{label}</td>
                <td>{pc['precision']:.4f}</td>
                <td>{pc['recall']:.4f}</td>
                <td>{pc['f1-score']:.4f}</td>
                <td>{int(pc['support'])}</td>
            </tr>
            """

    html += """
        </table>

        <h2>Visualizations</h2>
        <p>See confusion_matrix.png and pr_curves.png in the reports directory.</p>
    </body>
    </html>
    """

    save_path.write_text(html, encoding="utf-8")
    logger.info(f"Saved HTML report: {save_path}")


def evaluate(config: dict[str, Any], checkpoint_path: str) -> None:
    """
    Evaluate model from checkpoint.

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

    logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    # Build test dataloader
    transforms = {"test": get_transforms(config, is_train=False)}
    dataloaders = build_dataloaders(config, transforms)

    if "test" not in dataloaders:
        raise ValueError("Test split not found")

    # Evaluate
    labels = config["data"]["labels"]
    metrics = evaluate_model(model, dataloaders["test"], device, labels)

    # Save metrics
    output_dir = Path(config["evaluation"]["output_dir"])
    ensure_dir(output_dir)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        # Remove large arrays for JSON
        metrics_json = {
            k: v for k, v in metrics.items() if k not in ["predictions", "labels", "probabilities"]
        }
        json.dump(metrics_json, f, indent=2)

    logger.info(f"Saved metrics: {metrics_path}")

    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")

    # Plot PR curves
    probs = np.array(metrics["probabilities"])
    labels_true = np.array(metrics["labels"])
    plot_pr_curves(labels_true, probs, labels, output_dir / "pr_curves.png")

    # Generate HTML report
    if config["evaluation"]["generate_html"]:
        generate_html_report(metrics, labels, output_dir / "evaluation_report.html")

    # Log summary
    logger.info("Evaluation Summary:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Macro Avg F1: {metrics['macro_avg']['f1-score']:.4f}")
    for label in labels:
        if label in metrics["per_class"]:
            f1 = metrics["per_class"][label]["f1-score"]
            logger.info(f"  {label} F1: {f1:.4f}")
