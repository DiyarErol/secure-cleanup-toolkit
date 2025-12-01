"""Visualization utilities for training progress and results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_curves(
    history_csv: str | Path,
    output_dir: str | Path,
    metrics: list[str] | None = None,
) -> None:
    """
    Plot training curves from history CSV.

    Args:
        history_csv: Path to training history CSV
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load history
    df = pd.read_csv(history_csv)

    if metrics is None:
        # Auto-detect metrics (exclude epoch column)
        metrics = [col for col in df.columns if col != "epoch"]

    # Set style
    sns.set_style("whitegrid")

    for metric in metrics:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot metric
        ax.plot(df["epoch"], df[metric], marker="o", linewidth=2, markersize=4)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Training Progress: {metric.replace('_', ' ').title()}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Save plot
        output_path = output_dir / f"{metric}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confusion_matrix_advanced(
    cm: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    normalize: bool = True,
    show_percentages: bool = True,
) -> None:
    """
    Plot enhanced confusion matrix with percentages.

    Args:
        cm: Confusion matrix
        labels: Class labels
        output_path: Path to save plot
        normalize: Whether to normalize the confusion matrix
        show_percentages: Whether to show percentages in cells
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
    )

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if show_percentages and normalize:
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
            else:
                text = f"{cm[i, j]}"

            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    counts: dict[str, int], output_path: str | Path, title: str = "Class Distribution"
) -> None:
    """
    Plot class distribution bar chart.

    Args:
        counts: Dictionary mapping class names to counts
        output_path: Path to save plot
        title: Plot title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, values, color=sns.color_palette("husl", len(labels)))

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value}\n({value/total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate_schedule(history_csv: str | Path, output_path: str | Path) -> None:
    """
    Plot learning rate schedule over epochs.

    Args:
        history_csv: Path to training history CSV
        output_path: Path to save plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    if "lr" not in df.columns:
        print("Warning: 'lr' column not found in history CSV")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["epoch"], df["lr"], marker="o", linewidth=2, markersize=4, color="red")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_training_summary_report(
    history_csv: str | Path,
    metrics_json: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Create comprehensive training summary report with multiple plots.

    Args:
        history_csv: Path to training history CSV
        metrics_json: Path to evaluation metrics JSON
        output_dir: Directory to save report
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(history_csv)

    with open(metrics_json) as f:
        metrics = json.load(f)

    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    if "train_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], label="Train", marker="o")
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="Val", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    if "train_acc" in df.columns:
        ax2.plot(df["epoch"], df["train_acc"], label="Train", marker="o")
    if "val_acc" in df.columns:
        ax2.plot(df["epoch"], df["val_acc"], label="Val", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Learning rate
    ax3 = fig.add_subplot(gs[1, 0])
    if "lr" in df.columns:
        ax3.plot(df["epoch"], df["lr"], marker="o", color="red")
        ax3.set_yscale("log")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)

    # 4. Per-class metrics
    ax4 = fig.add_subplot(gs[1, 1])
    if "per_class" in metrics:
        class_names = list(metrics["per_class"].keys())
        f1_scores = [metrics["per_class"][cls]["f1-score"] for cls in class_names]
        ax4.bar(class_names, f1_scores, color=sns.color_palette("husl", len(class_names)))
        ax4.set_ylabel("F1-Score")
        ax4.set_title("Per-Class F1 Scores")
        ax4.grid(axis="y", alpha=0.3)

    # 5. Confusion matrix
    ax5 = fig.add_subplot(gs[2, :])
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        labels = list(metrics["per_class"].keys())
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax5
        )
        ax5.set_xlabel("Predicted")
        ax5.set_ylabel("True")
        ax5.set_title("Confusion Matrix")

    fig.suptitle("Training Summary Report", fontsize=16, fontweight="bold")

    # Save report
    output_path = output_dir / "training_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
