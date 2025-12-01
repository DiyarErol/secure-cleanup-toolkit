"""Data validation utilities for ensuring data quality."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validator for video dataset quality checks."""

    def __init__(self, data_dir: str | Path, labels: list[str]):
        """
        Initialize data validator.

        Args:
            data_dir: Root directory containing split subdirectories
            labels: List of class labels
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.issues: list[dict[str, Any]] = []

    def validate_structure(self) -> bool:
        """
        Validate dataset directory structure.

        Returns:
            True if structure is valid, False otherwise
        """
        logger.info("Validating dataset structure...")
        valid = True

        # Check splits
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                self.issues.append(
                    {"type": "missing_split", "split": split, "path": str(split_dir)}
                )
                logger.warning(f"Missing split directory: {split_dir}")
                valid = False
                continue

            # Check labels
            for label in self.labels:
                label_dir = split_dir / label
                if not label_dir.exists():
                    self.issues.append(
                        {
                            "type": "missing_label",
                            "split": split,
                            "label": label,
                            "path": str(label_dir),
                        }
                    )
                    logger.warning(f"Missing label directory: {label_dir}")
                    valid = False

        return valid

    def validate_videos(self, split: str = "train") -> dict[str, Any]:
        """
        Validate video quality in a split.

        Args:
            split: Dataset split to validate

        Returns:
            Validation report dictionary
        """
        logger.info(f"Validating videos in {split} split...")

        report: dict[str, Any] = {
            "total_videos": 0,
            "valid_videos": 0,
            "corrupted_videos": [],
            "empty_videos": [],
            "missing_frames": [],
            "resolution_stats": {"widths": [], "heights": []},
            "frame_count_stats": [],
        }

        split_dir = self.data_dir / split

        for label in self.labels:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue

            video_dirs = [d for d in label_dir.iterdir() if d.is_dir()]

            for video_dir in tqdm(video_dirs, desc=f"Validating {label}"):
                report["total_videos"] += 1

                # Get frame files
                frame_files = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))

                if len(frame_files) == 0:
                    report["empty_videos"].append(str(video_dir))
                    self.issues.append(
                        {"type": "empty_video", "path": str(video_dir), "label": label}
                    )
                    continue

                # Check first frame
                first_frame_path = frame_files[0]
                frame = cv2.imread(str(first_frame_path))

                if frame is None:
                    report["corrupted_videos"].append(str(video_dir))
                    self.issues.append(
                        {
                            "type": "corrupted_video",
                            "path": str(video_dir),
                            "frame": str(first_frame_path),
                        }
                    )
                    continue

                # Record statistics
                h, w = frame.shape[:2]
                report["resolution_stats"]["heights"].append(h)
                report["resolution_stats"]["widths"].append(w)
                report["frame_count_stats"].append(len(frame_files))

                report["valid_videos"] += 1

        # Compute summary statistics
        if report["resolution_stats"]["heights"]:
            report["resolution_summary"] = {
                "mean_height": np.mean(report["resolution_stats"]["heights"]),
                "mean_width": np.mean(report["resolution_stats"]["widths"]),
                "min_height": np.min(report["resolution_stats"]["heights"]),
                "max_height": np.max(report["resolution_stats"]["heights"]),
                "min_width": np.min(report["resolution_stats"]["widths"]),
                "max_width": np.max(report["resolution_stats"]["widths"]),
            }

        if report["frame_count_stats"]:
            report["frame_count_summary"] = {
                "mean_frames": np.mean(report["frame_count_stats"]),
                "min_frames": np.min(report["frame_count_stats"]),
                "max_frames": np.max(report["frame_count_stats"]),
                "std_frames": np.std(report["frame_count_stats"]),
            }

        return report

    def check_class_balance(self) -> dict[str, dict[str, int]]:
        """
        Check class balance across splits.

        Returns:
            Dictionary mapping splits to class counts
        """
        logger.info("Checking class balance...")

        balance: dict[str, dict[str, int]] = {}

        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue

            counts: dict[str, int] = {}
            for label in self.labels:
                label_dir = split_dir / label
                if label_dir.exists():
                    video_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
                    counts[label] = len(video_dirs)
                else:
                    counts[label] = 0

            balance[split] = counts

            # Check for severe imbalance
            if counts:
                max_count = max(counts.values())
                min_count = min(counts.values())
                if min_count > 0 and max_count / min_count > 10:
                    self.issues.append(
                        {
                            "type": "class_imbalance",
                            "split": split,
                            "max_count": max_count,
                            "min_count": min_count,
                            "ratio": max_count / min_count,
                        }
                    )
                    logger.warning(
                        f"Severe class imbalance in {split}: "
                        f"ratio {max_count}/{min_count} = {max_count/min_count:.2f}"
                    )

        return balance

    def generate_report(self, output_path: str | Path) -> None:
        """
        Generate validation report.

        Args:
            output_path: Path to save report
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "issues": self.issues,
            "num_issues": len(self.issues),
            "issue_types": {
                issue_type: sum(1 for i in self.issues if i["type"] == issue_type)
                for issue_type in {i["type"] for i in self.issues}
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

    def run_full_validation(self, output_dir: str | Path) -> bool:
        """
        Run complete validation suite.

        Args:
            output_dir: Directory to save validation reports

        Returns:
            True if all validations pass, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running full validation suite...")

        # 1. Structure validation
        self.validate_structure()

        # 2. Video validation for each split
        video_reports = {}
        for split in ["train", "val", "test"]:
            if (self.data_dir / split).exists():
                video_reports[split] = self.validate_videos(split)

        # 3. Class balance check
        balance = self.check_class_balance()

        # 4. Generate reports
        self.generate_report(output_dir / "validation_issues.json")

        # Save video reports
        import json

        with open(output_dir / "video_validation.json", "w") as f:
            json.dump(video_reports, f, indent=2)

        with open(output_dir / "class_balance.json", "w") as f:
            json.dump(balance, f, indent=2)

        # Summary
        logger.info(f"Validation complete. Found {len(self.issues)} issues.")

        return len(self.issues) == 0


def _validate_data_config(data_cfg: dict[str, Any]) -> list[str]:
    """Validate data configuration section."""
    errors = []
    if "root" not in data_cfg and "processed_dir" not in data_cfg:
        errors.append("Data config missing 'root' or 'processed_dir'")
    if "labels" not in data_cfg:
        errors.append("Data config missing 'labels'")
    if "num_classes" not in data_cfg:
        errors.append("Data config missing 'num_classes'")
    return errors


def _validate_model_config(model_cfg: dict[str, Any]) -> list[str]:
    """Validate model configuration section."""
    errors = []
    required_keys = ["backbone", "pretrained", "hidden_dim", "dropout"]
    for key in required_keys:
        if key not in model_cfg:
            errors.append(f"Model config missing '{key}'")
    return errors


def _validate_training_config(train_cfg: dict[str, Any]) -> list[str]:
    """Validate training configuration section."""
    errors = []
    required_keys = ["batch_size", "epochs", "learning_rate", "optimizer", "loss"]
    for key in required_keys:
        if key not in train_cfg:
            errors.append(f"Training config missing '{key}'")
    return errors


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required top-level keys
    required_keys = ["data", "model", "training"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: '{key}'")

    # Validate each section
    if "data" in config:
        errors.extend(_validate_data_config(config["data"]))
    if "model" in config:
        errors.extend(_validate_model_config(config["model"]))
    if "training" in config:
        errors.extend(_validate_training_config(config["training"]))

    return errors
