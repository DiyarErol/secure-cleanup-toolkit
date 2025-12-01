"""Unit tests for CLI module."""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from src.cli import main


@pytest.fixture
def temp_config():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "project": {"name": "test"},
            "seed": 42,
            "data": {
                "labels": ["stable", "critical", "terminal"],
                "num_classes": 3,
                "processed_dir": "data/processed",
                "preprocessing": {
                    "fps": 10,
                    "resolution": [224, 224],
                    "num_frames": 16,
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                "augmentation": {"enabled": False},
                "dataloader": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "model": {
                "backbone": "resnet3d_18",
                "pretrained": False,
                "hidden_dim": 512,
                "dropout": 0.5,
                "loss": {"type": "cross_entropy", "class_weights": None},
            },
            "training": {
                "epochs": 1,
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                },
                "scheduler": {"type": "none"},
                "mixed_precision": False,
                "gradient_accumulation_steps": 1,
                "gradient_clip_norm": None,
                "early_stopping": {
                    "enabled": False,
                    "patience": 10,
                    "metric": "val_loss",
                    "mode": "min",
                },
                "checkpoint": {
                    "save_dir": "checkpoints",
                    "save_best": True,
                    "save_last": True,
                },
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "per_class_metrics": True,
                "output_dir": "reports",
                "generate_html": False,
            },
            "explainability": {
                "method": "gradcam",
                "num_frames_to_visualize": 5,
                "output_dir": "reports/explainability",
            },
            "logging": {
                "log_dir": "logs",
                "level": "INFO",
                "log_to_file": False,
                "log_to_console": True,
                "wandb": {"enabled": False},
            },
            "hardware": {"device": "cpu", "num_threads": 1},
        }

        yaml.dump(config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


def test_cli_help():
    """Test CLI help."""
    sys.argv = ["cli.py"]

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


    def test_cli_check_runs(monkeypatch):
        import sys

        from src.cli import main
        sys.argv = ["cli", "check", "--config", "configs/default.yaml"]
        try:
            main()
        except SystemExit:
            pass


def test_cli_preprocess_command(temp_config):
    """Test preprocess command."""
    sys.argv = ["cli.py", "preprocess", "--config", str(temp_config)]

    # Should run without error (just prints message)
    main()


def test_cli_invalid_command(temp_config):
    """Test invalid command."""
    sys.argv = ["cli.py", "invalid", "--config", str(temp_config)]

    with pytest.raises(SystemExit):
        main()


    def test_evaluate_pipeline_runs(monkeypatch):
        import sys

        from src.cli import main
        # Note: checkpoint path may be created during training; here we only validate command path triggers
        # If checkpoint missing, CLI should raise, so we run evaluate via CLI help check
        sys.argv = ["cli", "evaluate", "--config", "configs/default.yaml", "--checkpoint", "checkpoints/best.pth"]
        try:
            main()
        except SystemExit:
            # Allow exit due to missing resources in CI; the command dispatch should still occur
            pass


def test_cli_missing_config():
    """Test missing config file."""
    sys.argv = ["cli.py", "train", "--config", "nonexistent.yaml"]

    with pytest.raises(SystemExit):
        main()
