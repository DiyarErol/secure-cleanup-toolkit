"""Test utilities and validation."""

from src.utils.validation import validate_config


def test_validate_config_complete():
    """Test config validation with complete config."""
    config = {
        "data": {
            "root": "/path/to/data",
            "labels": ["class1", "class2"],
            "num_classes": 2,
            "num_frames": 16,
            "input_size": 224,
        },
        "model": {
            "backbone": "resnet3d_18",
            "pretrained": True,
            "hidden_dim": 512,
            "dropout": 0.5,
        },
        "training": {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "loss": "cross_entropy",
        },
    }

    errors = validate_config(config)
    assert len(errors) == 0, f"Expected no errors, got: {errors}"


def test_validate_config_missing_keys():
    """Test config validation with missing required keys."""
    config = {
        "data": {
            "root": "/path/to/data",
        },
        "model": {
            "backbone": "resnet3d_18",
        },
    }

    errors = validate_config(config)
    assert len(errors) > 0, "Expected validation errors for incomplete config"
    assert any("training" in err.lower() for err in errors)


def test_validate_config_missing_data_keys():
    """Test config validation with missing data keys."""
    config = {
        "data": {},
        "model": {
            "backbone": "resnet3d_18",
            "pretrained": True,
            "hidden_dim": 512,
            "dropout": 0.5,
        },
        "training": {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "loss": "cross_entropy",
        },
    }

    errors = validate_config(config)
    assert len(errors) > 0
    assert any("labels" in err.lower() for err in errors)
