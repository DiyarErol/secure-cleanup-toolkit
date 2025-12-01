"""Model export utilities for deployment."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_torchscript(
    model: nn.Module,
    input_shape: tuple[int, ...],
    output_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Export model to TorchScript format.

    Args:
        model: Model to export
        input_shape: Input tensor shape for tracing
        output_path: Path to save TorchScript model
        device: Device to use for export
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Create example input
    example_input = torch.randn(*input_shape, device=device)

    # Trace model
    logger.info(f"Tracing model with input shape {input_shape}")
    traced_model = torch.jit.trace(model, example_input)

    # Save
    torch.jit.save(traced_model, str(output_path))
    logger.info(f"TorchScript model saved to {output_path}")


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple[int, ...],
    output_path: str | Path,
    device: torch.device = torch.device("cpu"),
    opset_version: int = 14,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: Model to export
        input_shape: Input tensor shape for export
        output_path: Path to save ONNX model
        device: Device to use for export
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification for variable-size inputs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Create example input
    example_input = torch.randn(*input_shape, device=device)

    # Default dynamic axes (batch size)
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Export
    logger.info(f"Exporting model to ONNX with input shape {input_shape}")
    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"ONNX model saved to {output_path}")


def quantize_model_dynamic(
    model: nn.Module, output_path: str | Path, dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to model.

    Args:
        model: Model to quantize
        output_path: Path to save quantized model
        dtype: Quantization dtype

    Returns:
        Quantized model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    logger.info("Applying dynamic quantization...")

    # Quantize
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv3d}, dtype=dtype
    )

    # Save
    torch.save(quantized_model.state_dict(), output_path)

    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

    logger.info(
        f"Model size: {original_size / 1e6:.2f}MB → {quantized_size / 1e6:.2f}MB "
        f"({100 * (1 - quantized_size / original_size):.1f}% reduction)"
    )

    return quantized_model


def create_deployment_package(
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    include_onnx: bool = True,
    include_torchscript: bool = True,
) -> None:
    """
    Create complete deployment package with model and metadata.

    Args:
        model_path: Path to trained model checkpoint
        config: Configuration dictionary
        output_dir: Directory to save deployment package
        include_onnx: Whether to include ONNX export
        include_torchscript: Whether to include TorchScript export
    """
    import json
    import shutil

    from src.models.builder import build_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating deployment package...")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Save PyTorch model
    pytorch_dir = output_dir / "pytorch"
    pytorch_dir.mkdir(exist_ok=True)
    shutil.copy(model_path, pytorch_dir / "model.pt")

    # Export formats
    input_shape = (
        1,
        config["data"]["num_frames"],
        3,
        config["data"]["input_size"],
        config["data"]["input_size"],
    )

    if include_torchscript:
        export_to_torchscript(model, input_shape, output_dir / "torchscript" / "model.pt", device)

    if include_onnx:
        export_to_onnx(model, input_shape, output_dir / "onnx" / "model.onnx", device)

    # Save metadata
    metadata = {
        "model_config": config["model"],
        "data_config": config["data"],
        "input_shape": input_shape,
        "labels": config["data"]["labels"],
        "preprocessing": {
            "mean": config["data"].get("mean", [0.485, 0.456, 0.406]),
            "std": config["data"].get("std", [0.229, 0.224, 0.225]),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme_content = f"""# Deployment Package

## Model Information
- Backbone: {config['model']['backbone']}
- Input Shape: {input_shape}
- Classes: {config['data']['labels']}

## Contents
- `pytorch/`: PyTorch checkpoint
- `torchscript/`: TorchScript model (if enabled)
- `onnx/`: ONNX model (if enabled)
- `metadata.json`: Model metadata and configuration

## Usage Example

```python
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('torchscript/model.pt', map_location=device)
model.eval()

# Inference
input_tensor = torch.randn{input_shape}.to(device)
with torch.no_grad():
    output = model(input_tensor)
    predictions = torch.softmax(output, dim=1)
```
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    logger.info(f"Deployment package created at {output_dir}")
