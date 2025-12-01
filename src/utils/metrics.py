"""Advanced metrics and performance monitoring utilities."""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class PerformanceMonitor:
    """Monitor model performance metrics during training and inference."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.start_times: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """
        Start a timer.

        Args:
            name: Timer name
        """
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """
        End a timer and record elapsed time.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.time() - self.start_times[name]
        self.metrics[name].append(elapsed)
        del self.start_times[name]
        return elapsed

    def record(self, name: str, value: float) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)

    def get_stats(self, name: str) -> dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with mean, std, min, max
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        values = self.metrics[name]
        return {
            "mean": sum(values) / len(values),
            "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all metrics.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {name: self.get_stats(name) for name in self.metrics.keys()}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


def compute_model_flops(
    model: nn.Module, input_shape: tuple[int, ...], device: torch.device
) -> dict[str, Any]:
    """
    Estimate model FLOPs (floating point operations).

    Args:
        model: Model to analyze
        input_shape: Input tensor shape (batch_size, channels, frames, height, width)
        device: Device to run on

    Returns:
        Dictionary with FLOPs estimates and model statistics
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Profile model
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(dummy_input)

    # Extract FLOPs
    events = prof.events()
    total_flops = sum(evt.flops for evt in events if evt.flops is not None) if events else 0

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_flops": total_flops,
        "total_flops_human": f"{total_flops / 1e9:.2f}G",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / (1024**2),  # Assuming float32
    }


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict[str, float]:
    """
    Measure model inference time.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "std_ms": (sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)) ** 0.5
        * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "fps": 1.0 / (sum(times) / len(times)),
    }


def export_metrics_to_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    """
    Export metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Output file path
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
