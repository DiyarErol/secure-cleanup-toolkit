"""Seed utilities for reproducible experiments."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None = None) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This function seeds Python's random module, NumPy, and PyTorch (CPU and CUDA).
    For full determinism on CUDA, additional flags are set, which may reduce performance.

    Args:
        seed: Random seed value. If None, no seeding is performed.

    Note:
        Full determinism on GPU requires additional settings and may reduce performance:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False

        For more information, see:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        return

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variables for CuDNN determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Full determinism on CUDA (may reduce performance)
    # Uncomment for strict reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # For DataLoader workers
    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker with unique seed."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Store worker_init_fn for DataLoader use
    torch.utils.data._utils.worker.worker_init_fn = worker_init_fn  # type: ignore
