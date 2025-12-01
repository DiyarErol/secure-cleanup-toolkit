"""Logging utilities for structured logging."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str | None = None,
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Setup structured logging with file and console handlers.

    Args:
        log_dir: Directory to store log files
        log_file: Log filename (if None, auto-generated with timestamp)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file

    Returns:
        Configured logger instance
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("secure_cleanup_toolkit")
    logger.setLevel(numeric_level)
    logger.handlers.clear()  # Remove existing handlers

    # Format string
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = ensure_dir(log_dir)
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"train_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
    name: str = "secure_cleanup_toolkit", log_dir: str | Path | None = None, level: str = "INFO"
) -> logging.Logger:
    """
    Get logger instance by name with optional file logging.

    Args:
        name: Logger name (typically module name)
        log_dir: Optional directory for log files
        level: Logging level

    Returns:
        Logger instance
    """
    # Avoid duplicate calls - just setup and return
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    # Only call setup_logging if log_dir is provided, otherwise use default
    if log_dir is not None:
        setup_logging(log_dir=log_dir, level=level)
    else:
        setup_logging(level=level)
    return logging.getLogger(name)
