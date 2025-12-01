"""I/O utilities for safe file operations."""

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Path to save YAML file
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary containing JSON contents

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Path to save JSON file
        indent: Indentation spaces for pretty printing
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def safe_read(path: str | Path, mode: str = "r") -> str | bytes:
    """
    Safely read file with error handling.

    Args:
        path: Path to file
        mode: Read mode ('r' for text, 'rb' for binary)

    Returns:
        File contents as string or bytes

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, mode, encoding="utf-8" if "b" not in mode else None) as f:
            return f.read()
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading file: {path}") from e


def safe_write(content: str | bytes, path: str | Path, mode: str = "w") -> None:
    """
    Safely write content to file with error handling.

    Args:
        content: Content to write
        path: Path to file
        mode: Write mode ('w' for text, 'wb' for binary)

    Raises:
        PermissionError: If file cannot be written
    """
    path = Path(path)
    ensure_dir(path.parent)

    try:
        with open(path, mode, encoding="utf-8" if "b" not in mode else None) as f:
            f.write(content)
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing file: {path}") from e
