"""Test metrics utilities."""

import pytest

from src.utils.metrics import PerformanceMonitor


def test_performance_monitor_timer():
    """Test performance monitor timer functionality."""
    monitor = PerformanceMonitor()

    monitor.start_timer("test")
    import time

    time.sleep(0.1)
    elapsed = monitor.end_timer("test")

    assert elapsed >= 0.1, f"Expected at least 0.1s, got {elapsed}s"
    assert "test" in monitor.metrics
    assert len(monitor.metrics["test"]) == 1


def test_performance_monitor_record():
    """Test performance monitor record functionality."""
    monitor = PerformanceMonitor()

    monitor.record("accuracy", 0.95)
    monitor.record("accuracy", 0.96)
    monitor.record("accuracy", 0.97)

    stats = monitor.get_stats("accuracy")
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(0.96, rel=1e-2)
    assert stats["min"] == 0.95
    assert stats["max"] == 0.97


def test_performance_monitor_get_all_stats():
    """Test getting all statistics."""
    monitor = PerformanceMonitor()

    monitor.record("loss", 0.5)
    monitor.record("accuracy", 0.9)

    all_stats = monitor.get_all_stats()

    assert "loss" in all_stats
    assert "accuracy" in all_stats
    assert all_stats["loss"]["count"] == 1
    assert all_stats["accuracy"]["count"] == 1


def test_performance_monitor_reset():
    """Test resetting monitor."""
    monitor = PerformanceMonitor()

    monitor.record("test", 1.0)
    monitor.reset()

    assert len(monitor.metrics) == 0
    assert len(monitor.start_times) == 0


def test_performance_monitor_timer_error():
    """Test timer error handling."""
    monitor = PerformanceMonitor()

    with pytest.raises(ValueError, match="Timer 'nonexistent' was not started"):
        monitor.end_timer("nonexistent")
