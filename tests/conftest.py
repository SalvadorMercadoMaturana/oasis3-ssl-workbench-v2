"""Shared pytest fixtures for the T5-SSL test suite."""

from __future__ import annotations

import os

import pytest

# Ensure CPU-safe execution on systems with incompatible NVIDIA drivers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as 'slow' (full dataset download).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Wire up custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests requiring full dataset download",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.slow`` tests unless ``--run-slow`` is passed."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
