"""
tests/conftest.py
=================
Shared pytest fixtures and configuration for CryptoViT tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Tests never need a GUI; the interactive Tk backend is flaky in test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the src package importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that download data",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (skipped by default unless --run-integration is passed)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def base_cfg():
    """Return the base OmegaConf configuration (CIFAR-10, no augmentation)."""
    from cryptovit.utils import load_config
    return load_config()


@pytest.fixture(scope="session")
def cifar100_cfg():
    """Return merged CIFAR-100 configuration."""
    from cryptovit.utils import load_config
    return load_config(extra_files=["cifar100.yaml"])

