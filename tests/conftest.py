"""
tests/conftest.py
=================
Shared pytest fixtures for CryptoViT tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the src package importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
