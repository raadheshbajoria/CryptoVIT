"""
tests/integration/test_phase1_pipeline.py
==========================================
Integration tests for Phase 1: dataset download and DataLoader pipeline.

These tests require an internet connection on first run to download CIFAR-10
(~170 MB).  They are marked with ``@pytest.mark.integration`` and are
skipped in the default ``pytest`` run unless ``--run-integration`` is passed.

To run:
    pytest tests/integration/ --run-integration -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with -m 'not integration')"
    )


@pytest.fixture(scope="module")
def cifar10_loaders(base_cfg):
    from cryptovit.datasets import get_dataloaders
    return get_dataloaders(base_cfg, seed=42)


@pytest.mark.integration
class TestCIFAR10Pipeline:
    def test_loader_lengths(self, cifar10_loaders, base_cfg):
        train, val, test = cifar10_loaders
        assert len(train.dataset) + len(val.dataset) == 50_000
        assert len(test.dataset) == 10_000

    def test_batch_shapes(self, cifar10_loaders, base_cfg):
        train, _, _ = cifar10_loaders
        images, labels = next(iter(train))
        assert images.shape == (
            base_cfg.training.batch_size, 3,
            base_cfg.dataset.image_size,
            base_cfg.dataset.image_size,
        )
        assert labels.shape == (base_cfg.training.batch_size,)

    def test_label_range(self, cifar10_loaders, base_cfg):
        _, _, test = cifar10_loaders
        for images, labels in test:
            assert labels.min() >= 0
            assert labels.max() < base_cfg.dataset.num_classes

    def test_pixel_value_range_after_normalisation(self, cifar10_loaders):
        train, _, _ = cifar10_loaders
        images, _ = next(iter(train))
        # After CIFAR normalisation values are not strictly in [0,1]
        # but should be roughly in [-3, 3]
        assert float(images.min()) > -5.0
        assert float(images.max()) < 5.0

    def test_train_val_no_overlap(self, cifar10_loaders):
        """Verify train and val indices do not overlap."""
        train, val, _ = cifar10_loaders
        from torch.utils.data import Subset
        if isinstance(train.dataset, Subset) and isinstance(val.dataset, Subset):
            train_idx = set(train.dataset.indices)
            val_idx = set(val.dataset.indices)
            assert len(train_idx & val_idx) == 0, "Train/val sets overlap!"