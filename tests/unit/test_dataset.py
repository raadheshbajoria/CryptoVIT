"""
tests/unit/test_dataset.py
===========================
Unit tests for the CIFAR dataset module.

These tests do NOT download data from the internet.
They use synthetic/in-memory datasets to verify transform logic,
split sizes, and DataLoader construction without network calls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _make_minimal_cfg(name: str = "cifar10", image_size: int = 32) -> OmegaConf:
    """Construct a minimal config for testing without loading files."""
    num_classes = {"cifar10": 10, "cifar100": 100}[name]
    cfg = OmegaConf.create({
        "project": {"seed": 0},
        "paths": {"project_root": ".", "raw_dir": "data/raw"},
        "dataset": {
            "name": name,
            "image_size": image_size,
            "num_classes": num_classes,
            "train_split": 0.9,
            "num_workers": 0,
            "pin_memory": False,
            "augmentation": {
                "random_crop": False,
                "crop_padding": 4,
                "random_horizontal_flip": False,
                "color_jitter": False,
                "color_jitter_strength": 0.4,
                "random_erasing": False,
                "normalize": True,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010],
            },
        },
        "training": {"batch_size": 8},
        "evaluation": {"batch_size": 8},
        "hardware": {"device": "cpu"},
    })
    return cfg


# -------------------------------------------------------------------------
# Transform tests
# -------------------------------------------------------------------------

class TestBuildTransforms:
    def test_train_transform_output_shape(self):
        from cryptovit.datasets import build_train_transform
        from PIL import Image
        cfg = _make_minimal_cfg(image_size=224)
        transform = build_train_transform(cfg)
        # Create a fake 32x32 RGB image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224), f"Unexpected shape: {tensor.shape}"
        assert tensor.dtype == torch.float32

    def test_eval_transform_deterministic(self):
        from cryptovit.datasets import build_eval_transform
        from PIL import Image
        cfg = _make_minimal_cfg(image_size=224)
        transform = build_eval_transform(cfg)
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2), "Eval transform is not deterministic"

    def test_train_transform_normalises(self):
        """After normalisation, mean should be close to 0."""
        from cryptovit.datasets import build_train_transform
        from PIL import Image
        cfg = _make_minimal_cfg(image_size=224)
        transform = build_train_transform(cfg)
        # White image (all 255)
        img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        tensor = transform(img)
        # After normalisation with CIFAR stats, max should be well above 1
        # (unnormalised max = 1.0, normalised > 1.0 after dividing by small std)
        assert tensor.max() > 1.0, "Normalisation not applied?"

    def test_no_normalise_flag(self):
        from cryptovit.datasets import build_eval_transform
        from PIL import Image
        cfg = _make_minimal_cfg(image_size=64)
        OmegaConf.update(cfg, "dataset.augmentation.normalize", False)
        transform = build_eval_transform(cfg)
        arr = np.full((32, 32, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        tensor = transform(img)
        # Without normalisation, max pixel value should be 1.0 (ToTensor divides by 255)
        assert float(tensor.max()) == pytest.approx(1.0, abs=1e-5)


# -------------------------------------------------------------------------
# Class names
# -------------------------------------------------------------------------

class TestGetClassNames:
    def test_cifar10_count(self):
        from cryptovit.datasets import get_class_names
        names = get_class_names("cifar10")
        assert len(names) == 10

    def test_cifar100_count(self):
        from cryptovit.datasets import get_class_names
        names = get_class_names("cifar100")
        assert len(names) == 100

    def test_cifar10_names_are_strings(self):
        from cryptovit.datasets import get_class_names
        for name in get_class_names("cifar10"):
            assert isinstance(name, str) and len(name) > 0

    def test_unknown_dataset_raises(self):
        from cryptovit.datasets import get_class_names
        with pytest.raises(ValueError):
            get_class_names("imagenet")


# -------------------------------------------------------------------------
# compute_dataset_stats
# -------------------------------------------------------------------------

class TestComputeDatasetStats:
    def test_known_stats(self):
        """For a tensor dataset of all 0.5, stats should be mean=0.5, std~0."""
        from cryptovit.datasets import compute_dataset_stats
        images = torch.full((100, 3, 32, 32), 0.5)
        labels = torch.zeros(100, dtype=torch.long)
        ds = TensorDataset(images, labels)
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        mean, std = compute_dataset_stats(loader, n_batches=20)
        assert len(mean) == 3
        for m in mean:
            assert abs(m - 0.5) < 1e-5, f"Expected mean~0.5, got {m}"
        for s in std:
            assert abs(s) < 1e-5, f"Expected std~0, got {s}"

    def test_returns_three_channels(self):
        from cryptovit.datasets import compute_dataset_stats
        images = torch.rand(50, 3, 16, 16)
        labels = torch.zeros(50, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=10)
        mean, std = compute_dataset_stats(loader)
        assert len(mean) == 3
        assert len(std) == 3
