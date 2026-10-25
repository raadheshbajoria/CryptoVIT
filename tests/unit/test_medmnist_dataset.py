from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from cryptovit.datasets import get_class_names
from cryptovit.datasets.medmnist_dataset import (
    MEDMNIST_DATASETS,
    MedMNISTAdapter,
    compute_channel_stats,
    is_medmnist,
)
from cryptovit.utils import load_config


class FakeMedMNIST(Dataset):
    def __init__(self, n=10, mode="RGB", num_classes=7, seed=0):
        rng = np.random.default_rng(seed)
        shape = (28, 28, 3) if mode == "RGB" else (28, 28)
        self.imgs = rng.integers(0, 256, (n, *shape), dtype=np.uint8)
        self.labels = (np.arange(n) % num_classes).reshape(-1, 1)
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return Image.fromarray(self.imgs[i]), self.labels[i]


def test_adapter_returns_rgb_pil_and_int_label():
    ds = MedMNISTAdapter(FakeMedMNIST(5))
    img, label = ds[3]
    assert isinstance(img, Image.Image) and img.mode == "RGB"
    assert img.size == (28, 28)
    assert isinstance(label, int) and label == 3
    assert ds.targets == [0, 1, 2, 3, 4]


def test_adapter_converts_grayscale_to_rgb():
    ds = MedMNISTAdapter(FakeMedMNIST(4, mode="L", num_classes=2))
    img, _ = ds[0]
    assert img.mode == "RGB"
    arr = np.asarray(img)
    assert np.array_equal(arr[..., 0], arr[..., 1])


def test_adapter_applies_transform():
    from torchvision import transforms

    ds = MedMNISTAdapter(FakeMedMNIST(3), transforms.ToTensor())
    x, y = ds[1]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 28, 28)
    assert y == 1


def test_is_medmnist_and_registry():
    assert is_medmnist("dermamnist")
    assert not is_medmnist("cifar10")
    assert MEDMNIST_DATASETS["dermamnist"][1] == 7


def test_channel_stats():
    imgs = [Image.fromarray(np.full((4, 4, 3), v, np.uint8)) for v in (0, 255)]
    stats = compute_channel_stats(imgs)
    assert stats["mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert stats["std"] == pytest.approx([0.5, 0.5, 0.5])


def test_class_names_come_from_official_metadata():
    names = get_class_names("dermamnist")
    assert len(names) == 7
    assert "melanoma" in names


def test_config_validation_accepts_medmnist_and_checks_class_count():
    cfg = load_config(
        overrides=["dataset.name=dermamnist", "dataset.num_classes=7", "model.num_classes=7"]
    )
    assert cfg.dataset.name == "dermamnist"
    with pytest.raises(ValueError):
        load_config(overrides=["dataset.name=dermamnist", "dataset.num_classes=10"])
    with pytest.raises(ValueError):
        load_config(overrides=["dataset.name=imagenet"])


def test_config_override_selects_yaml_file():
    cfg = load_config(overrides=["config=cifar100"])
    assert cfg.dataset.name == "cifar100"
    assert cfg.dataset.num_classes == 100
    cfg = load_config(overrides=["config=cifar100", "training.epochs=3"])
    assert cfg.training.epochs == 3
