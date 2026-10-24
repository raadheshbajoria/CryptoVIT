"""Tests for the encrypted data loaders: shared split, no test leakage,
deterministic transform and encrypted-image normalisation statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

from cryptovit.datasets import build_encrypted_loaders, get_split_indices
from cryptovit.datasets.encrypted_dataset import (
    build_encrypted_transform,
    compute_encrypted_stats,
)
from cryptovit.utils import load_config


class FakeCifar(Dataset):
    """Smooth gradient images (strongly non-uniform pixel statistics)."""

    def __init__(self, n, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:32, 0:32]
        self.images = []
        for _ in range(n):
            base = (xx * 3 + yy * 2 + rng.integers(0, 40)) % 128
            self.images.append(np.stack([base, base // 2, base // 3], -1).astype(np.uint8))
        self.labels = [int(i % 10) for i in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return Image.fromarray(self.images[i]), self.labels[i]


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=[
        "dataset.image_size=32", "dataset.num_workers=0",
        "dataset.pin_memory=false", "encryption.transient_steps=50",
        "training.batch_size=8", "evaluation.batch_size=8",
        "dataset.train_split=0.8", "model.image_size=32", "model.patch_size=8",
    ])


@pytest.fixture(scope="module")
def loaders(cfg):
    train = FakeCifar(50, seed=1)
    test = FakeCifar(20, seed=2)
    return build_encrypted_loaders(cfg, train, test, "unit-test-key", seed=7), train, test


def test_split_indices_match_original_random_split():
    expected_train, expected_val = random_split(
        range(100), [80, 20], generator=torch.Generator().manual_seed(42)
    )
    train_idx, val_idx = get_split_indices(100, 0.8, 42)
    assert train_idx == list(expected_train)
    assert val_idx == list(expected_val)


def test_train_and_val_partition_the_training_set(loaders):
    (train_l, val_l, _), train_base, _ = loaders
    tr, va = set(train_l.dataset.indices), set(val_l.dataset.indices)
    assert tr.isdisjoint(va)
    assert tr | va == set(range(len(train_base)))


def test_encrypted_split_equals_plaintext_split(loaders, cfg):
    (train_l, val_l, _), train_base, _ = loaders
    train_idx, val_idx = get_split_indices(len(train_base), cfg.dataset.train_split, 7)
    assert train_l.dataset.indices == train_idx
    assert val_l.dataset.indices == val_idx


def test_validation_loader_is_not_the_test_loader(loaders):
    (_, val_l, test_l), _, test_base = loaders
    assert val_l is not test_l
    assert len(test_l.dataset) == len(test_base)
    assert len(val_l.dataset) != len(test_l.dataset)


def test_batches_have_expected_shape_and_dtype(loaders):
    (train_l, val_l, test_l), _, _ = loaders
    for loader in (train_l, val_l, test_l):
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64


def test_eval_transform_is_deterministic(loaders):
    (_, val_l, _), _, _ = loaders
    ds = val_l.dataset.dataset
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_encrypted_stats_are_near_uniform_noise(cfg):
    base = FakeCifar(40, seed=3)
    stats = compute_encrypted_stats(base, "k", range(40), transient_steps=50)
    # Plaintext here has mean well below 0.5 and small spread; ciphertext
    # should be close to uniform: mean ~0.5, std ~0.289.
    assert all(0.44 < m < 0.56 for m in stats["mean"])
    assert all(0.25 < s < 0.33 for s in stats["std"])


def test_encrypted_transform_has_no_random_ops():
    t = build_encrypted_transform(32, [0.5] * 3, [0.29] * 3)
    names = [type(op).__name__ for op in t.transforms]
    assert names == ["Resize", "ToTensor", "Normalize"]
