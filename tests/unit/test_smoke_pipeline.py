"""End-to-end smoke test: every ablation variant trains for one epoch and is
evaluated, on tiny synthetic data (no download, runs in seconds on CPU).

Covers: encrypted/plaintext loaders on a shared split, forward + backward
pass, checkpoint saving and loading, and test-set evaluation for variants
A (plain ViT, plaintext), B (plain ViT, encrypted), C (CryptoPatchEmbedding)
and D (CryptoViT with key embedding).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from cryptovit.datasets import build_encrypted_loaders, get_split_indices
from cryptovit.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def _load_run_ablation():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_smoke", ROOT / "scripts" / "run_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCifar(Dataset):
    """Synthetic CIFAR-like data: class = dominant colour channel."""

    def __init__(self, n, seed, transform=None):
        rng = np.random.default_rng(seed)
        self.labels = [int(i % 3) for i in range(n)]
        self.images = []
        for label in self.labels:
            img = rng.integers(0, 80, (32, 32, 3), dtype=np.uint8)
            img[..., label] = rng.integers(150, 256, (32, 32), dtype=np.uint8)
            self.images.append(img)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=[
        "dataset.name=cifar10", "dataset.num_classes=10",
        "dataset.image_size=32", "dataset.num_workers=0",
        "dataset.pin_memory=false", "dataset.train_split=0.75",
        "model.image_size=32", "model.patch_size=8", "model.num_classes=10",
        "model.embed_dim=16", "model.depth=1", "model.num_heads=2",
        "model.dropout=0.0", "model.drop_path_rate=0.0",
        "training.epochs=1", "training.batch_size=8",
        "training.warmup_epochs=0", "training.mixed_precision=false",
        "training.save_every_n_epochs=100", "evaluation.batch_size=8",
        "encryption.transient_steps=50", "hardware.device=cpu",
    ])


def _plaintext_loaders(cfg, seed):
    tf = transforms.Compose([transforms.ToTensor()])
    train_full = FakeCifar(48, seed=1, transform=tf)
    test = FakeCifar(16, seed=2, transform=tf)
    train_idx, val_idx = get_split_indices(len(train_full), cfg.dataset.train_split, seed)
    return (
        DataLoader(Subset(train_full, train_idx), batch_size=8, shuffle=True, drop_last=True),
        DataLoader(Subset(train_full, val_idx), batch_size=8),
        DataLoader(test, batch_size=8),
    )


def _encrypted_loaders(cfg, seed):
    return build_encrypted_loaders(
        cfg, FakeCifar(48, seed=1), FakeCifar(16, seed=2),
        "smoke-test-key", seed=seed,
    )


@pytest.mark.parametrize("variant", ["A", "B", "C", "D"])
def test_variant_trains_and_evaluates(variant, cfg, tmp_path):
    ablation = _load_run_ablation()
    seed = cfg.project.seed
    loaders = (
        _plaintext_loaders(cfg, seed) if variant == "A" else _encrypted_loaders(cfg, seed)
    )

    results = ablation.run_variant(
        variant, cfg, torch.device("cpu"), tmp_path, loaders=loaders
    )

    assert results["variant"] == variant
    assert 0.0 <= results["accuracy"] <= 1.0
    assert results["num_samples"] == 16
    assert len(results["per_class_f1"]) == cfg.dataset.num_classes
    assert 0.0 <= results["best_val_acc"] <= 1.0

    ckpts = list((tmp_path / f"variant_{variant}" / "checkpoints").glob("*.pt"))
    assert ckpts, "a best checkpoint must have been saved"
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert "model_state_dict" in state and "val_acc" in state


def test_unknown_variant_is_rejected(cfg):
    ablation = _load_run_ablation()
    with pytest.raises(ValueError):
        ablation._build_variant("E", cfg, torch.device("cpu"))
