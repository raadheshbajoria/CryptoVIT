"""End-to-end smoke test of the medical configuration (config=dermamnist):
class-weighted loss, balanced-accuracy checkpointing, official-split encrypted
loaders (separate validation set) and the medical metrics, for variants A-D,
on tiny fake 7-class data (no download)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cryptovit.datasets import build_encrypted_loaders
from cryptovit.datasets.cifar_dataset import build_eval_transform, build_train_transform
from cryptovit.datasets.medmnist_dataset import MedMNISTAdapter
from cryptovit.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


class FakeMedMNIST(Dataset):
    """Imbalanced 7-class fake data shaped like DermaMNIST (28x28 RGB)."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        probs = np.array([0.05, 0.08, 0.1, 0.02, 0.1, 0.6, 0.05])
        self.labels = rng.choice(7, size=(n, 1), p=probs)
        self.imgs = rng.integers(0, 256, (n, 28, 28, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return Image.fromarray(self.imgs[i]), self.labels[i]


def _load_run_ablation():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_medical", ROOT / "scripts" / "run_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=[
        "config=dermamnist",
        "dataset.image_size=32", "dataset.num_workers=0", "dataset.pin_memory=false",
        "model.image_size=32", "model.patch_size=8",
        "model.embed_dim=16", "model.depth=1", "model.num_heads=2",
        "model.dropout=0.0", "model.drop_path_rate=0.0",
        "training.epochs=1", "training.batch_size=8", "training.warmup_epochs=0",
        "training.mixed_precision=false", "training.save_every_n_epochs=100",
        "evaluation.batch_size=8", "encryption.transient_steps=50",
        "hardware.device=cpu",
    ])


def test_dermamnist_config_values(cfg):
    assert cfg.dataset.name == "dermamnist"
    assert cfg.dataset.num_classes == 7 and cfg.model.num_classes == 7
    assert cfg.training.class_weights == "balanced"
    assert cfg.training.select_metric == "balanced_accuracy"
    assert cfg.dataset.augmentation.random_crop is False
    assert cfg.dataset.augmentation.random_horizontal_flip is False


def _plaintext_loaders(cfg):
    train = MedMNISTAdapter(FakeMedMNIST(64, 1), build_train_transform(cfg))
    val = MedMNISTAdapter(FakeMedMNIST(24, 2), build_eval_transform(cfg))
    test = MedMNISTAdapter(FakeMedMNIST(24, 3), build_eval_transform(cfg))
    return (
        DataLoader(train, batch_size=8, shuffle=True, drop_last=True),
        DataLoader(val, batch_size=8),
        DataLoader(test, batch_size=8),
    )


def _encrypted_loaders(cfg):
    return build_encrypted_loaders(
        cfg,
        MedMNISTAdapter(FakeMedMNIST(64, 1)),
        MedMNISTAdapter(FakeMedMNIST(24, 3)),
        "smoke-key", seed=cfg.project.seed,
        base_val=MedMNISTAdapter(FakeMedMNIST(24, 2)),
    )


def test_encrypted_loaders_use_the_official_splits(cfg):
    train_l, val_l, test_l = _encrypted_loaders(cfg)
    assert len(train_l.dataset) == 64          # whole train split, no re-split
    assert len(val_l.dataset) == 24
    assert len(test_l.dataset) == 24
    assert val_l.dataset is not test_l.dataset


@pytest.mark.parametrize("variant", ["A", "B", "C", "D"])
def test_medical_variant_end_to_end(variant, cfg, tmp_path):
    ablation = _load_run_ablation()
    loaders = _plaintext_loaders(cfg) if variant == "A" else _encrypted_loaders(cfg)

    res = ablation.run_variant(variant, cfg, torch.device("cpu"), tmp_path, loaders=loaders)

    assert res["select_metric"] == "balanced_accuracy"
    assert 0.0 <= res["balanced_accuracy"] <= 1.0
    assert len(res["per_class_sensitivity"]) == 7
    assert len(res["per_class_specificity"]) == 7
    assert len(res["per_class_auc"]) == 7
    assert res["epochs_trained"] == 1
    assert res["class_names"] == [] or len(res["class_names"]) == 7
