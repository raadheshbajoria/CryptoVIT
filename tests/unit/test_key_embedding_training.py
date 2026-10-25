"""Tests that the key embedding reaches the model in training and evaluation,
that per-class metrics keep a fixed length, and the trainer's optimiser and
schedule setup."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from cryptovit.evaluation import evaluate_model
from cryptovit.models.cryptovit import CryptoViT
from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding
from cryptovit.training import Trainer


def _cfg(epochs=1, accumulation_steps=1, warmup_epochs=0):
    return OmegaConf.create(
        {
            "training": {
                "epochs": epochs,
                "batch_size": 4,
                "accumulation_steps": accumulation_steps,
                "mixed_precision": False,
                "learning_rate": 1e-3,
                "weight_decay": 0.05,
                "betas": [0.9, 0.999],
                "warmup_epochs": warmup_epochs,
                "min_lr": 1e-6,
                "label_smoothing": 0.0,
                "clip_grad_norm": 1.0,
                "save_every_n_epochs": 100,
                "keep_best_k": 1,
                "early_stopping_patience": 100,
            },
            "logging": {"use_tensorboard": False},
        }
    )


def _tiny_cryptovit():
    return CryptoViT(
        image_size=16,
        patch_size=8,
        in_channels=3,
        num_classes=3,
        embed_dim=16,
        depth=1,
        num_heads=2,
        hidden_dim=4,
        dropout=0.0,
        drop_path_rate=0.0,
        key_embed_dim=32,
    )


def _loader(n=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _key_mlp_state(model):
    return [p.detach().clone() for p in model.pos_encoding.key_mlp.parameters()]


def test_trainer_with_key_embedding_trains_key_mlp(tmp_path):
    model = _tiny_cryptovit()
    before = _key_mlp_state(model)
    key = KeyAwarePositionalEncoding.key_to_embedding("k", embed_dim=32)
    Trainer(model, _cfg(), torch.device("cpu"), tmp_path, key_embedding=key).fit(
        _loader(), _loader()
    )
    after = _key_mlp_state(model)
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_trainer_without_key_embedding_leaves_key_mlp_untouched(tmp_path):
    model = _tiny_cryptovit()
    before = _key_mlp_state(model)
    Trainer(model, _cfg(), torch.device("cpu"), tmp_path).fit(_loader(), _loader())
    after = _key_mlp_state(model)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


class _RecordingModel(nn.Module):
    """Stub that records whether key_embedding was passed."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)
        self.seen_keys = []

    def forward(self, x, key_embedding=None):
        self.seen_keys.append(key_embedding)
        return self.fc(x.flatten(1))


def _small_loader(labels):
    x = torch.randn(len(labels), 3, 4, 4)
    return DataLoader(TensorDataset(x, torch.tensor(labels)), batch_size=4)


def test_evaluate_model_forwards_key_embedding():
    model = _RecordingModel()
    key = torch.ones(1, 8)
    evaluate_model(
        model, _small_loader([0, 1, 2, 0]), torch.device("cpu"), num_classes=3, key_embedding=key
    )
    assert model.seen_keys and all(k is not None for k in model.seen_keys)

    model = _RecordingModel()
    evaluate_model(model, _small_loader([0, 1, 2, 0]), torch.device("cpu"), num_classes=3)
    assert all(k is None for k in model.seen_keys)


def test_evaluate_model_per_class_length_is_fixed_when_class_missing():
    model = _RecordingModel(num_classes=5)
    results = evaluate_model(model, _small_loader([0, 1, 0, 1]), torch.device("cpu"), num_classes=5)
    assert len(results["per_class_precision"]) == 5
    assert len(results["per_class_recall"]) == 5
    assert len(results["per_class_f1"]) == 5
    cm = results["confusion_matrix"]
    assert len(cm) == 5 and all(len(row) == 5 for row in cm)
    assert sum(sum(row) for row in cm) == 4


def test_no_weight_decay_on_one_dimensional_params(tmp_path):
    model = _tiny_cryptovit()
    trainer = Trainer(model, _cfg(), torch.device("cpu"), tmp_path)
    decay_group, no_decay_group = trainer.optimizer.param_groups
    assert decay_group["weight_decay"] == pytest.approx(0.05)
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(p.ndim <= 1 for p in no_decay_group["params"])
    n_params = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    assert n_params == len(list(model.parameters()))


def test_lr_schedule_completes_with_gradient_accumulation(tmp_path):
    """With accumulation the schedule must still reach ~min_lr at the end."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    x = torch.randn(16, 3, 4, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    cfg = _cfg(epochs=3, accumulation_steps=2)
    trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
    history = trainer.fit(loader, loader)
    assert history["lr"][-1] == pytest.approx(1e-6, rel=0.05)
