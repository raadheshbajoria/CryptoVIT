from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from cryptovit.training import Trainer


def _cfg(epochs, patience=100):
    return OmegaConf.create(
        {
            "training": {
                "epochs": epochs,
                "batch_size": 4,
                "accumulation_steps": 1,
                "mixed_precision": False,
                "learning_rate": 1e-2,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "warmup_epochs": 1,
                "min_lr": 1e-5,
                "label_smoothing": 0.0,
                "clip_grad_norm": 1.0,
                "save_every_n_epochs": 100,
                "keep_best_k": 2,
                "early_stopping_patience": patience,
            },
            "logging": {"use_tensorboard": False},
        }
    )


def _data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 3, 4, 4, generator=g)
    y = torch.randint(0, 2, (16,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))


class _Interrupt(Exception):
    pass


def _train_with_crash_after(trainer, loader, n_epochs_before_crash):
    calls = {"n": 0}
    original = trainer._val_epoch

    def flaky(val_loader):
        calls["n"] += 1
        if calls["n"] > n_epochs_before_crash:
            raise _Interrupt()
        return original(val_loader)

    trainer._val_epoch = flaky
    with pytest.raises(_Interrupt):
        trainer.fit(loader, loader)


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    loader = _data()

    straight = Trainer(_model(), _cfg(4), torch.device("cpu"), tmp_path / "straight")
    straight_history = straight.fit(loader, loader)

    crashed = Trainer(_model(), _cfg(4), torch.device("cpu"), tmp_path / "resumed")
    _train_with_crash_after(crashed, loader, n_epochs_before_crash=2)

    resumed = Trainer(_model(), _cfg(4), torch.device("cpu"), tmp_path / "resumed")
    history = resumed.fit(loader, loader, resume=True)

    assert len(history["train_loss"]) == 4
    assert history["train_loss"] == pytest.approx(straight_history["train_loss"])
    assert history["lr"] == pytest.approx(straight_history["lr"])
    for a, b in zip(straight.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_resume_without_checkpoint_starts_from_scratch(tmp_path):
    loader = _data()
    trainer = Trainer(_model(), _cfg(2), torch.device("cpu"), tmp_path)
    history = trainer.fit(loader, loader, resume=True)
    assert len(history["train_loss"]) == 2


def test_finished_run_is_not_trained_again(tmp_path):
    loader = _data()
    first = Trainer(_model(), _cfg(2), torch.device("cpu"), tmp_path)
    first_history = copy.deepcopy(first.fit(loader, loader))

    second = Trainer(_model(), _cfg(2), torch.device("cpu"), tmp_path)
    history = second.fit(loader, loader, resume=True)

    assert history["train_loss"] == first_history["train_loss"]
    assert second.ckpt_manager.best_checkpoint() is not None


def test_best_checkpoint_survives_resume(tmp_path):
    loader = _data()
    crashed = Trainer(_model(), _cfg(3), torch.device("cpu"), tmp_path)
    _train_with_crash_after(crashed, loader, n_epochs_before_crash=1)

    resumed = Trainer(_model(), _cfg(3), torch.device("cpu"), tmp_path)
    resumed.fit(loader, loader, resume=True)
    resumed.load_best_checkpoint()


def test_last_checkpoint_has_no_leftover_temp_file(tmp_path):
    loader = _data()
    Trainer(_model(), _cfg(1), torch.device("cpu"), tmp_path).fit(loader, loader)
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert not list(ckpt_dir.glob("*.tmp"))
