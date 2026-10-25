from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from cryptovit.training import CheckpointManager, Trainer
from cryptovit.training.trainer import _cosine_schedule_with_warmup


class TestCheckpointManager:
    def test_save_and_best(self, tmp_path):
        mgr = CheckpointManager(tmp_path, keep_best_k=2)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        p1 = mgr.save(model, optimizer, epoch=1, val_acc=0.5)
        assert p1.exists()
        assert mgr.best_checkpoint() == p1

        p2 = mgr.save(model, optimizer, epoch=2, val_acc=0.8)
        assert p2.exists()
        assert mgr.best_checkpoint() == p2

        p3 = mgr.save(model, optimizer, epoch=3, val_acc=0.6)
        assert p3.exists()

        assert not p1.exists()
        assert p2.exists()
        assert p3.exists()
        assert mgr.best_checkpoint() == p2

    def test_save_periodic(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        p = mgr.save_periodic(model, optimizer, epoch=5)
        assert p.exists()


class TestLRScheduler:
    def test_warmup_and_cosine(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = _cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            min_lr=1e-5,
            base_lr=1e-3,
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        assert lrs[0] < lrs[9]
        assert lrs[10] >= lrs[9]
        assert lrs[-1] < lrs[10]


class TestTrainer:
    def test_fit_and_checkpoint_loading(self, tmp_path):
        device = torch.device("cpu")
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))

        cfg = OmegaConf.create(
            {
                "training": {
                    "epochs": 2,
                    "batch_size": 4,
                    "accumulation_steps": 1,
                    "mixed_precision": False,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "warmup_epochs": 1,
                    "min_lr": 1e-5,
                    "label_smoothing": 0.0,
                    "clip_grad_norm": 1.0,
                    "save_every_n_epochs": 1,
                    "keep_best_k": 2,
                    "early_stopping_patience": 5,
                },
                "logging": {"use_tensorboard": False},
            }
        )

        x = torch.randn(16, 3, 8, 8)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        trainer = Trainer(model, cfg, device, tmp_path)
        history = trainer.fit(loader, loader)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        assert trainer.ckpt_manager.best_checkpoint() is not None

        trainer.load_best_checkpoint()
