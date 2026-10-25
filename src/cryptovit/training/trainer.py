"""
cryptovit/training/trainer.py
==============================
Full training loop for CryptoViT / plain ViT.

Features
--------
* AdamW optimiser with cosine LR schedule + linear warmup
* PyTorch AMP (mixed precision) for RTX 3050 4GB VRAM
* Gradient clipping
* Checkpoint saving - best-K + periodic
* Early stopping
* Rich progress display via tqdm
* Minimal TensorBoard support (optional)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-6,
    base_lr: float = 3e-4,
) -> LambdaLR:
    """Return a LambdaLR that implements linear warmup then cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr / base_lr
        return max(scale, cosine * (1.0 - scale) + scale)

    return LambdaLR(optimizer, lr_lambda)


def balanced_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights: ``N / (num_classes * count_c)``.

    A class with no training samples gets the weight of a single-sample class
    instead of dividing by zero.
    """
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)
    weights = len(targets) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def build_class_weights(cfg: DictConfig, train_loader: DataLoader) -> Optional[torch.Tensor]:
    """Class weights requested by ``cfg.training.class_weights`` ("none"|"balanced")."""
    mode = cfg.training.get("class_weights", "none")
    if mode in (None, "none", False):
        return None
    if mode != "balanced":
        raise ValueError(f"training.class_weights must be 'none' or 'balanced', got '{mode}'.")
    from cryptovit.datasets.cifar_dataset import get_dataset_targets

    targets = get_dataset_targets(train_loader.dataset)
    weights = balanced_class_weights(targets, cfg.dataset.num_classes)
    logger.info("Using balanced class weights: %s", [round(w, 2) for w in weights.tolist()])
    return weights


def split_batch_for_memory(
    batch_size: int, accumulation_steps: int, max_micro_batch: int
) -> Tuple[int, int]:
    """Return ``(micro_batch, accumulation_steps)`` with the same effective batch.

    Used when a model does not fit in GPU memory at the configured batch
    size: the loader yields ``micro_batch`` images and gradients are
    accumulated so that ``micro_batch * accumulation_steps`` equals the
    original ``batch_size * accumulation_steps``.
    """
    if batch_size <= 0 or accumulation_steps <= 0 or max_micro_batch <= 0:
        raise ValueError("batch_size, accumulation_steps and max_micro_batch must be > 0")
    if batch_size <= max_micro_batch:
        return batch_size, accumulation_steps
    if batch_size % max_micro_batch != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be a multiple of max_micro_batch "
            f"({max_micro_batch}) to keep the effective batch size unchanged."
        )
    return max_micro_batch, accumulation_steps * (batch_size // max_micro_batch)


class CheckpointManager:
    """Saves best-K checkpoints and periodic snapshots."""

    def __init__(self, save_dir: Path, keep_best_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self._best: List[Tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
        extra: Optional[dict] = None,
    ) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            **(extra or {}),
        }
        path = self.save_dir / f"checkpoint_epoch{epoch:04d}_acc{val_acc:.4f}.pt"
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        self._best.append((val_acc, path))
        self._best.sort(key=lambda t: t[0], reverse=True)
        if len(self._best) > self.keep_best_k:
            _, old_path = self._best.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug("Removed old checkpoint: %s", old_path)

        return path

    def save_periodic(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_periodic_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        return path

    def best_checkpoint(self) -> Optional[Path]:
        if self._best:
            return self._best[0][1]
        return None


class Trainer:
    """Encapsulates the complete training loop.

    Parameters
    ----------
    model:
        The neural network to train.
    cfg:
        Full project configuration.
    device:
        torch.device to run on.
    experiment_dir:
        Root directory for checkpoints and logs.
    key_embedding:
        Optional ``(1, key_embed_dim)`` tensor passed to the model as
        ``key_embedding=`` on every forward pass (training and validation).
        Only models with a key-aware positional encoding accept it; leave
        ``None`` for every other model.
    class_weights:
        Optional per-class loss weights (see ``balanced_class_weights``), used
        to counter class imbalance.

    ``cfg.training.select_metric`` chooses what decides the "best" checkpoint
    and early stopping: ``"accuracy"`` (default) or ``"balanced_accuracy"``
    (mean per-class recall, the right choice for imbalanced medical data).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device,
        experiment_dir: Path,
        key_embedding: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.key_embedding = key_embedding.to(device) if key_embedding is not None else None

        t = cfg.training
        self.epochs: int = t.epochs
        self.accum_steps: int = t.accumulation_steps
        self.clip_norm: float = t.clip_grad_norm
        self.save_every: int = t.save_every_n_epochs
        self.patience: int = t.early_stopping_patience
        self.use_amp: bool = t.mixed_precision and device.type == "cuda"
        self.label_smoothing: float = t.label_smoothing

        self.select_metric: str = t.get("select_metric", "accuracy")
        if self.select_metric not in ("accuracy", "balanced_accuracy"):
            raise ValueError(
                "training.select_metric must be 'accuracy' or 'balanced_accuracy', "
                f"got '{self.select_metric}'."
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )

        decay, no_decay = [], []
        for p in model.parameters():
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
        self.optimizer = AdamW(
            [
                {"params": decay, "weight_decay": t.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=t.learning_rate,
            betas=tuple(t.betas),
        )

        self.scheduler: Optional[LambdaLR] = None

        device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cuda"
        self.scaler = GradScaler(device_type, enabled=self.use_amp)

        ckpt_dir = self.experiment_dir / "checkpoints"
        self.ckpt_manager = CheckpointManager(ckpt_dir, keep_best_k=t.keep_best_k)

        self.tb_writer = None
        if cfg.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.experiment_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                logger.warning("TensorBoard not available. Skipping TB logging.")

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_acc": [],
            "val_score": [],
            "lr": [],
        }
        self._best_val_acc: float = 0.0
        self._no_improve: int = 0
        self._last_val_balanced_acc: float = 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: bool = False,
    ) -> Dict[str, List[float]]:
        """Run the complete training loop.

        Parameters
        ----------
        train_loader:
            DataLoader for the training set.
        val_loader:
            DataLoader for the validation set.
        resume:
            If True and ``checkpoints/last.pt`` exists in the experiment
            directory, continue from the epoch after the one it holds
            (model, optimiser, LR schedule, AMP scaler, history and
            early-stopping state are restored). The data order of the
            resumed epochs is not bit-identical to an uninterrupted run.

        Returns
        -------
        dict
            Training history (losses and accuracies per epoch).
        """
        t = self.cfg.training
        steps_per_epoch = math.ceil(len(train_loader) / self.accum_steps)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = t.warmup_epochs * steps_per_epoch

        self.scheduler = _cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr=t.min_lr,
            base_lr=t.learning_rate,
        )

        logger.info(
            "Starting training | epochs=%d  steps/epoch=%d  warmup=%d",
            self.epochs,
            steps_per_epoch,
            warmup_steps,
        )

        start_epoch, finished = 1, False
        if resume:
            start_epoch, finished = self._load_last()
        if finished:
            logger.info("Checkpoint marks training as finished; nothing to do.")
            start_epoch = self.epochs + 1

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_loss, val_acc = self._val_epoch(val_loader)
            val_bal = self._last_val_balanced_acc
            score = val_bal if self.select_metric == "balanced_accuracy" else val_acc
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_balanced_acc"].append(val_bal)
            self.history["val_score"].append(score)
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f acc=%.4f | "
                "val_loss=%.4f acc=%.4f bal_acc=%.4f | lr=%.2e | %.1fs",
                epoch,
                self.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_bal,
                current_lr,
                elapsed,
            )

            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("Acc/train", train_acc, epoch)
                self.tb_writer.add_scalar("Acc/val", val_acc, epoch)
                self.tb_writer.add_scalar("LR", current_lr, epoch)

            if score > self._best_val_acc:
                self._best_val_acc = score
                self._no_improve = 0
                self.ckpt_manager.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    score,
                    extra={
                        "train_acc": train_acc,
                        "val_accuracy": val_acc,
                        "select_metric": self.select_metric,
                    },
                )
                logger.info("  New best %s: %.4f", self.select_metric, score)
            else:
                self._no_improve += 1

            if epoch % self.save_every == 0:
                self.ckpt_manager.save_periodic(self.model, self.optimizer, epoch)

            stop = self._no_improve >= self.patience
            self._save_last(epoch, finished=stop or epoch == self.epochs)
            if stop:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )
                break

        if self.tb_writer:
            self.tb_writer.close()

        return self.history

    @property
    def _last_path(self) -> Path:
        return self.ckpt_manager.save_dir / "last.pt"

    def _save_last(self, epoch: int, finished: bool) -> None:
        """Write the full training state so an interrupted run can resume.

        Written to a temporary file first and then renamed, so a crash while
        saving never leaves a corrupt ``last.pt``.
        """
        state = {
            "epoch": epoch,
            "finished": finished,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
            "best_val_acc": self._best_val_acc,
            "no_improve": self._no_improve,
            "best_checkpoints": [(acc, str(path)) for acc, path in self.ckpt_manager._best],
        }
        tmp = self._last_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self._last_path)

    def _load_last(self) -> Tuple[int, bool]:
        """Restore state from ``last.pt``; return ``(next_epoch, finished)``."""
        if not self._last_path.exists():
            logger.info("No resume checkpoint at %s; starting from epoch 1.", self._last_path)
            return 1, False
        state = torch.load(self._last_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.history = state["history"]
        self._best_val_acc = state["best_val_acc"]
        self._no_improve = state["no_improve"]
        self.ckpt_manager._best = [(acc, Path(p)) for acc, p in state["best_checkpoints"]]
        logger.info(
            "Resumed from %s: last completed epoch=%d  best_val_acc=%.4f",
            self._last_path,
            state["epoch"],
            self._best_val_acc,
        )
        return state["epoch"] + 1, bool(state["finished"])

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the model, supplying the key embedding when one is configured."""
        if self.key_embedding is None:
            return self.model(images)
        return self.model(images, key_embedding=self.key_embedding)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Train {epoch}/{self.epochs}",
            leave=False,
        )
        for step, (images, labels) in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=self.use_amp):
                logits = self._forward(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            with torch.no_grad():
                total_loss += loss.detach() * self.accum_steps * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum()
                total_samples += labels.size(0)

            if (step + 1) % 20 == 0 or step + 1 == len(loader):
                pbar.set_postfix(
                    loss=f"{float(total_loss) / total_samples:.4f}",
                    acc=f"{float(total_correct) / total_samples:.4f}",
                )

        if len(loader) % self.accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        return float(total_loss) / total_samples, float(total_correct) / total_samples

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_correct = class_count = None

        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=self.use_amp):
                logits = self._forward(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            hit = preds == labels
            total_correct += hit.sum().item()
            total_samples += labels.size(0)

            n_cls = logits.shape[1]
            if class_correct is None:
                class_correct = torch.zeros(n_cls, device=self.device)
                class_count = torch.zeros(n_cls, device=self.device)
            class_correct += torch.bincount(labels[hit], minlength=n_cls).float()
            class_count += torch.bincount(labels, minlength=n_cls).float()

        present = class_count > 0
        self._last_val_balanced_acc = float(
            (class_correct[present] / class_count[present]).mean().item()
        )
        return total_loss / total_samples, total_correct / total_samples

    def load_best_checkpoint(self) -> None:
        """Restore model weights from the best saved checkpoint."""
        best = self.ckpt_manager.best_checkpoint()
        if best is None:
            logger.warning("No checkpoint found - model weights unchanged.")
            return
        state = torch.load(best, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best checkpoint (val_acc=%.4f): %s", state["val_acc"], best)
