#!/usr/bin/env python
"""
scripts/train_review_baseline.py
=================================
Opt-in reproducible review training script for Monday review.

Trains a standard VisionTransformer (vit_tiny) on a verified subset of CIFAR-10:
- Train subset: 5,000 disjoint samples from the official training split
- Validation subset: 1,000 disjoint samples from the official training split
- Test set: official CIFAR-10 test set (10,000 samples), evaluated once at the end
- Does not modify configs/base.yaml or the default training workflow.
- Saves all artifacts into a timestamped directory under experiments/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, random_split

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cryptovit.datasets import (
    build_eval_transform,
    build_train_transform,
    get_class_names,
)
from cryptovit.datasets.cifar_dataset import _load_raw_dataset
from cryptovit.evaluation.metrics import evaluate_model
from cryptovit.models import build_model
from cryptovit.training import Trainer
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger("cryptovit.train_review")


def create_subset_dataloaders(
    cfg: DictConfig,
    train_size: int = 5000,
    val_size: int = 1000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int], List[int]]:
    """Build reproducible disjoint train, validation subsets and official test loader."""
    dataset_name = cfg.dataset.name
    data_root = PROJECT_ROOT / cfg.paths.raw_dir

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    # Load datasets
    full_train = _load_raw_dataset(
        dataset_name, data_root, train=True, transform=train_transform, download=False
    )
    full_train_eval = _load_raw_dataset(
        dataset_name, data_root, train=True, transform=eval_transform, download=False
    )
    test_dataset = _load_raw_dataset(
        dataset_name, data_root, train=False, transform=eval_transform, download=False
    )

    n_total = len(full_train)  # 50,000 for CIFAR-10
    assert train_size + val_size <= n_total, "Requested subset exceeds dataset size"

    generator = torch.Generator().manual_seed(seed)
    # Permute indices reproducibly
    shuffled_indices = torch.randperm(n_total, generator=generator).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size : train_size + val_size]

    # Verification: disjoint indices
    overlap = set(train_indices).intersection(set(val_indices))
    assert len(overlap) == 0, f"Data leakage detected! Overlapping indices count: {len(overlap)}"

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train_eval, val_indices)

    batch_train = cfg.training.batch_size
    batch_eval = cfg.evaluation.batch_size
    num_workers = cfg.dataset.num_workers
    pin_memory = cfg.dataset.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_indices, val_indices


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Val Acc (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: Path) -> None:
    """Plot and save normalized confusion matrix."""
    matrix = np.array(cm, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_cm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(norm_cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Test Confusion Matrix (Normalized)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{norm_cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if norm_cm[i, j] > 0.5 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Baseline Training Run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--image-size", type=int, default=96, help="Image resolution (96x96)")
    parser.add_argument("--train-samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # Load base configuration and override for vit_tiny review run
    cfg = load_config()

    cfg.project.seed = args.seed
    cfg.dataset.image_size = args.image_size
    cfg.dataset.num_workers = 2
    cfg.model.name = "vit_tiny"
    cfg.model.image_size = args.image_size
    cfg.model.patch_size = 16
    cfg.model.embed_dim = 96
    cfg.model.depth = 4
    cfg.model.num_heads = 3
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.lr
    cfg.training.warmup_epochs = 1
    cfg.training.save_every_n_epochs = 1
    cfg.training.keep_best_k = 3

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / f"review_baseline_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / "training.log"
    setup_logging(level=cfg.logging.level, log_file=log_file)

    logger.info("=" * 65)
    logger.info("CryptoViT - Monday Review Baseline Training Run")
    logger.info("Output directory : %s", exp_dir)
    logger.info("Model            : %s (embed_dim=%d, depth=%d, heads=%d)",
                cfg.model.name, cfg.model.embed_dim, cfg.model.depth, cfg.model.num_heads)
    logger.info("Image size       : %d x %d (patch_size=%d)",
                cfg.model.image_size, cfg.model.image_size, cfg.model.patch_size)
    logger.info("Dataset subset   : train=%d, val=%d (seed=%d)",
                args.train_samples, args.val_samples, args.seed)
    logger.info("Epochs           : %d, Batch size: %d, LR: %e",
                cfg.training.epochs, cfg.training.batch_size, cfg.training.learning_rate)
    logger.info("=" * 65)

    set_seed(cfg.project.seed)
    device = get_device(cfg)
    logger.info("Device           : %s", device)
    if device.type == "cuda":
        logger.info("GPU Name         : %s", torch.cuda.get_device_name(0))

    # DataLoaders
    logger.info("Preparing data loaders...")
    train_loader, val_loader, test_loader, train_idx, val_idx = create_subset_dataloaders(
        cfg,
        train_size=args.train_samples,
        val_size=args.val_samples,
        seed=cfg.project.seed,
    )
    class_names = get_class_names(cfg.dataset.name)
    logger.info("DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d",
                len(train_loader), len(val_loader), len(test_loader))

    # Build standard ViT model
    logger.info("Constructing standard VisionTransformer...")
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters : %d (%.2fM)", total_params, total_params / 1e6)

    # Train
    trainer = Trainer(model, cfg, device, exp_dir)
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0
    logger.info("Training finished in %.2f seconds (%.2f minutes).", train_time, train_time / 60)

    # Save history
    history_file = exp_dir / "history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_file)

    # Plot curves
    curve_plot = exp_dir / "loss_acc_curves.png"
    plot_training_history(history, curve_plot)

    # Load best checkpoint and evaluate on official test set
    trainer.load_best_checkpoint()
    best_ckpt_path = trainer.ckpt_manager.best_checkpoint()
    logger.info("Best checkpoint path: %s", best_ckpt_path)

    logger.info("Evaluating selected best checkpoint on official test set (10,000 images)...")
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        use_amp=use_amp,
        class_names=class_names,
        num_classes=cfg.dataset.num_classes,
    )

    test_metrics["train_time_seconds"] = train_time
    test_metrics["train_samples"] = args.train_samples
    test_metrics["val_samples"] = args.val_samples
    test_metrics["model_name"] = cfg.model.name
    test_metrics["dataset_name"] = cfg.dataset.name
    test_metrics["best_checkpoint"] = str(best_ckpt_path)

    metrics_file = exp_dir / "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Saved test metrics to %s", metrics_file)

    # Confusion matrix
    if "confusion_matrix" in test_metrics:
        cm_plot = exp_dir / "confusion_matrix.png"
        plot_confusion_matrix(test_metrics["confusion_matrix"], class_names, cm_plot)
        logger.info("Saved confusion matrix plot to %s", cm_plot)

    # Environment and run metadata
    env_info = {
        "timestamp": timestamp,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "total_parameters": total_params,
        "training_time_seconds": train_time,
        "best_checkpoint": str(best_ckpt_path),
        "history": history,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "train_indices_sample": train_idx[:20],
        "val_indices_sample": val_idx[:20],
    }
    meta_file = exp_dir / "run_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(env_info, f, indent=2)

    # Verification: test reloading checkpoint and doing inference on batch
    logger.info("Verifying checkpoint loading and inference...")
    eval_model = build_model(cfg).to(device)
    saved_state = torch.load(best_ckpt_path, map_location=device)
    eval_model.load_state_dict(saved_state["model_state_dict"])
    eval_model.eval()

    test_batch, test_targets = next(iter(test_loader))
    test_batch = test_batch.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = eval_model(test_batch)
    preds = logits.argmax(dim=1).cpu().tolist()
    sample_preds = [
        {
            "index": i,
            "pred_idx": preds[i],
            "pred_class": class_names[preds[i]],
            "true_idx": test_targets[i].item(),
            "true_class": class_names[test_targets[i].item()],
        }
        for i in range(min(10, len(preds)))
    ]
    sample_preds_file = exp_dir / "sample_predictions.json"
    with open(sample_preds_file, "w") as f:
        json.dump(sample_preds, f, indent=2)

    logger.info("Batch inference verified. Sample predictions saved.")
    logger.info("=" * 65)
    logger.info("Review baseline complete!")
    logger.info("Best Val Accuracy  : %.4f", max(history["val_acc"]))
    logger.info("Test Accuracy      : %.4f", test_metrics["accuracy"])
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

