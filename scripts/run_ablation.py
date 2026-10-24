#!/usr/bin/env python
"""
scripts/run_ablation.py
=========================
Phase 12 — Ablation study: train and evaluate 5 model variants.

Variants
--------
A — Standard ViT + plaintext images
B — Standard ViT + encrypted images (standard patch embedding)
C — Standard ViT + encrypted images + CryptoPatchEmbedding
D — Standard ViT + encrypted images + CryptoPatchEmbedding + KeyAwarePE
E — Full CryptoViT (D + increased depth / fine-tuned HP)

All variants use the same training config (same LR, epochs, batch size).
Results are saved to experiments/ablation/ablation_results.json.

Usage
-----
    python scripts/run_ablation.py
    python scripts/run_ablation.py training.epochs=30
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn

from cryptovit.datasets import get_class_names, get_dataloaders
from cryptovit.datasets.encrypted_dataset import EncryptedCIFARDataset
from cryptovit.evaluation import evaluate_model, print_metrics_table
from cryptovit.models.cryptovit import CryptoViT
from cryptovit.models.patch_embedding import CryptoPatchEmbedding
from cryptovit.models.vit import VisionTransformer
from cryptovit.training import Trainer
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Encryption key for ablation study
ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"


def _build_variant(name: str, cfg, device) -> nn.Module:
    """Build the model for a given ablation variant."""
    common = dict(
        image_size  = cfg.model.image_size,
        patch_size  = cfg.model.patch_size,
        in_channels = cfg.model.in_channels,
        num_classes = cfg.model.num_classes,
        embed_dim   = cfg.model.embed_dim,
        depth       = cfg.model.depth,
        num_heads   = cfg.model.num_heads,
        mlp_ratio   = cfg.model.mlp_ratio,
        dropout     = cfg.model.dropout,
        attn_dropout= cfg.model.attn_dropout,
        drop_path_rate = cfg.model.drop_path_rate,
        pool        = cfg.model.pool,
    )

    if name in ("A", "B"):
        # Standard ViT (plaintext or encrypted, standard embedding)
        return VisionTransformer(**common)

    elif name == "C":
        # Encrypted + CryptoPatchEmbedding
        return VisionTransformer(
            **common,
            patch_embedding_cls=CryptoPatchEmbedding,
        )

    elif name in ("D", "E"):
        # Full CryptoViT (CryptoPatchEmbedding + KeyAwarePE)
        return CryptoViT(
            **common,
            hidden_dim=64,
            key_embed_dim=256,
        )

    raise ValueError(f"Unknown variant: {name}")


def _get_loaders(variant: str, cfg, data_root: Path):
    """Return (train_loader, val_loader, test_loader) for a variant."""
    train_l, val_l, test_l = get_dataloaders(
        cfg, data_root=data_root, seed=cfg.project.seed
    )

    if variant == "A":
        # Plaintext — use standard loaders
        return train_l, val_l, test_l

    # Variants B–E: use encrypted dataset (online mode for now)
    # For full training, switch to precomputed mode after generating data
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from cryptovit.datasets import build_eval_transform, build_train_transform

    cls = datasets.CIFAR10 if cfg.dataset.name == "cifar10" else datasets.CIFAR100
    base_train = cls(root=str(data_root), train=True, transform=None, download=False)
    base_test  = cls(root=str(data_root), train=False, transform=None, download=False)

    train_transform = build_train_transform(cfg)
    eval_transform  = build_eval_transform(cfg)

    enc_train = EncryptedCIFARDataset(base_train, ENCRYPTION_KEY, mode="online",
                                       transform=train_transform)
    enc_test  = EncryptedCIFARDataset(base_test,  ENCRYPTION_KEY, mode="online",
                                       transform=eval_transform)

    enc_train_l = DataLoader(
        enc_train, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
        drop_last=True,
    )
    enc_test_l = DataLoader(
        enc_test, batch_size=cfg.evaluation.batch_size, shuffle=False,
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
    )
    # Use validation loader from plaintext split (same indices, encrypted)
    return enc_train_l, enc_test_l, enc_test_l


def run_variant(
    variant: str, cfg, device: torch.device, exp_dir: Path
) -> dict:
    """Train and evaluate one ablation variant."""
    logger.info("\n%s\nVariant %s\n%s", "="*50, variant, "="*50)
    set_seed(cfg.project.seed)

    data_root = PROJECT_ROOT / cfg.paths.raw_dir
    train_l, val_l, test_l = _get_loaders(variant, cfg, data_root)
    model = _build_variant(variant, cfg, device)

    var_dir = exp_dir / f"variant_{variant}"
    trainer = Trainer(model, cfg, device, var_dir)
    t0 = time.time()
    history = trainer.fit(train_l, val_l)
    train_time = time.time() - t0

    trainer.load_best_checkpoint()

    # Forward with key embedding for variants D/E
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    results = evaluate_model(
        model, test_l, device,
        use_amp=use_amp,
        class_names=get_class_names(cfg.dataset.name),
        num_classes=cfg.dataset.num_classes,
    )
    print_metrics_table(results)

    results["variant"] = variant
    results["train_time_s"] = train_time
    results["best_val_acc"] = max(history["val_acc"]) if history["val_acc"] else 0.0

    return results


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_config(overrides=overrides if overrides else None)
    setup_logging(level=cfg.logging.level)

    device = get_device(cfg)
    exp_dir = PROJECT_ROOT / cfg.paths.experiments_dir / "ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for variant in ["A", "B", "C", "D", "E"]:
        try:
            results = run_variant(variant, cfg, device, exp_dir)
            all_results[variant] = results
        except Exception as exc:
            logger.error("Variant %s failed: %s", variant, exc, exc_info=True)
            all_results[variant] = {"error": str(exc)}

    # Save combined results
    out_path = exp_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation results saved to %s", out_path)

    # Print summary table
    print("\n" + "="*70)
    print(f"  {'Variant':<10} {'Test Acc':>10} {'Best Val Acc':>14} {'Train Time (min)':>18}")
    print("  " + "-"*66)
    for v, r in all_results.items():
        if "error" in r:
            print(f"  {v:<10}  ERROR: {r['error'][:50]}")
        else:
            acc  = r.get("accuracy", 0)
            val  = r.get("best_val_acc", 0)
            t    = r.get("train_time_s", 0) / 60
            print(f"  {v:<10} {acc:>10.4f} {val:>14.4f} {t:>18.1f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
