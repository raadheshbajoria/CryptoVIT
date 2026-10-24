#!/usr/bin/env python
"""
scripts/run_ablation.py
=========================
Phase 12 — Ablation study: train and evaluate 4 model variants.

Variants
--------
A — Standard ViT + plaintext images
B — Standard ViT + encrypted images (standard patch embedding)
C — Standard ViT + encrypted images + CryptoPatchEmbedding
D — CryptoViT: C + KeyAwarePositionalEncoding, trained and evaluated with the
    key embedding derived from the encryption key

Protocol
--------
* All variants use the same seeded train/validation split of the official
  training set (``get_split_indices``) and the same training config.
* Checkpoints and early stopping use the validation split only. The official
  test split is used once per variant, after training, for the final score.
* Encrypted variants use no random crop/flip (they are meaningless on
  ciphertext) and normalise with statistics of the encrypted training images.
* C differs from D only by the key-aware positional offset, so D vs C
  isolates what the key embedding adds.

Results are saved to experiments/ablation/ablation_results.json.

Usage
-----
    python scripts/run_ablation.py
    python scripts/run_ablation.py training.epochs=30
    python scripts/run_ablation.py ablation.variants=A,D
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
from omegaconf import OmegaConf
from torchvision import datasets

from cryptovit.datasets import (
    build_encrypted_loaders,
    get_class_names,
    get_dataloaders,
)
from cryptovit.evaluation import evaluate_model, print_metrics_table
from cryptovit.models.cryptovit import CryptoViT
from cryptovit.models.patch_embedding import CryptoPatchEmbedding
from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding
from cryptovit.models.vit import VisionTransformer
from cryptovit.training import Trainer
from cryptovit.training.trainer import split_batch_for_memory
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Encryption key for ablation study
ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"
KEY_EMBED_DIM = 256
ALL_VARIANTS = ["A", "B", "C", "D"]

# CryptoPatchEmbedding runs 3x3 convolutions on the full 224x224 image. Measured
# on the 4 GB RTX 3050 with AMP: batch 32 -> 1.75 GB (0.15 s/step); batch 64 ->
# 3.4 GB and batch 128 -> 6.7 GB, which spill into shared memory and run 10-70x
# slower. C and D therefore use micro-batches of 32 with gradient accumulation,
# keeping the effective batch size identical to A and B.
CRYPTO_EMBED_MAX_MICRO_BATCH = 32
CRYPTO_EMBED_VARIANTS = ("C", "D")


def _variant_cfg(variant: str, cfg):
    """Return a config copy with memory-safe batching for the variant."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if variant in CRYPTO_EMBED_VARIANTS:
        micro, accum = split_batch_for_memory(
            cfg.training.batch_size,
            cfg.training.accumulation_steps,
            CRYPTO_EMBED_MAX_MICRO_BATCH,
        )
        cfg.training.batch_size = micro
        cfg.training.accumulation_steps = accum
    return cfg


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

    elif name == "D":
        # CryptoViT: CryptoPatchEmbedding + KeyAwarePositionalEncoding
        return CryptoViT(
            **common,
            hidden_dim=64,
            key_embed_dim=KEY_EMBED_DIM,
        )

    raise ValueError(f"Unknown variant: {name}. Choose from A, B, C, D.")


def _get_loaders(variant: str, cfg, data_root: Path):
    """Return (train_loader, val_loader, test_loader) for a variant.

    Every variant uses the same seeded train/val split and the official test
    split; only the image representation (plaintext vs encrypted) differs.
    """
    if variant == "A":
        return get_dataloaders(cfg, data_root=data_root, seed=cfg.project.seed)

    cls = datasets.CIFAR10 if cfg.dataset.name == "cifar10" else datasets.CIFAR100
    base_train = cls(root=str(data_root), train=True, transform=None, download=False)
    base_test  = cls(root=str(data_root), train=False, transform=None, download=False)
    return build_encrypted_loaders(
        cfg, base_train, base_test, ENCRYPTION_KEY, seed=cfg.project.seed
    )


def run_variant(
    variant: str, cfg, device: torch.device, exp_dir: Path
) -> dict:
    """Train and evaluate one ablation variant."""
    logger.info("\n%s\nVariant %s\n%s", "="*50, variant, "="*50)
    cfg = _variant_cfg(variant, cfg)
    logger.info(
        "Batching: micro-batch=%d x accumulation=%d (effective %d)",
        cfg.training.batch_size, cfg.training.accumulation_steps,
        cfg.training.batch_size * cfg.training.accumulation_steps,
    )
    set_seed(cfg.project.seed)

    data_root = PROJECT_ROOT / cfg.paths.raw_dir
    train_l, val_l, test_l = _get_loaders(variant, cfg, data_root)
    model = _build_variant(variant, cfg, device)

    # Variant D is trained and evaluated with the key embedding of the
    # encryption key; no other variant receives one.
    key_embedding = (
        KeyAwarePositionalEncoding.key_to_embedding(ENCRYPTION_KEY, KEY_EMBED_DIM)
        if variant == "D" else None
    )

    var_dir = exp_dir / f"variant_{variant}"
    trainer = Trainer(model, cfg, device, var_dir, key_embedding=key_embedding)
    t0 = time.time()
    history = trainer.fit(train_l, val_l)
    train_time = time.time() - t0

    trainer.load_best_checkpoint()

    # Test set is touched exactly once, after model selection on validation.
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    results = evaluate_model(
        model, test_l, device,
        use_amp=use_amp,
        class_names=get_class_names(cfg.dataset.name),
        num_classes=cfg.dataset.num_classes,
        key_embedding=key_embedding,
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

    requested = str(OmegaConf.select(cfg, "ablation.variants", default=",".join(ALL_VARIANTS)))
    variants = [v.strip().upper() for v in requested.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {ALL_VARIANTS}.")

    all_results = {}
    for variant in variants:
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
