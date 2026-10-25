#!/usr/bin/env python
"""
scripts/run_ablation.py
=========================
Ablation study: train and evaluate 4 model variants.

Variants
--------
A - Standard ViT + plaintext images
B - Standard ViT + encrypted images (standard patch embedding)
C - Standard ViT + encrypted images + CryptoPatchEmbedding
D - CryptoViT: C + KeyAwarePositionalEncoding, trained and evaluated with the
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

Each variant's result is saved as <experiments_dir>/ablation/result_<variant>.json
and training is resumable (finished variants are skipped on a rerun).

Usage
-----
    python scripts/run_ablation.py config=dermamnist
    python scripts/run_ablation.py config=dermamnist training.epochs=30
    python scripts/run_ablation.py config=dermamnist ablation.variants=A,D
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import datasets

from cryptovit.datasets import (
    build_encrypted_loaders,
    get_class_names,
    get_dataloaders,
)
from cryptovit.datasets.medmnist_dataset import is_medmnist, load_medmnist_splits
from cryptovit.evaluation import evaluate_model, print_metrics_table
from cryptovit.models.cryptovit import CryptoViT
from cryptovit.models.patch_embedding import CryptoPatchEmbedding
from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding
from cryptovit.models.vit import VisionTransformer
from cryptovit.training import Trainer
from cryptovit.training.trainer import build_class_weights, split_batch_for_memory
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"
KEY_EMBED_DIM = 256
ALL_VARIANTS = ["A", "B", "C", "D"]

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
        cfg.evaluation.batch_size = min(cfg.evaluation.batch_size, CRYPTO_EMBED_MAX_MICRO_BATCH)
    return cfg


def _build_variant(name: str, cfg, device) -> nn.Module:
    """Build the model for a given ablation variant."""
    common = dict(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        drop_path_rate=cfg.model.drop_path_rate,
        pool=cfg.model.pool,
    )

    if name in ("A", "B"):
        return VisionTransformer(**common)

    elif name == "C":
        return VisionTransformer(
            **common,
            patch_embedding_cls=CryptoPatchEmbedding,
        )

    elif name == "D":
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

    if is_medmnist(cfg.dataset.name):
        train_b, val_b, test_b = load_medmnist_splits(
            cfg.dataset.name,
            data_root / "medmnist",
            size=int(cfg.dataset.get("medmnist_size", 28)),
        )
        return build_encrypted_loaders(
            cfg, train_b, test_b, ENCRYPTION_KEY, seed=cfg.project.seed, base_val=val_b
        )

    cls = datasets.CIFAR10 if cfg.dataset.name == "cifar10" else datasets.CIFAR100
    base_train = cls(root=str(data_root), train=True, transform=None, download=False)
    base_test = cls(root=str(data_root), train=False, transform=None, download=False)
    return build_encrypted_loaders(
        cfg, base_train, base_test, ENCRYPTION_KEY, seed=cfg.project.seed
    )


def run_variant(variant: str, cfg, device: torch.device, exp_dir: Path, loaders=None) -> dict:
    """Train and evaluate one ablation variant.

    ``loaders`` is an optional pre-built ``(train, val, test)`` tuple (used by
    tests with synthetic data); by default the real CIFAR loaders are built.
    """
    logger.info("\n%s\nVariant %s\n%s", "=" * 50, variant, "=" * 50)
    cfg = _variant_cfg(variant, cfg)
    logger.info(
        "Batching: micro-batch=%d x accumulation=%d (effective %d)",
        cfg.training.batch_size,
        cfg.training.accumulation_steps,
        cfg.training.batch_size * cfg.training.accumulation_steps,
    )
    set_seed(cfg.project.seed)

    if loaders is None:
        data_root = PROJECT_ROOT / cfg.paths.raw_dir
        loaders = _get_loaders(variant, cfg, data_root)
    train_l, val_l, test_l = loaders
    model = _build_variant(variant, cfg, device)

    key_embedding = (
        KeyAwarePositionalEncoding.key_to_embedding(ENCRYPTION_KEY, KEY_EMBED_DIM)
        if variant == "D"
        else None
    )

    var_dir = exp_dir / f"variant_{variant}"
    trainer = Trainer(
        model,
        cfg,
        device,
        var_dir,
        key_embedding=key_embedding,
        class_weights=build_class_weights(cfg, train_l),
    )
    t0 = time.time()
    history = trainer.fit(train_l, val_l, resume=True)
    train_time = time.time() - t0

    trainer.load_best_checkpoint()

    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    results = evaluate_model(
        model,
        test_l,
        device,
        use_amp=use_amp,
        class_names=get_class_names(cfg.dataset.name),
        num_classes=cfg.dataset.num_classes,
        key_embedding=key_embedding,
    )
    print_metrics_table(results)

    results["variant"] = variant
    results["train_time_s"] = train_time
    results["best_val_acc"] = max(history["val_acc"]) if history["val_acc"] else 0.0
    results["select_metric"] = cfg.training.get("select_metric", "accuracy")
    results["best_val_score"] = max(history["val_score"]) if history.get("val_score") else 0.0
    results["epochs_trained"] = len(history["val_acc"])

    return results


def _free_gpu_memory(device: torch.device) -> None:
    """Release cached GPU memory and log what is still held."""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(
            "GPU memory before next variant: allocated=%.0f MB  reserved=%.0f MB",
            torch.cuda.memory_allocated(device) / 2**20,
            torch.cuda.memory_reserved(device) / 2**20,
        )


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_config(overrides=overrides if overrides else None)

    exp_dir = PROJECT_ROOT / cfg.paths.experiments_dir / "ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=cfg.logging.level, log_file=exp_dir / "run.log")

    device = get_device(cfg)

    requested = str(OmegaConf.select(cfg, "ablation.variants", default=",".join(ALL_VARIANTS)))
    variants = [v.strip().upper() for v in requested.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {ALL_VARIANTS}.")

    all_results = {}
    for variant in variants:
        result_path = exp_dir / f"result_{variant}.json"
        if result_path.exists():
            logger.info("Variant %s already finished; loading %s", variant, result_path)
            all_results[variant] = json.loads(result_path.read_text())
            continue

        _free_gpu_memory(device)

        for attempt in (1, 2):
            try:
                results = run_variant(variant, cfg, device, exp_dir)
                result_path.write_text(json.dumps(results, indent=2))
                all_results[variant] = results
                break
            except Exception as exc:
                logger.error(
                    "Variant %s failed (attempt %d/2): %s", variant, attempt, exc, exc_info=True
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if attempt == 2:
                    all_results[variant] = {"error": str(exc)}

    out_path = exp_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation results saved to %s", out_path)

    print("\n" + "=" * 70)
    print(
        f"  {'Variant':<9} {'Test Acc':>9} {'Bal.Acc':>9} {'AUC':>7} "
        f"{'Best Val':>9} {'Time (min)':>11}"
    )
    print("  " + "-" * 60)
    for v, r in all_results.items():
        if "error" in r:
            print(f"  {v:<9}  ERROR: {r['error'][:50]}")
        else:
            auc = r.get("auc_macro_ovr")
            auc_s = f"{auc:>7.4f}" if auc is not None else f"{'n/a':>7}"
            print(
                f"  {v:<9} {r.get('accuracy', 0):>9.4f} "
                f"{r.get('balanced_accuracy', 0):>9.4f} {auc_s} "
                f"{r.get('best_val_score', r.get('best_val_acc', 0)):>9.4f} "
                f"{r.get('train_time_s', 0) / 60:>11.1f}"
            )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
