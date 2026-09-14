#!/usr/bin/env python
"""
scripts/phase1_dataset_check.py
================================
Phase 1 verification script.

Performs end-to-end dataset sanity checks:
1. Downloads CIFAR-10 (and optionally CIFAR-100).
2. Verifies train/val/test split sizes.
3. Inspects batch shapes and dtype.
4. Computes and logs per-channel statistics.
5. Saves a visualisation grid.
6. Reports class distribution.

Run
---
    python scripts/phase1_dataset_check.py
    python scripts/phase1_dataset_check.py dataset.name=cifar100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from cryptovit.datasets import (
    compute_dataset_stats,
    get_class_names,
    get_dataloaders,
)
from cryptovit.utils import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 – dataset sanity check")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation config overrides, e.g. dataset.name=cifar100",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip saving visualisation (useful in headless environments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load configuration
    cfg = load_config(overrides=args.overrides)
    setup_logging(level=cfg.logging.level)

    logger.info("=" * 60)
    logger.info("CryptoViT – Phase 1: Dataset and Preprocessing")
    logger.info("=" * 60)
    logger.info("Dataset : %s", cfg.dataset.name)
    logger.info("Image   : %d x %d", cfg.dataset.image_size, cfg.dataset.image_size)
    logger.info("Seed    : %d", cfg.project.seed)

    # 2. Set seed
    set_seed(cfg.project.seed)

    # 3. Get device (informational only in Phase 1)
    device = get_device(cfg)

    # 4. Build DataLoaders
    logger.info("Building DataLoaders …")
    train_loader, val_loader, test_loader = get_dataloaders(cfg, seed=cfg.project.seed)

    # 5. Verify split sizes
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    logger.info("Split sizes  | train=%d  val=%d  test=%d", n_train, n_val, n_test)

    assert n_train > 0, "Training set is empty!"
    assert n_val > 0, "Validation set is empty!"
    assert n_test > 0, "Test set is empty!"

    # 6. Inspect a single batch
    images, labels = next(iter(train_loader))
    logger.info(
        "Batch shape  | images=%s  labels=%s  dtype=%s",
        tuple(images.shape), tuple(labels.shape), images.dtype,
    )
    assert images.ndim == 4, "Expected 4D image tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Expected 3-channel (RGB) images"
    assert images.shape[2] == cfg.dataset.image_size, "Unexpected image height"
    assert images.shape[3] == cfg.dataset.image_size, "Unexpected image width"
    assert images.dtype == torch.float32, "Expected float32 images"

    logger.info(
        "Pixel range  | min=%.4f  max=%.4f  mean=%.4f",
        float(images.min()), float(images.max()), float(images.mean()),
    )

    # 7. Compute per-channel statistics on the eval (unnormalised) loader
    # Build a temporary unnormalised eval loader for stat computation
    from cryptovit.datasets.cifar_dataset import (
        _TORCHVISION_CLS,
        _get_stats,
        build_eval_transform,
    )
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torchvision import transforms

    cfg_nonorm = OmegaConf.merge(
        cfg,
        OmegaConf.create({"dataset": {"augmentation": {"normalize": False}}}),
    )
    data_root = Path(cfg.paths.project_root) / cfg.paths.raw_dir
    raw_ds = _TORCHVISION_CLS[cfg.dataset.name](
        root=str(data_root),
        train=True,
        transform=build_eval_transform(cfg_nonorm),
        download=False,
    )
    raw_loader = DataLoader(raw_ds, batch_size=256, shuffle=False, num_workers=0)
    logger.info("Computing per-channel statistics (may take a moment) …")
    empirical_mean, empirical_std = compute_dataset_stats(raw_loader, n_batches=200)
    logger.info("Empirical  mean=%s  std=%s", empirical_mean, empirical_std)

    # 8. Class distribution on the test set
    all_labels = []
    for _, lbl in test_loader:
        all_labels.extend(lbl.tolist())

    class_names = get_class_names(cfg.dataset.name)
    import numpy as np
    counts = np.bincount(np.array(all_labels), minlength=cfg.dataset.num_classes)
    logger.info("Test class distribution (first 10 classes):")
    for i, (name, cnt) in enumerate(zip(class_names[:10], counts[:10])):
        logger.info("  [%3d] %-20s: %d", i, name, cnt)

    # 9. Optional visualisation
    if not args.no_vis:
        from cryptovit.utils.visualization import show_batch, plot_class_distribution
        aug = cfg.dataset.augmentation
        results_dir = Path(cfg.paths.project_root) / cfg.paths.results_dir / "phase1"

        show_batch(
            train_loader,
            class_names=class_names,
            mean=list(aug.mean),
            std=list(aug.std),
            n_images=16,
            title=f"Phase 1 – {cfg.dataset.name.upper()} training samples",
            save_path=results_dir / "sample_batch.png",
        )
        plot_class_distribution(
            all_labels,
            class_names=class_names,
            title=f"Phase 1 – {cfg.dataset.name.upper()} test class distribution",
            save_path=results_dir / "class_distribution.png",
        )
        logger.info("Visualisations saved to %s", results_dir)

    logger.info("=" * 60)
    logger.info("Phase 1 PASSED — Dataset pipeline is working correctly.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
