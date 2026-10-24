#!/usr/bin/env python
"""
scripts/generate_encrypted_data.py
=====================================
Phase 7 — Pre-generate encrypted CIFAR-10/100 dataset and save to disk.

This script encrypts every image in the training and test splits and
saves them as NumPy arrays so that subsequent training runs can load
encrypted images without repeating the encryption overhead.

Usage
-----
    python scripts/generate_encrypted_data.py
    python scripts/generate_encrypted_data.py dataset.name=cifar100

The encrypted dataset is saved to:
    data/encrypted/<dataset_name>_<key_hash>/
        train/images.npy  train/labels.npy
        val/images.npy    val/labels.npy
        test/images.npy   test/labels.npy
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from torchvision import datasets

from cryptovit.datasets.encrypted_dataset import generate_encrypted_dataset
from cryptovit.utils.config import load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger(__name__)

# The encryption key — in a real system this would be a secure random key
# stored in a key management system, NOT hard-coded.
DEFAULT_ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_config(overrides=overrides if overrides else None)
    setup_logging(level=cfg.logging.level)
    set_seed(cfg.project.seed)

    dataset_name = cfg.dataset.name
    key = DEFAULT_ENCRYPTION_KEY
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]

    data_root = PROJECT_ROOT / cfg.paths.raw_dir
    enc_root  = PROJECT_ROOT / "data" / "encrypted" / f"{dataset_name}_{key_hash}"

    logger.info("Dataset    : %s", dataset_name)
    logger.info("Key hash   : %s…", key_hash)
    logger.info("Output dir : %s", enc_root)

    cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    # Train split
    train_ds = cls(root=str(data_root), train=True, transform=None, download=True)
    logger.info("Encrypting TRAIN set (%d images) …", len(train_ds))
    generate_encrypted_dataset(train_ds, key, enc_root / "train")

    # Test split
    test_ds = cls(root=str(data_root), train=False, transform=None, download=True)
    logger.info("Encrypting TEST set (%d images) …", len(test_ds))
    generate_encrypted_dataset(test_ds, key, enc_root / "test")

    logger.info("Done! Encrypted dataset saved to %s", enc_root)


if __name__ == "__main__":
    main()
