"""
cryptovit.datasets
==================
Dataset management for CryptoViT.

Public API
----------
``get_dataloaders``            - Build train/val/test DataLoaders.
``build_train_transform``      - Training augmentation pipeline.
``build_eval_transform``       - Deterministic evaluation transform.
``compute_dataset_stats``      - Empirical per-channel mean/std.
``get_class_names``            - Ordered class label list.
``EncryptedCIFARDataset``      - Encrypted CIFAR dataset wrapper.
``generate_encrypted_dataset`` - Pre-generate encrypted dataset to disk.
"""

from cryptovit.datasets.cifar_dataset import (
    build_eval_transform,
    build_train_transform,
    compute_dataset_stats,
    get_class_names,
    get_dataloaders,
    get_split_indices,
)
from cryptovit.datasets.encrypted_dataset import (
    EncryptedCIFARDataset,
    build_encrypted_loaders,
    generate_encrypted_dataset,
)

__all__ = [
    "get_dataloaders",
    "get_split_indices",
    "build_encrypted_loaders",
    "build_train_transform",
    "build_eval_transform",
    "compute_dataset_stats",
    "get_class_names",
    "EncryptedCIFARDataset",
    "generate_encrypted_dataset",
]
