"""
cryptovit/datasets/encrypted_dataset.py
=========================================
Encrypted CIFAR dataset for CryptoViT Phases 7–11.

Two modes of operation
----------------------
1. **On-the-fly encryption** (``mode="online"``)
   The dataset encrypts images at load time.  Useful for development and
   debugging.  Slow for full training due to Python-level encryption loop.

2. **Pre-generated encrypted dataset** (``mode="precomputed"``)
   Encrypted images are generated once and saved to disk as NumPy arrays.
   Subsequent runs load from disk.  Recommended for actual training runs.

Usage
-----
    from cryptovit.datasets.encrypted_dataset import EncryptedCIFARDataset

    # On-the-fly (no disk I/O)
    ds = EncryptedCIFARDataset(
        base_dataset=base_cifar10_dataset,
        key="my_secret_key",
        mode="online",
        transform=eval_transform,
    )

    # Pre-generated (fast)
    ds = EncryptedCIFARDataset(
        base_dataset=base_cifar10_dataset,
        key="my_secret_key",
        mode="precomputed",
        encrypted_dir=Path("data/encrypted/cifar10_key1"),
        transform=eval_transform,
    )
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from cryptovit.encryption import encrypt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EncryptedCIFARDataset
# ---------------------------------------------------------------------------

class EncryptedCIFARDataset(Dataset):
    """Wraps a CIFAR torchvision dataset and serves encrypted images.

    Parameters
    ----------
    base_dataset : Dataset
        An un-transformed (or minimally-transformed) CIFAR dataset.
        Images should be returned as PIL Images or uint8 tensors/arrays.
    key : str
        Encryption key.
    mode : {"online", "precomputed"}
        Encryption strategy.
    encrypted_dir : Path, optional
        Path to the directory containing pre-generated encrypted images.
        Required when ``mode="precomputed"``.
    transform : callable, optional
        torchvision transform applied *after* the encrypted uint8 image
        is retrieved.  Should at minimum convert to tensor and normalise.
    transient_steps : int
        Warm-up steps for the chaotic system.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        key: str,
        mode: str = "online",
        encrypted_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
        transient_steps: int = 1000,
    ):
        assert mode in ("online", "precomputed"), (
            f"mode must be 'online' or 'precomputed', got '{mode}'"
        )
        self.base_dataset = base_dataset
        self.key = key
        self.mode = mode
        self.encrypted_dir = Path(encrypted_dir) if encrypted_dir else None
        self.transform = transform
        self.transient_steps = transient_steps

        if mode == "precomputed":
            if self.encrypted_dir is None:
                raise ValueError(
                    "encrypted_dir must be provided when mode='precomputed'"
                )
            self._encrypted_images, self._labels = self._load_precomputed()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.mode == "online":
            img_pil, label = self.base_dataset[idx]
            img_np = _pil_to_uint8(img_pil)
            enc_np, _ = encrypt(img_np, self.key, self.transient_steps)
        else:
            enc_np = self._encrypted_images[idx]
            label = int(self._labels[idx])

        # Convert to PIL for transforms (torchvision convention)
        from PIL import Image
        enc_pil = Image.fromarray(enc_np)

        if self.transform is not None:
            enc_tensor = self.transform(enc_pil)
        else:
            enc_tensor = transforms.ToTensor()(enc_pil)

        return enc_tensor, label

    # ------------------------------------------------------------------
    # Pre-generation
    # ------------------------------------------------------------------

    def _load_precomputed(self):
        """Load pre-generated encrypted images from disk."""
        images_path = self.encrypted_dir / "images.npy"
        labels_path = self.encrypted_dir / "labels.npy"
        if not images_path.exists():
            raise FileNotFoundError(
                f"Pre-generated encrypted dataset not found at {self.encrypted_dir}. "
                "Run scripts/generate_encrypted_data.py first."
            )
        images = np.load(str(images_path))
        labels = np.load(str(labels_path))
        logger.info("Loaded pre-generated encrypted dataset from %s (%d samples).",
                    self.encrypted_dir, len(images))
        return images, labels


# ---------------------------------------------------------------------------
# Dataset generation script helper
# ---------------------------------------------------------------------------

def generate_encrypted_dataset(
    base_dataset: Dataset,
    key: str,
    save_dir: Path,
    transient_steps: int = 1000,
) -> None:
    """Generate and save an encrypted version of a dataset.

    Encrypts each image in ``base_dataset`` and saves all encrypted images
    as a single ``.npy`` array.

    Parameters
    ----------
    base_dataset : Dataset
        CIFAR dataset (returns PIL images, not tensors).
    key : str
        Encryption key.
    save_dir : Path
        Directory to write ``images.npy`` and ``labels.npy``.
    transient_steps : int
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(base_dataset)
    logger.info("Generating encrypted dataset (%d samples) → %s", n, save_dir)

    # Get shape from first sample
    img0, _ = base_dataset[0]
    img0_np = _pil_to_uint8(img0)
    enc0, _ = encrypt(img0_np, key, transient_steps)

    images = np.empty((n, *enc0.shape), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)

    for i in tqdm(range(n), desc="Encrypting"):
        img_pil, label = base_dataset[i]
        img_np = _pil_to_uint8(img_pil)
        enc_np, _ = encrypt(img_np, key, transient_steps)
        images[i] = enc_np
        labels[i] = label

    np.save(str(save_dir / "images.npy"), images)
    np.save(str(save_dir / "labels.npy"), labels)
    logger.info("Saved encrypted dataset to %s.", save_dir)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pil_to_uint8(img) -> np.ndarray:
    """Convert PIL Image or tensor to HWC uint8 numpy array."""
    import numpy as np
    if hasattr(img, "numpy"):           # torch.Tensor
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return np.array(img, dtype=np.uint8)
