"""
cryptovit/datasets/encrypted_dataset.py
=========================================
Encrypted CIFAR dataset for CryptoViT Phases 7–11.

Two modes of operation
---------------------
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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from cryptovit.datasets.cifar_dataset import get_split_indices
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
        scheme: str = "v1",
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
        self.scheme = scheme

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
            enc_np, _ = encrypt(
                img_np, self.key, self.transient_steps, scheme=self.scheme
            )
        else:
            enc_np = self._encrypted_images[idx]
            label = int(self._labels[idx])

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
    scheme: str = "v1",
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
    enc0, _ = encrypt(img0_np, key, transient_steps, scheme=scheme)

    images = np.empty((n, *enc0.shape), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)

    for i in tqdm(range(n), desc="Encrypting"):
        img_pil, label = base_dataset[i]
        img_np = _pil_to_uint8(img_pil)
        enc_np, _ = encrypt(img_np, key, transient_steps, scheme=scheme)
        images[i] = enc_np
        labels[i] = label

    np.save(str(save_dir / "images.npy"), images)
    np.save(str(save_dir / "labels.npy"), labels)
    logger.info("Saved encrypted dataset to %s.", save_dir)


# ---------------------------------------------------------------------------
# Encrypted data loaders (shared split, no test-set leakage)
# ---------------------------------------------------------------------------

def compute_encrypted_stats(
    base_dataset: Dataset,
    key: str,
    indices: Sequence[int],
    n_samples: int = 2000,
    transient_steps: int = 1000,
    scheme: str = "v1",
) -> Dict[str, List[float]]:
    """Per-channel mean/std (0-1 scale) of encrypted images.

    Encrypted images are close to uniform noise, so the plaintext CIFAR
    statistics do not apply. Statistics must come from training images only.
    """
    chosen = list(indices)[:n_samples]
    if not chosen:
        raise ValueError("indices must contain at least one sample.")
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for i in chosen:
        img, _ = base_dataset[i]
        enc, _ = encrypt(_pil_to_uint8(img), key, transient_steps, scheme=scheme)
        px = enc.reshape(-1, enc.shape[-1]).astype(np.float64) / 255.0
        total += px.sum(axis=0)
        total_sq += (px ** 2).sum(axis=0)
        count += px.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist()}


def build_encrypted_transform(
    image_size: int, mean: Sequence[float], std: Sequence[float]
) -> transforms.Compose:
    """Deterministic transform for encrypted images.

    No random crop or flip: those operations on ciphertext do not correspond
    to any plaintext augmentation and only inject noise.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def build_encrypted_loaders(
    cfg: DictConfig,
    base_train: Dataset,
    base_test: Dataset,
    key: str,
    seed: int,
    base_val: Optional[Dataset] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train/val/test loaders over encrypted images.

    * Without ``base_val``, train and validation use the same index split as
      the plaintext ``get_dataloaders`` (``get_split_indices`` with the same
      seed), so plaintext and encrypted experiments see exactly the same
      images. With ``base_val`` (datasets that ship official splits, such as
      MedMNIST) ``base_train`` is used whole and ``base_val`` is the
      validation set.
    * The test set is only the official test split; it is never used for
      checkpoint selection or early stopping.
    * Normalisation statistics are measured on training images only.

    ``base_train`` / ``base_test`` return un-transformed PIL images (or uint8
    arrays) and labels.
    """
    if base_val is None:
        train_idx, val_idx = get_split_indices(
            len(base_train), cfg.dataset.train_split, seed
        )
    else:
        train_idx, val_idx = list(range(len(base_train))), None
    scheme = cfg.encryption.get("scheme", "v1")
    stats = compute_encrypted_stats(
        base_train, key, train_idx,
        transient_steps=cfg.encryption.transient_steps, scheme=scheme,
    )
    transform = build_encrypted_transform(
        cfg.dataset.image_size, stats["mean"], stats["std"]
    )
    kwargs = dict(
        key=key, mode="online", transform=transform,
        transient_steps=cfg.encryption.transient_steps, scheme=scheme,
    )
    enc_train = EncryptedCIFARDataset(base_train, **kwargs)
    enc_test = EncryptedCIFARDataset(base_test, **kwargs)

    loader_kwargs = dict(
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory
    )
    train_loader = DataLoader(
        Subset(enc_train, train_idx),
        batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        # Keep workers alive between epochs (Windows re-spawns them otherwise).
        persistent_workers=cfg.dataset.num_workers > 0, **loader_kwargs,
    )
    val_dataset = (
        Subset(enc_train, val_idx) if base_val is None
        else EncryptedCIFARDataset(base_val, **kwargs)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        enc_test, batch_size=cfg.evaluation.batch_size, shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pil_to_uint8(img) -> np.ndarray:
    """Convert PIL Image or tensor to HWC uint8 numpy array."""
    if hasattr(img, "numpy"):           # torch.Tensor
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return np.array(img, dtype=np.uint8)
