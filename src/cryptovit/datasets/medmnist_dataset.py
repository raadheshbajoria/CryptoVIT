"""
cryptovit/datasets/medmnist_dataset.py
========================================
MedMNIST (https://medmnist.com) loading for CryptoViT.

MedMNIST provides small, standardised medical-image classification datasets
with fixed official train / val / test splits. Only single-label classification
datasets are supported here (multi-label ChestMNIST and ordinal RetinaMNIST are
not). All images are converted to RGB so grayscale datasets (X-ray, OCT) work
with the 3-channel models.

MedMNIST is for research benchmarking and is not intended for clinical use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MEDMNIST_DATASETS: Dict[str, Tuple[str, int]] = {
    "dermamnist": ("DermaMNIST", 7),
    "pathmnist": ("PathMNIST", 9),
    "bloodmnist": ("BloodMNIST", 8),
    "pneumoniamnist": ("PneumoniaMNIST", 2),
    "octmnist": ("OCTMNIST", 4),
}


def is_medmnist(dataset_name: str) -> bool:
    return dataset_name in MEDMNIST_DATASETS


class MedMNISTAdapter(Dataset):
    """Uniform ``(PIL RGB image, int label)`` view of a MedMNIST split.

    MedMNIST returns labels as 1-element arrays and keeps grayscale images
    single-channel; this adapter flattens the label to ``int`` and converts the
    image to RGB so it behaves like the CIFAR datasets elsewhere in the code.

    Parameters
    ----------
    base:
        A ``medmnist`` dataset built *without* a transform, or any object with
        ``__len__``, ``__getitem__`` returning ``(PIL image, label array)`` and
        a ``labels`` array.
    transform:
        Optional torchvision transform applied to the RGB PIL image.
    """

    def __init__(self, base: Dataset, transform: Optional[Callable] = None):
        self.base = base
        self.transform = transform
        self.targets: List[int] = [int(x) for x in np.asarray(base.labels).ravel()]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, _ = self.base[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def _medmnist_class(dataset_name: str):
    if dataset_name not in MEDMNIST_DATASETS:
        raise ValueError(
            f"Unsupported MedMNIST dataset '{dataset_name}'. Choose from {list(MEDMNIST_DATASETS)}."
        )
    try:
        import medmnist
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("The 'medmnist' package is required: pip install medmnist") from exc
    return getattr(medmnist, MEDMNIST_DATASETS[dataset_name][0])


def medmnist_class_names(dataset_name: str) -> List[str]:
    """Class names in label-index order, from the official MedMNIST metadata."""
    import medmnist

    labels = medmnist.INFO[dataset_name]["label"]
    return [labels[str(i)] for i in range(len(labels))]


def load_medmnist_splits(
    dataset_name: str,
    root: Path,
    size: int = 28,
    download: bool = True,
) -> Tuple[MedMNISTAdapter, MedMNISTAdapter, MedMNISTAdapter]:
    """Load the official train / val / test splits as transform-free adapters."""
    cls = _medmnist_class(dataset_name)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splits = [
        MedMNISTAdapter(cls(split=s, root=str(root), size=size, download=download, transform=None))
        for s in ("train", "val", "test")
    ]
    logger.info(
        "MedMNIST '%s' (%dpx) | train=%d  val=%d  test=%d",
        dataset_name,
        size,
        *(len(s) for s in splits),
    )
    return tuple(splits)  # type: ignore[return-value]


def get_medmnist_dataloaders(
    cfg: DictConfig,
    data_root: Optional[Path] = None,
    download: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test loaders using the *official* MedMNIST splits."""
    from cryptovit.datasets.cifar_dataset import (
        build_eval_transform,
        build_train_transform,
    )

    if data_root is None:
        data_root = Path(cfg.paths.project_root) / cfg.paths.raw_dir
    root = Path(data_root) / "medmnist"
    size = int(cfg.dataset.get("medmnist_size", 28))

    train_base, val_base, test_base = load_medmnist_splits(
        cfg.dataset.name, root, size=size, download=download
    )
    train_ds = MedMNISTAdapter(train_base.base, build_train_transform(cfg))
    eval_tf = build_eval_transform(cfg)
    val_ds = MedMNISTAdapter(val_base.base, eval_tf)
    test_ds = MedMNISTAdapter(test_base.base, eval_tf)

    loader_kwargs = dict(num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=cfg.dataset.num_workers > 0,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.evaluation.batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.evaluation.batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader


def compute_channel_stats(
    images: Sequence[Image.Image],
) -> Dict[str, List[float]]:
    """Per-channel mean/std (0-1 scale) of RGB PIL images."""
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.float64) for im in images])
    arr /= 255.0
    return {
        "mean": arr.mean(axis=(0, 1, 2)).tolist(),
        "std": arr.std(axis=(0, 1, 2)).tolist(),
    }
