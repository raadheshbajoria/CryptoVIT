"""
cryptovit/datasets/cifar_dataset.py
====================================
CIFAR-10 and CIFAR-100 dataset management for CryptoViT.

Design principles
-----------------
* Pure PyTorch / torchvision — no Hugging Face dependency.
* Deterministic train/val split derived from the fixed seed.
* ``get_dataloaders`` is the single public entry point used by all
  training and evaluation scripts.
* Transforms are returned separately so the encrypted-dataset pipeline
  (Phase 9) can reuse them without re-downloading.

Normalisation statistics
------------------------
CIFAR-10  : mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
CIFAR-100 : mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
Both are per-channel values computed on the respective training sets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-dataset normalisation statistics
# ---------------------------------------------------------------------------
_STATS: Dict[str, Dict[str, List[float]]] = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
}

_TORCHVISION_CLS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def build_train_transform(cfg: DictConfig) -> transforms.Compose:
    """Return the training-time augmentation pipeline.

    Parameters
    ----------
    cfg:
        Full project config (uses ``cfg.dataset`` and ``cfg.dataset.augmentation``).

    Returns
    -------
    transforms.Compose
    """
    aug = cfg.dataset.augmentation
    img_size: int = cfg.dataset.image_size
    dataset_name: str = cfg.dataset.name

    stats = _get_stats(cfg)
    pipeline: List = []

    # 1. Resize to ViT input resolution
    pipeline.append(transforms.Resize((img_size, img_size)))

    # 2. Optional random crop (with padding)
    if aug.random_crop:
        pipeline.append(
            transforms.RandomCrop(img_size, padding=aug.crop_padding)
        )

    # 3. Optional horizontal flip
    if aug.random_horizontal_flip:
        pipeline.append(transforms.RandomHorizontalFlip())

    # 4. Optional colour jitter
    if aug.color_jitter:
        s = aug.color_jitter_strength
        pipeline.append(
            transforms.ColorJitter(
                brightness=s, contrast=s, saturation=s, hue=s * 0.5
            )
        )

    # 5. To tensor
    pipeline.append(transforms.ToTensor())

    # 6. Optional random erasing (applied *after* ToTensor)
    if aug.random_erasing:
        pipeline.append(transforms.RandomErasing(p=0.25))

    # 7. Normalise
    if aug.normalize:
        pipeline.append(transforms.Normalize(mean=stats["mean"], std=stats["std"]))

    return transforms.Compose(pipeline)


def build_eval_transform(cfg: DictConfig) -> transforms.Compose:
    """Return the deterministic evaluation transform (no augmentation).

    Parameters
    ----------
    cfg:
        Full project config.

    Returns
    -------
    transforms.Compose
    """
    aug = cfg.dataset.augmentation
    img_size: int = cfg.dataset.image_size
    stats = _get_stats(cfg)

    pipeline: List = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if aug.normalize:
        pipeline.append(transforms.Normalize(mean=stats["mean"], std=stats["std"]))

    return transforms.Compose(pipeline)


def _get_stats(cfg: DictConfig) -> Dict[str, List[float]]:
    """Return normalisation statistics, preferring config overrides."""
    dataset_name = cfg.dataset.name
    aug = cfg.dataset.augmentation
    # Config-specified values take precedence
    if hasattr(aug, "mean") and aug.mean and hasattr(aug, "std") and aug.std:
        return {"mean": list(aug.mean), "std": list(aug.std)}
    return _STATS[dataset_name]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_raw_dataset(
    dataset_name: str,
    root: Path,
    train: bool,
    transform: transforms.Compose,
    download: bool = True,
) -> Dataset:
    """Instantiate a torchvision CIFAR dataset."""
    if dataset_name not in _TORCHVISION_CLS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from {list(_TORCHVISION_CLS)}."
        )
    cls = _TORCHVISION_CLS[dataset_name]
    return cls(root=str(root), train=train, transform=transform, download=download)


def get_dataloaders(
    cfg: DictConfig,
    data_root: Optional[Path] = None,
    download: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    The *training* split is obtained by performing a random split on the
    official 50 000-image CIFAR training set using ``cfg.dataset.train_split``.
    The *test* set uses the official 10 000-image CIFAR test split.

    Parameters
    ----------
    cfg:
        Project configuration (OmegaConf DictConfig).
    data_root:
        Directory where the raw dataset is stored / downloaded.
        Defaults to ``cfg.paths.raw_dir`` resolved relative to the project root.
    download:
        Download the dataset if not present.
    seed:
        RNG seed for the train/val split.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    dataset_name: str = cfg.dataset.name
    batch_train: int = cfg.training.batch_size
    batch_eval: int = cfg.evaluation.batch_size
    num_workers: int = cfg.dataset.num_workers
    pin_memory: bool = cfg.dataset.pin_memory
    train_split: float = cfg.dataset.train_split

    # Resolve data root
    if data_root is None:
        data_root = Path(cfg.paths.project_root) / cfg.paths.raw_dir
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    # --- Full training dataset (with augmentation) ---
    full_train: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=True,
        transform=train_transform, download=download
    )

    # --- Full training dataset (eval transform, used for val split) ---
    full_train_eval: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=True,
        transform=eval_transform, download=False
    )

    # --- Test dataset ---
    test_dataset: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=False,
        transform=eval_transform, download=False
    )

    # --- Deterministic train/val split ---
    n_total = len(full_train)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    # Use the index lists to create Subsets on the *correct* transform datasets
    train_dataset = Subset(full_train, list(train_indices))
    val_dataset = Subset(full_train_eval, list(val_indices))

    logger.info(
        "Dataset '%s' | train=%d  val=%d  test=%d",
        dataset_name, len(train_dataset), len(val_dataset), len(test_dataset)
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
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

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Convenience: dataset statistics verification
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    loader: DataLoader,
    n_batches: int = 100,
) -> Tuple[List[float], List[float]]:
    """Empirically compute per-channel mean and std from a DataLoader.

    Useful to verify that normalisation is applied correctly and to
    compute stats for new datasets.

    Parameters
    ----------
    loader:
        DataLoader whose images are ``(B, C, H, W)`` float tensors.
    n_batches:
        Cap on the number of batches used (for speed).

    Returns
    -------
    (mean_per_channel, std_per_channel)
    """
    mean = torch.zeros(3)
    var = torch.zeros(3)
    n_pixels = 0

    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        b, c, h, w = images.shape
        pixels = b * h * w
        mean += images.sum(dim=(0, 2, 3))
        var += (images ** 2).sum(dim=(0, 2, 3))
        n_pixels += pixels

    mean /= n_pixels
    std = ((var / n_pixels) - mean ** 2).sqrt()

    logger.info("Computed mean: %s", mean.tolist())
    logger.info("Computed std:  %s", std.tolist())
    return mean.tolist(), std.tolist()


def get_class_names(dataset_name: str) -> List[str]:
    """Return ordered class names for CIFAR-10 or CIFAR-100."""
    if dataset_name == "cifar10":
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    elif dataset_name == "cifar100":
        # 100 fine-grained classes in alphabetical order as stored by torchvision
        return [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
            "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
            "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
            "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
            "flatfish", "forest", "fox", "girl", "hamster", "house",
            "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
            "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
            "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate",
            "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray",
            "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
            "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
            "streetcar", "sunflower", "sweet_pepper", "table", "tank",
            "telephone", "television", "tiger", "tractor", "train", "trout",
            "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
            "woman", "worm",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
