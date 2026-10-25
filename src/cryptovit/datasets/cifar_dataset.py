from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

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


def build_train_transform(cfg: DictConfig) -> transforms.Compose:
    aug = cfg.dataset.augmentation
    img_size: int = cfg.dataset.image_size
    stats = _get_stats(cfg)
    pipeline: List = []

    pipeline.append(transforms.Resize((img_size, img_size)))

    if aug.random_crop:
        pipeline.append(transforms.RandomCrop(img_size, padding=aug.crop_padding))

    if aug.random_horizontal_flip:
        pipeline.append(transforms.RandomHorizontalFlip())

    if aug.color_jitter:
        s = aug.color_jitter_strength
        pipeline.append(transforms.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s * 0.5))

    pipeline.append(transforms.ToTensor())

    if aug.random_erasing:
        pipeline.append(transforms.RandomErasing(p=0.25))

    if aug.normalize:
        pipeline.append(transforms.Normalize(mean=stats["mean"], std=stats["std"]))

    return transforms.Compose(pipeline)


def build_eval_transform(cfg: DictConfig) -> transforms.Compose:
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
    dataset_name = cfg.dataset.name
    aug = cfg.dataset.augmentation
    if hasattr(aug, "mean") and aug.mean and hasattr(aug, "std") and aug.std:
        return {"mean": list(aug.mean), "std": list(aug.std)}
    return _STATS[dataset_name]


def _load_raw_dataset(
    dataset_name: str,
    root: Path,
    train: bool,
    transform: transforms.Compose,
    download: bool = True,
) -> Dataset:
    if dataset_name not in _TORCHVISION_CLS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose from {list(_TORCHVISION_CLS)}."
        )
    cls = _TORCHVISION_CLS[dataset_name]
    return cls(root=str(root), train=train, transform=transform, download=download)


def get_dataset_targets(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent = get_dataset_targets(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    if hasattr(dataset, "base_dataset"):
        return get_dataset_targets(dataset.base_dataset)
    if hasattr(dataset, "targets"):
        return [int(t) for t in dataset.targets]
    raise TypeError(
        f"Cannot read labels from {type(dataset).__name__}; it needs a "
        "'targets' attribute, a 'base_dataset', or to be a Subset of one."
    )


def get_split_indices(n_total: int, train_split: float, seed: int) -> Tuple[List[int], List[int]]:
    n_train = int(n_total * train_split)
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_total - n_train], generator=generator
    )
    return list(train_idx), list(val_idx)


def get_dataloaders(
    cfg: DictConfig,
    data_root: Optional[Path] = None,
    download: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_name: str = cfg.dataset.name

    from cryptovit.datasets.medmnist_dataset import (
        get_medmnist_dataloaders,
        is_medmnist,
    )

    if is_medmnist(dataset_name):
        return get_medmnist_dataloaders(cfg, data_root, download, seed)

    batch_train: int = cfg.training.batch_size
    batch_eval: int = cfg.evaluation.batch_size
    num_workers: int = cfg.dataset.num_workers
    pin_memory: bool = cfg.dataset.pin_memory
    train_split: float = cfg.dataset.train_split

    if data_root is None:
        data_root = Path(cfg.paths.project_root) / cfg.paths.raw_dir
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    full_train: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=True, transform=train_transform, download=download
    )

    full_train_eval: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=True, transform=eval_transform, download=False
    )

    test_dataset: Dataset = _load_raw_dataset(
        dataset_name, data_root, train=False, transform=eval_transform, download=False
    )

    train_indices, val_indices = get_split_indices(len(full_train), train_split, seed)
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    logger.info(
        "Dataset '%s' | train=%d  val=%d  test=%d",
        dataset_name,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=num_workers > 0,
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


def compute_dataset_stats(
    loader: DataLoader,
    n_batches: int = 100,
) -> Tuple[List[float], List[float]]:
    mean = torch.zeros(3)
    var = torch.zeros(3)
    n_pixels = 0

    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        b, c, h, w = images.shape
        pixels = b * h * w
        mean += images.sum(dim=(0, 2, 3))
        var += (images**2).sum(dim=(0, 2, 3))
        n_pixels += pixels

    mean /= n_pixels
    std = ((var / n_pixels) - mean**2).sqrt()

    logger.info("Computed mean: %s", mean.tolist())
    logger.info("Computed std:  %s", std.tolist())
    return mean.tolist(), std.tolist()


def get_class_names(dataset_name: str) -> List[str]:
    from cryptovit.datasets.medmnist_dataset import (
        is_medmnist,
        medmnist_class_names,
    )

    if is_medmnist(dataset_name):
        return medmnist_class_names(dataset_name)
    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif dataset_name == "cifar100":
        return [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
