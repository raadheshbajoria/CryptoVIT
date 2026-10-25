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
    except ImportError as exc:
        raise ImportError("The 'medmnist' package is required: pip install medmnist") from exc
    return getattr(medmnist, MEDMNIST_DATASETS[dataset_name][0])


def medmnist_class_names(dataset_name: str) -> List[str]:
    import medmnist

    labels = medmnist.INFO[dataset_name]["label"]
    return [labels[str(i)] for i in range(len(labels))]


def load_medmnist_splits(
    dataset_name: str,
    root: Path,
    size: int = 28,
    download: bool = True,
) -> Tuple[MedMNISTAdapter, MedMNISTAdapter, MedMNISTAdapter]:
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
    return tuple(splits)


def get_medmnist_dataloaders(
    cfg: DictConfig,
    data_root: Optional[Path] = None,
    download: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.float64) for im in images])
    arr /= 255.0
    return {
        "mean": arr.mean(axis=(0, 1, 2)).tolist(),
        "std": arr.std(axis=(0, 1, 2)).tolist(),
    }
