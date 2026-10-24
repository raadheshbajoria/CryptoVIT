"""
cryptovit/utils/visualization.py
==================================
Lightweight visualisation utilities for dataset inspection and analysis.

Used in Phase 1 for sanity-checking the dataset pipeline and in later
phases for visualising encrypted images, confusion matrices, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Reverse channel-wise normalisation.

    Parameters
    ----------
    tensor:
        Float tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
    mean, std:
        Per-channel statistics used during normalisation.

    Returns
    -------
    torch.Tensor
        Denormalised tensor clamped to [0, 1].
    """
    t = tensor.clone()
    mean_t = torch.tensor(mean, dtype=t.dtype)
    std_t = torch.tensor(std, dtype=t.dtype)

    if t.ndim == 3:  # (C, H, W)
        for c in range(t.shape[0]):
            t[c] = t[c] * std_t[c] + mean_t[c]
    elif t.ndim == 4:  # (B, C, H, W)
        for c in range(t.shape[1]):
            t[:, c] = t[:, c] * std_t[c] + mean_t[c]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t.ndim}D.")

    return t.clamp(0.0, 1.0)


def show_batch(
    loader: DataLoader,
    class_names: List[str],
    mean: Sequence[float],
    std: Sequence[float],
    n_images: int = 16,
    ncols: int = 8,
    title: str = "Sample batch",
    save_path: Optional[Path] = None,
) -> None:
    """Display a grid of images from a DataLoader.

    Parameters
    ----------
    loader:
        DataLoader yielding ``(images, labels)`` tuples.
    class_names:
        Ordered list of class label strings.
    mean, std:
        Normalisation statistics for denormalisation.
    n_images:
        Number of images to display.
    ncols:
        Number of columns in the grid.
    title:
        Figure title.
    save_path:
        If provided, save the figure to this path instead of showing it.
    """
    images, labels = next(iter(loader))
    images = images[:n_images]
    labels = labels[:n_images]

    images_dn = denormalize(images, mean, std)

    nrows = (n_images + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8))
    axes = np.array(axes).flatten()

    for i, (img, lbl) in enumerate(zip(images_dn, labels)):
        ax = axes[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(class_names[int(lbl)], fontsize=7)
        ax.axis("off")

    # Hide unused subplots
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_class_distribution(
    labels: Sequence[int],
    class_names: List[str],
    title: str = "Class distribution",
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart of label frequencies.

    Parameters
    ----------
    labels:
        Flat sequence (or array) of integer class labels.
    class_names:
        Ordered list of class label strings.
    title:
        Figure title.
    save_path:
        If provided, save instead of showing.
    """
    counts = np.bincount(np.asarray(labels), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.4), 4))
    ax.bar(range(len(class_names)), counts)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
