"""
tests/unit/test_visualization.py
================================
Unit tests for visualization and denormalization utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cryptovit.utils.visualization import denormalize, plot_class_distribution, show_batch


class TestVisualization:
    def test_denormalize_3d_tensor(self):
        t = torch.zeros(3, 8, 8)
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        out = denormalize(t, mean, std)
        assert out.shape == (3, 8, 8)
        assert torch.allclose(out, torch.full((3, 8, 8), 0.5))

    def test_denormalize_4d_tensor(self):
        t = torch.zeros(2, 3, 8, 8)
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        out = denormalize(t, mean, std)
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out, torch.full((2, 3, 8, 8), 0.5))

    def test_denormalize_invalid_ndim_raises(self):
        t = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            denormalize(t, [0.5], [0.2])

    def test_show_batch_saves_file(self, tmp_path):
        x = torch.rand(4, 3, 16, 16)
        y = torch.tensor([0, 1, 0, 1])
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)
        save_file = tmp_path / "batch.png"

        show_batch(
            loader=loader,
            class_names=["classA", "classB"],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            n_images=4,
            ncols=2,
            save_path=save_file,
        )
        assert save_file.exists()

    def test_plot_class_distribution_saves_file(self, tmp_path):
        labels = [0, 0, 1, 1, 1]
        save_file = tmp_path / "dist.png"

        plot_class_distribution(
            labels=labels,
            class_names=["classA", "classB"],
            save_path=save_file,
        )
        assert save_file.exists()
