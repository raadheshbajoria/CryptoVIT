from __future__ import annotations

import pytest
from omegaconf import DictConfig


class TestLoadConfig:
    def test_returns_dictconfig(self, base_cfg):
        assert isinstance(base_cfg, DictConfig)

    def test_required_sections_present(self, base_cfg):
        for section in ("project", "paths", "dataset", "model", "training", "evaluation"):
            assert hasattr(base_cfg, section), f"Missing section: {section}"

    def test_default_dataset_is_cifar10(self, base_cfg):
        assert base_cfg.dataset.name == "cifar10"
        assert base_cfg.dataset.num_classes == 10

    def test_cifar100_override(self, cifar100_cfg):
        assert cifar100_cfg.dataset.name == "cifar100"
        assert cifar100_cfg.dataset.num_classes == 100

    def test_dot_notation_override(self):
        from cryptovit.utils import load_config

        cfg = load_config(overrides=["training.batch_size=32"])
        assert cfg.training.batch_size == 32

    def test_image_divisible_by_patch(self, base_cfg):
        assert base_cfg.model.image_size % base_cfg.model.patch_size == 0

    def test_invalid_dataset_name_raises(self):
        from cryptovit.utils import load_config

        with pytest.raises(ValueError, match="dataset.name"):
            load_config(overrides=["dataset.name=imagenet"])

    def test_project_root_injected(self, base_cfg):
        assert "project_root" in base_cfg.paths
        from pathlib import Path

        root = Path(base_cfg.paths.project_root)
        assert root.is_absolute()
        assert (root / "configs" / "base.yaml").exists()

    def test_mismatched_num_classes_raises(self):
        from cryptovit.utils import load_config

        with pytest.raises(ValueError, match="num_classes"):
            load_config(overrides=["dataset.name=cifar10", "dataset.num_classes=100"])
