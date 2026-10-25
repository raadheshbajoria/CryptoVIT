from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(
    base: str = "base.yaml",
    overrides: Optional[List[str]] = None,
    extra_files: Optional[List[Union[str, Path]]] = None,
) -> DictConfig:
    base_path = CONFIGS_DIR / base
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    overrides = list(overrides or [])
    cli_configs = [o.split("=", 1)[1] for o in overrides if o.startswith("config=")]
    overrides = [o for o in overrides if not o.startswith("config=")]
    extra_files = list(extra_files or []) + [
        c if c.endswith((".yaml", ".yml")) else f"{c}.yaml" for c in cli_configs
    ]

    cfg: DictConfig = OmegaConf.load(base_path)
    logger.debug("Loaded base config: %s", base_path)

    for extra in extra_files or []:
        extra_path = Path(extra) if Path(extra).is_absolute() else CONFIGS_DIR / extra
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra config not found: {extra_path}")
        extra_cfg = OmegaConf.load(extra_path)
        cfg = OmegaConf.merge(cfg, extra_cfg)
        logger.debug("Merged extra config: %s", extra_path)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        logger.debug("Applied CLI overrides: %s", overrides)

    OmegaConf.update(cfg, "paths.project_root", str(PROJECT_ROOT), merge=False)

    _validate_config(cfg)
    logger.info("Configuration loaded successfully.")
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    num_classes_map = {
        "cifar10": 10,
        "cifar100": 100,
        "dermamnist": 7,
        "pathmnist": 9,
        "bloodmnist": 8,
        "pneumoniamnist": 2,
        "octmnist": 4,
    }
    allowed_datasets = set(num_classes_map)
    if cfg.dataset.name not in allowed_datasets:
        raise ValueError(
            f"dataset.name must be one of {sorted(allowed_datasets)}, got '{cfg.dataset.name}'"
        )

    expected = num_classes_map[cfg.dataset.name]
    if cfg.dataset.num_classes != expected:
        raise ValueError(
            f"dataset.num_classes={cfg.dataset.num_classes} does not match "
            f"dataset.name='{cfg.dataset.name}' (expected {expected})."
        )

    if cfg.model.patch_size not in (8, 16, 32):
        raise ValueError(f"model.patch_size must be 8, 16, or 32; got {cfg.model.patch_size}")

    if cfg.model.image_size % cfg.model.patch_size != 0:
        raise ValueError(
            f"model.image_size ({cfg.model.image_size}) must be divisible by "
            f"model.patch_size ({cfg.model.patch_size})."
        )


def get_device(cfg: DictConfig):
    import torch

    preference = cfg.hardware.device
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    logger.info("Using device: %s", device)
    return device


def pretty_print(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
