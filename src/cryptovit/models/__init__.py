"""
cryptovit/models/__init__.py
============================
Model factory for CryptoViT.

Provides a single ``build_model(cfg)`` entry-point that constructs the
correct model variant based on the configuration key ``cfg.model.name``.

Supported names
---------------
* ``vit_tiny``   — embed_dim=96,  depth=4, num_heads=3
* ``vit_small``  — embed_dim=192, depth=6, num_heads=3  (default)
* ``vit_base``   — embed_dim=384, depth=8, num_heads=6
* ``cryptovit``  — full CryptoViT (CryptoPatchEmbedding + KeyAwarePE)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from cryptovit.models.vit import VisionTransformer
from cryptovit.models.cryptovit import CryptoViT, build_cryptovit
from cryptovit.models.patch_embedding import CryptoPatchEmbedding, StandardPatchEmbedding
from cryptovit.models.positional_encoding import (
    KeyAwarePositionalEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Preset hyper-parameters for standard ViT sizes
_VIT_PRESETS = {
    "vit_tiny": dict(embed_dim=96,  depth=4, num_heads=3),
    "vit_small": dict(embed_dim=192, depth=6, num_heads=3),
    "vit_base": dict(embed_dim=384, depth=8, num_heads=6),
}


def build_model(cfg: DictConfig) -> "nn.Module":
    """Construct the model specified by ``cfg.model.name``.

    Parameters
    ----------
    cfg:
        Full project configuration (OmegaConf DictConfig).

    Returns
    -------
    nn.Module
        Instantiated model.
    """
    name: str = cfg.model.name.lower()

    if name in _VIT_PRESETS:
        # Allow config to override preset values
        preset = _VIT_PRESETS[name]
        embed_dim  = cfg.model.get("embed_dim",  preset["embed_dim"])
        depth      = cfg.model.get("depth",      preset["depth"])
        num_heads  = cfg.model.get("num_heads",  preset["num_heads"])

        model = VisionTransformer(
            image_size     = cfg.model.image_size,
            patch_size     = cfg.model.patch_size,
            in_channels    = cfg.model.in_channels,
            num_classes    = cfg.model.num_classes,
            embed_dim      = embed_dim,
            depth          = depth,
            num_heads      = num_heads,
            mlp_ratio      = cfg.model.get("mlp_ratio",   4.0),
            dropout        = cfg.model.get("dropout",     0.1),
            attn_dropout   = cfg.model.get("attn_dropout", 0.0),
            drop_path_rate = cfg.model.get("drop_path_rate", 0.1),
            pool           = cfg.model.get("pool",        "cls"),
        )
        logger.info("Built model '%s' with %.2fM parameters.", name, model.num_parameters() / 1e6)
        return model

    if name == "cryptovit":
        return build_cryptovit(cfg)

    raise ValueError(
        f"Unknown model name '{name}'. "
        f"Choose from {list(_VIT_PRESETS)} or 'cryptovit'."
    )


__all__ = [
    "build_model",
    "VisionTransformer",
    "CryptoViT",
    "build_cryptovit",
    "CryptoPatchEmbedding",
    "StandardPatchEmbedding",
    "KeyAwarePositionalEncoding",
    "LearnedPositionalEncoding",
    "SinusoidalPositionalEncoding",
]
