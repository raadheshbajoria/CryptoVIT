from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from cryptovit.models.cryptovit import CryptoViT, build_cryptovit
from cryptovit.models.patch_embedding import CryptoPatchEmbedding, StandardPatchEmbedding
from cryptovit.models.positional_encoding import (
    KeyAwarePositionalEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from cryptovit.models.vit import VisionTransformer

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

_VIT_PRESETS = {
    "vit_tiny": dict(embed_dim=96, depth=4, num_heads=3),
    "vit_small": dict(embed_dim=192, depth=6, num_heads=3),
    "vit_base": dict(embed_dim=384, depth=8, num_heads=6),
}


def build_model(cfg: DictConfig) -> "nn.Module":
    name: str = cfg.model.name.lower()

    if name in _VIT_PRESETS:
        preset = _VIT_PRESETS[name]
        embed_dim = cfg.model.get("embed_dim", preset["embed_dim"])
        depth = cfg.model.get("depth", preset["depth"])
        num_heads = cfg.model.get("num_heads", preset["num_heads"])

        model = VisionTransformer(
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=cfg.model.get("mlp_ratio", 4.0),
            dropout=cfg.model.get("dropout", 0.1),
            attn_dropout=cfg.model.get("attn_dropout", 0.0),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.1),
            pool=cfg.model.get("pool", "cls"),
        )
        logger.info("Built model '%s' with %.2fM parameters.", name, model.num_parameters() / 1e6)
        return model

    if name == "cryptovit":
        return build_cryptovit(cfg)

    raise ValueError(
        f"Unknown model name '{name}'. Choose from {list(_VIT_PRESETS)} or 'cryptovit'."
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
