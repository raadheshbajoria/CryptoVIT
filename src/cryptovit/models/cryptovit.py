from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cryptovit.models.patch_embedding import CryptoPatchEmbedding
from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding
from cryptovit.models.vit import TransformerBlock

logger = logging.getLogger(__name__)


class CryptoViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "cls",
        hidden_dim: int = 64,
        key_embed_dim: int = 256,
    ):
        super().__init__()
        assert pool in ("cls", "mean")
        self.pool = pool
        self.embed_dim = embed_dim

        self.patch_embed = CryptoPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_encoding = KeyAwarePositionalEncoding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=dropout,
            key_embed_dim=key_embed_dim,
        )
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        logger.info(
            "CryptoViT | image=%d patch=%d embed=%d depth=%d heads=%d params=%.2fM",
            image_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            self.num_parameters() / 1e6,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_encoding(x, key_embedding=key_embedding)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x[:, 0] if self.pool == "cls" else x[:, 1:].mean(dim=1)

        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_features(
        self,
        x: torch.Tensor,
        key_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x, key_embedding=key_embedding)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0] if self.pool == "cls" else x[:, 1:].mean(dim=1)


def build_cryptovit(cfg: DictConfig) -> CryptoViT:
    m = cfg.model
    model = CryptoViT(
        image_size=m.image_size,
        patch_size=m.patch_size,
        in_channels=m.in_channels,
        num_classes=m.num_classes,
        embed_dim=m.get("embed_dim", 192),
        depth=m.get("depth", 6),
        num_heads=m.get("num_heads", 3),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        dropout=m.get("dropout", 0.1),
        attn_dropout=m.get("attn_dropout", 0.0),
        drop_path_rate=m.get("drop_path_rate", 0.1),
        pool=m.get("pool", "cls"),
        hidden_dim=m.get("hidden_dim", 64),
        key_embed_dim=m.get("key_embed_dim", 256),
    )
    logger.info("Built CryptoViT with %.2fM parameters.", model.num_parameters() / 1e6)
    return model
