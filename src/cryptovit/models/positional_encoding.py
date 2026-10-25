from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pos_embed)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(num_patches + 1, embed_dim)
        position = torch.arange(num_patches + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe)


class KeyAwarePositionalEncoding(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.0,
        key_embed_dim: int = 256,
    ):
        super().__init__()
        self.num_tokens = num_patches + 1
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.key_mlp = nn.Sequential(
            nn.Linear(key_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.num_tokens * embed_dim),
        )
        self.drop = nn.Dropout(dropout)

        logger.info(
            "KeyAwarePositionalEncoding | tokens=%d  embed=%d  key_embed=%d",
            self.num_tokens,
            embed_dim,
            key_embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_embedding is None:
            return self.drop(x + self.pos_embed)

        B = x.shape[0]
        if key_embedding.dim() == 1:
            key_embedding = key_embedding.unsqueeze(0)
        if key_embedding.shape[0] == 1 and B > 1:
            key_embedding = key_embedding.expand(B, -1)

        offset = self.key_mlp(key_embedding)
        offset = offset.view(B, self.num_tokens, self.embed_dim)
        offset = offset * 0.1

        return self.drop(x + self.pos_embed + offset)

    @staticmethod
    def key_to_embedding(key: str | bytes, embed_dim: int = 256) -> torch.Tensor:
        if isinstance(key, str):
            key = key.encode("utf-8")
        digest = hashlib.sha256(key).digest()
        n_repeats = math.ceil(embed_dim / 32)
        extended = (digest * n_repeats)[:embed_dim]
        values = [b / 127.5 - 1.0 for b in extended]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)
