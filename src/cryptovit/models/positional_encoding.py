"""
cryptovit/models/positional_encoding.py
=========================================
Positional encoding variants for CryptoViT.

Background
----------
Standard ViT uses learnable 1-D positional embeddings added to token
embeddings before the transformer encoder.  The positional embedding
encodes the spatial position of each patch.

For encrypted images the spatial positions have been permuted by the
encryption.  The ``KeyAwarePositionalEncoding`` incorporates information
derived from the encryption key to modulate the positional signal -
making the transformer implicitly "key-aware" even without explicit
key injection into every layer.

Variants
--------
* ``LearnedPositionalEncoding``   - standard ViT 1-D learnable PE.
* ``SinusoidalPositionalEncoding`` - fixed sinusoidal PE (used as baseline).
* ``KeyAwarePositionalEncoding``   - learnable PE + key-derived additive offset.

All variants include the [CLS] token position (index 0).

Design note on KeyAwarePositionalEncoding
-----------------------------------------
The key-derived offset is computed as:
    offset = MLP(key_embedding)   shape: (1, num_patches + 1, embed_dim)

where ``key_embedding`` is a fixed (per-call) vector derived from the
SHA-256 hash of the key.  The MLP is small (256 -> embed_dim) and its
output is added as a residual to the standard learnable PE.

This allows the positional information to shift based on the key, which
can help the model learn that different encrypted representations of the
same image (under different keys) have different positional statistics.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LearnedPositionalEncoding(nn.Module):
    """Standard ViT learnable 1-D positional embeddings.

    Parameters
    ----------
    num_patches : int
        Number of image patch tokens (not counting [CLS]).
    embed_dim : int
        Token embedding dimensionality.
    dropout : float
        Dropout applied after adding positional embeddings.
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to token sequence.

        Parameters
        ----------
        x : (B, N+1, D) - token sequence with [CLS] prepended

        Returns
        -------
        (B, N+1, D)
        """
        return self.drop(x + self.pos_embed)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learned parameters).

    Follows the formulation from Vaswani et al. (2017).
    """

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
    """Learnable PE with a key-derived additive offset.

    The key offset is computed on-the-fly from a key embedding vector.
    The key can be provided as a string, bytes, or a pre-computed
    float tensor (the key embedding).

    Parameters
    ----------
    num_patches : int
    embed_dim : int
    dropout : float
    key_embed_dim : int
        Dimensionality of the key embedding fed into the MLP.
    """

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
        """Add key-aware positional embeddings.

        Parameters
        ----------
        x : (B, N+1, D) - token sequence with [CLS] prepended
        key_embedding : (B, key_embed_dim) or (1, key_embed_dim) float tensor.
            If None, falls back to standard learnable PE (no key offset).

        Returns
        -------
        (B, N+1, D)
        """
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
        """Convert a key string/bytes to a fixed float embedding vector.

        Uses SHA-256 to produce a deterministic 256-dim float32 vector.

        Parameters
        ----------
        key : str or bytes
        embed_dim : int - must be a multiple of 8

        Returns
        -------
        (1, embed_dim) float32 tensor (values in [-1, 1])
        """
        if isinstance(key, str):
            key = key.encode("utf-8")
        digest = hashlib.sha256(key).digest()
        n_repeats = math.ceil(embed_dim / 32)
        extended = (digest * n_repeats)[:embed_dim]
        values = [b / 127.5 - 1.0 for b in extended]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)
