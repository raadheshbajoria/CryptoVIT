"""
cryptovit/models/patch_embedding.py
=====================================
Patch embedding variants for CryptoViT.

Standard ViT uses a simple linear projection (Conv2d) to embed patches.
CryptoPatchEmbedding extends this with a two-path design that can learn
a more robust mapping from encrypted patch representations.

Design
------
* ``StandardPatchEmbedding``  - identical to the one in vit.py; re-exported
  here for clarity and for use in ablation experiments.
* ``CryptoPatchEmbedding``    - uses a small convolutional front-end
  before the linear projection.  The extra convolutions give the model
  capacity to handle the statistical properties of encrypted patches,
  which have higher entropy and lower spatial correlation than plaintext.

Both classes share the same interface (``__init__`` signature and
``forward`` signature) so they can be swapped into ``VisionTransformer``
via the ``patch_embedding_cls`` parameter.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class StandardPatchEmbedding(nn.Module):
    """Standard ViT patch embedding: Conv2d(in_channels, embed_dim, patch_size).

    This is identical to the PatchEmbedding in vit.py.  It is duplicated
    here so that ablation experiments can import both from one place.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, embed_dim)"""
        x = self.projection(x)
        return rearrange(x, "b d h w -> b (h w) d")


class CryptoPatchEmbedding(nn.Module):
    """Custom patch embedding adapted for encrypted / high-entropy images.

    Architecture
    ------------
    1. A two-layer convolutional front-end (3x3 conv + BatchNorm + GELU, twice)
       runs over the *whole* image at full resolution, so the receptive field
       of a feature near a patch border extends into the neighbouring patch.
       Its memory use therefore scales with ``hidden_dim * H * W`` per image.
    2. A final Conv2d with stride=patch_size performs the linear projection
       (same as the standard embedding).

    Compared to ``StandardPatchEmbedding``, this adds ~3.04M parameters at
    ``hidden_dim=64`` (the 16x16 projection now reads 64 channels instead of
    3), versus ~0.15M for the standard embedding. Any accuracy gain over the
    standard embedding must be read with that capacity difference in mind.
    Whether it helps on encrypted images is an empirical question tested by
    the ablation study.

    Parameters
    ----------
    image_size, patch_size, in_channels, embed_dim:
        Same as ``StandardPatchEmbedding``.
    hidden_dim:
        Number of channels in the convolutional front-end (default 64).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        hidden_dim: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.conv_front = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.projection = nn.Conv2d(
            hidden_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        logger.info(
            "CryptoPatchEmbedding | patch=%d  hidden=%d  embed=%d  patches=%d",
            patch_size,
            hidden_dim,
            embed_dim,
            self.num_patches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, embed_dim)"""
        x = self.conv_front(x)
        x = self.projection(x)
        return rearrange(x, "b d h w -> b (h w) d")
