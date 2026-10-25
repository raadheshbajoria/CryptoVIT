"""
cryptovit/models/vit.py
=======================
From-scratch Vision Transformer (ViT) implementation.

Architecture follows Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words"
with minor adaptations for CIFAR-scale experiments:
  * Pre-LN (LayerNorm before attention and MLP) for training stability.
  * Learnable 1-D positional embeddings.
  * [CLS] token classification head.
  * Drop-path (stochastic depth) regularisation.

All components are pure PyTorch - no timm dependency in this module.

Usage
-----
    from cryptovit.models import build_model
    cfg = load_config()
    model = build_model(cfg)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Per-sample drop-path (stochastic depth) regularisation.

    Randomly drops the entire residual branch during training.
    At test time, the layer is the identity.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class PatchEmbedding(nn.Module):
    """Divide an image into non-overlapping patches and project each patch.

    Parameters
    ----------
    image_size:
        Spatial size of the input image (assumed square).
    patch_size:
        Side length of each patch (assumed square). Must divide ``image_size``.
    in_channels:
        Number of input channels (3 for RGB).
    embed_dim:
        Projection dimensionality.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        tokens : (B, N, embed_dim)  where N = num_patches
        """
        x = self.projection(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    Parameters
    ----------
    embed_dim:
        Model dimensionality.
    num_heads:
        Number of attention heads.
    attn_dropout:
        Dropout applied to attention weights.
    proj_dropout:
        Dropout applied to the output projection.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D)

        Returns
        -------
        out : (B, N, D)
        """
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """Two-layer MLP with GELU activation used inside transformer blocks.

    Parameters
    ----------
    in_features:
        Input dimensionality.
    hidden_features:
        Hidden layer size (typically ``in_features * mlp_ratio``).
    out_features:
        Output dimensionality (defaults to ``in_features``).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-layer-norm.

    Residual path:   x -> LN -> MHSA -> DropPath -> + x
    Feed-forward:    x -> LN -> MLP  -> DropPath -> + x

    Parameters
    ----------
    embed_dim:
        Model dimensionality.
    num_heads:
        Number of attention heads.
    mlp_ratio:
        MLP hidden size as a multiple of ``embed_dim``.
    dropout:
        Dropout probability for MLP and attention output.
    attn_dropout:
        Dropout inside attention weights.
    drop_path_rate:
        Stochastic depth probability.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Parameters
    ----------
    image_size:
        Input image spatial size (H = W).
    patch_size:
        Patch size (H = W). Must divide ``image_size``.
    in_channels:
        Number of input channels.
    num_classes:
        Number of output classes.
    embed_dim:
        Token embedding dimensionality.
    depth:
        Number of transformer encoder blocks.
    num_heads:
        Number of attention heads per block.
    mlp_ratio:
        MLP hidden size multiplier.
    dropout:
        General dropout probability.
    attn_dropout:
        Attention weight dropout.
    drop_path_rate:
        Max stochastic depth rate (linearly scaled across blocks).
    pool:
        Pooling strategy. ``"cls"`` uses the [CLS] token;
        ``"mean"`` uses global average pooling over all patch tokens.
    patch_embedding_cls:
        Patch embedding class to use (allows swapping in CryptoPatchEmbedding).
    """

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
        patch_embedding_cls=None,
    ):
        super().__init__()
        assert pool in ("cls", "mean"), f"pool must be 'cls' or 'mean', got '{pool}'"
        self.pool = pool
        self.embed_dim = embed_dim

        patch_embedding_cls = patch_embedding_cls or PatchEmbedding
        self.patch_embed = patch_embedding_cls(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
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

        self._init_weights()

        logger.info(
            "VisionTransformer | image=%d patch=%d embed=%d depth=%d heads=%d params=%.2fM",
            image_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            self.num_parameters() / 1e6,
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)

        return self.head(x)

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features (before classification head).

        Useful for ablation studies and visualisation.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0] if self.pool == "cls" else x[:, 1:].mean(dim=1)
