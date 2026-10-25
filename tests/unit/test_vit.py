"""
tests/unit/test_vit.py
=======================
Unit tests for the Vision Transformer model.

All tests run on CPU and use tiny configurations to stay fast.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf


def _tiny_cfg(
    image_size: int = 32,
    patch_size: int = 8,
    embed_dim: int = 48,
    depth: int = 2,
    num_heads: int = 3,
    num_classes: int = 10,
    pool: str = "cls",
) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "name": "vit_small",
                "image_size": image_size,
                "patch_size": patch_size,
                "in_channels": 3,
                "num_classes": num_classes,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "attn_dropout": 0.0,
                "drop_path_rate": 0.0,
                "pool": pool,
            }
        }
    )


class TestPatchEmbedding:
    def test_output_shape(self):
        from cryptovit.models.vit import PatchEmbedding

        pe = PatchEmbedding(image_size=32, patch_size=8, in_channels=3, embed_dim=48)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 16, 48), f"Got {out.shape}"

    def test_num_patches_224(self):
        from cryptovit.models.vit import PatchEmbedding

        pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=192)
        assert pe.num_patches == 196

    def test_invalid_patch_size_raises(self):
        from cryptovit.models.vit import PatchEmbedding

        with pytest.raises(AssertionError):
            PatchEmbedding(image_size=32, patch_size=7)


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        from cryptovit.models.vit import MultiHeadSelfAttention

        attn = MultiHeadSelfAttention(embed_dim=48, num_heads=3)
        x = torch.randn(2, 16, 48)
        out = attn(x)
        assert out.shape == x.shape

    def test_invalid_heads_raises(self):
        from cryptovit.models.vit import MultiHeadSelfAttention

        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=48, num_heads=7)


class TestTransformerBlock:
    def test_output_shape(self):
        from cryptovit.models.vit import TransformerBlock

        block = TransformerBlock(embed_dim=48, num_heads=3)
        x = torch.randn(2, 17, 48)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero weights the block should approximate identity."""
        from cryptovit.models.vit import TransformerBlock

        block = TransformerBlock(embed_dim=48, num_heads=3, drop_path_rate=0.0)
        for p in block.parameters():
            p.data.zero_()
        x = torch.randn(1, 5, 48)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5), "Residual connection broken"


class TestVisionTransformer:
    def test_forward_shape_cls(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            in_channels=3,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
            pool="cls",
        )
        x = torch.randn(4, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (4, 10), f"Got {logits.shape}"

    def test_forward_shape_mean_pool(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            in_channels=3,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
            pool="mean",
        )
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 10)

    def test_get_features_shape(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
        )
        x = torch.randn(3, 3, 32, 32)
        feats = model.get_features(x)
        assert feats.shape == (3, 48), f"Got {feats.shape}"

    def test_num_parameters_positive(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
        )
        assert model.num_parameters() > 0

    def test_different_batch_sizes(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
        )
        model.eval()
        for B in [1, 4, 16]:
            x = torch.randn(B, 3, 32, 32)
            out = model(x)
            assert out.shape == (B, 10)

    def test_invalid_pool_raises(self):
        from cryptovit.models.vit import VisionTransformer

        with pytest.raises(AssertionError):
            VisionTransformer(
                image_size=32,
                patch_size=8,
                num_classes=10,
                embed_dim=48,
                depth=2,
                num_heads=3,
                pool="invalid",
            )

    def test_no_nan_in_output(self):
        from cryptovit.models.vit import VisionTransformer

        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
        )
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert not torch.isnan(logits).any(), "NaN in model output"


class TestBuildModel:
    def test_vit_small(self):
        from cryptovit.models import build_model

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "vit_small",
                    "image_size": 32,
                    "patch_size": 8,
                    "in_channels": 3,
                    "num_classes": 10,
                    "embed_dim": 192,
                    "depth": 6,
                    "num_heads": 3,
                    "mlp_ratio": 4.0,
                    "dropout": 0.0,
                    "attn_dropout": 0.0,
                    "drop_path_rate": 0.0,
                    "pool": "cls",
                }
            }
        )
        model = build_model(cfg)
        assert model is not None

    def test_unknown_model_raises(self):
        from cryptovit.models import build_model

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "unknown_model",
                    "image_size": 32,
                    "patch_size": 8,
                    "in_channels": 3,
                    "num_classes": 10,
                }
            }
        )
        with pytest.raises(ValueError):
            build_model(cfg)


class TestDropPath:
    def test_identity_at_eval(self):
        from cryptovit.models.vit import DropPath

        dp = DropPath(drop_prob=0.9)
        dp.eval()
        x = torch.randn(4, 10)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_zero_prob_identity(self):
        from cryptovit.models.vit import DropPath

        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(4, 10)
        out = dp(x)
        assert torch.allclose(out, x)
