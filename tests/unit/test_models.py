from __future__ import annotations

import torch
from omegaconf import OmegaConf

from cryptovit.models import (
    CryptoPatchEmbedding,
    CryptoViT,
    KeyAwarePositionalEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    StandardPatchEmbedding,
    build_cryptovit,
    build_model,
)


class TestPatchEmbeddings:
    def test_standard_patch_embedding_shape(self):
        emb = StandardPatchEmbedding(image_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = emb(x)
        assert out.shape == (2, 16, 64)
        assert emb.num_patches == 16

    def test_crypto_patch_embedding_shape(self):
        emb = CryptoPatchEmbedding(
            image_size=32, patch_size=8, in_channels=3, embed_dim=64, hidden_dim=32
        )
        x = torch.randn(2, 3, 32, 32)
        out = emb(x)
        assert out.shape == (2, 16, 64)
        assert emb.num_patches == 16


class TestPositionalEncodings:
    def test_learned_pe_shape(self):
        pe = LearnedPositionalEncoding(num_patches=16, embed_dim=48, dropout=0.0)
        x = torch.randn(2, 17, 48)
        out = pe(x)
        assert out.shape == (2, 17, 48)

    def test_sinusoidal_pe_shape(self):
        pe = SinusoidalPositionalEncoding(num_patches=16, embed_dim=48, dropout=0.0)
        x = torch.randn(2, 17, 48)
        out = pe(x)
        assert out.shape == (2, 17, 48)

    def test_key_aware_pe_fallback_without_key(self):
        pe = KeyAwarePositionalEncoding(num_patches=16, embed_dim=48, key_embed_dim=64)
        x = torch.randn(2, 17, 48)
        out = pe(x, key_embedding=None)
        assert out.shape == (2, 17, 48)

    def test_key_aware_pe_with_key_embedding(self):
        pe = KeyAwarePositionalEncoding(num_patches=16, embed_dim=48, key_embed_dim=64)
        x = torch.randn(2, 17, 48)
        key_emb = torch.randn(1, 64)
        out = pe(x, key_embedding=key_emb)
        assert out.shape == (2, 17, 48)

    def test_key_to_embedding_deterministic(self):
        emb1 = KeyAwarePositionalEncoding.key_to_embedding("test_key", embed_dim=64)
        emb2 = KeyAwarePositionalEncoding.key_to_embedding("test_key", embed_dim=64)
        assert emb1.shape == (1, 64)
        assert torch.allclose(emb1, emb2)

    def test_key_to_embedding_different_keys_differ(self):
        emb1 = KeyAwarePositionalEncoding.key_to_embedding("key_alpha", embed_dim=64)
        emb2 = KeyAwarePositionalEncoding.key_to_embedding("key_beta", embed_dim=64)
        assert not torch.allclose(emb1, emb2)


class TestCryptoViT:
    def _tiny_cryptovit(self, pool="cls") -> CryptoViT:
        return CryptoViT(
            image_size=32,
            patch_size=8,
            in_channels=3,
            num_classes=10,
            embed_dim=48,
            depth=2,
            num_heads=3,
            mlp_ratio=2.0,
            dropout=0.0,
            attn_dropout=0.0,
            drop_path_rate=0.0,
            pool=pool,
            hidden_dim=32,
            key_embed_dim=64,
        )

    def test_forward_without_key(self):
        model = self._tiny_cryptovit(pool="cls")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_with_key(self):
        model = self._tiny_cryptovit(pool="cls")
        x = torch.randn(2, 3, 32, 32)
        key_emb = KeyAwarePositionalEncoding.key_to_embedding("secret", embed_dim=64)
        out = model(x, key_embedding=key_emb)
        assert out.shape == (2, 10)

    def test_forward_mean_pool(self):
        model = self._tiny_cryptovit(pool="mean")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_get_features(self):
        model = self._tiny_cryptovit(pool="cls")
        x = torch.randn(2, 3, 32, 32)
        feats = model.get_features(x)
        assert feats.shape == (2, 48)

    def test_num_parameters(self):
        model = self._tiny_cryptovit()
        assert model.num_parameters() > 0

    def test_build_cryptovit_via_factory(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "cryptovit",
                    "image_size": 32,
                    "patch_size": 8,
                    "in_channels": 3,
                    "num_classes": 10,
                    "embed_dim": 48,
                    "depth": 2,
                    "num_heads": 3,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "attn_dropout": 0.0,
                    "drop_path_rate": 0.0,
                    "pool": "cls",
                    "hidden_dim": 32,
                    "key_embed_dim": 64,
                }
            }
        )
        model = build_cryptovit(cfg)
        assert isinstance(model, CryptoViT)

        model_from_build = build_model(cfg)
        assert isinstance(model_from_build, CryptoViT)
