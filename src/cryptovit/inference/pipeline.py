from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image

from cryptovit.encryption import encrypt

logger = logging.getLogger(__name__)

VARIANT_INFO: Dict[str, Dict] = {
    "A": {"label": "Plain ViT (plaintext image)", "encrypted": False},
    "B": {"label": "Plain ViT (encrypted image)", "encrypted": True},
    "C": {"label": "CryptoPatchEmbedding (encrypted image)", "encrypted": True},
    "D": {
        "label": "CryptoViT: patch embedding + key-aware positions (encrypted image)",
        "encrypted": True,
    },
}

_UNIFORM_NOISE_STATS = {"mean": [0.5] * 3, "std": [0.2887] * 3}

_CKPT_PATTERN = re.compile(r"checkpoint_epoch(\d+)_acc([0-9.]+)\.pt$")


def build_variant_model(name: str, cfg: DictConfig) -> nn.Module:
    from cryptovit.models.cryptovit import CryptoViT
    from cryptovit.models.patch_embedding import CryptoPatchEmbedding
    from cryptovit.models.vit import VisionTransformer

    if name not in VARIANT_INFO:
        raise ValueError(f"Unknown variant '{name}'. Choose from {list(VARIANT_INFO)}.")
    m = cfg.model
    common = dict(
        image_size=m.image_size,
        patch_size=m.patch_size,
        in_channels=m.in_channels,
        num_classes=m.num_classes,
        embed_dim=m.embed_dim,
        depth=m.depth,
        num_heads=m.num_heads,
        mlp_ratio=m.mlp_ratio,
        dropout=m.dropout,
        attn_dropout=m.attn_dropout,
        drop_path_rate=m.drop_path_rate,
        pool=m.pool,
    )
    if name in ("A", "B"):
        return VisionTransformer(**common)
    if name == "C":
        return VisionTransformer(**common, patch_embedding_cls=CryptoPatchEmbedding)
    return CryptoViT(**common, hidden_dim=64, key_embed_dim=256)


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    best: Optional[Tuple[float, int, Path]] = None
    for path in Path(checkpoint_dir).glob("checkpoint_epoch*_acc*.pt"):
        match = _CKPT_PATTERN.search(path.name)
        if not match:
            continue
        key = (float(match.group(2)), int(match.group(1)), path)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def read_results(ablation_dir: Path) -> List[Dict]:
    out = []
    for variant in VARIANT_INFO:
        path = Path(ablation_dir) / f"result_{variant}.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["variant"] = variant
            data["label"] = VARIANT_INFO[variant]["label"]
            out.append(data)
    return out


class ClassifierService:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        ablation_dir: Path,
        train_key: str,
        preferred_variant: str = "D",
    ):
        self.cfg = cfg
        self.device = device
        self.ablation_dir = Path(ablation_dir)
        self.train_key = train_key
        self.preferred_variant = preferred_variant
        self.scheme = cfg.encryption.get("scheme", "v1")
        self.transient_steps = int(cfg.encryption.get("transient_steps", 1000))
        self.models: Dict[str, nn.Module] = {}
        self.trained: Dict[str, bool] = {}
        self._encrypted_stats: Optional[Dict] = None

    def load(self) -> None:
        for name in VARIANT_INFO:
            ckpt = find_best_checkpoint(self.ablation_dir / f"variant_{name}" / "checkpoints")
            if ckpt is None and name not in ("A", "D"):
                continue
            try:
                model = build_variant_model(name, self.cfg)
                if ckpt is not None:
                    state = torch.load(ckpt, map_location=self.device, weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                    logger.info("Variant %s: loaded %s", name, ckpt)
                else:
                    logger.warning(
                        "Variant %s: no checkpoint found, using an UNTRAINED model.", name
                    )
                self.models[name] = model.to(self.device).eval()
                self.trained[name] = ckpt is not None
            except Exception as exc:
                logger.error("Variant %s failed to load: %s", name, exc)

    @property
    def class_names(self) -> List[str]:
        from cryptovit.datasets import get_class_names

        return get_class_names(self.cfg.dataset.name)

    @property
    def native_size(self) -> int:
        from cryptovit.datasets.medmnist_dataset import is_medmnist

        if is_medmnist(self.cfg.dataset.name):
            return int(self.cfg.dataset.get("medmnist_size", 28))
        return 32

    def status(self) -> Dict[str, Dict]:
        return {
            name: {
                "label": info["label"],
                "encrypted_input": info["encrypted"],
                "loaded": name in self.models,
                "trained": self.trained.get(name, False),
            }
            for name, info in VARIANT_INFO.items()
        }

    def encrypted_stats(self) -> Dict[str, List[float]]:
        if self._encrypted_stats is not None:
            return self._encrypted_stats

        key_hash = hashlib.sha256(self.train_key.encode()).hexdigest()
        cache = self.ablation_dir / "encrypted_stats.json"
        if cache.exists():
            data = json.loads(cache.read_text())
            if (
                data.get("key_hash") == key_hash
                and data.get("scheme") == self.scheme
                and data.get("transient_steps") == self.transient_steps
            ):
                self._encrypted_stats = {"mean": data["mean"], "std": data["std"]}
                return self._encrypted_stats

        stats = None
        try:
            from cryptovit.datasets.encrypted_dataset import compute_encrypted_stats
            from cryptovit.datasets.medmnist_dataset import is_medmnist, load_medmnist_splits

            if is_medmnist(self.cfg.dataset.name):
                root = Path(self.cfg.paths.project_root) / self.cfg.paths.raw_dir / "medmnist"
                train_base = load_medmnist_splits(
                    self.cfg.dataset.name, root, size=self.native_size, download=False
                )[0]
                stats = compute_encrypted_stats(
                    train_base,
                    self.train_key,
                    range(len(train_base)),
                    transient_steps=self.transient_steps,
                    scheme=self.scheme,
                )
        except Exception as exc:
            logger.warning("Could not compute encrypted-image statistics (%s).", exc)

        if stats is None:
            logger.warning("Using uniform-noise normalisation for encrypted images.")
            stats = dict(_UNIFORM_NOISE_STATS)
        else:
            try:
                self.ablation_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(
                    json.dumps(
                        {
                            "key_hash": key_hash,
                            "scheme": self.scheme,
                            "transient_steps": self.transient_steps,
                            **stats,
                        },
                        indent=2,
                    )
                )
            except OSError as exc:
                logger.warning("Could not cache encrypted-image statistics: %s", exc)
        self._encrypted_stats = stats
        return stats

    def plaintext_stats(self) -> Dict[str, List[float]]:
        from cryptovit.datasets.cifar_dataset import _get_stats

        return _get_stats(self.cfg)

    def to_tensor(self, img_np: np.ndarray, encrypted: bool) -> torch.Tensor:
        from cryptovit.datasets.encrypted_dataset import build_encrypted_transform

        stats = self.encrypted_stats() if encrypted else self.plaintext_stats()
        transform = build_encrypted_transform(
            self.cfg.model.image_size, stats["mean"], stats["std"]
        )
        return transform(Image.fromarray(img_np)).unsqueeze(0)

    def encrypt_image(self, img_np: np.ndarray, key: str) -> np.ndarray:
        enc, _ = encrypt(img_np, key, self.transient_steps, scheme=self.scheme)
        return enc

    @torch.no_grad()
    def _classify(self, variant: str, tensor: torch.Tensor, key: Optional[str]) -> Dict:
        if variant not in self.models:
            raise KeyError(f"Variant {variant} is not loaded.")
        model = self.models[variant]
        tensor = tensor.to(self.device)
        if variant == "D":
            from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding

            key_emb = KeyAwarePositionalEncoding.key_to_embedding(key or self.train_key).to(
                self.device
            )
            logits = model(tensor, key_embedding=key_emb)
        else:
            logits = model(tensor)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        names = self.class_names
        top = probs.topk(min(5, probs.numel())).indices.tolist()
        best = top[0]
        return {
            "variant": variant,
            "model": VARIANT_INFO[variant]["label"],
            "trained": self.trained.get(variant, False),
            "class_id": best,
            "class_name": names[best] if best < len(names) else str(best),
            "confidence": float(probs[best]),
            "top5": [
                {"class": names[i] if i < len(names) else str(i), "prob": float(probs[i])}
                for i in top
            ],
        }

    def predict_plaintext(self, img_np: np.ndarray) -> Dict:
        return self._classify("A", self.to_tensor(img_np, encrypted=False), None)

    def predict_encrypted(
        self,
        img_np: np.ndarray,
        key: str,
        variant: Optional[str] = None,
        encrypt_first: bool = True,
    ) -> Dict:
        variant = variant or self.preferred_variant
        if variant not in self.models or not VARIANT_INFO[variant]["encrypted"]:
            candidates = [v for v in ("D", "C", "B") if v in self.models]
            if not candidates:
                raise KeyError("No encrypted-image model is loaded.")
            variant = candidates[0]
        enc = self.encrypt_image(img_np, key) if encrypt_first else img_np
        return self._classify(variant, self.to_tensor(enc, encrypted=True), key)

    def compare(self, img_np: np.ndarray, key: str) -> Dict:
        out: Dict = {"plaintext": None, "encrypted": {}}
        if "A" in self.models:
            out["plaintext"] = self.predict_plaintext(img_np)
        enc = self.encrypt_image(img_np, key)
        tensor = self.to_tensor(enc, encrypted=True)
        for variant in ("B", "C", "D"):
            if variant in self.models:
                out["encrypted"][variant] = self._classify(variant, tensor, key)
        return out
