"""
api/main.py
============
FastAPI inference backend for CryptoViT.

Endpoints
---------
GET  /health              — Health check
POST /encrypt             — Upload an image, receive its encrypted version
POST /decrypt             — Decrypt an image (returns plaintext for demo only)
POST /predict             — Classify an encrypted image with CryptoViT
POST /predict_plaintext   — Classify a plaintext image with baseline ViT
GET  /metrics             — Security metrics for the last processed image
GET  /model_info          — Model architecture and parameter count

Running
-------
    pip install fastapi uvicorn python-multipart pillow
    python -m uvicorn api.main:app --reload --port 8000

Or from project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from cryptovit.encryption import decrypt, encrypt
from cryptovit.evaluation.security_metrics import compute_all_security_metrics
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Global state (loaded at startup)
_cfg = None
_device = None
_baseline_model = None     # Standard ViT on plaintext
_cryptovit_model = None    # Full CryptoViT on encrypted images
_last_metrics: Dict = {}   # Security metrics from last /encrypt call

ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _device, _baseline_model, _cryptovit_model

    _cfg = load_config()
    setup_logging(level=_cfg.logging.level)
    _device = get_device(_cfg)

    logger.info("CryptoViT API starting up on device: %s", _device)

    # Load models
    _baseline_model = _try_load_baseline()
    _cryptovit_model = _try_load_cryptovit()
    yield


app = FastAPI(
    title="CryptoViT API",
    description=(
        "Privacy-preserving image classification via chaos-assisted encryption "
        "and Vision Transformer (CryptoViT)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _try_load_baseline():
    """Attempt to load the baseline ViT checkpoint."""
    from cryptovit.models import build_model
    try:
        model = build_model(_cfg).to(_device)
        exp_dir = PROJECT_ROOT / _cfg.paths.experiments_dir
        # Search for baseline checkpoint
        ckpt_dirs = list(exp_dir.glob("baseline_*/checkpoints/*.pt"))
        if ckpt_dirs:
            ckpt = sorted(ckpt_dirs)[-1]  # take the latest
            state = torch.load(ckpt, map_location=_device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            logger.info("Loaded baseline model from %s", ckpt)
            return model
        else:
            logger.warning("No baseline checkpoint found. Using untrained model.")
            model.eval()
            return model
    except Exception as e:
        logger.error("Failed to load baseline model: %s", e)
        return None


def _try_load_cryptovit():
    """Attempt to load a trained CryptoViT checkpoint."""
    try:
        from omegaconf import OmegaConf
        cryptovit_cfg = OmegaConf.merge(
            _cfg,
            OmegaConf.create({"model": {"name": "cryptovit"}})
        )
        from cryptovit.models import build_model
        model = build_model(cryptovit_cfg).to(_device)
        exp_dir = PROJECT_ROOT / _cfg.paths.experiments_dir
        ckpt_dirs = list(exp_dir.glob("*cryptovit*/checkpoints/*.pt"))
        if ckpt_dirs:
            ckpt = sorted(ckpt_dirs)[-1]
            state = torch.load(ckpt, map_location=_device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            logger.info("Loaded CryptoViT from %s", ckpt)
        else:
            logger.warning("No CryptoViT checkpoint found. Using untrained model.")
            model.eval()
        return model
    except Exception as e:
        logger.error("Failed to load CryptoViT: %s", e)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(img: Image.Image, target_size: int = 32) -> np.ndarray:
    """Resize PIL image and return HWC uint8 numpy array."""
    img = img.resize((target_size, target_size), Image.LANCZOS).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _numpy_to_tensor(img_np: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert HWC uint8 numpy to (1, C, H, W) normalised float tensor."""
    from torchvision import transforms
    pil = Image.fromarray(img_np)
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])(pil)
    return t.unsqueeze(0)


def _image_to_bytes(img_np: np.ndarray) -> bytes:
    """Convert HWC uint8 numpy array to PNG bytes."""
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


@torch.no_grad()
def _classify(model, tensor: torch.Tensor, key_emb=None) -> dict:
    """Run inference and return class + confidence."""
    tensor = tensor.to(_device)
    if key_emb is not None and hasattr(model, "pos_encoding"):
        logits = model(tensor, key_embedding=key_emb)
    else:
        logits = model(tensor)
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = probs.argmax().item()
    return {
        "class_id": top_idx,
        "class_name": CIFAR10_CLASSES[top_idx] if top_idx < len(CIFAR10_CLASSES) else str(top_idx),
        "confidence": float(probs[top_idx]),
        "top5": [
            {"class": CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else str(i),
             "prob": float(probs[i])}
            for i in probs.topk(5).indices.tolist()
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(_device),
        "baseline_loaded": _baseline_model is not None,
        "cryptovit_loaded": _cryptovit_model is not None,
    }


@app.post("/encrypt")
async def encrypt_image(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
):
    """Encrypt an uploaded image.

    Returns the encrypted image as a PNG file.
    Also computes and stores security metrics for /metrics.
    """
    global _last_metrics

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes))
        img_np = _pil_to_numpy(pil)

        enc_np, meta = encrypt(img_np, key)

        # Compute and cache security metrics
        _last_metrics = compute_all_security_metrics(img_np, enc_np)
        _last_metrics["key_hash"] = meta["key_hash"]

        return Response(
            content=_image_to_bytes(enc_np),
            media_type="image/png",
            headers={"X-Encryption-KeyHash": meta["key_hash"]},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decrypt")
async def decrypt_image(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
):
    """Decrypt an encrypted image (authorised demo endpoint).

    WARNING: In a production system this endpoint would not exist.
    It is included only for demonstration purposes.
    """
    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes))
        enc_np = np.array(pil.convert("RGB"), dtype=np.uint8)

        restored_np = decrypt(enc_np, key)

        return Response(content=_image_to_bytes(restored_np), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_encrypted(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
    encrypt_first: bool = Form(default=True),
):
    """Classify an image using CryptoViT.

    If ``encrypt_first=True`` (default), the uploaded plaintext image is
    encrypted before classification.  If False, the image is assumed to be
    already encrypted.
    """
    if _cryptovit_model is None:
        raise HTTPException(status_code=503, detail="CryptoViT model not loaded.")

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes))
        img_np = _pil_to_numpy(pil)

        if encrypt_first:
            img_np, _ = encrypt(img_np, key)

        tensor = _numpy_to_tensor(img_np, image_size=_cfg.model.image_size)

        from cryptovit.models.positional_encoding import KeyAwarePositionalEncoding
        key_emb = KeyAwarePositionalEncoding.key_to_embedding(key).to(_device)

        result = _classify(_cryptovit_model, tensor, key_emb=key_emb)
        result["model"] = "CryptoViT"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_plaintext")
async def predict_plaintext(file: UploadFile = File(...)):
    """Classify a plaintext image using the baseline ViT."""
    if _baseline_model is None:
        raise HTTPException(status_code=503, detail="Baseline model not loaded.")

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes))
        img_np = _pil_to_numpy(pil)
        tensor = _numpy_to_tensor(img_np, image_size=_cfg.model.image_size)

        result = _classify(_baseline_model, tensor)
        result["model"] = "BaselineViT"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Return security metrics from the last /encrypt call."""
    if not _last_metrics:
        raise HTTPException(
            status_code=404,
            detail="No metrics available. Call /encrypt first."
        )
    # Convert nested numpy floats to plain Python floats
    return JSONResponse(content=json.loads(json.dumps(_last_metrics, default=float)))


@app.get("/model_info")
async def model_info():
    """Return model architecture info."""
    info = {}
    if _baseline_model is not None:
        info["baseline_vit"] = {
            "parameters": sum(p.numel() for p in _baseline_model.parameters()),
        }
    if _cryptovit_model is not None:
        info["cryptovit"] = {
            "parameters": sum(p.numel() for p in _cryptovit_model.parameters()),
        }
    return JSONResponse(content=info)
