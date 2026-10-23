"""
api/main.py
============
FastAPI inference and demonstration backend for CryptoViT.

Endpoints
---------
GET  /health              — Health check and trained-model readiness
GET  /model_info          — Architecture and parameter count
POST /encrypt             — Encrypt an image and return PNG bytes
POST /decrypt             — Decrypt an image and return PNG bytes
POST /predict_plaintext   — Classify a plaintext image with the trained baseline ViT
POST /predict             — CryptoViT prediction (marked pending until validated checkpoint exists)
GET  /metrics             — Security metrics from the last /encrypt call

Running
-------
    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

app = FastAPI(
    title="CryptoViT API",
    description=(
        "Backend API supporting the CryptoViT project review dashboard. "
        "Provides validated Standard ViT baseline inference and hyperchaotic encryption/decryption demo."
    ),
    version="0.1.0",
)

# Global state
_cfg = None
_device = None
_baseline_model = None           # Trained Standard ViT
_baseline_ckpt_path: Optional[str] = None
_baseline_cfg: Optional[dict] = None
_cryptovit_model = None          # CryptoViT (strictly requires trained checkpoint)
_cryptovit_ckpt_path: Optional[str] = None
_last_metrics: Dict = {}         # Security metrics from last /encrypt call

DEFAULT_KEY = "CryptoViT_Research_Key_2025"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _find_baseline_checkpoint() -> Optional[Path]:
    """Search for the audited review baseline checkpoint or any valid baseline checkpoint."""
    exp_dir = PROJECT_ROOT / "experiments"
    # 1. Prefer audited review baseline directory if available
    specific_ckpt = exp_dir / "review_baseline_20261022_211230" / "checkpoints" / "checkpoint_epoch0005_acc0.3110.pt"
    if specific_ckpt.is_file():
        return specific_ckpt

    # 2. Look for review_baseline_* directories
    review_ckpts = list(exp_dir.glob("review_baseline_*/checkpoints/*.pt"))
    if review_ckpts:
        return sorted(review_ckpts)[-1]

    # 3. Look for standard baseline_* directories
    baseline_ckpts = list(exp_dir.glob("baseline_*/checkpoints/*.pt"))
    if baseline_ckpts:
        return sorted(baseline_ckpts)[-1]

    return None


def _load_baseline_model():
    """Load the trained baseline ViT checkpoint strictly.
    Returns (model, ckpt_path, cfg_dict). If no valid checkpoint exists, returns (None, None, None).
    """
    from cryptovit.models import build_model
    ckpt_path = _find_baseline_checkpoint()
    if ckpt_path is None or not ckpt_path.is_file():
        logger.warning("No trained baseline checkpoint found. Baseline model will NOT be loaded.")
        return None, None, None

    try:
        # Check if there is run_metadata.json next to checkpoints/
        meta_path = ckpt_path.parents[1] / "run_metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            model_cfg_dict = meta.get("config", {}).get("model", {})
        else:
            model_cfg_dict = {}

        # Reconstruct configuration matching review baseline
        base_cfg = load_config()
        base_cfg.model.name = model_cfg_dict.get("name", "vit_tiny")
        base_cfg.model.image_size = model_cfg_dict.get("image_size", 96)
        base_cfg.model.patch_size = model_cfg_dict.get("patch_size", 16)
        base_cfg.model.embed_dim = model_cfg_dict.get("embed_dim", 96)
        base_cfg.model.depth = model_cfg_dict.get("depth", 4)
        base_cfg.model.num_heads = model_cfg_dict.get("num_heads", 3)
        base_cfg.model.num_classes = model_cfg_dict.get("num_classes", 10)

        model = build_model(base_cfg).to(_device)
        state = torch.load(ckpt_path, map_location=_device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        logger.info("Successfully loaded trained baseline model from %s (strict=True)", ckpt_path)
        return model, str(ckpt_path), dict(base_cfg.model)
    except Exception as e:
        logger.error("Failed to load baseline checkpoint from %s: %s", ckpt_path, e)
        return None, None, None


def _load_cryptovit_model():
    """Load a trained CryptoViT checkpoint strictly.
    Does NOT instantiate a random untrained model if checkpoint is absent.
    """
    exp_dir = PROJECT_ROOT / "experiments"
    ckpt_dirs = list(exp_dir.glob("*cryptovit*/checkpoints/*.pt"))
    if not ckpt_dirs:
        logger.info("No trained CryptoViT checkpoint exists. CryptoViT inference remains pending.")
        return None, None

    ckpt = sorted(ckpt_dirs)[-1]
    try:
        from omegaconf import OmegaConf
        from cryptovit.models import build_model
        cryptovit_cfg = OmegaConf.merge(_cfg, OmegaConf.create({"model": {"name": "cryptovit"}}))
        model = build_model(cryptovit_cfg).to(_device)
        state = torch.load(ckpt, map_location=_device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        logger.info("Successfully loaded trained CryptoViT from %s", ckpt)
        return model, str(ckpt)
    except Exception as e:
        logger.error("Failed to load CryptoViT checkpoint: %s", e)
        return None, None


@app.on_event("startup")
async def startup_event():
    global _cfg, _device, _baseline_model, _baseline_ckpt_path, _baseline_cfg, _cryptovit_model, _cryptovit_ckpt_path

    _cfg = load_config()
    setup_logging(level=_cfg.logging.level)
    _device = get_device(_cfg)
    logger.info("CryptoViT API starting up on device: %s", _device)

    _baseline_model, _baseline_ckpt_path, _baseline_cfg = _load_baseline_model()
    _cryptovit_model, _cryptovit_ckpt_path = _load_cryptovit_model()


# ---------------------------------------------------------------------------
# Preprocessing and Classification Helpers
# ---------------------------------------------------------------------------

def _pil_to_eval_tensor(pil_img: Image.Image, image_size: int = 96) -> torch.Tensor:
    """Preprocess PIL image using exact CIFAR-10 evaluation transforms (Resize, ToTensor, Normalize)."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])
    tensor = t(pil_img.convert("RGB"))
    return tensor.unsqueeze(0)


@torch.no_grad()
def _classify_tensor(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """Run model in eval mode, verify finite logits, and compute softmax probabilities."""
    model.eval()
    tensor = tensor.to(_device)
    logits = model(tensor)
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced non-finite (NaN or Inf) logits.")

    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = probs.argmax().item()

    top5_indices = probs.topk(min(5, len(CIFAR10_CLASSES))).indices.tolist()
    top5 = [
        {
            "class": CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else str(i),
            "prob": float(probs[i]),
        }
        for i in top5_indices
    ]

    return {
        "class_id": top_idx,
        "class_name": CIFAR10_CLASSES[top_idx] if top_idx < len(CIFAR10_CLASSES) else str(top_idx),
        "confidence": float(probs[top_idx]),
        "logits": [float(val) for val in logits[0].tolist()],
        "top5": top5,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check distinguishing API availability from trained-model readiness."""
    return {
        "status": "ok",
        "device": str(_device),
        "baseline_ready": _baseline_model is not None,
        "baseline_checkpoint": _baseline_ckpt_path,
        "baseline_architecture": _baseline_cfg.get("name") if _baseline_cfg else None,
        "cryptovit_ready": _cryptovit_model is not None,
        "cryptovit_checkpoint": _cryptovit_ckpt_path,
        "cryptovit_status": "ready" if _cryptovit_model is not None else "pending_training_validation",
    }


@app.get("/model_info")
async def model_info():
    """Return model architecture info for loaded models."""
    info = {}
    if _baseline_model is not None:
        info["baseline_vit"] = {
            "status": "trained",
            "model_name": _baseline_cfg.get("name", "vit_tiny") if _baseline_cfg else "vit_tiny",
            "image_size": _baseline_cfg.get("image_size", 96) if _baseline_cfg else 96,
            "parameters": sum(p.numel() for p in _baseline_model.parameters()),
            "checkpoint": _baseline_ckpt_path,
        }
    else:
        info["baseline_vit"] = {
            "status": "not_loaded",
            "reason": "Audited checkpoint not found or failed to load.",
        }

    if _cryptovit_model is not None:
        info["cryptovit"] = {
            "status": "trained",
            "parameters": sum(p.numel() for p in _cryptovit_model.parameters()),
            "checkpoint": _cryptovit_ckpt_path,
        }
    else:
        info["cryptovit"] = {
            "status": "pending",
            "message": "CryptoViT training and evaluation are pending validation. No trained weights loaded.",
        }
    return JSONResponse(content=info)


@app.post("/encrypt")
async def encrypt_image(
    file: UploadFile = File(...),
    # Required at the framework layer: FastAPI substitutes the Form default when a
    # multipart value is empty, so a defaulted key let an empty key silently encrypt
    # under the well-known DEFAULT_KEY (verified live). Required + strip() check
    # rejects empty (422) and whitespace-only (400) keys over real HTTP.
    key: str = Form(...),
):
    """Encrypt an uploaded image array using hyperchaotic permutation and diffusion.
    Computes security metrics for the encryption demonstration.
    """
    global _last_metrics

    if not key or not key.strip():
        raise HTTPException(status_code=400, detail="Encryption key must not be empty.")

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil, dtype=np.uint8)

        enc_np, meta = encrypt(img_np, key)

        # Compute and cache metrics for demonstration
        _last_metrics = compute_all_security_metrics(img_np, enc_np)
        _last_metrics["key_hash"] = meta["key_hash"]
        _last_metrics["image_shape"] = list(img_np.shape)

        buf = io.BytesIO()
        Image.fromarray(enc_np).save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"X-Encryption-KeyHash": meta["key_hash"]},
        )
    except Exception as e:
        logger.error("Encryption error: %s", e)
        raise HTTPException(status_code=400, detail=f"Encryption failed: {str(e)}")


@app.post("/decrypt")
async def decrypt_image(
    file: UploadFile = File(...),
    key: str = Form(...),  # Same rationale as /encrypt: avoid silent DEFAULT_KEY fallback.
):
    """Decrypt an encrypted image with the provided key (demonstration endpoint)."""
    if not key or not key.strip():
        raise HTTPException(status_code=400, detail="Decryption key must not be empty.")

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        enc_np = np.array(pil, dtype=np.uint8)

        restored_np = decrypt(enc_np, key)

        buf = io.BytesIO()
        Image.fromarray(restored_np).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Decryption error: %s", e)
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")


@app.post("/predict_plaintext")
async def predict_plaintext(file: UploadFile = File(...)):
    """Classify a plaintext image using the audited standard ViT baseline."""
    if _baseline_model is None:
        raise HTTPException(
            status_code=503,
            detail="Standard ViT baseline model is not loaded. Ensure the audited checkpoint exists.",
        )

    try:
        img_bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_size = _baseline_cfg.get("image_size", 96) if _baseline_cfg else 96
        tensor = _pil_to_eval_tensor(pil, image_size=img_size)

        result = _classify_tensor(_baseline_model, tensor)
        result["model"] = "Standard ViT (vit_tiny)"
        result["checkpoint"] = _baseline_ckpt_path
        result["input_resolution"] = f"{img_size}x{img_size}"
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict")
async def predict_encrypted(
    file: UploadFile = File(...),
    key: str = Form(default=DEFAULT_KEY),
    encrypt_first: bool = Form(default=True),
):
    """CryptoViT prediction route.
    Strictly blocked if no verified CryptoViT checkpoint has been trained and validated.
    """
    if _cryptovit_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "CryptoViT model is not available for trained inference. "
                "CryptoViT training and evaluation are pending validation. "
                "No randomly initialized model will be used."
            ),
        )

    # In the future when CryptoViT checkpoint is available:
    raise HTTPException(status_code=501, detail="CryptoViT inference route pending validation.")


@app.get("/metrics")
async def get_metrics():
    """Return security and statistical metrics from the last /encrypt operation."""
    if not _last_metrics:
        raise HTTPException(
            status_code=404,
            detail="No metrics available. Please run an encryption via /encrypt first.",
        )
    return JSONResponse(content=json.loads(json.dumps(_last_metrics, default=float)))
