from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from cryptovit.encryption import decrypt, encrypt
from cryptovit.evaluation.security_metrics import compute_all_security_metrics
from cryptovit.inference.pipeline import ClassifierService, read_results
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_cfg = None
_device = None
_service: Optional[ClassifierService] = None
_ablation_dir: Optional[Path] = None
_last_metrics: Dict = {}

ENCRYPTION_KEY = "CryptoViT_Research_Key_2025"


def _config_overrides() -> list:
    overrides = []
    for name in os.environ.get("CRYPTOVIT_CONFIG", "").split(","):
        if name.strip():
            overrides.append(f"config={name.strip()}")
    if os.environ.get("CRYPTOVIT_EXPERIMENTS"):
        overrides.append(f"paths.experiments_dir={os.environ['CRYPTOVIT_EXPERIMENTS']}")
    if os.environ.get("CRYPTOVIT_DEVICE"):
        overrides.append(f"hardware.device={os.environ['CRYPTOVIT_DEVICE']}")
    return overrides


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _device, _service, _ablation_dir

    _cfg = load_config(overrides=_config_overrides() or None)
    setup_logging(level=_cfg.logging.level)
    _device = get_device(_cfg)
    logger.info("CryptoViT API starting up on device: %s", _device)

    _ablation_dir = PROJECT_ROOT / _cfg.paths.experiments_dir / "ablation"
    _service = ClassifierService(
        _cfg,
        _device,
        _ablation_dir,
        ENCRYPTION_KEY,
        preferred_variant=os.environ.get("CRYPTOVIT_VARIANT", "D").upper(),
    )
    _service.load()
    yield


app = FastAPI(
    title="CryptoViT API",
    description=(
        "Privacy-preserving image classification via chaos-assisted encryption "
        "and Vision Transformer (CryptoViT). Research prototype; not for clinical use."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


def _pil_to_numpy(img: Image.Image, target_size: int = 32) -> np.ndarray:
    img = img.resize((target_size, target_size), Image.LANCZOS).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _image_to_bytes(img_np: np.ndarray) -> bytes:
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _load_upload(img_bytes: bytes) -> np.ndarray:
    return _pil_to_numpy(Image.open(io.BytesIO(img_bytes)), _service.native_size)


def _require_service() -> ClassifierService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialised.")
    return _service


@app.get("/health")
async def health():
    svc = _require_service()
    status = svc.status()
    return {
        "status": "ok",
        "device": str(_device),
        "dataset": _cfg.dataset.name,
        "class_names": svc.class_names,
        "image_size": svc.native_size,
        "encryption_scheme": svc.scheme,
        "baseline_loaded": status["A"]["loaded"],
        "cryptovit_loaded": status["D"]["loaded"],
        "variants": status,
        "disclaimer": "Research prototype. Not a medical device; not for clinical use.",
    }


@app.post("/encrypt")
async def encrypt_image(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
):
    global _last_metrics
    svc = _require_service()

    try:
        img_np = _load_upload(await file.read())
        enc_np, meta = encrypt(img_np, key, svc.transient_steps, scheme=svc.scheme)

        _last_metrics = compute_all_security_metrics(
            img_np,
            enc_np,
            key=key,
            encrypt_fn=lambda img, k: encrypt(img, k, svc.transient_steps, scheme=svc.scheme),
        )
        _last_metrics["key_hash"] = meta["key_hash"]

        return Response(
            content=_image_to_bytes(enc_np),
            media_type="image/png",
            headers={"X-Encryption-KeyHash": meta["key_hash"]},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/decrypt")
async def decrypt_image(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
):
    svc = _require_service()
    try:
        pil = Image.open(io.BytesIO(await file.read()))
        enc_np = np.array(pil.convert("RGB"), dtype=np.uint8)
        restored_np = decrypt(enc_np, key, None, svc.transient_steps, scheme=svc.scheme)
        return Response(content=_image_to_bytes(restored_np), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict")
async def predict_encrypted(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
    encrypt_first: bool = Form(default=True),
    variant: Optional[str] = Form(default=None),
):
    svc = _require_service()
    try:
        img_np = _load_upload(await file.read())
        return JSONResponse(
            content=svc.predict_encrypted(
                img_np,
                key,
                variant=variant.upper() if variant else None,
                encrypt_first=encrypt_first,
            )
        )
    except KeyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict_plaintext")
async def predict_plaintext(file: UploadFile = File(...)):
    svc = _require_service()
    try:
        return JSONResponse(content=svc.predict_plaintext(_load_upload(await file.read())))
    except KeyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/compare")
async def compare_variants(
    file: UploadFile = File(...),
    key: str = Form(default=ENCRYPTION_KEY),
):
    svc = _require_service()
    try:
        return JSONResponse(content=svc.compare(_load_upload(await file.read()), key))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/results")
async def get_results():
    _require_service()
    results = read_results(_ablation_dir)
    if not results:
        raise HTTPException(status_code=404, detail="No trained-variant results found yet.")
    return JSONResponse(content=json.loads(json.dumps(results, default=float)))


@app.get("/metrics")
async def get_metrics():
    if not _last_metrics:
        raise HTTPException(status_code=404, detail="No metrics available. Call /encrypt first.")
    return JSONResponse(content=json.loads(json.dumps(_last_metrics, default=float)))


@app.get("/model_info")
async def model_info():
    svc = _require_service()
    info = {}
    for name, model in svc.models.items():
        info[name] = {
            "label": svc.status()[name]["label"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "trained": svc.trained.get(name, False),
        }
    if "A" in info:
        info["baseline_vit"] = {"parameters": info["A"]["parameters"]}
    if "D" in info:
        info["cryptovit"] = {"parameters": info["D"]["parameters"]}
    return JSONResponse(content=info)
