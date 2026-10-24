# CryptoViT

> **Chaos-Assisted Vision Transformer for Privacy-Preserving Classification of Encrypted Images**

Final-year B.Tech capstone project.

## Quick Start

```bash
# Install the package in editable mode with development dependencies
pip install -e ".[dev,api,dashboard]"

# Run unit test suite (fast, offline)
pytest tests/ -v

# Run integration tests (downloads CIFAR-10)
pytest tests/integration/ --run-integration -v

# Run FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run Streamlit dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
CryptoViT/
├── configs/           YAML configurations (base, CIFAR-100, experiments)
├── data/              Dataset storage (raw, processed, encrypted)
├── src/cryptovit/     Main Python package
│   ├── datasets/      CIFAR-10/100 loading, transforms & encrypted dataset
│   ├── encryption/    4D hyperchaotic permutation + DNA diffusion engine
│   ├── models/        ViT baseline, CryptoPatchEmbedding, KeyAwarePE, CryptoViT
│   ├── training/      Trainer, AMP mixed precision, AdamW cosine warmup
│   ├── evaluation/    Accuracy metrics, confusion matrix & security metrics
│   ├── inference/     Inference pipeline
│   └── utils/         Config management, logging, seeding, visualisation
├── scripts/           Phase scripts (dataset check, training, ablation, download)
├── experiments/       Checkpoints, logs, evaluation results
├── tests/             Pytest test suite (unit tests and integration tests)
├── api/               FastAPI REST backend with encryption & classification
└── dashboard/         Streamlit interactive web demonstration app
```

## Features & Implementation

- **4D Hyperchaotic Map & DNA Diffusion**: Lossless pixel-level scrambling followed by DNA-XOR and chained-XOR diffusion (`decrypt(encrypt(x)) == x` exactly).
  - **Key sensitivity** is high: ciphertexts under keys that differ by one bit differ in about 99.7% of pixels (UACI about 33.9%) on a 32x32 test image.
  - **Plaintext sensitivity is limited.** Changing one plaintext pixel changed only about 1.6% (corner pixel) to 8% (middle pixel) of ciphertext pixels in the same test, because the diffusion is per-pixel with a forward-only chain. The textbook NPCR > 99% / UACI ~ 33.46% targets are therefore *not* met by this test; see `plaintext_sensitivity` in `cryptovit.evaluation.security_metrics`.
  - **Entropy** of a 32x32 channel cannot exceed about 7.82 even for ideal noise; the usual "> 7.99" applies to images of roughly 256x256 or larger (`expected_uniform_entropy` gives the reference for any size).
  - These are statistical checks, not a security proof. The scheme is a research prototype and should not be used to protect real data.
- **CryptoViT Architecture**: Native Vision Transformer tailored for protected image classification with convolutional patch embedding front-end and key-aware positional encoding.
- **Robust Training Pipeline**: Cosine learning rate decay with linear warmup, mixed precision (AMP), gradient accumulation, and checkpoint manager.
- **Interactive Stack**: FastAPI REST inference engine and Streamlit UI for visual inspection and live security diagnostics.

## Hardware

Optimised for: **NVIDIA RTX 3050 Laptop GPU (4 GB)** / CPU fallback.
