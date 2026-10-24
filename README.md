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

- **4D Hyperchaotic Map & DNA Diffusion**: Reversible pixel-level scrambling and DNA algebraic XOR diffusion yielding entropy > 7.99, NPCR > 99%, and UACI ~ 33.46%.
- **CryptoViT Architecture**: Native Vision Transformer tailored for protected image classification with convolutional patch embedding front-end and key-aware positional encoding.
- **Robust Training Pipeline**: Cosine learning rate decay with linear warmup, mixed precision (AMP), gradient accumulation, and checkpoint manager.
- **Interactive Stack**: FastAPI REST inference engine and Streamlit UI for visual inspection and live security diagnostics.

## Hardware

Optimised for: **NVIDIA RTX 3050 Laptop GPU (4 GB)** / CPU fallback.
