# CryptoViT

> **Chaos-Assisted Vision Transformer for Privacy-Preserving Classification of Encrypted Images**

Final-year B.Tech capstone project.

## Quick Start

```bash
# Install the package in editable mode
pip install -e ".[dev]"

# Run Phase 1 dataset sanity check (downloads CIFAR-10)
python scripts/phase1_dataset_check.py

# Run unit tests
pytest tests/ -v
```

## Project Structure

```
CryptoViT/
├── configs/           YAML configurations
├── data/              Dataset storage (gitignored)
├── src/cryptovit/     Main Python package
│   ├── datasets/      Dataset loading & preprocessing
│   ├── encryption/    Hyperchaotic encryption engine (Phase 3+)
│   ├── models/        ViT & CryptoViT (Phase 2+)
│   ├── training/      Training loops (Phase 2+)
│   ├── evaluation/    Metrics & security analysis (Phase 8+)
│   ├── inference/     Inference pipeline (Phase 17+)
│   └── utils/         Config, logging, seeding, visualisation
├── scripts/           Phase-specific run scripts
├── experiments/       Saved results, checkpoints, logs
├── tests/             pytest test suite
├── api/               FastAPI backend (Phase 17+)
├── dashboard/         Streamlit app (Phase 18+)
└── notebooks/         Exploratory notebooks
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset & preprocessing | ✅ |
| 2 | Standard ViT baseline | ⏳ |
| 3 | 4D hyperchaotic key generator | ⏳ |
| … | … | … |

## Hardware

Developed on: **NVIDIA RTX 3050 Laptop GPU (4 GB)**
