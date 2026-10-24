#!/usr/bin/env python
"""
scripts/make_demo_pack.py
==========================
Pick live-demo images from real model predictions and build a picture pack.

    python scripts/make_demo_pack.py --block 2 --variant D --replace-demo-images

What it does
------------
1. Scores test images of the chosen classes with the plaintext model (A) and an
   encrypted-image model (default D, the block-encrypted CryptoViT).
2. Selects, per class, the test image that BOTH models classify correctly with
   the highest confidence (the safe live-demo images), and one honest failure:
   an image the plaintext model gets right but the encrypted-image model gets wrong.
3. Writes the selected 28x28 originals to ``demo_images/`` (what the dashboard
   offers in its dropdown) and, in ``presentation/``, enlarged pictures of each
   image: original, per-pixel encrypted, block encrypted, and a side-by-side strip,
   plus ``SELECTION.md`` with the real predictions.

Everything runs on the CPU so it does not disturb training. Research prototype;
images are from the public DermaMNIST dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("make_demo_pack")

DEFAULT_CLASSES = ["melanoma", "melanocytic nevi", "basal cell carcinoma"]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def choose_images(records: Sequence[Dict], classes: Sequence[str],
                  min_confidence: float = 0.0) -> Dict:
    """Pick one safe image per class and one honest failure.

    ``records``: dicts with ``index``, ``label`` (class name), ``plain_pred``,
    ``plain_conf``, ``enc_pred``, ``enc_conf``.
    Safe = both models correct; best by the lower of the two confidences.
    Failure = plaintext model correct, encrypted model wrong; best by the
    encrypted model's (wrong) confidence, i.e. the most visible mistake.
    """
    chosen: Dict[str, Dict] = {}
    for cls in classes:
        pool = [
            r for r in records
            if r["label"] == cls and r["plain_pred"] == cls and r["enc_pred"] == cls
            and min(r["plain_conf"], r["enc_conf"]) >= min_confidence
        ]
        if pool:
            chosen[cls] = max(pool, key=lambda r: min(r["plain_conf"], r["enc_conf"]))
    failures = [
        r for r in records
        if r["label"] in classes and r["plain_pred"] == r["label"] and r["enc_pred"] != r["label"]
    ]
    failure = max(failures, key=lambda r: r["enc_conf"]) if failures else None
    return {"safe": chosen, "failure": failure}


def demo_filename(stem: str, role: str, position: int) -> str:
    """File name for the dashboard dropdown, which lists files alphabetically.

    Safe images get 1_, 2_, 3_ ... so they come first; the honest failure gets 9_
    so it is last and nobody picks it by accident at the start of a demo.
    """
    prefix = 9 if role == "failure" else position
    return f"{prefix}_{stem}.png"


def enlarge(img: Image.Image, scale: int = 8) -> Image.Image:
    """Nearest-neighbour enlargement so tiny images stay crisp on a projector."""
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def triptych(panels: Sequence[Image.Image], titles: Sequence[str], caption: str = "",
             gap: int = 16, title_h: int = 28, caption_h: int = 30) -> Image.Image:
    """Side-by-side strip of equally sized panels with a title above each."""
    if len(panels) != len(titles):
        raise ValueError("panels and titles must have the same length")
    w, h = panels[0].size
    total_w = gap + len(panels) * (w + gap)
    total_h = title_h + h + (caption_h if caption else 0) + gap
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x = gap + i * (w + gap)
        draw.text((x, 8), title, fill=(0, 0, 0))
        canvas.paste(panel.convert("RGB"), (x, title_h))
    if caption:
        draw.text((gap, title_h + h + 8), caption, fill=(0, 0, 0))
    return canvas


# ---------------------------------------------------------------------------
# Scoring and output
# ---------------------------------------------------------------------------

def score_images(service, test_ds, class_names: Sequence[str], classes: Sequence[str],
                 per_class: int, key: str, variant: str) -> List[Dict]:
    """Predict with A (plaintext) and an encrypted-image model for test images."""
    records = []
    for cls in classes:
        class_id = list(class_names).index(cls)
        indices = [i for i, t in enumerate(test_ds.targets) if t == class_id][:per_class]
        for i in indices:
            arr = np.array(test_ds[i][0])
            plain = service.predict_plaintext(arr)
            enc = service.predict_encrypted(arr, key, variant=variant)
            records.append({
                "index": i, "label": cls,
                "plain_pred": plain["class_name"], "plain_conf": plain["confidence"],
                "enc_pred": enc["class_name"], "enc_conf": enc["confidence"],
            })
    return records


def write_pack(rec: Dict, role: str, test_ds, key: str, block: int, out_dir: Path,
               demo_dir: Optional[Path], position: int = 1) -> Dict:
    """Write the originals and the enlarged pictures for one selected image."""
    from cryptovit.encryption import encrypt

    arr = np.array(test_ds[rec["index"]][0])
    stem = f"{role}_{slug(rec['label'])}_test{rec['index']}"
    pixel_enc, _ = encrypt(arr, key, scheme="v1")
    block_enc, _ = encrypt(arr, key, scheme=f"block{block}")

    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [Image.fromarray(a) for a in (arr, pixel_enc, block_enc)]
    names = ["original", "per_pixel_encrypted", f"block{block}_encrypted"]
    for panel, name in zip(panels, names):
        enlarge(panel).save(out_dir / f"{stem}_{name}.png")
    strip = triptych(
        [enlarge(p) for p in panels],
        ["Original", "Per-pixel encryption", f"Block encryption ({block}x{block})"],
        caption=f"True class: {rec['label']}.  Plaintext model: {rec['plain_pred']} "
                f"({rec['plain_conf'] * 100:.0f}%).  Block-encrypted model: "
                f"{rec['enc_pred']} ({rec['enc_conf'] * 100:.0f}%).",
    )
    strip.save(out_dir / f"{stem}_side_by_side.png")
    if demo_dir is not None:
        demo_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(demo_dir / demo_filename(stem, role, position))
    return {**rec, "role": role, "file_stem": stem}


def write_selection_md(rows: List[Dict], out_dir: Path, variant: str, block: int,
                       n_scored: int) -> None:
    lines = [
        "# Demo image selection",
        "",
        f"Chosen from {n_scored} DermaMNIST test images by comparing the plaintext model (A) "
        f"with the block{block}-encrypted model (variant {variant}). Predictions below are real, "
        "from the saved checkpoints.",
        "",
        "| Role | File | True class | Plaintext model | Encrypted-image model |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        role = "safe (both correct)" if r["role"] == "safe" else "honest failure"
        lines.append(
            f"| {role} | `{r['file_stem']}` | {r['label']} | "
            f"{r['plain_pred']} ({r['plain_conf'] * 100:.0f}%) | "
            f"{r['enc_pred']} ({r['enc_conf'] * 100:.0f}%) |"
        )
    lines += ["", "Images are 28x28 test images from the public DermaMNIST dataset, "
                  "enlarged 8x for display. Research prototype; not for clinical use.", ""]
    (out_dir / "SELECTION.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pick demo images and build the picture pack.")
    p.add_argument("--block", type=int, default=2, help="block size of the block scheme")
    p.add_argument("--variant", default="D", choices=["B", "C", "D"])
    p.add_argument("--experiments", default=None,
                   help="experiments dir (default: experiments/dermamnist_block<B>)")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    p.add_argument("--per-class", type=int, default=30, help="test images scored per class")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--key", default="CryptoViT_Research_Key_2025")
    p.add_argument("--out", default="presentation")
    p.add_argument("--demo-dir", default="demo_images")
    p.add_argument("--replace-demo-images", action="store_true",
                   help="delete the old PNGs in the demo dir before writing the selection")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("cryptovit").setLevel(logging.WARNING)

    import torch
    from cryptovit.datasets import get_class_names
    from cryptovit.datasets.medmnist_dataset import load_medmnist_splits
    from cryptovit.inference.pipeline import ClassifierService
    from cryptovit.utils import load_config

    experiments = args.experiments or f"experiments/dermamnist_block{args.block}"
    cfg = load_config(overrides=["config=dermamnist", f"config=block{args.block}",
                                 "hardware.device=cpu", f"paths.experiments_dir={experiments}",
                                 "dataset.num_workers=0"])
    service = ClassifierService(cfg, torch.device("cpu"),
                                PROJECT_ROOT / experiments / "ablation", args.key)
    service.load()
    for needed in ("A", args.variant):
        if not service.trained.get(needed):
            raise SystemExit(f"Variant {needed} has no trained checkpoint in {experiments}.")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    names = get_class_names("dermamnist")
    unknown = [c for c in classes if c not in names]
    if unknown:
        raise SystemExit(f"Unknown classes {unknown}; choose from {names}.")
    test_ds = load_medmnist_splits("dermamnist", PROJECT_ROOT / "data/raw/medmnist", 28,
                                   download=False)[2]

    records = score_images(service, test_ds, names, classes, args.per_class, args.key, args.variant)
    picked = choose_images(records, classes, args.min_confidence)
    missing = [c for c in classes if c not in picked["safe"]]
    if missing:
        logger.warning("No image correct for both models for: %s", missing)

    out_dir, demo_dir = PROJECT_ROOT / args.out, PROJECT_ROOT / args.demo_dir
    if args.replace_demo_images:
        # Remove files from earlier runs so the folders match SELECTION.md exactly.
        for folder, patterns in ((demo_dir, ["*.png"]), (out_dir, ["safe_*.png", "failure_*.png"])):
            for pattern in patterns:
                for old in folder.glob(pattern) if folder.is_dir() else []:
                    old.unlink()
    rows = [write_pack(r, "safe", test_ds, args.key, args.block, out_dir, demo_dir, position=i)
            for i, r in enumerate(picked["safe"].values(), start=1)]
    if picked["failure"]:
        rows.append(write_pack(picked["failure"], "failure", test_ds, args.key, args.block,
                               out_dir, demo_dir))
    else:
        logger.warning("No failure case found among the scored images.")
    write_selection_md(rows, out_dir, args.variant, args.block, len(records))
    (out_dir / "selection.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        logger.info("%-8s test #%-4d %-22s plain=%s (%.0f%%)  enc=%s (%.0f%%)", r["role"],
                    r["index"], r["label"], r["plain_pred"], r["plain_conf"] * 100,
                    r["enc_pred"], r["enc_conf"] * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
