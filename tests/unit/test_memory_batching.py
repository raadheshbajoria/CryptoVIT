from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cryptovit.training.trainer import split_batch_for_memory
from cryptovit.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def _load_run_ablation():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_under_test", ROOT / "scripts" / "run_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "batch, accum, limit, expected",
    [
        (128, 1, 32, (32, 4)),
        (128, 2, 32, (32, 8)),
        (32, 1, 32, (32, 1)),
        (16, 3, 32, (16, 3)),
    ],
)
def test_split_batch_keeps_effective_batch(batch, accum, limit, expected):
    micro, steps = split_batch_for_memory(batch, accum, limit)
    assert (micro, steps) == expected
    if batch > limit:
        assert micro * steps == batch * accum


def test_split_batch_rejects_non_multiple():
    with pytest.raises(ValueError):
        split_batch_for_memory(100, 1, 32)


@pytest.mark.parametrize("args", [(0, 1, 32), (8, 0, 32), (8, 1, 0)])
def test_split_batch_rejects_non_positive(args):
    with pytest.raises(ValueError):
        split_batch_for_memory(*args)


def test_variant_cfg_only_changes_crypto_embedding_variants():
    ablation = _load_run_ablation()
    cfg = load_config()
    assert cfg.training.batch_size == 128

    for variant in ("A", "B"):
        v = ablation._variant_cfg(variant, cfg)
        assert (v.training.batch_size, v.training.accumulation_steps) == (128, 1)

    for variant in ("C", "D"):
        v = ablation._variant_cfg(variant, cfg)
        assert (v.training.batch_size, v.training.accumulation_steps) == (32, 4)

    assert cfg.training.batch_size == 128
    assert cfg.training.accumulation_steps == 1
