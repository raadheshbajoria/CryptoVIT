from __future__ import annotations

import logging
import math
import time
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def expected_uniform_entropy(n_samples: int, levels: int = 256) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return math.log2(levels) - (levels - 1) / (2.0 * n_samples * math.log(2))


def compute_entropy(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    C = image.shape[2]
    results = {}
    all_entropy = []

    for c in range(C):
        channel = image[:, :, c].ravel()
        hist, _ = np.histogram(channel, bins=256, range=(0, 255))
        hist = hist[hist > 0].astype(np.float64)
        total = hist.sum()
        probs = hist / total
        ent = float(-np.sum(probs * np.log2(probs)))
        results[f"channel_{c}"] = ent
        all_entropy.append(ent)

    results["overall"] = float(np.mean(all_entropy))
    results["expected_for_uniform_noise"] = expected_uniform_entropy(
        image.shape[0] * image.shape[1]
    )
    return results


def histogram_analysis(
    image: np.ndarray,
) -> Dict[str, float]:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    C = image.shape[2]
    n_pixels = image.shape[0] * image.shape[1]
    expected = n_pixels / 256.0
    results = {}
    chi_all = []

    for c in range(C):
        channel = image[:, :, c].ravel()
        hist, _ = np.histogram(channel, bins=256, range=(0, 255))
        chi2 = float(np.sum((hist - expected) ** 2 / expected))
        results[f"chi_squared_channel_{c}"] = chi2
        chi_all.append(chi2)

    results["chi_squared_overall"] = float(np.mean(chi_all))
    return results


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    num = np.mean((a - ma) * (b - mb))
    den = np.sqrt(np.mean((a - ma) ** 2) * np.mean((b - mb) ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def adjacent_pixel_correlation(
    image: np.ndarray,
    n_samples: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    H, W, C = image.shape
    rng = np.random.default_rng(seed)

    results = {}
    h_all, v_all, d_all = [], [], []

    for c in range(C):
        ch = image[:, :, c].astype(np.float64)

        rs_h = rng.integers(0, H, n_samples)
        cs_h = rng.integers(0, W - 1, n_samples)
        hcorr = _pearson_correlation(ch[rs_h, cs_h], ch[rs_h, cs_h + 1])

        rs_v = rng.integers(0, H - 1, n_samples)
        cs_v = rng.integers(0, W, n_samples)
        vcorr = _pearson_correlation(ch[rs_v, cs_v], ch[rs_v + 1, cs_v])

        rs_d = rng.integers(0, H - 1, n_samples)
        cs_d = rng.integers(0, W - 1, n_samples)
        dcorr = _pearson_correlation(ch[rs_d, cs_d], ch[rs_d + 1, cs_d + 1])

        results[f"horizontal_c{c}"] = hcorr
        results[f"vertical_c{c}"] = vcorr
        results[f"diagonal_c{c}"] = dcorr
        h_all.append(hcorr)
        v_all.append(vcorr)
        d_all.append(dcorr)

    results["horizontal"] = float(np.mean(h_all))
    results["vertical"] = float(np.mean(v_all))
    results["diagonal"] = float(np.mean(d_all))
    return results


def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = (img1 != img2).astype(np.float64)
    return float(diff.mean() * 100.0)


def uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    return float(diff.mean() / 255.0 * 100.0)


def key_sensitivity(
    image: np.ndarray,
    key: str,
    encrypt_fn: Callable,
    delta_chars: int = 1,
) -> Dict[str, float]:
    enc1 = encrypt_fn(image, key)

    key_modified = key[:-delta_chars] + chr(ord(key[-1]) ^ 1)
    enc2 = encrypt_fn(image, key_modified)

    return {
        "npcr": npcr(enc1, enc2),
        "uaci": uaci(enc1, enc2),
    }


def plaintext_sensitivity(
    image: np.ndarray,
    key: str,
    encrypt_fn: Callable,
    n_trials: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    baseline = encrypt_fn(image, key)
    rng = np.random.default_rng(seed)
    npcrs, uacis = [], []
    for _ in range(n_trials):
        idx = tuple(int(rng.integers(0, s)) for s in image.shape)
        modified = image.copy()
        modified[idx] ^= np.uint8(1)
        other = encrypt_fn(modified, key)
        npcrs.append(npcr(baseline, other))
        uacis.append(uaci(baseline, other))
    return {
        "npcr": float(np.mean(npcrs)),
        "uaci": float(np.mean(uacis)),
        "npcr_min": float(np.min(npcrs)),
        "npcr_max": float(np.max(npcrs)),
        "n_trials": n_trials,
    }


def measure_encryption_time(
    image: np.ndarray,
    encrypt_fn: Callable,
    decrypt_fn: Callable,
    key: str,
    n_trials: int = 5,
) -> Dict[str, float]:
    enc_times = []
    dec_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        encrypted, meta = encrypt_fn(image, key)
        enc_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        decrypt_fn(encrypted, key, meta)
        dec_times.append((time.perf_counter() - t0) * 1000)

    return {
        "encrypt_ms": float(np.mean(enc_times)),
        "decrypt_ms": float(np.mean(dec_times)),
    }


def compute_all_security_metrics(
    plaintext: np.ndarray,
    encrypted: np.ndarray,
    key: Optional[str] = None,
    encrypt_fn: Optional[Callable] = None,
) -> Dict:
    results: Dict = {}

    results["entropy"] = compute_entropy(encrypted)

    results["histogram"] = histogram_analysis(encrypted)

    results["correlation_encrypted"] = adjacent_pixel_correlation(encrypted)

    results["correlation_plaintext"] = adjacent_pixel_correlation(plaintext)

    results["npcr_plain_vs_enc"] = npcr(plaintext, encrypted)
    results["uaci_plain_vs_enc"] = uaci(plaintext, encrypted)

    if key is not None and encrypt_fn is not None:
        cipher_only = lambda img, k: encrypt_fn(img, k)[0]  # noqa: E731

        results["key_sensitivity"] = key_sensitivity(plaintext, key, encrypt_fn=cipher_only)
        results["plaintext_sensitivity"] = plaintext_sensitivity(
            plaintext, key, encrypt_fn=cipher_only
        )

    _log_security_summary(results)
    return results


def _log_security_summary(results: Dict) -> None:
    ent = results.get("entropy", {}).get("overall", 0)
    ent_ref = results.get("entropy", {}).get("expected_for_uniform_noise", 0)
    hcorr = results.get("correlation_encrypted", {}).get("horizontal", 0)
    ps = results.get("plaintext_sensitivity")

    msg = "Security | entropy=%.4f (uniform-noise ref %.4f)  h_corr=%.4f"
    args = [ent, ent_ref, hcorr]
    if ps:
        msg += "  plaintext-sensitivity npcr=%.2f%% uaci=%.2f%%"
        args += [ps["npcr"], ps["uaci"]]
    logger.info(msg, *args)
