"""
cryptovit/evaluation/security_metrics.py
==========================================
Image-encryption security metrics.

Metrics implemented
-------------------
* Shannon entropy (per channel and overall)
* Histogram uniformity (chi-squared statistic and visual check)
* Adjacent-pixel correlation (horizontal, vertical, diagonal)
* NPCR  — Number of Pixels Change Rate
* UACI  — Unified Average Changing Intensity
* Key sensitivity — divergence after a tiny key perturbation
* Encryption/decryption time

These metrics are standard in the image-encryption literature.
Reference values for a good encryption scheme:
  Entropy  > 7.99  (max 8.0 for 8-bit images)
  NPCR     > 99%   (ideally ~99.6%)
  UACI     ~ 33%   (ideal 33.46%)
  Correlation ≈ 0  (should be < 0.01 in absolute value)

References
----------
Chai et al. (2017), Signal Processing: Image Communication.
Zhu & Zhu (2020), Entropy.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shannon Entropy
# ---------------------------------------------------------------------------

def compute_entropy(image: np.ndarray) -> Dict[str, float]:
    """Compute Shannon entropy of each channel and overall.

    Parameters
    ----------
    image : (H, W) or (H, W, C) uint8 array

    Returns
    -------
    dict with keys ``channel_{c}`` for each channel, and ``overall``.
    """
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
    return results


# ---------------------------------------------------------------------------
# Histogram Uniformity
# ---------------------------------------------------------------------------

def histogram_analysis(
    image: np.ndarray,
) -> Dict[str, float]:
    """Compute chi-squared statistic against a uniform histogram.

    For a perfectly encrypted image each intensity level appears equally
    often → chi-squared ≈ 0.

    Parameters
    ----------
    image : (H, W) or (H, W, C) uint8 array

    Returns
    -------
    dict with ``chi_squared_channel_{c}`` and ``chi_squared_overall``.
    """
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


# ---------------------------------------------------------------------------
# Adjacent-Pixel Correlation
# ---------------------------------------------------------------------------

def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
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
    """Measure horizontal, vertical, and diagonal adjacent-pixel correlations.

    Parameters
    ----------
    image : (H, W) or (H, W, C) uint8 array
    n_samples : int
        Number of random pixel pairs to sample (for speed).
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        horizontal, vertical, diagonal  (per channel + overall)
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    H, W, C = image.shape
    rng = np.random.default_rng(seed)

    results = {}
    h_all, v_all, d_all = [], [], []

    for c in range(C):
        ch = image[:, :, c].astype(np.float64)

        # Horizontal pairs
        rs_h = rng.integers(0, H, n_samples)
        cs_h = rng.integers(0, W - 1, n_samples)
        hcorr = _pearson_correlation(ch[rs_h, cs_h], ch[rs_h, cs_h + 1])

        # Vertical pairs
        rs_v = rng.integers(0, H - 1, n_samples)
        cs_v = rng.integers(0, W, n_samples)
        vcorr = _pearson_correlation(ch[rs_v, cs_v], ch[rs_v + 1, cs_v])

        # Diagonal pairs
        rs_d = rng.integers(0, H - 1, n_samples)
        cs_d = rng.integers(0, W - 1, n_samples)
        dcorr = _pearson_correlation(ch[rs_d, cs_d], ch[rs_d + 1, cs_d + 1])

        results[f"horizontal_c{c}"] = hcorr
        results[f"vertical_c{c}"]   = vcorr
        results[f"diagonal_c{c}"]   = dcorr
        h_all.append(hcorr)
        v_all.append(vcorr)
        d_all.append(dcorr)

    results["horizontal"] = float(np.mean(h_all))
    results["vertical"]   = float(np.mean(v_all))
    results["diagonal"]   = float(np.mean(d_all))
    return results


# ---------------------------------------------------------------------------
# NPCR — Number of Pixels Change Rate
# ---------------------------------------------------------------------------

def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute NPCR between two images encrypted with near-identical keys.

    NPCR = (number of pixels that differ) / (total pixels) × 100%

    Parameters
    ----------
    img1, img2 : same-shape uint8 arrays

    Returns
    -------
    float in [0, 100]
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = (img1 != img2).astype(np.float64)
    return float(diff.mean() * 100.0)


# ---------------------------------------------------------------------------
# UACI — Unified Average Changing Intensity
# ---------------------------------------------------------------------------

def uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute UACI between two images.

    UACI = mean(|c1 - c2| / 255) × 100%
    Ideal value for a good cipher is ~33.46%.

    Parameters
    ----------
    img1, img2 : same-shape uint8 arrays

    Returns
    -------
    float in [0, 100]
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    return float(diff.mean() / 255.0 * 100.0)


# ---------------------------------------------------------------------------
# Key Sensitivity
# ---------------------------------------------------------------------------

def key_sensitivity(
    image: np.ndarray,
    key: str,
    encrypt_fn: Callable,
    delta_chars: int = 1,
) -> Dict[str, float]:
    """Measure how much the encrypted output changes when the key changes slightly.

    Parameters
    ----------
    image : uint8 array
    key : str — original encryption key
    encrypt_fn : callable(image, key) → encrypted_image
        Should accept an image and key string and return the encrypted array.
    delta_chars : int
        Number of characters to flip at the end of the key.

    Returns
    -------
    dict with ``npcr`` and ``uaci`` between encryptions under key and key+delta.
    """
    enc1 = encrypt_fn(image, key)

    # Slightly different key: flip the last character
    key_modified = key[:-delta_chars] + chr(ord(key[-1]) ^ 1)
    enc2 = encrypt_fn(image, key_modified)

    return {
        "npcr": npcr(enc1, enc2),
        "uaci": uaci(enc1, enc2),
    }


# ---------------------------------------------------------------------------
# Encryption / Decryption Timing
# ---------------------------------------------------------------------------

def measure_encryption_time(
    image: np.ndarray,
    encrypt_fn: Callable,
    decrypt_fn: Callable,
    key: str,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Measure average encryption and decryption time.

    Parameters
    ----------
    image : uint8 array
    encrypt_fn : callable(image, key) → (encrypted, meta)
    decrypt_fn : callable(encrypted, key, meta) → image
    key : str
    n_trials : int

    Returns
    -------
    dict with ``encrypt_ms`` and ``decrypt_ms`` (milliseconds).
    """
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


# ---------------------------------------------------------------------------
# Combined: all metrics in one call
# ---------------------------------------------------------------------------

def compute_all_security_metrics(
    plaintext: np.ndarray,
    encrypted: np.ndarray,
    key: Optional[str] = None,
    encrypt_fn: Optional[Callable] = None,
) -> Dict:
    """Compute the full suite of security metrics.

    Parameters
    ----------
    plaintext : uint8 array — original image
    encrypted : uint8 array — encrypted image (same shape)
    key : str, optional — required for key_sensitivity test
    encrypt_fn : callable, optional — required for key_sensitivity test

    Returns
    -------
    dict with all metric sub-dicts merged together.
    """
    results: Dict = {}

    # Entropy of encrypted image (should be > 7.99)
    results["entropy"] = compute_entropy(encrypted)

    # Histogram uniformity
    results["histogram"] = histogram_analysis(encrypted)

    # Adjacent-pixel correlation of encrypted image (should be ~0)
    results["correlation_encrypted"] = adjacent_pixel_correlation(encrypted)

    # Adjacent-pixel correlation of plaintext (for comparison)
    results["correlation_plaintext"] = adjacent_pixel_correlation(plaintext)

    # NPCR / UACI between plaintext and encrypted (structural change)
    results["npcr_plain_vs_enc"] = npcr(plaintext, encrypted)
    results["uaci_plain_vs_enc"] = uaci(plaintext, encrypted)

    # Key sensitivity (optional)
    if key is not None and encrypt_fn is not None:
        ks = key_sensitivity(
            plaintext, key,
            encrypt_fn=lambda img, k: encrypt_fn(img, k)[0],
        )
        results["key_sensitivity"] = ks

    _log_security_summary(results)
    return results


def _log_security_summary(results: Dict) -> None:
    ent = results.get("entropy", {}).get("overall", 0)
    npcr_val = results.get("npcr_plain_vs_enc", 0)
    uaci_val = results.get("uaci_plain_vs_enc", 0)
    hcorr = results.get("correlation_encrypted", {}).get("horizontal", 0)

    logger.info(
        "Security | entropy=%.4f  npcr=%.2f%%  uaci=%.2f%%  h_corr=%.4f",
        ent, npcr_val, uaci_val, hcorr,
    )
