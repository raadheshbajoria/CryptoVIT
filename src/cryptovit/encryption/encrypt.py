from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from cryptovit.encryption.block_scheme import (
    block_decrypt,
    block_encrypt,
    parse_block_scheme,
)
from cryptovit.encryption.diffusion import diffuse, inverse_diffuse
from cryptovit.encryption.key_generator import KeyGenerator
from cryptovit.encryption.permutation import (
    inverse_permute_image,
    permute_image,
)

logger = logging.getLogger(__name__)


SCHEMES = ("v1", "v2")
DEFAULT_SCHEME = "v1"


def _check_scheme(scheme: str) -> str:
    if scheme not in SCHEMES and parse_block_scheme(scheme) is None:
        raise ValueError(
            f"Unknown encryption scheme '{scheme}'; choose from {SCHEMES} "
            "or 'block<B>' (e.g. 'block2')."
        )
    return scheme


@lru_cache(maxsize=8)
def _key_streams(
    key: bytes,
    n: int,
    transient_steps: int,
    channels: int = 1,
    scheme: str = DEFAULT_SCHEME,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kg = KeyGenerator(key, transient_steps=transient_steps)
    if scheme == "v2":
        perm = kg.channel_permutations(n, channels)
        keystream = kg.channel_keystreams(n, channels)
    else:
        perm = kg.pixel_permutation(n)
        keystream = kg.diffusion_keystream(n)
    rules = kg.dna_rule_sequence(n)
    for arr in (perm, keystream, rules):
        arr.setflags(write=False)
    return perm, keystream, rules


def _streams_for(
    key: str | bytes, image: np.ndarray, transient_steps: int, scheme: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = image.shape[0] * image.shape[1]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return _key_streams(
        _as_key_bytes(key),
        n,
        transient_steps,
        channels if scheme == "v2" else 1,
        scheme,
    )


def _as_key_bytes(key: str | bytes) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else key


def encrypt(
    image: np.ndarray,
    key: str | bytes,
    transient_steps: int = 1000,
    scheme: str = DEFAULT_SCHEME,
) -> Tuple[np.ndarray, Dict]:
    if image.dtype != np.uint8:
        raise TypeError(f"Image must be uint8; got {image.dtype}.")
    _check_scheme(scheme)

    key_bytes = _as_key_bytes(key)
    block = parse_block_scheme(scheme)
    if block is not None:
        encrypted = block_encrypt(image, key_bytes, block, transient_steps)
        meta = {
            "shape": image.shape,
            "key_hash": hashlib.sha256(key_bytes).hexdigest(),
            "scheme": scheme,
        }
        logger.debug("Encrypted image shape=%s scheme=%s", image.shape, scheme)
        return encrypted, meta

    perm, ks, rules = _streams_for(key_bytes, image, transient_steps, scheme)

    permuted = permute_image(image, perm)

    encrypted = diffuse(permuted, ks, rules)

    meta = {
        "shape": image.shape,
        "key_hash": hashlib.sha256(key_bytes).hexdigest(),
        "scheme": scheme,
    }

    logger.debug("Encrypted image shape=%s scheme=%s", image.shape, scheme)
    return encrypted, meta


def decrypt(
    encrypted: np.ndarray,
    key: str | bytes,
    meta: Dict | None = None,
    transient_steps: int = 1000,
    scheme: str | None = None,
) -> np.ndarray:
    if encrypted.dtype != np.uint8:
        raise TypeError(f"Encrypted array must be uint8; got {encrypted.dtype}.")
    if scheme is None:
        scheme = (meta or {}).get("scheme", DEFAULT_SCHEME)
    _check_scheme(scheme)

    block = parse_block_scheme(scheme)
    if block is not None:
        return block_decrypt(encrypted, _as_key_bytes(key), block, transient_steps)

    perm, ks, rules = _streams_for(key, encrypted, transient_steps, scheme)

    undiffused = inverse_diffuse(encrypted, ks, rules)

    restored = inverse_permute_image(undiffused, perm)

    logger.debug("Decrypted image shape=%s", restored.shape)
    return restored


def verify_roundtrip(
    image: np.ndarray,
    key: str | bytes,
    transient_steps: int = 1000,
    scheme: str = DEFAULT_SCHEME,
) -> bool:
    encrypted, meta = encrypt(image, key, transient_steps, scheme)
    restored = decrypt(encrypted, key, meta, transient_steps)
    ok = bool(np.array_equal(image, restored))
    if ok:
        logger.info("Round-trip verification PASSED.")
    else:
        diff = np.sum(image != restored)
        logger.error("Round-trip verification FAILED! %d pixels differ.", diff)
    return ok
