from __future__ import annotations

import hashlib
import logging
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = dict(a=35.0, b=3.0, d=28.0, r=-1.0)

_INIT_SCALE = 15.0


def _ode(
    state: np.ndarray,
    a: float,
    b: float,
    d: float,
    r: float,
) -> np.ndarray:
    x, y, z, w = state
    dx = a * (y - x)
    dy = d * x - x * z + w
    dz = x * y - b * z
    dw = -y * z + r * w
    return np.array([dx, dy, dz, dw], dtype=np.float64)


def _rk4_step(
    state: np.ndarray,
    h: float,
    a: float,
    b: float,
    d: float,
    r: float,
) -> np.ndarray:
    k1 = _ode(state, a, b, d, r)
    k2 = _ode(state + h / 2 * k1, a, b, d, r)
    k3 = _ode(state + h / 2 * k2, a, b, d, r)
    k4 = _ode(state + h * k3, a, b, d, r)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class HyperchaosSystem:
    def __init__(
        self,
        key: bytes | str,
        params: Optional[dict] = None,
        h: float = 0.01,
        transient_steps: int = 1000,
    ):
        if isinstance(key, str):
            key = key.encode("utf-8")

        p = params or _DEFAULT_PARAMS
        self.a: float = float(p.get("a", _DEFAULT_PARAMS["a"]))
        self.b: float = float(p.get("b", _DEFAULT_PARAMS["b"]))
        self.d: float = float(p.get("d", _DEFAULT_PARAMS["d"]))
        self.r: float = float(p.get("r", _DEFAULT_PARAMS["r"]))
        self.h: float = h
        self.transient_steps: int = transient_steps

        self._state: np.ndarray = self._derive_init(key)

        self._run_transient()

        logger.debug(
            "HyperchaosSystem initialised | params=(a=%.1f b=%.1f d=%.1f r=%.1f) state0=%s",
            self.a,
            self.b,
            self.d,
            self.r,
            self._state.tolist(),
        )

    @staticmethod
    def _derive_init(key: bytes) -> np.ndarray:
        digest = hashlib.sha256(key).digest()
        raw = [struct.unpack(">Q", digest[i : i + 8])[0] for i in range(0, 32, 8)]
        max_u64 = (1 << 64) - 1
        init = np.array(
            [(v / max_u64) * 2 * _INIT_SCALE - _INIT_SCALE for v in raw],
            dtype=np.float64,
        )
        if np.all(np.abs(init) < 1e-10):
            init += 1.0
        return init

    def _run_transient(self) -> None:
        state = self._state.copy()
        for _ in range(self.transient_steps):
            state = _rk4_step(state, self.h, self.a, self.b, self.d, self.r)
        self._state = state

    def generate(self, n: int) -> np.ndarray:
        seq = np.empty(n, dtype=np.float64)
        state = self._state
        for i in range(n):
            state = _rk4_step(state, self.h, self.a, self.b, self.d, self.r)
            seq[i] = state[0]
        self._state = state
        return seq

    def generate_bytes(self, n: int) -> np.ndarray:
        raw = self.generate(n)
        frac = np.abs(raw) % 1.0
        return (frac * 256).astype(np.uint8)

    def generate_uniform(self, n: int) -> np.ndarray:
        return (np.abs(self.generate(n)) * 1e6) % 1.0

    def generate_permutation(self, n: int) -> np.ndarray:
        u = self.generate_uniform(n)
        perm = np.arange(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            j = int(u[i] * (i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def generate_indices(self, n: int) -> np.ndarray:
        seq = self.generate(n)
        return np.argsort(seq)

    def reset(self, key: bytes | str) -> None:
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._state = self._derive_init(key)
        self._run_transient()

    def sensitivity_test(self, n: int = 1000, delta: float = 1e-10) -> float:
        saved = self._state.copy()

        state1 = saved.copy()
        for _ in range(200):
            state1 = _rk4_step(state1, self.h, self.a, self.b, self.d, self.r)
        seq1 = np.empty(n, dtype=np.uint8)
        for i in range(n):
            state1 = _rk4_step(state1, self.h, self.a, self.b, self.d, self.r)
            seq1[i] = int(abs(state1[0]) % 1.0 * 256) & 0xFF

        state2 = saved.copy()
        state2[0] += delta
        for _ in range(200):
            state2 = _rk4_step(state2, self.h, self.a, self.b, self.d, self.r)
        seq2 = np.empty(n, dtype=np.uint8)
        for i in range(n):
            state2 = _rk4_step(state2, self.h, self.a, self.b, self.d, self.r)
            seq2[i] = int(abs(state2[0]) % 1.0 * 256) & 0xFF

        self._state = saved

        divergence = float(np.mean(seq1 != seq2))
        logger.debug("Sensitivity (delta=%.2e): divergence=%.4f", delta, divergence)
        return divergence
