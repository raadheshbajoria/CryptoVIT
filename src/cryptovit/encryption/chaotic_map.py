"""
cryptovit/encryption/chaotic_map.py
=====================================
4D Hyperchaotic dynamical system for key-stream generation.

System definition
-----------------
We use the 4D hyperchaotic system proposed in the literature (Chai et al. 2017,
Wang & Zhao 2021) with four coupled state variables (x, y, z, w):

    dx/dt = a * (y - x)
    dy/dt = d * x - x * z + w
    dz/dt = x * y - b * z
    dw/dt = -y * z + r * w

Default parameters (a=35, b=3, d=28, r=-1) produce hyperchaotic behaviour
with two positive Lyapunov exponents.

Integration
-----------
4th-order Runge-Kutta (RK4) with step size h=0.01.
A configurable transient (default 1000 steps) is discarded to ensure
the trajectory has settled onto the attractor before sampling begins.

Key derivation
--------------
The secret key (bytes or hex string) is hashed with SHA-256 to produce
a 256-bit seed, which is split into four 64-bit floats and used as the
initial state (x0, y0, z0, w0) after mapping into the chaotic attractor
region (~[-30, 30]).

References
----------
Chai et al. (2017), Signal Processing: Image Communication.
Wang & Zhao (2021), Optics & Laser Technology.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default system parameters
_DEFAULT_PARAMS = dict(a=35.0, b=3.0, d=28.0, r=-1.0)

# Attractor scale: initial conditions are mapped into [-SCALE, SCALE]
_INIT_SCALE = 15.0


# ---------------------------------------------------------------------------
# ODE definition
# ---------------------------------------------------------------------------

def _ode(
    state: np.ndarray,
    a: float, b: float, d: float, r: float,
) -> np.ndarray:
    """Compute the time derivative of the 4D hyperchaotic system.

    Parameters
    ----------
    state : (4,) array — [x, y, z, w]
    a, b, d, r : system parameters

    Returns
    -------
    dstate : (4,) array
    """
    x, y, z, w = state
    dx = a * (y - x)
    dy = d * x - x * z + w
    dz = x * y - b * z
    dw = -y * z + r * w
    return np.array([dx, dy, dz, dw], dtype=np.float64)


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def _rk4_step(
    state: np.ndarray,
    h: float,
    a: float, b: float, d: float, r: float,
) -> np.ndarray:
    """One RK4 integration step."""
    k1 = _ode(state,         a, b, d, r)
    k2 = _ode(state + h/2*k1, a, b, d, r)
    k3 = _ode(state + h/2*k2, a, b, d, r)
    k4 = _ode(state + h*k3,   a, b, d, r)
    return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HyperchaosSystem:
    """4D Hyperchaotic key-stream generator.

    Parameters
    ----------
    key : bytes or str
        Secret encryption key.  A str is UTF-8 encoded before hashing.
    params : dict, optional
        System parameters ``{a, b, d, r}``.  Defaults to the standard
        hyperchaotic regime.
    h : float
        RK4 step size.
    transient_steps : int
        Number of initial steps discarded (to settle on the attractor).
    """

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

        # Derive initial conditions from key
        self._state: np.ndarray = self._derive_init(key)

        # Discard transient
        self._run_transient()

        logger.debug(
            "HyperchaosSystem initialised | params=(a=%.1f b=%.1f d=%.1f r=%.1f) "
            "state0=%s",
            self.a, self.b, self.d, self.r, self._state.tolist(),
        )

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_init(key: bytes) -> np.ndarray:
        """Hash the key with SHA-256 and map to initial (x0, y0, z0, w0)."""
        digest = hashlib.sha256(key).digest()   # 32 bytes = 256 bits
        # Split into four 8-byte chunks → four float64 values in [0, 1)
        raw = [struct.unpack(">Q", digest[i:i+8])[0] for i in range(0, 32, 8)]
        # Normalise to uint64 max and shift to [-SCALE, SCALE]
        max_u64 = (1 << 64) - 1
        init = np.array(
            [(v / max_u64) * 2 * _INIT_SCALE - _INIT_SCALE for v in raw],
            dtype=np.float64,
        )
        # Avoid trivial zero-state (which is a fixed point)
        if np.all(np.abs(init) < 1e-10):
            init += 1.0
        return init

    def _run_transient(self) -> None:
        """Advance the system by transient_steps without recording output."""
        state = self._state.copy()
        for _ in range(self.transient_steps):
            state = _rk4_step(state, self.h, self.a, self.b, self.d, self.r)
        self._state = state

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n: int) -> np.ndarray:
        """Generate ``n`` floating-point samples from the chaotic trajectory.

        The samples are drawn by advancing the system ``n`` steps and
        collecting one output value (the x-component) per step.  The
        internal state is updated so successive calls continue from where
        the last call left off.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        seq : (n,) float64 array in approximately [-50, 50]
        """
        seq = np.empty(n, dtype=np.float64)
        state = self._state
        for i in range(n):
            state = _rk4_step(state, self.h, self.a, self.b, self.d, self.r)
            seq[i] = state[0]   # use x-component
        self._state = state
        return seq

    def generate_bytes(self, n: int) -> np.ndarray:
        """Generate ``n`` samples mapped to the range [0, 255] (uint8).

        Useful for pixel-level diffusion.

        Parameters
        ----------
        n : int

        Returns
        -------
        (n,) uint8 array
        """
        raw = self.generate(n)
        # Map to [0, 255] via modular normalisation
        # 1. Take absolute value and fractional part
        frac = np.abs(raw) % 1.0       # in [0, 1)
        return (frac * 256).astype(np.uint8)

    def generate_indices(self, n: int) -> np.ndarray:
        """Generate a chaotic permutation of indices [0, n).

        Uses the argsort of a chaotic sequence — a standard technique for
        generating key-dependent pseudo-random permutations.

        Parameters
        ----------
        n : int

        Returns
        -------
        (n,) int64 array — a permutation of [0, n)
        """
        seq = self.generate(n)
        return np.argsort(seq)

    def reset(self, key: bytes | str) -> None:
        """Re-derive initial conditions from a new key and re-run transient."""
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._state = self._derive_init(key)
        self._run_transient()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sensitivity_test(
        self, n: int = 1000, delta: float = 1e-10
    ) -> float:
        """Measure sensitivity to initial conditions.

        Creates a copy of the current settled state, perturbs one component
        by ``delta``, advances both trajectories 200 extra steps to let the
        perturbation grow via chaotic divergence, then compares byte sequences.

        Returns
        -------
        float in [0, 1] — fraction of differing bytes (should be > 0.5)
        """
        saved = self._state.copy()

        # Sequence 1: from current settled state (advance 200 steps + n)
        state1 = saved.copy()
        for _ in range(200):
            state1 = _rk4_step(state1, self.h, self.a, self.b, self.d, self.r)
        seq1 = np.empty(n, dtype=np.uint8)
        for i in range(n):
            state1 = _rk4_step(state1, self.h, self.a, self.b, self.d, self.r)
            seq1[i] = int(abs(state1[0]) % 1.0 * 256) & 0xFF

        # Sequence 2: same but with perturbation in one component
        state2 = saved.copy()
        state2[0] += delta
        for _ in range(200):
            state2 = _rk4_step(state2, self.h, self.a, self.b, self.d, self.r)
        seq2 = np.empty(n, dtype=np.uint8)
        for i in range(n):
            state2 = _rk4_step(state2, self.h, self.a, self.b, self.d, self.r)
            seq2[i] = int(abs(state2[0]) % 1.0 * 256) & 0xFF

        # Restore internal state
        self._state = saved

        divergence = float(np.mean(seq1 != seq2))
        logger.debug("Sensitivity (delta=%.2e): divergence=%.4f", delta, divergence)
        return divergence
