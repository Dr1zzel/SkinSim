"""Independent closed-form reference implementations, used only to check the fast path.

The toolbox computes spoiled-GRE signal by composing affine maps on Mz
(``affine.py``) and applying them n times with ``power``. That is fast and
general, but it is not obviously correct. These two functions are the textbook
closed forms written out directly, with no affine machinery behind them, so a
test that compares the two is a real check rather than a tautology.

They live under tests/ deliberately: they exist to validate, nothing in the
toolbox imports them, and keeping them out of the source tree is what makes the
comparison independent.
"""
from __future__ import annotations

import math

import numpy as np


def ernst_signal(m0: float, flip_deg: float, tr_ms: float, t1_ms: float) -> float:
    """Return the ideally spoiled steady-state transverse magnetisation just after the pulse."""
    e1 = math.exp(-tr_ms / t1_ms)
    alpha = math.radians(flip_deg)
    return m0 * math.sin(alpha) * (1 - e1) / (1 - e1 * math.cos(alpha))


def spoiled_train(
    mz_initial: float | np.ndarray,
    m0: float | np.ndarray,
    flip_deg: float | np.ndarray,
    tr_ms: float,
    t1_ms: float | np.ndarray,
    n_pulses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (signal just after each pulse, Mz just before each pulse) for an ideally spoiled train, in closed form, broadcasting over inputs with pulses on the last axis."""
    e1 = np.exp(-tr_ms / np.asarray(t1_ms, dtype=float))[..., None]
    alpha = np.radians(np.asarray(flip_deg, dtype=float))[..., None]
    m0 = np.asarray(m0, dtype=float)[..., None]
    mz_initial = np.asarray(mz_initial, dtype=float)[..., None]
    q = e1 * np.cos(alpha)
    mz_ss = m0 * (1 - e1) / (1 - q)
    mz_before = mz_ss + (mz_initial - mz_ss) * q ** np.arange(n_pulses)
    return mz_before * np.sin(alpha), mz_before
