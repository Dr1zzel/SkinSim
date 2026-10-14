"""Affine maps on longitudinal magnetisation: the core of the spoiled-GRE signal model.

With ideal spoiling only Mz carries from one event to the next, and every event
(relaxation, an excitation, a saturation pulse, a whole TR or a whole cycle) is
an affine map Mz_out = A * Mz_in + B. Composing maps gives one map, its fixed
point B / (1 - A) is the periodic steady state, and its n-th power is n
identical repetitions in closed form.

A and B may be NumPy arrays: maps broadcast elementwise, so one map can carry a
whole sample axis (n_samples,) or a grid (n_samples, n_shells).
Units: ms.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

Affine = tuple[ArrayLike, ArrayLike]  # Mz_out = A * Mz_in + B
IDENTITY: Affine = (1.0, 0.0)


def compose(*maps: Affine) -> Affine:
    """Return the single affine map that applies the given maps in order."""
    return walk(maps)


def walk(maps: Iterable[Affine]) -> Affine:
    """Return the composition of a sequence of per-pulse maps, for trains whose TRs differ (power() needs them identical)."""
    a, b = IDENTITY
    for a_i, b_i in maps:
        a, b = a_i * a, a_i * b + b_i
    return a, b


def apply(m: Affine, mz: ArrayLike) -> ArrayLike:
    """Return Mz after the affine map."""
    return m[0] * mz + m[1]


def power(m: Affine, n: ArrayLike) -> Affine:
    """Return the affine map applied n times; n = inf gives the constant map to its fixed point (needs |A| < 1)."""
    if np.ndim(n) == 0 and n == 0:
        return IDENTITY
    a_n = np.power(m[0], n)
    return a_n, m[1] * (1 - a_n) / (1 - m[0])


def fixed_point(m: Affine) -> ArrayLike:
    """Return the Mz the affine map leaves unchanged, B / (1 - A)."""
    return m[1] / (1 - m[0])


def relax_map(duration_ms: ArrayLike, t1_ms: ArrayLike, m0: ArrayLike = 1.0) -> Affine:
    """Return T1 recovery towards m0 over duration_ms."""
    e1 = np.exp(-np.asarray(duration_ms, dtype=float) / np.asarray(t1_ms, dtype=float))
    return e1, m0 * (1 - e1)
