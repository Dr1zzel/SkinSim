"""CNR efficiency: absolute contrast per square root of the time spent per k-space line.

Ported from SkinMRA (skinmra_sim/analysis.py). Sequence.time_per_line_ms does
the harder half, amortising any preparation and recovery over the lines it
serves; this is the metric on top of it:

    efficiency = (S_blood - S_background) / sqrt(time per k-space line)

It is an ABSOLUTE difference, never a ratio: in the sensitivity-limited regime
the background can be near zero, where a ratio is meaningless. With the same
voxel, matrix and bandwidth the noise per acquired line is the same for every
protocol, so for a fixed number of lines this is CNR per sqrt(scan time) up to
one common sensitivity factor (noise.py). Signed: negative means blood darker.
Units: signals in units of dermis M0, times in ms; returned per sqrt(second).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cnr_efficiency(s_blood: ArrayLike, s_background: ArrayLike, time_per_line_ms: ArrayLike) -> np.ndarray:
    """Return (S_blood - S_background) / sqrt(time per k-space line) per sqrt(s), broadcasting, refusing non-positive times."""
    t = np.asarray(time_per_line_ms, dtype=float)
    if np.any(t <= 0):
        raise ValueError("time per k-space line must be positive")
    diff = np.asarray(s_blood, dtype=float) - np.asarray(s_background, dtype=float)
    return diff / np.sqrt(t / 1000.0)
