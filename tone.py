"""TONE: the through-slab flip-angle ramp of fl3d1r.

Every 13 October series runs TONE 70 %, so the flip angle varies through the
slab and a flowing spin sees a different flip at each pulse.

CONVENTION ASSUMED, NOT VERIFIED: the ramp percentage is the peak-to-peak
variation about the nominal angle, linear in the normalised through-slab
coordinate and increasing along the declared flow direction, so that
alpha(z) = alpha_mean * (1 + ramp/100 * (z - 1/2)), z = 0 at the face where
declared flow enters. This sets the sign of the predicted through-slab
profile; if Siemens defines the ramp differently (e.g. as an exit/entry
ratio), the predicted profile changes shape. Confirm against the vendor
documentation or a phantom before quoting a profile.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def flip_profile(z_frac: ArrayLike, ramp_pct: float, alpha_mean_deg: float) -> np.ndarray:
    """Return the local flip angle in degrees at normalised through-slab position z_frac along the declared flow direction."""
    if not 0.0 <= ramp_pct < 200.0:
        raise ValueError(f"ramp must lie in [0, 200) % to keep every flip positive, got {ramp_pct}")
    z = np.clip(np.asarray(z_frac, dtype=float), 0.0, 1.0)
    return alpha_mean_deg * (1.0 + ramp_pct / 100.0 * (z - 0.5))
