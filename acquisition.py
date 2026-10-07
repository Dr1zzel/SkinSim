"""The scanner's side: excitation slabs, vessel path through them, and coverage cost.

Units: mm. Angles are degrees at the API boundary, radians inside.
A tilted slab's normal lies in the x-z plane of the skin frame (see skin.py),
so a vessel's azimuth is measured from the slab's tilt direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skin import Vessel


@dataclass(frozen=True)
class Slab:
    """Excitation slab whose normal makes orientation_deg with the skin surface normal (0 = slab parallel to skin)."""

    thickness_mm: float
    orientation_deg: float

    def __post_init__(self) -> None:
        if self.thickness_mm <= 0:
            raise ValueError(f"slab thickness must be positive, got {self.thickness_mm} mm")

    def normal(self) -> np.ndarray:
        """Return the slab's unit normal in the skin frame, tilted from z towards x."""
        tilt = math.radians(self.orientation_deg)
        return np.array([math.sin(tilt), 0.0, math.cos(tilt)])


def path_length(slab: Slab, vessel: Vessel, max_path_mm: float) -> float:
    """Return the vessel's path through the slab, thickness / |direction . normal|, capped at max_path_mm, a bound with no justification (slab lateral extent or straight-segment length) that must be swept."""
    if max_path_mm <= 0:
        raise ValueError(f"max_path_mm must be positive, got {max_path_mm}")
    cos_angle = abs(float(np.dot(vessel.direction(), slab.normal())))
    if cos_angle * max_path_mm <= slab.thickness_mm:
        return max_path_mm
    return slab.thickness_mm / cos_angle


# Floating-point slack so an exact fit is not rounded up to an extra slab.
_TILING_SLACK = 1e-9


def slabs_to_cover(slab: Slab, target_depth_mm: float, target_width_mm: float) -> int:
    """Return how many slabs, stacked along their normal, tile a target from the skin surface to target_depth_mm across target_width_mm in the tilt direction."""
    if target_depth_mm <= 0 or target_width_mm <= 0:
        raise ValueError("target depth and width must be positive")
    tilt = math.radians(slab.orientation_deg)
    # Extent of the target's cross-section projected onto the slab normal; each
    # slab is assumed wide enough in-plane to span the target.
    extent_mm = target_width_mm * abs(math.sin(tilt)) + target_depth_mm * abs(math.cos(tilt))
    return max(1, math.ceil(extent_mm / slab.thickness_mm - _TILING_SLACK))
