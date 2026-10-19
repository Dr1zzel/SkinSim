"""Skin anatomy and the excitation slab: layers by depth, vessels, flow profiles, slab path.

Frame: z is the skin surface normal pointing into tissue (depth); x and y lie
in the skin plane, and vessel azimuth is measured from x. A tilted slab's
normal lies in the x-z plane, so a vessel's azimuth is measured from the slab's
tilt direction.

Units: mm, mm/s. Angles are degrees at the API boundary, radians inside.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tissues import DERMIS, EPIDERMIS, HYPODERMIS, MUSCLE, Tissue


# Lower boundary of each layer, in mm below the skin surface. Real layer
# thicknesses vary with body site; these are the project's stated boundaries.
LAYERS: tuple[tuple[float, Tissue], ...] = (
    (0.1, EPIDERMIS),
    (2.0, DERMIS),
    (10.0, HYPODERMIS),
    (math.inf, MUSCLE),
)


def tissue_at_depth(depth_mm: float) -> Tissue:
    """Return the background tissue at a depth below the skin surface, each layer half-open [top, bottom)."""
    if depth_mm < 0:
        raise ValueError(f"depth must be non-negative, got {depth_mm} mm")
    return next(tissue for bottom_mm, tissue in LAYERS if depth_mm < bottom_mm)


@dataclass(frozen=True)
class Vessel:
    """Straight vessel segment: depth decides its background, polar_deg (from the skin normal, 90 = in-plane) and azimuth_deg decide its path."""

    depth_mm: float
    polar_deg: float
    azimuth_deg: float
    diameter_mm: float

    def __post_init__(self) -> None:
        if self.depth_mm < 0:
            raise ValueError(f"vessel depth must be non-negative, got {self.depth_mm} mm")
        if self.diameter_mm <= 0:
            raise ValueError(f"vessel diameter must be positive, got {self.diameter_mm} mm")

    def direction(self) -> np.ndarray:
        """Return the vessel's unit direction vector in the skin frame."""
        polar = math.radians(self.polar_deg)
        azimuth = math.radians(self.azimuth_deg)
        return np.array([
            math.sin(polar) * math.cos(azimuth),
            math.sin(polar) * math.sin(azimuth),
            math.cos(polar),
        ])

    def background(self) -> Tissue:
        """Return the tissue surrounding the vessel, decided by its depth alone."""
        return tissue_at_depth(self.depth_mm)


def laminar_velocities(mean_velocity: float, n_shells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (shell velocities in mm/s, annular-area weights summing to 1) for Poiseuille flow in equal-width shells, innermost first."""
    if n_shells < 1:
        raise ValueError(f"n_shells must be at least 1, got {n_shells}")
    edges = np.linspace(0.0, 1.0, n_shells + 1)  # radius as a fraction of vessel radius
    inner, outer = edges[:-1], edges[1:]
    weights = outer**2 - inner**2
    # Area average of v_max (1 - r^2) over each annulus, so the weighted mean
    # equals mean_velocity exactly for any n_shells; v_max = 2 * mean_velocity.
    velocities = 2.0 * mean_velocity * (1.0 - 0.5 * (inner**2 + outer**2))
    return velocities, weights


# --- The scanner's side: excitation slabs and the vessel path through them ---


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


@dataclass(frozen=True)
class SatBand:
    """Travelling saturation band beside the slab, on side +1 or -1 along the slab normal, moving with it."""

    thickness_mm: float
    gap_mm: float
    side: int  # +1: beyond the slab's +normal face; -1: beyond its -normal face

    def __post_init__(self) -> None:
        if self.side not in (1, -1):
            raise ValueError("side must be +1 or -1")


def entry_side(slab: Slab, vessel: Vessel) -> int:
    """Return the side (+1 or -1 along the normal) through which blood flowing along the vessel enters, 0 if parallel to the slab."""
    along = float(np.dot(vessel.direction(), slab.normal()))
    if abs(along) < 1e-12:
        return 0
    return -1 if along > 0 else 1


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
