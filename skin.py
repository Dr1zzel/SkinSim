"""Skin anatomy: layers by depth, vessels and their flow profiles.

Units: mm, mm/s. Angles are degrees at the API boundary, radians inside.
Frame: z is the skin surface normal pointing into tissue (depth); x and y lie
in the skin plane, and vessel azimuth is measured from x.
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
