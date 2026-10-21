"""The composition layer: one protocol and one vessel in, contrast and a detection verdict out.

Nothing here is new physics. ``sequence.py`` turns tissue parameters into
signals and ``detect.py`` turns a contrast into a detection statement, but
until this module existed nothing joined them, so the efficiency metric
CLAUDE.md names as the one that decides the protocol had no caller.

The join needs one convention stated explicitly, because it is what makes the
two halves commensurable. Simulated signals are relative, dermis M0 = 1
(``tissues.pd_rel``), and ``detect`` works in units of noise sigma. The bridge
is the REFERENCE SNR: the SNR a voxel would have if its simulated signal were
exactly 1. Absolute sensitivity is not derivable here - it needs the coil's
sensitivity at depth, which ``detect.thermal_snr`` deliberately leaves as an
unknown constant - so reference SNR is never stored as a number. It is an
argument to ``Prediction.cnr`` and nothing else, which is what CLAUDE.md means
by keeping ``snr_available`` a hook rather than a value.

Under TONE the background is evaluated at the vessel's own through-slab
position, not at the slab centre: the flip the blood sees at a position is the
flip the surrounding tissue sees there too, so a single background number would
compare the vessel against tissue excited at a different angle.

The join also fixes where background texture comes from. ``detect`` wants an
absolute texture sigma, in the same units as the contrast. Stored as an
absolute constant it would be unphysical - 6 % of a unit reference signal is
2.5x the entire suppressed-fat background - so the stored parameter
(``tissues.ACQUISITION["background_structure_rel"]``) is the background's
structure relative to its OWN mean, and the absolute figure is derived here as
that fraction times the simulated background. Fat saturation then suppresses
the texture floor by the same factor as the background, which is the real
reason it decides detectability rather than merely brightness.

Shapes: positions (P,) and a sample axis (N,) from the Latin-hypercube draw.
``blood`` is (P, N); ``background`` is (P, N) under TONE and (N,) without it.

Units: ms, mm, mm/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

import detect
from detect import ROSE
from geometry import Vessel
from protocols import Protocol
from sequence import FlowGeometry, TissueSample


def cross_section_voxel(protocol: Protocol, vessel: Vessel) -> tuple[float, float]:
    """Return the voxel extents whose product is the area of the voxel's shadow projected along the vessel."""
    # For a long cylinder of cross-section A_v crossing a voxel of volume V, the
    # intersection is A_v times the chord along the vessel, and the mean chord of
    # a convex body along a direction is V / A_proj. So the blood fraction is
    # A_v / A_proj and the area detect.py needs is the projected shadow,
    # A_proj = inplane * partition * sin + inplane^2 * cos for a box with two
    # in-plane edges and one along the slab normal. One returned edge is the true
    # in-plane extent; the other carries the rest. Exact at both limits (inplane^2
    # end-on, inplane * partition in-plane) and LARGER than either in between,
    # because a box's shadow is widest along a diagonal.
    cos = abs(float(np.dot(vessel.direction(), protocol.slab().normal())))
    sin = math.sqrt(max(1.0 - cos * cos, 0.0))
    return protocol.inplane_mm, protocol.partition_mm * sin + protocol.inplane_mm * cos


@dataclass(frozen=True)
class Prediction:
    """Signals along one vessel and the detection quantities that follow, for one protocol and one draw set."""

    protocol_name: str
    background_name: str
    positions_mm: np.ndarray
    blood: np.ndarray
    background: np.ndarray
    time_per_line_ms: float
    voxel_mm: tuple[float, float]
    diameter_mm: float
    background_structure_rel: ArrayLike

    @property
    def contrast(self) -> np.ndarray:
        """Return the absolute blood-minus-background contrast at each position, in units of dermis M0."""
        return self.blood - self.background

    @property
    def structure_fraction(self) -> np.ndarray:
        """Return the absolute background texture detect.py has to beat, the relative structure times the background."""
        return np.asarray(self.background_structure_rel, dtype=float) * self.background

    @property
    def cnr_efficiency(self) -> np.ndarray:
        """Return the absolute contrast per square root of the time spent per k-space line, per sqrt(s)."""
        return detect.cnr_efficiency(self.blood, self.background, self.time_per_line_ms)

    @property
    def blood_fraction(self) -> float:
        """Return the peak voxel's blood volume fraction, with kappa derived from the vessel and voxel."""
        return detect.blood_fraction(self.diameter_mm, self.voxel_mm)

    @property
    def snr_required(self) -> np.ndarray:
        """Return the reference SNR needed to reach Rose CNR 4, infinite where texture alone defeats the contrast."""
        return detect.snr_required(self.contrast, self.diameter_mm, self.voxel_mm,
                                   self.structure_fraction)

    def cnr(self, snr_available: ArrayLike) -> np.ndarray:
        """Return the peak-voxel CNR at a given reference SNR, the SNR of a voxel whose simulated signal is 1."""
        return detect.cnr(self.contrast, self.diameter_mm, self.voxel_mm, snr_available,
                          self.structure_fraction)

    def detected(self, snr_available: ArrayLike, cnr_threshold: float = ROSE) -> np.ndarray:
        """Return whether each position and draw clears the detection threshold at that reference SNR."""
        return self.cnr(snr_available) >= cnr_threshold


def predict(protocol: Protocol, blood: TissueSample, background: TissueSample, flow: FlowGeometry,
            positions_mm: np.ndarray, *, background_structure_rel: ArrayLike = 0.0,
            band_pulse_residual: ArrayLike = 0.0) -> Prediction:
    """Return the Prediction for one protocol, vessel and background, refusing a background off the vessel's depth."""
    expected = flow.vessel.background().name
    if background.name != expected:
        raise ValueError(f"vessel at {flow.vessel.depth_mm} mm lies in {expected}, not {background.name}")
    sequence = protocol.sequence(band_pulse_residual)
    positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
    z = sequence.blood_z(flow, positions)[:, None] if protocol.tone_pct else None
    signal = np.asarray(sequence.blood_signal(blood, flow, positions), dtype=float)
    return Prediction(
        protocol_name=protocol.name,
        background_name=background.name,
        positions_mm=positions,
        blood=signal[:, None] if signal.ndim == 1 else signal,  # positions first, draws last
        background=np.asarray(sequence.static_signal(background, z), dtype=float),
        time_per_line_ms=sequence.time_per_line_ms(),
        voxel_mm=cross_section_voxel(protocol, flow.vessel),
        diameter_mm=flow.vessel.diameter_mm,
        background_structure_rel=background_structure_rel,
    )
