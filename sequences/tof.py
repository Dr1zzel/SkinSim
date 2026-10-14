"""Time-of-flight as fl3d1r runs it: spoiled GRE with inflow, laminar shells, saturation modules and a travelling band.

Every TR is one affine map on Mz (sequences/prep.py): the excitation, then
relaxation with any non-selective saturation pulses (fat saturation, MTC).
Static tissue sits at that map's fixed point. Blood arrives at the slab face
with the inflow boundary condition (the upstream steady state of the
non-selective pulses, saturated further if the travelling band lies on the
entry side) and, at distance s along the vessel, has received
n = floor(s / (v TR)) excitations; power() gives the result in closed form.
Assumes an ideal rectangular slab profile and ideal spoiling.

Shapes: positions (P,), laminar shells (S,), and a sample axis (N,) when the
TissueSample fields are arrays; blood_signal returns (P,) or (P, N).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike

from acquisition import SatBand, entry_side
from affine import Affine, apply, compose, fixed_point, power, relax_map
from sequences.base import FlowGeometry, Sequence, TissueSample
from sequences.prep import (
    SaturationModule,
    SaturationPulse,
    band_exit_mz,
    between_excitations,
    excitation_map,
)
from skin import laminar_velocities


def mz_before_pulse(
    n_pulses: np.ndarray, mz_initial: float, m0: float, flip_deg: float, tr_ms: float, t1_ms: float
) -> np.ndarray:
    """Return Mz just before the next pulse after n_pulses plain spoiled pulses, n_pulses = inf giving the steady state."""
    e1 = np.exp(-tr_ms / np.asarray(t1_ms, dtype=float))
    q = e1 * np.cos(np.radians(flip_deg))
    mz_ss = m0 * (1 - e1) / (1 - q)
    return mz_ss + (mz_initial - mz_ss) * np.power(q, n_pulses)


def pulses_received(positions_mm: np.ndarray, velocity_mm_s: np.ndarray, tr_ms: float) -> np.ndarray:
    """Return how many whole TRs blood at velocity_mm_s has spent travelling positions_mm into the slab, inf when static."""
    step_mm = np.asarray(velocity_mm_s, dtype=float) * tr_ms / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.floor(np.asarray(positions_mm, dtype=float) / step_mm)
    return np.where(step_mm > 0, n, np.inf)


def transit_ms(distance_mm: ArrayLike, velocity_mm_s: ArrayLike) -> np.ndarray:
    """Return the time to cover distance_mm at velocity_mm_s, inf when static."""
    with np.errstate(divide="ignore"):
        return 1000.0 * np.asarray(distance_mm, dtype=float) / np.asarray(velocity_mm_s, dtype=float)


def as_sample_axis(tissue: TissueSample) -> TissueSample:
    """Return the sample with every numeric field at least one-dimensional, the sample axis last."""
    return TissueSample(tissue.name, **{f.name: np.atleast_1d(np.asarray(getattr(tissue, f.name), dtype=float))
                                        for f in fields(tissue) if f.name != "name"})


@dataclass(frozen=True, kw_only=True)
class TofSequence(Sequence):
    """Spoiled-GRE time-of-flight with non-selective saturation modules and an optional travelling saturation band."""

    flip_deg: float
    modules: tuple[SaturationModule, ...] = ()
    sat_band: SatBand | None = None
    band_pulse_residual: ArrayLike = 0.0  # ideal 90 deg band pulse; sample ACQUISITION["sat_band_pulse_residual"]

    def pulses(self, tissue: TissueSample) -> tuple[SaturationPulse, ...]:
        """Return the non-selective saturation pulses as this tissue experiences them."""
        return tuple(module.pulse(tissue) for module in self.modules)

    def tr_map(self, tissue: TissueSample) -> Affine:
        """Return the map from just before one excitation to just before the next, unit M0."""
        relax = between_excitations(self.pulses(tissue), tissue, self.timing.tr_ms)
        return compose(excitation_map(self.flip_deg), relax)

    def static_signal(self, tissue: TissueSample) -> ArrayLike:
        """Return the echo signal of a static tissue at its steady state."""
        return self.echo(fixed_point(self.tr_map(tissue)), self.flip_deg, tissue)

    def inflow_mz(self, blood: TissueSample, flow: FlowGeometry, velocities: np.ndarray) -> np.ndarray:
        """Return Mz / M0 of blood reaching the slab face per laminar shell (rows), band-saturated if on the entry side."""
        upstream = fixed_point(between_excitations(self.pulses(blood), blood, self.timing.tr_ms))
        shells = np.ones((len(velocities), 1)) * upstream
        side = entry_side(flow.slab, flow.vessel)
        if self.sat_band is None or side == 0 or self.sat_band.side != side:
            return shells
        along = abs(float(np.dot(flow.vessel.direction(), flow.slab.normal())))
        v = velocities[:, None] * along  # speed along the slab normal
        n_band = np.floor(transit_ms(self.sat_band.thickness_mm, v) / self.timing.tr_ms)
        exit_mz = band_exit_mz(shells, n_band, self.band_pulse_residual, blood, self.timing.tr_ms)
        return apply(relax_map(transit_ms(self.sat_band.gap_mm, v), blood.t1_ms), exit_mz)

    def blood_signal(self, blood: TissueSample, flow: FlowGeometry, positions_mm: np.ndarray) -> np.ndarray:
        """Return the area-weighted laminar echo signal of blood at distances positions_mm from slab entry."""
        positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
        if positions.min() < 0 or positions.max() > flow.path_mm():
            raise ValueError(f"positions must lie within the slab path, 0 to {flow.path_mm():.3g} mm")
        scalar, blood = np.ndim(blood.t1_ms) == 0, as_sample_axis(blood)
        velocities, weights = laminar_velocities(flow.mean_velocity_mm_s, flow.n_shells)
        n = pulses_received(positions[:, None], velocities[None, :], self.timing.tr_ms)[..., None]
        mz = apply(power(self.tr_map(blood), n), self.inflow_mz(blood, flow, velocities)[None])
        out = np.einsum("psn,s->pn", self.echo(mz, self.flip_deg, blood), weights)
        return out[:, 0] if scalar else out
