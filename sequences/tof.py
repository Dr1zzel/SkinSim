"""Time-of-flight as fl3d1r runs it: spoiled GRE with inflow, laminar shells, saturation modules and a travelling band.

Every TR is one affine map on Mz (sequences/prep.py): the excitation, then
relaxation with any non-selective saturation pulses (fat saturation, MTC).
Static tissue sits at that map's fixed point. Blood arrives at the slab face
with the inflow boundary condition (the upstream steady state of the
non-selective pulses, saturated further if the travelling band lies on the
entry side) and, at distance s along the vessel, has received
n = floor(s / (v TR)) excitations; power() gives the result in closed form.
Under TONE the flip varies with through-slab position z (0 at the slab's
-normal face, 1 at its +normal face), so each pulse has its own map and the
spin's pulses are walked one by one (affine.walk semantics), sequentially in
the pulse index and vectorised over positions, shells and samples.
Assumes an ideal rectangular slab profile and ideal spoiling.

Shapes: positions (P,), laminar shells (S,), and a sample axis (N,) when the
TissueSample fields or the flow's mean velocity are arrays; shell velocities
are (S, N) and blood_signal returns (P,) or (P, N).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import math

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
from tone import flip_profile


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
    tone_pct: float = 0.0
    tone_direction: int = 1  # +1: the TONE ramp rises towards +normal, i.e. the declared flow direction
    modules: tuple[SaturationModule, ...] = ()
    sat_band: SatBand | None = None
    band_pulse_residual: ArrayLike = 0.0  # ideal 90 deg band pulse; sample ACQUISITION["sat_band_pulse_residual"]

    def pulses(self, tissue: TissueSample) -> tuple[SaturationPulse, ...]:
        """Return the non-selective saturation pulses as this tissue experiences them."""
        return tuple(module.pulse(tissue) for module in self.modules)

    def flip_at(self, z_frac: ArrayLike) -> np.ndarray:
        """Return the local flip angle at through-slab position z_frac (0 at the -normal face)."""
        along_flow = np.asarray(z_frac, dtype=float) if self.tone_direction > 0 else 1 - np.asarray(z_frac)
        return flip_profile(along_flow, self.tone_pct, self.flip_deg)

    def tr_map(self, tissue: TissueSample, flip_deg: ArrayLike | None = None) -> Affine:
        """Return the map from just before one excitation to just before the next, unit M0."""
        relax = between_excitations(self.pulses(tissue), tissue, self.timing.tr_ms)
        return compose(excitation_map(self.flip_deg if flip_deg is None else flip_deg), relax)

    def static_flip(self, z_frac: ArrayLike | None) -> ArrayLike:
        """Return the flip static tissue sees, refusing to guess a position when TONE is on."""
        if self.tone_pct == 0:
            return self.flip_deg
        if z_frac is None:
            raise ValueError("under TONE a static signal needs its through-slab position z_frac")
        return self.flip_at(z_frac)

    def static_signal(self, tissue: TissueSample, z_frac: ArrayLike | None = None) -> ArrayLike:
        """Return the echo signal of a static tissue at its steady state, at z_frac when TONE is on."""
        flip = self.static_flip(z_frac)
        return self.echo(fixed_point(self.tr_map(tissue, flip)), flip, tissue)

    def blood_z(self, flow: FlowGeometry, distance_mm: ArrayLike) -> np.ndarray:
        """Return the through-slab position of blood distance_mm along the vessel from where it entered the slab."""
        side = entry_side(flow.slab, flow.vessel)
        if side == 0:
            if flow.entry_z_frac is None:
                raise ValueError("a vessel parallel to the slab needs flow.entry_z_frac under TONE")
            return np.full(np.shape(distance_mm), flow.entry_z_frac)
        along = abs(float(np.dot(flow.vessel.direction(), flow.slab.normal())))
        z = np.asarray(distance_mm, dtype=float) * along / flow.slab.thickness_mm
        return z if side < 0 else 1 - z

    def inflow_mz(self, blood: TissueSample, flow: FlowGeometry, velocities: np.ndarray) -> np.ndarray:
        """Return Mz / M0 of blood reaching the slab face for shell velocities (S, N), band-saturated if on the entry side."""
        upstream = fixed_point(between_excitations(self.pulses(blood), blood, self.timing.tr_ms))
        shells = np.zeros_like(velocities) + upstream
        side = entry_side(flow.slab, flow.vessel)
        if self.sat_band is None or side == 0 or self.sat_band.side != side:
            return shells
        along = abs(float(np.dot(flow.vessel.direction(), flow.slab.normal())))
        v = velocities * along  # speed along the slab normal
        n_band = np.floor(transit_ms(self.sat_band.thickness_mm, v) / self.timing.tr_ms)
        exit_mz = band_exit_mz(shells, n_band, self.band_pulse_residual, blood, self.timing.tr_ms)
        return apply(relax_map(transit_ms(self.sat_band.gap_mm, v), blood.t1_ms), exit_mz)

    def memory_pulses(self, relax_a: ArrayLike) -> int:
        """Return how many past pulses can still change Mz by 1e-12 of M0, from the slowest per-pulse contraction."""
        cos = np.abs(np.cos(np.radians(self.flip_at(np.array([0.0, 1.0])))))
        q = float(np.max(np.abs(relax_a)) * np.max(cos))
        return 0 if q == 0 else math.ceil(math.log(1e-12) / math.log(q))

    def walked_mz(self, blood: TissueSample, flow: FlowGeometry, positions: np.ndarray, velocities: np.ndarray,
                  inflow: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Return Mz / M0 before the current pulse, walking each spin's pulses with the flip at its position at each one."""
        relax = between_excitations(self.pulses(blood), blood, self.timing.tr_ms)
        finite = n[np.isfinite(n)]
        k_top = int(min(self.memory_pulses(relax[0]), finite.max() if finite.size else 0))
        step = velocities[None] * self.timing.tr_ms / 1000.0

        def pulse_map(k: int) -> Affine:
            """Return the per-TR map for the pulse k TRs before now, at each spin's position then."""
            z = self.blood_z(flow, np.maximum(positions[:, None, None] - k * step, 0.0))
            return compose(excitation_map(self.flip_at(z)), relax)

        mz = np.where(n > k_top, fixed_point(pulse_map(k_top)), inflow[None])  # older history < 1e-12
        for k in range(k_top, 0, -1):
            mz = np.where(n >= k, apply(pulse_map(k), mz), mz)
        return mz

    def blood_signal(self, blood: TissueSample, flow: FlowGeometry, positions_mm: np.ndarray) -> np.ndarray:
        """Return the area-weighted laminar echo signal of blood at distances positions_mm from slab entry."""
        positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
        if positions.min() < 0 or positions.max() > flow.path_mm():
            raise ValueError(f"positions must lie within the slab path, 0 to {flow.path_mm():.3g} mm")
        scalar = np.ndim(blood.t1_ms) == 0 and np.ndim(flow.mean_velocity_mm_s) == 0
        blood = as_sample_axis(blood)
        shell_factors, weights = laminar_velocities(1.0, flow.n_shells)  # shell velocities are linear in the mean
        velocities = shell_factors[:, None] * np.atleast_1d(np.asarray(flow.mean_velocity_mm_s, dtype=float))
        n = pulses_received(positions[:, None, None], velocities[None], self.timing.tr_ms)
        inflow = self.inflow_mz(blood, flow, velocities)
        if self.tone_pct == 0:
            mz, flip = apply(power(self.tr_map(blood), n), inflow[None]), self.flip_deg
        else:
            mz = self.walked_mz(blood, flow, positions, velocities, inflow, n)
            flip = self.flip_at(self.blood_z(flow, positions))[:, None, None]
        out = np.einsum("psn,s->pn", self.echo(mz, flip, blood), weights)
        return out[:, 0] if scalar else out
