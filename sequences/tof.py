"""Time-of-flight: fresh blood flows into a spoiled-GRE slab whose static tissue sits at steady state.

Assumes an ideal rectangular slab profile (every spin inside sees flip_deg) and
ideal spoiling, so only Mz carries between pulses (the scalar path of bloch.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sequences.base import FlowGeometry, Sequence, TissueSample
from skin import laminar_velocities


def mz_before_pulse(
    n_pulses: np.ndarray, mz_initial: float, m0: float, flip_deg: float, tr_ms: float, t1_ms: float
) -> np.ndarray:
    """Return Mz just before the next pulse after n_pulses ideally spoiled pulses, n_pulses = inf giving the steady state."""
    e1 = math.exp(-tr_ms / t1_ms)
    q = e1 * math.cos(math.radians(flip_deg))
    mz_ss = m0 * (1 - e1) / (1 - q)
    return mz_ss + (mz_initial - mz_ss) * np.power(q, n_pulses)


def pulses_received(positions_mm: np.ndarray, velocity_mm_s: np.ndarray, tr_ms: float) -> np.ndarray:
    """Return how many whole TRs blood at velocity_mm_s has spent travelling positions_mm into the slab, inf when static."""
    step_mm = np.asarray(velocity_mm_s, dtype=float) * tr_ms / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.floor(np.asarray(positions_mm, dtype=float) / step_mm)
    return np.where(step_mm > 0, n, np.inf)


@dataclass(frozen=True, kw_only=True)
class TofSequence(Sequence):
    """Spoiled-GRE time-of-flight with inflow of unsaturated blood."""

    flip_deg: float
    inflow_mz_fraction: float = 1.0  # fully relaxed inflow; lower it for blood that crossed an upstream slab

    def static_signal(self, tissue: TissueSample) -> float:
        """Return the echo signal of a static tissue at its spoiled steady state."""
        mz = mz_before_pulse(np.inf, tissue.pd_rel, tissue.pd_rel, self.flip_deg,
                             self.timing.tr_ms, tissue.t1_ms)
        return float(mz) * math.sin(math.radians(self.flip_deg)) * self.readout_decay(tissue.t2star_ms)

    def blood_signal(self, blood: TissueSample, flow: FlowGeometry, positions_mm: np.ndarray) -> np.ndarray:
        """Return the area-weighted laminar echo signal of blood at distances positions_mm from slab entry."""
        positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
        if positions.min() < 0 or positions.max() > flow.path_mm():
            raise ValueError(f"positions must lie within the slab path, 0 to {flow.path_mm():.3g} mm")
        velocities, weights = laminar_velocities(flow.mean_velocity_mm_s, flow.n_shells)
        n = pulses_received(positions[:, None], velocities[None, :], self.timing.tr_ms)
        mz = mz_before_pulse(n, self.inflow_mz_fraction * blood.pd_rel, blood.pd_rel,
                             self.flip_deg, self.timing.tr_ms, blood.t1_ms)
        echo = math.sin(math.radians(self.flip_deg)) * self.readout_decay(blood.t2star_ms)
        return (mz * echo) @ weights
