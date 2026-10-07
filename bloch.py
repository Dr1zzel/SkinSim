"""Bloch engine: spin state, RF rotation, relaxation with off-resonance precession, spoiling.

Units: ms, mm, Hz for off-resonance. Angles are degrees at the API boundary,
radians inside. Rotations are right-handed: an RF pulse of phase 0 rotates
about +x, and positive off-resonance precesses counter-clockwise about +z.

Two paths:
- the matrix engine acts on full (Mx, My, Mz) vectors of N isochromats and
  never spoils implicitly, so transverse coherence persists across TRs unless
  spoil() is called, as bSSFP requires;
- the scalar path tracks Mz only and assumes ideal spoiling before every pulse,
  for spoiled GRE and TOF.
Signals are taken immediately after the pulse; decay to TE belongs to the readout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SpinState:
    """Magnetisation of N isochromats with their equilibrium magnetisation and spatial positions."""

    m: np.ndarray  # (N, 3): Mx, My, Mz
    m0: np.ndarray  # (N,): equilibrium longitudinal magnetisation
    position_mm: np.ndarray  # (N, 3) in the skin frame; carried for FSD, unused so far

    def __post_init__(self) -> None:
        n = self.m.shape[0]
        if self.m.shape != (n, 3) or self.m0.shape != (n,) or self.position_mm.shape != (n, 3):
            raise ValueError("need m (N, 3), m0 (N,) and position_mm (N, 3)")

    @property
    def transverse(self) -> np.ndarray:
        """Return the complex transverse magnetisation Mx + i My of each isochromat."""
        return self.m[:, 0] + 1j * self.m[:, 1]


def equilibrium(
    m0: float | np.ndarray, n_spins: int = 1, position_mm: np.ndarray | None = None
) -> SpinState:
    """Return N isochromats at rest along +z with magnitude m0, at the origin unless positions are given."""
    m0_arr = np.broadcast_to(np.asarray(m0, dtype=float), (n_spins,)).copy()
    m = np.zeros((n_spins, 3))
    m[:, 2] = m0_arr
    if position_mm is None:
        position_mm = np.zeros((n_spins, 3))
    return SpinState(m, m0_arr, np.asarray(position_mm, dtype=float).reshape(n_spins, 3))


def rf_rotation(flip_deg: float, phase_deg: float) -> np.ndarray:
    """Return the 3x3 right-handed rotation by flip_deg about the transverse axis at phase_deg from +x."""
    alpha, phi = math.radians(flip_deg), math.radians(phase_deg)
    ux, uy = math.cos(phi), math.sin(phi)
    cross = np.array([[0.0, 0.0, uy], [0.0, 0.0, -ux], [-uy, ux, 0.0]])
    outer = np.array([[ux * ux, ux * uy, 0.0], [ux * uy, uy * uy, 0.0], [0.0, 0.0, 0.0]])
    return math.cos(alpha) * np.eye(3) + math.sin(alpha) * cross + (1 - math.cos(alpha)) * outer


def apply_rf(state: SpinState, flip_deg: float, phase_deg: float) -> SpinState:
    """Return the state after an instantaneous RF pulse of flip_deg at phase_deg."""
    return replace(state, m=state.m @ rf_rotation(flip_deg, phase_deg).T)


def relax(
    state: SpinState,
    duration_ms: float,
    t1_ms: float | np.ndarray,
    t2_ms: float | np.ndarray,
    off_resonance_hz: float | np.ndarray = 0.0,
) -> SpinState:
    """Return the state after free precession at off_resonance_hz with T1 recovery and T2 decay for duration_ms."""
    if duration_ms < 0:
        raise ValueError(f"duration must be non-negative, got {duration_ms} ms")
    e1 = np.exp(-duration_ms / np.asarray(t1_ms, dtype=float))
    e2 = np.exp(-duration_ms / np.asarray(t2_ms, dtype=float))
    theta = 2.0 * np.pi * np.asarray(off_resonance_hz, dtype=float) * duration_ms / 1000.0
    c, s = np.cos(theta), np.sin(theta)
    mx, my, mz = state.m.T
    m = np.stack([e2 * (c * mx - s * my), e2 * (s * mx + c * my), state.m0 + (mz - state.m0) * e1], axis=1)
    return replace(state, m=m)


def spoil(state: SpinState) -> SpinState:
    """Return the state with all transverse magnetisation destroyed (ideal spoiling)."""
    m = state.m.copy()
    m[:, :2] = 0.0
    return replace(state, m=m)


def ernst_signal(m0: float, flip_deg: float, tr_ms: float, t1_ms: float) -> float:
    """Return the ideally spoiled steady-state transverse magnetisation just after the pulse."""
    e1 = math.exp(-tr_ms / t1_ms)
    alpha = math.radians(flip_deg)
    return m0 * math.sin(alpha) * (1 - e1) / (1 - e1 * math.cos(alpha))


def spoiled_train(
    mz_initial: float | np.ndarray,
    m0: float | np.ndarray,
    flip_deg: float | np.ndarray,
    tr_ms: float,
    t1_ms: float | np.ndarray,
    n_pulses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (signal just after each pulse, Mz just before each pulse) for an ideally spoiled train, in closed form, broadcasting over inputs with pulses on the last axis."""
    e1 = np.exp(-tr_ms / np.asarray(t1_ms, dtype=float))[..., None]
    alpha = np.radians(np.asarray(flip_deg, dtype=float))[..., None]
    m0 = np.asarray(m0, dtype=float)[..., None]
    mz_initial = np.asarray(mz_initial, dtype=float)[..., None]
    q = e1 * np.cos(alpha)
    mz_ss = m0 * (1 - e1) / (1 - q)
    mz_before = mz_ss + (mz_initial - mz_ss) * q ** np.arange(n_pulses)
    return mz_before * np.sin(alpha), mz_before
