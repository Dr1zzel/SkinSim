"""Non-selective saturation pulses within each TR: one mechanism for fat saturation and MTC.

A SaturationPulse leaves a fraction s of Mz and is played delay_ms before each
excitation. With the excitation, the map from just before one excitation to
just before the next is

    excite(alpha) -> relax(TR - delay) -> saturate(s) -> relax(delay)

whose fixed point is Mz just before the excitation, and S = sin(alpha) Mz.
(This is the order that gives the closed form A = s cos(alpha) E_d E_r,
B = M0 ((1 - E_r) s E_d + (1 - E_d)); it is derived here by composition and
the closed form is only a test.) Engines run with unit M0; signal_weighting
supplies proton density and T2*.

The per-tissue s values live on Tissue (fatsat_s, mtc_s). Two caveats:
- Dermis fatsat_s below 1 is an MT proxy, not spectral suppression.
- MTC as a per-pulse multiplicative s is crude: real MT saturation builds up in
  the bound pool across pulses and transfers continuously. It is the honest
  level of detail for a quantity whose only planned measurement is one on/off
  comparison, but s is then an effective number, not a pulse property.

The travelling saturation band is spatially selective, so it is not here: it
sets the inflow boundary condition (sequences/tof.py, band_exit_mz below).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from affine import Affine, apply, compose, power, relax_map
from tissues import TissueSample


def excitation_map(flip_deg: ArrayLike) -> Affine:
    """Return the effect of an excitation of flip_deg on Mz, followed by ideal spoiling."""
    return np.cos(np.radians(flip_deg)), 0.0


def saturation_map(s: ArrayLike) -> Affine:
    """Return a pulse that leaves a fraction s of Mz."""
    return s, 0.0


@dataclass(frozen=True)
class SaturationPulse:
    """A spatially non-selective pulse that leaves a fraction s of Mz, played delay_ms before each excitation."""

    s: ArrayLike
    delay_ms: float

    def map(self, tissue: TissueSample, tr_ms: float) -> Affine:
        """Return the map from just after one excitation to just before the next, unit M0."""
        return between_excitations((self,), tissue, tr_ms)


def between_excitations(pulses: tuple[SaturationPulse, ...], tissue: TissueSample, tr_ms: float) -> Affine:
    """Return relaxation across one TR with each pulse played delay_ms before the next excitation, unit M0."""
    maps, elapsed = [], 0.0
    for pulse in sorted(pulses, key=lambda p: -p.delay_ms):
        if not 0.0 <= pulse.delay_ms <= tr_ms:
            raise ValueError(f"delay {pulse.delay_ms} ms must lie within TR {tr_ms} ms")
        maps += [relax_map(tr_ms - pulse.delay_ms - elapsed, tissue.t1_ms), saturation_map(pulse.s)]
        elapsed = tr_ms - pulse.delay_ms
    return compose(*maps, relax_map(tr_ms - elapsed, tissue.t1_ms))


@dataclass(frozen=True)
class SaturationModule:
    """A sequence's non-selective saturation module: which per-tissue factor it applies and its delay."""

    factor: Literal["fatsat_s", "mtc_s"]
    delay_ms: float  # module pulse centre to excitation

    def pulse(self, tissue: TissueSample) -> SaturationPulse:
        """Return the pulse as this tissue experiences it."""
        return SaturationPulse(s=getattr(tissue, self.factor), delay_ms=self.delay_ms)


def mean_recovery_map(tr_ms: float, t1_ms: ArrayLike) -> Affine:
    """Return T1 recovery over a delay uniformly distributed in [0, TR), unit M0."""
    a = np.asarray(t1_ms, dtype=float) / tr_ms * (1 - np.exp(-tr_ms / np.asarray(t1_ms, dtype=float)))
    return a, 1 - a


def band_exit_mz(mz_in: ArrayLike, n_band_pulses: ArrayLike, pulse_residual: ArrayLike,
                 tissue: TissueSample, tr_ms: float) -> ArrayLike:
    """Return Mz / M0 of blood leaving a band that saturated it once per TR, n_band_pulses times, exiting at a random phase."""
    band_tr = compose(saturation_map(pulse_residual), relax_map(tr_ms, tissue.t1_ms))
    before_last = apply(power(band_tr, np.maximum(np.asarray(n_band_pulses) - 1, 0)), mz_in)
    after_last = apply(compose(saturation_map(pulse_residual), mean_recovery_map(tr_ms, tissue.t1_ms)), before_last)
    return np.where(np.asarray(n_band_pulses) > 0, after_last, mz_in)
