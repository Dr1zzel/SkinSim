"""Time-of-flight as fl3d1r runs it: readout timing, saturation pulses, TONE and the TOF signal model.

Every TR is one affine map on Mz (affine.py): the excitation, then relaxation
with any non-selective saturation pulses (fat saturation, MTC). Static tissue
sits at that map's fixed point. Blood arrives at the slab face with the inflow
boundary condition - the upstream steady state of the non-selective pulses,
saturated further if a travelling band lies on the entry side - and at distance
s along the vessel has received n = floor(s / (v TR)) excitations, which
power() gives in closed form.

Under TONE the flip varies with through-slab position z (0 at the slab's
-normal face, 1 at its +normal face), so each pulse has its own map and the
spin's pulses are walked one by one, sequentially in the pulse index and
vectorised over positions, shells and samples.

Assumes an ideal rectangular slab profile and ideal spoiling.

Shapes: positions (P,), laminar shells (S,), and a sample axis (N,) when the
TissueSample fields or the flow's mean velocity are arrays; shell velocities
are (S, N) and blood_signal returns (P,) or (P, N).

Units: ms, mm, mm/s, Hz for bandwidth. Angles are degrees at the API boundary.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from affine import Affine, apply, compose, fixed_point, power, relax_map
from geometry import SatBand, Slab, Vessel, entry_side, laminar_velocities, path_length
from tissues import TissueSample, signal_weighting  # TissueSample re-exported for callers


@dataclass(frozen=True)
class ReadoutTiming:
    """Named components of one readout TR, played back to back with no overlap."""

    rf_duration_ms: float
    rephaser_ms: float  # slice-select rephaser
    prephaser_ms: float  # readout prephaser
    bandwidth_per_px_hz: float  # readout duration = 1 / bandwidth per pixel
    spoiler_ms: float  # spoiler, or the balancing rewinder for bSSFP
    dead_time_ms: float

    def __post_init__(self) -> None:
        if self.bandwidth_per_px_hz <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth_per_px_hz} Hz/px")
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(f"{f.name} must be non-negative")

    @property
    def readout_ms(self) -> float:
        """Return the readout duration, 1 / bandwidth per pixel."""
        return 1000.0 / self.bandwidth_per_px_hz

    @property
    def tr_ms(self) -> float:
        """Return TR as the sum of its components."""
        return (self.rf_duration_ms + self.rephaser_ms + self.prephaser_ms
                + self.readout_ms + self.spoiler_ms + self.dead_time_ms)

    @property
    def te_ms(self) -> float:
        """Return TE from the RF centre to the centre of a symmetric (full) echo."""
        return self.rf_duration_ms / 2 + self.rephaser_ms + self.prephaser_ms + self.readout_ms / 2


@dataclass(frozen=True)
class Preparation:
    """A preparation module and the recovery after its train, both amortised over the lines it serves."""

    duration_ms: float
    recovery_ms: float  # wait after the train before the next preparation
    lines_per_train: int

    def __post_init__(self) -> None:
        if self.duration_ms < 0 or self.recovery_ms < 0 or self.lines_per_train < 1:
            raise ValueError("need non-negative durations and at least one line per train")


@dataclass(frozen=True)
class FlowGeometry:
    """Flow of one vessel through one slab, with laminar shells and the explicit path cap."""

    vessel: Vessel
    slab: Slab
    mean_velocity_mm_s: float | np.ndarray  # an array runs along the sample axis (TOF only)
    n_shells: int
    max_path_mm: float  # no justification; must be swept (see acquisition.path_length)
    entry_z_frac: float | None = None  # through-slab position of a vessel parallel to the slab, needed under TONE

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.mean_velocity_mm_s) < 0):
            raise ValueError("mean velocity must be non-negative; direction is set by the vessel")

    def path_mm(self) -> float:
        """Return the vessel's path length through the slab."""
        return path_length(self.slab, self.vessel, self.max_path_mm)


@dataclass(frozen=True)
class Contrast:
    """Blood signal along a vessel and the static background, labelled with the background tissue."""

    background_name: str
    blood: np.ndarray
    background: float

    @property
    def ratio(self) -> np.ndarray:
        """Return blood / background at each position."""
        return self.blood / self.background


@dataclass(frozen=True, kw_only=True)
class Sequence(ABC):
    """An optional preparation followed by a readout train."""

    timing: ReadoutTiming
    preparation: Preparation | None = None

    def time_per_line_ms(self) -> float:
        """Return the readout TR plus any preparation and recovery amortised over the train."""
        if self.preparation is None:
            return self.timing.tr_ms
        prep = self.preparation
        return self.timing.tr_ms + (prep.duration_ms + prep.recovery_ms) / prep.lines_per_train

    def echo(self, mz_over_m0: np.ndarray, flip_deg: float, tissue: TissueSample) -> np.ndarray:
        """Return the echo signal from Mz / M0 just before a pulse of flip_deg, via signal_weighting."""
        return mz_over_m0 * np.sin(np.radians(flip_deg)) * signal_weighting(tissue, self.timing.te_ms)

    @abstractmethod
    def static_signal(self, tissue: TissueSample) -> float:
        """Return the echo signal of a static tissue."""

    @abstractmethod
    def blood_signal(self, blood: TissueSample, flow: FlowGeometry, positions_mm: np.ndarray) -> np.ndarray:
        """Return the echo signal of blood at distances positions_mm along the vessel from slab entry."""

    def contrast(self, blood: TissueSample, background: TissueSample, flow: FlowGeometry,
                 positions_mm: np.ndarray) -> Contrast:
        """Return blood and background signals, refusing a background that does not match the vessel depth."""
        expected = flow.vessel.background().name
        if background.name != expected:
            raise ValueError(f"vessel at {flow.vessel.depth_mm} mm lies in {expected}, not {background.name}")
        return Contrast(background.name, self.blood_signal(blood, flow, positions_mm),
                        self.static_signal(background))


# --- Non-selective saturation pulses within each TR: fat saturation and MTC ---


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


# --- TONE: the through-slab flip-angle ramp ---


def flip_profile(z_frac: ArrayLike, ramp_pct: float, alpha_mean_deg: float) -> np.ndarray:
    """Return the local flip angle in degrees at normalised through-slab position z_frac along the declared flow direction."""
    if not 0.0 <= ramp_pct < 200.0:
        raise ValueError(f"ramp must lie in [0, 200) % to keep every flip positive, got {ramp_pct}")
    z = np.clip(np.asarray(z_frac, dtype=float), 0.0, 1.0)
    return alpha_mean_deg * (1.0 + ramp_pct / 100.0 * (z - 0.5))


# --- The TOF sequence itself ---


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
