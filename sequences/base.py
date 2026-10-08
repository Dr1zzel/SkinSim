"""Shared sequence interface: an optional preparation followed by a readout train.

Units: ms, mm, mm/s, Hz for bandwidth. Angles are degrees at the API boundary.
Every timing is derived from named components; nothing is assumed.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from functools import cache

import numpy as np

from acquisition import Slab, path_length
from bloch import spoiled_train
from skin import Vessel, laminar_velocities
from tissues import Tissue


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
class TissueSample:
    """One point drawn from a tissue's parameter intervals, the values a simulation runs on."""

    name: str
    t1_ms: float
    t2_ms: float
    t2star_ms: float
    pd_rel: float

    @classmethod
    def nominal(cls, tissue: Tissue) -> TissueSample:
        """Return the tissue's nominal values; sweeps construct samples directly."""
        return cls(tissue.name, tissue.t1_ms.nominal, tissue.t2_ms.nominal,
                   tissue.t2star_ms.nominal, tissue.pd_rel.nominal)


@dataclass(frozen=True)
class FlowGeometry:
    """Flow of one vessel through one slab, with laminar shells and the explicit path cap."""

    vessel: Vessel
    slab: Slab
    mean_velocity_mm_s: float
    n_shells: int
    max_path_mm: float  # no justification; must be swept (see acquisition.path_length)

    def __post_init__(self) -> None:
        if self.mean_velocity_mm_s < 0:
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

    def readout_decay(self, t2star_ms: float) -> float:
        """Return the T2* decay from the RF centre to the echo."""
        return math.exp(-self.timing.te_ms / t2star_ms)

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


# --- Segmented sequences: preparation -> spoiled readout train -> recovery -> repeat ---
# With ideal spoiling every segment is an affine map on Mz, Mz_out = A * Mz_in + B,
# so a whole cycle is one affine map and its periodic steady state is B / (1 - A).

Affine = tuple[float, float]
IDENTITY: Affine = (1.0, 0.0)


def compose(*maps: Affine) -> Affine:
    """Return the single affine map that applies the given maps in order."""
    a, b = IDENTITY
    for a_i, b_i in maps:
        a, b = a_i * a, a_i * b + b_i
    return a, b


def apply(m: Affine, mz: float) -> float:
    """Return Mz after the affine map."""
    return m[0] * mz + m[1]


def power(m: Affine, n: float) -> Affine:
    """Return the affine map applied n times; n = inf gives the constant map to its fixed point."""
    if n == 0:
        return IDENTITY
    a_n = m[0] ** n
    return a_n, m[1] * (1 - a_n) / (1 - m[0])


def fixed_point(m: Affine) -> float:
    """Return the Mz the affine map leaves unchanged, B / (1 - A)."""
    return m[1] / (1 - m[0])


def relax_map(duration_ms: float, tissue: TissueSample) -> Affine:
    """Return T1 recovery towards the tissue's proton density over duration_ms."""
    e1 = math.exp(-duration_ms / tissue.t1_ms)
    return e1, tissue.pd_rel * (1 - e1)


def train_map(n_pulses: int, flip_deg: float, tr_ms: float, tissue: TissueSample) -> Affine:
    """Return the map across n spoiled pulses each followed by TR, derived from bloch.spoiled_train at two inputs."""
    if n_pulses == 0:
        return IDENTITY
    _, mz = spoiled_train(np.array([0.0, 1.0]), tissue.pd_rel, flip_deg, tr_ms, tissue.t1_ms, n_pulses + 1)
    b = float(mz[0, n_pulses])
    return float(mz[1, n_pulses]) - b, b


def time_in_slab_ms(position_mm: float, velocity_mm_s: float) -> float:
    """Return how long blood at velocity_mm_s has taken to travel position_mm into the slab, inf when static."""
    if velocity_mm_s <= 0:
        return math.inf
    return 1000.0 * position_mm / velocity_mm_s


@dataclass(frozen=True, kw_only=True)
class SegmentedSequence(Sequence):
    """Preparation, a spoiled readout train and a recovery delay, repeated to a periodic steady state.

    The preparation is non-selective, so inflowing blood was prepared upstream
    exactly like static tissue; the readout is slab-selective, so only time spent
    inside the slab saturates. Times below are relative to the first readout pulse.
    """

    flip_deg: float
    lines_per_train: int
    recovery_ms: float  # end of the train's last TR to the next preparation
    center_line: int = 0  # line sampling the k-space centre: 0 centric, lines_per_train // 2 linear
    preparation: Preparation | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 0 <= self.center_line < self.lines_per_train:
            raise ValueError(f"center_line must lie in 0..{self.lines_per_train - 1}")
        object.__setattr__(self, "preparation",
                           Preparation(self.pre_train_ms, self.recovery_ms, self.lines_per_train))

    @property
    @abstractmethod
    def pre_train_ms(self) -> float:
        """Return the time from the start of the preparation to the first readout pulse."""

    @abstractmethod
    def prep_map(self, tissue: TissueSample) -> Affine:
        """Return the map on Mz from the start of the preparation to the first readout pulse."""

    @property
    def train_and_recovery_ms(self) -> float:
        """Return the time from the first readout pulse to the next preparation."""
        return self.lines_per_train * self.timing.tr_ms + self.recovery_ms

    def in_slab_map(self, tissue: TissueSample, start_ms: float, stop_ms: float) -> Affine:
        """Return the map from start_ms to just before stop_ms for a spin inside the slab, readout pulses included."""
        tr, n_lines = self.timing.tr_ms, self.lines_per_train
        first = min(n_lines, math.ceil(start_ms / tr - 1e-9))
        last = n_lines if stop_ms >= n_lines * tr else math.ceil(stop_ms / tr - 1e-9)
        n = max(0, last - first)
        if n == 0:
            return relax_map(stop_ms - start_ms, tissue)
        tail_ms = max(0.0, stop_ms - (first + n) * tr)
        return compose(relax_map(first * tr - start_ms, tissue),
                       train_map(n, self.flip_deg, tr, tissue), relax_map(tail_ms, tissue))

    @cache
    def train_start_mz(self, tissue: TissueSample, inside: bool) -> float:
        """Return the periodic steady-state Mz at the first readout pulse, inside or outside the slab."""
        rest = self.train_and_recovery_ms
        after = self.in_slab_map(tissue, 0.0, rest) if inside else relax_map(rest, tissue)
        return fixed_point(compose(after, self.prep_map(tissue)))

    def train_mz(self, tissue: TissueSample) -> np.ndarray:
        """Return the signed steady-state Mz of a static tissue just before each readout pulse."""
        mz0 = self.train_start_mz(tissue, inside=True)
        _, mz = spoiled_train(mz0, tissue.pd_rel, self.flip_deg, self.timing.tr_ms,
                              tissue.t1_ms, self.lines_per_train)
        return mz

    def train_signals(self, tissue: TissueSample) -> np.ndarray:
        """Return the echo signal magnitude of a static tissue at every line of the train."""
        echo = math.sin(math.radians(self.flip_deg)) * self.readout_decay(tissue.t2star_ms)
        return np.abs(self.train_mz(tissue)) * echo

    def static_signal(self, tissue: TissueSample) -> float:
        """Return the echo signal of a static tissue at the k-space centre line."""
        return float(self.train_signals(tissue)[self.center_line])

    def blood_mz(self, blood: TissueSample, tau_ms: float) -> float:
        """Return signed Mz just before the centre line for blood that has spent tau_ms inside the slab."""
        now = self.center_line * self.timing.tr_ms
        entry = now - tau_ms
        if entry >= 0:  # entered during the current train, prepared but unsaturated until then
            mz = apply(relax_map(entry, blood), self.train_start_mz(blood, inside=False))
            return apply(self.in_slab_map(blood, entry, now), mz)
        mz, full_cycles = self.mz_at_next_train_start(blood, entry)
        cycle = compose(self.in_slab_map(blood, 0.0, self.train_and_recovery_ms), self.prep_map(blood))
        mz = apply(power(cycle, full_cycles), mz)
        return apply(self.in_slab_map(blood, 0.0, now), mz)

    def mz_at_next_train_start(self, blood: TissueSample, entry_ms: float) -> tuple[float, float]:
        """Return (Mz at the first train start after an entry at entry_ms < 0, whole in-slab cycles left before now)."""
        rest, outside = self.train_and_recovery_ms, self.train_start_mz(blood, inside=False)
        period = rest + self.pre_train_ms
        if math.isinf(entry_ms):  # static blood
            return outside, math.inf
        if math.isinf(period):  # entered during an infinitely long recovery: relaxed, then prepared
            return outside, 0
        cycles_back = math.ceil(-entry_ms / period)
        phase = entry_ms + cycles_back * period  # time after that cycle's first readout pulse
        if phase >= rest:  # entered during the non-selective preparation, so in = out
            return outside, cycles_back - 1
        mz = apply(relax_map(phase, blood), outside)
        mz = apply(compose(self.in_slab_map(blood, phase, rest), self.prep_map(blood)), mz)
        return mz, cycles_back - 1

    def blood_signal(self, blood: TissueSample, flow: FlowGeometry, positions_mm: np.ndarray) -> np.ndarray:
        """Return the area-weighted laminar echo signal of blood at the centre line at distances positions_mm from slab entry."""
        positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
        if positions.min() < 0 or positions.max() > flow.path_mm():
            raise ValueError(f"positions must lie within the slab path, 0 to {flow.path_mm():.3g} mm")
        velocities, weights = laminar_velocities(flow.mean_velocity_mm_s, flow.n_shells)
        echo = math.sin(math.radians(self.flip_deg)) * self.readout_decay(blood.t2star_ms)
        mz = np.array([[self.blood_mz(blood, time_in_slab_ms(s, v)) for v in velocities] for s in positions])
        return (np.abs(mz) * echo) @ weights
