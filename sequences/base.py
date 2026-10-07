"""Shared sequence interface: an optional preparation followed by a readout train.

Units: ms, mm, mm/s, Hz for bandwidth. Angles are degrees at the API boundary.
Every timing is derived from named components; nothing is assumed.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import numpy as np

from acquisition import Slab, path_length
from skin import Vessel
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
