"""T2 preparation: non-selective 90 - n refocusing pulses - (-90), a spoiled readout train, recovery, repeat.

The prep stores Mz_in * exp(-prep_ms / T2) back along z; T1 recovery during the
prep is neglected because the magnetisation is transverse throughout it.
Blood T2 depends on the refocusing interval (Zhao et al., MRM 2007): the blood
T2 passed in must be the value for refocus_interval_ms = prep_ms / n_refocus,
not a generic blood T2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sequences.base import Affine, SegmentedSequence, TissueSample, compose, relax_map


@dataclass(frozen=True, kw_only=True)
class T2PrepSequence(SegmentedSequence):
    """Segmented T2-prepared spoiled GRE; blood T2 must match prep_ms / n_refocus (Zhao, MRM 2007)."""

    prep_ms: float  # tip-down to tip-up
    n_refocus: int
    crusher_ms: float  # tip-up to first readout pulse; T1 recovers during it

    def __post_init__(self) -> None:
        if self.prep_ms < 0 or self.crusher_ms < 0 or self.n_refocus < 1:
            raise ValueError("need non-negative durations and at least one refocusing pulse")
        super().__post_init__()

    @property
    def refocus_interval_ms(self) -> float:
        """Return the spacing between refocusing pulses, which sets the effective blood T2."""
        return self.prep_ms / self.n_refocus

    @property
    def pre_train_ms(self) -> float:
        """Return the prep plus crusher, tip-down to first readout pulse."""
        return self.prep_ms + self.crusher_ms

    def prep_map(self, tissue: TissueSample) -> Affine:
        """Return T2 decay over the prep, with no T1 recovery while transverse, then T1 recovery over the crusher."""
        return compose((math.exp(-self.prep_ms / tissue.t2_ms), 0.0), relax_map(self.crusher_ms, tissue))
