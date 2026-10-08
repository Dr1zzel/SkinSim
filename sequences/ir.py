"""Inversion recovery: non-selective inversion, TI, a spoiled readout train, recovery, repeat.

TI runs from the inversion to the first readout pulse; the k-space centre is
sampled center_line TRs later. Magnetisation does not start from M0 at each
inversion: the train and a finite recovery leave it partly saturated, so the
periodic steady state is solved rather than full recovery assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from sequences.base import Affine, SegmentedSequence, TissueSample, compose, relax_map


@dataclass(frozen=True, kw_only=True)
class IrSequence(SegmentedSequence):
    """Segmented inversion recovery with a spoiled GRE readout."""

    ti_ms: float
    inversion_efficiency: float = 1.0  # Mz -> -efficiency * Mz; 1 is a perfect inversion

    @property
    def pre_train_ms(self) -> float:
        """Return TI, inversion to first readout pulse."""
        return self.ti_ms

    def prep_map(self, tissue: TissueSample) -> Affine:
        """Return inversion followed by T1 recovery over TI."""
        return compose((-self.inversion_efficiency, 0.0), relax_map(self.ti_ms, tissue))


def nulling_ti(tissue: TissueSample, sequence: IrSequence) -> float:
    """Return the TI that nulls the tissue at the centre line under the segmented periodic steady state."""
    if sequence.inversion_efficiency <= 0:
        raise ValueError("no null without inversion")

    def centre_mz(ti_ms: float) -> float:
        return float(replace(sequence, ti_ms=ti_ms).train_mz(tissue)[sequence.center_line])

    if centre_mz(0.0) >= 0:
        raise ValueError(f"{tissue.name} recovers past zero within {sequence.center_line} lines "
                         "even at TI = 0; move the centre line earlier")
    lo, hi = 0.0, tissue.t1_ms
    while centre_mz(hi) < 0:
        lo, hi = hi, 2 * hi
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if centre_mz(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def textbook_nulling_ti(tissue: TissueSample) -> float:
    """Return T1 ln 2, the null for perfect inversion from full recovery with the null on the first line."""
    return tissue.t1_ms * math.log(2)
