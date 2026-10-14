import math
from dataclasses import replace

import numpy as np
import pytest

from bloch import ernst_signal, spoiled_train
from sequences.base import ReadoutTiming, TissueSample
from sequences.t2prep import T2PrepSequence
from tissues import ARTERIAL_BLOOD, DERMIS, HYPODERMIS

TIMING = ReadoutTiming(rf_duration_ms=1.0, rephaser_ms=0.5, prephaser_ms=1.0,
                       bandwidth_per_px_hz=250.0, spoiler_ms=1.5, dead_time_ms=0.2)
TISSUES = [TissueSample.nominal(t) for t in (ARTERIAL_BLOOD, DERMIS, HYPODERMIS)]


def t2prep(**changes) -> T2PrepSequence:
    seq = T2PrepSequence(timing=TIMING, flip_deg=12.0, lines_per_train=48, recovery_ms=400.0,
                         prep_ms=60.0, n_refocus=4, crusher_ms=3.0)
    return replace(seq, **changes)


@pytest.mark.parametrize("tissue", TISSUES, ids=lambda t: t.name)
def test_periodic_steady_state_is_a_fixed_point(tissue):
    seq = t2prep()
    mz0 = seq.train_start_mz(tissue, inside=True)
    _, mz = spoiled_train(mz0, tissue.pd_rel, seq.flip_deg, TIMING.tr_ms, tissue.t1_ms, seq.lines_per_train + 1)
    after_train = mz[-1]
    e1 = math.exp(-seq.recovery_ms / tissue.t1_ms)
    before_prep = tissue.pd_rel + (after_train - tissue.pd_rel) * e1
    a, b = seq.prep_map(tissue)
    assert a * before_prep + b == pytest.approx(mz0, rel=1e-12)


@pytest.mark.parametrize("tissue", TISSUES, ids=lambda t: t.name)
def test_zero_length_prep_reduces_to_the_readout_train_alone(tissue):
    seq = t2prep(prep_ms=0.0, crusher_ms=0.0, recovery_ms=0.0)
    assert seq.prep_map(tissue) == (1.0, 0.0)
    ernst_mz = ernst_signal(tissue.pd_rel, seq.flip_deg, TIMING.tr_ms, tissue.t1_ms) / math.sin(
        math.radians(seq.flip_deg))
    np.testing.assert_allclose(seq.train_mz(tissue), ernst_mz, rtol=1e-12)


def test_time_per_line_amortises_prep_and_recovery():
    seq = t2prep()
    assert seq.time_per_line_ms() == pytest.approx(TIMING.tr_ms + (60.0 + 3.0 + 400.0) / 48)
