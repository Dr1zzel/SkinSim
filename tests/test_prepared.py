import math
from dataclasses import replace

import numpy as np
import pytest

from acquisition import Slab
from sequences.base import FlowGeometry, ReadoutTiming, SegmentedSequence, TissueSample
from sequences.ir import IrSequence, nulling_ti, textbook_nulling_ti
from sequences.t2prep import T2PrepSequence
from skin import Vessel
from tissues import ARTERIAL_BLOOD, DERMIS, HYPODERMIS

TIMING = ReadoutTiming(rf_duration_ms=1.0, rephaser_ms=0.5, prephaser_ms=1.0,
                       bandwidth_per_px_hz=250.0, spoiler_ms=1.5, dead_time_ms=0.2)
BLOOD, DERMIS_S, FAT = (TissueSample.nominal(t) for t in (ARTERIAL_BLOOD, DERMIS, HYPODERMIS))


def ir(**changes) -> IrSequence:
    seq = IrSequence(timing=TIMING, flip_deg=10.0, lines_per_train=64, recovery_ms=500.0, ti_ms=200.0)
    return replace(seq, **changes)


def t2prep(**changes) -> T2PrepSequence:
    seq = T2PrepSequence(timing=TIMING, flip_deg=10.0, lines_per_train=64, recovery_ms=500.0,
                         prep_ms=50.0, n_refocus=4, crusher_ms=2.0)
    return replace(seq, **changes)


@pytest.mark.parametrize("efficiency", [1.0, 0.9])
def test_ir_with_infinite_recovery_reduces_to_textbook(efficiency):
    seq = ir(recovery_ms=math.inf, inversion_efficiency=efficiency)
    expected = FAT.pd_rel * (1 - (1 + efficiency) * math.exp(-seq.ti_ms / FAT.t1_ms))
    assert seq.train_mz(FAT)[0] == pytest.approx(expected, rel=1e-12)


def test_t2prep_with_infinite_recovery_reduces_to_textbook():
    assert t2prep(recovery_ms=math.inf, crusher_ms=0.0).train_mz(BLOOD)[0] == pytest.approx(
        BLOOD.pd_rel * math.exp(-50.0 / BLOOD.t2_ms), rel=1e-12)


def test_ir_with_infinite_recovery_nulls_at_t1_ln2():
    assert nulling_ti(FAT, ir(recovery_ms=math.inf)) == pytest.approx(textbook_nulling_ti(FAT), abs=1e-4)


@pytest.mark.parametrize("tissue", [FAT, BLOOD, DERMIS_S])
@pytest.mark.parametrize("center_line", [0, 8])
def test_segmented_nulling_ti_is_shorter_than_t1_ln2(tissue, center_line):
    seq = ir(center_line=center_line)
    ti = nulling_ti(tissue, seq)
    assert ti < textbook_nulling_ti(tissue)
    assert replace(seq, ti_ms=ti).train_mz(tissue)[center_line] == pytest.approx(0.0, abs=1e-8)


def test_nulling_is_refused_when_train_recovers_tissue_before_centre_line():
    with pytest.raises(ValueError, match="move the centre line earlier"):
        nulling_ti(FAT, ir(center_line=32))


@pytest.mark.parametrize("background", [DERMIS_S, FAT])
def test_t2prep_contrast_falls_monotonically_along_train(background):
    seq = t2prep()
    ratio = seq.train_signals(BLOOD) / seq.train_signals(background)
    assert np.all(np.diff(ratio) < 0)


def test_time_per_line_amortises_preparation_and_recovery():
    assert ir().time_per_line_ms() == pytest.approx(TIMING.tr_ms + (200.0 + 500.0) / 64)
    assert t2prep().time_per_line_ms() == pytest.approx(TIMING.tr_ms + (50.0 + 2.0 + 500.0) / 64)
    assert replace(ir(), ti_ms=300.0).preparation.duration_ms == 300.0


def brute_force_blood_mz(seq: SegmentedSequence, tissue: TissueSample, tau_ms: float) -> float:
    """Step one spin event by event from far in the past; readout pulses act only after it enters the slab."""
    tr, n, cos_a = seq.timing.tr_ms, seq.lines_per_train, math.cos(math.radians(seq.flip_deg))
    period = seq.train_and_recovery_ms + seq.pre_train_ms
    now, entry = seq.center_line * tr, seq.center_line * tr - tau_ms
    first_cycle = -(math.ceil(tau_ms / period) + 300)
    mz, t = tissue.pd_rel, first_cycle * period
    relax = lambda m, dt: tissue.pd_rel + (m - tissue.pd_rel) * math.exp(-dt / tissue.t1_ms)
    for cycle in range(first_cycle, 1):
        for j in range(n):
            pulse_t = cycle * period + j * tr
            mz, t = relax(mz, pulse_t - t), pulse_t
            if cycle == 0 and j == seq.center_line:
                return mz
            mz *= cos_a if pulse_t >= entry else 1.0
        mz = relax(mz, cycle * period + seq.train_and_recovery_ms - t)
        a, b = seq.prep_map(tissue)
        mz, t = a * mz + b, (cycle + 1) * period
    raise AssertionError("unreachable")


@pytest.mark.parametrize("make", [lambda: ir(lines_per_train=20, recovery_ms=300.0, ti_ms=150.0, center_line=5),
                                  lambda: t2prep(lines_per_train=20, recovery_ms=300.0, center_line=5)])
def test_flowing_blood_matches_event_by_event_simulation(make):
    seq = make()
    period = seq.train_and_recovery_ms + seq.pre_train_ms
    now = seq.center_line * seq.timing.tr_ms
    for tau in [0.0, 0.5 * TIMING.tr_ms, 3.3 * TIMING.tr_ms, now + 100.0,
                now + 0.5 * seq.pre_train_ms, 2.5 * period, 7.2 * period]:
        assert seq.blood_mz(BLOOD, tau) == pytest.approx(brute_force_blood_mz(seq, BLOOD, tau), abs=1e-12)


@pytest.mark.parametrize("make", [ir, t2prep])
def test_blood_reduces_to_static_steady_state_as_velocity_goes_to_zero(make):
    seq = make()
    vessel = Vessel(depth_mm=1.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.2)
    static = seq.static_signal(BLOOD)
    for velocity, tolerance in [(0.01, 1e-9), (0.0, 1e-12)]:
        flow = FlowGeometry(vessel, Slab(10.0, 0.0), velocity, 8, 30.0)
        assert seq.blood_signal(BLOOD, flow, [5.0])[0] == pytest.approx(static, abs=tolerance)
