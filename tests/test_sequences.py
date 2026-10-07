import math

import numpy as np
import pytest

from acquisition import Slab
from bloch import spoiled_train
from sequences.base import FlowGeometry, Preparation, ReadoutTiming, TissueSample
from sequences.tof import TofSequence, mz_before_pulse
from skin import Vessel
from tissues import ARTERIAL_BLOOD, DERMIS, HYPODERMIS

TIMING = ReadoutTiming(
    rf_duration_ms=1.0, rephaser_ms=0.5, prephaser_ms=0.6,
    bandwidth_per_px_hz=250.0, spoiler_ms=1.2, dead_time_ms=0.3,
)
BLOOD = TissueSample.nominal(ARTERIAL_BLOOD)
FAT = TissueSample.nominal(HYPODERMIS)


def flow(mean_velocity_mm_s: float, n_shells: int = 8) -> FlowGeometry:
    vessel = Vessel(depth_mm=4.0, polar_deg=60.0, azimuth_deg=0.0, diameter_mm=0.3)
    return FlowGeometry(vessel, Slab(5.0, 0.0), mean_velocity_mm_s, n_shells, max_path_mm=40.0)


def tof() -> TofSequence:
    return TofSequence(timing=TIMING, flip_deg=25.0)


def test_derived_tr_is_sum_of_stated_components():
    assert TIMING.readout_ms == pytest.approx(4.0)
    assert TIMING.tr_ms == pytest.approx(1.0 + 0.5 + 0.6 + 4.0 + 1.2 + 0.3)
    assert TIMING.te_ms == pytest.approx(0.5 + 0.5 + 0.6 + 2.0)


def test_time_per_line_amortises_preparation_and_recovery():
    seq = TofSequence(timing=TIMING, flip_deg=25.0, preparation=Preparation(40.0, 160.0, 50))
    assert seq.time_per_line_ms() == pytest.approx(TIMING.tr_ms + 200.0 / 50)
    assert tof().time_per_line_ms() == pytest.approx(TIMING.tr_ms)


def test_tof_signal_falls_monotonically_along_vessel():
    geometry = flow(10.0)
    signal = tof().blood_signal(BLOOD, geometry, np.linspace(0.0, geometry.path_mm(), 200))
    assert np.all(np.diff(signal) <= 1e-15)
    assert signal[-1] < signal[0]


def test_tof_reduces_to_static_steady_state_as_velocity_goes_to_zero():
    static = tof().static_signal(BLOOD)
    errors = [abs(tof().blood_signal(BLOOD, flow(v), [2.0])[0] - static) for v in (10.0, 3.0, 1.0)]
    assert errors[0] > errors[1] > errors[2] > 0
    assert tof().blood_signal(BLOOD, flow(0.01), [2.0])[0] == pytest.approx(static, abs=1e-12)
    assert tof().blood_signal(BLOOD, flow(0.0), [0.0, 2.0]) == pytest.approx([static, static])


def test_tof_inflow_matches_bloch_scalar_path():
    _, mz = spoiled_train(BLOOD.pd_rel, BLOOD.pd_rel, 25.0, TIMING.tr_ms, BLOOD.t1_ms, 50)
    n = np.arange(50)
    np.testing.assert_allclose(
        mz_before_pulse(n, BLOOD.pd_rel, BLOOD.pd_rel, 25.0, TIMING.tr_ms, BLOOD.t1_ms), mz, rtol=1e-12)


def test_contrast_states_background_and_refuses_the_wrong_one():
    geometry = flow(10.0)
    result = tof().contrast(BLOOD, FAT, geometry, [0.0, 1.0])
    assert result.background_name == HYPODERMIS.name
    assert result.ratio[0] > result.ratio[1]
    with pytest.raises(ValueError):
        tof().contrast(BLOOD, TissueSample.nominal(DERMIS), geometry, [0.0])


def test_positions_beyond_slab_path_are_refused():
    geometry = flow(10.0)
    with pytest.raises(ValueError):
        tof().blood_signal(BLOOD, geometry, [geometry.path_mm() + 1.0])


def test_fresh_blood_at_entry_has_full_magnetisation():
    signal = tof().blood_signal(BLOOD, flow(10.0), [0.0])[0]
    expected = BLOOD.pd_rel * math.sin(math.radians(25.0)) * math.exp(-TIMING.te_ms / BLOOD.t2star_ms)
    assert signal == pytest.approx(expected)
