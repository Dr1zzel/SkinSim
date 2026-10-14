import math
from dataclasses import replace

import numpy as np
import pytest

from acquisition import Slab
from affine import compose, power, walk
from protocols import MEASURED
from sequences.base import FlowGeometry
from sequences.tof import as_sample_axis, mz_before_pulse, pulses_received
from skin import Vessel, laminar_velocities
from tissues import ARTERIAL_BLOOD, DERMIS, TissueSample, signal_weighting
from tone import flip_profile

BLOOD, DERMIS_S = TissueSample.nominal(ARTERIAL_BLOOD), TissueSample.nominal(DERMIS)
MEASURED_SEQ = MEASURED["13oct_0.35mm"].sequence()  # TONE 70 %, fat sat, band


def through_slab(velocity_mm_s: float, n_shells: int = 4, azimuth_deg: float = 180.0) -> FlowGeometry:
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=azimuth_deg, diameter_mm=0.5)
    return FlowGeometry(vessel, Slab(11.04, 90.0), velocity_mm_s, n_shells, max_path_mm=30.0)


def test_flip_profile_is_a_linear_ramp_about_the_nominal_angle():
    z = np.linspace(0.0, 1.0, 101)
    alpha = flip_profile(z, 70.0, 18.0)
    assert alpha.mean() == pytest.approx(18.0)
    assert alpha[-1] - alpha[0] == pytest.approx(0.7 * 18.0)
    assert np.all(np.diff(alpha) > 0)
    np.testing.assert_array_equal(flip_profile(z, 0.0, 18.0), 18.0)


def test_walk_of_identical_maps_equals_power():
    m = (0.93, 0.04)
    for n in (1, 5, 40):
        assert walk([m] * n) == pytest.approx(power(m, n), rel=1e-12)
    assert walk([(0.5, 0.1), (0.9, 0.2)]) == pytest.approx(compose((0.5, 0.1), (0.9, 0.2)))


def test_tone_zero_reproduces_the_constant_flip_result():
    """Plain spoiled GRE, no saturation modules: the pre-TONE closed form, element for element."""
    seq = replace(MEASURED_SEQ, tone_pct=0.0, modules=(), sat_band=None)
    flow = through_slab(20.0)
    positions = np.linspace(0.0, flow.path_mm(), 40)
    velocities, weights = laminar_velocities(flow.mean_velocity_mm_s, flow.n_shells)
    n = pulses_received(positions[:, None], velocities[None, :], seq.timing.tr_ms)
    mz = mz_before_pulse(n, 1.0, 1.0, seq.flip_deg, seq.timing.tr_ms, BLOOD.t1_ms)
    expected = (mz * math.sin(math.radians(seq.flip_deg)) * signal_weighting(BLOOD, seq.timing.te_ms)) @ weights
    np.testing.assert_allclose(seq.blood_signal(BLOOD, flow, positions), expected, rtol=1e-12)


def test_walked_path_with_a_flat_ramp_equals_the_closed_form():
    seq = replace(MEASURED_SEQ, tone_pct=0.0)
    flow = through_slab(8.0)
    positions = np.linspace(0.0, flow.path_mm(), 25)
    closed = seq.blood_signal(BLOOD, flow, positions)
    velocities, weights = laminar_velocities(flow.mean_velocity_mm_s, flow.n_shells)
    n = pulses_received(positions[:, None], velocities[None, :], seq.timing.tr_ms)
    blood = as_sample_axis(BLOOD)
    walked = seq.walked_mz(blood, flow, positions, velocities, seq.inflow_mz(blood, flow, velocities), n)
    walked_signal = np.einsum("psn,s->pn", seq.echo(walked, seq.flip_deg, blood), weights)[:, 0]
    np.testing.assert_allclose(walked_signal, closed, rtol=1e-10)


def test_tone_walk_matches_a_pulse_by_pulse_loop_for_one_spin():
    seq = replace(MEASURED_SEQ, modules=(), sat_band=None)  # TONE 70 % only
    flow = through_slab(15.0, n_shells=1)  # one shell: the spin moves at the mean velocity
    s_now = 9.0
    step = flow.mean_velocity_mm_s * seq.timing.tr_ms / 1000.0
    n = math.floor(s_now / step)
    mz, e1 = 1.0, math.exp(-seq.timing.tr_ms / BLOOD.t1_ms)
    for k in range(n, 0, -1):
        z = 1 - (s_now - k * step) / flow.slab.thickness_mm  # entered at the +normal face
        mz = mz * math.cos(math.radians(18.0 * (1 + 0.7 * (z - 0.5))))
        mz = 1 + (mz - 1) * e1
    alpha_now = 18.0 * (1 + 0.7 * ((1 - s_now / flow.slab.thickness_mm) - 0.5))
    expected = mz * math.sin(math.radians(alpha_now)) * signal_weighting(BLOOD, seq.timing.te_ms)
    assert seq.blood_signal(BLOOD, flow, [s_now])[0] == pytest.approx(expected, rel=1e-12)


def test_tone_static_signal_needs_a_position_and_matches_constant_flip_at_mid_slab():
    with pytest.raises(ValueError):
        MEASURED_SEQ.static_signal(DERMIS_S)
    flat = replace(MEASURED_SEQ, tone_pct=0.0)
    assert MEASURED_SEQ.static_signal(DERMIS_S, z_frac=0.5) == pytest.approx(flat.static_signal(DERMIS_S), rel=1e-12)


def test_slow_blood_under_tone_converges_to_the_local_static_signal():
    """Slow blood still drifts through the ramp during its memory, so it reaches the local static value only as v -> 0."""
    s = 6.0
    static = MEASURED_SEQ.static_signal(BLOOD, z_frac=MEASURED_SEQ.blood_z(through_slab(1.0), s))
    errors = [abs(MEASURED_SEQ.blood_signal(BLOOD, through_slab(v), [s])[0] - static) for v in (0.1, 0.01, 0.001)]
    assert errors[0] > errors[1] > errors[2]
    assert MEASURED_SEQ.blood_signal(BLOOD, through_slab(0.0), [s])[0] == pytest.approx(static, rel=1e-12)


def test_vessel_parallel_to_slab_needs_its_through_slab_position_under_tone():
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=90.0, diameter_mm=0.5)
    flow = FlowGeometry(vessel, Slab(11.04, 90.0), 20.0, 4, max_path_mm=30.0)
    with pytest.raises(ValueError, match="entry_z_frac"):
        MEASURED_SEQ.blood_signal(BLOOD, flow, [1.0])
    flow = replace(flow, entry_z_frac=0.25)
    assert np.isfinite(MEASURED_SEQ.blood_signal(BLOOD, flow, [1.0, 10.0])).all()
