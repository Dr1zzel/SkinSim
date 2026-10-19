"""Saturation pulses within each TR: fat saturation, MTC and the travelling band."""
from __future__ import annotations

import math

import numpy as np
import pytest

from affine import compose, fixed_point
from geometry import SatBand, Slab, Vessel
from protocols import CANDIDATES, MEASURED
from reference import ernst_signal
from sequence import (
    FlowGeometry, ReadoutTiming, SaturationModule, SaturationPulse, TofSequence,
    band_exit_mz, between_excitations, excitation_map, mean_recovery_map,
)
from tissues import ARTERIAL_BLOOD, DERMIS, HYPODERMIS, TissueSample

TR_MS, FLIP_DEG = 45.22, 18.0
BLOOD, DERMIS_S, FAT = (TissueSample.nominal(t) for t in (ARTERIAL_BLOOD, DERMIS, HYPODERMIS))
TIMING = ReadoutTiming(rf_duration_ms=1.0, rephaser_ms=1.0, prephaser_ms=1.0,
                       bandwidth_per_px_hz=186.0, spoiler_ms=3.0, dead_time_ms=TR_MS - 6.0 - 1000 / 186.0)


def closed_form_mz(s: float, delay_ms: float, flip_deg: float, tr_ms: float, t1_ms: float) -> float:
    """Section 4a: A = s cos(a) E_d E_r, B = (1 - E_r) s E_d + (1 - E_d), unit M0, Mz just before excitation."""
    e_d, e_r = math.exp(-delay_ms / t1_ms), math.exp(-(tr_ms - delay_ms) / t1_ms)
    a = s * math.cos(math.radians(flip_deg)) * e_d * e_r
    b = (1 - e_r) * s * e_d + (1 - e_d)
    return b / (1 - a)


@pytest.mark.parametrize("tissue", [BLOOD, DERMIS_S, FAT], ids=lambda t: t.name)
def test_unit_saturation_reproduces_plain_spoiled_gre(tissue):
    m = compose(excitation_map(FLIP_DEG), SaturationPulse(s=1.0, delay_ms=8.0).map(tissue, TR_MS))
    ernst_mz = ernst_signal(1.0, FLIP_DEG, TR_MS, tissue.t1_ms) / math.sin(math.radians(FLIP_DEG))
    assert fixed_point(m) == pytest.approx(ernst_mz, rel=1e-12)


@pytest.mark.parametrize("s, delay_ms", [(0.15, 8.0), (0.0, 3.0), (0.6, 30.0), (-0.2, 10.0)])
def test_closed_form_agrees_with_composition_and_500_tr_brute_force(s, delay_ms):
    t1 = FAT.t1_ms
    composed = fixed_point(compose(excitation_map(FLIP_DEG), SaturationPulse(s, delay_ms).map(FAT, TR_MS)))
    mz = 1.0
    for _ in range(500):
        mz *= math.cos(math.radians(FLIP_DEG))
        mz = 1 + (mz - 1) * math.exp(-(TR_MS - delay_ms) / t1)
        mz *= s
        mz = 1 + (mz - 1) * math.exp(-delay_ms / t1)
    expected = closed_form_mz(s, delay_ms, FLIP_DEG, TR_MS, t1)
    assert composed == pytest.approx(expected, rel=1e-12)
    assert mz == pytest.approx(expected, rel=1e-12)


def test_pulse_order_follows_delays_not_argument_order_and_delays_must_fit_in_tr():
    fatsat, mtc = SaturationPulse(0.2, 8.0), SaturationPulse(0.7, 20.0)
    assert between_excitations((fatsat, mtc), DERMIS_S, TR_MS) == pytest.approx(
        between_excitations((mtc, fatsat), DERMIS_S, TR_MS))
    with pytest.raises(ValueError):
        between_excitations((SaturationPulse(0.2, TR_MS + 1.0),), DERMIS_S, TR_MS)


def test_fat_saturation_suppresses_fat_and_leaves_blood_alone():
    plain = TofSequence(timing=TIMING, flip_deg=FLIP_DEG)
    fatsat = TofSequence(timing=TIMING, flip_deg=FLIP_DEG, modules=(SaturationModule("fatsat_s", 8.0),))
    assert fatsat.static_signal(BLOOD) == pytest.approx(plain.static_signal(BLOOD), rel=1e-12)
    assert fatsat.static_signal(FAT) < 0.5 * plain.static_signal(FAT)


def flow(polar_deg: float, azimuth_deg: float, velocity_mm_s: float = 40.0) -> FlowGeometry:
    vessel = Vessel(depth_mm=5.0, polar_deg=polar_deg, azimuth_deg=azimuth_deg, diameter_mm=0.5)
    return FlowGeometry(vessel, Slab(11.04, 90.0), velocity_mm_s, n_shells=4, max_path_mm=30.0)


def test_band_saturates_only_blood_entering_from_its_side():
    band = SatBand(thickness_mm=40.0, gap_mm=10.0, side=-1)  # blood flowing along +normal enters at -1
    seq = TofSequence(timing=TIMING, flip_deg=FLIP_DEG, sat_band=band)
    plain = TofSequence(timing=TIMING, flip_deg=FLIP_DEG)
    v = np.array([20.0, 40.0])
    assert np.all(seq.inflow_mz(BLOOD, flow(90.0, 0.0), v) < 0.5)  # through-slab, from the band side
    np.testing.assert_allclose(seq.inflow_mz(BLOOD, flow(90.0, 180.0), v), plain.inflow_mz(BLOOD, flow(90.0, 180.0), v))
    np.testing.assert_allclose(seq.inflow_mz(BLOOD, flow(90.0, 90.0), v), 1.0)  # parallel to the slab


def test_ideal_band_exit_then_gap_recovery_matches_hand_calculation():
    seq = TofSequence(timing=TIMING, flip_deg=FLIP_DEG, sat_band=SatBand(40.0, 10.0, side=-1))
    v = 40.0
    a_mean, b_mean = mean_recovery_map(TR_MS, BLOOD.t1_ms)
    exit_mz = b_mean  # an ideal pulse zeroes Mz, then a random fraction of TR recovers
    expected = 1 - (1 - exit_mz) * math.exp(-1000 * 10.0 / v / BLOOD.t1_ms)
    assert seq.inflow_mz(BLOOD, flow(90.0, 0.0, v), np.array([[v]]))[0, 0] == pytest.approx(expected, rel=1e-9)
    assert band_exit_mz(1.0, 0, 0.0, BLOOD, TR_MS) == 1.0  # no band pulse, no change


def test_protocols_build_their_saturation_modules():
    measured = MEASURED["13oct_0.35mm"].sequence()
    assert [m.factor for m in measured.modules] == ["fatsat_s"]
    assert measured.sat_band is not None
    assert CANDIDATES["fatsat_off"].sequence().modules == ()


def test_static_signal_vectorises_over_a_sample_axis():
    seq = MEASURED["13oct_0.35mm"].sequence()
    fats = TissueSample("fat", t1_ms=np.array([360.0, 400.0]), t2_ms=70.0, t2star_ms=25.0, pd_rel=1.4,
                        fatsat_s=np.array([0.05, 0.5]))
    out = seq.static_signal(fats, z_frac=0.5)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(seq.static_signal(TissueSample("fat", 360.0, 70.0, 25.0, 1.4, fatsat_s=0.05), 0.5))
