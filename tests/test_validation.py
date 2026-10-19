"""Literature validation, ported from SkinMRA tests/test_engines.py.

These check the signal model against textbook physics and a published
operating point. They are not fits to this project's data, and they are what
makes a simulation-only prediction credible to a reviewer. The Ernst steady
state is checked against tests/reference.py.
"""
import math

import numpy as np
import pytest

from geometry import Slab, Vessel
from sequence import FlowGeometry, ReadoutTiming, TofSequence
from reference import ernst_signal, spoiled_train
from tissues import TissueSample


def fresh_spin_signal(flip_deg: float, tr_ms: float, t1_ms: float, k: int) -> float:
    """Classical fresh-spin inflow signal after k prior pulses, unit M0 (Bernstein, King & Zhou ch. 13;
    Haacke et al. ch. 17): Mz_k = (1 - E1)(1 - r^k)/(1 - r) + r^k, r = E1 cos(alpha)."""
    e1 = math.exp(-tr_ms / t1_ms)
    r = e1 * math.cos(math.radians(flip_deg))
    return ((1 - e1) * (1 - r**k) / (1 - r) + r**k) * math.sin(math.radians(flip_deg))


def test_inflow_enhancement_matches_classical_expression_and_brain_tof_operating_point():
    """Typical 3 T brain 3D TOF (TR ~25 ms, FA ~20 deg) with mean cerebral arterial speed of a few hundred
    mm/s through a several-cm slab gives inflow signal several-fold above the saturated steady state
    (Parker et al., MRM 1991)."""
    flip_deg, slab_mm, velocity_mm_s, t1_ms = 20.0, 50.0, 300.0, 1650.0
    timing = ReadoutTiming(rf_duration_ms=2.0, rephaser_ms=1.0, prephaser_ms=1.0,
                           bandwidth_per_px_hz=200.0, spoiler_ms=2.0, dead_time_ms=14.0)
    assert timing.tr_ms == pytest.approx(25.0)
    blood = TissueSample("blood", t1_ms=t1_ms, t2_ms=200.0, t2star_ms=1e12, pd_rel=1.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=3.0)
    flow = FlowGeometry(vessel, Slab(slab_mm, 90.0), velocity_mm_s, n_shells=1, max_path_mm=1e3)
    assert flow.path_mm() == pytest.approx(slab_mm)  # through-slab flow, brain-like geometry

    k = math.floor(slab_mm / (velocity_mm_s * timing.tr_ms / 1000.0))  # prior pulses at slab exit
    assert 1 <= k <= 15
    seq = TofSequence(timing=timing, flip_deg=flip_deg)
    exit_signal = seq.blood_signal(blood, flow, [slab_mm])[0]
    _, mz = spoiled_train(1.0, 1.0, flip_deg, timing.tr_ms, t1_ms, k + 1)
    assert exit_signal == pytest.approx(fresh_spin_signal(flip_deg, timing.tr_ms, t1_ms, k), rel=1e-10)
    assert mz[k] * math.sin(math.radians(flip_deg)) == pytest.approx(exit_signal, rel=1e-10)
    assert 2.0 < exit_signal / ernst_signal(1.0, flip_deg, timing.tr_ms, t1_ms) < 8.0
