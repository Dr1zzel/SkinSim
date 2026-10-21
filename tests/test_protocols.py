import math

import pytest

from protocols import (
    ALL_PROTOCOLS,
    CANDIDATES,
    CONSOLE_TE_MS,
    CONSOLE_TR_MS,
    OCT20_TE_MS,
    MEASURED,
    MEASURED_13OCT,
    MEASURED_20OCT,
    PRISMA_XR,
    RF_DURATION_MS,
    GradientSystem,
    Protocol,
    encoding_area,
    fl3d1r_timing,
    infer_rf_duration_ms,
    min_lobe_ms,
)


@pytest.mark.parametrize("protocol", MEASURED.values(), ids=lambda p: p.name)
def test_readout_timing_is_internally_consistent(protocol):
    """TR is the sum of its named components and the readout is the reciprocal bandwidth."""
    t = protocol.timing
    assert t.readout_ms == pytest.approx(1000.0 / t.bandwidth_per_px_hz)
    assert t.dead_time_ms >= 0
    assert t.tr_ms == pytest.approx(t.rf_duration_ms + t.rephaser_ms + t.prephaser_ms + t.readout_ms
                                    + t.spoiler_ms + t.dead_time_ms)


@pytest.mark.parametrize("session, console_te_ms, console_tr_ms", [
    (MEASURED_13OCT, CONSOLE_TE_MS, CONSOLE_TR_MS),
    (MEASURED_20OCT, OCT20_TE_MS, None),  # 20 October: TE shared, TR differs per arm
])
def test_each_session_reproduces_its_own_console_te(session, console_te_ms, console_tr_ms):
    """TE comes back by construction for every arm, and TR too where the session shares one."""
    for protocol in session.values():
        assert protocol.timing.te_ms == pytest.approx(console_te_ms, abs=0.2)
        if console_tr_ms is not None:
            assert protocol.timing.tr_ms == pytest.approx(console_tr_ms, abs=0.2)


def test_inferred_rf_duration_is_positive_and_below_a_few_ms():
    assert 0.0 < RF_DURATION_MS < 5.0


def test_te_shorter_than_the_lobes_is_refused_and_so_is_an_overfull_tr():
    with pytest.raises(ValueError, match="component model is wrong"):
        infer_rf_duration_ms(4.0, 0.2578, 0.23, 186.0, PRISMA_XR)
    with pytest.raises(ValueError):
        fl3d1r_timing(0.2578, 0.23, 186.0, 2.0, 8.0, PRISMA_XR)


def test_encoding_area_and_lobe_shapes():
    assert encoding_area(0.2578) == pytest.approx(45.55, abs=0.01)  # 1 / (2 gamma dx) in mT ms/m
    g = GradientSystem(80.0, 200.0)
    corner = 80.0**2 / 200.0  # largest area a triangle delivers
    assert min_lobe_ms(corner, g) == pytest.approx(0.8)
    assert min_lobe_ms(corner * (1 - 1e-9), g) == pytest.approx(min_lobe_ms(corner * (1 + 1e-9), g))
    assert min_lobe_ms(4 * corner, g) == pytest.approx(4 * corner / 80.0 + 0.4)


def test_derived_counts_for_the_finest_measured_series():
    p = MEASURED["13oct_0.26mm"]
    assert p.slab_thickness_mm == pytest.approx(11.04)
    assert p.voxel_volume_mm3 == pytest.approx(0.2578**2 * 0.23)
    n_phase = round(55.7 / 0.2578)
    assert p.lines_per_partition == math.ceil((n_phase - 32) / 2) + 32
    assert p.acquired_partitions == math.ceil(48 * 1.25 * 7 / 8)
    assert p.scan_time_s == pytest.approx(p.total_lines * 45.22 / 1000.0, rel=1e-6)


def test_no_acceleration_acquires_every_phase_line():
    p = CANDIDATES["no_acceleration"]
    assert p.lines_per_partition == round(p.phase_fov_mm / p.inplane_mm)


def test_finer_inplane_needs_a_longer_prephaser_so_te_grows():
    assert CANDIDATES["inplane_0.18mm"].timing.te_ms > MEASURED["13oct_0.26mm"].timing.te_ms


def test_measured_and_candidate_protocols_share_one_type():
    assert all(isinstance(p, Protocol) for p in ALL_PROTOCOLS.values())
    assert all(p.acquired is None and p.console_ta_s is None for p in CANDIDATES.values())
    assert all(p.acquired is not None for p in MEASURED.values())
    assert len(MEASURED_13OCT) == 5 and len(MEASURED_20OCT) == 4
    assert MEASURED == {**MEASURED_13OCT, **MEASURED_20OCT}
