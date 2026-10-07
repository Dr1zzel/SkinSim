import math

import numpy as np
import pytest

from bloch import (
    apply_rf,
    equilibrium,
    ernst_signal,
    relax,
    spoil,
    spoiled_train,
)

M0, T1_MS, T2_MS, TR_MS, FLIP_DEG = 1.3, 1600.0, 180.0, 5.0, 20.0


def matrix_spoiled_signals(n_pulses, t2_ms, spoiled, off_resonance_hz=0.0):
    state, signals = equilibrium(M0), []
    for _ in range(n_pulses):
        state = apply_rf(state, FLIP_DEG, 0.0)
        signals.append(abs(state.transverse[0]))
        state = relax(state, TR_MS, T1_MS, t2_ms, off_resonance_hz)
        if spoiled:
            state = spoil(state)
    return np.array(signals)


@pytest.mark.parametrize("flip_deg, phase_deg", [(90.0, 0.0), (90.0, 37.0), (30.0, 90.0), (135.0, 200.0)])
def test_pulse_from_equilibrium_leaves_sin_transverse_and_cos_longitudinal(flip_deg, phase_deg):
    state = apply_rf(equilibrium(M0), flip_deg, phase_deg)
    alpha = math.radians(flip_deg)
    assert abs(state.transverse[0]) == pytest.approx(M0 * math.sin(alpha), abs=1e-12)
    assert state.m[0, 2] == pytest.approx(M0 * math.cos(alpha), abs=1e-12)


def test_static_spins_converge_to_ernst_steady_state():
    expected = ernst_signal(M0, FLIP_DEG, TR_MS, T1_MS)
    assert matrix_spoiled_signals(3000, T2_MS, spoiled=True)[-1] == pytest.approx(expected, rel=1e-9)
    signal, _ = spoiled_train(M0, M0, FLIP_DEG, TR_MS, T1_MS, 3000)
    assert signal[-1] == pytest.approx(expected, rel=1e-9)


def test_scalar_and_matrix_paths_agree_under_ideal_spoiling():
    scalar, _ = spoiled_train(M0, M0, FLIP_DEG, TR_MS, T1_MS, 300)
    matrix = matrix_spoiled_signals(300, T2_MS, spoiled=True, off_resonance_hz=73.0)
    np.testing.assert_allclose(matrix, scalar, rtol=1e-12, atol=1e-14)


def test_scalar_and_matrix_paths_agree_unspoiled_when_t2_kills_coherence():
    scalar, _ = spoiled_train(M0, M0, FLIP_DEG, TR_MS, T1_MS, 300)
    matrix = matrix_spoiled_signals(300, TR_MS / 100.0, spoiled=False)
    np.testing.assert_allclose(matrix, scalar, rtol=1e-12, atol=1e-14)


def test_unspoiled_bssfp_reaches_analytic_on_resonance_steady_state():
    t1_ms, t2_ms, flip_deg = 300.0, 100.0, 40.0
    state = equilibrium(M0)
    for k in range(4000):
        state = apply_rf(state, flip_deg, 180.0 * (k % 2))
        signal = abs(state.transverse[0])
        state = relax(state, TR_MS, t1_ms, t2_ms)
    e1, e2, alpha = math.exp(-TR_MS / t1_ms), math.exp(-TR_MS / t2_ms), math.radians(flip_deg)
    expected = M0 * math.sin(alpha) * (1 - e1) / (1 - (e1 - e2) * math.cos(alpha) - e1 * e2)
    assert signal == pytest.approx(expected, rel=1e-9)


def test_operations_preserve_positions():
    positions = np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 5.0]])
    state = spoil(relax(apply_rf(equilibrium(M0, 2, positions), 30.0, 0.0), TR_MS, T1_MS, T2_MS))
    np.testing.assert_array_equal(state.position_mm, positions)
