"""Ported from SkinMRA tests/test_analysis.py: the CNR-efficiency metric."""
import numpy as np
import pytest

from analysis import cnr_efficiency


def test_value_and_units():
    assert cnr_efficiency(0.5, 0.2, 1000.0) == pytest.approx(0.3)
    assert cnr_efficiency(0.5, 0.2, 10.0) == pytest.approx(0.3 / np.sqrt(0.01))


def test_quartering_time_per_line_doubles_efficiency():
    assert cnr_efficiency(1.0, 0.5, 5.0) == pytest.approx(2 * cnr_efficiency(1.0, 0.5, 20.0))


def test_signed_and_antisymmetric():
    assert cnr_efficiency(0.4, 0.4, 20.0) == 0.0
    assert cnr_efficiency(0.2, 0.5, 20.0) == pytest.approx(-cnr_efficiency(0.5, 0.2, 20.0))


def test_absolute_not_ratio_so_a_near_zero_background_stays_finite():
    assert np.isfinite(cnr_efficiency(0.3, 0.0, 45.0))
    assert cnr_efficiency(3 * 0.5, 3 * 0.2, 20.0) == pytest.approx(3 * cnr_efficiency(0.5, 0.2, 20.0))


def test_broadcasts():
    s_blood = np.array([[0.2], [0.4]])
    out = cnr_efficiency(s_blood, 0.1, np.array([10.0, 20.0, 40.0]))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, (s_blood - 0.1) / np.sqrt(np.array([10.0, 20.0, 40.0]) / 1000.0))


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_rejects_non_positive_time(bad):
    with pytest.raises(ValueError):
        cnr_efficiency(0.5, 0.2, bad)
