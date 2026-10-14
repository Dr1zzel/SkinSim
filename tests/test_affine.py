import numpy as np
import pytest

from affine import IDENTITY, apply, compose, fixed_point, power, relax_map

M1 = (0.9, 0.05)
M2 = (-0.4, 0.3)


def test_compose_applies_maps_in_order():
    assert apply(compose(M1, M2), 0.7) == pytest.approx(apply(M2, apply(M1, 0.7)))
    assert compose() == IDENTITY


def test_power_matches_repeated_composition():
    for n in (1, 2, 7):
        a, b = power(M1, n)
        a_ref, b_ref = compose(*[M1] * n)
        assert (a, b) == pytest.approx((a_ref, b_ref), rel=1e-12)
    assert power(M1, 0) == IDENTITY


def test_infinite_power_is_the_constant_map_to_the_fixed_point():
    a, b = power(M2, np.inf)
    assert a == 0.0
    assert b == pytest.approx(fixed_point(M2))
    assert apply(M2, fixed_point(M2)) == pytest.approx(fixed_point(M2))


def test_relax_map_limits_and_broadcasting():
    assert relax_map(0.0, 1000.0, 1.3) == pytest.approx((1.0, 0.0))
    a, b = relax_map(np.inf, 1000.0, 1.3)
    assert (a, b) == pytest.approx((0.0, 1.3))
    a, b = relax_map(100.0, np.array([370.0, 1650.0]), np.array([1.4, 1.25]))
    np.testing.assert_allclose(a, np.exp(-100.0 / np.array([370.0, 1650.0])))
    np.testing.assert_allclose(b, np.array([1.4, 1.25]) * (1 - a))
