import pytest

from skin import laminar_velocities, tissue_at_depth
from tissues import DERMIS, EPIDERMIS, HYPODERMIS, MUSCLE


@pytest.mark.parametrize(
    "depth_mm, expected",
    [(0.0, EPIDERMIS), (0.1, DERMIS), (2.0, HYPODERMIS), (10.0, MUSCLE)],
)
def test_tissue_at_depth_at_each_boundary(depth_mm, expected):
    assert tissue_at_depth(depth_mm) is expected


@pytest.mark.parametrize(
    "depth_mm, expected",
    [(0.0999, EPIDERMIS), (1.999, DERMIS), (9.999, HYPODERMIS)],
)
def test_tissue_at_depth_just_above_each_boundary(depth_mm, expected):
    assert tissue_at_depth(depth_mm) is expected


def test_laminar_area_weighted_mean_is_half_centreline():
    velocities, weights = laminar_velocities(10.0, 1000)
    mean = float((velocities * weights).sum())
    assert weights.sum() == pytest.approx(1.0)
    assert mean == pytest.approx(10.0)
    assert mean == pytest.approx(0.5 * velocities[0], rel=1e-5)


def test_laminar_mean_is_exact_for_few_shells():
    velocities, weights = laminar_velocities(7.0, 3)
    assert float((velocities * weights).sum()) == pytest.approx(7.0)
