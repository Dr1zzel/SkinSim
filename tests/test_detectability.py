import math

import numpy as np
import pytest

from detectability import (
    ROSE,
    blood_fraction,
    circle_rect_area,
    cnr,
    derive_kappa,
    geometric_max_fraction,
    min_detectable_diameter,
    optimal_voxel,
    peak_fractions,
    sensitivity_gain_needed,
    snr_required,
    apparent_fwhm,
)


def test_circle_rectangle_overlap_matches_known_areas():
    r = 0.7
    big = np.array([-5.0]), np.array([5.0]), np.array([-5.0]), np.array([5.0])
    assert circle_rect_area(r, *big)[0] == pytest.approx(math.pi * r * r, rel=1e-4)
    half = np.array([0.0]), np.array([5.0]), np.array([-5.0]), np.array([5.0])
    assert circle_rect_area(r, *half)[0] == pytest.approx(math.pi * r * r / 2, rel=1e-4)
    quarter = np.array([0.0]), np.array([5.0]), np.array([0.0]), np.array([5.0])
    assert circle_rect_area(r, *quarter)[0] == pytest.approx(math.pi * r * r / 4, rel=1e-4)


def test_peak_fraction_derivation_agrees_with_point_sampling():
    """Independent check of one offset: count random points of the vessel cross-section in each voxel."""
    d, offset = 0.9, np.array([[0.8, 0.3]])  # vessel centre in voxel units
    rng = np.random.default_rng(1)
    pts = rng.uniform(-d / 2, d / 2, (2_000_000, 2))
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= d / 2] + offset[0]
    counts = np.unique(np.floor(pts).astype(int), axis=0, return_counts=True)[1]
    sampled = counts.max() / len(pts) * (math.pi * d * d / 4)
    assert peak_fractions(d, 1.0, offset)[0] == pytest.approx(sampled, abs=2e-3)


@pytest.mark.parametrize("d_over_a", [0.05, 0.2, 0.5, 0.886, 1.0, 1.2, 1.5, 2.0, 4.0])
def test_kappa_derived_from_random_offsets_lands_in_half_to_one(d_over_a):
    assert 0.5 <= derive_kappa(d_over_a * 0.35, 0.35) <= 1.0


def test_kappa_is_scale_invariant_and_blood_fraction_uses_it():
    assert derive_kappa(0.3, 0.3) == pytest.approx(derive_kappa(0.6, 0.6))
    assert blood_fraction(0.3, 0.35) == pytest.approx(derive_kappa(0.3, 0.35) * geometric_max_fraction(0.3, 0.35))
    assert blood_fraction(0.3, 0.35, kappa=0.7) == pytest.approx(0.7 * geometric_max_fraction(0.3, 0.35))


def test_optimal_voxel_maximises_cnr_by_direct_search():
    """Fixed kappa, fixed partition and scan time, so thermal SNR scales with in-plane area; texture makes the optimum sharp."""
    d, texture, contrast = 0.4, 0.05, 0.8
    voxels = np.linspace(0.1, 1.2, 2201)
    cnrs = [cnr(contrast, d, a, snr=60.0 * a * a, structure_fraction=texture, kappa=0.7) for a in voxels]
    assert voxels[int(np.argmax(cnrs))] == pytest.approx(optimal_voxel(d), abs=voxels[1] - voxels[0])


@pytest.mark.parametrize("kappa", [0.7, None])
def test_min_detectable_diameter_and_snr_required_are_inverses(kappa):
    contrast, voxel, snr = 0.6, 0.35, 25.0
    d_min = min_detectable_diameter(contrast, voxel, snr, structure_fraction=0.01, kappa=kappa)
    assert snr_required(contrast, d_min, voxel, structure_fraction=0.01, kappa=kappa) == pytest.approx(snr, rel=1e-6)
    assert cnr(contrast, d_min, voxel, snr, structure_fraction=0.01, kappa=kappa) == pytest.approx(ROSE, rel=1e-6)


def test_texture_alone_can_defeat_the_contrast():
    f = blood_fraction(0.2, 0.35, kappa=0.7)
    assert snr_required(0.5, 0.2, 0.35, structure_fraction=f * 0.5 / ROSE * 1.01, kappa=0.7) == np.inf
    assert min_detectable_diameter(0.5, 0.35, 1.0) == math.inf  # SNR 1 cannot reach CNR 4 at any size
    assert sensitivity_gain_needed(0.3, 0.35, 0.5, 10.0) == pytest.approx(snr_required(0.5, 0.3, 0.35) / 10.0)


def test_apparent_fwhm_saturates_at_the_psf_below_it_and_is_linear_above_it():
    voxel, psf_voxels = 0.35, 1.2
    psf = psf_voxels * voxel
    for d in (0.01 * psf, 0.05 * psf):
        assert apparent_fwhm(d, voxel, psf_voxels) == pytest.approx(psf, rel=0.01)
    big = np.array([8.0, 10.0]) * psf
    widths = np.array([apparent_fwhm(d, voxel, psf_voxels) for d in big])
    np.testing.assert_allclose(widths, big, rtol=0.02)
    assert (widths[1] - widths[0]) / (big[1] - big[0]) == pytest.approx(1.0, abs=0.02)
