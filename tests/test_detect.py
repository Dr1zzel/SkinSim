"""Channel noise statistics, CNR efficiency and detectability."""
from __future__ import annotations

# --- from test_noise.py ---
import math

import numpy as np
import pytest

from detect import noise_floor_mean, sigma_from_background, thermal_snr, true_signal

N_CHANNELS, SIGMA = 4, 1.0


def sos_magnitudes(amplitude: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate sum-of-squares magnitudes of 4 channels sharing the signal unequally, complex Gaussian noise."""
    weights = np.array([0.6, 0.3, 0.08, 0.02])  # channel power fractions, as from a surface array
    signal = amplitude * np.sqrt(weights)
    noise = rng.normal(0, SIGMA, (n_samples, N_CHANNELS)) + 1j * rng.normal(0, SIGMA, (n_samples, N_CHANNELS))
    return np.sqrt(np.sum(np.abs(signal + noise) ** 2, axis=1))


def test_noise_floor_mean_is_rayleigh_for_one_channel_and_grows_with_channels():
    assert noise_floor_mean(1) == pytest.approx(math.sqrt(math.pi / 2))
    assert noise_floor_mean(4) == pytest.approx(math.sqrt(2) * math.gamma(4.5) / math.gamma(4))
    assert noise_floor_mean(4) > noise_floor_mean(1)


@pytest.mark.parametrize("snr", [0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
def test_true_signal_recovers_input_from_simulated_four_channel_sos(snr):
    rng = np.random.default_rng(20261014)
    m = sos_magnitudes(snr * SIGMA, 400_000, rng)
    rms = np.sqrt(np.mean(m**2))
    assert true_signal(rms, SIGMA, N_CHANNELS) == pytest.approx(snr * SIGMA, rel=0.05)
    if snr <= 1.0:  # the naive magnitude is badly biased exactly where this study lives
        assert np.mean(m) > 2 * snr * SIGMA


def test_sigma_from_background_round_trip():
    rng = np.random.default_rng(7)
    m = sos_magnitudes(0.0, 400_000, rng)
    assert sigma_from_background(float(np.mean(m)), N_CHANNELS) == pytest.approx(SIGMA, rel=0.01)


def test_thermal_snr_scales_with_volume_root_time_and_g():
    assert thermal_snr(2.0, 4.0) == pytest.approx(2 * thermal_snr(1.0, 4.0))
    assert thermal_snr(1.0, 4.0) == pytest.approx(2 * thermal_snr(1.0, 1.0))
    assert thermal_snr(1.0, 1.0, g_factor=1.25) == pytest.approx(0.8)


# --- from test_analysis.py ---
import numpy as np
import pytest

from detect import cnr_efficiency


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


# --- from test_detectability.py ---
import math

import numpy as np
import pytest

from detect import (
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
