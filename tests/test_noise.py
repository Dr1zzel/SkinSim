import math

import numpy as np
import pytest

from noise import noise_floor_mean, sigma_from_background, thermal_snr, true_signal

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
