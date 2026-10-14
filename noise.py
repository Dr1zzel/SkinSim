"""Noise physics for sum-of-squares magnitude images from N receive channels.

A sum-of-squares combination of N channels, each with independent complex
Gaussian noise of standard deviation sigma per real and imaginary component,
gives magnitudes M distributed as a non-central chi with 2N degrees of freedom,
not Rician (which is the N = 1 case). Its second moment is exact and simple,
E[M^2] = A^2 + 2 N sigma^2, which is what true_signal inverts. Getting this
wrong makes every low-SNR result wrong, which is where this study lives.

Assumes equal, uncorrelated channel noise; correlated channels lower the
effective number of degrees of freedom.

Nothing here stores a measured number: sigma_from_background takes the
signal-free mean as an argument (the hook for later validation, CLAUDE.md).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def noise_floor_mean(n_channels: int) -> float:
    """Return E[chi_2N] / sigma, the mean magnitude of a signal-free region, sqrt(2) Gamma(N + 1/2) / Gamma(N)."""
    if n_channels < 1:
        raise ValueError("need at least one channel")
    return math.sqrt(2.0) * math.exp(math.lgamma(n_channels + 0.5) - math.lgamma(n_channels))


def true_signal(measured: ArrayLike, sigma: ArrayLike, n_channels: int = 4) -> np.ndarray:
    """Return sqrt(max(M^2 - 2 N sigma^2, 0)), exact when M is the root-mean-square magnitude of a uniform region."""
    m = np.asarray(measured, dtype=float)
    return np.sqrt(np.maximum(m**2 - 2 * n_channels * np.asarray(sigma, dtype=float) ** 2, 0.0))


def sigma_from_background(mean_signal_free: float, n_channels: int = 4) -> float:
    """Return the per-channel noise sigma from the mean magnitude of a signal-free region."""
    return mean_signal_free / noise_floor_mean(n_channels)


def thermal_snr(voxel_volume_mm3: ArrayLike, scan_time_s: ArrayLike, g_factor: ArrayLike = 1.0) -> np.ndarray:
    """Return relative thermal SNR, V sqrt(T) / g, for the actual (accelerated) scan time T."""
    # Relative: one sensitivity constant, set by coil, depth and sample and not
    # derivable here, multiplies every protocol alike. Acceleration enters through
    # the shorter T and through g; a separate R factor would count it twice.
    return np.asarray(voxel_volume_mm3, dtype=float) * np.sqrt(np.asarray(scan_time_s, dtype=float)) / g_factor
