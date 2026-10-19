"""From signal to detection: channel noise statistics, CNR efficiency, and what a vessel needs to be seen.

Three layers, none of which contains a measured number:

- NOISE. A sum-of-squares combination of N channels gives magnitudes
  distributed as a non-central chi with 2N degrees of freedom, not Rician
  (which is the N = 1 case). Its second moment is exact and simple,
  E[M^2] = A^2 + 2 N sigma^2, which is what true_signal inverts.
- EFFICIENCY. Absolute contrast per square root of the time spent per k-space
  line. Never a ratio: in the sensitivity-limited regime the background can be
  near zero, where a ratio is meaningless.
- DETECTABILITY. What SNR a vessel of diameter d needs at voxel size a, and
  what width it appears to have. The peak voxel holds a blood fraction
  f = kappa * f_max, with f_max the area-ratio maximum and kappa the
  grid-registration factor, DERIVED by averaging the exact peak-voxel fraction
  over uniformly random sub-voxel offsets. CNR of the peak voxel is
  f |contrast| SNR / sqrt(1 + (structure_fraction SNR)^2), and detection needs
  CNR >= the Rose threshold.

The vessel is a straight cylinder; voxel_mm is its extent in the plane
perpendicular to it. The point-spread function enters only apparent_fwhm,
never the blood fraction.

Units: mm, ms.
"""
from __future__ import annotations

import math
from functools import lru_cache

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


# --- Efficiency: absolute contrast per sqrt(time per k-space line) ---


def cnr_efficiency(s_blood: ArrayLike, s_background: ArrayLike, time_per_line_ms: ArrayLike) -> np.ndarray:
    """Return (S_blood - S_background) / sqrt(time per k-space line) per sqrt(s), broadcasting, refusing non-positive times."""
    t = np.asarray(time_per_line_ms, dtype=float)
    if np.any(t <= 0):
        raise ValueError("time per k-space line must be positive")
    diff = np.asarray(s_blood, dtype=float) - np.asarray(s_background, dtype=float)
    return diff / np.sqrt(t / 1000.0)


# --- Detectability: blood fraction, required SNR and apparent width ---


ROSE = 4.0  # CNR threshold for reliable visual detection (Rose, 1948)
KAPPA_SEED, KAPPA_OFFSETS = 20261014, 4096  # recorded: the derivation of kappa is deterministic
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


def _pair(voxel_mm: float | tuple[float, float]) -> tuple[float, float]:
    return (voxel_mm, voxel_mm) if np.ndim(voxel_mm) == 0 else (float(voxel_mm[0]), float(voxel_mm[1]))


def geometric_max_fraction(diameter_mm: float, voxel_mm: float | tuple[float, float]) -> float:
    """Return the area-ratio maximum blood fraction, min(1, vessel cross-section / voxel cross-section)."""
    ay, az = _pair(voxel_mm)
    return min(1.0, math.pi * diameter_mm**2 / 4 / (ay * az))


def circle_rect_area(r: float, x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """Return the area where a centred circle of radius r overlaps rectangles [x0, x1] x [y0, y1], by Gauss-Legendre."""
    lo, hi = np.maximum(x0, -r), np.minimum(x1, r)
    width = np.maximum(hi - lo, 0.0)
    x = 0.5 * (lo + hi)[..., None] + 0.5 * width[..., None] * _NODES
    h = np.sqrt(np.maximum(r * r - x * x, 0.0))
    chord = np.maximum(np.minimum(y1[..., None], h) - np.maximum(y0[..., None], -h), 0.0)
    return 0.5 * width * np.sum(chord * _WEIGHTS, axis=-1)


def peak_fractions(diameter_mm: float, voxel_mm: float | tuple[float, float], offsets: np.ndarray) -> np.ndarray:
    """Return the exact peak-voxel blood fraction for each sub-voxel offset (rows, in voxel units) of the vessel centre."""
    ay, az = _pair(voxel_mm)
    i, j = np.meshgrid([-1, 0, 1], [-1, 0, 1])  # the peak voxel is the centre's voxel or a neighbour
    x0 = (i.ravel()[None, :] - offsets[:, :1]) * ay
    y0 = (j.ravel()[None, :] - offsets[:, 1:]) * az
    area = circle_rect_area(diameter_mm / 2, x0, x0 + ay, y0, y0 + az)
    return np.max(area, axis=1) / (ay * az)


@lru_cache(maxsize=4096)
def _kappa(d_over_ay: float, d_over_az: float) -> float:
    offsets = np.random.default_rng(KAPPA_SEED).uniform(0.0, 1.0, (KAPPA_OFFSETS, 2))
    mean_peak = float(np.mean(peak_fractions(d_over_ay, (1.0, d_over_ay / d_over_az), offsets)))
    return mean_peak / geometric_max_fraction(d_over_ay, (1.0, d_over_ay / d_over_az))


def derive_kappa(diameter_mm: float, voxel_mm: float | tuple[float, float]) -> float:
    """Return kappa, the mean exact peak-voxel fraction over random sub-voxel offsets divided by the area-ratio maximum."""
    ay, az = _pair(voxel_mm)
    return _kappa(round(diameter_mm / ay, 9), round(diameter_mm / az, 9))


def blood_fraction(diameter_mm: float, voxel_mm: float | tuple[float, float], kappa: float | None = None) -> float:
    """Return the blood volume fraction of the peak voxel, kappa times the area-ratio maximum, kappa derived if not given."""
    k = derive_kappa(diameter_mm, voxel_mm) if kappa is None else kappa
    return k * geometric_max_fraction(diameter_mm, voxel_mm)


def cnr(contrast: ArrayLike, diameter_mm: float, voxel_mm: float | tuple[float, float], snr: ArrayLike,
        structure_fraction: ArrayLike = 0.0, kappa: float | None = None) -> np.ndarray:
    """Return the peak-voxel CNR, f |contrast| SNR / sqrt(1 + (structure_fraction SNR)^2)."""
    f = blood_fraction(diameter_mm, voxel_mm, kappa)
    snr = np.asarray(snr, dtype=float)
    return f * np.abs(contrast) * snr / np.sqrt(1 + (np.asarray(structure_fraction) * snr) ** 2)


def snr_required(contrast: ArrayLike, diameter_mm: float, voxel_mm: float | tuple[float, float],
                 structure_fraction: ArrayLike = 0.0, cnr_threshold: float = ROSE,
                 kappa: float | None = None) -> np.ndarray:
    """Return the reference SNR needed to reach the threshold, inf when texture alone defeats the contrast."""
    signal = blood_fraction(diameter_mm, voxel_mm, kappa) * np.abs(np.asarray(contrast, dtype=float))
    floor = cnr_threshold * np.asarray(structure_fraction, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(signal > floor, cnr_threshold / np.sqrt(signal**2 - floor**2), np.inf)


def min_detectable_diameter(contrast: float, voxel_mm: float | tuple[float, float], snr_available: float,
                            structure_fraction: float = 0.0, cnr_threshold: float = ROSE,
                            kappa: float | None = None) -> float:
    """Return the smallest diameter whose required SNR is met by snr_available (bisection in log d), inf if none is."""
    a = min(_pair(voxel_mm))
    need = lambda d: snr_required(contrast, d, voxel_mm, structure_fraction, cnr_threshold, kappa)  # noqa: E731
    lo, hi = 1e-4 * a, 20.0 * a
    if need(hi) > snr_available:
        return math.inf
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if need(mid) > snr_available else (lo, mid)
    return hi


def optimal_voxel(diameter_mm: float) -> float:
    """Return sqrt(pi / 4) d, the voxel whose area equals the vessel cross-section (exact for a fixed kappa)."""
    # With kappa derived from d / a the optimum moves with texture: no finite
    # optimum without texture (larger voxels stop splitting the vessel), about
    # 1.1 d at structure_fraction 0.05, about 0.86 d at 0.1 (fixed scan time).
    return math.sqrt(math.pi / 4) * diameter_mm


def sensitivity_gain_needed(diameter_mm: float, voxel_mm: float | tuple[float, float], contrast: ArrayLike,
                            snr_available: ArrayLike, structure_fraction: ArrayLike = 0.0,
                            cnr_threshold: float = ROSE, kappa: float | None = None) -> np.ndarray:
    """Return the factor by which SNR must rise to detect the vessel, snr_required / snr_available."""
    need = snr_required(contrast, diameter_mm, voxel_mm, structure_fraction, cnr_threshold, kappa)
    return need / np.asarray(snr_available, dtype=float)


def apparent_fwhm(diameter_mm: float, voxel_mm: float, psf_voxels: float = 1.2) -> float:
    """Return the FWHM of the vessel's profile blurred by a Gaussian PSF of FWHM psf_voxels voxels; it saturates at the PSF."""
    sigma = psf_voxels * voxel_mm / (2 * math.sqrt(2 * math.log(2)))
    r = diameter_mm / 2
    xs = np.linspace(-r, r, 801)  # source columns across the vessel
    across = np.array([math.erf(math.sqrt(max(r * r - x * x, 0.0)) / (sigma * math.sqrt(2))) for x in xs])
    grid = np.linspace(-(r + 6 * sigma), r + 6 * sigma, 4001)
    profile = np.exp(-((grid[:, None] - xs[None, :]) ** 2) / (2 * sigma**2)) @ across
    above = grid[profile >= profile.max() / 2]
    return float(above[-1] - above[0])
