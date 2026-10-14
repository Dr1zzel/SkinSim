"""Detectability: what SNR a vessel of diameter d needs at voxel size a, and what width it appears to have.

Physics only, no measured numbers (CLAUDE.md). Definitions:

- S_ref is a chosen reference signal and SNR = S_ref / sigma. contrast is
  (S_vessel - S_background) / S_ref. With S_ref = the background, SNR is the
  background SNR of QUESTIONS.md Q1; against a suppressed background, whose
  SNR tends to zero, pass S_ref = the blood signal so the answer stays finite.
- structure_fraction is background texture, a standard deviation as a
  fraction of S_ref that does not average down with thermal noise.
- The peak voxel holds a blood fraction f = kappa * f_max, with f_max the
  area-ratio maximum min(1, (pi d^2 / 4) / voxel area) and kappa the
  grid-registration factor, DERIVED by averaging the exact peak-voxel fraction
  over uniformly random sub-voxel offsets of the vessel (derive_kappa). It
  depends on d / a; it is never asserted.
- CNR of the peak voxel = f |contrast| SNR / sqrt(1 + (structure_fraction SNR)^2),
  and detection needs CNR >= the Rose threshold.

Geometry: the vessel is a straight cylinder; voxel_mm is the voxel's extent in
the plane perpendicular to it (a scalar for a square cross-section, or a pair).
The PSF enters only apparent_fwhm, not the blood fraction.
Units: mm.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

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
