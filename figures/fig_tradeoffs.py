"""General figure 2: what decides whether a cutaneous vessel is seen at all.

(a) A transit-time threshold. Against unsuppressed subcutaneous fat, inflowing
    blood is brighter only while it crosses the excited slab faster than a
    critical transit time. That threshold is a TIME, not a velocity: it barely
    moves with slab thickness, TR or flip angle, so it cannot be escaped by
    retuning the sequence. Suppressing the fat removes it altogether.

(b) Voxel size relative to vessel diameter. At fixed scan time the available
    SNR scales with voxel volume while the required SNR scales with the inverse
    peak-voxel blood fraction, and both depend on the voxel-to-diameter ratio
    alone. Detection plateaus once the voxel exceeds the vessel; apparent width
    is only truthful well below it. The two regimes do not overlap.

Run: python figures/fig_general_2_tradeoffs.py
"""
from __future__ import annotations

import math

import numpy as np
from common import INK, INK_2, SERIES, N_SHELLS, band, plt, save, style

from acquisition import Slab
from detectability import apparent_fwhm, blood_fraction
from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from sequences.base import FlowGeometry, TissueSample
from sequences.prep import SaturationModule
from sequences.tof import TofSequence
from skin import Vessel
from tissues import sample_tissues

FIELDS = ("t1_ms", "t2_ms", "t2star_ms", "pd_rel", "fatsat_s", "mtc_s")
FATSAT = (SaturationModule("fatsat_s", 8.0),)
TR_MS, FLIP_DEG, SLAB_MM = 35.0, 20.0, 12.0
TRANSITS_S = np.geomspace(0.05, 20.0, 30)
RATIOS = np.geomspace(0.15, 6.0, 120)
PSF_VOXELS = 1.2
N_DRAWS_HERE = 96


def sequence(fat_sat: bool) -> TofSequence:
    """Return a spoiled-GRE TOF readout with or without a per-TR fat-saturation pulse."""
    timing = fl3d1r_timing(0.3, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, TR_MS, PRISMA_XR)
    return TofSequence(timing=timing, flip_deg=FLIP_DEG, modules=FATSAT if fat_sat else ())


def contrast_curve(blood: TissueSample, background: TissueSample, fat_sat: bool) -> np.ndarray:
    """Return signed slab-mean blood minus background against slab transit time."""
    seq = sequence(fat_sat)
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
    g = seq.static_signal(background)
    out = []
    for transit in TRANSITS_S:
        flow = FlowGeometry(vessel, slab, SLAB_MM / transit, N_SHELLS, SLAB_MM * 1.001)
        blood_signal = np.mean(seq.blood_signal(blood, flow, np.linspace(0.0, flow.path_mm(), 32)), axis=0)
        out.append(blood_signal - g)
    return np.array(out)


def threshold_s(blood: TissueSample, background: TissueSample, fat_sat: bool = False) -> float:
    """Return the transit time at which slab-mean blood equals the background, 0 if it never falls below."""
    seq = sequence(fat_sat)
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
    g = float(seq.static_signal(background))

    def diff(transit: float) -> float:
        flow = FlowGeometry(vessel, slab, SLAB_MM / transit, N_SHELLS, SLAB_MM * 1.001)
        return float(np.mean(seq.blood_signal(blood, flow, np.linspace(0.0, flow.path_mm(), 32)))) - g

    lo, hi = 1e-3, 200.0
    if diff(hi) >= 0:
        return math.inf
    if diff(lo) <= 0:
        return 0.0
    for _ in range(36):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if diff(mid) < 0 else (mid, hi)
    return math.sqrt(lo * hi)


def samples() -> list[tuple[TissueSample, TissueSample]]:
    """Return paired blood and hypodermis samples from one Latin-hypercube draw."""
    tissues, _ = sample_tissues(["arterial_blood", "hypodermis"], N_DRAWS_HERE, 20261014)
    pairs = []
    for i in range(N_DRAWS_HERE):
        pairs.append(tuple(
            TissueSample(key, **{f: float(np.atleast_1d(getattr(tissues[key], f))[i]) for f in FIELDS})
            for key in ("arterial_blood", "hypodermis")))
    return pairs


def panel_a(ax, pairs) -> None:
    """Draw signed contrast against slab transit time, with and without fat suppression."""
    for fat_sat, colour, label in ((False, SERIES[3], "no fat suppression"), (True, SERIES[0], "fat suppressed")):
        curves = np.stack([contrast_curve(b, g, fat_sat) for b, g in pairs], axis=-1)
        band(ax, TRANSITS_S, curves, colour, label)
    ax.axhline(0.0, color=INK, lw=1.0)
    thresholds = np.array([threshold_s(b, g, False) for b, g in pairs])
    finite = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
    lo, mid, hi = np.percentile(finite, [5, 50, 95])
    ax.axvspan(lo, hi, color=SERIES[3], alpha=0.10, lw=0)
    ax.axvline(mid, color=SERIES[3], lw=1.3)
    ax.annotate(f"blood darker than fat\nbeyond {mid:.2f} s [{lo:.2f}-{hi:.2f}]",
                (mid, ax.get_ylim()[0]), xytext=(5, 8), textcoords="offset points",
                fontsize=7.5, color=SERIES[3], ha="left", va="bottom")
    for slab in (8.0, 20.0):
        ax.annotate(f"{slab:g} mm slab: {slab / mid:.0f} mm/s", (mid, 0), xytext=(5, -10 - 12 * (slab > 10)),
                    textcoords="offset points", fontsize=7, color=INK_2, ha="left", va="top")
    ax.set(xscale="log", xlim=(TRANSITS_S[0], TRANSITS_S[-1]))
    ax.set_xlabel("time for blood to cross the excited slab (s)", color=INK_2)
    ax.set_ylabel("slab-mean blood minus background (dermis M0)", color=INK_2)
    ax.set_title(f"a  A transit-time threshold, not a velocity one (TR {TR_MS:g} ms, {FLIP_DEG:g} deg)",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper right")
    style(ax)
    n = len(thresholds)
    print(f"  no fat suppression: {len(finite)}/{n} draws have a finite threshold "
          f"({np.sum(np.isinf(thresholds))} never invert, {np.sum(thresholds == 0)} always dark); "
          f"threshold {mid:.2f} s [{lo:.2f}-{hi:.2f}]")
    with_fat = np.array([threshold_s(b, g, True) for b, g in pairs])
    inverting = np.sum(np.isfinite(with_fat) & (with_fat > 0)) + np.sum(with_fat == 0)
    print(f"  fat suppressed: {inverting}/{n} draws are ever darker than the background at any transit time")


def panel_b(ax) -> None:
    """Draw the detection margin and apparent-width fidelity against voxel / diameter."""
    margin = np.array([r**2 * blood_fraction(1.0, r) for r in RATIOS])
    margin /= margin.max()
    fidelity = np.array([apparent_fwhm(1.0, r, PSF_VOXELS) for r in RATIOS])
    ax.plot(RATIOS, margin, color=SERIES[0], lw=2.2, label="detection margin at fixed scan time")
    ax.plot(RATIOS, fidelity, color=SERIES[1], lw=2.2, label="apparent width / true width")
    ax.axhline(1.0, color=INK_2, lw=0.8, ls=":")
    detect_from = RATIOS[np.argmax(margin >= 0.9)]
    measure_below = RATIOS[::-1][np.argmax(fidelity[::-1] <= 1.1)]
    ax.axvspan(detect_from, RATIOS[-1], color=SERIES[0], alpha=0.08, lw=0)
    ax.axvspan(RATIOS[0], measure_below, color=SERIES[1], alpha=0.08, lw=0)
    ax.annotate(f"detect: voxel > {detect_from:.1f} d", (RATIOS[-1], 0.45), xytext=(-6, 0),
                textcoords="offset points", fontsize=7.5, color=SERIES[0], ha="right", va="center")
    ax.annotate(f"measure: voxel < {measure_below:.2f} d", (RATIOS[0], 1.9), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=SERIES[1], ha="left", va="center")
    ax.set(xscale="log", xlim=(RATIOS[0], RATIOS[-1]), ylim=(0, 2.6))
    ax.set_xlabel("voxel size / vessel diameter", color=INK_2)
    ax.set_ylabel("normalised", color=INK_2)
    ax.set_title("b  Detect, or measure: the two regimes do not overlap", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper center")
    style(ax)
    print(f"  detection within 10 % of best for voxel > {detect_from:.2f} d; width within 10 % of true for "
          f"voxel < {measure_below:.2f} d; gap {detect_from / measure_below:.1f}x")


def main() -> None:
    """Build both panels and report the numbers."""
    pairs = samples()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    print("a  transit-time threshold")
    panel_a(axes[0], pairs)
    print("b  voxel size")
    panel_b(axes[1])
    fig.suptitle("What decides whether a cutaneous vessel is seen, and whether its width can be believed",
                 fontsize=11, color=INK)
    save(fig, "fig_tradeoffs", n=N_DRAWS_HERE,
         note=f"Literature parameters only, no measured data. Panel a: LHS seed 20261014, {N_DRAWS_HERE} draws, "
              "band 5-95 %. Panel b: geometry and noise statistics alone.")


if __name__ == "__main__":
    main()
