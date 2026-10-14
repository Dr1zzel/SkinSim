"""Required SNR against true vessel diameter over the protocols, and apparent width against true width (Q1, Q2).

Left: the SNR each protocol needs to detect a vessel along the leg in fat-
suppressed hypodermis (Rose CNR 4 on the peak voxel), expressed as the blood
SNR the 13 Oct 0.35 mm acquisition would have to show for that protocol to
succeed: one unknown absolute sensitivity scales every protocol alike, so this
ranks protocols without it, and it is the number a later measurement can
supply (detectability.snr_available). Reference signal = blood; voxel volume,
scan time and GRAPPA g-factor convert between protocols. Solid: no texture;
dashed: background texture of STRUCTURE_FRACTION of the background signal,
an illustrative value that is not set from the images.
Right: apparent FWHM against true diameter for each in-plane voxel size.
Run: python figures/fig_detectability.py
"""
from __future__ import annotations

import numpy as np
from common import (HYPODERMIS_DEPTH_MM, INK, INK_2, SERIES, along_leg, band, draw, drawn_parameters, plt,
                    report_dominance, save, style, summary, vessel_velocity)

from detectability import apparent_fwhm, blood_fraction, snr_required
from noise import thermal_snr
from protocols import ALL_PROTOCOLS, CANDIDATES, MEASURED, Protocol

REFERENCE = MEASURED["13oct_0.35mm"]
SHOWN = ["inplane_0.18mm", "13oct_0.26mm", "13oct_0.35mm", "13oct_0.46mm", "no_acceleration", "fatsat_off"]
DIAMETERS_MM = np.geomspace(0.08, 1.5, 32)
STRUCTURE_FRACTION = 0.2  # illustrative, unjustified; the texture regime, not a measured level
INPLANE_MM = sorted({p.inplane_mm for p in ALL_PROTOCOLS.values()})
PSF_VOXELS = 1.2  # FWHM of the fully sampled Fourier PSF in voxels


def sensitivity(protocol: Protocol, tissues, others) -> dict[str, np.ndarray]:
    """Return blood and background signal and the relative blood SNR (signal x V sqrt(T) / g) over the draws."""
    seq = protocol.sequence(others["sat_band_pulse_residual"])
    flow = along_leg(protocol, HYPODERMIS_DEPTH_MM, vessel_velocity(others))
    blood = seq.blood_signal(tissues["arterial_blood"], flow, np.linspace(0, flow.path_mm(), 40)).mean(axis=0)
    background = seq.static_signal(tissues["hypodermis"], 0.5)
    g = others["grappa_r2_g_factor"] if protocol.grappa_r == 2 else 1.0
    return {"blood": blood, "background": background,
            "snr_scale": blood * thermal_snr(protocol.voxel_volume_mm3, protocol.scan_time_s, g)}


def required_at_reference(protocol: Protocol, s: dict, ref: dict, texture: float) -> np.ndarray:
    """Return, per diameter and draw, the reference acquisition's blood SNR this protocol needs."""
    contrast = (s["blood"] - s["background"]) / s["blood"]
    structure = texture * s["background"] / s["blood"]
    need = np.stack([snr_required(contrast, d, protocol.inplane_mm, structure) for d in DIAMETERS_MM])
    return need * ref["snr_scale"] / s["snr_scale"]


def left_panel(ax, tissues, others) -> dict[str, np.ndarray]:
    """Draw required reference SNR against diameter for the shown protocols; return the no-texture curves."""
    ref = sensitivity(REFERENCE, tissues, others)
    curves = {}
    for name, colour in zip(SHOWN, SERIES):
        protocol = ALL_PROTOCOLS[name]
        s = sensitivity(protocol, tissues, others)
        curves[name] = required_at_reference(protocol, s, ref, 0.0)
        band(ax, DIAMETERS_MM, curves[name], colour, f"{name} ({protocol.inplane_mm:g} mm, {protocol.scan_time_s:.0f} s)")
        with np.errstate(invalid="ignore"):  # texture can make a draw's requirement infinite
            textured = np.percentile(required_at_reference(protocol, s, ref, STRUCTURE_FRACTION), 50, axis=-1)
        ax.plot(DIAMETERS_MM, textured, color=colour, lw=1, ls="--")
    ax.plot([], [], color=INK_2, lw=1, ls="--", label=f"dashed: texture {STRUCTURE_FRACTION:g} x background (illustrative)")
    ax.set(xscale="log", yscale="log", ylim=(1, 1e4))
    ax.set_xlabel("True vessel diameter (mm)", color=INK_2)
    ax.set_ylabel("Blood SNR needed at the 13 Oct 0.35 mm operating point", color=INK_2)
    ax.set_title("Vessel along the leg in fat-suppressed hypodermis, Rose CNR 4", fontsize=9, color=INK, loc="left")
    return curves


def right_panel(ax) -> None:
    """Draw apparent FWHM against true diameter for each in-plane voxel size."""
    for a, colour in zip(INPLANE_MM, SERIES):
        ax.plot(DIAMETERS_MM, [apparent_fwhm(d, a, PSF_VOXELS) for d in DIAMETERS_MM], color=colour, lw=2,
                label=f"{a:g} mm voxel")
    ax.plot(DIAMETERS_MM, DIAMETERS_MM, color=INK_2, lw=1, ls=":", label="apparent = true")
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("True vessel diameter (mm)", color=INK_2)
    ax.set_ylabel("Apparent FWHM (mm)", color=INK_2)
    ax.set_title(f"Apparent width, Gaussian PSF of {PSF_VOXELS:g} voxels FWHM", fontsize=9, color=INK, loc="left")


def report_resolution() -> None:
    """Print Q2 per voxel size: where partial volume ends and where apparent width starts reporting true width."""
    ratios = np.geomspace(0.1, 6.0, 240)  # d / a: kappa and FWHM / a depend on nothing else
    peak = np.array([blood_fraction(r, 1.0) for r in ratios])
    fwhm = np.array([apparent_fwhm(r, 1.0, PSF_VOXELS) for r in ratios])
    peak_09 = ratios[np.argmax(peak >= 0.9)]
    within = ratios[np.argmax(np.abs(fwhm / ratios - 1) <= 0.1)]
    for a in INPLANE_MM:
        print(f"  {a:g} mm: area ratio reaches 1 at d = {2 * a / np.sqrt(np.pi):.2f} mm; mean peak-voxel fraction "
              f">= 0.9 from d = {peak_09 * a:.2f} mm; apparent FWHM within 10 % of d from d = {within * a:.2f} mm "
              f"(below about {PSF_VOXELS * a:.2f} mm it reports the PSF)")


def main() -> None:
    """Draw, compute, plot and print."""
    tissues, others = draw()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    curves = left_panel(left, tissues, others)
    right_panel(right)
    for ax in (left, right):
        style(ax)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2)
    print("Q1: blood SNR needed at the 13 Oct 0.35 mm operating point, median [5-95 %], no texture")
    for name, c in curves.items():
        cells = [f"d {d:g} mm {summary(c[np.argmin(np.abs(DIAMETERS_MM - d))])}" for d in (0.3, 0.5, 1.0)]
        print(f"  {name}: " + "; ".join(cells))
    report_dominance("the 0.3 mm requirement, 0.35 mm protocol",
                     curves["13oct_0.35mm"][np.argmin(np.abs(DIAMETERS_MM - 0.3))], drawn_parameters(tissues, others))
    print("Q2: resolution")
    report_resolution()
    fig.suptitle("fl3d1r TOF: what it takes to detect a vessel, and what width it appears to have", fontsize=11, color=INK)
    save(fig, "fig_detectability")


if __name__ == "__main__":
    main()
