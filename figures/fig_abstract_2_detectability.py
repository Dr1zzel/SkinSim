"""Abstract figure 2: detection is sensitivity-limited, and measured width reports the PSF (Q1, Q2).

Left: the blood SNR each voxel size needs to reach Rose CNR 4 on the peak voxel
for a vessel along the leg in fat-suppressed hypodermis, expressed at one common
operating point so that an unknown absolute sensitivity cancels. Turning fat
saturation off is shown for scale: it is a contrast lever an order of magnitude
larger than any voxel-size choice.
Right: apparent FWHM against true diameter. Below the point-spread function
every vessel measures the same width, so a line profile there reports the
protocol rather than the anatomy.

Run: python figures/fig_abstract_2_detectability.py
"""
from __future__ import annotations

import numpy as np
from common import (HYPODERMIS_DEPTH_MM, INK, INK_2, SERIES, along_leg, band, draw, plt, save, style, summary,
                    vessel_velocity)

from detectability import apparent_fwhm, blood_fraction, snr_required
from noise import thermal_snr
from protocols import ALL_PROTOCOLS, MEASURED, Protocol

REFERENCE = MEASURED["13oct_0.26mm"]
VOXEL_FAMILY = ["inplane_0.18mm", "13oct_0.26mm", "13oct_0.35mm", "13oct_0.46mm"]
LEVERS = ["no_acceleration", "fatsat_off"]
DIAMETERS_MM = np.geomspace(0.1, 1.5, 32)
PSF_VOXELS = 1.2


def sensitivity(protocol: Protocol, tissues, others) -> dict[str, np.ndarray]:
    """Return blood and background signal and the relative blood SNR (signal x V sqrt(T) / g) over the draws."""
    seq = protocol.sequence(others["sat_band_pulse_residual"])
    flow = along_leg(protocol, HYPODERMIS_DEPTH_MM, vessel_velocity(others))
    blood = seq.blood_signal(tissues["arterial_blood"], flow, np.linspace(0, flow.path_mm(), 40)).mean(axis=0)
    background = seq.static_signal(tissues["hypodermis"], 0.5)
    g = others["grappa_r2_g_factor"] if protocol.grappa_r == 2 else 1.0
    return {"blood": blood, "background": background,
            "snr_scale": blood * thermal_snr(protocol.voxel_volume_mm3, protocol.scan_time_s, g)}


def required(protocol: Protocol, s: dict, ref: dict) -> np.ndarray:
    """Return, per diameter and draw, the reference acquisition's blood SNR this protocol needs."""
    contrast = (s["blood"] - s["background"]) / s["blood"]
    need = np.stack([snr_required(contrast, d, protocol.inplane_mm) for d in DIAMETERS_MM])
    return need * ref["snr_scale"] / s["snr_scale"]


def main() -> None:
    """Draw, compute, plot and print."""
    tissues, others = draw()
    ref = sensitivity(REFERENCE, tissues, others)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)

    print("Blood SNR needed at the 0.26 mm operating point, median [5-95 %]")
    for name, colour in zip(VOXEL_FAMILY, SERIES):
        protocol = ALL_PROTOCOLS[name]
        curve = required(protocol, sensitivity(protocol, tissues, others), ref)
        band(left, DIAMETERS_MM, curve, colour, f"{protocol.inplane_mm:g} mm, {protocol.scan_time_s:.0f} s")
        cells = [f"d {d:g} mm {summary(curve[np.argmin(np.abs(DIAMETERS_MM - d))])}" for d in (0.2, 0.3, 0.5)]
        print(f"  {name}: " + "; ".join(cells))
    for name, style_ in zip(LEVERS, ("-.", "--")):
        protocol = ALL_PROTOCOLS[name]
        curve = required(protocol, sensitivity(protocol, tissues, others), ref)
        left.plot(DIAMETERS_MM, np.median(curve, axis=-1), color=INK_2, lw=1.3, ls=style_,
                  label=f"{name.replace('_', ' ')} (at {protocol.inplane_mm:g} mm)")
        cells = [f"d {d:g} mm {summary(curve[np.argmin(np.abs(DIAMETERS_MM - d))])}" for d in (0.2, 0.3, 0.5)]
        print(f"  {name}: " + "; ".join(cells))

    left.set(xscale="log", yscale="log", ylim=(1, 1e3))
    left.set_xlabel("True vessel diameter (mm)", color=INK_2)
    left.set_ylabel("Blood SNR needed (at the 0.26 mm operating point)", color=INK_2)
    left.set_title("a  Detection threshold, Rose CNR 4", fontsize=9.5, color=INK, loc="left")
    left.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper right")

    inplane = sorted({ALL_PROTOCOLS[n].inplane_mm for n in VOXEL_FAMILY})
    for a, colour in zip(inplane, SERIES):
        right.plot(DIAMETERS_MM, [apparent_fwhm(d, a, PSF_VOXELS) for d in DIAMETERS_MM], color=colour, lw=2,
                   label=f"{a:g} mm voxel")
    right.plot(DIAMETERS_MM, DIAMETERS_MM, color=INK_2, lw=1, ls=":", label="apparent = true")
    right.set(xscale="log", yscale="log")
    right.set_xlabel("True vessel diameter (mm)", color=INK_2)
    right.set_ylabel("Apparent FWHM (mm)", color=INK_2)
    right.set_title("b  Measured width below the PSF reports the protocol", fontsize=9.5, color=INK, loc="left")
    right.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left")

    for ax in (left, right):
        style(ax)

    ratios = np.geomspace(0.1, 6.0, 240)
    peak = np.array([blood_fraction(r, 1.0) for r in ratios])
    fwhm = np.array([apparent_fwhm(r, 1.0, PSF_VOXELS) for r in ratios])
    peak_09 = ratios[np.argmax(peak >= 0.9)]
    within = ratios[np.argmax(np.abs(fwhm / ratios - 1) <= 0.1)]
    print("Resolution")
    for a in inplane:
        print(f"  {a:g} mm: peak-voxel fraction >= 0.9 from d = {peak_09 * a:.2f} mm; apparent FWHM within 10 % "
              f"of d from d = {within * a:.2f} mm (below about {PSF_VOXELS * a:.2f} mm it reports the PSF)")

    fig.suptitle("fl3d1r TOF: what it takes to see a cutaneous vessel, and what width it appears to have",
                 fontsize=11, color=INK)
    save(fig, "fig_abstract_2_detectability")


if __name__ == "__main__":
    main()
