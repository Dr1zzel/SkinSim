"""Abstract Figure 5 (simulation part): what resolution costs, and what width can be trusted.

(a) Sensitivity each candidate voxel needs to detect a vessel of a given true
    diameter, every protocol expressed in units of the sensitivity it delivers
    itself (V sqrt(T) / g, detect.thermal_snr). Lower is easier, and where two
    curves cross the better choice of voxel changes. The contrast is simulated
    rather than assumed, and the background texture is swept over its interval,
    so each curve carries a band and each voxel has a diameter below which no
    sensitivity suffices at all.
(b) Apparent width against true width: below the point-spread function every
    vessel measures the same, so finer voxels buy measurement, not detection.

No measured quantity appears here. The vertical axis is relative: its absolute
scale needs the coil's sensitivity at depth, which the toolbox deliberately
leaves unknown, so what the panel supports is the comparison BETWEEN curves.
"""
from __future__ import annotations

import numpy as np
from common import (HYPODERMIS_DEPTH_MM, INK, INK_2, MAX_PATH_MM, N_SHELLS, SERIES, band, draw,
                    percentiles, plt, save, style, summary)
from detect import apparent_fwhm, snr_required, thermal_snr
from geometry import Vessel
from predict import predict
from protocols import ALL_PROTOCOLS, REFERENCE_PROTOCOL, RESOLUTION_SWEEP
from sequence import FlowGeometry

D = np.geomspace(0.08, 1.5, 160)
PSF = 1.2  # point-spread FWHM in voxels, stated not derived (detect.apparent_fwhm)
VESSEL_AZIMUTH_DEG = 60.0  # in-plane vessel, obliquely crossing a transversal slab

tissues, others = draw()
velocity = others["target_vessel_velocity"] * others["thermoregulatory_factor"]
structure_rel = others["background_structure_rel"]
reference_sensitivity = thermal_snr(REFERENCE_PROTOCOL.voxel_volume_mm3,
                                    REFERENCE_PROTOCOL.scan_time_s,
                                    others["grappa_r2_g_factor"])


def best_position(protocol):
    """Return the contrast and background texture at each draw's best position along the vessel."""
    vessel = Vessel(depth_mm=HYPODERMIS_DEPTH_MM, polar_deg=90.0,
                    azimuth_deg=VESSEL_AZIMUTH_DEG, diameter_mm=1.0)  # diameter enters later, per sweep point
    flow = FlowGeometry(vessel, protocol.slab(), velocity, N_SHELLS, MAX_PATH_MM, entry_z_frac=0.5)
    pred = predict(protocol, tissues["arterial_blood"], tissues["hypodermis"], flow,
                   np.linspace(0.0, flow.path_mm(), 24),
                   background_structure_rel=structure_rel,
                   band_pulse_residual=others["sat_band_pulse_residual"])
    peak = np.argmax(np.abs(pred.contrast), axis=0)[None, :]
    contrast = np.take_along_axis(pred.contrast, peak, axis=0)[0]
    texture = np.take_along_axis(np.broadcast_to(pred.structure_fraction, pred.contrast.shape),
                                 peak, axis=0)[0]
    return pred, contrast, texture


def relative_cost(protocol):
    """Return required sensitivity against diameter, in units of what the reference protocol delivers."""
    # Matched scan time, not each protocol's own: a coarser voxel also runs
    # shorter, and crediting it with that too would conflate resolution with
    # duration and make coarse win for the wrong reason.
    pred, contrast, texture = best_position(protocol)
    delivered = thermal_snr(protocol.voxel_volume_mm3, REFERENCE_PROTOCOL.scan_time_s,
                            others["grappa_r2_g_factor"]) / reference_sensitivity
    need = np.array([snr_required(contrast, d, pred.voxel_mm, texture) for d in D])
    return need / delivered, pred


fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

A = ax[0]
costs = {label: relative_cost(ALL_PROTOCOLS[name]) for name, label in RESOLUTION_SWEEP.items()}
floors = {}
for (label, (cost, pred)), colour in zip(costs.items(), SERIES):
    band(A, D, cost, colour, f"{label}  ({pred.voxel_mm[0]:.2f} x {pred.voxel_mm[1]:.2f} mm seen)")
    reachable = np.isfinite(percentiles(cost)[1])
    floors[label] = D[np.argmax(reachable)] if reachable.any() else np.inf
    if floors[label] > D[0]:  # a floor inside the sweep is worth drawing; below it, there is none
        A.axvline(floors[label], color=colour, lw=1.0, ls=":", alpha=0.8)
A.set(xscale="log", yscale="log", xlim=(D[0], D[-1]))
A.set_xlabel("true vessel diameter (mm)", color=INK_2, fontsize=10)
A.set_ylabel("sensitivity needed, in units of what the\n"
             f"{REFERENCE_PROTOCOL.inplane_mm:.2f} mm protocol delivers", color=INK_2, fontsize=10)
A.set_title("a   At matched scan time, finer voxels cost sensitivity", fontsize=11, color=INK,
            loc="left")
A.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right", title="in-plane voxel",
         title_fontsize=8)
style(A)

B = ax[1]
for (name, label), colour in zip(RESOLUTION_SWEEP.items(), SERIES):
    a = ALL_PROTOCOLS[name].inplane_mm
    width = np.array([apparent_fwhm(d, a, PSF) for d in D])
    B.plot(D, width, color=colour, lw=2.2, label=f"{label}")
    truthful = D[np.argmax(np.abs(width / D - 1) <= 0.1)]
    B.plot([truthful], [apparent_fwhm(truthful, a, PSF)], marker="o", ms=7, color=colour, zorder=5)
B.plot(D, D, color=INK_2, lw=1.1, ls=":", label="apparent = true")
B.set(xscale="log", yscale="log", xlim=(D[0], D[-1]))
B.set_xlabel("true vessel diameter (mm)", color=INK_2, fontsize=10)
B.set_ylabel("apparent FWHM (mm)", color=INK_2, fontsize=10)
B.set_title("b   Below the PSF, every vessel measures the same width",
            fontsize=11, color=INK, loc="left")
B.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
style(B)

print(f"Matched scan time {REFERENCE_PROTOCOL.scan_time_s:.0f} s; sensitivity in units of the "
      f"{REFERENCE_PROTOCOL.inplane_mm:.2f} mm protocol's own")
for label, (cost, _) in costs.items():
    at_300 = cost[np.argmin(np.abs(D - 0.3))]
    floor = f"{floors[label]:.3f} mm" if floors[label] > D[0] else f"below {D[0]:.2f} mm"
    print(f"  {label:<22} at 0.30 mm: {summary(at_300):<34} texture floor {floor}")
print("\n  marker in (b): diameter above which the apparent width is within 10 % of the true one")

fig.suptitle("Figure 5 — what finer voxels cost, and what width they can report",
             fontsize=12.5, color=INK)
save(fig, "fig5_detectability")
