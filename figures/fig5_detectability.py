"""Abstract Figure 5 (simulation part): detection is sensitivity-limited, and width reports the PSF.

(a) Peak-voxel SNR needed to reach Rose CNR 4, against true vessel diameter, for the
    achieved voxel and a finer one, with the measured and matched-time predicted
    sensitivity drawn across them.
(b) Apparent width against true width: below the point-spread function every vessel
    measures the same, so finer voxels buy measurement, not detection.
"""
from __future__ import annotations
import numpy as np
from common import INK, INK_2, SERIES, plt, save, style
from detectability import ROSE, apparent_fwhm, blood_fraction

D = np.geomspace(0.08, 1.5, 200)
ACHIEVED_A, FINE_A, PART = 0.245, 0.15, 0.22
SNR_MEASURED = 15.0                      # peak-voxel vessel SNR, 15 Oct, 0.245 mm
SNR_FINE = SNR_MEASURED * (FINE_A/ACHIEVED_A)**2     # matched scan time: SNR ~ voxel volume
PSF = 1.2

need = lambda a: np.array([ROSE/blood_fraction(d, a) for d in D])
def dmin(a, snr):
    r = need(a); ok = np.where(r <= snr)[0]
    return D[ok[0]] if ok.size else np.nan

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

A = ax[0]
for a, snr, c, lab in ((ACHIEVED_A, SNR_MEASURED, SERIES[0], f"{ACHIEVED_A} mm (achieved)"),
                       (FINE_A, SNR_FINE, SERIES[1], f"{FINE_A} mm (matched scan time)")):
    A.plot(D, need(a), color=c, lw=2.4, label=f"SNR needed, {lab}")
    A.axhline(snr, color=c, lw=1.4, ls="--")
    d0 = dmin(a, snr)
    A.plot([d0], [snr], marker="o", ms=9, color=c, zorder=5)
    A.annotate(f"{d0:.2f} mm", (d0, snr), xytext=(6, 8), textcoords="offset points",
               fontsize=10, color=c, weight="bold")
    A.annotate(f"sensitivity {'measured' if a==ACHIEVED_A else 'predicted'} = {snr:.0f}",
               (D[-1], snr), xytext=(-4, 5), textcoords="offset points",
               fontsize=8.5, color=c, ha="right")
A.annotate("", xy=(0.17, SNR_MEASURED), xytext=(0.17, SNR_FINE),
           arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
A.annotate(f"x{SNR_MEASURED/SNR_FINE:.1f} sensitivity\nneeded to hold the line", (0.175, np.sqrt(SNR_MEASURED*SNR_FINE)),
           fontsize=9, color=INK, va="center", weight="bold")
A.set(xscale="log", yscale="log", xlim=(D[0], D[-1]), ylim=(2, 400))
A.set_xlabel("true vessel diameter (mm)", color=INK_2, fontsize=10)
A.set_ylabel("peak-voxel SNR needed (Rose CNR 4)", color=INK_2, fontsize=10)
A.set_title("a   Finer voxels cost more sensitivity than they save", fontsize=11, color=INK, loc="left")
A.legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper right")
style(A)

B = ax[1]
for a, c in ((ACHIEVED_A, SERIES[0]), (FINE_A, SERIES[1])):
    B.plot(D, [apparent_fwhm(d, a, PSF) for d in D], color=c, lw=2.4, label=f"{a} mm voxel")
    w = D[np.argmax(np.abs(np.array([apparent_fwhm(d, a, PSF) for d in D])/D - 1) <= 0.1)]
    B.plot([w], [apparent_fwhm(w, a, PSF)], marker="o", ms=8, color=c, zorder=5)
    B.annotate(f"width truthful\nabove {w:.2f} mm", (w, apparent_fwhm(w, a, PSF)),
               xytext=(8, -14), textcoords="offset points", fontsize=9, color=c, weight="bold")
B.plot(D, D, color=INK_2, lw=1.1, ls=":", label="apparent = true")
B.set(xscale="log", yscale="log", xlim=(D[0], D[-1]))
B.set_xlabel("true vessel diameter (mm)", color=INK_2, fontsize=10)
B.set_ylabel("apparent FWHM (mm)", color=INK_2, fontsize=10)
B.set_title("b   Below the PSF, every vessel measures the same width", fontsize=11, color=INK, loc="left")
B.legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")
style(B)

print(f"achieved {ACHIEVED_A} mm, SNR {SNR_MEASURED:.0f} -> d_min {dmin(ACHIEVED_A, SNR_MEASURED):.3f} mm")
print(f"fine     {FINE_A} mm, SNR {SNR_FINE:.1f} -> d_min {dmin(FINE_A, SNR_FINE):.3f} mm")
print(f"sensitivity gain to make 0.15 mm match the achieved d_min: "
      f"x{need(FINE_A)[np.argmin(np.abs(D-dmin(ACHIEVED_A,SNR_MEASURED)))]/SNR_FINE:.1f}")
fig.suptitle("Figure 5 — what sensitivity the next step requires", fontsize=12.5, color=INK)
save(fig, "fig5_detectability", note="Simulation (geometry and noise statistics) with the measured 15 October "
     "operating point overlaid; the finer voxel assumes matched scan time.")
