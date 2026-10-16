"""Abstract Figure 1: why cutaneous TOF is a different problem.

(a) Inflow collapses onto one dimensionless group, the slab transit time over the
    saturation time constant; intracranial TOF sits where inflow is barely spent,
    cutaneous acquisitions an order of magnitude further in.
(b) Against unsuppressed subcutaneous fat there is a transit-time threshold beyond
    which blood is the darker tissue. Fat suppression removes it.
"""
from __future__ import annotations
import math
import numpy as np
from common import INK, INK_2, SERIES, N_SHELLS, band, plt, save, style
from acquisition import Slab
from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from sequences.base import FlowGeometry, TissueSample
from sequences.prep import SaturationModule
from sequences.tof import TofSequence
from skin import Vessel
from tissues import ARTERIAL_BLOOD, sample_tissues

BLOOD = TissueSample.nominal(ARTERIAL_BLOOD)
F = ("t1_ms","t2_ms","t2star_ms","pd_rel","fatsat_s","mtc_s")
FATSAT = (SaturationModule("fatsat_s", 8.0),)
VES = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)

def timing(tr): return fl3d1r_timing(0.3, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, tr, PRISMA_XR)
def t_sat(tr, flip, t1=BLOOD.t1_ms):
    q = math.exp(-tr/t1)*math.cos(math.radians(flip));  return -tr/math.log(q)

def retained(tr, flip, slab, v):
    seq = TofSequence(timing=timing(tr), flip_deg=flip)
    flow = FlowGeometry(VES, Slab(thickness_mm=slab, orientation_deg=90.0), v, N_SHELLS, slab*1.001)
    p = np.linspace(0, flow.path_mm(), 40)
    return float(np.mean(seq.blood_signal(BLOOD, flow, p)))/float(seq.blood_signal(BLOOD, flow, [0.0])[0])

def panel_a(ax):
    for (tr, flip), c in zip([(25,12),(25,30),(45,18),(45,40),(70,25)], SERIES):
        s = t_sat(tr, flip); xs, ys = [], []
        for slab in (10., 20., 50.):
            for v in np.geomspace(1, 500, 30):
                xs.append(1000*slab/v/s); ys.append(retained(tr, flip, slab, v))
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], color=c, lw=1.7, alpha=.9,
                label=f"TR {tr} ms, {flip} deg")
    for (name, lo, hi), c, y in zip([("intracranial TOF", .12, 1.93), ("peripheral run-off", .55, 19.9),
                                     ("cutaneous vessels", .12, 75.1)],
                                    (SERIES[4], SERIES[5], INK), (.94, .86, .78)):
        ax.plot([lo, hi], [y, y], color=c, lw=6, alpha=.33, solid_capstyle="butt")
        ax.annotate(name, (math.sqrt(lo*hi), y), xytext=(0, 6), textcoords="offset points",
                    fontsize=9, color=c, ha="center", va="bottom", weight="bold")
    ax.set(xscale="log", xlim=(.03, 150), ylim=(0, 1.04))
    ax.set_xlabel("slab transit time / saturation time constant", color=INK_2, fontsize=10)
    ax.set_ylabel("slab-mean blood / fully relaxed blood", color=INK_2, fontsize=10)
    ax.set_title("a   One group governs inflow — and skin sits at its far end", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower left")
    style(ax)

def panel_b(ax):
    tis, _ = sample_tissues(["arterial_blood","hypodermis"], 96, 20261014)
    pairs = [tuple(TissueSample(k, **{f: float(np.atleast_1d(getattr(tis[k], f))[i]) for f in F})
                   for k in ("arterial_blood","hypodermis")) for i in range(96)]
    T = np.geomspace(.05, 20, 28); SLAB = 12.
    def curve(b, g, fs):
        seq = TofSequence(timing=timing(35.), flip_deg=20., modules=FATSAT if fs else ())
        gg = seq.static_signal(g); out = []
        for t in T:
            flow = FlowGeometry(VES, Slab(thickness_mm=SLAB, orientation_deg=90.), SLAB/t, N_SHELLS, SLAB*1.001)
            out.append(np.mean(seq.blood_signal(b, flow, np.linspace(0, flow.path_mm(), 28)), axis=0) - gg)
        return np.array(out)
    for fs, c, lab in ((False, SERIES[3], "no fat suppression"), (True, SERIES[0], "fat suppressed")):
        band(ax, T, np.stack([curve(b, g, fs) for b, g in pairs], -1), c, lab)
    ax.axhline(0, color=INK, lw=1.1)
    ax.axvspan(.27, 1.73, color=SERIES[3], alpha=.10, lw=0)
    ax.axvline(.73, color=SERIES[3], lw=1.4)
    ax.annotate("blood DARKER than fat\nbeyond 0.73 s [0.27–1.73]", (.73, ax.get_ylim()[0]),
                xytext=(7, 10), textcoords="offset points", fontsize=9, color=SERIES[3], va="bottom", weight="bold")
    ax.annotate("8 mm slab → 11 mm/s\n20 mm slab → 27 mm/s", (.73, 0), xytext=(7, -8),
                textcoords="offset points", fontsize=8, color=INK_2, va="top")
    ax.set(xscale="log", xlim=(T[0], T[-1]))
    ax.set_xlabel("time for blood to cross the excited slab (s)", color=INK_2, fontsize=10)
    ax.set_ylabel("blood − background (dermis M0)", color=INK_2, fontsize=10)
    ax.set_title("b   Fat suppression removes a transit-time threshold", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_2, loc="upper right")
    style(ax)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
panel_a(ax[0]); panel_b(ax[1])
fig.suptitle("Figure 1 — cutaneous time-of-flight is not intracranial time-of-flight at a smaller scale",
             fontsize=12.5, color=INK)
save(fig, "ABS_fig1_regime", n=96,
     note="Simulation only; literature tissue parameters, no measured data. "
          "Panel b: LHS seed 20261014, 96 draws, band 5–95 %.")
