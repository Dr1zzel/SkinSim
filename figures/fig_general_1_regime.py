"""General figure 1: the inflow regime, and where cutaneous vessels sit in it.

Nothing here is tied to one protocol. Inflow enhancement in a spoiled-GRE slab
is governed by a single dimensionless group: the transit time of blood through
the excited slab divided by the saturation time constant

    T_sat = -TR / ln(E1 cos alpha),

the 1/e time of the approach to the static steady state. Panel (a) shows that
the slab-mean blood signal collapses onto one curve in that group across a wide
range of TR and flip angle, and marks where cerebral and cutaneous acquisitions
fall. Panel (b) maps the same quantity over velocity and slab thickness, the
two variables the operator actually sets.

Run: python figures/fig_general_1_regime.py
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from common import INK, INK_2, SERIES, GRID, N_SHELLS, plt, save, style

from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from sequences.base import FlowGeometry, TissueSample
from sequences.tof import TofSequence
from skin import Vessel
from acquisition import Slab
from tissues import ARTERIAL_BLOOD

BLOOD = TissueSample.nominal(ARTERIAL_BLOOD)
INPLANE_MM = 0.3
N_PATH = 60

# Operating regions, as ranges of (velocity mm/s, slab thickness mm, TR ms, flip deg).
REGIONS = {
    "intracranial TOF": ((200.0, 600.0), (40.0, 70.0), (20.0, 30.0), (15.0, 25.0)),
    "peripheral run-off TOF": ((80.0, 250.0), (40.0, 70.0), (20.0, 35.0), (25.0, 50.0)),
    "cutaneous vessels": ((3.0, 60.0), (8.0, 20.0), (25.0, 50.0), (10.0, 40.0)),
}


def timing(tr_ms: float):
    """Return a spoiled-GRE readout of the given TR with lobes set by the gradient limits."""
    return fl3d1r_timing(INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, tr_ms, PRISMA_XR)


def t_sat_ms(tr_ms: float, flip_deg: float, t1_ms: float = BLOOD.t1_ms) -> float:
    """Return the 1/e saturation time constant, -TR / ln(E1 cos alpha)."""
    q = math.exp(-tr_ms / t1_ms) * math.cos(math.radians(flip_deg))
    return math.inf if q >= 1 else -tr_ms / math.log(q)


def retained(tr_ms: float, flip_deg: float, slab_mm: float, velocity_mm_s: float) -> float:
    """Return slab-mean blood signal divided by the signal of fully relaxed blood, for a vessel crossing the slab."""
    seq = TofSequence(timing=timing(tr_ms), flip_deg=flip_deg)
    slab = Slab(thickness_mm=slab_mm, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
    flow = FlowGeometry(vessel, slab, velocity_mm_s, N_SHELLS, slab_mm * 1.001)
    path = np.linspace(0.0, flow.path_mm(), N_PATH)
    mean = float(np.mean(seq.blood_signal(BLOOD, flow, path)))
    fresh = float(seq.blood_signal(BLOOD, flow, [0.0])[0])
    return mean / fresh


def region_group(name: str) -> tuple[float, float]:
    """Return the min and max of transit / T_sat over a region's stated ranges."""
    (vlo, vhi), (llo, lhi), (trlo, trhi), (flo, fhi) = REGIONS[name]
    taus = [1000.0 * L / v for v in (vlo, vhi) for L in (llo, lhi)]
    sats = [t_sat_ms(tr, f) for tr in (trlo, trhi) for f in (flo, fhi)]
    ratios = [t / s for t in taus for s in sats]
    return min(ratios), max(ratios)


def panel_a(ax) -> None:
    """Draw the collapse of slab-mean retention onto transit / T_sat across TR and flip."""
    combos = [(25.0, 12.0), (25.0, 30.0), (45.0, 18.0), (45.0, 40.0), (70.0, 25.0)]
    velocities = np.geomspace(1.0, 500.0, 36)
    for (tr, flip), colour in zip(combos, SERIES):
        sat = t_sat_ms(tr, flip)
        xs, ys = [], []
        for slab in (10.0, 20.0, 50.0):
            for v in velocities:
                xs.append(1000.0 * slab / v / sat)
                ys.append(retained(tr, flip, slab, v))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=colour, lw=1.6, alpha=0.9,
                label=f"TR {tr:g} ms, {flip:g} deg (T_sat {sat:.0f} ms)")
    for (name, colour), y in zip(zip(REGIONS, (SERIES[4], SERIES[5], INK)), (0.93, 0.85, 0.77)):
        lo, hi = region_group(name)
        ax.plot([lo, hi], [y, y], color=colour, lw=5, alpha=0.35, solid_capstyle="butt")
        ax.annotate(name, (math.sqrt(lo * hi), y), xytext=(0, 5), textcoords="offset points",
                    fontsize=7.5, color=colour, ha="center", va="bottom")
    ax.set(xscale="log", xlim=(0.02, 200), ylim=(0, 1.02))
    ax.set_xlabel("slab transit time / saturation time constant", color=INK_2)
    ax.set_ylabel("slab-mean blood signal / fully relaxed blood", color=INK_2)
    ax.set_title("a  One dimensionless group governs inflow", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="lower left")
    style(ax)


def panel_b(ax) -> None:
    """Map slab-mean retention over velocity and slab thickness at a mid-range TR and flip."""
    tr, flip = 35.0, 20.0
    velocities = np.geomspace(2.0, 400.0, 44)
    slabs = np.linspace(4.0, 70.0, 34)
    grid = np.array([[retained(tr, flip, L, v) for v in velocities] for L in slabs])
    levels = np.linspace(0.1, 0.9, 9)
    cs = ax.contourf(velocities, slabs, grid, levels=np.linspace(0, 1, 21), cmap="Blues_r", alpha=0.9)
    ax.contour(velocities, slabs, grid, levels=levels, colors=[INK_2], linewidths=0.5, alpha=0.5)
    half = ax.contour(velocities, slabs, grid, levels=[0.5], colors=[INK], linewidths=1.8)
    ax.clabel(half, fmt={0.5: "50 %"}, fontsize=7.5, inline_spacing=12)
    for name, colour in zip(REGIONS, (SERIES[4], SERIES[5], SERIES[1])):
        (vlo, vhi), (llo, lhi), _, _ = REGIONS[name]
        ax.add_patch(plt.Rectangle((vlo, llo), vhi - vlo, lhi - llo, fill=False, ec=colour, lw=1.6))
        ax.annotate(name, (math.sqrt(vlo * vhi), 0.5 * (llo + lhi)), fontsize=7.5, color=colour,
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    ax.set(xscale="log", xlim=(velocities[0], velocities[-1]), ylim=(slabs[0], slabs[-1]))
    ax.set_xlabel("mean blood velocity (mm/s)", color=INK_2)
    ax.set_ylabel("slab thickness (mm)", color=INK_2)
    ax.set_title(f"b  The same quantity over what the operator sets (TR {tr:g} ms, {flip:g} deg)",
                 fontsize=9.5, color=INK, loc="left")
    plt.colorbar(cs, ax=ax, fraction=0.045, label="slab-mean blood / relaxed blood")
    ax.grid(color=GRID, lw=0.4, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def main() -> None:
    """Build both panels and report the regime boundaries."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    panel_a(axes[0])
    panel_b(axes[1])
    print("transit / T_sat by regime (min-max over the stated velocity, slab, TR and flip ranges):")
    for name in REGIONS:
        lo, hi = region_group(name)
        print(f"  {name:24s} {lo:7.2f} - {hi:7.2f}")
    print("\nslab-mean blood / relaxed blood at TR 35 ms, 20 deg:")
    for name, (v, L) in (("intracranial", (350.0, 55.0)), ("cutaneous, thick slab", (20.0, 20.0)),
                         ("cutaneous, thin slab", (20.0, 8.0)), ("cutaneous, slow", (5.0, 12.0))):
        print(f"  {name:24s} v {v:5.0f} mm/s, slab {L:4.0f} mm -> {retained(35.0, 20.0, L, v):.2f}")
    fig.suptitle("Inflow enhancement in a spoiled-GRE slab: one governing group, and where skin falls",
                 fontsize=11, color=INK)
    save(fig, "fig_general_1_regime",
         note="Nominal literature parameters, no measured data and no parameter draw: "
              "this figure shows the structure of the design space, not its uncertainty.")


if __name__ == "__main__":
    main()
