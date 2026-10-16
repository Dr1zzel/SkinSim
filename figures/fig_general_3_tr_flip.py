"""General figure 3: the TR and flip angle that go together, for any transit time.

TR and flip angle are not independent settings. Both act on inflow only through
the saturation time constant

    T_sat = 1 / (1/T1 + |ln cos alpha| / TR),

so every (TR, flip) pair that gives the same T_sat gives the same contrast.
The CNR-optimal setting is therefore a RIDGE in the (TR, flip) plane, not a
point: pick TR for whatever else constrains it, then read the flip angle off
the ridge.

Panel (a) maps CNR per sqrt(scan time), relative to its own maximum, over the
plane for one transit time, with the ridge and the band within 10 % of best.
Panel (b) gives the ridge for a range of transit times, with the static-tissue
Ernst angle as the zero-flow limit.

Run: python figures/fig_general_3_tr_flip.py
"""
from __future__ import annotations

import math

import numpy as np
from common import GRID, INK, INK_2, SERIES, N_SHELLS, plt, save, style

from acquisition import Slab
from analysis import cnr_efficiency
from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from sequences.base import FlowGeometry, TissueSample
from sequences.prep import SaturationModule
from sequences.tof import TofSequence
from skin import Vessel
from tissues import ARTERIAL_BLOOD, HYPODERMIS

BLOOD, BACKGROUND = TissueSample.nominal(ARTERIAL_BLOOD), TissueSample.nominal(HYPODERMIS)
FATSAT = (SaturationModule("fatsat_s", 8.0),)
SLAB_MM = 12.0
TRS_MS = np.linspace(15.0, 80.0, 14)
FLIPS_DEG = np.arange(2.0, 71.0, 2.0)
TRANSITS_S = (0.1, 0.25, 0.5, 1.0, 2.5)
MAP_TRANSIT_S = 0.5


def t_sat_ms(tr_ms: float, flip_deg: float, t1_ms: float = BLOOD.t1_ms) -> float:
    """Return the saturation time constant of the approach to the static steady state."""
    return 1.0 / (1.0 / t1_ms + abs(math.log(math.cos(math.radians(flip_deg)))) / tr_ms)


def ernst_deg(tr_ms: float, t1_ms: float = BLOOD.t1_ms) -> float:
    """Return the Ernst angle, the optimum for fully saturated spins."""
    return math.degrees(math.acos(math.exp(-tr_ms / t1_ms)))


def efficiency(tr_ms: float, flip_deg: float, transit_s: float) -> float:
    """Return signed CNR per sqrt(second) of blood crossing the slab in transit_s."""
    seq = TofSequence(
        timing=fl3d1r_timing(0.3, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, tr_ms, PRISMA_XR),
        flip_deg=flip_deg, modules=FATSAT)
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
    flow = FlowGeometry(vessel, slab, SLAB_MM / transit_s, N_SHELLS, SLAB_MM * 1.001)
    blood = float(np.mean(seq.blood_signal(BLOOD, flow, np.linspace(0.0, flow.path_mm(), 28))))
    return float(cnr_efficiency(blood, float(seq.static_signal(BACKGROUND)), seq.time_per_line_ms()))


def grid_for(transit_s: float) -> np.ndarray:
    """Return CNR per sqrt(second) over the (TR, flip) plane, shape (flips, TRs)."""
    return np.array([[efficiency(tr, flip, transit_s) for tr in TRS_MS] for flip in FLIPS_DEG])


def ridge(grid: np.ndarray) -> np.ndarray:
    """Return the optimal flip angle at each TR."""
    return FLIPS_DEG[np.argmax(grid, axis=0)]


def panel_a(ax, grid: np.ndarray) -> None:
    """Map CNR relative to its own maximum, with the ridge and the within-10 % band."""
    rel = grid / grid.max()
    mesh = ax.contourf(TRS_MS, FLIPS_DEG, rel, levels=np.linspace(0.3, 1.0, 15), cmap="Blues", extend="min")
    ax.contour(TRS_MS, FLIPS_DEG, rel, levels=[0.9], colors=[INK], linewidths=1.4)
    ax.plot(TRS_MS, ridge(grid), color=INK, lw=2.2, label="optimal flip")
    sat = np.array([[t_sat_ms(tr, flip) for tr in TRS_MS] for flip in FLIPS_DEG])
    iso = ax.contour(TRS_MS, FLIPS_DEG, sat, levels=[150, 275, 500, 900], colors=[SERIES[1]],
                     linewidths=0.9, linestyles="--")
    ax.clabel(iso, fmt=lambda v: f"T_sat {v:.0f} ms", fontsize=6.5, inline_spacing=6)
    ax.set_xlabel("TR (ms)", color=INK_2)
    ax.set_ylabel("Nominal flip angle (deg)", color=INK_2)
    ax.set_title(f"a  CNR per sqrt(time) relative to best, transit {MAP_TRANSIT_S:g} s",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left")
    plt.colorbar(mesh, ax=ax, fraction=0.045, label="fraction of best")
    ax.grid(color=GRID, lw=0.4, alpha=0.4)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_b(ax, ridges: dict[float, np.ndarray]) -> None:
    """Draw the optimal flip against TR for each transit time, with the Ernst angle as the slow limit."""
    for (transit, best), colour in zip(ridges.items(), SERIES):
        ax.plot(TRS_MS, best, color=colour, lw=2, marker="o", ms=3,
                label=f"{transit:g} s  ({SLAB_MM / transit:.0f} mm/s at {SLAB_MM:g} mm)")
    ax.plot(TRS_MS, [ernst_deg(tr) for tr in TRS_MS], color=INK_2, lw=1.4, ls=":",
            label="Ernst angle (no inflow)")
    ax.set_xlabel("TR (ms)", color=INK_2)
    ax.set_ylabel("CNR-optimal flip angle (deg)", color=INK_2)
    ax.set_title("b  The ridge, by slab transit time", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left", title="slab transit time",
              title_fontsize=7.5)
    style(ax)


def main() -> None:
    """Compute the map and the ridges, plot, and print the lookup table."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    grid = grid_for(MAP_TRANSIT_S)
    panel_a(axes[0], grid)
    ridges = {t: ridge(grid_for(t)) if t != MAP_TRANSIT_S else ridge(grid) for t in TRANSITS_S}
    panel_b(axes[1], ridges)

    print("Optimal flip angle (deg) by TR and slab transit time")
    print("  TR (ms) " + "".join(f"{t:>9.2f} s" for t in TRANSITS_S))
    for i, tr in enumerate(TRS_MS):
        print(f"  {tr:7.0f} " + "".join(f"{ridges[t][i]:>9.0f}  " for t in TRANSITS_S)
              + f"   Ernst {ernst_deg(tr):4.1f}")
    print("\nT_sat at the optimum (ms) -- constant along each ridge is the point:")
    for t in TRANSITS_S:
        sats = [t_sat_ms(tr, f) for tr, f in zip(TRS_MS, ridges[t])]
        print(f"  transit {t:5.2f} s: T_sat {np.mean(sats):6.0f} ms "
              f"(spread {np.min(sats):.0f}-{np.max(sats):.0f}), ratio T_sat/transit {np.mean(sats)/1000/t:.2f}")
    best_per_tr = grid.max(axis=0)
    print(f"\nAt transit {MAP_TRANSIT_S:g} s, CNR at the ridge varies {best_per_tr.min():.3f}-{best_per_tr.max():.3f} "
          f"across TR {TRS_MS[0]:.0f}-{TRS_MS[-1]:.0f} ms, i.e. {100 * (1 - best_per_tr.min() / best_per_tr.max()):.0f} %")

    fig.suptitle("TR and flip angle are one setting: every pair on the ridge gives the same contrast",
                 fontsize=11, color=INK)
    save(fig, "fig_general_3_tr_flip",
         note="Nominal literature parameters, no measured data and no parameter draw: "
              "this figure shows the structure of the design space, not its uncertainty.")


if __name__ == "__main__":
    main()
