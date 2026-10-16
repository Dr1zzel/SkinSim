"""Figure 1b: what inflow buys in skin, and which background it has to beat.

One panel. Blood signal against velocity, with the four levels it is measured
against drawn as flat references:

    blood, stagnant       the spoiled-GRE fixed point - a vessel with no flow.
                          The gap between it and the curve IS the inflow
                          enhancement, so the figure shows inflow rather than
                          asking the reader to assume it.
    subcutaneous fat      the background at the 1-10 mm target depth.
    fat, suppressed       the same background with a fat-saturation pulse.
    dermis                the background for the shallowest vessels.

Two things follow, and they are different claims:

  - inflow still works: 1.4x at 3 mm/s to 3.4x at 70 mm/s over the stagnant
    floor, so it is not exhausted at cutaneous velocities;
  - but it is not what makes the vessel visible. Stagnant blood already beats
    dermis and suppressed fat. Inflow is load-bearing only against UNSUPPRESSED
    fat, where stagnant blood sits at 0.42x the background - and there it
    fails, inverting in every parameter draw.

Layout: the velocity regimes sit on a rail below the axes rather than as shaded
bands inside it, and the numbers belong in the caption rather than in callouts,
so the plot area carries curves and nothing else.

Every number is read from tissues.py and protocols.py. No measured data.

Run: python figures/fig1b_regime.py
"""
from __future__ import annotations

import numpy as np
from common import GRID, INK, INK_2, N_SHELLS, SERIES, plt, save

from acquisition import Slab
from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from sequences.base import FlowGeometry, TissueSample
from sequences.prep import SaturationModule
from sequences.tof import TofSequence
from skin import Vessel
from tissues import HAEMODYNAMICS, PARAM_FIELDS, sample_tissues

# Validated two-hue pair (dataviz validator, light surface): blood vs fat.
BLOOD_C, FAT_C = SERIES[1], SERIES[0]
DOTTED = (0, (1.2, 1.8))

SEED, N_DRAWS = 20261014, 96
TR_MS, FLIP_DEG, SLAB_MM, INPLANE_MM = 35.0, 20.0, 12.0, 0.3
VELOCITIES = np.geomspace(2.0, 300.0, 64)
CONVENTIONAL_TOF = (200.0, 600.0)  # intracranial / run-off TOF, for scale
TICKS = (2, 3, 5, 10, 20, 30, 50, 100, 200, 300)
KEYS = ("arterial_blood", "hypodermis", "dermis")


def draw() -> tuple[list[dict[str, TissueSample]], np.ndarray]:
    """Return one joint Latin-hypercube draw: per-sample tissue dicts and that sample's vessel velocity."""
    tissues, others = sample_tissues(list(KEYS), N_DRAWS, SEED,
                                     haemodynamic_keys=("target_vessel_velocity", "thermoregulatory_factor"))
    samples = [{k: TissueSample(k, **{f: float(np.atleast_1d(getattr(tissues[k], f))[i]) for f in PARAM_FIELDS})
                for k in KEYS} for i in range(N_DRAWS)]
    return samples, np.asarray(others["target_vessel_velocity"]) * np.asarray(others["thermoregulatory_factor"])


def sequences() -> tuple[TofSequence, TofSequence]:
    """Return the TOF readout without and with a per-TR fat-saturation pulse."""
    timing = fl3d1r_timing(INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, TR_MS, PRISMA_XR)
    return (TofSequence(timing=timing, flip_deg=FLIP_DEG),
            TofSequence(timing=timing, flip_deg=FLIP_DEG, modules=(SaturationModule("fatsat_s", 8.0),)))


def blood_at(seq: TofSequence, sample: TissueSample, velocity: float) -> float:
    """Return slab-mean blood signal for one tissue sample at one velocity."""
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    flow = FlowGeometry(vessel, Slab(thickness_mm=SLAB_MM, orientation_deg=90.0),
                        velocity, N_SHELLS, SLAB_MM * 1.001)
    return float(np.mean(seq.blood_signal(sample, flow, np.linspace(0.0, flow.path_mm(), 32))))


def crossing_velocities(seq: TofSequence, samples: list[dict[str, TissueSample]], key: str) -> np.ndarray:
    """Return, per draw, the velocity at which that draw's blood falls below its own background.

    Solved by bisection in log velocity, not read off the plotting grid, so the
    bounds do not move when the grid is refined.
    """
    from scipy.optimize import brentq

    out = []
    for s in samples:
        level = float(seq.static_signal(s[key]))

        def excess(log_v: float, sample=s["arterial_blood"], g=level) -> float:
            return blood_at(seq, sample, float(np.exp(log_v))) - g

        lo, hi = np.log(VELOCITIES[0]), np.log(VELOCITIES[-1])
        if excess(lo) * excess(hi) < 0:
            out.append(float(np.exp(brentq(excess, lo, hi, xtol=1e-4))))
    return np.array(out)


def reference(ax, values, colour: str, style, label: str, band: bool = False) -> float:
    """Draw a velocity-independent level as a line, label it in the right margin, return its median."""
    lo, mid, hi = np.percentile(np.asarray(values, dtype=float), [5, 50, 95])
    if band:
        ax.axhspan(lo, hi, color=colour, alpha=0.055, lw=0, zorder=1)
    ax.axhline(mid, color=colour, lw=1.7, ls=style, zorder=3)
    ax.annotate(label, (1.0, mid), xycoords=("axes fraction", "data"), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=colour,
                annotation_clip=False)
    return float(mid)


def regime_rail(ax, target: tuple[float, float]) -> None:
    """Draw the velocity regimes on a rail below the axes, keeping the plot area for curves."""
    t = ax.get_xaxis_transform()
    for (lo, hi), colour, text in ((target, INK, "target cutaneous vessels"),
                                   (CONVENTIONAL_TOF, INK_2, "conventional TOF")):
        drawn_hi = min(hi, VELOCITIES[-1])
        ax.plot([lo, drawn_hi], [-0.09, -0.09], transform=t, color=colour, lw=3.2, alpha=0.5,
                solid_capstyle="butt", clip_on=False, zorder=5)
        ax.annotate(f"{text}\n{lo:.0f}-{hi:.0f} mm/s", (np.sqrt(lo * drawn_hi), -0.125), xycoords=t,
                    ha="center", va="top", fontsize=8, color=colour, clip_on=False)


def main() -> None:
    """Build the panel and print every number the caption needs."""
    samples, drawn_v = draw()
    plain, fatsat = sequences()
    v_lo, v_hi = np.percentile(drawn_v, [5, 95])

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(left=0.115, right=0.775, top=0.85, bottom=0.26)

    blood = np.array([[blood_at(plain, s["arterial_blood"], v) for s in samples] for v in VELOCITIES])
    lo, mid, hi = np.percentile(blood, [5, 50, 95], axis=-1)

    fat_u = reference(ax, [plain.static_signal(s["hypodermis"]) for s in samples], FAT_C, "-",
                      "subcutaneous fat", band=True)
    fat_s = reference(ax, [fatsat.static_signal(s["hypodermis"]) for s in samples], FAT_C, "--",
                      "fat, suppressed", band=True)
    derm = reference(ax, [plain.static_signal(s["dermis"]) for s in samples], INK_2, DOTTED, "dermis")
    stat = reference(ax, [plain.static_signal(s["arterial_blood"]) for s in samples], BLOOD_C, DOTTED,
                     "blood, stagnant")

    # The inversion as an area, not a crossing the reader has to decode.
    ax.fill_between(VELOCITIES, mid, fat_u, where=mid < fat_u, color=BLOOD_C, alpha=0.26,
                    lw=0, zorder=2, interpolate=True)
    ax.fill_between(VELOCITIES, lo, hi, color=BLOOD_C, alpha=0.15, lw=0, zorder=4)
    ax.plot(VELOCITIES, mid, color=BLOOD_C, lw=2.5, zorder=5)
    ax.annotate("blood, inflowing", (1.0, mid[-1]), xycoords=("axes fraction", "data"), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=BLOOD_C,
                weight="bold", annotation_clip=False)

    xs = crossing_velocities(plain, samples, "hypodermis")
    x5, x50, x95 = np.percentile(xs, [5, 50, 95])
    frac = float(np.mean([blood_at(plain, s["arterial_blood"], float(v)) < float(plain.static_signal(s["hypodermis"]))
                          for s, v in zip(samples, drawn_v)]))
    ax.plot([x50], [fat_u], "o", ms=8, mfc="white", mec=BLOOD_C, mew=2.1, zorder=7)
    ax.annotate(f"darker than fat\n{frac:.0%} of draws", (VELOCITIES[0] * 1.95, fat_u * 0.78),
                ha="center", va="center", fontsize=8.5, color=BLOOD_C, weight="bold", zorder=8)

    ax.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, 0.45), xticks=TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("mean blood velocity (mm/s)", color=INK_2, labelpad=40)
    ax.set_ylabel("simulated signal\n(1 = full dermis magnetisation)", color=INK_2)
    ax.set_title("Inflow is not what makes a cutaneous vessel visible — the background is",
                 fontsize=10.5, color=INK, loc="left", pad=16)
    ax.text(0, 1.015, f"spoiled-GRE TOF, TR {TR_MS:g} ms, {FLIP_DEG:g}°, {SLAB_MM:g} mm slab; "
                      f"gap to the stagnant line = the inflow gain",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    regime_rail(ax, (v_lo, v_hi))

    i_lo, i_hi = (int(np.argmin(np.abs(VELOCITIES - v))) for v in (v_lo, v_hi))
    print(f"Target velocity, 5-95 % of draws: {v_lo:.0f}-{v_hi:.0f} mm/s")
    print(f"Levels: stagnant blood {stat:.3f} | dermis {derm:.3f} | fat {fat_u:.3f} | suppressed fat {fat_s:.3f}")
    print(f"Stagnant blood already beats dermis {stat / derm:.2f}x and suppressed fat {stat / fat_s:.1f}x, "
          f"but unsuppressed fat only {stat / fat_u:.2f}x")
    print(f"Inflow enhancement over stagnant: {mid[i_lo] / stat:.2f}x at {v_lo:.0f} mm/s, "
          f"{mid[i_hi] / stat:.2f}x at {v_hi:.0f} mm/s")
    print(f"Inversion vs unsuppressed fat: {len(xs)}/{N_DRAWS} draws invert, at {x50:.0f} mm/s "
          f"[{x5:.0f}-{x95:.0f}] (transit {SLAB_MM / x50:.2f} s); {frac:.0%} of joint draws sit below it")
    print(f"Against suppressed fat the median never inverts: minimum {np.min(mid) / fat_s:.1f}x")

    save(fig, "fig1b_regime", n=N_DRAWS, seed=SEED,
         note=f"Simulation only; literature tissue parameters, no measured data. "
              f"LHS seed {SEED}, {N_DRAWS} draws; lines median, bands 5-95 %.")


if __name__ == "__main__":
    main()
