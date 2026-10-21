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

  - inflow still works: 1.4x at 3 mm/s to 4.5x at 80 mm/s over the stagnant
    floor, so it is not exhausted at cutaneous velocities;
  - but it is not what makes the vessel visible. Stagnant blood already beats
    dermis and suppressed fat. Inflow is load-bearing only against UNSUPPRESSED
    fat, where stagnant blood sits at 0.40x the background - and there it
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

from geometry import Slab, Vessel
from sequence import FlowGeometry, SaturationModule, TissueSample, TofSequence
from protocols import MEASURED
from tissues import HAEMODYNAMICS, PARAM_FIELDS, sample_tissues

# Validated two-hue pair (dataviz validator, light surface): blood vs fat.
BLOOD_C, FAT_C = SERIES[1], SERIES[0]
DOTTED = (0, (1.2, 1.8))

SEED, N_DRAWS = 20261014, 96

# The readout is the 20 October arm WITHOUT fat saturation, because that is the
# TR the sequence reaches when nothing else is played: adding a fat-sat pulse at
# the same TR is then the counterfactual this figure needs, rather than a
# comparison that also changes the timing. Every number comes from the protocol.
BASE = MEASURED["20oct_nosat"]
TR_MS, FLIP_DEG = BASE.timing.tr_ms, BASE.flip_deg
SLAB_MM, INPLANE_MM = BASE.slab_thickness_mm, BASE.inplane_mm
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
    """Return the TOF readout without and with a per-TR fat-saturation pulse, at one shared TR.

    TONE and the travelling saturation band are deliberately left OFF, although
    the acquired protocol runs both. This figure isolates what inflow alone buys
    against each background; a ramp that varies the flip with depth and a band
    that pre-saturates the inflow would both fold into the same curve and could
    not be read apart from it. Figure 4 is where those arms are compared.
    """
    return (TofSequence(timing=BASE.timing, flip_deg=FLIP_DEG),
            TofSequence(timing=BASE.timing, flip_deg=FLIP_DEG,
                        modules=(SaturationModule("fatsat_s", BASE.fat_sat_delay_ms),)))


def blood_at(seq: TofSequence, sample: TissueSample, velocity: float) -> float:
    """Return slab-mean blood signal for one tissue sample at one velocity.

    Geometry: polar 90 deg with the slab at orientation 90 deg makes the vessel
    direction PARALLEL TO THE SLAB NORMAL, so the vessel crosses the slab end-on
    over the full 12 mm and the whole velocity counts towards the crossing. That
    is the BEST case for inflow - any other angle lengthens the path through the
    slab and saturates the blood further - so the inversion reported here is the
    conservative end of the result, not the worst case.

    The returned value is the mean over 32 points along that path, so it mixes
    fresh blood at the entry face with saturated blood at the exit. A maximum
    intensity projection would weight the entry end more heavily.
    """
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


def reference(ax, values, colour: str, style, label: str, band: bool = True) -> float:
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
    fig.subplots_adjust(left=0.115, right=0.775, top=0.90, bottom=0.26)

    blood = np.array([[blood_at(plain, s["arterial_blood"], v) for s in samples] for v in VELOCITIES])
    lo, mid, hi = np.percentile(blood, [5, 50, 95], axis=-1)

    fat_u = reference(ax, [plain.static_signal(s["hypodermis"]) for s in samples], FAT_C, "-",
                      "subcutaneous fat", band=True)
    fat_s = reference(ax, [fatsat.static_signal(s["hypodermis"]) for s in samples], FAT_C, "--",
                      "fat, suppressed", band=True)
    derm_v = np.array([float(plain.static_signal(s["dermis"])) for s in samples])
    stat_v = np.array([float(plain.static_signal(s["arterial_blood"])) for s in samples])
    derm = reference(ax, derm_v, INK_2, DOTTED, "dermis")
    stat = reference(ax, stat_v, BLOOD_C, DOTTED, "blood, stagnant")

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

    ax.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, 0.45), xticks=TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("mean blood velocity (mm/s)", color=INK_2, labelpad=40)
    ax.set_ylabel("simulated signal\n(1 = full dermis magnetisation)", color=INK_2)
    # No in-figure headline: the claim belongs in the caption, not on the axes.
    ax.text(0, 1.02, f"spoiled-GRE TOF, TR {TR_MS:g} ms, {FLIP_DEG:g}°, {SLAB_MM:g} mm slab",
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
    for name, level in (("dermis", derm_v), ("suppressed fat",
                        np.array([float(fatsat.static_signal(s["hypodermis"])) for s in samples]))):
        d = stat_v - level
        print(f"  paired, stagnant blood - {name}: {np.median(d):+.3f} "
              f"[{np.percentile(d, 5):+.3f}, {np.percentile(d, 95):+.3f}], brighter in {np.mean(d > 0):.0%} of draws")

    save(fig, "fig1b_regime", n=N_DRAWS, seed=SEED, footer=False,
         note=f"Simulation only; literature tissue parameters, no measured data. "
              f"LHS seed {SEED}, {N_DRAWS} draws; lines median, bands 5-95 %.")


if __name__ == "__main__":
    main()
