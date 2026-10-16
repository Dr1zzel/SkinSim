"""Figure 1b: why cutaneous TOF is a different problem, in absolute signal.

One panel, one message. As blood slows into the cutaneous range, inflow
enhancement is progressively spent and the slab-mean blood signal falls toward
the static steady state. Whether the vessel is still visible there is decided
by the background it sits against, which in skin is set by depth:

  - against DERMIS (shallow), blood stays above the background at every
    physiological velocity;
  - against UNSUPPRESSED SUBCUTANEOUS FAT (1-10 mm, where the target vessels
    are), blood falls BELOW the background and the vessel inverts. Every draw
    in the literature parameter ranges inverts somewhere; only the velocity at
    which it happens is uncertain, and that velocity lands inside the target
    range;
  - with the fat suppressed, the background collapses to near zero and the
    inversion disappears entirely.

Conventional TOF operates two decades to the right of this, on the plateau,
which is why its parameter choices do not transfer.

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

SEED, N_DRAWS = 20261014, 96
TR_MS, FLIP_DEG = 35.0, 20.0
SLAB_MM = 12.0  # stated: the transit time that governs inflow is SLAB_MM / velocity
INPLANE_MM = 0.3
VELOCITIES = np.geomspace(1.5, 800.0, 64)
INTRACRANIAL_MM_S = (200.0, 600.0)  # typical TOF operating range, for scale only
KEYS = ("arterial_blood", "hypodermis", "dermis")


def cutaneous_velocities(n: int = 4096) -> np.ndarray:
    """Return drawn cutaneous velocities, target velocity times the thermoregulatory factor."""
    _, others = sample_tissues(["arterial_blood"], n, SEED,
                               haemodynamic_keys=("target_vessel_velocity", "thermoregulatory_factor"))
    return np.asarray(others["target_vessel_velocity"]) * np.asarray(others["thermoregulatory_factor"])


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


def _flow(velocity: float) -> FlowGeometry:
    """Return the laminar flow geometry of a target vessel lying in the slab plane."""
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    return FlowGeometry(vessel, slab, velocity, N_SHELLS, SLAB_MM * 1.001)


def blood_at(seq: TofSequence, sample: TissueSample, velocity: float) -> float:
    """Return slab-mean blood signal for one tissue sample at one velocity."""
    flow = _flow(velocity)
    return float(np.mean(seq.blood_signal(sample, flow, np.linspace(0.0, flow.path_mm(), 32))))


def blood_curve(seq: TofSequence, samples: list[dict[str, TissueSample]]) -> np.ndarray:
    """Return slab-mean blood signal, shape (velocities, draws)."""
    return np.array([[blood_at(seq, s["arterial_blood"], v) for s in samples] for v in VELOCITIES])


def flat(seq: TofSequence, samples: list[dict[str, TissueSample]], key: str) -> np.ndarray:
    """Return the static steady-state signal of one tissue over the draws."""
    return np.array([float(seq.static_signal(s[key])) for s in samples])


def crossing_velocities(seq: TofSequence, samples: list[dict[str, TissueSample]], key: str) -> np.ndarray:
    """Return, per draw, the velocity at which that draw's blood falls below that draw's own background.

    Solved by bisection in log velocity rather than read off the plotting grid,
    so the quoted bounds do not move when the grid is refined.
    """
    from scipy.optimize import brentq

    out = []
    for s in samples:
        background = float(seq.static_signal(s[key]))

        def excess(log_v: float, sample=s["arterial_blood"], g=background) -> float:
            return blood_at(seq, sample, float(np.exp(log_v))) - g

        lo, hi = np.log(VELOCITIES[0]), np.log(VELOCITIES[-1])
        if excess(lo) * excess(hi) < 0:
            out.append(float(np.exp(brentq(excess, lo, hi, xtol=1e-4))))
    return np.array(out)


def hline(ax, values: np.ndarray, colour: str, style: str, alpha: float = 0.07) -> float:
    """Draw a velocity-independent background as a median line with its 5-95 % band; return the median."""
    lo, mid, hi = np.percentile(values, [5, 50, 95])
    if alpha:
        ax.axhspan(lo, hi, color=colour, alpha=alpha, lw=0, zorder=1)
    ax.axhline(mid, color=colour, lw=1.8, ls=style, zorder=3)
    return float(mid)


def main() -> None:
    """Build the panel and print the numbers behind it."""
    samples, draw_velocities = draw()
    plain, fatsat = sequences()
    blood = blood_curve(plain, samples)  # blood is on-resonance: the fat pulse leaves it untouched
    v25, v75 = np.percentile(cutaneous_velocities(), [25, 75])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=0.10, right=0.775, top=0.835, bottom=0.145)
    ax.axvspan(v25, v75, color=INK, alpha=0.06, lw=0, zorder=0)

    fat_u = hline(ax, flat(plain, samples, "hypodermis"), FAT_C, "-")
    fat_s = hline(ax, flat(fatsat, samples, "hypodermis"), FAT_C, "--")
    derm = hline(ax, flat(plain, samples, "dermis"), INK_2, ":", alpha=0.0)

    lo, mid, hi = np.percentile(blood, [5, 50, 95], axis=-1)
    ax.fill_between(VELOCITIES, lo, hi, color=BLOOD_C, alpha=0.16, lw=0, zorder=4)
    ax.plot(VELOCITIES, mid, color=BLOOD_C, lw=2.4, zorder=5)

    # Where each draw's vessel goes dark against its own unsuppressed fat.
    xs = crossing_velocities(plain, samples, "hypodermis")
    x5, x50, x95 = np.percentile(xs, [5, 50, 95])
    ax.plot([x5, x95], [fat_u, fat_u], color=BLOOD_C, lw=3.4, solid_capstyle="butt", alpha=0.45, zorder=6)
    ax.plot([x50], [fat_u], "o", ms=8.5, mfc="white", mec=BLOOD_C, mew=2.2, zorder=7)

    frac = float(np.mean([blood_at(plain, s["arterial_blood"], float(v)) < float(plain.static_signal(s["hypodermis"]))
                          for s, v in zip(samples, draw_velocities)]))
    ax.annotate(f"every draw inverts somewhere in {x5:.0f}-{x95:.0f} mm/s\n"
                f"below it the vessel is darker than the fat around it",
                (x50, fat_u), xytext=(16, -44), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=INK, zorder=8,
                bbox=dict(fc="white", ec="none", alpha=0.88, pad=2.0),
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.9, connectionstyle="arc3,rad=-0.3"))

    ax.annotate(f"target cutaneous vessels\n{v25:.0f}-{v75:.0f} mm/s", (np.sqrt(v25 * v75), 0.487),
                ha="center", va="top", fontsize=8, color=INK)
    ic = INTRACRANIAL_MM_S
    ax.plot(ic, [0.468, 0.468], color=INK_2, lw=2.6, solid_capstyle="butt", alpha=0.5)
    ax.annotate("conventional TOF\noperates here", (np.sqrt(ic[0] * ic[1]), 0.461), ha="center", va="top",
                fontsize=8, color=INK_2)

    for y, text, colour, weight in ((mid[-1], "blood, inflowing", BLOOD_C, "bold"),
                                    (fat_u, "subcutaneous fat", FAT_C, "normal"),
                                    (derm, "dermis", INK_2, "normal"),
                                    (fat_s, "fat, suppressed", FAT_C, "normal")):
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=colour,
                    weight=weight, annotation_clip=False)

    ax.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, 0.50))
    ax.set_xlabel(f"mean blood velocity (mm/s)      —      {SLAB_MM:g} mm slab, so 10 mm/s is a "
                  f"{SLAB_MM / 10:.1f} s crossing", color=INK_2, fontsize=9)
    ax.set_ylabel("signal, relative to dermis $M_0$", color=INK_2)
    ax.set_title("Inflow fades across the cutaneous range — and the background decides what is left",
                 fontsize=10.5, color=INK, loc="left", pad=17)
    ax.text(0, 1.016, f"spoiled-GRE TOF, TR {TR_MS:g} ms, {FLIP_DEG:g}°, {SLAB_MM:g} mm slab, "
                      f"vessel lying in the slab plane", transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    print(f"Backgrounds (median, dermis M0): fat {fat_u:.3f}, fat suppressed {fat_s:.3f}, dermis {derm:.3f}")
    print(f"Target cutaneous velocity IQR {v25:.0f}-{v75:.0f} mm/s")
    print(f"Inversion against unsuppressed fat: {len(xs)}/{N_DRAWS} draws invert; "
          f"velocity {x50:.0f} mm/s [{x5:.0f}-{x95:.0f}], transit {SLAB_MM / x50:.2f} s "
          f"[{SLAB_MM / x95:.2f}-{SLAB_MM / x5:.2f}]")
    print(f"Joint draws (own velocity AND own tissue parameters) darker than unsuppressed fat: {frac:.0%}")
    print(f"Against suppressed fat the median curve never inverts: min ratio {np.min(mid) / fat_s:.1f}x")
    for v in (5.0, 20.0, 100.0, 400.0):
        i = int(np.argmin(np.abs(VELOCITIES - v)))
        print(f"  {VELOCITIES[i]:6.1f} mm/s: blood {mid[i]:.3f} = {mid[i] / fat_u:4.2f}x fat, {mid[i] / derm:4.2f}x dermis")

    save(fig, "fig1b_regime", n=N_DRAWS, seed=SEED,
         note=f"Simulation only; literature tissue parameters, no measured data. "
              f"LHS seed {SEED}, {N_DRAWS} draws; lines median, bands 5-95 %.")


if __name__ == "__main__":
    main()
