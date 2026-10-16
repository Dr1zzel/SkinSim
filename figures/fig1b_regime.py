"""Figure 1b: why cutaneous TOF is a different problem, in absolute signal.

One panel, one message. As blood slows into the cutaneous range, inflow
enhancement is progressively spent and the slab-mean blood signal falls toward
the static steady state. Whether the vessel is still visible there is decided
entirely by the background it sits against:

  - against DERMIS, blood stays above the background at every physiological
    velocity, so a dermal vessel is in principle always positive-contrast;
  - against UNSUPPRESSED SUBCUTANEOUS FAT, blood falls below the background
    inside the physiological range, and the vessel inverts and disappears;
  - with the fat suppressed, the background collapses to near zero and blood
    stays far above it across the whole range.

So the limiting problem at 1-10 mm depth is not the slow flow by itself: it is
slow flow measured against bright fat. That is a statement the simulation makes
before any image is acquired.

Every number is read from tissues.py and protocols.py. No measured data.

Run: python figures/fig1b_regime.py
"""
from __future__ import annotations

import numpy as np
from common import GRID, INK, INK_2, N_SHELLS, SERIES, plt, save  # noqa: F401

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
VELOCITIES = np.geomspace(2.0, 500.0, 48)
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


def joint_fraction_darker(samples: list[dict[str, TissueSample]], velocities: np.ndarray,
                          seq: TofSequence, key: str) -> float:
    """Return the fraction of joint draws whose own vessel, at its own velocity, is darker than that background."""
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    darker = 0
    for s, v in zip(samples, velocities):
        flow = FlowGeometry(vessel, slab, float(v), N_SHELLS, SLAB_MM * 1.001)
        b = float(np.mean(seq.blood_signal(s["arterial_blood"], flow, np.linspace(0.0, flow.path_mm(), 32))))
        darker += b < float(seq.static_signal(s[key]))
    return darker / len(samples)


def sequences() -> tuple[TofSequence, TofSequence]:
    """Return the TOF readout without and with a per-TR fat-saturation pulse."""
    timing = fl3d1r_timing(INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, TR_MS, PRISMA_XR)
    return (TofSequence(timing=timing, flip_deg=FLIP_DEG),
            TofSequence(timing=timing, flip_deg=FLIP_DEG, modules=(SaturationModule("fatsat_s", 8.0),)))


def blood_curve(seq: TofSequence, samples: list[dict[str, TissueSample]]) -> np.ndarray:
    """Return slab-mean blood signal, shape (velocities, draws)."""
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    out = np.empty((len(VELOCITIES), len(samples)))
    for i, v in enumerate(VELOCITIES):
        flow = FlowGeometry(vessel, slab, v, N_SHELLS, SLAB_MM * 1.001)
        path = np.linspace(0.0, flow.path_mm(), 32)
        out[i] = [float(np.mean(seq.blood_signal(s["arterial_blood"], flow, path))) for s in samples]
    return out


def flat(seq: TofSequence, samples: list[dict[str, TissueSample]], key: str) -> np.ndarray:
    """Return the static steady-state signal of one tissue over the draws."""
    return np.array([float(seq.static_signal(s[key])) for s in samples])


def hline(ax, values: np.ndarray, colour: str, style: str, label: str, band: bool = True) -> float:
    """Draw a velocity-independent background as a median line with its 5-95 % band; return the median."""
    lo, mid, hi = np.percentile(values, [5, 50, 95])
    if band:
        ax.axhspan(lo, hi, color=colour, alpha=0.12, lw=0)
    ax.axhline(mid, color=colour, lw=1.8, ls=style, label=label)
    return float(mid)


def crossing(curve: np.ndarray, level: float) -> float | None:
    """Return the velocity at which the median blood curve crosses a flat background, by log interpolation."""
    med = np.median(curve, axis=-1) - level
    s = np.where(np.diff(np.sign(med)))[0]
    if not len(s):
        return None
    i = s[0]
    w = med[i] / (med[i] - med[i + 1])
    return float(np.exp(np.log(VELOCITIES[i]) + w * (np.log(VELOCITIES[i + 1]) - np.log(VELOCITIES[i]))))


def main() -> None:
    """Build the panel and print the numbers behind it."""
    samples, draw_velocities = draw()
    plain, fatsat = sequences()
    blood = blood_curve(plain, samples)  # blood is on-resonance: the fat pulse leaves it untouched
    v25, v50, v75 = np.percentile(cutaneous_velocities(), [25, 50, 75])

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    fig.subplots_adjust(left=0.105, right=0.775, top=0.845, bottom=0.145)

    ax.axvspan(v25, v75, color=INK, alpha=0.06, lw=0, zorder=0)
    ax.annotate(f"target cutaneous vessels\n{v25:.0f}-{v75:.0f} mm/s", (np.sqrt(v25 * v75), 0.425),
                ha="center", va="top", fontsize=8, color=INK)

    fat_u = hline(ax, flat(plain, samples, "hypodermis"), FAT_C, "-", "subcutaneous fat")
    fat_s = hline(ax, flat(fatsat, samples, "hypodermis"), FAT_C, "--", "fat, suppressed")
    derm = hline(ax, flat(plain, samples, "dermis"), INK_2, ":", "dermis", band=False)

    lo, mid, hi = np.percentile(blood, [5, 50, 95], axis=-1)
    ax.fill_between(VELOCITIES, lo, hi, color=BLOOD_C, alpha=0.16, lw=0, zorder=4)
    ax.plot(VELOCITIES, mid, color=BLOOD_C, lw=2.4, zorder=5)

    v_cross = crossing(blood, fat_u)
    frac = joint_fraction_darker(samples, draw_velocities, plain, "hypodermis")
    if v_cross is not None:
        ax.plot([v_cross], [fat_u], "o", ms=8.5, mfc="white", mec=BLOOD_C, mew=2.2, zorder=6)
        ax.annotate(f"below {v_cross:.0f} mm/s the vessel is\ndarker than the fat around it\n"
                    f"— {frac:.0%} of parameter draws",
                    (v_cross, fat_u), xytext=(14, -46), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=INK, zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.9,
                                    connectionstyle="arc3,rad=-0.3"))

    for y, text, colour, weight in ((mid[-1], "blood, inflowing", BLOOD_C, "bold"),
                                    (fat_u, "subcutaneous fat", FAT_C, "normal"),
                                    (derm, "dermis", INK_2, "normal"),
                                    (fat_s, "fat, suppressed", FAT_C, "normal")):
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=colour,
                    weight=weight, annotation_clip=False)

    ax.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, 0.44))
    ax.set_xlabel(f"mean blood velocity (mm/s)   —   {SLAB_MM:g} mm slab, so 10 mm/s is a {SLAB_MM / 10:.1f} s crossing",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("signal (units of dermis $M_0$)", color=INK_2)
    ax.set_title("Inflow fades across the cutaneous range; the background decides what survives",
                 fontsize=10.5, color=INK, loc="left", pad=16)
    ax.text(0, 1.015, f"spoiled-GRE TOF, TR {TR_MS:g} ms, {FLIP_DEG:g}°, {SLAB_MM:g} mm slab, vessel in the slab plane",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    print(f"Backgrounds (median, dermis M0): fat {fat_u:.3f}, fat suppressed {fat_s:.3f}, dermis {derm:.3f}")
    print(f"Target cutaneous velocity: {v50:.0f} mm/s [IQR {v25:.0f}-{v75:.0f}]")
    for v in (3.0, 10.0, 30.0, 100.0, 400.0):
        i = int(np.argmin(np.abs(VELOCITIES - v)))
        b = np.median(blood[i])
        print(f"  {VELOCITIES[i]:6.1f} mm/s: blood {b:.3f} = {b / fat_u:4.2f}x fat, "
              f"{b / fat_s:5.1f}x suppressed fat, {b / derm:4.2f}x dermis")
    if v_cross is not None:
        print(f"Blood crosses unsuppressed fat at {v_cross:.1f} mm/s (transit {SLAB_MM / v_cross:.2f} s)")
    print(f"Joint draws (own velocity AND own tissue parameters) darker than unsuppressed fat: {frac:.0%}")
    print(f"Blood never falls below suppressed fat: min ratio {np.min(mid) / fat_s:.1f}x")

    save(fig, "fig1b_regime", n=N_DRAWS, seed=SEED,
         note=f"Simulation only; literature tissue parameters, no measured data. "
              f"LHS seed {SEED}, {N_DRAWS} draws; lines median, bands 5-95 %.")


if __name__ == "__main__":
    main()
