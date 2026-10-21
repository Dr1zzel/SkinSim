"""Abstract Figure 4: CNR efficiency over the plane the method has to work in.

CLAUDE.md names absolute contrast per sqrt(time per k-space line) as the metric
that decides the protocol. This is the figure it decides from.

(a) Efficiency over velocity x vessel orientation at the operating point. Velocity
    is a SWEPT AXIS here, not an uncertainty, so the drawn velocity is overridden
    and only the tissue and hardware draws remain; the map is their median.
(b) Efficiency against vessel orientation for each contrast mechanism, with the
    drawn velocity restored, so the band marginalises over everything uncertain.
    Where a curve crosses zero the vessel is darker than its background.
(c) Efficiency against flip angle, which is the one knob that costs nothing to
    change, against the flip the operating point actually used.

Backgrounds: the vessel sits in the hypodermis, so the background is fat -
suppressed or not, depending on the arm. Absolute contrast, never a ratio: with
fat saturation the background is close to zero and a ratio is meaningless.
"""
from __future__ import annotations

import numpy as np
from common import (GRID, HYPODERMIS_DEPTH_MM, INK, INK_2, MAX_PATH_MM, N_DRAWS, N_SHELLS, SEED,
                    SERIES, band, draw, plt, save, style, summary, tilted)
from predict import predict
from protocols import ALL_PROTOCOLS, CANDIDATE_FLIPS_DEG, CONTRAST_ARMS, MEASURED
from tissues import HAEMODYNAMICS, PARAM_FIELDS, TissueSample

OPERATING = MEASURED["13oct_0.35mm"]
VELOCITY = HAEMODYNAMICS["target_vessel_velocity"]
ANGLES_DEG = np.linspace(0.0, 90.0, 10)   # 0 = crosses the slab end-on, 90 = lies in the slab plane
VELOCITIES = np.geomspace(VELOCITY.lo, VELOCITY.hi, 14)
N_POSITIONS = 8

tissues, others = draw()


def tile(sample: TissueSample, reps: int) -> TissueSample:
    """Return the sample with every field repeated reps times along the sample axis."""
    return TissueSample(sample.name, **{
        f: np.tile(np.atleast_1d(np.asarray(getattr(sample, f), dtype=float)), reps)
        for f in PARAM_FIELDS})


def efficiency(protocol, flow, blood, background, rel, residual):
    """Return the CNR efficiency averaged along the vessel's whole course through the excited volume."""
    # Averaged, not taken at the best position: blood is fully fresh where it
    # enters, so a peak-position convention reports the same unsaturated blood
    # whatever the orientation or velocity and erases the inflow physics this
    # figure is about. The average over the residence inside the slab is what
    # differs between a vessel that crosses and one that lies in the plane.
    pred = predict(protocol, blood, background, flow, np.linspace(0.0, flow.path_mm(), N_POSITIONS),
                   background_structure_rel=rel, band_pulse_residual=residual)
    return pred.cnr_efficiency.mean(axis=0)


def over_velocity(protocol, angle_deg):
    """Return efficiency on the velocity grid, (velocity, draw), with velocity swept rather than drawn."""
    velocity = np.repeat(VELOCITIES, N_DRAWS)
    flow = tilted(protocol.slab(), angle_deg, velocity)
    return efficiency(protocol, flow, tile(tissues["arterial_blood"], len(VELOCITIES)),
                      tile(tissues["hypodermis"], len(VELOCITIES)),
                      np.tile(others["background_structure_rel"], len(VELOCITIES)),
                      np.tile(others["sat_band_pulse_residual"], len(VELOCITIES))
                      ).reshape(len(VELOCITIES), N_DRAWS)


def over_angle(protocol):
    """Return efficiency against vessel orientation, (angle, draw), with velocity drawn as usual."""
    velocity = others["target_vessel_velocity"] * others["thermoregulatory_factor"]
    return np.array([efficiency(protocol, tilted(protocol.slab(), a, velocity),
                                tissues["arterial_blood"], tissues["hypodermis"],
                                others["background_structure_rel"],
                                others["sat_band_pulse_residual"]) for a in ANGLES_DEG])


fig = plt.figure(figsize=(15.0, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], left=0.055, right=0.985, top=0.80,
                      bottom=0.14, wspace=0.30)

# --- (a) the plane --------------------------------------------------------------
grid = np.array([np.median(over_velocity(OPERATING, a), axis=-1) for a in ANGLES_DEG])
A = fig.add_subplot(gs[0, 0])
# Sequential, over the range the data actually occupy: the operating point never
# changes sign, so a diverging scale centred on zero would spend half its range
# on values that do not occur. Panel (b) is where the sign matters.
mesh = A.pcolormesh(VELOCITIES, ANGLES_DEG, grid, cmap="magma_r", shading="gouraud")
A.contour(VELOCITIES, ANGLES_DEG, grid, levels=np.round(np.linspace(grid.min(), grid.max(), 6), 2),
          colors=[INK], linewidths=0.6, alpha=0.45)
A.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, 90), yticks=[0, 30, 60, 90])
A.set_xticks([5, 10, 20, 50])
A.set_xticklabels(["5", "10", "20", "50"])
A.minorticks_off()  # the log minor ticks relabel the axis on top of the ticks set above
A.set_xlabel("mean velocity (mm/s)", color=INK_2, fontsize=10)
A.set_ylabel("vessel angle to the slab normal (deg)", color=INK_2, fontsize=10)
A.set_title("a   Where the operating point earns contrast", fontsize=11, color=INK, loc="left")
bar = fig.colorbar(mesh, ax=A, pad=0.02)
bar.set_label("CNR efficiency (per sqrt s)", color=INK_2, fontsize=8.5)
bar.ax.tick_params(colors=INK_2, labelsize=7.5)

# --- (b) the mechanisms ---------------------------------------------------------
B = fig.add_subplot(gs[0, 1])
arms = {label: over_angle(ALL_PROTOCOLS[name]) for name, label in CONTRAST_ARMS.items()}
for (label, values), colour in zip(arms.items(), SERIES):
    band(B, ANGLES_DEG, values, colour, label)
B.axhline(0.0, color=INK, lw=1.0, ls="--")
B.annotate("vessel darker than fat", (1, 0), xytext=(2, -11), textcoords="offset points",
           fontsize=8, color=INK_2)
B.set(xlim=(0, 90), xticks=[0, 30, 60, 90])
B.set_xlabel("vessel angle to the slab normal (deg)", color=INK_2, fontsize=10)
B.set_ylabel("CNR efficiency (per sqrt s)", color=INK_2, fontsize=10)
B.set_title("b   Fat saturation decides the sign", fontsize=11, color=INK, loc="left")
B.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower left", borderaxespad=0.2)
style(B)

# --- (c) the free knob ----------------------------------------------------------
C = fig.add_subplot(gs[0, 2])
flips = sorted({*CANDIDATE_FLIPS_DEG, OPERATING.flip_deg})
velocity = others["target_vessel_velocity"] * others["thermoregulatory_factor"]
by_flip = np.array([
    efficiency(ALL_PROTOCOLS.get(f"flip_{f:g}deg", OPERATING) if f != OPERATING.flip_deg else OPERATING,
               tilted(OPERATING.slab(), HAEMODYNAMICS["vessel_angle_to_slab_normal"].nominal, velocity),
               tissues["arterial_blood"], tissues["hypodermis"],
               others["background_structure_rel"], others["sat_band_pulse_residual"])
    for f in flips])
band(C, np.array(flips), by_flip, SERIES[0], "fat-saturated TOF")
best = flips[int(np.argmax(np.median(by_flip, axis=-1)))]
C.axvline(OPERATING.flip_deg, color=INK_2, lw=1.1, ls="--")
C.annotate(f"used {OPERATING.flip_deg:g}°", (OPERATING.flip_deg, C.get_ylim()[1]), xytext=(4, -12),
           textcoords="offset points", fontsize=8.5, color=INK_2, va="top")
C.plot([best], [np.median(by_flip[flips.index(best)])], marker="o", ms=8, color=SERIES[0], zorder=5)
C.annotate(f"best {best:g}°", (best, np.median(by_flip[flips.index(best)])), xytext=(6, 6),
           textcoords="offset points", fontsize=9, color=SERIES[0], weight="bold")
C.set(xlim=(min(flips) - 1, max(flips) + 1))
C.set_xlabel("flip angle (deg)", color=INK_2, fontsize=10)
C.set_ylabel("CNR efficiency (per sqrt s)", color=INK_2, fontsize=10)
C.set_title("c   The flip the protocol could change", fontsize=11, color=INK, loc="left")
C.grid(color=GRID, lw=0.6)
style(C)

print(f"CNR efficiency at the nominal vessel angle "
      f"({HAEMODYNAMICS['vessel_angle_to_slab_normal'].nominal:g} deg to the slab normal):")
for label, values in arms.items():
    at_nominal = values[int(np.argmin(np.abs(ANGLES_DEG - 75.0)))]
    print(f"  {label:<34} {summary(at_nominal)}")
gain = np.median(by_flip[flips.index(best)]) / np.median(by_flip[flips.index(OPERATING.flip_deg)])
print(f"\nFlip angle: the efficiency optimum is {best:g} deg; the {OPERATING.flip_deg:g} deg used "
      f"sits within {max(abs(1 - 1 / gain), 0.001):.1%} of it, so the flip is already close to right")
end_on, in_plane = np.median(over_velocity(OPERATING, 0.0)), np.median(over_velocity(OPERATING, 90.0))
print(f"Orientation: end-on {end_on:.3f} vs in-plane {in_plane:.3f} "
      f"({end_on / in_plane:.2f}x) - the inflow gain, and the reason orientation matters more than velocity")
print(f"NOTE: the in-plane limit depends on MAX_PATH_MM = {MAX_PATH_MM:g} mm, the unjustified "
      f"lateral-extent cap in geometry.path_length, which must be swept")

fig.suptitle("Figure 4 — CNR efficiency: which contrast mechanism, and at which flip",
             fontsize=12.5, color=INK, x=0.055, ha="left")
save(fig, "fig4_efficiency", n=N_DRAWS, seed=SEED)
