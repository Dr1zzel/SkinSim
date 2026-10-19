"""Trace one draw through the whole Figure 1b pipeline, printing every intermediate.

Companion to docs/figure1b-walkthrough.md. The walkthrough quotes numbers; this
prints them, so the document can be checked against the code rather than
believed. Nothing here is used to build the figure - it re-derives the same
quantities by hand, step by step, from the same modules.

Run:
    python figures/trace_fig1b.py
    python figures/trace_fig1b.py --draw 42 --velocity 5
"""
from __future__ import annotations

import argparse

import numpy as np
from common import N_SHELLS, SEED  # same draw as the figure

from geometry import Slab, Vessel, laminar_velocities
from sequence import FlowGeometry, SaturationModule, TofSequence, between_excitations, excitation_map, pulses_received, saturation_map
from affine import apply, compose, fixed_point, power, relax_map
from protocols import BANDWIDTH_PER_PX_HZ, PARTITION_MM, PRISMA_XR, RF_DURATION_MS, fl3d1r_timing
from tissues import PARAM_FIELDS, TISSUES, TissueSample, sample_tissues, signal_weighting

TR_MS, FLIP_DEG, SLAB_MM, INPLANE_MM = 35.0, 20.0, 12.0, 0.3
N_DRAWS, N_POSITIONS = 96, 32
KEYS = ("arterial_blood", "hypodermis", "dermis")


def head(title: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def one_draw(index: int) -> dict[str, TissueSample]:
    """Return the tissues of a single draw of the figure's own Latin hypercube."""
    tissues, _ = sample_tissues(list(KEYS), N_DRAWS, SEED,
                                haemodynamic_keys=("target_vessel_velocity", "thermoregulatory_factor"))
    return {k: TissueSample(k, **{f: float(np.atleast_1d(getattr(tissues[k], f))[index])
                                  for f in PARAM_FIELDS}) for k in KEYS}


def show_intervals() -> None:
    """Print the parameter table that the draw is taken from."""
    head("STEP 1  tissues.py: the table holds intervals, not numbers")
    for key in KEYS:
        for f in ("t1_ms", "t2star_ms", "pd_rel", "fatsat_s", "mtc_s"):
            p = getattr(TISSUES[key], f)
            print(f"  {key + '.' + f:<28} {p.lo:>8.4g} .. {p.hi:<8.4g} "
                  f"nominal {p.nominal:<8.4g} {'log' if p.log else 'lin':>3}  {p.confidence.value}")
    print("\n  `nominal` is never read by a figure. The interval is what gets sampled.")


def show_draw(samples: dict[str, TissueSample], index: int) -> None:
    """Print the one coherent world this draw represents."""
    head(f"STEP 2  the joint Latin hypercube: draw #{index} of {N_DRAWS}, seed {SEED}")
    print(f"  {'tissue':<16}{'T1 (ms)':>10}{'T2* (ms)':>10}{'PD':>8}{'fatsat_s':>10}{'mtc_s':>8}")
    for k, s in samples.items():
        print(f"  {k:<16}{s.t1_ms:>10.1f}{s.t2star_ms:>10.2f}{s.pd_rel:>8.3f}"
              f"{s.fatsat_s:>10.3f}{s.mtc_s:>8.3f}")
    print("\n  One hypercube covers every parameter at once, so this blood sits beside")
    print("  ITS OWN dermis and fat. That is what makes the paired statistics valid.")


def show_timing():
    """Print TR and TE assembled from named components, and return the timing."""
    head("STEP 3  protocols.py: TR and TE are built from components, not assumed")
    t = fl3d1r_timing(INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS, TR_MS, PRISMA_XR)
    for name, value in (("RF", t.rf_duration_ms), ("slice rephaser", t.rephaser_ms),
                        ("read prephaser", t.prephaser_ms), ("readout", t.readout_ms),
                        ("spoiler", t.spoiler_ms), ("dead time", t.dead_time_ms)):
        print(f"    {name:<16} {value:7.3f} ms")
    print(f"    {'':<16} {'-' * 10}\n    {'TR':<16} {t.tr_ms:7.3f} ms")
    print(f"    {'TE':<16} {t.te_ms:7.3f} ms   = RF/2 + rephaser + prephaser + readout/2")
    print(f"\n  dead time is {t.dead_time_ms / t.tr_ms:.0%} of TR: the slack a lower readout bandwidth would use")
    return t


def show_geometry() -> FlowGeometry:
    """Print the slab and vessel geometry and return the flow."""
    head("STEP 4  geometry: the vessel CROSSES the slab, it does not lie in it")
    slab = Slab(thickness_mm=SLAB_MM, orientation_deg=90.0)
    vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
    dot = abs(float(np.dot(vessel.direction(), slab.normal())))
    print(f"  slab normal       {np.round(slab.normal(), 3)}   (orientation 90 deg)")
    print(f"  vessel direction  {np.round(vessel.direction(), 3)}   (polar 90, azimuth 0)")
    print(f"  |direction . normal| = {dot:.3f}  ->  path = thickness / that = {SLAB_MM / dot:.3f} mm")
    print("\n  Parallel to the normal = end-on crossing = the SHORTEST path and the")
    print("  whole velocity counted. Best case for inflow, so the result is conservative.")
    return slab, vessel


def show_shells(velocity: float) -> tuple[np.ndarray, np.ndarray]:
    """Print the laminar shells and return velocities and weights."""
    head(f"STEP 5  geometry.py: the lumen is {N_SHELLS} shells, not a plug")
    vel, w = laminar_velocities(velocity, N_SHELLS)
    print(f"  {'shell':>6}{'v (mm/s)':>12}{'area weight':>14}")
    for i, (v, wi) in enumerate(zip(vel, w)):
        tag = "  <- centre" if i == 0 else ("  <- wall" if i == len(vel) - 1 else "")
        print(f"  {i:>6}{v:>12.2f}{wi:>14.4f}{tag}")
    print(f"\n  weights sum to {w.sum():.6f}; weighted mean velocity = {np.sum(vel * w):.4f} mm/s")
    print(f"  the wall shell is the SLOWEST and carries the LARGEST area weight:")
    print(f"  {w[-1]:.0%} of the lumen moves at {vel[-1] / velocity:.2f}x the mean. Plug flow overestimates TOF.")
    return vel, w


def show_affine(blood: TissueSample, timing) -> tuple:
    """Print the one-TR affine map and its fixed point, and return the map."""
    head("STEP 6  affine.py: one TR is a map  Mz -> A*Mz + B")
    relax = between_excitations((), blood, timing.tr_ms)
    m = compose(excitation_map(FLIP_DEG), relax)
    print(f"  relaxation over TR   A = exp(-TR/T1) = {relax[0]:.5f}   B = 1 - A = {relax[1]:.5f}")
    print(f"  excitation           A = cos({FLIP_DEG:g} deg) = {np.cos(np.radians(FLIP_DEG)):.5f}   B = 0")
    print(f"  composed one-TR map  A = {m[0]:.5f}   B = {m[1]:.5f}")
    print(f"  fixed point B/(1-A)  = {fixed_point(m):.5f}   <- STAGNANT blood's Mz/M0")
    print("\n  The engine runs at unit M0: B = 1 - A, not PD*(1 - A).")
    print("  Proton density is applied later, at the echo step, so maps stay tissue-agnostic.")
    print(f"\n  power() needs no special cases:")
    for n in (0, 1, 5, np.inf):
        print(f"    n = {str(n):>4}  ->  Mz from a fully relaxed inflow = {apply(power(m, n), 1.0):.6f}")
    print(f"  n = inf reproduces the fixed point exactly: "
          f"{abs(float(apply(power(m, np.inf), 1.0)) - float(fixed_point(m))) < 1e-12}")
    return m, relax


def show_saturation(fat: TissueSample, blood: TissueSample, timing) -> None:
    """Print how a saturation pulse enters the TR, and what it does to the inflow."""
    head("STEP 7  sequence.py: where a saturation pulse sits, and what it costs")
    delay = 8.0
    coded = between_excitations((SaturationModule("fatsat_s", delay).pulse(fat),), fat, timing.tr_ms)
    manual = compose(relax_map(timing.tr_ms - delay, fat.t1_ms),
                     saturation_map(fat.fatsat_s), relax_map(delay, fat.t1_ms))
    print(f"  one TR is: excite -> relax({timing.tr_ms - delay:.0f} ms) -> saturate(s) -> relax({delay:.0f} ms)")
    print(f"    from the code  A = {coded[0]:.6f}  B = {coded[1]:.6f}")
    print(f"    by hand        A = {manual[0]:.6f}  B = {manual[1]:.6f}   match = {np.allclose(coded, manual)}")
    print("\n  The inflow boundary condition is the fixed point of relaxation + pulses")
    print("  WITHOUT excitation: upstream blood is never excited, but a non-selective")
    print("  saturation pulse still reaches it.")
    for label, module in (("none", None), ("fat saturation", "fatsat_s"), ("MTC", "mtc_s")):
        pulses = () if module is None else (SaturationModule(module, delay).pulse(blood),)
        upstream = float(fixed_point(between_excitations(pulses, blood, timing.tr_ms)))
        factor = "-" if module is None else f"{getattr(blood, module):.3f}"
        print(f"    {label:<16} blood factor {factor:>6}  ->  upstream Mz/M0 = {upstream:.4f}")
    recovery = 1 - np.exp(-timing.tr_ms / blood.t1_ms)
    print(f"\n  MTC costs {1 - blood.mtc_s:.3f} per TR against only {recovery:.3f} T1 recovery per TR,")
    print("  so it compounds: blood reaches the slab ALREADY heavily saturated, before")
    print("  any excitation. A non-selective MT pulse trades background suppression")
    print("  against the very inflow that TOF depends on.")


def show_signal(blood: TissueSample, flow: FlowGeometry, timing, velocity: float,
                vel: np.ndarray, m) -> None:
    """Print the pulse count, the echo conversion and the path mean."""
    head("STEP 8  sequence.py: excitations received, then Mz -> echo")
    seq = TofSequence(timing=timing, flip_deg=FLIP_DEG)
    pos = np.linspace(0.0, flow.path_mm(), N_POSITIONS)
    n = pulses_received(pos[:, None], vel[None, :], timing.tr_ms)
    print(f"  one TR advances the centreline by v*TR = {vel[0] * timing.tr_ms / 1000:.4f} mm")
    print(f"  n = floor(distance / step), so a spin 8.7 TRs in counts as 8, never 8.7")
    print(f"  at the exit face ({pos[-1]:.2f} mm):")
    for i in (0, N_SHELLS // 2, N_SHELLS - 1):
        print(f"    shell {i} (v = {vel[i]:6.2f} mm/s): n = {n[-1, i]:.0f} excitations")
    mz = float(apply(power(m, n[-1, 0]), 1.0))
    w = float(signal_weighting(blood, timing.te_ms))
    print(f"\n  centreline at the exit: Mz/M0 = {mz:.5f}")
    print(f"  signal_weighting = PD * exp(-TE/T2*) = {blood.pd_rel:.4f} * "
          f"exp(-{timing.te_ms:.3f}/{blood.t2star_ms:.2f}) = {w:.5f}")
    print(f"  echo = Mz * sin(flip) * weighting = {mz:.5f} * {np.sin(np.radians(FLIP_DEG)):.5f} * {w:.5f}")
    print(f"       = {mz * np.sin(np.radians(FLIP_DEG)) * w:.5f}")
    print("\n  Dermis has pd_rel = 1.0 BY DEFINITION, which is what '1' on the y axis means.")

    sig = seq.blood_signal(blood, flow, pos)
    head("STEP 9  area-weighted over shells, then averaged along the path")
    print(f"  blood_signal returns one value per position ({N_POSITIONS} of them)")
    print(f"    entry face {sig[0]:.5f}  ->  exit face {sig[-1]:.5f}")
    print(f"  the figure plots the MEAN along the path: {float(np.mean(sig)):.5f}")
    print(f"  that single number is ONE dot on the curve, for this draw at v = {velocity:g} mm/s")
    print("\n  The mean mixes fresh blood at entry with saturated blood at the exit.")
    print("  A MIP would weight the entry end more, so the figure is pessimistic.")

    static = float(seq.static_signal(blood))
    crawl = float(seq.blood_signal(blood, FlowGeometry(flow.vessel, flow.slab, 1e-9, N_SHELLS,
                                                       SLAB_MM * 1.001), np.array([6.0]))[0])
    print(f"\n  CHECK  static_signal = {static:.6f}; blood_signal as v -> 0 = {crawl:.6f}; "
          f"match = {abs(static - crawl) < 1e-6}")


def show_ensemble(flow: FlowGeometry, timing, velocity: float, index: int) -> None:
    """Print how the 96 draws become a median and a band."""
    head("STEP 10  all 96 draws -> median and 5-95 % band")
    tissues, _ = sample_tissues(list(KEYS), N_DRAWS, SEED,
                                haemodynamic_keys=("target_vessel_velocity", "thermoregulatory_factor"))
    seq = TofSequence(timing=timing, flip_deg=FLIP_DEG)
    pos = np.linspace(0.0, flow.path_mm(), N_POSITIONS)
    values = np.array([float(np.mean(seq.blood_signal(
        TissueSample("arterial_blood", **{f: float(np.atleast_1d(getattr(tissues["arterial_blood"], f))[i])
                                          for f in PARAM_FIELDS}), flow, pos))) for i in range(N_DRAWS)])
    lo, mid, hi = np.percentile(values, [5, 50, 95])
    print(f"  at v = {velocity:g} mm/s across {N_DRAWS} draws:")
    print(f"    median {mid:.4f}   5 % {lo:.4f}   95 % {hi:.4f}")
    print(f"  draw #{index} gave {values[index]:.4f}, the {100 * np.mean(values < values[index]):.0f}th percentile")
    print(f"\n  The orange line is that median at each of 64 velocities; the band is 5-95 %.")
    print("  Everything else on the figure is static_signal on a tissue, same machinery")
    print("  with n -> infinity.")


def main() -> None:
    """Trace one draw at one velocity through every stage."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", type=int, default=0, help="which draw of the hypercube (0-95)")
    ap.add_argument("--velocity", type=float, default=20.0, help="mean blood velocity, mm/s")
    args = ap.parse_args()

    samples = one_draw(args.draw)
    show_intervals()
    show_draw(samples, args.draw)
    timing = show_timing()
    slab, vessel = show_geometry()
    vel, _ = show_shells(args.velocity)
    m, _ = show_affine(samples["arterial_blood"], timing)
    show_saturation(samples["hypodermis"], samples["arterial_blood"], timing)
    flow = FlowGeometry(vessel, slab, args.velocity, N_SHELLS, SLAB_MM * 1.001)
    show_signal(samples["arterial_blood"], flow, timing, args.velocity, vel, m)
    show_ensemble(flow, timing, args.velocity, args.draw)
    print("\n" + "=" * 74)
    print("Full prose version: docs/figure1b-walkthrough.md")
    print("=" * 74)


if __name__ == "__main__":
    main()
