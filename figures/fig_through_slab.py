"""Signal against through-slab position for static tissue and flowing blood, TONE 0 / 70 %, both band positions (Q5, Q6).

The directly measurable prediction. Geometry as acquired: transversal slab on
the leg, a vessel along the leg crossing it in the declared flow direction, so
distance along the vessel equals through-slab position. Blood lines are at the
low, nominal and high target velocity; bands cover the tissue and hardware
draws. Band 'downstream' is the acquired position for this flow direction;
'upstream' is the acquired opposite-band series.
Run: python figures/fig_through_slab.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from common import (label_grid, HYPODERMIS_DEPTH_MM, INK, INK_2, SERIES, along_leg, band, draw, plt, save, style, summary,
                    tilted)

from protocols import MEASURED, Protocol
from tissues import HAEMODYNAMICS

BASE = MEASURED["13oct_0.35mm"]
BAND_POSITIONS = {"band downstream (as acquired)": BASE, "band upstream (opposite series)": MEASURED[
    "13oct_0.35mm_band_opposite"]}
TONES_PCT = (0.0, BASE.tone_pct)
V = HAEMODYNAMICS["target_vessel_velocity"]
VELOCITIES_MM_S = (V.lo, V.nominal, V.hi)
ANGLES_TO_NORMAL_DEG = (0.0, 30.0, 60.0, 80.0, 90.0)
Z_MM = np.linspace(0.0, BASE.slab_thickness_mm, 60)
STATIC = {"dermis": ("dermis (static)", "--"), "hypodermis": ("fat, suppressed (static)", ":")}


def profiles(protocol: Protocol, tone_pct: float, tissues, others) -> dict[str, np.ndarray]:
    """Return static and blood signals against through-slab position, (positions, draws)."""
    seq = replace(protocol, tone_pct=tone_pct).sequence(others["sat_band_pulse_residual"])
    z_frac = (Z_MM / protocol.slab_thickness_mm)[:, None]
    out = {key: np.zeros_like(z_frac) + seq.static_signal(tissues[key], z_frac) for key in STATIC}  # TONE 0: no z
    for v in VELOCITIES_MM_S:
        out[v] = seq.blood_signal(tissues["arterial_blood"], along_leg(protocol, HYPODERMIS_DEPTH_MM, v), Z_MM)
    return out


def panel(ax, p: dict[str, np.ndarray], title: str) -> None:
    """Draw one TONE and band combination."""
    for key, (label, ls) in STATIC.items():
        band(ax, Z_MM, p[key], INK_2, label, ls=ls)
    for v, colour in zip(VELOCITIES_MM_S, SERIES):
        band(ax, Z_MM, p[v], colour, f"blood, {v:g} mm/s")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    style(ax)


def report_tone(p0: dict[str, np.ndarray], p70: dict[str, np.ndarray]) -> None:
    """Print Q5: exit / entry signal for static tissue (TONE alone) and blood (inflow, with and without TONE)."""
    ratio = lambda s: s[-1] / s[0]  # noqa: E731
    for key in STATIC:
        print(f"  {key}: exit/entry TONE 0 {summary(ratio(p0[key]))}, TONE {TONES_PCT[1]:g} {summary(ratio(p70[key]))}")
    for v in VELOCITIES_MM_S:
        print(f"  blood {v:g} mm/s: exit/entry TONE 0 {summary(ratio(p0[v]))}, TONE {TONES_PCT[1]:g} "
              f"{summary(ratio(p70[v]))}")


def report_band(tissues, others) -> None:
    """Print Q6: blood signal with the band downstream minus upstream, slab mean, by velocity and angle to the normal."""
    for angle in ANGLES_TO_NORMAL_DEG:
        cells = []
        for v in VELOCITIES_MM_S:
            means = []
            for protocol in BAND_POSITIONS.values():
                flow = tilted(protocol.slab(), angle, v)
                seq = protocol.sequence(others["sat_band_pulse_residual"])
                means.append(seq.blood_signal(tissues["arterial_blood"], flow,
                                              np.linspace(0, flow.path_mm(), 40)).mean(axis=0))
            cells.append(f"{v:g} mm/s {summary(means[0] - means[1])}")
        print(f"  {angle:4g} deg to the slab normal: " + "; ".join(cells))


def main() -> None:
    """Draw, compute, plot and print."""
    tissues, others = draw()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    results = {}
    for row, tone in zip(axes, TONES_PCT):
        for ax, (name, protocol) in zip(row, BAND_POSITIONS.items()):
            results[tone, name] = profiles(protocol, tone, tissues, others)
            panel(ax, results[tone, name], f"TONE {tone:g} %; {name}")
    label_grid(axes, "Through-slab position from the inflow face (mm)", "Echo signal (units of dermis M0)")
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=INK_2)
    first = next(iter(BAND_POSITIONS))
    print("Q5, band downstream: exit/entry signal ratio, median [5-95 %]")
    report_tone(results[TONES_PCT[0], first], results[TONES_PCT[1], first])
    print("Q6: slab-mean blood signal, band downstream minus upstream, TONE as acquired")
    report_band(tissues, others)
    fig.suptitle("fl3d1r TOF through the slab: TONE and travelling-band position", fontsize=11, color=INK)
    save(fig, "fig_through_slab")


if __name__ == "__main__":
    main()
