"""Abstract figure 1: what the simulation says the protocol should be (Q3, Q4).

Two panels at the achievable TR. Left: flowing-blood signal and the hypodermis
background with fat saturation on and off, showing that fat saturation moves the
background alone. Right: absolute CNR per sqrt(scan time), the quantity that
decides the protocol once the background is near the noise floor, with the
optimal flip angle and the inherited 18 deg marked.

Run: python figures/fig_abstract_1_protocol.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from common import (DERMIS_DEPTH_MM, HYPODERMIS_DEPTH_MM, INK, INK_2, SERIES, along_leg, band, draw,
                    drawn_parameters, plt, report_dominance, save, style, summary, vessel_velocity)

from analysis import cnr_efficiency
from protocols import CANDIDATES, MEASURED, Protocol

BASE = MEASURED["13oct_0.35mm"]
FATSAT_OFF = CANDIDATES["fatsat_off"]
FLIPS_DEG = np.arange(2.0, 91.0, 2.0)
MID_SLAB = 0.5
BACKGROUNDS = {"dermis": DERMIS_DEPTH_MM, "hypodermis": HYPODERMIS_DEPTH_MM}


def sweep(protocol: Protocol, tissues, others) -> dict[str, np.ndarray]:
    """Return flowing-blood and background signals as (flips, draws)."""
    flow = along_leg(protocol, HYPODERMIS_DEPTH_MM, vessel_velocity(others))
    path = np.linspace(0.0, flow.path_mm(), 40)
    rows = []
    for flip in FLIPS_DEG:
        seq = replace(protocol, flip_deg=flip).sequence(others["sat_band_pulse_residual"])
        row = {"flowing": seq.blood_signal(tissues["arterial_blood"], flow, path).mean(axis=0)}
        row |= {name: seq.static_signal(tissues[name], MID_SLAB) for name in BACKGROUNDS}
        rows.append(row)
    shape = rows[0]["flowing"].shape
    return {k: np.stack([np.broadcast_to(r[k], shape) for r in rows]) for k in rows[0]}


def efficiency(protocol: Protocol, s: dict[str, np.ndarray], background: str) -> np.ndarray:
    """Return |absolute CNR per sqrt(time)| against one background, (flips, draws)."""
    return np.abs(cnr_efficiency(s["flowing"], s[background], protocol.sequence().time_per_line_ms()))


def plateau(eff: np.ndarray, within: float = 0.9) -> tuple[float, float]:
    """Return the flip-angle range over which the median CNR stays within `within` of its peak."""
    median = np.median(eff, axis=-1)
    ok = FLIPS_DEG[median >= within * median.max()]
    return float(ok.min()), float(ok.max())


def cost_of(flip_deg: float, eff: np.ndarray) -> float:
    """Return the median CNR at a flip angle as a fraction of the median peak."""
    median = np.median(eff, axis=-1)
    return float(median[int(np.argmin(np.abs(FLIPS_DEG - flip_deg)))] / median.max())


def mark_optimum(ax, eff: np.ndarray, colour: str) -> None:
    """Shade where the median CNR is within 10 % of its peak and draw the peak."""
    lo, hi = plateau(eff)
    mid = FLIPS_DEG[int(np.argmax(np.median(eff, axis=-1)))]
    ax.axvspan(lo, hi, color=colour, alpha=0.07, lw=0)
    ax.axvline(mid, color=colour, lw=1.3, alpha=0.9)
    ax.annotate(f"peak {mid:.0f} deg\nwithin 10 % over {lo:.0f}-{hi:.0f} deg",
                (mid, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points",
                fontsize=7.5, color=colour, ha="left", va="top")


def main() -> None:
    """Draw the parameter sample, sweep both protocols, plot and print."""
    tissues, others = draw()
    on, off = sweep(BASE, tissues, others), sweep(FATSAT_OFF, tissues, others)
    params = drawn_parameters(tissues, others)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    ax = axes[0]
    band(ax, FLIPS_DEG, on["flowing"], SERIES[0], "blood, flowing")
    band(ax, FLIPS_DEG, on["hypodermis"], SERIES[1], "hypodermis, fat sat on", ls="-")
    band(ax, FLIPS_DEG, off["hypodermis"], SERIES[3], "hypodermis, fat sat off", ls="--")
    ax.axvline(BASE.flip_deg, color=INK_2, lw=1, ls=":")
    ax.set_title("a  Signal: fat saturation moves only the background", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Nominal flip angle (deg)", color=INK_2)
    ax.set_ylabel("Echo signal (units of dermis M0)", color=INK_2)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left")
    ax.set_xlim(0, 90)
    style(ax)

    ax = axes[1]
    series = [("hypodermis, fat sat on", on, "hypodermis", BASE, SERIES[1]),
              ("hypodermis, fat sat off", off, "hypodermis", FATSAT_OFF, SERIES[3]),
              ("dermis, fat sat on", on, "dermis", BASE, SERIES[2])]
    for label, s, background, protocol, colour in series:
        eff = efficiency(protocol, s, background)
        band(ax, FLIPS_DEG, eff, colour, label)
        best = FLIPS_DEG[np.argmax(eff, axis=0)]
        lo, hi = plateau(eff)
        print(f"{label}: optimal flip {summary(best)} deg; median curve within 10 % of peak over "
              f"{lo:.0f}-{hi:.0f} deg; at the acquired {BASE.flip_deg:g} deg you keep "
              f"{100 * cost_of(BASE.flip_deg, eff):.0f} % of the peak CNR per sqrt(t)")
        report_dominance("the optimal flip", best, params)
        if background == "hypodermis" and protocol is BASE:
            mark_optimum(ax, eff, INK)
    ax.axvline(BASE.flip_deg, color=INK_2, lw=1.2, ls=":")
    ax.annotate("acquired 18 deg", (BASE.flip_deg, ax.get_ylim()[1]), xytext=(3, -10),
                textcoords="offset points", fontsize=7.5, color=INK_2, ha="left", va="top")
    ax.set_title("b  The quantity that decides the protocol", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Nominal flip angle (deg)", color=INK_2)
    ax.set_ylabel("Absolute CNR per sqrt(second)", color=INK_2)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower right")
    ax.set_xlim(0, 90)
    style(ax)

    i = int(np.argmin(np.abs(FLIPS_DEG - BASE.flip_deg)))
    blood_term = on["flowing"][i] - off["flowing"][i]
    bg_term = off["hypodermis"][i] - on["hypodermis"][i]
    print(f"at {BASE.flip_deg:g} deg, hypodermis: fat sat changes blood by {summary(blood_term)} "
          f"and background by {summary(bg_term)} M0")

    fig.suptitle(f"fl3d1r TOF at TR {BASE.timing.tr_ms:.2f} ms, vessel at {HYPODERMIS_DEPTH_MM:g} mm depth",
                 fontsize=11, color=INK)
    save(fig, "fig_abstract_1_protocol")


if __name__ == "__main__":
    main()
