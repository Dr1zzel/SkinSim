"""Blood and background signal against flip angle at the achievable TR/TE, with and without fat saturation (Q3, Q4).

Marks the flip angle maximising absolute CNR per sqrt(time), not the contrast
ratio: in a sensitivity-limited regime the ratio optimises the wrong thing.
Blood: fresh (slab entry), flowing (averaged along a vessel crossing the slab at
the drawn velocity) and saturated (static). Background and saturated blood are
taken at mid-slab, where TONE gives the nominal flip.
Run: python figures/fig_contrast_vs_flip.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from common import (label_grid, DERMIS_DEPTH_MM, HYPODERMIS_DEPTH_MM, INK, INK_2, SERIES, along_leg, band, draw,
                    drawn_parameters, plt, report_dominance, save, style, summary, vessel_velocity)

from analysis import cnr_efficiency
from protocols import CANDIDATES, MEASURED, Protocol

BASE = MEASURED["13oct_0.35mm"]
PROTOCOLS = {"fat sat on (as acquired)": BASE, "fat sat off (candidate)": CANDIDATES["fatsat_off"]}
BACKGROUNDS = {"dermis": DERMIS_DEPTH_MM, "hypodermis": HYPODERMIS_DEPTH_MM}
FLIPS_DEG = np.arange(2.0, 91.0, 1.0)  # plotting axis
MID_SLAB = 0.5
CURVES = {"fresh": ("blood, fresh (slab entry)", SERIES[1]), "flowing": ("blood, flowing (slab mean)", SERIES[0]),
          "saturated": ("blood, saturated (static)", SERIES[2])}


def signals(protocol: Protocol, flip_deg: float, tissues, others) -> dict[str, np.ndarray]:
    """Return each signal over the draws at one flip angle."""
    seq = replace(protocol, flip_deg=flip_deg).sequence(others["sat_band_pulse_residual"])
    flow = along_leg(protocol, HYPODERMIS_DEPTH_MM, vessel_velocity(others))
    blood = tissues["arterial_blood"]
    path = np.linspace(0.0, flow.path_mm(), 40)
    out = {"fresh": seq.blood_signal(blood, flow, [0.0])[0], "flowing": seq.blood_signal(blood, flow, path).mean(axis=0),
           "saturated": seq.static_signal(blood, MID_SLAB)}
    return out | {name: seq.static_signal(tissues[name], MID_SLAB) for name in BACKGROUNDS}


def sweep(protocol: Protocol, tissues, others) -> dict[str, np.ndarray]:
    """Return every signal as (flips, draws)."""
    rows = [signals(protocol, f, tissues, others) for f in FLIPS_DEG]
    return {key: np.stack([np.broadcast_to(r[key], rows[0]["flowing"].shape) for r in rows]) for key in rows[0]}


def efficiency(protocol: Protocol, s: dict[str, np.ndarray], background: str) -> np.ndarray:
    """Return |absolute CNR per sqrt(time)| of flowing blood against the background, (flips, draws)."""
    return np.abs(cnr_efficiency(s["flowing"], s[background], protocol.sequence().time_per_line_ms()))


def panel(ax, protocol: Protocol, s: dict[str, np.ndarray], background: str, title: str) -> np.ndarray:
    """Draw one background's signals, the operating flip and the CNR-optimal flip; return the optimal flip per draw."""
    for key, (label, colour) in CURVES.items():
        band(ax, FLIPS_DEG, s[key], colour, label)
    band(ax, FLIPS_DEG, s[background], INK_2, f"{background} (background)", ls="--")
    best = FLIPS_DEG[np.argmax(efficiency(protocol, s, background), axis=0)]
    lo, mid, hi = np.percentile(best, [5, 50, 95])
    ax.axvspan(lo, hi, color=INK_2, alpha=0.08, lw=0)
    ax.axvline(mid, color=INK, lw=1.2, label="CNR/sqrt(t)-optimal flip (median, 5-95 %)")
    ax.axvline(BASE.flip_deg, color=INK_2, lw=1, ls=":", label=f"acquired {BASE.flip_deg:g} deg")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    style(ax)
    return best


def report_fat_sat(on: dict[str, np.ndarray], off: dict[str, np.ndarray], background: str) -> None:
    """Print Q4 at the acquired flip: absolute CNR change, split into blood and background terms, beside the ratio."""
    i = int(np.argmin(np.abs(FLIPS_DEG - BASE.flip_deg)))
    blood_term = on["flowing"][i] - off["flowing"][i]
    background_term = off[background][i] - on[background][i]
    gain = blood_term + background_term
    ratio_gain = (on["flowing"][i] / on[background][i]) / (off["flowing"][i] / off[background][i])
    print(f"  {background}: delta(S_blood - S_bg) from fat sat {summary(gain)}; of which blood {summary(blood_term)}, "
          f"background suppression {summary(background_term)}; ratio blood/bg changes x{summary(ratio_gain)}")


def main() -> None:
    """Draw, sweep, plot and print."""
    tissues, others = draw()
    sweeps = {name: sweep(p, tissues, others) for name, p in PROTOCOLS.items()}
    print(f"TR {BASE.timing.tr_ms:.2f} ms, TE {BASE.timing.te_ms:.2f} ms, time per line {BASE.sequence().time_per_line_ms():.2f} ms")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for row, (name, s) in zip(axes, sweeps.items()):
        for ax, background in zip(row, BACKGROUNDS):
            best = panel(ax, PROTOCOLS[name], s, background, f"{name}; vessel in {background}")
            print(f"{name}, {background}: CNR/sqrt(t)-optimal flip {summary(best)} deg")
            report_dominance("the optimal flip", best, drawn_parameters(tissues, others))
    label_grid(axes, "Nominal flip angle (deg)", "Echo signal (units of dermis M0)")
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=INK_2)
    print(f"Q4 at the acquired {BASE.flip_deg:g} deg (fat sat on minus off):")
    for background in BACKGROUNDS:
        report_fat_sat(sweeps["fat sat on (as acquired)"], sweeps["fat sat off (candidate)"], background)
    fig.suptitle(f"fl3d1r TOF at TR {BASE.timing.tr_ms:.2f} ms: signal against flip angle", fontsize=11, color=INK)
    save(fig, "fig_contrast_vs_flip")


if __name__ == "__main__":
    main()
