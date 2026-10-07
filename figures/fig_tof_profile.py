"""First look at TOF: blood/background ratio along an in-plane vessel, per background.

Calls only what exists; no new physics. All tissue values nominal.
Run from anywhere: python figures/fig_tof_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition import Slab  # noqa: E402
from sequences.base import FlowGeometry, ReadoutTiming, TissueSample  # noqa: E402
from sequences.tof import TofSequence  # noqa: E402
from skin import Vessel  # noqa: E402
from tissues import ARTERIAL_BLOOD  # noqa: E402

SLAB = Slab(thickness_mm=10.0, orientation_deg=0.0)
MAX_PATH_MM = 30.0  # no justification; must be swept
N_SHELLS = 16
DEPTHS_MM = (1.0, 5.0)
VELOCITIES_MM_S = (5.0, 15.0, 50.0)
THRESHOLDS = (2.0, 1.5)
POSITION_STEP_MM = 0.01
RATIO_TICKS = (0.25, 0.5, 1, 1.5, 2, 4, 8, 16)
TOF = TofSequence(
    timing=ReadoutTiming(rf_duration_ms=1.0, rephaser_ms=0.5, prephaser_ms=1.0,
                         bandwidth_per_px_hz=250.0, spoiler_ms=1.5, dead_time_ms=0.2),
    flip_deg=20.0,
)
BLOOD = TissueSample.nominal(ARTERIAL_BLOOD)

# Reference palette, light mode: categorical slots 1-3 and text/grid tokens.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def flow(depth_mm: float, velocity_mm_s: float) -> FlowGeometry:
    """Return the flow geometry for an in-plane vessel at depth_mm."""
    vessel = Vessel(depth_mm=depth_mm, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.2)
    return FlowGeometry(vessel, SLAB, velocity_mm_s, N_SHELLS, MAX_PATH_MM)


def ratio_profile(depth_mm: float, velocity_mm_s: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (positions, blood/background ratio, background name) along the vessel."""
    geometry = flow(depth_mm, velocity_mm_s)
    positions = np.arange(0.0, geometry.path_mm() + POSITION_STEP_MM / 2, POSITION_STEP_MM)
    background = TissueSample.nominal(geometry.vessel.background())
    result = TOF.contrast(BLOOD, background, geometry, positions)
    return positions, result.ratio, result.background_name


def saturated_ratio(depth_mm: float) -> float:
    """Return the velocity -> 0 blood/background ratio at depth_mm."""
    background = TissueSample.nominal(flow(depth_mm, 0.0).vessel.background())
    return TOF.static_signal(BLOOD) / TOF.static_signal(background)


def first_below(positions: np.ndarray, ratio: np.ndarray, threshold: float) -> str:
    """Return the first distance at which the ratio is below threshold, or a not-reached note."""
    below = np.flatnonzero(ratio < threshold)
    if below.size == 0:
        return f"not within {positions[-1]:.0f} mm"
    return f"{positions[below[0]]:.2f} mm"


def report() -> None:
    """Print derived timing, saturated ratios and threshold distances."""
    print(f"TR = {TOF.timing.tr_ms:.2f} ms, TE = {TOF.timing.te_ms:.2f} ms "
          f"(readout {TOF.timing.readout_ms:.2f} ms)")
    for depth in DEPTHS_MM:
        _, _, name = ratio_profile(depth, VELOCITIES_MM_S[0])
        print(f"\nVessel at {depth:g} mm, background {name}: "
              f"saturated ratio {saturated_ratio(depth):.3f}")
        for v in VELOCITIES_MM_S:
            positions, ratio, _ = ratio_profile(depth, v)
            cuts = ", ".join(f"< {t:g} at {first_below(positions, ratio, t)}" for t in THRESHOLDS)
            print(f"  {v:>4g} mm/s: entry ratio {ratio[0]:.3f}; {cuts}")


def draw_panel(ax: plt.Axes, depth_mm: float) -> None:
    """Draw one background's ratio profiles, saturated asymptote and ratio = 1 line."""
    for v, colour in zip(VELOCITIES_MM_S, SERIES):
        positions, ratio, name = ratio_profile(depth_mm, v)
        ax.plot(positions, ratio, color=colour, lw=2, label=f"{v:g} mm/s")
    sat = saturated_ratio(depth_mm)
    ax.axhline(sat, color=TEXT_2, lw=1.2, ls="--", label=f"saturated (v → 0): {sat:.2f}")
    ax.axhline(1.0, color=TEXT_2, lw=0.8, ls=":", label="ratio = 1")
    ax.set_title(f"Vessel at {depth_mm:g} mm — {name} background", fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("Distance along vessel from slab entry (mm)", color=TEXT_2)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)


def main() -> None:
    """Print the numbers and save the two-panel figure."""
    report()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, depth in zip(axes, DEPTHS_MM):
        draw_panel(ax, depth)
        ax.set_xlim(0, MAX_PATH_MM)
        ax.set_yscale("log")
        ax.set_yticks(RATIO_TICKS, [f"{t:g}" for t in RATIO_TICKS])
        ax.minorticks_off()
        ax.tick_params(colors=TEXT_2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Blood / background signal (log scale)", color=TEXT_2)
    fig.suptitle(f"TOF, {SLAB.thickness_mm:g} mm slab parallel to skin, in-plane vessel, "
                 f"flip {TOF.flip_deg:g}°, TR {TOF.timing.tr_ms:.1f} ms", fontsize=11, color=TEXT)
    out = ROOT / "figures" / "fig_tof_profile.png"
    fig.savefig(out, dpi=200, metadata={"Date": None})
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
