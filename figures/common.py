"""Shared figure machinery: the parameter draw, the vessel geometry, bands and reproducible saving.

Every physical parameter comes from protocols.py and tissues.py. The draw is a
Latin hypercube with a fixed seed, recorded on every figure and printout. No
measured data appears in any figure.
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
from protocols import Protocol  # noqa: E402
from sequences.base import FlowGeometry  # noqa: E402
from skin import Vessel  # noqa: E402
from tissues import HAEMODYNAMICS, PARAM_FIELDS, TissueSample, sample_tissues  # noqa: E402

SEED, N_DRAWS = 20261014, 128
TISSUE_KEYS = ["arterial_blood", "dermis", "hypodermis", "muscle"]
HAEMO_KEYS = ("target_vessel_velocity", "thermoregulatory_factor")
ACQ_KEYS = ("sat_band_pulse_residual", "grappa_r2_g_factor")
N_SHELLS = 8  # numerical resolution of the laminar profile, not a physical parameter
MAX_PATH_MM = 30.0  # unjustified cap (acquisition.path_length); inactive for vessels crossing the slab
DERMIS_DEPTH_MM, HYPODERMIS_DEPTH_MM = 1.0, 5.0  # one vessel depth inside each background layer

# Reference categorical palette (dataviz skill, light mode), slots 1-6, and neutral ink.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def draw(n: int = N_DRAWS, seed: int = SEED) -> tuple[dict[str, TissueSample], dict[str, np.ndarray]]:
    """Return the joint Latin-hypercube draw of tissues, haemodynamics and hardware uncertainties."""
    return sample_tissues(TISSUE_KEYS, n, seed, haemodynamic_keys=HAEMO_KEYS, acquisition_keys=ACQ_KEYS)


def vessel_velocity(others: dict[str, np.ndarray]) -> np.ndarray:
    """Return the drawn mean vessel velocity, target velocity times the thermoregulatory factor."""
    return others["target_vessel_velocity"] * others["thermoregulatory_factor"]


def along_leg(protocol: Protocol, depth_mm: float, velocity_mm_s, azimuth_deg: float = 0.0) -> FlowGeometry:
    """Return a vessel running along the leg (in the skin plane, azimuth 0 = the declared flow direction) through the protocol's slab."""
    vessel = Vessel(depth_mm=depth_mm, polar_deg=90.0, azimuth_deg=azimuth_deg,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    return FlowGeometry(vessel, protocol.slab(), velocity_mm_s, N_SHELLS, MAX_PATH_MM)


def tilted(slab: Slab, angle_to_normal_deg: float, velocity_mm_s) -> FlowGeometry:
    """Return an in-plane vessel at angle_to_normal_deg from a transversal slab's normal, crossing its centre."""
    vessel = Vessel(depth_mm=HYPODERMIS_DEPTH_MM, polar_deg=90.0, azimuth_deg=angle_to_normal_deg,
                    diameter_mm=HAEMODYNAMICS["target_diameter"].nominal)
    return FlowGeometry(vessel, slab, velocity_mm_s, N_SHELLS, MAX_PATH_MM, entry_z_frac=0.5)


def band(ax: plt.Axes, x: np.ndarray, values: np.ndarray, colour: str, label: str, **line) -> None:
    """Plot the median over draws (last axis) as a line and the 5-95 % range as a band."""
    with np.errstate(invalid="ignore"):  # draws whose requirement is infinite (texture wins) are allowed
        lo, mid, hi = np.percentile(values, [5, 50, 95], axis=-1)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.plot(x, mid, color=colour, lw=2, label=label, **line)


def summary(values: np.ndarray) -> str:
    """Return 'median [5-95 %]' of draws along the last axis."""
    with np.errstate(invalid="ignore"):
        lo, mid, hi = np.percentile(values, [5, 50, 95], axis=-1)
    return f"{mid:.3g} [{lo:.3g}-{hi:.3g}]"


def style(ax: plt.Axes) -> None:
    """Apply recessive grid and axes."""
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def drawn_parameters(tissues: dict[str, TissueSample], others: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return every drawn parameter that actually varies, keyed '<tissue>.<field>' or by bare name."""
    out = {f"{k}.{f}": np.asarray(getattr(t, f)) for k, t in tissues.items() for f in PARAM_FIELDS}
    out |= {k: np.asarray(v) for k, v in others.items()}
    return {k: v for k, v in out.items() if np.ndim(v) == 1 and np.ptp(v) > 0}


def report_dominance(label: str, outcome: np.ndarray, parameters: dict[str, np.ndarray], top: int = 4) -> None:
    """Print the parameters whose draws rank-correlate most strongly with an outcome (Spearman)."""
    from scipy.stats import spearmanr

    rho = {k: spearmanr(v, outcome).statistic for k, v in parameters.items()}
    ranked = sorted(rho.items(), key=lambda kv: -abs(np.nan_to_num(kv[1])))[:top]
    noise = 2 / np.sqrt(len(outcome) - 1)
    print(f"  dominant for {label} (|rho| < {noise:.2f} is sampling noise): "
          + ", ".join(f"{k} (rho {r:+.2f})" for k, r in ranked))


def label_grid(axes: np.ndarray, xlabel: str, ylabel: str) -> None:
    """Label the bottom row's x axes and the left column's y axes of a 2-D grid of panels."""
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK_2)


def save(fig: plt.Figure, name: str, n: int = N_DRAWS, seed: int = SEED, note: str | None = None) -> Path:
    """Save the figure reproducibly with the draw recorded on it and in its metadata.

    ``note`` overrides the default provenance line, for figures built on nominal
    parameters rather than a Latin-hypercube draw.
    """
    if note is None:
        note = f"Literature parameters only, no measured data. LHS seed {seed}, {n} draws; bands 5-95 %, lines median."
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    fig.text(0.01, 0.005, note, fontsize=7, color=INK_2, ha="left", va="bottom")
    out = ROOT / "figures" / f"{name}.png"
    fig.savefig(out, dpi=200, metadata={"Date": None, "Description": note})
    print(f"\nSaved {out.relative_to(ROOT)}  ({note})")
    return out
