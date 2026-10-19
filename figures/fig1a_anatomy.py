"""Figure 1a: the cutaneous vascular tree, drawn to scale in depth, with the target calibre on it.

The anatomical cartoon that usually opens a skin-imaging paper carries no
numbers, so it cannot say why the problem is hard. This one is dimensioned:

  - depth is to scale, so the 0.1 mm epidermis is a hairline against the 12 mm
    slab and the reader can see for themselves why it is not modelled;
  - every level of the vascular tree is labelled with its measured calibre,
    from 10 um capillary loops to the ~1.9 mm perforators;
  - the target band sits between the dermal plexus ceiling (~120 um) and the
    perforators, which is the gap no in vivo imaging study resolves;
  - a true-scale calibre key puts the target vessels beside the voxel, which is
    the whole detectability argument in one glance.

Layer boundaries come from geometry.LAYERS and the calibres from
geometry.PLEXUS, so this figure cannot drift from the model. Vessel line widths
are exaggerated and labelled as such; DEPTHS are to scale, calibres are not.

Run: python figures/fig1a_anatomy.py
"""
from __future__ import annotations

import numpy as np
from common import GRID, INK, INK_2, plt, save
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from geometry import LAYERS, PLEXUS, TARGET_DEPTH_MM
from protocols import MEASURED

# Arterial / venous convention, muted to sit under the annotation layer.
ART, VEN = "#c0392b", "#2b6ca3"
LAYER_FILL = {"epidermis": "#f0d7b4", "dermis": "#fbe9e7",
              "hypodermis (subcutaneous fat)": "#fdf3cd", "muscle": "#e5cfcf"}

DEPTH_MM = 12.0          # drawn depth, matched to the slab in fig1b
WIDTH_MM = 17.0          # lateral extent of the cartoon, arbitrary
ZOOM_MM = 2.0            # the inset magnifies 0 .. ZOOM_MM
VOXEL_MM = MEASURED["13oct_0.35mm"].inplane_mm
TARGET = next(v for v in PLEXUS if v.name == "subcutaneous vessels")


def layer_rows() -> list[tuple[str, float, float]]:
    """Return (name, top_mm, bottom_mm) for every layer, the last one clipped to the drawn depth."""
    rows, top = [], 0.0
    for bottom, tissue in LAYERS:
        rows.append((tissue.name, top, min(bottom, DEPTH_MM)))
        if bottom >= DEPTH_MM:
            break
        top = bottom
    return rows


def draw_layers(ax, width_mm: float, depth_mm: float, label: bool = True) -> None:
    """Fill each skin layer as a tinted band with its boundary drawn and, optionally, named."""
    for name, top, bottom in layer_rows():
        if top >= depth_mm:
            break
        bottom = min(bottom, depth_mm)
        ax.add_patch(Rectangle((0, top), width_mm, bottom - top, facecolor=LAYER_FILL[name],
                               edgecolor="none", zorder=0))
        ax.plot([0, width_mm], [bottom, bottom], color="#b9b4ac", lw=0.8, zorder=1)
        if label:
            full = next(b for b, tis in LAYERS if tis.name == name)
            span = f"{top:g}–{bottom:g} mm" if np.isfinite(full) else f"below {top:g} mm"
            ax.annotate(f"{name.split(' (')[0]}\n{span}", (-0.35, (top + bottom) / 2),
                        ha="right", va="center", fontsize=8, color=INK_2, linespacing=1.35)


def wavy(ax, x0: float, x1: float, depth: float, amp: float, cycles: float, colour: str,
         lw: float, phase: float = 0.0, zorder: int = 3) -> None:
    """Draw one roughly horizontal vessel as a sine path at a given depth."""
    x = np.linspace(x0, x1, 240)
    y = depth + amp * np.sin(2 * np.pi * cycles * (x - x0) / max(x1 - x0, 1e-9) + phase)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", zorder=zorder)


def ascending(ax, x: float, top: float, bottom: float, colour: str, lw: float, bow: float = 0.0,
              zorder: int = 3) -> None:
    """Draw a vessel climbing between two plexus levels, bowed sideways so it does not read as a ruler."""
    t = np.linspace(0, 1, 120)
    ax.plot(x + bow * np.sin(np.pi * t), bottom + (top - bottom) * t, color=colour, lw=lw,
            solid_capstyle="round", zorder=zorder)


def capillary_loops(ax, x0: float, x1: float, n: int, crest: float, base: float) -> None:
    """Draw hairpin capillary loops rising into the dermal papillae, arterial limb up, venous limb down."""
    for x in np.linspace(x0, x1, n):
        t = np.linspace(0, np.pi, 60)
        w = 0.085
        ax.plot(x - w * np.cos(t) * 0.5 - w / 2, base - (base - crest) * np.sin(t), color=ART,
                lw=0.9, zorder=3)
        ax.plot(x + w * np.cos(t) * 0.5 + w / 2, base - (base - crest) * np.sin(t), color=VEN,
                lw=0.9, zorder=3)


def level(name: str):
    """Return the PLEXUS entry of that name."""
    return next(v for v in PLEXUS if v.name == name)


def calibre_label(ax, text: str, xy, xytext, colour: str = INK_2, fontsize: float = 7.6,
                  arrow: bool = True) -> None:
    """Annotate a vessel level with its measured calibre."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=colour, ha="left", va="center",
                linespacing=1.3, zorder=8,
                arrowprops=dict(arrowstyle="-", color="#9a948c", lw=0.7,
                                shrinkA=2, shrinkB=2) if arrow else None)


def main_panel(ax) -> None:
    """Draw the full-depth cartoon: layers to scale, the vascular tree, the slab and the target band."""
    draw_layers(ax, WIDTH_MM, DEPTH_MM)

    sup, low, deep = (level(n) for n in ("superficial (subpapillary) plexus", "lower dermis",
                                         "deep (dermal-subcutaneous) plexus"))
    sup_d = (sup.top_mm + sup.bottom_mm) / 2
    deep_d = deep.top_mm + 0.18

    capillary_loops(ax, 0.6, WIDTH_MM - 0.6, 22, crest=0.17, base=0.5)

    # superficial plexus: a paired artery and vein running the width of the panel
    wavy(ax, 0.3, WIDTH_MM - 0.3, sup_d - 0.07, 0.07, 6, ART, 1.5)
    wavy(ax, 0.3, WIDTH_MM - 0.3, sup_d + 0.09, 0.07, 6, VEN, 1.9, phase=0.5)

    # lower dermis, then the deep plexus at the dermal-fat junction
    wavy(ax, 0.3, WIDTH_MM - 0.3, (low.top_mm + low.bottom_mm) / 2, 0.05, 4, ART, 1.9, phase=1.1)
    wavy(ax, 0.3, WIDTH_MM - 0.3, deep_d, 0.09, 3.5, ART, 2.4)
    wavy(ax, 0.3, WIDTH_MM - 0.3, deep_d + 0.3, 0.09, 3.5, VEN, 3.0, phase=0.9)

    # ascending vessels linking the levels (Braverman: spaced 1.5-7 mm apart)
    for i, x in enumerate(np.arange(1.6, WIDTH_MM - 1.0, 3.1)):
        ascending(ax, x, 0.55, sup_d, ART, 1.2, bow=0.18 * (-1) ** i)
        ascending(ax, x + 0.45, sup_d, deep_d, VEN, 1.7, bow=-0.22 * (-1) ** i)
        ascending(ax, x - 0.3, sup_d, low.bottom_mm, ART, 1.4, bow=0.2 * (-1) ** i)

    # subcutaneous vessels: the target, running through the fat at every depth
    rng = np.random.default_rng(7)
    for depth in np.linspace(TARGET.top_mm + 1.4, min(TARGET.bottom_mm, DEPTH_MM) - 0.8, 4):
        jitter = rng.uniform(-0.2, 0.2)
        wavy(ax, 0.3, WIDTH_MM - 0.3, depth + jitter, 0.3, rng.uniform(1.4, 2.2), ART, 2.0,
             phase=rng.uniform(0, 6))
        wavy(ax, 0.3, WIDTH_MM - 0.3, depth + jitter + 0.5, 0.3, rng.uniform(1.4, 2.2), VEN, 2.6,
             phase=rng.uniform(0, 6))
    for x in (3.4, 9.1, 14.2):  # branches climbing from the fat to the deep plexus
        ascending(ax, x, deep_d + 0.4, TARGET.top_mm + 2.4, ART, 1.7, bow=0.5)

    # a perforator crossing the deep fascia
    perf = level("perforators")
    ascending(ax, 11.6, deep_d + 0.5, DEPTH_MM - 0.1, ART, 3.4, bow=1.0)
    ascending(ax, 12.5, deep_d + 0.8, DEPTH_MM - 0.1, VEN, 4.2, bow=1.0)

    # --- calibres, each read from the PLEXUS table -------------------------------
    x_lab = WIDTH_MM + 0.55
    for text, y_vessel, y_label in (
        (f"capillary loops\n{level('capillary loops').diameter_um[0]:.0f}–"
         f"{level('capillary loops').diameter_um[1]:.0f} µm", (WIDTH_MM - 1.2, 0.3), (x_lab, 0.25)),
        (f"superficial plexus\n{sup.diameter_um[0]:.0f}–{sup.diameter_um[1]:.0f} µm",
         (WIDTH_MM - 1.2, sup_d), (x_lab, sup_d)),
        (f"deep plexus\n{deep.diameter_um[0]:.0f}–{deep.diameter_um[1]:.0f} µm",
         (WIDTH_MM - 1.2, deep_d + 0.15), (x_lab, deep.bottom_mm + 0.05)),
        (f"perforators\n{perf.diameter_um[1] / 1000:.1f} mm", (12.6, DEPTH_MM - 0.9), (x_lab, DEPTH_MM - 0.5)),
    ):
        calibre_label(ax, text, y_vessel, y_label)

    # the target depth range, as stated by the project, not an anatomical boundary
    t_top, t_bot = TARGET_DEPTH_MM
    ax.add_patch(Rectangle((0, t_top), WIDTH_MM, min(t_bot, DEPTH_MM) - t_top,
                           facecolor="none", edgecolor=INK, lw=1.1, ls=(0, (4, 3)), zorder=6))
    ax.annotate(f"TARGET  {TARGET.diameter_um[0]:.0f}–{TARGET.diameter_um[1]:.0f} µm\n"
                f"at {t_top:g}–{min(t_bot, DEPTH_MM):g} mm depth",
                (0.35, TARGET.top_mm + 0.45), fontsize=8.6, color=INK, weight="bold",
                ha="left", va="top", linespacing=1.35, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
    ax.annotate("vessels of this calibre exist only below the dermis:\nthe plexus above tops out at "
                f"{deep.diameter_um[1]:.0f} µm",
                (0.35, TARGET.top_mm + 1.5), fontsize=7.8, color=INK_2, ha="left", va="top",
                linespacing=1.35, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    ax.set(xlim=(-3.4, WIDTH_MM + 4.6), ylim=(DEPTH_MM, 0))
    ax.axis("off")


def zoom_panel(ax) -> None:
    """Magnify the top 2 mm, where the whole dermal plexus ladder is a hairline in the main panel."""
    w = 4.2
    draw_layers(ax, w, ZOOM_MM, label=False)
    sup = level("superficial (subpapillary) plexus")
    sup_d = (sup.top_mm + sup.bottom_mm) / 2

    capillary_loops(ax, 0.35, w - 0.35, 9, crest=0.15, base=0.48)
    wavy(ax, 0.1, w - 0.1, sup_d - 0.07, 0.05, 3, ART, 2.0)
    wavy(ax, 0.1, w - 0.1, sup_d + 0.1, 0.05, 3, VEN, 2.6, phase=0.6)
    wavy(ax, 0.1, w - 0.1, 1.78, 0.05, 2.5, ART, 2.6, phase=1.2)
    for i, x in enumerate(np.linspace(0.8, w - 0.8, 3)):
        ascending(ax, x, 0.52, sup_d, ART, 1.5, bow=0.1 * (-1) ** i)
        ascending(ax, x + 0.25, sup_d, 1.78, VEN, 2.0, bow=-0.12 * (-1) ** i)

    for name, depth, side in (("epidermis  0.1 mm", 0.05, "top"),
                              ("capillary loops\nin the dermal papillae", 0.32, "mid"),
                              (f"superficial plexus\n{sup.top_mm:g}–{sup.bottom_mm:g} mm", sup_d, "mid"),
                              ("lower dermis\n40–50 µm", 1.78, "mid")):
        ax.annotate(name, (w + 0.12, depth), fontsize=7.4, color=INK_2, ha="left", va="center",
                    linespacing=1.3)

    # a depth ruler, so the magnification is readable rather than asserted
    for d in np.arange(0.0, ZOOM_MM + 0.01, 0.5):
        ax.plot([-0.22, -0.08], [d, d], color="#9a948c", lw=0.8, zorder=4)
        ax.annotate(f"{d:g}", (-0.3, d), ha="right", va="center", fontsize=7, color=INK_2)
    ax.plot([-0.15, -0.15], [0, ZOOM_MM], color="#9a948c", lw=0.8, zorder=4)
    ax.annotate("mm", (-0.3, -0.12), ha="right", va="center", fontsize=7, color=INK_2)

    ax.set(xlim=(-0.95, w + 2.9), ylim=(ZOOM_MM, 0))
    ax.axis("off")
    ax.set_title(f"top {ZOOM_MM:g} mm, magnified — the whole dermal plexus",
                 fontsize=8.6, color=INK_2, loc="left", pad=6)


def calibre_key(ax) -> None:
    """Draw the target vessels beside the voxel in mm, at EQUAL ASPECT so the comparison is honest."""
    ax.set_aspect("equal")
    ax.axis("off")
    v = VOXEL_MM
    ax.add_patch(Rectangle((0.0, -v / 2), v, v, facecolor="none", edgecolor=INK, lw=1.3, zorder=3))
    ax.annotate(f"voxel\n{v * 1000:.0f} µm", (v / 2, -v / 2 - 0.045), ha="center", va="top",
                fontsize=7.6, color=INK, linespacing=1.3)

    x = v + 0.26
    for d_um in (TARGET.diameter_um[0], 300.0, TARGET.diameter_um[1]):
        r = d_um / 1000 / 2
        ax.add_patch(Circle((x + r, 0.0), r, facecolor=ART, alpha=0.45, edgecolor=ART, lw=1.1, zorder=3))
        ax.annotate(f"{d_um:.0f}", (x + r, -v / 2 - 0.045), ha="center", va="top", fontsize=7.6,
                    color=INK_2)
        x += 2 * r + 0.16
    ax.annotate("µm", (x - 0.1, -v / 2 - 0.045), ha="left", va="top", fontsize=7.6, color=INK_2)

    ax.set(xlim=(-0.05, x + 0.1), ylim=(-0.52, 0.42))
    ax.annotate("drawn at equal scale: a 300 µm vessel is one voxel across",
                (0.5, 1.0), xycoords="axes fraction", ha="center", va="top", fontsize=8, color=INK_2)


def main() -> None:
    """Build the three-part anatomy panel and print the table it was drawn from."""
    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[3.1, 1.0],
                          left=0.055, right=0.985, top=0.9, bottom=0.07, wspace=0.08, hspace=0.22)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_key = fig.add_subplot(gs[1, 1])

    main_panel(ax_main)
    zoom_panel(ax_zoom)
    calibre_key(ax_key)

    ax_main.set_title("Cutaneous vessels of the target calibre lie below the dermal plexus, in fat",
                      fontsize=11.5, color=INK, loc="left", pad=22)
    ax_main.text(0, 1.028, "depths to scale, lateral extent schematic; vessel line widths exaggerated "
                           "and labelled with their measured calibre",
                 transform=ax_main.transAxes, fontsize=8.4, color=INK_2)

    print(f"{'level':<38}{'depth (mm)':>14}{'diameter (um)':>18}")
    for v in PLEXUS:
        print(f"  {v.name:<36}{v.top_mm:>6.1f}–{v.bottom_mm:<7.1f}"
              f"{v.diameter_um[0]:>8.0f}–{v.diameter_um[1]:<9.0f}")
    print(f"\nThe dermal plexus tops out at {level('deep (dermal-subcutaneous) plexus').diameter_um[1]:.0f} um, "
          f"so the {TARGET.diameter_um[0]:.0f}-{TARGET.diameter_um[1]:.0f} um target band lies above every "
          f"measured dermal level\nand below the {level('perforators').diameter_um[1] / 1000:.1f} mm perforators: "
          f"the calibre gap no in vivo study resolves.")
    print(f"At the achieved {VOXEL_MM * 1000:.0f} um voxel, a 300 um vessel spans "
          f"{300 / (VOXEL_MM * 1000):.1f} voxels across.")

    save(fig, "fig1a_anatomy", note="Anatomy from the literature (Braverman 2000; perforator calibres "
                                    "J Reconstr Microsurg 2021). Depths to scale, vessel widths not. "
                                    "No measured data.")


if __name__ == "__main__":
    main()
