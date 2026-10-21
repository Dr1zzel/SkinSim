"""Figure 1a: the cutaneous vascular tree, drawn to scale in depth and dimensioned.

The anatomical cartoon that usually opens a skin-imaging paper carries no
numbers, so it cannot say why the problem is hard. This one is dimensioned:
depth is to scale, every level carries its measured calibre, and a true-scale
key puts the target vessels beside the voxel.

Vessels are drawn as ACCOMPANYING PAIRS, an artery with its vein alongside,
because that is how the cutaneous supply is actually built: a perforator pair
crosses the fascia, branches through the fat, and feeds the deep plexus at the
dermal-subcutaneous junction; ascending pairs climb to the superficial plexus;
capillary loops rise into the papillae. Drawing separate arterial and venous
trees doubles the ink for no anatomy, which is why this version pairs them.

Element counts are deliberately low - two perforators, six terminals, three
ascending units, twelve loops - because the figure has to be read at abstract
size. The spacing of the ascending units is kept inside the 1.5-7 mm Braverman
reports.

Layer boundaries come from geometry.LAYERS, calibres and level depths from
geometry.PLEXUS, the target band from geometry.TARGET_DEPTH_MM and the voxel
from protocols.REFERENCE_PROTOCOL, so the figure cannot drift from the model.
DEPTHS are to scale; vessel line widths are exaggerated and labelled as such.

Run: python figures/fig1a_anatomy.py
"""
from __future__ import annotations

import numpy as np
from common import INK, INK_2, plt, save
from matplotlib.patches import Circle, Rectangle

from geometry import LAYERS, PLEXUS, TARGET_DEPTH_MM
from protocols import REFERENCE_PROTOCOL

ART, VEN = "#c0392b", "#2b6ca3"
LAYER_FILL = {"epidermis": "#f3dcbb", "dermis": "#fbe6e4",
              "hypodermis (subcutaneous fat)": "#fdf5d6", "muscle": "#e7d2d2"}
RULE = "#9a948c"

DEPTH_MM, WIDTH_MM = 12.0, 18.0
ZOOM_MM = 3.0
VOXEL_MM = REFERENCE_PROTOCOL.inplane_mm


def level(name: str):
    """Return the PLEXUS entry of that name."""
    return next(v for v in PLEXUS if v.name == name)


SUP, LOW, DEEP = (level(n) for n in ("superficial (subpapillary) plexus", "lower dermis",
                                     "deep (dermal-subcutaneous) plexus"))
TARGET, PERF, CAP = level("subcutaneous vessels"), level("perforators"), level("capillary loops")
SUP_D = (SUP.top_mm + SUP.bottom_mm) / 2
DEEP_D = (DEEP.top_mm + DEEP.bottom_mm) / 2
TRUNK_DEPTH = 11.3                       # the large subcutaneous vessels the perforators rise from
PERF_X = (4.4, 13.6)                     # two perforator pairs, the only trunks in the fat
RISING_X = (3.2, 9.0, 14.8)              # ascending pairs, 5.8 mm apart (Braverman: 1.5-7 mm)


# --- drawing primitives -------------------------------------------------------


def bowed(p0, p1, bow=0.0, n=140):
    """Return a path from p0 to p1 bowed sideways by bow mm at its midpoint."""
    t = np.linspace(0.0, 1.0, n)
    x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    span = float(np.hypot(dx, dy)) or 1.0
    swell = np.sin(np.pi * t) * bow
    return x - dy / span * swell, y + dx / span * swell


def pair(ax, p0, p1, lw, bow=0.0, sep=0.17, zorder=4):
    """Draw an artery with its accompanying vein, offset either side of one path."""
    x, y = bowed(p0, p1, bow)
    dx, dy = np.gradient(x), np.gradient(y)
    span = np.hypot(dx, dy)
    span[span == 0] = 1.0
    nx, ny = -dy / span, dx / span
    ax.plot(x + nx * sep / 2, y + ny * sep / 2, color=ART, lw=lw, solid_capstyle="round",
            zorder=zorder)
    ax.plot(x - nx * sep / 2, y - ny * sep / 2, color=VEN, lw=lw * 1.15, solid_capstyle="round",
            zorder=zorder)


def plexus(ax, x0, x1, depth, cycles, amp, lw, sep=0.17, phase=0.0, zorder=5):
    """Draw a horizontal plexus as one arterial and one venous wavy vessel running together."""
    x = np.linspace(x0, x1, 400)
    y = depth + amp * np.sin(2 * np.pi * cycles * (x - x0) / (x1 - x0) + phase)
    ax.plot(x, y - sep / 2, color=ART, lw=lw, solid_capstyle="round", zorder=zorder)
    ax.plot(x, y + sep / 2, color=VEN, lw=lw * 1.15, solid_capstyle="round", zorder=zorder)


def capillary_loops(ax, x0, x1, n, crest, base, rng, width=0.1, lw=1.0):
    """Draw hairpin loops rising into the papillae, arterial limb up and venous limb down."""
    for x in np.linspace(x0, x1, n) + rng.uniform(-0.1, 0.1, n):
        t = np.linspace(0, np.pi, 50)
        height = (base - crest) * rng.uniform(0.85, 1.0)
        ax.plot(x - width / 2 - width * np.cos(t) * 0.5, base - height * np.sin(t), color=ART,
                lw=lw, solid_capstyle="round", zorder=5)
        ax.plot(x + width / 2 + width * np.cos(t) * 0.5, base - height * np.sin(t), color=VEN,
                lw=lw, solid_capstyle="round", zorder=5)


def perforator_unit(ax, x, deep_depth):
    """Draw one perforator pair crossing the fascia and branching through the fat to the deep plexus."""
    pair(ax, (x + 0.5, TRUNK_DEPTH), (x, PERF.top_mm - 0.4), 3.0, bow=0.3, sep=0.26)
    for side, bow in ((-1, -0.5), (1, 0.45)):
        node = (x + side * 3.0, 4.5)
        pair(ax, (x, PERF.top_mm - 0.4), node, 2.1, bow=bow, sep=0.21)
        for reach, end_bow in ((3.4, -0.22), (1.4, 0.2)):
            pair(ax, node, (x + side * reach, deep_depth), 1.4, bow=end_bow, sep=0.17)


def depth_ruler(ax, ticks, x, depth_mm, length=0.16, fontsize=7.2):
    """Draw a depth scale, so the magnification is readable rather than asserted."""
    ax.plot([x, x], [0, depth_mm], color=RULE, lw=0.9, zorder=6)
    for d in ticks:
        ax.plot([x - length, x], [d, d], color=RULE, lw=0.9, zorder=6)
        ax.annotate(f"{d:g}", (x - length - 0.08, d), ha="right", va="center", fontsize=fontsize,
                    color=INK_2, zorder=6)
    ax.annotate("mm", (x, -0.24), ha="center", va="bottom", fontsize=fontsize, color=INK_2)


def layer_rows(depth_mm):
    """Return (name, top_mm, bottom_mm) for every layer, the last clipped to the drawn depth."""
    rows, top = [], 0.0
    for bottom, tissue in LAYERS:
        rows.append((tissue.name, top, min(bottom, depth_mm)))
        if bottom >= depth_mm:
            break
        top = bottom
    return rows


def draw_layers(ax, width_mm, depth_mm, names=False):
    """Fill each skin layer as a tinted band with its boundary drawn."""
    for name, top, bottom in layer_rows(depth_mm):
        ax.add_patch(Rectangle((0, top), width_mm, bottom - top, facecolor=LAYER_FILL[name],
                               edgecolor="none", zorder=0))
        ax.plot([0, width_mm], [bottom, bottom], color="#c9c2b8", lw=0.8, zorder=1)
        if names:
            ax.annotate(name.split(" (")[0], (0.3, (top + bottom) / 2), fontsize=7.6,
                        color="#6f6a63", ha="left", va="center", zorder=2, style="italic")


# --- panels -------------------------------------------------------------------


def main_panel(ax):
    """Draw the full-depth tree: layers to scale, paired vessels, the target band."""
    rng = np.random.default_rng(11)
    draw_layers(ax, WIDTH_MM, DEPTH_MM, names=True)
    t_top, t_bot = TARGET_DEPTH_MM
    ax.add_patch(Rectangle((0, t_top), WIDTH_MM, min(t_bot, DEPTH_MM) - t_top, facecolor="white",
                           alpha=0.3, edgecolor="none", zorder=1))

    # the large subcutaneous vessels everything rises from
    plexus(ax, 0.5, WIDTH_MM - 0.5, TRUNK_DEPTH, 0.5, 0.1, 3.4, sep=0.3)
    for x in PERF_X:
        perforator_unit(ax, x, DEEP_D)

    plexus(ax, 0.4, WIDTH_MM - 0.4, DEEP_D, 3, 0.11, 2.0, sep=0.22)
    for x in RISING_X:
        pair(ax, (x, DEEP_D), (x, SUP_D), 1.3, bow=0.18, sep=0.17)
    plexus(ax, 0.4, WIDTH_MM - 0.4, SUP_D, 5, 0.06, 1.4, sep=0.13)
    for x in RISING_X:
        pair(ax, (x, SUP_D), (x, CAP.bottom_mm + 0.12), 1.0, sep=0.11)
    # the subpapillary arcade the loops actually arise from, so they are not left floating
    plexus(ax, 0.9, WIDTH_MM - 0.9, CAP.bottom_mm + 0.12, 6, 0.03, 0.9, sep=0.07)
    capillary_loops(ax, 1.0, WIDTH_MM - 1.0, 12, crest=CAP.top_mm + 0.05,
                    base=CAP.bottom_mm + 0.1, rng=rng)

    depth_ruler(ax, [0, 0.1, 1, 2, 2.6, 4, 6, 8, 10, 12], -0.55, DEPTH_MM)

    for text, xy, xytext in (
        (f"capillary loops  {CAP.diameter_um[0]:.0f}–{CAP.diameter_um[1]:.0f} µm",
         (WIDTH_MM - 1.6, CAP.top_mm + 0.15), (WIDTH_MM + 1.1, -0.3)),
        (f"superficial plexus  {SUP.diameter_um[0]:.0f}–{SUP.diameter_um[1]:.0f} µm\n"
         f"{SUP.top_mm:g}–{SUP.bottom_mm:g} mm", (WIDTH_MM - 1.6, SUP_D), (WIDTH_MM + 1.1, 1.25)),
        (f"deep plexus  {DEEP.diameter_um[0]:.0f}–{DEEP.diameter_um[1]:.0f} µm\n"
         f"{DEEP.top_mm:g}–{DEEP.bottom_mm:g} mm", (WIDTH_MM - 1.6, DEEP_D),
         (WIDTH_MM + 1.1, 3.0)),
        (f"subcutaneous vessels\n{TARGET.diameter_um[0]:.0f}–{TARGET.diameter_um[1]:.0f} µm"
         f"   ← TARGET", (17.0, 2.9), (WIDTH_MM + 1.1, 5.6)),
        (f"perforator  {PERF.diameter_um[1] / 1000:.1f} mm\nacross the fascia, "
         f"{PERF.top_mm:g}–{PERF.bottom_mm:g} mm", (14.1, 9.4), (WIDTH_MM + 1.1, 9.1)),
        ("large subcutaneous vessels", (16.0, TRUNK_DEPTH), (WIDTH_MM + 1.1, 11.4)),
    ):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.8, color=INK_2, ha="left", va="center",
                    linespacing=1.35, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7, shrinkA=3, shrinkB=3,
                                    connectionstyle="arc3,rad=0.1"))

    ax.plot([-2.35, -2.55, -2.55, -2.35], [t_top, t_top, t_bot, t_bot], color=INK, lw=1.2,
            zorder=7, solid_joinstyle="miter")
    ax.annotate(f"TARGET  {t_top:g}–{t_bot:g} mm", (-2.75, (t_top + t_bot) / 2), rotation=90,
                ha="center", va="center", fontsize=8.6, color=INK, weight="bold", zorder=7)
    ax.annotate(f"the dermal plexus tops out at {DEEP.diameter_um[1]:.0f} µm, so vessels of the\n"
                "target calibre exist only below the dermis, in fat",
                (0.45, 8.6), fontsize=7.9, color=INK_2, ha="left", va="bottom", linespacing=1.4,
                zorder=9, bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#e4e3df", alpha=0.9))
    ax.annotate("artery", (0.55, TRUNK_DEPTH - 0.95), color=ART, fontsize=7.8, ha="left",
                va="center", zorder=9, weight="bold")
    ax.annotate("vein", (2.4, TRUNK_DEPTH - 0.95), color=VEN, fontsize=7.8, ha="left",
                va="center", zorder=9, weight="bold")

    ax.set(xlim=(-3.5, WIDTH_MM + 6.4), ylim=(DEPTH_MM + 0.25, -0.75))
    ax.axis("off")


def zoom_panel(ax):
    """Magnify the top 3 mm, where the whole dermal ladder is a hairline in the main panel."""
    rng = np.random.default_rng(5)
    width = 5.0
    draw_layers(ax, width, ZOOM_MM)
    rising = np.linspace(1.0, width - 1.0, 3)

    plexus(ax, 0.15, width - 0.15, DEEP_D, 2, 0.09, 2.6, sep=0.2)
    for x in rising:
        pair(ax, (x, DEEP_D), (x, SUP_D), 1.8, bow=0.12, sep=0.15)
        pair(ax, (x, SUP_D), (x, CAP.bottom_mm + 0.08), 1.3, sep=0.11)
    plexus(ax, 0.15, width - 0.15, SUP_D, 3, 0.05, 1.9, sep=0.12)
    plexus(ax, 0.35, width - 0.35, CAP.bottom_mm + 0.08, 4, 0.02, 1.2, sep=0.06)
    capillary_loops(ax, 0.45, width - 0.45, 8, crest=CAP.top_mm + 0.03,
                    base=CAP.bottom_mm + 0.08, rng=rng, width=0.09, lw=1.2)

    depth_ruler(ax, [0, 0.5, 1.0, 1.5, 2.0, 2.5], -0.18, ZOOM_MM, length=0.12, fontsize=7.0)
    for text, depth in ((f"epidermis {LAYERS[0][0]:g} mm", 0.05),
                        ("capillary loops in the papillae", (CAP.top_mm + CAP.bottom_mm) / 2),
                        ("superficial plexus", SUP_D),
                        ("deep plexus, at the\ndermal–fat junction", DEEP_D + 0.1)):
        ax.annotate(text, (width + 0.18, depth), fontsize=7.2, color=INK_2, ha="left", va="center")

    ax.set(xlim=(-1.0, width + 5.4), ylim=(ZOOM_MM, -0.12))
    ax.axis("off")
    ax.set_title(f"top {ZOOM_MM:g} mm magnified — the dermal ladder", fontsize=8.6, color=INK_2,
                 loc="left", pad=4)


def calibre_key(ax):
    """Draw the target vessels beside the voxel at EQUAL ASPECT, so the comparison is honest."""
    ax.set_aspect("equal")
    ax.axis("off")
    v = VOXEL_MM
    ax.add_patch(Rectangle((0.0, -v / 2), v, v, facecolor="none", edgecolor=INK, lw=1.4, zorder=3))
    ax.annotate(f"voxel\n{v * 1000:.0f} µm", (v / 2, -v / 2 - 0.04), ha="center", va="top",
                fontsize=7.4, color=INK, linespacing=1.3)
    x = v + 0.22
    for d_um in (TARGET.diameter_um[0], 300.0, TARGET.diameter_um[1]):
        r = d_um / 2000
        ax.add_patch(Circle((x + r, 0.0), r, facecolor=ART, alpha=0.45, edgecolor=ART, lw=1.1,
                            zorder=3))
        ax.annotate(f"{d_um:.0f}", (x + r, -v / 2 - 0.04), ha="center", va="top", fontsize=7.4,
                    color=INK_2)
        x += 2 * r + 0.13
    ax.annotate("µm", (x - 0.07, -v / 2 - 0.04), ha="left", va="top", fontsize=7.4, color=INK_2)
    ax.set(xlim=(-0.04, x + 0.06), ylim=(-0.46, 0.38))
    ax.annotate(f"equal scale: a 300 µm vessel is {300 / (v * 1000):.1f} voxels across",
                (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom", fontsize=7.8,
                color=INK_2)


def main():
    """Build the three-part anatomy figure and print the table it was drawn from."""
    fig = plt.figure(figsize=(13.8, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[2.9, 1.0],
                          left=0.045, right=0.99, top=0.845, bottom=0.055, wspace=0.02, hspace=0.3)
    main_panel(fig.add_subplot(gs[:, 0]))
    zoom_panel(fig.add_subplot(gs[0, 1]))
    calibre_key(fig.add_subplot(gs[1, 1]))

    fig.suptitle("Cutaneous vessels of the target calibre lie below the dermal plexus, in fat",
                 fontsize=12.5, x=0.045, ha="left", y=0.982, color=INK)
    fig.text(0.045, 0.945,
             "Arteries drawn with their accompanying veins. Perforators cross the fascia, branch through "
             "the fat and feed\nthe deep plexus; ascending pairs climb to the superficial plexus and end in "
             "capillary loops. Depths to scale;\nvessel widths exaggerated, so every level carries its "
             "measured calibre instead.",
             fontsize=8.3, color=INK_2, va="top", linespacing=1.5)

    print(f"{'level':<40}{'depth (mm)':>16}{'diameter (um)':>18}")
    for entry in PLEXUS:
        print(f"  {entry.name:<38}{entry.top_mm:>6.1f}–{entry.bottom_mm:<9.1f}"
              f"{entry.diameter_um[0]:>8.0f}–{entry.diameter_um[1]:<9.0f}")
    print(f"\nThe dermal plexus tops out at {DEEP.diameter_um[1]:.0f} um, so the "
          f"{TARGET.diameter_um[0]:.0f}-{TARGET.diameter_um[1]:.0f} um target band lies above every measured "
          f"dermal level and\nbelow the {PERF.diameter_um[1] / 1000:.1f} mm perforators: the calibre gap no "
          f"in vivo study resolves.")
    print(f"At the {VOXEL_MM * 1000:.0f} um voxel of {REFERENCE_PROTOCOL.name}, a 300 um vessel spans "
          f"{300 / (VOXEL_MM * 1000):.1f} voxels across.")

    save(fig, "fig1a_anatomy",
         note="Anatomy from the literature (Braverman, J Invest Dermatol Symp Proc 2000; perforator "
              "calibres J Reconstr Microsurg 2021). Depths to scale, vessel widths not. No measured data.")


if __name__ == "__main__":
    main()
