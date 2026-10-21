"""Figure 1a: the cutaneous vascular tree, drawn to scale in depth and dimensioned.

The anatomical cartoon that usually opens a skin-imaging paper carries no
numbers, so it cannot say why the problem is hard. This one is dimensioned:

  - depth is to scale, so the 0.1 mm epidermis is a hairline against the 12 mm
    the slab has to cover, and the reader can see why it is not modelled;
  - every level is labelled with its measured calibre, from 10 um capillary
    loops to the ~1.9 mm perforators;
  - the target band sits between the dermal plexus ceiling (~120 um) and the
    perforators - the calibre gap no in vivo study resolves;
  - a true-scale calibre key puts the target vessels beside the voxel, which is
    the detectability argument in one glance.

The vasculature is drawn as one connected tree, not as stacked wavy lines: a
perforator crosses the fascia, branches through the fat, feeds the deep plexus
at the dermal-subcutaneous junction, sends ascending arterioles to the
superficial plexus, and ends in capillary loops in the papillae. The venous
side mirrors it. That architecture is what makes inflow a depth-dependent
question, so a figure that shows unconnected stripes cannot introduce it.

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


# --- drawing primitives -------------------------------------------------------


def bowed(p0, p1, bow=0.0, n=90):
    """Return a path from p0 to p1 bowed sideways by bow mm at its midpoint."""
    t = np.linspace(0.0, 1.0, n)
    x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    span = float(np.hypot(dx, dy)) or 1.0
    swell = np.sin(np.pi * t) * bow
    return x - dy / span * swell, y + dx / span * swell


def vessel(ax, p0, p1, lw, colour, bow=0.0, zorder=4):
    """Draw one vessel segment and return its path, so a child branch can start on it."""
    x, y = bowed(p0, p1, bow)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", zorder=zorder)
    return x, y


def plexus(ax, x0, x1, depth, cycles, amp, colour, lw, phase=0.0, links=True, zorder=4):
    """Draw a horizontal plexus as two interleaved wavy vessels with cross-links: a net, not a line."""
    x = np.linspace(x0, x1, 500)
    k = 2 * np.pi * cycles / (x1 - x0)
    upper = depth + amp * np.sin(k * (x - x0) + phase)
    lower = depth + amp * np.sin(k * (x - x0) + phase + np.pi) + amp * 0.9
    ax.plot(x, upper, color=colour, lw=lw, solid_capstyle="round", zorder=zorder)
    ax.plot(x, lower, color=colour, lw=lw * 0.8, solid_capstyle="round", zorder=zorder)
    if links:
        for xc in np.linspace(x0, x1, int(cycles) * 2 + 1)[1:-1]:
            i = int((xc - x0) / (x1 - x0) * (len(x) - 1))
            ax.plot([x[i], x[i]], [upper[i], lower[i]], color=colour, lw=lw * 0.55,
                    solid_capstyle="round", zorder=zorder - 1)
    return upper.mean()


def capillary_loops(ax, x0, x1, n, crest, base, rng, width=0.09, lw=0.9):
    """Draw hairpin loops rising into the dermal papillae, arterial limb up and venous limb down."""
    for x in np.linspace(x0, x1, n) + rng.uniform(-0.09, 0.09, n):
        t = np.linspace(0, np.pi, 50)
        height = (base - crest) * rng.uniform(0.82, 1.0)
        ax.plot(x - width / 2 - width * np.cos(t) * 0.5, base - height * np.sin(t),
                color=ART, lw=lw, solid_capstyle="round", zorder=4)
        ax.plot(x + width / 2 + width * np.cos(t) * 0.5, base - height * np.sin(t),
                color=VEN, lw=lw, solid_capstyle="round", zorder=4)


# The tree's shape, in mm, as (node, line width, bow). It fans symmetrically
# outwards from one trunk so the fat is filled rather than striped, and the
# third generation lands on the deep plexus at the spacing Braverman reports
# for ascending vessels (1.5-7 mm apart).
TRUNK_X, TRUNK_TOP = 9.6, 8.0
FIRST = [((5.0, 6.0), 2.5, -0.8), ((13.3, 5.9), 2.4, 0.7)]
SECOND = {0: [((2.4, 4.3), 1.8, -0.4), ((6.9, 4.1), 1.7, 0.3)],
          1: [((11.2, 4.0), 1.7, -0.35), ((15.8, 4.4), 1.5, 0.4)]}
THIRD = {(0, 0): [(1.2, 1.2), (3.6, 1.1)], (0, 1): [(5.8, 1.2), (8.2, 1.1)],
         (1, 0): [(10.2, 1.2), (12.4, 1.1)], (1, 1): [(14.9, 1.1), (17.0, 1.0)]}


def arterial_tree(ax, colour, lw_scale, x_shift, deep_depth):
    """Draw a perforator branching up through the fat to the deep plexus: three generations."""
    root = (TRUNK_X + x_shift, DEPTH_MM - 0.05)
    trunk_top = (TRUNK_X - 0.6 + x_shift, TRUNK_TOP)
    vessel(ax, root, trunk_top, 3.6 * lw_scale, colour, bow=0.3)
    for i, (node, width, bow) in enumerate(FIRST):
        node = (node[0] + x_shift, node[1])
        vessel(ax, trunk_top, node, width * lw_scale, colour, bow=bow)
        for j, (child, child_width, child_bow) in enumerate(SECOND[i]):
            child = (child[0] + x_shift, child[1])
            vessel(ax, node, child, child_width * lw_scale, colour, bow=child_bow)
            for x_end, end_width in THIRD[(i, j)]:
                vessel(ax, child, (x_end + x_shift, deep_depth), end_width * lw_scale, colour,
                       bow=0.2 * (-1) ** j)


def ascending(ax, xs, top, bottom, colour, lw, bows):
    """Draw the arterioles that climb between two plexus levels."""
    for x, bow in zip(xs, bows):
        vessel(ax, (x, bottom), (x, top), lw, colour, bow=bow)


def depth_ruler(ax, ticks, x, depth_mm, label="mm", length=0.16, fontsize=7.2):
    """Draw a depth scale so the magnification is readable rather than asserted."""
    ax.plot([x, x], [0, depth_mm], color=RULE, lw=0.9, zorder=6)
    for d in ticks:
        ax.plot([x - length, x], [d, d], color=RULE, lw=0.9, zorder=6)
        ax.annotate(f"{d:g}", (x - length - 0.08, d), ha="right", va="center", fontsize=fontsize,
                    color=INK_2, zorder=6)
    ax.annotate(label, (x, -0.22), ha="center", va="bottom", fontsize=fontsize, color=INK_2)


def layer_rows(depth_mm):
    """Return (name, top_mm, bottom_mm) for every layer, the last clipped to the drawn depth."""
    rows, top = [], 0.0
    for bottom, tissue in LAYERS:
        rows.append((tissue.name, top, min(bottom, depth_mm)))
        if bottom >= depth_mm:
            break
        top = bottom
    return rows


def draw_layers(ax, width_mm, depth_mm, label=True, x0=0.0):
    """Fill each skin layer as a tinted band with its boundary drawn and, optionally, named."""
    for name, top, bottom in layer_rows(depth_mm):
        ax.add_patch(Rectangle((x0, top), width_mm, bottom - top, facecolor=LAYER_FILL[name],
                               edgecolor="none", zorder=0))
        ax.plot([x0, x0 + width_mm], [bottom, bottom], color="#c9c2b8", lw=0.8, zorder=1)
        if label:
            span = f"{top:g}–{bottom:g} mm" if np.isfinite(bottom) else f"below {top:g} mm"
            ax.annotate(f"{name.split(' (')[0]}\n{span}", (x0 + width_mm + 0.25,
                                                           (top + bottom) / 2),
                        ha="left", va="center", fontsize=7.8, color=INK_2, linespacing=1.35)


# --- panels -------------------------------------------------------------------


def main_panel(ax):
    """Draw the full-depth tree: layers to scale, one connected vascular tree, the target band."""
    rng = np.random.default_rng(11)
    draw_layers(ax, WIDTH_MM, DEPTH_MM, label=False)

    # the target depth band, tinted under the drawing rather than boxed over it
    t_top, t_bot = TARGET_DEPTH_MM
    ax.add_patch(Rectangle((0, t_top), WIDTH_MM, min(t_bot, DEPTH_MM) - t_top,
                           facecolor="white", alpha=0.32, edgecolor="none", zorder=1))

    arterial_tree(ax, ART, 1.0, 0.0, DEEP_D - 0.1)
    arterial_tree(ax, VEN, 1.25, 0.9, DEEP_D + 0.45)

    # layer names on the drawing, where the depth ruler already gives the numbers
    for name, top, bottom in layer_rows(DEPTH_MM):
        ax.annotate(name.split(" (")[0], (0.35, (top + bottom) / 2), fontsize=7.6, color="#6f6a63",
                    ha="left", va="center", zorder=2, style="italic")

    plexus(ax, 0.4, WIDTH_MM - 0.4, DEEP_D - 0.12, 4, 0.1, ART, 2.1)
    plexus(ax, 0.4, WIDTH_MM - 0.4, DEEP_D + 0.42, 4, 0.1, VEN, 2.5, phase=0.7)

    rising = np.array([2.3, 5.7, 9.1, 12.5, 15.9])
    bows = [0.16, -0.14, 0.17, -0.15, 0.13]
    ascending(ax, rising, LOW.bottom_mm - 0.05, DEEP_D - 0.14, ART, 1.3, bows)
    ascending(ax, rising + 0.5, DEEP_D + 0.4, LOW.bottom_mm, VEN, 1.6, [-b for b in bows])
    plexus(ax, 0.4, WIDTH_MM - 0.4, LOW.top_mm + 0.15, 5, 0.07, ART, 1.6, links=False)
    ascending(ax, rising + 0.25, SUP_D, LOW.top_mm + 0.15, ART, 1.1, bows)
    ascending(ax, rising + 0.75, SUP_D + 0.12, LOW.top_mm + 0.2, VEN, 1.3, [-b for b in bows])

    plexus(ax, 0.4, WIDTH_MM - 0.4, SUP_D, 7, 0.05, ART, 1.4, links=False)
    plexus(ax, 0.4, WIDTH_MM - 0.4, SUP_D + 0.16, 7, 0.05, VEN, 1.7, phase=0.6, links=False)
    for x in rising:
        vessel(ax, (x + 0.1, SUP_D), (x + 0.1, CAP.bottom_mm + 0.08), 0.9, ART, bow=0.05)
        vessel(ax, (x + 0.65, CAP.bottom_mm + 0.08), (x + 0.65, SUP_D + 0.16), 1.1, VEN, bow=-0.05)
    capillary_loops(ax, 0.7, WIDTH_MM - 0.7, 26, crest=CAP.top_mm + 0.04,
                    base=CAP.bottom_mm + 0.08, rng=rng)

    depth_ruler(ax, [0, 0.1, 1, 2, 2.6, 4, 6, 8, 10, 12], -0.55, DEPTH_MM)

    # calibre annotations, every number read from PLEXUS
    for text, xy, xytext in (
        (f"capillary loops  {CAP.diameter_um[0]:.0f}–{CAP.diameter_um[1]:.0f} µm",
         (WIDTH_MM - 1.0, CAP.top_mm + 0.12), (WIDTH_MM + 1.3, -0.3)),
        (f"superficial plexus  {SUP.diameter_um[0]:.0f}–{SUP.diameter_um[1]:.0f} µm\n"
         f"{SUP.top_mm:g}–{SUP.bottom_mm:g} mm", (WIDTH_MM - 1.0, SUP_D), (WIDTH_MM + 1.3, 1.2)),
        (f"deep plexus  {DEEP.diameter_um[0]:.0f}–{DEEP.diameter_um[1]:.0f} µm\n"
         f"{DEEP.top_mm:g}–{DEEP.bottom_mm:g} mm", (WIDTH_MM - 1.0, DEEP_D + 0.1),
         (WIDTH_MM + 1.3, 2.95)),
        (f"subcutaneous vessels\n{TARGET.diameter_um[0]:.0f}–{TARGET.diameter_um[1]:.0f} µm  ← TARGET",
         (15.9, 4.5), (WIDTH_MM + 1.3, 5.3)),
        (f"perforator  {PERF.diameter_um[1] / 1000:.1f} mm\ncrosses the fascia at "
         f"{PERF.top_mm:g}–{PERF.bottom_mm:g} mm", (9.9, 10.9), (WIDTH_MM + 1.3, 10.4)),
    ):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.8, color=INK_2, ha="left", va="center",
                    linespacing=1.35, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7, shrinkA=3, shrinkB=3,
                                    connectionstyle="arc3,rad=0.12"))

    # the target bracket, outside the drawing
    ax.plot([-2.35, -2.55, -2.55, -2.35], [t_top, t_top, t_bot, t_bot], color=INK, lw=1.2,
            zorder=7, solid_joinstyle="miter")
    ax.annotate(f"TARGET  {t_top:g}–{t_bot:g} mm", (-2.75, (t_top + t_bot) / 2), rotation=90,
                ha="center", va="center", fontsize=8.6, color=INK, weight="bold", zorder=7)

    ax.annotate(f"the dermal plexus tops out at {DEEP.diameter_um[1]:.0f} µm, so vessels of the\n"
                f"target calibre exist only below the dermis, in fat",
                (0.45, 9.55), fontsize=7.9, color=INK_2, ha="left", va="bottom",
                linespacing=1.4, zorder=9,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#e4e3df", alpha=0.9))

    ax.set(xlim=(-3.5, WIDTH_MM + 6.0), ylim=(DEPTH_MM + 0.25, -0.75))
    ax.axis("off")


def zoom_panel(ax):
    """Magnify the top 2.8 mm, where the whole dermal ladder is a hairline in the main panel."""
    rng = np.random.default_rng(5)
    width = 5.0
    draw_layers(ax, width, ZOOM_MM, label=False)

    rising = np.linspace(0.85, width - 0.85, 4)
    bows = [0.1, -0.09, 0.1, -0.09]
    plexus(ax, 0.15, width - 0.15, DEEP_D - 0.1, 3, 0.07, ART, 2.4)
    plexus(ax, 0.15, width - 0.15, DEEP_D + 0.4, 3, 0.07, VEN, 2.9, phase=0.7)
    ascending(ax, rising, LOW.bottom_mm, DEEP_D - 0.12, ART, 1.7, bows)
    ascending(ax, rising + 0.22, DEEP_D + 0.38, LOW.bottom_mm, VEN, 2.1, [-b for b in bows])
    plexus(ax, 0.15, width - 0.15, LOW.top_mm + 0.12, 3, 0.06, ART, 2.0, links=False)
    ascending(ax, rising + 0.1, SUP_D, LOW.top_mm + 0.12, ART, 1.5, bows)
    plexus(ax, 0.15, width - 0.15, SUP_D, 4, 0.05, ART, 1.8, links=False)
    plexus(ax, 0.15, width - 0.15, SUP_D + 0.14, 4, 0.05, VEN, 2.2, phase=0.6, links=False)
    for x, bow in zip(rising, bows):
        vessel(ax, (x + 0.05, SUP_D), (x + 0.05, CAP.bottom_mm + 0.06), 1.2, ART, bow=bow * 0.7)
        vessel(ax, (x + 0.42, CAP.bottom_mm + 0.06), (x + 0.42, SUP_D + 0.14), 1.4, VEN,
               bow=-bow * 0.7)
    capillary_loops(ax, 0.35, width - 0.35, 11, crest=CAP.top_mm + 0.03,
                    base=CAP.bottom_mm + 0.06, rng=rng, width=0.08, lw=1.1)

    depth_ruler(ax, [0, 0.5, 1.0, 1.5, 2.0, 2.5], -0.18, ZOOM_MM, length=0.12, fontsize=7.0)
    for text, depth in ((f"epidermis {LAYERS[0][0]:g} mm", 0.05),
                        ("capillary loops in the papillae", (CAP.top_mm + CAP.bottom_mm) / 2),
                        ("superficial plexus", SUP_D),
                        (f"lower dermis  {LOW.diameter_um[0]:.0f}–{LOW.diameter_um[1]:.0f} µm",
                         LOW.top_mm + 0.12),
                        ("deep plexus, at the\ndermal–fat junction", DEEP_D + 0.2)):
        ax.annotate(text, (width + 0.18, depth), fontsize=7.2, color=INK_2, ha="left", va="center")

    ax.set(xlim=(-1.0, width + 5.6), ylim=(ZOOM_MM, -0.12))
    ax.axis("off")
    ax.set_title(f"top {ZOOM_MM:g} mm magnified — the whole dermal ladder", fontsize=8.6,
                 color=INK_2, loc="left", pad=4)


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
    ax.annotate(f"equal scale: a 300 µm vessel is "
                f"{300 / (v * 1000):.1f} voxels across", (0.5, 1.02), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=7.8, color=INK_2)


def main():
    """Build the three-part anatomy figure and print the table it was drawn from."""
    fig = plt.figure(figsize=(13.8, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[2.9, 1.0],
                          left=0.045, right=0.99, top=0.885, bottom=0.055, wspace=0.02, hspace=0.3)
    main_panel(fig.add_subplot(gs[:, 0]))
    zoom_panel(fig.add_subplot(gs[0, 1]))
    calibre_key(fig.add_subplot(gs[1, 1]))

    fig.suptitle("Cutaneous vessels of the target calibre lie below the dermal plexus, in fat",
                 fontsize=12.5, x=0.045, ha="left", y=0.975, color=INK)
    fig.text(0.045, 0.932,
             "One connected tree: a perforator crosses the fascia, branches through the fat, feeds the deep "
             "plexus, and ends in capillary loops. Arterial red, venous blue.\nDepths are to scale; lateral "
             "extent is schematic and vessel line widths are exaggerated, so every level carries its measured "
             "calibre instead.",
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
