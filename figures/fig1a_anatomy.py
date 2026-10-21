"""Figure 1a: the cutaneous vascular tree in one frame, drawn to scale in depth.

The cartoon that usually opens a skin-imaging paper carries no numbers, so it
cannot say why the problem is hard. This one is dimensioned: depth is to scale,
each vascular level is named with its measured calibre, and a corner inset puts
the target vessels beside the voxel at equal scale.

Arteries are drawn with their accompanying veins, because that is how the
cutaneous supply is built: a perforator pair crosses the fascia, branches
through the fat, feeds the deep plexus at the dermal-subcutaneous junction;
ascending pairs climb to the superficial plexus and end in capillary loops.

Layer boundaries come from geometry.LAYERS, calibres and level depths from
geometry.PLEXUS, the target band from geometry.TARGET_DEPTH_MM and the voxel
from protocols.REFERENCE_PROTOCOL, so the figure cannot drift from the model.
DEPTHS are to scale; vessel line widths are exaggerated.

Run: python figures/fig1a_anatomy.py
"""
from __future__ import annotations

import numpy as np
from common import INK, INK_2, plt, save
from matplotlib.patches import Circle, Polygon, Rectangle
import matplotlib.patheffects as pe

from geometry import LAYERS, PLEXUS, TARGET_DEPTH_MM
from protocols import REFERENCE_PROTOCOL

ART, VEN = "#c0392b", "#2b6ca3"
EPI_FILL, EPI_EDGE = "#eec5a6", "#d2a07c"
DERM_FILL, DERM_LINE = "#fae3e0", "#eec6c1"
FAT_FILL, FAT_EDGE = "#fdf0a4", "#e3c85f"
MUSCLE_FILL = "#e3cfcf"
HAIR = "#b98c5e"
RULE = "#9a948c"

DEPTH_MM, WIDTH_MM = 12.0, 18.0
VOXEL_MM = REFERENCE_PROTOCOL.inplane_mm


def level(name: str):
    """Return the PLEXUS entry of that name."""
    return next(v for v in PLEXUS if v.name == name)


SUP, DEEP = level("superficial (subpapillary) plexus"), level("deep (dermal-subcutaneous) plexus")
TARGET, PERF, CAP = level("subcutaneous vessels"), level("perforators"), level("capillary loops")
SUP_D, DEEP_D = (SUP.top_mm + SUP.bottom_mm) / 2, (DEEP.top_mm + DEEP.bottom_mm) / 2
EPI_BOTTOM, DERM_BOTTOM = LAYERS[0][0], LAYERS[1][0]
FAT_BOTTOM = LAYERS[2][0]
TRUNK_DEPTH = 11.3
PERF_X = (4.4, 13.6)
RISING_X = (3.2, 9.0, 14.8)
ARCADE_D = CAP.bottom_mm + 0.12


# --- tissue ------------------------------------------------------------------


def rete_ridges(ax, x0, x1, bottom, amplitude=0.055, period=0.85):
    """Return the scalloped dermo-epidermal junction and fill the epidermis above it."""
    x = np.linspace(x0, x1, 900)
    y = bottom + amplitude * (1 + np.sin(2 * np.pi * x / period)) / 2
    ax.fill_between(x, 0.0, y, color=EPI_FILL, zorder=2)
    ax.plot(x, y, color=EPI_EDGE, lw=0.7, zorder=3)
    ax.plot([x0, x1], [0, 0], color=EPI_EDGE, lw=0.9, zorder=3)


def fat_lobules(ax, x0, x1, top, bottom, rng, clip, n=720):
    """Pack the hypodermis with adipose lobules, drawn as the reference illustrations do."""
    placed = []
    for _ in range(n):
        r = rng.uniform(0.17, 0.33)
        x, y = rng.uniform(x0 - r, x1 + r), rng.uniform(top + r * 0.3, bottom - r * 0.3)
        if all((x - px) ** 2 + (y - py) ** 2 > (r + pr) ** 2 * 0.62 for px, py, pr in placed):
            placed.append((x, y, r))
            lobule = Circle((x, y), r, facecolor=FAT_FILL, edgecolor=FAT_EDGE, lw=0.6, zorder=1)
            ax.add_patch(lobule)
            lobule.set_clip_path(clip)
    return len(placed)


def collagen(ax, x0, x1, top, bottom, rng, n=26):
    """Suggest the dermal collagen with a few faint horizontal bundles."""
    for depth in rng.uniform(top + 0.12, bottom - 0.05, n):
        x = np.linspace(x0, x1, 200)
        ax.plot(x, depth + 0.035 * np.sin(2 * np.pi * x / rng.uniform(2.5, 5.0)
                                          + rng.uniform(0, 6)),
                color=DERM_LINE, lw=0.7, zorder=1)


def hair_follicle(ax, x, bottom, tilt=0.9, width=0.19):
    """Draw a hair follicle running obliquely into the dermis, with its shaft above the surface."""
    t = np.linspace(0, 1, 60)
    cx = x + tilt * t
    cy = bottom * t
    half = width * (1 - 0.45 * t) * np.where(t > 0.86, 1.7, 1.0)
    ax.add_patch(Polygon(np.column_stack([np.r_[cx - half, cx[::-1] + half[::-1]],
                                          np.r_[cy, cy[::-1]]]),
                         closed=True, facecolor="#e8cba6", edgecolor=HAIR, lw=0.8, zorder=4))
    shaft = np.linspace(0, 1, 40)
    ax.plot(x - 0.8 * shaft, -1.25 * shaft, color=HAIR, lw=1.6, solid_capstyle="round", zorder=5)


# --- vessels ------------------------------------------------------------------


def bowed(p0, p1, bow=0.0, n=140):
    """Return a path from p0 to p1 bowed sideways by bow mm at its midpoint."""
    t = np.linspace(0.0, 1.0, n)
    x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    span = float(np.hypot(dx, dy)) or 1.0
    swell = np.sin(np.pi * t) * bow
    return x - dy / span * swell, y + dx / span * swell


def _stroke(lw):
    """Return a white casing so vessels stay legible over the lobules."""
    return [pe.withStroke(linewidth=lw + 1.3, foreground="white")]


def pair(ax, p0, p1, lw, bow=0.0, sep=0.17, zorder=7):
    """Draw an artery with its accompanying vein, offset either side of one path."""
    x, y = bowed(p0, p1, bow)
    dx, dy = np.gradient(x), np.gradient(y)
    span = np.hypot(dx, dy)
    span[span == 0] = 1.0
    nx, ny = -dy / span, dx / span
    for sign, colour, width in ((+1, ART, lw), (-1, VEN, lw * 1.15)):
        ax.plot(x + nx * sign * sep / 2, y + ny * sign * sep / 2, color=colour, lw=width,
                solid_capstyle="round", zorder=zorder, path_effects=_stroke(width))


def plexus(ax, x0, x1, depth, cycles, amp, lw, sep=0.17, phase=0.0, zorder=8):
    """Draw a horizontal plexus as one arterial and one venous vessel running together."""
    x = np.linspace(x0, x1, 500)
    y = depth + amp * np.sin(2 * np.pi * cycles * (x - x0) / (x1 - x0) + phase)
    for sign, colour, width in ((-1, ART, lw), (+1, VEN, lw * 1.15)):
        ax.plot(x, y + sign * sep / 2, color=colour, lw=width, solid_capstyle="round",
                zorder=zorder, path_effects=_stroke(width))


def twigs(ax, p0, p1, rng, n=3, length=0.55, lw=0.8):
    """Sprout short side branches along a segment, which is what makes a tree read as one."""
    x, y = bowed(p0, p1)
    for frac in rng.uniform(0.25, 0.85, n):
        i = int(frac * (len(x) - 1))
        angle = rng.uniform(0.5, 1.4) * rng.choice([-1, 1])
        dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
        span = np.hypot(dx, dy) or 1.0
        ex = x[i] + (dx * np.cos(angle) - dy * np.sin(angle)) / span * length
        ey = y[i] + (dx * np.sin(angle) + dy * np.cos(angle)) / span * length
        for sign, colour in ((+1, ART), (-1, VEN)):
            ax.plot([x[i] + sign * 0.05, ex], [y[i], ey], color=colour, lw=lw,
                    solid_capstyle="round", zorder=6, path_effects=_stroke(lw))


def perforator_unit(ax, x, rng):
    """Draw one perforator pair crossing the fascia and branching through the fat."""
    pair(ax, (x + 0.5, TRUNK_DEPTH), (x, PERF.top_mm - 0.4), 3.0, bow=0.3, sep=0.27)
    for side, bow in ((-1, -0.5), (1, 0.45)):
        node = (x + side * 3.0, 4.5)
        pair(ax, (x, PERF.top_mm - 0.4), node, 2.1, bow=bow, sep=0.21)
        for reach, end_bow in ((3.4, -0.22), (1.4, 0.2)):
            pair(ax, node, (x + side * reach, DEEP_D), 1.4, bow=end_bow, sep=0.17)


def capillary_loops(ax, x0, x1, n, crest, base, rng, width=0.1, lw=1.0):
    """Draw hairpin loops rising into the papillae, arterial limb up and venous limb down."""
    for x in np.linspace(x0, x1, n) + rng.uniform(-0.1, 0.1, n):
        t = np.linspace(0, np.pi, 50)
        height = (base - crest) * rng.uniform(0.85, 1.0)
        for sign, colour in ((-1, ART), (+1, VEN)):
            ax.plot(x + sign * (width / 2 + width * np.cos(t) * 0.5), base - height * np.sin(t),
                    color=colour, lw=lw, solid_capstyle="round", zorder=9)


# --- labels -------------------------------------------------------------------


def level_label(ax, depth, name, calibre, x_tick=-0.95, x_text=-1.5):
    """Name one vascular level at its own depth, with its calibre under it."""
    ax.plot([x_text + 0.1, x_tick], [depth, depth], color=RULE, lw=0.7, zorder=6)
    ax.annotate(name, (x_text, depth - 0.1), ha="right", va="bottom", fontsize=8.2, color=INK)
    ax.annotate(calibre, (x_text, depth + 0.1), ha="right", va="top", fontsize=7.6, color=INK_2)


def layer_bracket(ax, top, bottom, name, x=WIDTH_MM + 0.5, tip=0.22):
    """Bracket one anatomical layer on the right, the only layer text in the figure."""
    ax.plot([x, x + tip, x + tip, x], [top, top, bottom, bottom], color=RULE, lw=1.0, zorder=6)
    ax.annotate(name, (x + tip + 0.25, (top + bottom) / 2), ha="left", va="center", fontsize=8.6,
                color=INK)


def calibre_inset(ax):
    """Put the target vessels beside the voxel at equal scale, in the clear space above the frame."""
    scale = 2.0                        # mm of drawing per mm of vessel: an explicit magnification
    x, y = 1.45, -1.0
    side = VOXEL_MM * scale
    ax.add_patch(Rectangle((x, y - side / 2), side, side, facecolor="white", edgecolor=INK,
                           lw=1.2, zorder=10))
    ax.annotate(f"voxel {VOXEL_MM * 1000:.0f} µm", (x + side / 2, y - side / 2 - 0.08),
                ha="center", va="bottom", fontsize=7.2, color=INK, zorder=10)
    x += side + 0.5
    for d_um in (TARGET.diameter_um[0], 300.0, TARGET.diameter_um[1]):
        r = d_um / 2000 * scale
        ax.add_patch(Circle((x + r, y), r, facecolor=ART, alpha=0.5, edgecolor=ART, lw=1.0,
                            zorder=10))
        ax.annotate(f"{d_um:.0f}", (x + r, y + 0.62), ha="center", va="top", fontsize=7.2,
                    color=INK_2, zorder=10)
        x += 2 * r + 0.26
    ax.annotate("µm, equal scale", (x - 0.1, y), ha="left", va="center", fontsize=7.6,
                color=INK_2, zorder=10)


# --- the frame ----------------------------------------------------------------


def draw(ax):
    """Draw the single frame: tissue, the vascular tree, and the few labels it needs."""
    rng = np.random.default_rng(11)
    frame = Rectangle((0, 0), WIDTH_MM, DEPTH_MM, facecolor="none", edgecolor="none", zorder=0)
    ax.add_patch(frame)
    ax.add_patch(Rectangle((0, EPI_BOTTOM), WIDTH_MM, DERM_BOTTOM - EPI_BOTTOM,
                           facecolor=DERM_FILL, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, DERM_BOTTOM), WIDTH_MM, FAT_BOTTOM - DERM_BOTTOM,
                           facecolor="#fffbe8", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, FAT_BOTTOM), WIDTH_MM, DEPTH_MM - FAT_BOTTOM,
                           facecolor=MUSCLE_FILL, edgecolor="none", zorder=0))
    collagen(ax, 0, WIDTH_MM, EPI_BOTTOM, DERM_BOTTOM, rng)
    fat_lobules(ax, 0, WIDTH_MM, DERM_BOTTOM, FAT_BOTTOM, rng, frame)
    rete_ridges(ax, 0, WIDTH_MM, EPI_BOTTOM)
    for x, tilt in ((5.6, 0.8), (12.2, -0.7)):
        hair_follicle(ax, x, DERM_BOTTOM - 0.15, tilt)

    plexus(ax, 0.5, WIDTH_MM - 0.5, TRUNK_DEPTH, 0.5, 0.1, 3.4, sep=0.31)
    for x in PERF_X:
        perforator_unit(ax, x, rng)
    plexus(ax, 0.4, WIDTH_MM - 0.4, DEEP_D, 3, 0.11, 2.0, sep=0.22)
    for x in RISING_X:
        pair(ax, (x, DEEP_D), (x, SUP_D), 1.3, bow=0.18, sep=0.17)
        pair(ax, (x, SUP_D), (x, ARCADE_D), 1.0, sep=0.11)
    plexus(ax, 0.4, WIDTH_MM - 0.4, SUP_D, 5, 0.06, 1.4, sep=0.13)
    plexus(ax, 0.9, WIDTH_MM - 0.9, ARCADE_D, 6, 0.03, 0.9, sep=0.07, zorder=9)
    capillary_loops(ax, 1.0, WIDTH_MM - 1.0, 12, CAP.top_mm + 0.05, CAP.bottom_mm + 0.1, rng)

    ax.plot([0, WIDTH_MM, WIDTH_MM, 0, 0], [0, 0, DEPTH_MM, DEPTH_MM, 0], color="#c9c2b8", lw=0.9,
            zorder=6)

    for depth, name, calibre in (
        (CAP.top_mm + 0.15, "capillary loops", f"{CAP.diameter_um[0]:.0f}–{CAP.diameter_um[1]:.0f} µm"),
        (SUP_D, "superficial plexus", f"{SUP.diameter_um[0]:.0f}–{SUP.diameter_um[1]:.0f} µm"),
        (DEEP_D, "deep plexus", f"{DEEP.diameter_um[0]:.0f}–{DEEP.diameter_um[1]:.0f} µm"),
        (5.4, "subcutaneous vessels", f"{TARGET.diameter_um[0]:.0f}–{TARGET.diameter_um[1]:.0f} µm"),
        (9.3, "perforator", f"{PERF.diameter_um[1] / 1000:.1f} mm"),
    ):
        level_label(ax, depth, name, calibre)

    for depth in (0, DERM_BOTTOM, FAT_BOTTOM, DEPTH_MM):
        ax.plot([-0.55, -0.3], [depth, depth], color=RULE, lw=0.9, zorder=6)
        ax.annotate(f"{depth:g}", (-0.62, depth), ha="right", va="center", fontsize=7.4,
                    color=INK_2, zorder=6)
    ax.plot([-0.42, -0.42], [0, DEPTH_MM], color=RULE, lw=0.9, zorder=6)
    ax.annotate("mm", (-0.42, -0.3), ha="center", va="bottom", fontsize=7.4, color=INK_2)

    layer_bracket(ax, 0, EPI_BOTTOM, "epidermis")
    layer_bracket(ax, EPI_BOTTOM, DERM_BOTTOM, "dermis")
    layer_bracket(ax, DERM_BOTTOM, FAT_BOTTOM, "hypodermis")
    layer_bracket(ax, FAT_BOTTOM, DEPTH_MM, "muscle")

    t_top, t_bot = TARGET_DEPTH_MM
    ax.plot([WIDTH_MM + 2.95, WIDTH_MM + 3.15, WIDTH_MM + 3.15, WIDTH_MM + 2.95],
            [t_top, t_top, t_bot, t_bot], color=INK, lw=1.3, zorder=7)
    ax.annotate(f"TARGET  {t_top:g}–{t_bot:g} mm", (WIDTH_MM + 3.5, (t_top + t_bot) / 2),
                rotation=90, ha="center", va="center", fontsize=8.8, color=INK, weight="bold",
                zorder=7)
    calibre_inset(ax)

    ax.set(xlim=(-5.4, WIDTH_MM + 4.1), ylim=(DEPTH_MM + 0.4, -2.0))
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    """Build the single-frame anatomy figure and print the table it was drawn from."""
    fig = plt.figure(figsize=(11.6, 7.4))
    ax = fig.add_axes((0.01, 0.045, 0.98, 0.9))
    draw(ax)
    fig.suptitle("Target-calibre cutaneous vessels lie below the dermal plexus, in fat",
                 fontsize=12.5, x=0.035, ha="left", y=0.975, color=INK)

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
