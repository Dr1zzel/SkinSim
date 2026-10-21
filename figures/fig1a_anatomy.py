"""Figure 1a: the cutaneous vascular tree in one frame, drawn to scale in depth.

The cartoon that usually opens a skin-imaging paper carries no numbers, so it
cannot say why the problem is hard. This one is dimensioned: depth is to scale,
each level is named with its measured calibre, and an inset puts the target
vessels beside the voxel at equal scale.

Vessels are drawn as TAPERING FILLED SHAPES along smooth splines, not as
constant-width polylines, because a vessel that does not narrow as it branches
reads as a plot rather than an illustration. The drawn width is the square root
of the true calibre (``drawn_width``), so the hierarchy from a 1.9 mm
perforator down to an 11 um capillary survives the exaggeration that makes the
fine vessels visible at all. Arteries run with their accompanying veins.

Layer boundaries come from geometry.LAYERS, calibres and level depths from
geometry.PLEXUS, the target band from geometry.TARGET_DEPTH_MM and the voxel
from protocols.REFERENCE_PROTOCOL, so the figure cannot drift from the model.
DEPTHS are to scale; widths are exaggerated but ordered.

Run: python figures/fig1a_anatomy.py
"""
from __future__ import annotations

import math

import numpy as np
from common import INK, INK_2, plt, save
from matplotlib.patches import Circle, Polygon, Rectangle

from geometry import LAYERS, PLEXUS, TARGET_DEPTH_MM
from protocols import REFERENCE_PROTOCOL

ART, ART_EDGE = "#cc2f2f", "#8f1f1f"
VEN, VEN_EDGE = "#2f6fb5", "#1d4876"
EPI_FILL, EPI_EDGE = "#f0c9a8", "#d3a079"
DERM_FILL, DERM_LINE = "#fbe7e3", "#f0cac4"
FAT_FILL, FAT_EDGE = "#fdf0a4", "#e0c257"
FASCIA = "#cdbfae"
HAIR = "#b2855a"
RULE = "#9a948c"

DEPTH_MM, WIDTH_MM = 10.4, 18.0
FASCIA_D = 10.0
VOXEL_MM = REFERENCE_PROTOCOL.inplane_mm
_CLIP = None                           # the frame patch; vessels are clipped to it
WIDTH_K = 0.0079                       # drawn mm per sqrt(um): 11 um -> 0.026, 1930 um -> 0.35


def level(name: str):
    """Return the PLEXUS entry of that name."""
    return next(v for v in PLEXUS if v.name == name)


SUP, DEEP = level("superficial (subpapillary) plexus"), level("deep (dermal-subcutaneous) plexus")
TARGET, PERF, CAP = level("subcutaneous vessels"), level("perforators"), level("capillary loops")
SUP_D, DEEP_D = (SUP.top_mm + SUP.bottom_mm) / 2, (DEEP.top_mm + DEEP.bottom_mm) / 2
EPI_BOTTOM, DERM_BOTTOM = LAYERS[0][0], LAYERS[1][0]
ARCADE_D = CAP.bottom_mm + 0.12
TRUNK_D = 9.3


def drawn_width(calibre_um: float) -> float:
    """Return the drawn width in mm, the square root of the true calibre so the hierarchy survives."""
    return WIDTH_K * math.sqrt(calibre_um)


W_PERF = drawn_width(sum(PERF.diameter_um) / 2)
W_SUB_HI, W_SUB_LO = drawn_width(TARGET.diameter_um[1]), drawn_width(TARGET.diameter_um[0])
W_DEEP = drawn_width(sum(DEEP.diameter_um) / 2)
W_SUP = drawn_width(sum(SUP.diameter_um) / 2)
W_CAP = drawn_width(sum(CAP.diameter_um) / 2)


# --- geometry -----------------------------------------------------------------


def smooth(points, n=220):
    """Return a Catmull-Rom spline through the control points, so paths meander instead of kinking."""
    p = np.asarray(points, dtype=float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    per = max(n // (len(p) - 3), 3)
    out = []
    for i in range(1, len(p) - 2):
        t = np.linspace(0, 1, per, endpoint=(i == len(p) - 3))[:, None]
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def normals(path):
    """Return unit normals along a path."""
    d = np.gradient(path, axis=0)
    span = np.hypot(d[:, 0], d[:, 1])
    span[span == 0] = 1.0
    return np.column_stack([-d[:, 1] / span, d[:, 0] / span])


def taper(ax, path, w0, w1, colour, edge, zorder=7):
    """Fill one vessel as a shape that narrows from w0 to w1, with a rounded tip."""
    nrm = normals(path)
    half = np.linspace(w0, w1, len(path))[:, None] / 2
    poly = np.vstack([path + nrm * half, (path - nrm * half)[::-1]])
    body = Polygon(poly, closed=True, facecolor=colour, edgecolor=edge, lw=0.45,
                   joinstyle="round", zorder=zorder)
    tip = Circle(path[-1], w1 / 2, facecolor=colour, edgecolor=edge, lw=0.45, zorder=zorder)
    for patch in (body, tip):
        ax.add_patch(patch)
        if _CLIP is not None:
            patch.set_clip_path(_CLIP)


def vessel_pair(ax, points, w0, w1, sep=None, zorder=7):
    """Draw an artery with its accompanying vein along one course, both tapering."""
    path = smooth(points)
    gap = (w0 + w1) * 0.62 if sep is None else sep
    nrm = normals(path)
    taper(ax, path + nrm * gap / 2, w0, w1, ART, ART_EDGE, zorder)
    taper(ax, path - nrm * gap / 2, w0 * 1.12, w1 * 1.12, VEN, VEN_EDGE, zorder)
    return path


def meander(x0, x1, depth, swing, lobes, rng, n=260):
    """Return an irregular horizontal course, which reads as a plexus where a sine reads as a graph."""
    xs = np.linspace(x0, x1, lobes)
    ys = depth + rng.uniform(-swing, swing, lobes)
    ys[0] = ys[-1] = depth
    return smooth(np.column_stack([xs, ys]), n)


def plexus(ax, x0, x1, depth, lobes, swing, width, rng, gap=None):
    """Draw a horizontal plexus as an arterial and a venous course running together."""
    path = meander(x0, x1, depth, swing, lobes, rng)
    gap = width * 1.5 if gap is None else gap
    nrm = normals(path)
    taper(ax, path + nrm * gap / 2, width, width, ART, ART_EDGE, zorder=8)
    taper(ax, path - nrm * gap / 2, width * 1.12, width * 1.12, VEN, VEN_EDGE, zorder=8)
    return path


def sprout(ax, path, fraction, length, angle, w0, rng):
    """Grow one fine branch off a course, tapering to nothing so it reads as a twig, not a stick."""
    i = int(np.clip(fraction, 0.02, 0.98) * (len(path) - 1))
    nrm = normals(path)[i]
    tip = path[i] + nrm * length * np.array([math.cos(angle), math.sin(angle)])
    mid = (path[i] + tip) / 2 + nrm * length * 0.18 * rng.uniform(-1, 1)
    branch = smooth([path[i], mid, tip], n=60)
    taper(ax, branch, w0, w0 * 0.3, ART, ART_EDGE, zorder=6)
    taper(ax, branch - normals(branch) * w0 * 1.5, w0 * 1.1, w0 * 0.33, VEN, VEN_EDGE, zorder=6)


# --- tissue -------------------------------------------------------------------


def rete_ridges(ax, x0, x1, bottom, amplitude=0.06, period=0.8):
    """Fill the epidermis above a scalloped dermo-epidermal junction."""
    x = np.linspace(x0, x1, 1200)
    y = bottom + amplitude * (1 + np.sin(2 * np.pi * x / period)) / 2
    ax.fill_between(x, 0.0, y, color=EPI_FILL, zorder=3)
    ax.plot(x, y, color=EPI_EDGE, lw=0.7, zorder=4)


def fat_lobules(ax, x0, x1, top, bottom, rng, clip, n=900):
    """Pack the hypodermis with adipose lobules, as the anatomical illustrations do."""
    placed = []
    for _ in range(n):
        r = rng.uniform(0.16, 0.34)
        x, y = rng.uniform(x0 - r, x1 + r), rng.uniform(top + r * 0.2, bottom - r * 0.2)
        if all((x - px) ** 2 + (y - py) ** 2 > (r + pr) ** 2 * 0.6 for px, py, pr in placed):
            placed.append((x, y, r))
            lobule = Circle((x, y), r, facecolor=FAT_FILL, edgecolor=FAT_EDGE, lw=0.55, zorder=1)
            ax.add_patch(lobule)
            lobule.set_clip_path(clip)


def collagen(ax, x0, x1, top, bottom, rng, n=30):
    """Suggest the dermal collagen with faint horizontal bundles."""
    for depth in rng.uniform(top + 0.1, bottom - 0.04, n):
        x = np.linspace(x0, x1, 200)
        ax.plot(x, depth + 0.035 * np.sin(2 * np.pi * x / rng.uniform(2.5, 5.0)
                                          + rng.uniform(0, 6)),
                color=DERM_LINE, lw=0.7, zorder=2)


def hair_follicle(ax, x, bottom, tilt):
    """Draw a follicle running obliquely into the dermis, with its shaft above the surface."""
    t = np.linspace(0, 1, 70)
    cx, cy = x + tilt * t, bottom * t
    half = 0.14 * (1 - 0.28 * t) + 0.17 * np.exp(-((t - 0.94) / 0.1) ** 2)
    ax.add_patch(Polygon(np.column_stack([np.r_[cx - half, cx[::-1] + half[::-1]],
                                          np.r_[cy, cy[::-1]]]), closed=True,
                         facecolor="#ecd3b2", edgecolor=HAIR, lw=0.7, zorder=5))
    shaft = np.linspace(0, 1, 40)
    ax.plot(x - 0.5 * shaft, -1.1 * shaft, color=HAIR, lw=2.2, solid_capstyle="round", zorder=6)


# --- the tree -----------------------------------------------------------------


def perforator_unit(ax, x):
    """Draw one perforator pair crossing the fascia and branching through the fat."""
    trunk = vessel_pair(ax, [(x + 0.45, DEPTH_MM + 0.2), (x + 0.3, TRUNK_D),
                             (x - 0.1, 7.3), (x, 6.5)], W_PERF, W_SUB_HI)
    for side in (-1, 1):
        node = (x + side * 3.1, 4.3)
        vessel_pair(ax, [trunk[-1], (x + side * 1.3, 5.6), node], W_SUB_HI, W_SUB_HI * 0.72)
        for reach, mid_x in ((3.3, side * 3.9), (1.3, side * 1.9)):
            vessel_pair(ax, [node, (x + mid_x, 3.3), (x + side * reach, DEEP_D + 0.08)],
                        W_SUB_HI * 0.72, W_SUB_LO)


def capillary_loops(ax, x0, x1, n, crest, base, rng):
    """Draw hairpin loops rising into the papillae, varied in height so they do not read as a comb."""
    for x in np.linspace(x0, x1, n) + rng.uniform(-0.12, 0.12, n):
        height = (base - crest) * rng.uniform(0.75, 1.05)
        width = rng.uniform(0.085, 0.125)
        t = np.linspace(0, np.pi, 44)
        for sign, colour, edge in ((-1, ART, ART_EDGE), (1, VEN, VEN_EDGE)):
            limb = np.column_stack([x + sign * (width / 2 + width * np.cos(t) * 0.5),
                                    base - height * np.sin(t)])
            taper(ax, limb, W_CAP, W_CAP, colour, edge, zorder=9)


# --- labels -------------------------------------------------------------------


def level_label(ax, depth, name, calibre, x_tick=-0.95, x_text=-1.45):
    """Name one vascular level at its own depth, with its calibre under it."""
    ax.plot([x_text + 0.1, x_tick], [depth, depth], color=RULE, lw=0.7, zorder=6)
    ax.annotate(name, (x_text, depth - 0.1), ha="right", va="bottom", fontsize=8.4, color=INK)
    ax.annotate(calibre, (x_text, depth + 0.1), ha="right", va="top", fontsize=7.7, color=INK_2)


def layer_bracket(ax, top, bottom, name, x=WIDTH_MM + 0.45, tip=0.22):
    """Bracket one anatomical layer on the right."""
    ax.plot([x, x + tip, x + tip, x], [top, top, bottom, bottom], color=RULE, lw=1.0, zorder=6)
    ax.annotate(name, (x + tip + 0.25, (top + bottom) / 2), ha="left", va="center", fontsize=8.8,
                color=INK)


def calibre_inset(ax):
    """Put the target vessels beside the voxel at equal scale, above the frame."""
    scale, x, y = 2.0, 1.6, -1.0
    side = VOXEL_MM * scale
    ax.add_patch(Rectangle((x, y - side / 2), side, side, facecolor="white", edgecolor=INK,
                           lw=1.2, zorder=10))
    ax.annotate(f"voxel {VOXEL_MM * 1000:.0f} µm", (x + side / 2, y - side / 2 - 0.1),
                ha="center", va="bottom", fontsize=7.4, color=INK, zorder=10)
    x += side + 0.5
    for d_um in (TARGET.diameter_um[0], 300.0, TARGET.diameter_um[1]):
        r = d_um / 2000 * scale
        ax.add_patch(Circle((x + r, y), r, facecolor=ART, alpha=0.55, edgecolor=ART_EDGE, lw=0.8,
                            zorder=10))
        ax.annotate(f"{d_um:.0f}", (x + r, y + 0.6), ha="center", va="top", fontsize=7.2,
                    color=INK_2, zorder=10)
        x += 2 * r + 0.26
    ax.annotate("µm, equal scale", (x - 0.08, y), ha="left", va="center", fontsize=7.6,
                color=INK_2, zorder=10)


# --- the frame ----------------------------------------------------------------


def draw(ax):
    """Draw the single frame: tissue, the vascular tree, and the few labels it needs."""
    global _CLIP
    rng = np.random.default_rng(3)
    frame = Rectangle((0, 0), WIDTH_MM, DEPTH_MM, facecolor="none", edgecolor="none", zorder=0)
    ax.add_patch(frame)
    _CLIP = frame
    ax.add_patch(Rectangle((0, EPI_BOTTOM), WIDTH_MM, DERM_BOTTOM - EPI_BOTTOM,
                           facecolor=DERM_FILL, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, DERM_BOTTOM), WIDTH_MM, FASCIA_D - DERM_BOTTOM,
                           facecolor="#fffbe8", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, FASCIA_D), WIDTH_MM, DEPTH_MM - FASCIA_D, facecolor=FASCIA,
                           edgecolor="none", zorder=0))
    collagen(ax, 0, WIDTH_MM, EPI_BOTTOM, DERM_BOTTOM, rng)
    fat_lobules(ax, 0, WIDTH_MM, DERM_BOTTOM, FASCIA_D, rng, frame)
    rete_ridges(ax, 0, WIDTH_MM, EPI_BOTTOM)
    for x, tilt in ((5.3, 0.75), (12.6, -0.65)):
        hair_follicle(ax, x, DERM_BOTTOM - 0.12, tilt)
    ax.plot([0, WIDTH_MM], [FASCIA_D, FASCIA_D], color="#a89a86", lw=1.1, zorder=5)

    plexus(ax, 0.3, WIDTH_MM - 0.3, TRUNK_D, 11, 0.22, W_PERF * 0.8, rng)
    for x in (4.3, 13.5):
        perforator_unit(ax, x)

    deep = plexus(ax, 0.3, WIDTH_MM - 0.3, DEEP_D, 9, 0.11, W_DEEP, rng)
    for frac in (0.12, 0.33, 0.56, 0.78, 0.93):
        sprout(ax, deep, frac, 0.75, rng.uniform(-0.45, 0.45) + math.pi / 2, W_DEEP * 0.7, rng)
    for x in (1.9, 5.2, 8.8, 12.3, 15.9):
        vessel_pair(ax, [(x, DEEP_D), (x + 0.25, 1.9), (x - 0.15, SUP_D + 0.2), (x, SUP_D)],
                    W_DEEP * 0.8, W_SUP)
        vessel_pair(ax, [(x, SUP_D), (x + 0.12, 0.75), (x, ARCADE_D)], W_SUP, W_CAP * 1.3)
    plexus(ax, 0.3, WIDTH_MM - 0.3, SUP_D, 13, 0.055, W_SUP, rng)
    plexus(ax, 0.8, WIDTH_MM - 0.8, ARCADE_D, 17, 0.028, W_CAP * 1.15, rng)
    capillary_loops(ax, 1.0, WIDTH_MM - 1.0, 13, CAP.top_mm + 0.05, CAP.bottom_mm + 0.1, rng)

    ax.plot([0, WIDTH_MM, WIDTH_MM, 0, 0], [0, 0, DEPTH_MM, DEPTH_MM, 0], color="#c9c2b8", lw=0.9,
            zorder=6)

    for depth, name, calibre in (
        (CAP.top_mm + 0.15, "capillary loops",
         f"{CAP.diameter_um[0]:.0f}–{CAP.diameter_um[1]:.0f} µm"),
        (SUP_D, "superficial plexus", f"{SUP.diameter_um[0]:.0f}–{SUP.diameter_um[1]:.0f} µm"),
        (DEEP_D, "deep plexus", f"{DEEP.diameter_um[0]:.0f}–{DEEP.diameter_um[1]:.0f} µm"),
        (5.0, "subcutaneous vessels",
         f"{TARGET.diameter_um[0]:.0f}–{TARGET.diameter_um[1]:.0f} µm"),
        (TRUNK_D, "perforator trunks",
         f"{PERF.diameter_um[0] / 1000:.1f}–{PERF.diameter_um[1] / 1000:.1f} mm"),
    ):
        level_label(ax, depth, name, calibre)

    for depth in (0, DERM_BOTTOM, FASCIA_D):
        ax.plot([-0.55, -0.3], [depth, depth], color=RULE, lw=0.9, zorder=6)
        ax.annotate(f"{depth:g}", (-0.62, depth), ha="right", va="center", fontsize=7.6,
                    color=INK_2, zorder=6)
    ax.plot([-0.42, -0.42], [0, FASCIA_D], color=RULE, lw=0.9, zorder=6)
    ax.annotate("mm", (-0.42, -0.3), ha="center", va="bottom", fontsize=7.6, color=INK_2)

    layer_bracket(ax, 0, EPI_BOTTOM, "epidermis")
    layer_bracket(ax, EPI_BOTTOM, DERM_BOTTOM, "dermis")
    layer_bracket(ax, DERM_BOTTOM, FASCIA_D, "hypodermis")
    ax.annotate("fascia", (WIDTH_MM + 0.7, (FASCIA_D + DEPTH_MM) / 2), ha="left", va="center",
                fontsize=8.0, color=INK_2)

    t_top, t_bot = TARGET_DEPTH_MM
    ax.plot([WIDTH_MM + 3.0, WIDTH_MM + 3.2, WIDTH_MM + 3.2, WIDTH_MM + 3.0],
            [t_top, t_top, t_bot, t_bot], color=INK, lw=1.3, zorder=7)
    ax.annotate(f"TARGET  {t_top:g}–{t_bot:g} mm", (WIDTH_MM + 3.55, (t_top + t_bot) / 2),
                rotation=90, ha="center", va="center", fontsize=8.8, color=INK, weight="bold",
                zorder=7)
    calibre_inset(ax)

    ax.set(xlim=(-5.2, WIDTH_MM + 4.2), ylim=(DEPTH_MM + 0.35, -1.95))
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    """Build the single-frame anatomy figure and print the table it was drawn from."""
    fig = plt.figure(figsize=(12.0, 7.0))
    draw(fig.add_axes((0.01, 0.045, 0.98, 0.9)))
    fig.suptitle("Target-calibre cutaneous vessels lie below the dermal plexus, in fat",
                 fontsize=12.5, x=0.035, ha="left", y=0.975, color=INK)

    print(f"{'level':<40}{'depth (mm)':>16}{'diameter (um)':>18}{'drawn (mm)':>13}")
    for entry in PLEXUS:
        print(f"  {entry.name:<38}{entry.top_mm:>6.1f}–{entry.bottom_mm:<9.1f}"
              f"{entry.diameter_um[0]:>8.0f}–{entry.diameter_um[1]:<9.0f}"
              f"{drawn_width(sum(entry.diameter_um) / 2):>12.3f}")
    print(f"\nWidths are drawn as {WIDTH_K} x sqrt(calibre in um): a {PERF.diameter_um[1] / 1000:.1f} mm "
          f"perforator is {drawn_width(PERF.diameter_um[1]) / drawn_width(CAP.diameter_um[1]):.0f}x the width "
          f"of an {CAP.diameter_um[1]:.0f} um loop,\nagainst {PERF.diameter_um[1] / CAP.diameter_um[1]:.0f}x "
          f"in truth: exaggerated, but ordered.")
    print(f"The dermal plexus tops out at {DEEP.diameter_um[1]:.0f} um, so the "
          f"{TARGET.diameter_um[0]:.0f}-{TARGET.diameter_um[1]:.0f} um target band lies above every measured "
          f"dermal level and below\nthe {PERF.diameter_um[1] / 1000:.1f} mm perforators: the calibre gap no "
          f"in vivo study resolves.")
    print(f"At the {VOXEL_MM * 1000:.0f} um voxel of {REFERENCE_PROTOCOL.name}, a 300 um vessel spans "
          f"{300 / (VOXEL_MM * 1000):.1f} voxels across.")

    save(fig, "fig1a_anatomy",
         note="Anatomy from the literature (Braverman, J Invest Dermatol Symp Proc 2000; perforator "
              "calibres J Reconstr Microsurg 2021). Depths to scale, widths exaggerated. No measured data.")


if __name__ == "__main__":
    main()
