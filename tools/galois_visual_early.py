"""Geometric figures for radical choices, coordinates, and fixed fields."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.path import Path

from galois_visual_style import (
    BLUE, GOLD, GRID, INK, MUTED, PALE, RED, TEAL,
    arrow, canvas, clean, label, panel, point,
)


def fig01(mobile):
    """Two intersections make the two square-root choices visible."""
    fig = canvas(mobile, 510 if mobile else 470)
    ax = panel(fig, [.15 if mobile else .10, .15, .78 if mobile else .85, .76])
    x = np.linspace(-2.05, 2.05, 500)
    ax.plot(x, x*x, color=BLUE, lw=2.3)
    ax.axhline(2, color=GOLD, lw=1.6)
    ax.axhline(0, color=GRID, lw=.9)
    ax.axvline(0, color=GRID, lw=.9)
    root = np.sqrt(2)
    for a, color in [(-root, TEAL), (root, TEAL)]:
        ax.plot([a, a], [0, 2], color=GRID, ls=(0, (3, 3)), lw=1.2)
        point(ax, (a, 2), color, size=100 if mobile else 82)
    ax.set(xlim=(-2.25, 2.25), ylim=(-.20, 4.55))
    ax.set_xticks([-root, 0, root], [r"$-\sqrt{2}$", "0", r"$\sqrt{2}$"])
    ax.set_yticks([0, 4], ["", "4"])
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.set_xlabel(r"$x$", fontsize=22, labelpad=16)
    ax.xaxis.set_label_coords(1.01, -.045)
    ax.set_ylabel(r"$y$", fontsize=22, rotation=0)
    ax.yaxis.set_label_coords(.51, 1.01)
    clean(ax)
    label(ax, 1.18, 4.12, r"$y=x^2$", color=BLUE, ha="center",
          size=23 if mobile else 22)
    label(ax, .55, 2.38, r"$y=2$", color=GOLD, ha="center",
          size=23 if mobile else 22)
    ax.tick_params(axis="x", pad=9, labelsize=21 if mobile else 19)
    for item in ax.get_xticklabels():
        if "sqrt" in item.get_text():
            item.set_color(TEAL)
    return fig


def _coordinate_panel(fig, rect, which):
    is_multiply = which == "multiply"
    color = BLUE if is_multiply else TEAL
    formula = (r"(a) $M_\alpha(a,b)=(2b,a)$" if is_multiply
               else r"(b) $\sigma(a,b)=(a,-b)$")
    ax = panel(fig, rect, formula)
    ax.set(xlim=(-1.3, 2.65), ylim=(-1.55, 2.4))
    ax.set_aspect("equal", adjustable="box")
    clean(ax, grid=True)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1, 2], ["−1", "", "1", "2"])
    ax.spines[["left", "bottom"]].set_position("zero")
    ax.spines[["left", "bottom"]].set_color("#a7b6bf")
    # Unit square and its exact image under the two linear operators.
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    matrix = np.array([[0, 2], [1, 0]]) if is_multiply else np.diag([1, -1])
    transformed = square @ matrix.T
    ax.add_patch(Polygon(transformed, closed=True, facecolor=color,
                         edgecolor=color, linewidth=1.4, alpha=.12, zorder=2))
    ax.add_patch(Polygon(square, closed=True, facecolor="none", edgecolor=MUTED,
                         linewidth=1.0, linestyle=(0, (4, 3)), zorder=3))
    u = np.array([1, 1]); image = matrix @ u
    arrow(ax, (0, 0), u, MUTED, lw=1.5, scale=13,
          linestyle=(0, (3, 3)), zorder=4)
    arrow(ax, (0, 0), image, color, lw=2.2, scale=17, zorder=5)
    point(ax, u, MUTED, size=48)
    point(ax, image, color, size=70)
    label(ax, .86, 1.27, r"$u$", color=MUTED, ha="right", size=23)
    if is_multiply:
        label(ax, 2.21, 1.30, r"$M_\alpha u$", color=color, ha="right",
              size=24 if fig._mobile else 23)
    else:
        label(ax, 1.11, -1.14, r"$\sigma(u)$", color=color,
              size=24 if fig._mobile else 23)
    label(ax, 2.55, -.18, r"$a$", color=MUTED, ha="right", size=21)
    label(ax, .16, 2.23, r"$b$", color=MUTED, ha="left", size=21)
    return ax


def fig02(mobile):
    """Show stretching/interchange versus reflection, not another formula list."""
    fig = canvas(mobile, 920 if mobile else 515)
    rects = ([.13, .57, .82, .345], [.13, .105, .82, .345]) if mobile else (
        [.07, .12, .39, .73], [.57, .12, .39, .73])
    _coordinate_panel(fig, rects[0], "multiply")
    _coordinate_panel(fig, rects[1], "conjugate")
    return fig


def _real_axis(ax, xmin, xmax, family_labels=True):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.08, 1.98)
    ax.axis("off")
    arrow(ax, (xmin+.03, 0), (xmax-.03, 0), color=GRID, lw=1.2, scale=12)
    ax.plot([0, 0], [-.045, .045], color=MUTED, lw=1)
    if xmin < 0 < xmax:
        label(ax, 0, -.24, "0", size=19 if ax.figure._mobile else 17.5,
              color=MUTED, ha="center")
    root2, root3 = np.sqrt(2), np.sqrt(3)
    for a, color, text, yy in [
            (-root3, GOLD, r"$-\sqrt{3}$", -.84),
            (-root2, BLUE, r"$-\sqrt{2}$", -.25),
            (root2, BLUE, r"$\sqrt{2}$", -.25),
            (root3, GOLD, r"$\sqrt{3}$", -.84)]:
        if not xmin < a < xmax:
            continue
        point(ax, (a, 0), color, size=68 if ax.figure._mobile else 65)
        label(ax, a, yy, text, color=color, ha="center",
              size=23 if ax.figure._mobile else 22)
        if abs(yy) > .4:
            ax.plot([a, a], [-.06, yy+.20], color=color, lw=.7, alpha=.6)
    return root2, root3


def _double_arc(ax, x0, x1, height, color):
    span = x1-x0
    path = Path([(x0, .055), (x0+.22*span, height),
                 (x1-.22*span, height), (x1, .055)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    p = FancyArrowPatch(path=path, arrowstyle="<->",
                        mutation_scale=14, lw=1.7, color=color)
    ax.add_patch(p)
    return p


def fig03(mobile):
    """True real-axis positions, and a clearly marked zoom for the invalid swap."""
    fig = canvas(mobile, 800 if mobile else 480)
    rects = ([.065, .58, .87, .33], [.13, .105, .78, .33]) if mobile else (
        [.055, .14, .43, .68], [.57, .14, .38, .68])
    ax = panel(fig, rects[0], "(a) Independent sign flips")
    r2, r3 = _real_axis(ax, -2.15, 2.15)
    _double_arc(ax, -r2, r2, .95, BLUE)
    _double_arc(ax, -r3, r3, 1.75, GOLD)
    label(ax, 0, 1.02, r"$\sigma$", color=BLUE, ha="center", size=26)
    label(ax, 0, 1.64, r"$\tau$", color=GOLD, ha="center", size=26)
    ax2 = panel(fig, rects[1], "(b) Invalid swap · zoom")
    ax2.axis("off"); ax2.set_xlim(1.24, 1.91); ax2.set_ylim(-.96, 1.98)
    arrow(ax2, (1.25, 0), (1.9, 0), color=GRID, lw=1.2, scale=12)
    for a, c, text in [(r2, BLUE, r"$\sqrt{2}$"), (r3, GOLD, r"$\sqrt{3}$")]:
        point(ax2, (a, 0), c, size=75)
        label(ax2, a, -.25, text, color=c, ha="center", size=23)
    _double_arc(ax2, r2, r3, 1.55, RED)
    label(ax2, (r2+r3)/2, .60, r"$\times$", color=RED,
          ha="center", size=32)
    label(ax2, r2, -.71, r"$r^2=2$", color=BLUE, ha="center", size=22)
    label(ax2, r3, -.71, r"$r^2=3$", color=GOLD, ha="center", size=22)
    return fig


def fig04(mobile):
    """One matched Hasse diagram; opposite inclusion directions share its edges."""
    fig = canvas(mobile, 710 if mobile else 590)
    ax = fig.add_axes([.10, .06, .80, .88])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    positions = [(.5, .85), (.19, .50), (.50, .50), (.81, .50), (.5, .15)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
    for hi, lo in edges:
        a, b = np.array(positions[hi]), np.array(positions[lo])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=1.4, zorder=1)
        direction = a - b
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal) * (.013 if mobile else .009)
        mid = (a+b)/2
        # Blue arrows follow field inclusion upward; teal arrows reverse it.
        start, end = mid - .095*direction, mid + .095*direction
        arrow(ax, start+normal, end+normal, BLUE, scale=13, lw=1.5, zorder=2)
        arrow(ax, end-normal, start-normal, TEAL, scale=13, lw=1.5, zorder=2)
    fields = [r"$L$", r"$\mathbb{Q}(\sqrt{2})$",
              r"$\mathbb{Q}(\sqrt{6})$", r"$\mathbb{Q}(\sqrt{3})$",
              r"$\mathbb{Q}$"]
    groups = [r"$\{e\}$", r"$\langle\tau\rangle$",
              r"$\langle\sigma\tau\rangle$", r"$\langle\sigma\rangle$", r"$G$"]
    bg = {"facecolor": "white", "edgecolor": "none", "pad": 1.6}
    for j, ((x, y), field, group) in enumerate(zip(positions, fields, groups)):
        point(ax, (x, y), MUTED, size=20)
        label(ax, x, y+.055, field, color=BLUE, ha="center",
              size=22 if mobile else 24, bbox=bg)
        label(ax, x, y-.057, group, color=TEAL, ha="center",
              size=22 if mobile else 24, bbox=bg)
    # These side arrows expose the anti-isomorphism without duplicating graphs.
    outside_arrow = -.005 if mobile else -.035
    outside_label = -.065 if mobile else -.085
    arrow(ax, (outside_arrow, .24), (outside_arrow, .76), BLUE, lw=1.7, scale=15, clip_on=False)
    arrow(ax, (1-outside_arrow, .76), (1-outside_arrow, .24), TEAL, lw=1.7, scale=15, clip_on=False)
    label(ax, outside_label, .50, "Field inclusion", color=BLUE, ha="center",
          rotation=90, size=20 if mobile else 18, clip_on=False)
    label(ax, 1-outside_label, .50, "Subgroup inclusion", color=TEAL, ha="center",
          rotation=-90, size=20 if mobile else 18, clip_on=False)
    label(ax, .5, 1.02, r"$E\ \longleftrightarrow\ \operatorname{Gal}(L/E)$",
          ha="center", va="top", color=MUTED, size=23 if mobile else 24)
    return fig
