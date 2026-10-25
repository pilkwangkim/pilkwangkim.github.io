"""Geometry-first mathematical plates for the Galois article (figures 5, 6, 8, 9)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from galois_visual_style import (
    BLUE, GOLD, GRID, INK, MUTED, TEAL, arrow, canvas, clean, label, panel, point,
)


def _complex_axes(ax, limits=1.85, real_label_y=-.16):
    ax.set(xlim=(-limits, limits), ylim=(-limits, limits), aspect="equal")
    ax.axis("off")
    ax.plot([-limits + .05, limits - .08], [0, 0], color=GRID, lw=1.05, zorder=0)
    ax.plot([0, 0], [-limits + .08, limits - .05], color=GRID, lw=1.05, zorder=0)
    label(ax, limits - .11, real_label_y, "Re", ha="right", color=MUTED,
          size=20 if ax.figure._mobile else 18)
    label(ax, .09, limits - .09, "Im", color=MUTED,
          size=20 if ax.figure._mobile else 18)


def _circular_arrow(ax, radius, start, stop, color=TEAL, lw=1.65):
    angles = np.linspace(start, stop, 90)
    xy = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=2)
    arrow(ax, xy[-4], xy[-1], color=color, lw=lw, scale=14, zorder=3)


def _cubic_roots(ax):
    radius = 2 ** (1 / 3)
    angles = 2 * np.pi * np.arange(3) / 3
    roots = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    names = [r"$\alpha$", r"$\omega\alpha$", r"$\omega^2\alpha$"]
    positions = [(1.43, .24), (-.85, 1.41), (-.85, -1.43)]
    for xy, name, position in zip(roots, names, positions):
        point(ax, xy, color=BLUE, size=78)
        label(ax, *position, name, size=24 if ax.figure._mobile else 22, ha="center")
    return radius, roots


def fig05(mobile):
    fig = canvas(mobile, 960 if mobile else 550)
    if mobile:
        rectangles = [(.13, .56, .76, .34), (.13, .10, .76, .34)]
    else:
        rectangles = [(.07, .16, .40, .69), (.56, .16, .40, .69)]
    names = [r"(a) $\rho$: rotation", r"(b) $c$: conjugation"]
    axes = [panel(fig, rect, name) for rect, name in zip(rectangles, names)]
    for ax in axes:
        _complex_axes(ax)
    radius, roots = _cubic_roots(axes[0])
    for j in range(3):
        _circular_arrow(axes[0], radius, 2*np.pi*j/3 + .16,
                        2*np.pi*(j+1)/3 - .16)
    label(axes[0], 0, 0, r"$120^\circ$", color=TEAL, ha="center",
          size=24 if mobile else 22)
    _cubic_roots(axes[1])
    axes[1].plot([-1.73, 1.70], [0, 0], color=BLUE, alpha=.40, lw=2.3, zorder=0)
    # The same pair is exchanged in both directions; the positive real root stays put.
    for start, end, bend in [(roots[1], roots[2], -.20), (roots[2], roots[1], -.20)]:
        direction = (end - start) / np.linalg.norm(end - start)
        arrow(axes[1], start + .13*direction, end - .13*direction,
              color=TEAL, curve=bend, lw=1.8, scale=17)
    axes[1].add_patch(Circle(roots[0], .115, fill=False, edgecolor=TEAL, lw=1.4, zorder=4))
    label(axes[1], .45, .77, r"$z\mapsto\bar z$", color=TEAL,
          ha="center", size=23 if mobile else 21)
    fig.text(.5, .025, r"$x^3-2=0,\qquad\alpha=\sqrt[3]{2}$",
             ha="center", va="bottom", fontsize=21 if mobile else 19, color=MUTED)
    fig._visual_semantics = {"roots": roots.tolist(), "rho_degrees": 120,
                             "conjugation": "reflection across real axis"}
    return fig


def fig06(mobile):
    """A nontrivial prime-degree radical step, not an unconditional Cp assertion."""
    fig = canvas(mobile, 650 if mobile else 560)
    rect = (.10, .14, .80, .75) if mobile else (.21, .10, .58, .81)
    ax = panel(fig, rect)
    ax.set(xlim=(-1.75, 1.75), ylim=(-1.75, 1.75), aspect="equal")
    ax.axis("off")
    radius = 2 ** .2
    angles = 2*np.pi*np.arange(5)/5
    roots = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    full = np.linspace(0, 2*np.pi, 300)
    ax.plot(radius*np.cos(full), radius*np.sin(full), color=GRID, lw=1.2)
    for xy in roots:
        ax.plot([0, xy[0]], [0, xy[1]], color=GRID, lw=.8, ls=(0, (2, 4)), zorder=0)
    positions = [(1.45, .02), (.40, 1.46), (-1.28, .90), (-1.28, -.90), (.40, -1.47)]
    for j, (xy, position) in enumerate(zip(roots, positions)):
        point(ax, xy, color=BLUE, size=85)
        text = r"$\beta$" if j == 0 else (
            r"$\zeta_5\beta$" if j == 1 else rf"$\zeta_5^{j}\beta$")
        label(ax, *position, text, ha="center", size=24 if mobile else 22)
        _circular_arrow(ax, radius, angles[j]+.12,
                        angles[j]+2*np.pi/5-.12,
                        color=TEAL if j == 0 else MUTED, lw=2.1 if j == 0 else 1.2)
    _circular_arrow(ax, .31, .03, 2*np.pi/5-.03, color=TEAL, lw=1.2)
    label(ax, .47, .37, r"$72^\circ$", color=TEAL, ha="center",
          size=21 if mobile else 19)
    label(ax, -.08, -.23, r"$\beta^5=2$", color=MUTED, ha="center",
          size=23 if mobile else 21)
    fig.text(.5, .97, r"$K=\mathbb{Q}(\zeta_5),\qquad\beta=\sqrt[5]{2}$",
             fontsize=22 if mobile else 21, ha="center", va="top")
    fig.text(.5, .035, r"$\sigma(\beta)=\zeta_5\beta,\qquad\sigma^5=e$",
             fontsize=22 if mobile else 21, ha="center", va="bottom", color=TEAL)
    fig._visual_semantics = {"base": "Q(zeta5)", "beta": "5th root of 2",
                             "extension_degree": 5, "rotation_degrees": 72,
                             "group": "Gal(K(beta)/K) = C5"}
    return fig


def commutator_tracks():
    g = (2, 1, 4, 3, 5)
    h = (2, 1, 5, 4, 3)
    trajectories = []
    for source in range(1, 6):
        row = [source]
        for operation in (h, g, h, g):
            row.append(operation[row[-1] - 1])
        trajectories.append(row)
    assert trajectories == [[1, 2, 1, 2, 1], [2, 1, 2, 1, 2],
                             [3, 5, 5, 3, 4], [4, 4, 3, 5, 5], [5, 3, 4, 4, 3]]
    return trajectories


def fig08(mobile):
    fig = canvas(mobile, 910 if mobile else 560)
    ax = fig.add_axes((.09, .16, .84, .66) if mobile else (.11, .16, .78, .64))
    ax.axis("off")
    colors = ["#87959e", "#87959e", BLUE, TEAL, GOLD]
    tracks = commutator_tracks()
    if mobile:
        ax.set(xlim=(.1, 5.9), ylim=(4.65, -.65))
        for stage in range(5):
            ax.plot([.72, 5.28], [stage, stage], color=GRID, lw=.85, zorder=0)
        for stage, operation in enumerate([r"$h^{-1}$", r"$g^{-1}$", r"$h$", r"$g$"]):
            label(ax, .36, stage+.50, operation, ha="center", size=23, color=MUTED)
    else:
        ax.set(xlim=(-.50, 4.50), ylim=(.45, 5.95))
        for stage in range(5):
            ax.plot([stage, stage], [.80, 5.22], color=GRID, lw=.85, zorder=0)
        for stage, operation in enumerate([r"$h^{-1}$", r"$g^{-1}$", r"$h$", r"$g$"]):
            label(ax, stage+.50, 5.65, operation, ha="center", size=22, color=MUTED)
    t = np.linspace(0, 1, 100)
    ease = 3*t*t - 2*t*t*t
    for source, (track, color) in enumerate(zip(tracks, colors), 1):
        width = 1.25 if source < 3 else 2.4
        for stage in range(4):
            value = track[stage] + (track[stage+1]-track[stage])*ease
            if mobile:
                ax.plot(value, stage+t, color=color, lw=width, zorder=2 if source < 3 else 4)
            else:
                ax.plot(stage+t, 6-value, color=color, lw=width, zorder=2 if source < 3 else 4)
        if mobile:
            ax.scatter(track, range(5), color=color, s=28 if source < 3 else 40,
                       edgecolors="white", linewidths=.8, zorder=5)
            label(ax, source, -.35, str(source), color=color, ha="center", size=25)
            label(ax, track[-1], 4.35, str(track[-1]), color=color, ha="center", size=25)
        else:
            ax.scatter(range(5), 6-np.array(track), color=color, s=25 if source < 3 else 40,
                       edgecolors="white", linewidths=.8, zorder=5)
            label(ax, -.27, 6-source, str(source), color=color, ha="center", size=24)
            label(ax, 4.27, 6-track[-1], str(track[-1]), color=color, ha="center", size=24)
    if mobile:
        fig.text(.5, .965, r"$g=(12)(34)$", fontsize=24, ha="center", va="top", color=BLUE)
        fig.text(.5, .91, r"$h=(12)(35)$", fontsize=24, ha="center", va="top", color=TEAL)
    else:
        fig.text(.5, .945, r"$g=(12)(34),\qquad h=(12)(35)$",
                 fontsize=23, ha="center", va="top")
    fig.text(.5, .045, r"$[g,h]=ghg^{-1}h^{-1}=(345)$", color=INK,
             fontsize=25 if mobile else 24, ha="center", va="bottom")
    fig._visual_semantics = {"operation_order": ["h^-1", "g^-1", "h", "g"],
                             "tracks": tracks, "commutator": "(345)"}
    return fig


def quintic_roots():
    roots = np.roots([1, 0, 0, 0, -5, 1])
    real = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-8)
    pair = sorted((complex(z) for z in roots if abs(z.imag) >= 1e-8), key=lambda z: z.imag)
    assert len(real) == 3 and len(pair) == 2
    assert np.allclose(pair[0], pair[1].conjugate())
    return real, pair


def fig09(mobile):
    fig = canvas(mobile, 940 if mobile else 570)
    rectangles = ([ (.16, .58, .76, .33), (.16, .075, .76, .36)]
                  if mobile else [(.075, .18, .38, .67), (.57, .16, .38, .71)])
    real, pair = quintic_roots()
    ax = panel(fig, rectangles[0], "(a) Real roots")
    x = np.linspace(-1.75, 1.75, 800)
    ax.plot(x, x**5-5*x+1, color=BLUE, lw=2.1, zorder=2)
    ax.set(xlim=(-1.8, 1.8), ylim=(-7, 9.3))
    clean(ax)
    ax.spines["bottom"].set_position("zero")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-5, 5])
    ax.spines["left"].set_color(GRID)
    label(ax, 1.84, -.68, r"$x$", ha="left", color=MUTED, size=21 if mobile else 19)
    label(ax, -.02, 8.00, r"$f(x)$", color=MUTED, size=21 if mobile else 19)
    for j, root in enumerate(real, 1):
        point(ax, (root, 0), color=TEAL, size=76)
        label(ax, root + (.20 if j == 1 else 0), 1.35 if j == 2 else -1.45, rf"$r_{j}$",
              color=TEAL, ha="center", size=24 if mobile else 22)
    for xy, pos, text in [((-1, 5), (-1.10, 6.45), r"$(-1,5)$"),
                          ((1, -3), (.82, -4.90), r"$(1,-3)$")]:
        point(ax, xy, color=BLUE, size=46)
        label(ax, *pos, text, color=BLUE, ha="center", size=21 if mobile else 19)
    cx = panel(fig, rectangles[1], "(b) Complex roots")
    _complex_axes(cx, 1.95, real_label_y=.30)
    for j, root in enumerate(real, 1):
        point(cx, (root, 0), color=TEAL, size=76)
        label(cx, root, -.29, rf"$r_{j}$", color=TEAL, ha="center",
              size=23 if mobile else 21)
    for root, text, yoffset in [(pair[1], r"$z$", -.12), (pair[0], r"$\bar z$", -.05)]:
        point(cx, (root.real, root.imag), color=GOLD, size=84)
        label(cx, root.real+.22, root.imag+yoffset, text, color=GOLD,
              size=25 if mobile else 23)
    cx.add_patch(FancyArrowPatch((pair[1].real, pair[1].imag-.17),
                                (pair[0].real, pair[0].imag+.17),
                                arrowstyle="<->", mutation_scale=16, color=GOLD,
                                linewidth=1.55, linestyle=(0, (4, 3)), zorder=1))
    label(cx, -.42, .66, r"$c$", color=GOLD, size=25 if mobile else 23, ha="center")
    fig.text(.5, .025, r"$f(x)=x^5-5x+1$", ha="center", va="bottom",
             color=MUTED, fontsize=22 if mobile else 20)
    fig._visual_semantics = {"real_roots": real,
                             "nonreal_roots": [[z.real, z.imag] for z in pair],
                             "complex_conjugation": "one transposition"}
    return fig
