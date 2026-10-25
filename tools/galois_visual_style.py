"""Shared, restrained scientific-figure style for the Galois article."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import FancyArrowPatch

INK, MUTED = "#243746", "#697982"
BLUE, TEAL, GOLD, RED = "#315f87", "#16877d", "#b17b3f", "#b4544b"
GRID, PALE = "#dce4e8", "#f3f6f7"


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 18,
        "mathtext.fontset": "stix", "svg.fonttype": "path",
        "svg.hashsalt": "galois-scientific-redesign-v2",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": .85, "lines.solid_capstyle": "round",
    })


def canvas(mobile, height, width=None):
    width = width or (480 if mobile else 960)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig._mobile, fig._width, fig._height = mobile, width, height
    return fig


def label(ax, x, y, value, size=None, color=INK, ha="left", va="center", **kwargs):
    value = re.sub(r"\\sqrt([0-9])", r"\\sqrt{\1}", value)
    value = re.sub(r"\\mathbb ([A-Z])", r"\\mathbb{\1}", value)
    return ax.text(x, y, value, fontsize=size or (20 if ax.figure._mobile else 18),
                   color=color, ha=ha, va=va, **kwargs)


def panel(fig, rect, name=None, color=INK):
    """A normal data axis; panel labels are concise, above the plotting region."""
    ax = fig.add_axes(rect)
    if name:
        label(ax, 0, 1.10, name, size=21 if fig._mobile else 19,
              color=color, transform=ax.transAxes, weight="medium")
    return ax


def clean(ax, grid=False, ticks=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=19 if ax.figure._mobile else 17.5,
                   length=3, width=.7, pad=6)
    if grid:
        ax.grid(color=GRID, linewidth=.7, alpha=.7)
        ax.set_axisbelow(True)
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
    return ax


def arrow(ax, start, end, color=BLUE, curve=0, lw=1.5, scale=15, **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>",
                            connectionstyle=f"arc3,rad={curve}",
                            mutation_scale=scale, linewidth=lw,
                            color=color, shrinkA=0, shrinkB=0, **kwargs)
    ax.add_patch(patch)
    return patch


def point(ax, xy, color=BLUE, size=65, **kwargs):
    return ax.scatter([xy[0]], [xy[1]], s=size, color=color,
                      edgecolors="white", linewidths=1.3, zorder=5, **kwargs)


def plane(ax, xlim, ylim, ticks=None):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    clean(ax)
    ax.spines[["left", "bottom"]].set_position("zero")
    if ticks is not None:
        ax.set_xticks(ticks); ax.set_yticks(ticks)
    return ax


def polar_roots(order, radius=1):
    angle = 2 * np.pi * np.arange(order) / order
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
