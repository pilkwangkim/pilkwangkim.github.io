#!/usr/bin/env python3
"""Reproducible, outlined sorting diagrams at article and mobile widths.

The diagrams use deliberately small, completely enumerated examples.  Their
mathematical inputs and independently checked outputs are recorded alongside
the assets; figure captions belong to the article, not to the bitmap.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-10-24-sorting-from-scratch"
INK, MUTED, BLUE = "#233240", "#64717c", "#315d80"
TEAL, GOLD, CORAL = "#187a75", "#9b701c", "#ad5149"
PALE, GRID, LIGHT = "#f1f5f8", "#d3dce2", "#ffffff"


def canvas(mobile, height):
    width = 480 if mobile else 960
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    fig._sorting_labels = []
    fig._sorting_mobile = mobile
    return fig, ax


def text(ax, x, y, value, size=None, color=INK, weight="normal", ha="center", **kw):
    mobile = ax.figure._sorting_mobile
    size = size or (18 if mobile else 21)
    artist = ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                     ha=ha, va="center", linespacing=1.32, zorder=8, **kw)
    ax.figure._sorting_labels.append(artist)
    return artist


def heading(ax, x, y, value):
    if ax.figure._sorting_mobile and len(value) > 27:
        value = "\n".join(textwrap.wrap(value, width=27))
    return text(ax, x, y, value, size=20 if ax.figure._sorting_mobile else 23,
                weight="bold")


def panel_title(ax, x, y, letter, value, size=None):
    """Left-aligned panel labels share the article's publication-style grid."""
    label = f"({letter})  {value}"
    if ax.figure._sorting_mobile:
        label = "\n".join(textwrap.wrap(label, width=28))
    return text(ax, x, y, label, ha="left", weight="bold",
                size=size or (19 if ax.figure._sorting_mobile else 22))


def panel(ax, x, y, w, h, color=GRID, face=LIGHT):
    patch = Rectangle((x, y), w, h, linewidth=1.1,
                      edgecolor=color, facecolor=face, zorder=0)
    ax.add_patch(patch)
    return patch


def line(ax, x1, y1, x2, y2, color=GRID, lw=1.4, **kw):
    return ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1, **kw)[0]


def arrow(ax, start, end, color=MUTED, curve=0, lw=1.5):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color, shrinkA=1, shrinkB=2,
                        connectionstyle=f"arc3,rad={curve}", zorder=2)
    ax.add_patch(p)
    return p


def array(ax, values, y, x=None, cell=None, colors=None, indices=False,
          face=PALE, gap=5, size=None):
    mobile = ax.figure._sorting_mobile
    cell = cell or (65 if mobile else 76)
    width = len(values)*cell+(len(values)-1)*gap
    x = ((480 if mobile else 960)-width)/2 if x is None else x
    for i, value in enumerate(values):
        left = x+i*(cell+gap)
        col = colors[i] if colors else BLUE
        rect = Rectangle((left, y), cell, 58, facecolor=face,
                         edgecolor=col, linewidth=1.3, zorder=4)
        ax.add_patch(rect)
        text(ax, left+cell/2, y+29, str(value), color=col, weight="bold", size=size)
        if indices:
            text(ax, left+cell/2, y+83, str(i), color=MUTED, size=16 if mobile else 18)
    return [(x+i*(cell+gap)+cell/2, y+29) for i in range(len(values))]


def down(ax, x, y1, y2, color=MUTED):
    arrow(ax, (x, y1), (x, y2), color)


def fig01(mobile):
    """Six concrete mechanisms, not a text taxonomy disguised as an image."""
    fig, ax = canvas(mobile, 1920 if mobile else 1020)
    titles = ["Local update", "Merge", "Partition", "Heap order",
              "Key distribution", "Fixed schedule"]

    def boxes(values, cx, y, colors=None, cell=48, gap=5):
        width = len(values)*cell+(len(values)-1)*gap
        left = cx-width/2
        centers = []
        for k, value in enumerate(values):
            x = left+k*(cell+gap)
            color = (colors or [BLUE]*len(values))[k]
            ax.add_patch(Rectangle((x, y), cell, 43, facecolor=PALE,
                                   edgecolor=color, linewidth=1.2, zorder=4))
            text(ax, x+cell/2, y+21.5, str(value), size=19,
                 color=color, weight="bold")
            centers.append((x+cell/2, y+21.5))
        return centers

    def node(x, y, value, color=BLUE):
        ax.add_patch(Circle((x, y), 22, facecolor=LIGHT,
                            edgecolor=color, linewidth=1.6, zorder=4))
        text(ax, x, y, str(value), size=19, color=color, weight="bold")

    for j, title in enumerate(titles):
        x = 28 if mobile else 32+(j % 2)*480
        y = 18+j*318 if mobile else 18+(j // 2)*334
        width = 424 if mobile else 416
        cx = x+width/2
        line(ax, x, y, x+width, y, GRID, lw=1)
        panel_title(ax, x+4, y+32, chr(97+j), title, size=21)
        if j == 0:
            text(ax, cx, y+73, "Bubble sort: one adjacent swap", size=17, color=MUTED)
            boxes([3, 1, 2], cx, y+107, [CORAL, CORAL, BLUE])
            # The two arrows connect exactly the exchanged record identities.
            boxes([1, 3, 2], cx, y+215, [TEAL, TEAL, BLUE])
            left, right = cx-53, cx
            arrow(ax, (left, y+155), (right, y+210), CORAL)
            arrow(ax, (right, y+155), (left, y+210), TEAL)
        elif j == 1:
            text(ax, cx, y+73, "Two sorted runs → one sorted run", size=17, color=MUTED)
            boxes([1, 4], cx-94, y+107, [BLUE]*2)
            boxes([2, 3], cx+94, y+107, [TEAL]*2)
            line(ax, cx-94, y+157, cx, y+181, BLUE)
            line(ax, cx+94, y+157, cx, y+181, TEAL)
            down(ax, cx, y+181, y+208)
            boxes([1, 2, 3, 4], cx, y+215, [BLUE, TEAL, TEAL, BLUE])
        elif j == 2:
            text(ax, cx, y+73, "Quicksort: pivot = 3", size=17, color=MUTED)
            boxes([4, 2, 5, 1, 3], cx, y+105,
                  [CORAL, BLUE, CORAL, BLUE, TEAL], cell=43)
            down(ax, cx, y+155, y+185)
            # This is a valid grouping, not a trace of a specific partition code.
            boxes([2, 1], cx-113, y+200, [BLUE]*2, cell=40)
            boxes([3], cx, y+200, [TEAL], cell=40)
            boxes([4, 5], cx+113, y+200, [CORAL]*2, cell=40)
            for offset, label, color in [(-113, "< 3", BLUE), (0, "= 3", TEAL), (113, "> 3", CORAL)]:
                text(ax, cx+offset, y+272, label, color=color, size=19)
        elif j == 3:
            text(ax, cx, y+73, "Max-heap: only parent ≥ child", size=17, color=MUTED)
            line(ax, cx, y+130, cx-84, y+213, BLUE)
            line(ax, cx, y+130, cx+84, y+213, BLUE)
            node(cx, y+130, 5, TEAL)
            node(cx-84, y+213, 3)
            node(cx+84, y+213, 4)
            text(ax, cx, y+276, "Siblings need not be sorted", size=17, color=MUTED)
        elif j == 4:
            text(ax, cx, y+73, "Key value selects a destination", size=17, color=MUTED)
            starts = boxes([2, 0, 2, 1], cx, y+106,
                           [GOLD, BLUE, GOLD, TEAL], cell=42)
            targets = [cx-119, cx, cx+119]
            for start, value in zip(starts, [2, 0, 2, 1]):
                arrow(ax, (start[0], y+154), (targets[value], y+207),
                      [BLUE, TEAL, GOLD][value], lw=1.2)
            boxes([0], targets[0], y+214, [BLUE], cell=40)
            boxes([1], targets[1], y+214, [TEAL], cell=40)
            boxes([2, 2], targets[2], y+214, [GOLD]*2, cell=40)
            for value, target in enumerate(targets):
                text(ax, target, y+279, f"Key {value}", size=17, color=MUTED)
        else:
            text(ax, cx, y+73, "4-wire bitonic sorting network", size=17, color=MUTED)
            left, right = cx-151, cx+151
            ys = [y+121+i*42 for i in range(4)]
            for yy in ys:
                line(ax, left, yy, right, yy, MUTED, lw=1.2)
            for yy, incoming, outgoing in zip(ys, [3, 1, 4, 2], [1, 2, 3, 4]):
                text(ax, cx-180, yy, str(incoming), size=19, color=BLUE)
                text(ax, cx+180, yy, str(outgoing), size=19, color=TEAL)
            # Each arrow points toward the smaller output; the first stage
            # has opposite directions, exactly as in the article's network.
            for xx, stage in zip([cx-105, cx, cx+105],
                                 [[(0, 1, True), (2, 3, False)],
                                  [(0, 2, True), (1, 3, True)],
                                  [(0, 1, True), (2, 3, True)]]):
                for k, (a, b, ascending) in enumerate(stage):
                    xxx = xx+(k*16-8 if abs(a-b) == 2 else 0)
                    end, start = (ys[a], ys[b]) if ascending else (ys[b], ys[a])
                    arrow(ax, (xxx, start), (xxx, end), TEAL, lw=1.6)
                    for yy in [ys[a], ys[b]]:
                        ax.add_patch(Circle((xxx, yy), 3, facecolor=TEAL,
                                            edgecolor=TEAL, zorder=5))
            text(ax, cx, y+282, "→ time     arrowhead = min output", size=16, color=MUTED)
    return fig


def fig02(mobile):
    fig, ax = canvas(mobile, 785 if mobile else 610)
    w = 480 if mobile else 960
    heading(ax, w/2, 42, "Adjacent swaps & inversions")
    panel_title(ax, 28 if mobile else 48, 96, "a", "Identify the inverted pairs")
    points = array(ax, [3, 1, 2], 200, colors=[CORAL, BLUE, BLUE])
    # The two curved arcs have endpoints above the exact two offending pairs.
    arrow(ax, (points[0][0], 192), (points[1][0], 192), CORAL, curve=-.60)
    arrow(ax, (points[0][0], 180), (points[2][0], 180), CORAL, curve=-.58)
    text(ax, w/2, 303, "(3, 1) and (3, 2)  →  I = 2", color=CORAL,
         size=19 if mobile else 22)
    if mobile:
        line(ax, 28, 354, w-28, 354)
        panel_title(ax, 28, 395, "b", "One inversion per swap")
        array(ax, [1, 3, 2], 450, colors=[BLUE, CORAL, BLUE])
        text(ax, w/2, 550, "Swap 3 and 1:  I = 1", size=18)
        down(ax, w/2, 580, 620)
        array(ax, [1, 2, 3], 650, colors=[BLUE, BLUE, BLUE])
        text(ax, w/2, 750, "Swap 3 and 2:  I = 0", size=18, color=TEAL)
    else:
        line(ax, 48, 347, 912, 347)
        panel_title(ax, 48, 389, "b", "Remove one inversion per adjacent swap")
        array(ax, [1, 3, 2], 445, x=115, colors=[BLUE, CORAL, BLUE])
        arrow(ax, (405, 474), (550, 474), TEAL)
        array(ax, [1, 2, 3], 445, x=570, colors=[BLUE]*3)
        text(ax, 234, 551, "Swap 3 and 1: I = 1", size=20)
        text(ax, 689, 551, "Swap 3 and 2: I = 0", size=20, color=TEAL)
    return fig


def fig03(mobile):
    fig, ax = canvas(mobile, 1110 if mobile else 850)
    w = 480 if mobile else 960
    heading(ax, w/2, 38, "Merge sort: linear work per level")
    groups = [
        [[8], [3], [6], [1], [7], [2], [5], [4]],
        [[3, 8], [1, 6], [2, 7], [4, 5]],
        [[1, 3, 6, 8], [2, 4, 5, 7]],
        [[1, 2, 3, 4, 5, 6, 7, 8]],
    ]
    ys = [128, 355, 582, 809] if mobile else [115, 305, 495, 685]
    cell = 37 if mobile else 58
    gap = 3 if mobile else 5
    full = 8*cell+7*gap
    x0 = (w-full)/2
    for level, group in enumerate(groups):
        y = ys[level]
        offset = 0
        if level:
            text(ax, w/2 if mobile else 850, y-56 if mobile else y+29,
                 f"Level {level}: scan 8 items" if mobile else f"Level {level}\n8 items",
                 color=TEAL, size=17 if mobile else 19)
        for j, values in enumerate(group):
            color = BLUE
            array(ax, values, y, x=x0+offset*(cell+gap), cell=cell, gap=gap,
                  colors=[color]*len(values), size=17 if mobile else 20)
            offset += len(values)
        if level < 3:
            for j in range(2**(2-level)):
                block = 2**(level+1)
                midpoint = x0+(j*block+block/2)*(cell+gap)-gap/2
                # Each pair of sorted child runs feeds one wider parent run.
                child_width = (block/2)*(cell+gap)
                fork_y, target_y = y+(92 if mobile else 76), ys[level+1]-91
                left_center, right_center = midpoint-child_width/2, midpoint+child_width/2
                line(ax, left_center, y+66, left_center, fork_y)
                line(ax, right_center, y+66, right_center, fork_y)
                line(ax, left_center, fork_y, right_center, fork_y)
                down(ax, midpoint, fork_y, target_y, color=MUTED)
    text(ax, w/2, 940 if mobile else 809,
         "3 merge levels × 8 items" if mobile else r"$\log_2 n$ merge levels × $n$ items = $\Theta(n\log n)$",
         color=BLUE, weight="bold", size=19 if mobile else 23)
    if mobile:
        text(ax, w/2, 1001, r"$\log_2 n$ levels × $n$ items", color=BLUE)
        text(ax, w/2, 1053, r"$\Theta(n\log n)$", color=BLUE, weight="bold", size=23)
    return fig


def flag_regions(ax, y, mobile, title=True):
    w = 480 if mobile else 960
    left, width = (32, 416) if mobile else (90, 780)
    ratios = [0, .22, .44, .78, 1]
    names = ["0", "1", "?", "2"] if mobile else ["all 0", "all 1", "unknown", "all 2"]
    cols = [BLUE, TEAL, GOLD, CORAL]
    for j in range(4):
        x, r = left+width*ratios[j], left+width*ratios[j+1]
        ax.add_patch(Rectangle((x, y), r-x, 65, fc=cols[j], alpha=.12, ec=cols[j], lw=2))
        text(ax, (x+r)/2, y+32.5, names[j], color=cols[j], weight="bold")
    for ratio, label in zip([.22, .44, .78], ["low", "mid", "hi + 1"]):
        x = left+width*ratio
        down(ax, x, y-48, y-7, MUTED)
        text(ax, x, y-71, label, color=MUTED, size=16 if mobile else 19)
    text(ax, left, y+99, "0", color=MUTED, ha="left", size=16 if mobile else 19)
    text(ax, left+width, y+99, "n", color=MUTED, ha="right", size=16 if mobile else 19)
    if title:
        text(ax, w/2, y+155, "Inspect a[mid]; shrink\nthe unknown region." if mobile else "Inspect a[mid]; shrink the unknown region.",
             size=17 if mobile else 21)


def fig04(mobile):
    fig, ax = canvas(mobile, 1030 if mobile else 810)
    w = 480 if mobile else 960
    heading(ax, w/2, 38, "Three-way partition")
    panel_title(ax, 28 if mobile else 48, 97, "a", "Group keys around pivot = 4")
    array(ax, [4, 2, 4, 1, 5, 4, 3], 139, cell=53 if mobile else 76,
          gap=4, size=17 if mobile else 21)
    down(ax, w/2, 222, 265)
    text(ax, w/2, 302, "A valid grouping; not a full sort", color=MUTED,
         size=17 if mobile else 21)
    if mobile:
        array(ax, [2, 1, 3], 348, x=50, cell=58, colors=[BLUE]*3)
        text(ax, 353, 377, "< 4", color=BLUE)
        array(ax, [4, 4, 4], 435, x=50, cell=58, colors=[TEAL]*3)
        text(ax, 353, 464, "= 4", color=TEAL)
        array(ax, [5], 522, x=113, cell=58, colors=[CORAL])
        text(ax, 353, 551, "> 4", color=CORAL)
        line(ax, 35, 637, 445, 637)
        panel_title(ax, 28, 686, "b", "Dutch-flag invariant")
        flag_regions(ax, 840, mobile)
    else:
        array(ax, [2, 1, 3], 351, x=55, cell=70, colors=[BLUE]*3)
        array(ax, [4, 4, 4], 351, x=375, cell=70, colors=[TEAL]*3)
        array(ax, [5], 351, x=800, cell=70, colors=[CORAL])
        for x, label, color in [(165, "< 4", BLUE), (485, "= 4", TEAL), (835, "> 4", CORAL)]:
            text(ax, x, 453, label, color=color, weight="bold")
        line(ax, 45, 508, 915, 508)
        panel_title(ax, 48, 552, "b", "Dutch-flag invariant")
        flag_regions(ax, 677, mobile, title=False)
    return fig


def decision(ax, x, y, label, width=126):
    panel(ax, x-width/2, y-30, width, 60, color=BLUE, face=PALE)
    text(ax, x, y, label, color=BLUE, size=18 if ax.figure._sorting_mobile else 20)


def branch(ax, a, b, label, mobile):
    # Edge labels use a small opaque background to preserve the connecting line.
    arrow(ax, (a[0], a[1]+31), (b[0], b[1]-34), GRID)
    text(ax, (a[0]+b[0])/2, (a[1]+b[1])/2, label,
         size=16 if mobile else 18, color=MUTED,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


def fig05(mobile):
    fig, ax = canvas(mobile, 1140 if mobile else 800)
    w = 480 if mobile else 960
    heading(ax, w/2, 38, "The comparison decision tree")
    if not mobile:
        p = {"root": (480, 127), "left": (250, 283), "right": (710, 283),
             "abc": (100, 443), "l2": (350, 443),
             "bac": (610, 443), "r2": (780, 443),
             "acb": (260, 608), "cab": (440, 608),
             "bca": (685, 608), "cba": (875, 608)}
        for key, label in [("root", "a < b ?"), ("left", "b < c ?"),
                           ("right", "a < c ?"), ("l2", "a < c ?"), ("r2", "b < c ?")]:
            decision(ax, *p[key], label)
        for a, b, yn in [("root", "left", "yes"), ("root", "right", "no"),
                         ("left", "abc", "yes"), ("left", "l2", "no"),
                         ("right", "bac", "yes"), ("right", "r2", "no"),
                         ("l2", "acb", "yes"), ("l2", "cab", "no"),
                         ("r2", "bca", "yes"), ("r2", "cba", "no")]:
            branch(ax, p[a], p[b], yn, mobile)
        for key in ["abc", "bac", "acb", "cab", "bca", "cba"]:
            text(ax, *p[key], " < ".join(key), color=TEAL, size=20, weight="bold")
        text(ax, 480, 702, "3 distinct keys · 6 possible orders · worst-case height 3", size=21)
        text(ax, 480, 758, r"$2^h\geq 3!\quad\Longrightarrow\quad h\geq\lceil\log_2(3!)\rceil=3$", size=23)
    else:
        text(ax, w/2, 98, "First comparison: a < b ?", color=BLUE, weight="bold")
        for j, (condition, first, short, second, leaves) in enumerate([
                ("If yes", "b < c ?", "a < b < c", "a < c ?", ["a < c < b", "c < a < b"]),
                ("If no", "a < c ?", "b < a < c", "b < c ?", ["b < c < a", "c < b < a"]) ]):
            y = 175+j*390
            panel(ax, 20, y, 440, 340)
            text(ax, 70, y+33, condition, color=GOLD, size=17)
            p = {"first": (240, y+75), "short": (112, y+190),
                 "second": (305, y+190), "l": (230, y+298), "r": (380, y+298)}
            decision(ax, *p["first"], first)
            decision(ax, *p["second"], second)
            for a, b, yn in [("first", "short", "yes"), ("first", "second", "no"),
                             ("second", "l", "yes"), ("second", "r", "no")]:
                branch(ax, p[a], p[b], yn, mobile)
            text(ax, *p["short"], short, color=TEAL, size=16, weight="bold")
            for key, value in zip(["l", "r"], leaves):
                text(ax, *p[key], value, color=TEAL, size=16, weight="bold")
        text(ax, w/2, 1004, "6 possible orders", weight="bold")
        text(ax, w/2, 1054, "Worst-case height: 3", color=TEAL)
        text(ax, w/2, 1104, r"$h\geq\lceil\log_2(3!)\rceil=3$", size=21)
    return fig


def fig06(mobile):
    fig, ax = canvas(mobile, 1170 if mobile else 1050)
    w = 480 if mobile else 960
    heading(ax, w/2, 40, "Stable distribution")
    panel_title(ax, 28 if mobile else 48, 100, "a", "Counting sort: reserve key intervals")
    array(ax, ["2a", "1a", "2b", "0a", "1b"], 146,
          cell=66 if mobile else 85, colors=[TEAL, GOLD, TEAL, BLUE, GOLD])
    cols = [89, 211, 299, 387] if mobile else [280, 430, 555, 680]
    left, right = (44, 432) if mobile else (210, 750)
    for yy in [225, 276, 324, 376]:
        line(ax, left, yy, right, yy, lw=1.1)
    line(ax, 154 if mobile else 356, 225, 154 if mobile else 356, 376, lw=1.1)
    for row, y in zip([["Key", "0", "1", "2"], ["Count", "1", "2", "2"], ["Start", "0", "1", "3"]], [252, 300, 348]):
        for x, value in zip(cols, row):
            text(ax, x, y, value, color=TEAL if row[0] == "Start" else INK,
                 size=18 if mobile else 21)
    down(ax, w/2, 389, 435)
    array(ax, ["0a", "1a", "1b", "2a", "2b"], 466,
          cell=66 if mobile else 85, colors=[BLUE, GOLD, GOLD, TEAL, TEAL])
    text(ax, w/2, 570, "Exclusive prefix counts\ngive bucket starts." if mobile else "Exclusive prefix counts give bucket starts.",
         size=17 if mobile else 21)
    line(ax, 30, 625, w-30, 625)
    panel_title(ax, 28 if mobile else 48, 678, "b", "LSD radix: stable digit passes")
    rowys = [750, 895, 1040] if mobile else [725, 835, 945]
    vals = [["21a", "13a", "21b", "12a"],
            ["21a", "21b", "12a", "13a"],
            ["12a", "13a", "21a", "21b"]]
    labels = ["input", "ones digit", "tens digit"]
    for y, values, label in zip(rowys, vals, labels):
        if mobile:
            text(ax, w/2, y-27, label, size=17, color=MUTED)
            array(ax, values, y, cell=85, size=17)
        else:
            text(ax, 145, y+29, label, size=20, color=MUTED)
            array(ax, values, y, x=300, cell=105, size=21)
    for y, next_y in zip(rowys, rowys[1:]):
        down(ax, w/2 if mobile else 517.5, y+69, next_y-53 if mobile else next_y-14)
    return fig


def heap_panel(ax, y, mobile):
    w = 480 if mobile else 960
    panel_title(ax, 28 if mobile else 48, y, "a", "Heap order: parent ≥ children")
    coords = [(w/2, y+82), (w*.28, y+172), (w*.72, y+172),
              (w*.15, y+262), (w*.41, y+262),
              (w*.59, y+262), (w*.85, y+262)]
    for i in range(1, 7):
        a, b = coords[(i-1)//2], coords[i]
        # Tree edges are relations, not a chronological sequence of moves.
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = (dx*dx+dy*dy)**.5
        line(ax, a[0]+dx*31/length, a[1]+dy*31/length,
             b[0]-dx*31/length, b[1]-dy*31/length, MUTED)
    for (x, yy), value in zip(coords, [9, 7, 8, 3, 4, 2, 6]):
        ax.add_patch(Circle((x, yy), 30, edgecolor=BLUE, facecolor=PALE, lw=1.3, zorder=3))
        text(ax, x, yy, str(value), color=BLUE, weight="bold")
    text(ax, w/2, y+307, "Array in level order", color=MUTED,
         size=16 if mobile else 18)
    array(ax, [9, 7, 8, 3, 4, 2, 6], y+330,
          cell=51 if mobile else 66, gap=3 if mobile else 5,
          size=17 if mobile else 20)
    text(ax, w/2, y+418, r"Build: $\Theta(n)$", color=TEAL, size=18 if mobile else 21)
    text(ax, w/2, y+463, r"Worst-case sort: $\Theta(n\log n)$", color=TEAL,
         size=18 if mobile else 21)


def network_panel(ax, y, mobile):
    w = 480 if mobile else 960
    panel_title(ax, 28 if mobile else 48, y, "b", "Bitonic network: fixed stages")
    xmin, xmax = (48, 432) if mobile else (170, 790)
    ys = [y+100+j*74 for j in range(4)]
    stages = [(.16, [(0, 1, True), (2, 3, False)]),
              (.50, [(0, 2, True), (1, 3, True)]),
              (.84, [(0, 1, True), (2, 3, True)])]
    for k, yy in enumerate(ys):
        line(ax, xmin, yy, xmax, yy, MUTED, lw=1.15)
        text(ax, xmin-22, yy, str(k), color=MUTED, size=16 if mobile else 18)
    # Separate x positions inside a stage stop simultaneous comparators from
    # visually merging. They remain one parallel stage (disjoint wires).
    for fraction, comparators in stages:
        base = xmin+fraction*(xmax-xmin)
        ax.add_patch(Rectangle((base-(30 if mobile else 43), y+65),
                              60 if mobile else 86, 290,
                              facecolor=PALE, edgecolor="none", zorder=0))
        for j, (a, b, asc) in enumerate(comparators):
            x = base+(j-.5)*(26 if mobile else 40)
            col = TEAL if asc else CORAL
            arrow(ax, (x, ys[b] if asc else ys[a]),
                  (x, ys[a] if asc else ys[b]), col, lw=1.8)
            for k in [a, b]:
                ax.plot(x, ys[k], "o", ms=5, color=col, zorder=4)
        text(ax, base, y+382, f"S{round((fraction-.16)/.34)+1}", color=MUTED,
             size=16 if mobile else 18)
    text(ax, w/2, y+437, "Arrow points to the smaller output.", size=16 if mobile else 20)
    text(ax, w/2, y+485, "6 comparators · 3 parallel stages", color=BLUE,
         size=17 if mobile else 21)
    text(ax, w/2, y+541, r"Work: $\Theta(n\log^2 n)$", color=TEAL)
    text(ax, w/2, y+590, r"Depth: $\Theta(\log^2 n)$", color=TEAL)


def fig07(mobile):
    fig, ax = canvas(mobile, 1210 if mobile else 1250)
    w = 480 if mobile else 960
    heap_panel(ax, 40, mobile)
    line(ax, 30, 540, w-30, 540)
    network_panel(ax, 595, mobile)
    return fig


def fig08(mobile):
    fig, ax = canvas(mobile, 1490 if mobile else 1450)
    w = 480 if mobile else 960
    ys = [40, 300, 560, 820, 1120] if mobile else [40, 290, 540, 790, 1080]
    for j, y in enumerate(ys):
        if j:
            line(ax, 30, y-26, w-30, y-26)
    panel_title(ax, 28 if mobile else 48, ys[0]+15, "1", "Stability: preserve equal-key order")
    array(ax, ["2a", "1a", "2b"], ys[0]+88, cell=82 if mobile else 90,
          colors=[TEAL, BLUE, TEAL])
    text(ax, w/2, ys[0]+190, "Compare the number; a / b mark identity.", size=16 if mobile else 20)
    panel_title(ax, 28 if mobile else 48, ys[1]+15, "2", "Inversions: minimum adjacent swaps")
    array(ax, [4, 3, 2, 1], ys[1]+88, cell=75 if mobile else 90)
    text(ax, w/2, ys[1]+190,
         "Count all reversed pairs,\nincluding non-neighboring pairs." if mobile else
         "Count all reversed pairs, including non-neighboring pairs.",
         size=16 if mobile else 20)
    panel_title(ax, 28 if mobile else 48, ys[2]+15, "3", "Partition: process equal keys")
    array(ax, [4, 4, 4, 4], ys[2]+88, cell=75 if mobile else 90,
          colors=[TEAL]*4)
    text(ax, w/2, ys[2]+190, "Compare two-way and\nthree-way partition." if mobile else "Compare two-way and three-way partition.", size=16 if mobile else 20)
    panel_title(ax, 28 if mobile else 48, ys[3]+15, "4", "Counting: account for key range")
    array(ax, [0, 2, 7, 9, "10⁶"], ys[3]+88, cell=67 if mobile else 105,
          size=17 if mobile else 21)
    text(ax, w/2, ys[3]+190, "n = 5; keys lie in 0 … 1,000,000", size=16 if mobile else 20)
    panel_title(ax, 28 if mobile else 48, ys[4]+15, "5", "Invariant: update the boundaries")
    # This is the exact quiz state, before its proposed swap.  In particular,
    # low == 0 means the zero region is empty: index 0 is a classified 1.
    cell, gap = (78, 8) if mobile else (100, 10)
    left = (w-(3*cell+2*gap))/2
    top = ys[4]+175
    points = array(ax, [1], top, x=left, cell=cell, gap=gap,
                   colors=[TEAL], face="#f1f8f7")
    points += array(ax, [2, 0], top, x=left+cell+gap, cell=cell, gap=gap,
                    colors=[GOLD, GOLD], face="#faf5e9")
    for i, ((x, _), name) in enumerate(zip(points, ["low", "mid", "high"])):
        text(ax, x, top-66, f"{name}\n= {i}", color=MUTED,
             size=16 if mobile else 20)
        down(ax, x, top-36, top-8)
        text(ax, x, top+83, str(i), color=MUTED,
             size=16 if mobile else 18)
    unknown_left, unknown_right = left+cell+gap, left+3*cell+2*gap
    for x in [unknown_left, unknown_right]:
        line(ax, x, top+105, x, top+118, GOLD)
    line(ax, unknown_left, top+118, unknown_right, top+118, GOLD)
    text(ax, (unknown_left+unknown_right)/2, top+151, "Unknown: indexes 1–2",
         color=GOLD, size=16 if mobile else 20)
    return fig


def cover():
    """A permutation flow: every path joins the same record before and after."""
    fig, ax = canvas(False, 900)
    fig.set_size_inches(16, 9)
    ax.set(xlim=(0, 1600), ylim=(900, 0))
    text(ax, 90, 106, "Sorting algorithms", size=67, color=INK,
         weight="medium", ha="left")
    line(ax, 92, 185, 250, 185, TEAL, lw=4)
    text(ax, 231, 263, "Input", size=51, color=MUTED)
    text(ax, 1369, 263, "Sorted", size=51, color=MUTED)
    values = [5, 1, 6, 2, 4, 3]
    ys = [356+i*88 for i in range(6)]
    # Muted paths carry all records; one accent makes record identity easy to
    # follow without suggesting that the curves are an algorithm execution.
    for value in [1, 2, 3, 5, 6, 4]:
        start = ys[values.index(value)]
        end = ys[value-1]
        vertices = [(303, start), (652, start), (948, end), (1297, end)]
        path = PlotPath(vertices, [PlotPath.MOVETO]+[PlotPath.CURVE4]*3)
        ax.add_patch(PathPatch(path, fill=False, color=TEAL if value == 4 else "#8095a3",
                               linewidth=5.5 if value == 4 else 3, zorder=2 if value == 4 else 1))
    for sequence, x in [(values, 174), (sorted(values), 1312)]:
        for value, yy in zip(sequence, ys):
            color = TEAL if value == 4 else BLUE
            ax.add_patch(Rectangle((x, yy-42), 114, 84,
                                   facecolor="#e5f2f0" if value == 4 else PALE,
                                   edgecolor=color, linewidth=1.3, zorder=4))
            text(ax, x+57, yy, str(value), size=51, color=color,
                 weight="medium")
    return fig


def validate(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    boxes, errors = [], []
    for item in fig._sorting_labels:
        bb = item.get_window_extent(renderer)
        if not (bb.x0 >= 3 and bb.y0 >= 3 and bb.x1 <= bounds.x1-3 and bb.y1 <= bounds.y1-3):
            errors.append(f"out-of-bounds {item.get_text()!r}: {tuple(round(x, 1) for x in bb.extents)}")
        boxes.append((item.get_text(), bb))
    for j, (a, ba) in enumerate(boxes):
        for b, bb in boxes[j+1:]:
            overlap_x = min(ba.x1, bb.x1)-max(ba.x0, bb.x0)
            overlap_y = min(ba.y1, bb.y1)-max(ba.y0, bb.y0)
            if overlap_x > 1 and overlap_y > 1:
                errors.append(f"overlapping text: {a!r} / {b!r}")
    if errors:
        raise AssertionError(stem+": "+"; ".join(errors))
    target = 358 if fig._sorting_mobile else 690
    scale = target/fig.bbox.width
    return {"labels": len(boxes), "bounds": "PASS", "text_collisions": "PASS",
            "target_css_width": target,
            "minimum_label_css_px": round(min(t.get_fontsize()*100/72*scale
                                                for t in fig._sorting_labels), 2)}


def sanity():
    local_input, local_output = [3, 1, 2], [1, 3, 2]
    assert local_output == [local_input[1], local_input[0], local_input[2]]
    merge_input, merge_output = [[1, 4], [2, 3]], [1, 2, 3, 4]
    assert all(run == sorted(run) for run in merge_input)
    assert merge_output == sorted(itertools.chain.from_iterable(merge_input))
    family_partition = [[2, 1], [3], [4, 5]]
    assert sorted(itertools.chain.from_iterable(family_partition)) == sorted([4, 2, 5, 1, 3])
    assert all(x < 3 for x in family_partition[0]) and family_partition[1] == [3]
    assert all(x > 3 for x in family_partition[2])
    assert family_partition[0] != sorted(family_partition[0])
    assert 5 >= 3 and 5 >= 4
    distribution = [2, 0, 2, 1]
    destinations = [[x for x in distribution if x == k] for k in range(3)]
    assert destinations == [[0], [1], [2, 2]]
    inv = lambda a: sum(a[i] > a[j] for i in range(len(a)) for j in range(i+1, len(a)))
    assert [inv(a) for a in [[3, 1, 2], [1, 3, 2], [1, 2, 3]]] == [2, 1, 0]
    counting = [(2, "a"), (1, "a"), (2, "b"), (0, "a"), (1, "b")]
    counts = [sum(key == k for key, identity in counting) for k in range(3)]
    starts = [sum(counts[:k]) for k in range(3)]
    assert counts == [1, 2, 2] and starts == [0, 1, 3]
    assert sorted(counting, key=lambda p: p[0]) == [(0, "a"), (1, "a"), (1, "b"), (2, "a"), (2, "b")]
    radix = [(21, "a"), (13, "a"), (21, "b"), (12, "a")]
    ones = sorted(radix, key=lambda p: p[0] % 10)
    tens = sorted(ones, key=lambda p: p[0] // 10)
    assert ones == [(21, "a"), (21, "b"), (12, "a"), (13, "a")]
    assert tens == [(12, "a"), (13, "a"), (21, "a"), (21, "b")]
    stages = [[(0, 1, True), (2, 3, False)], [(0, 2, True), (1, 3, True)], [(0, 1, True), (2, 3, True)]]
    network_inputs = list(itertools.permutations(range(4))) + list(itertools.product([0, 1], repeat=4))
    for permutation in network_inputs:
        a = list(permutation)
        for stage in stages:
            used = []
            for i, j, ascending in stage:
                used.extend([i, j])
                if (a[i] > a[j]) == ascending:
                    a[i], a[j] = a[j], a[i]
            assert len(set(used)) == 4
        assert a == sorted(permutation)
    paths = []
    for perm in itertools.permutations("abc"):
        rank = {k: perm.index(k) for k in perm}
        compare = lambda a, b: rank[a] < rank[b]
        if compare("a", "b"):
            path = [True]
            if compare("b", "c"):
                path += [True]
                found = "abc"
            else:
                path += [False, compare("a", "c")]
                found = "acb" if path[-1] else "cab"
        else:
            path = [False]
            if compare("a", "c"):
                path += [True]
                found = "bac"
            else:
                path += [False, compare("b", "c")]
                found = "bca" if path[-1] else "cba"
        assert found == "".join(perm)
        paths.append({"order": found, "outcomes": path})
    assert len({p["order"] for p in paths}) == 6 and max(len(p["outcomes"]) for p in paths) == 3
    heap = [9, 7, 8, 3, 4, 2, 6]
    assert all(heap[(child-1)//2] >= heap[child] for child in range(1, len(heap)))
    partition_input = [4, 2, 4, 1, 5, 4, 3]
    partition_groups = [[2, 1, 3], [4, 4, 4], [5]]
    assert sorted(itertools.chain.from_iterable(partition_groups)) == sorted(partition_input)
    assert all(x < 4 for x in partition_groups[0])
    assert all(x == 4 for x in partition_groups[1])
    assert all(x > 4 for x in partition_groups[2])
    return {"families": {"local_swap": {"input": local_input, "output": local_output},
                         "merge": {"input_runs": merge_input, "output": merge_output},
                         "partition": {"input": [4, 2, 5, 1, 3], "pivot": 3,
                                       "groups": family_partition},
                         "heap": [5, 3, 4],
                         "distribution": {"input": distribution, "destinations": destinations},
                         "network": {"input": [3, 1, 4, 2], "output": [1, 2, 3, 4],
                                     "stages": stages, "arrowhead": "minimum output"}},
            "inversions": [2, 1, 0], "merge": {"n": 8, "merge_levels": 3, "items_per_level": 8},
            "partition": {"input": [4, 2, 4, 1, 5, 4, 3], "pivot": 4,
                          "valid_partition": [[2, 1, 3], [4, 4, 4], [5]], "kind": "valid grouping, not an implementation trace"},
            "counting": {"input": counting, "counts": counts, "exclusive_offsets": starts,
                         "stable_output": sorted(counting, key=lambda p: p[0])},
            "radix": {"input": radix, "ones": ones, "tens": tens},
            "decision_tree": paths,
            "bitonic": {"n": 4, "comparators": 6, "depth": 3, "stages": stages,
                        "permutations_checked": 24, "binary_inputs_checked": 16},
            "heap": {"array": [9, 7, 8, 3, 4, 2, 6], "build": "Theta(n)", "worst_case_sort": "Theta(n log n)"},
            "exercises": {"q1": ["2a", "1a", "2b"], "q2": [4, 3, 2, 1], "q3": [4, 4, 4, 4],
                          "q4": {"values": [0, 2, 7, 9, 1000000], "range_inclusive": [0, 1000000]},
                          "q5": {"input": [1, 2, 0], "low": 0, "mid": 1, "high": 2,
                                 "a_mid": 2, "classified_one_indices": [0], "unknown_indices": [1, 2],
                                 "state": "before the swap; no solution or pointer update shown"}}}


def contact_sheet(mobile):
    width = 358 if mobile else 690
    cols = 4 if mobile else 2
    thumbs = []
    suffix = "-mobile" if mobile else ""
    for key in NAMES:
        with Image.open(OUT / f"{key}{suffix}.png") as im:
            thumbs.append(im.convert("RGB").resize((width, round(im.height*width/im.width)), Image.Resampling.LANCZOS))
    heights = [max(t.height for t in thumbs[j:j+cols]) for j in range(0, len(thumbs), cols)]
    result = Image.new("RGB", (cols*width+20*(cols-1), sum(heights)+20*(len(heights)-1)), "#dce5eb")
    y = 0
    for j, h in enumerate(heights):
        for k, thumb in enumerate(thumbs[j*cols:(j+1)*cols]):
            result.paste(thumb, (k*(width+20), y))
        y += h+20
    result.save(OUT / ("contact-sheet-mobile.png" if mobile else "contact-sheet-desktop.png"))


NAMES = ["fig-01-families", "fig-02-inversions", "fig-03-merge-levels", "fig-04-partition",
         "fig-05-lower-bound", "fig-06-counting-radix", "fig-07-heap-network", "fig-08-exercises"]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "sorting-from-scratch-v2", "mathtext.fontset": "stix",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    for mobile in (False, True):
        for key, maker in zip(NAMES, [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08]):
            fig = maker(mobile)
            stem = key+("-mobile" if mobile else "")
            check = validate(fig, stem)
            for extension in ["svg", "png"]:
                path = OUT / (stem+"."+extension)
                kwargs = {"metadata": {"Date": None}} if extension == "svg" else {"metadata": {"Software": "Matplotlib"}}
                fig.savefig(path, dpi=160, **kwargs)
            assets.append({"stem": stem, "width": round(fig.get_figwidth()*160),
                           "height": round(fig.get_figheight()*160), "layout": check})
            plt.close(fig)
    fig = cover()
    check = validate(fig, "cover-sorting")
    fig.savefig(OUT / "cover-sorting.svg", metadata={"Date": None})
    fig.savefig(OUT / "cover-sorting.png", dpi=100, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    assets.append({"stem": "cover-sorting", "width": 1600, "height": 900, "layout": check})
    for mobile in [False, True]:
        contact_sheet(mobile)
    record = {"generator": "tools/sorting_figures.py", "assets": assets, "mathematics": sanity(),
              "design": {"revision": 3, "background": "white", "strokes": "thin",
                         "palette": {"structure": BLUE, "stable_or_equal": TEAL,
                                     "unknown_or_digit_key": GOLD, "inverted_or_greater": CORAL},
                         "families": ["local updates", "merge", "partition", "heap order",
                                      "key distribution", "fixed schedule"],
                         "dimensions": "Figure 1: six mechanism panels; other figure and cover dimensions unchanged",
                         "cover": {"input": [5, 1, 6, 2, 4, 3], "output": [1, 2, 3, 4, 5, 6],
                                   "highlight_key": 4, "paths": "record identity, not execution steps"}},
              "svg_text": "Outlined paths; no installed-font dependency",
              "inspection": "Contact sheets resize each figure to 690px desktop / 358px mobile; browser QA is separate."}
    (OUT / "figure-data.json").write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    print(json.dumps({"assets": assets, "files": len(hashes), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
