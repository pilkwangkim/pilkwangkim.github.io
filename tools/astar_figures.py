#!/usr/bin/env python3
"""Deterministic, article-readable A* figures and their exact input/results.

Run with a Python environment providing NumPy and Matplotlib. Grid movement is
four-way, entering a cell costs its terrain value, and the start costs zero.
Neighbor order is east, south, west, north; heaps break ties by h then insertion.
An expansion examines neighbors; popping the goal is not an expansion.
"""
from __future__ import annotations

from collections import deque
import heapq
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as PlotPath
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-10-24-astar-from-scratch"
BLUE, GREEN, GOLD, RED = "#235789", "#18794e", "#be7a10", "#c23a38"
INK, MUTED = "#1c3045", "#576879"
MAP = [
    "#############",
    "#...........#",
    "#..###.###..#",
    "#..#.....#..#",
    "#S.5555555.G#",
    "#..#..5..#..#",
    "#..###.###..#",
    "#...........#",
    "#############",
]
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
START, GOAL = (1, 4), (11, 4)


def cost_at(p):
    c = MAP[p[1]][p[0]]
    return int(c) if c.isdigit() else 1


def neighbors(p):
    for dx, dy in DIRS:
        q = (p[0] + dx, p[1] + dy)
        if 0 <= q[1] < len(MAP) and 0 <= q[0] < len(MAP[0]) and MAP[q[1]][q[0]] != "#":
            yield q


def heuristic(p):
    return abs(p[0] - GOAL[0]) + abs(p[1] - GOAL[1])


def search(method="astar", weight=1):
    """Reference implementation independent of the article's JS engine."""
    seq = itertools.count()
    dist, parent, expanded, order = {START: 0}, {}, [], []
    pending = deque([START]) if method in ("bfs", "dfs") else [(weight * heuristic(START), heuristic(START), next(seq), START, 0)]
    while pending:
        if method in ("bfs", "dfs"):
            p = pending.popleft() if method == "bfs" else pending.pop()
            g = dist[p]
        else:
            _, _, _, p, g = heapq.heappop(pending)
            if g != dist.get(p):
                continue
        order.append(p)
        if p == GOAL:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            return {"method": method, "weight": weight, "cost": g,
                    "steps": len(path) - 1, "expanded": len(expanded),
                    "path": list(reversed(path)), "expanded_nodes": expanded,
                    "pop_order": order}
        expanded.append(p)
        adjacent = list(neighbors(p))
        # Reverse pushes so DFS visits the first stated neighbor first.
        if method == "dfs":
            adjacent.reverse()
        for q in adjacent:
            candidate = g + cost_at(q)
            if method in ("bfs", "dfs"):
                if q in dist:
                    continue
                dist[q], parent[q] = candidate, p
                pending.append(q)
            elif candidate < dist.get(q, float("inf")):
                dist[q], parent[q] = candidate, p
                priority = (candidate if method == "dijkstra" else
                            heuristic(q) if method == "greedy" else
                            candidate + weight * heuristic(q))
                heapq.heappush(pending, (priority, heuristic(q), next(seq), q, candidate))
    raise RuntimeError("Map must have a path")


def style(mobile):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15 if mobile else 18,
        "axes.titlesize": 16 if mobile else 20,
        "axes.labelsize": 16 if mobile else 19,
        "xtick.labelsize": 14 if mobile else 16,
        "ytick.labelsize": 14 if mobile else 16,
        "legend.fontsize": 14 if mobile else 16,
        "mathtext.fontset": "stix", "svg.fonttype": "path",
        "svg.hashsalt": "astar-from-scratch-v1", "savefig.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK,
        "axes.titlepad": 15, "axes.labelpad": 10,
        "figure.constrained_layout.h_pad": .14,
        "figure.constrained_layout.w_pad": .10,
    })


def panels(mobile, n=2, height=4.5):
    cols = 1 if mobile else 2
    rows = (n + cols - 1) // cols
    f, axes = plt.subplots(rows, cols, figsize=(5 if mobile else 10.8, rows*height),
                           layout="constrained", squeeze=False)
    return f, list(axes.flat)


def bare(ax, xlim=(-.5, 4.5), ylim=(-.5, 2.0)):
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")


def node(ax, xy, name, face="white", color=BLUE, radius=.22):
    ax.add_patch(Circle(xy, radius, facecolor=face, edgecolor=color, lw=2.1, zorder=4))
    ax.text(*xy, name, ha="center", va="center", color=color, weight="bold", zorder=5)


def edge(ax, start, end, label="", color=MUTED, width=1.7, style="-", rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                shrinkA=18, shrinkB=18, linewidth=width, color=color,
                                linestyle=style, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        x, y = (start[0]+end[0])/2, (start[1]+end[1])/2
        ax.text(x, y+.13, label, ha="center", va="center", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.8}, zorder=3)


SMALL_POS = {"S": (0, .6), "A": (1.35, 1.45), "B": (1.15, -.05),
             "C": (2.6, -.05), "G": (3.9, .6)}
SMALL_EDGES = [("S", "A", 2), ("A", "G", 9), ("S", "B", 3),
               ("B", "C", 1), ("C", "G", 1)]


def small_graph(ax, highlight=()):
    bare(ax, (-.45, 4.35), (-.5, 2.0))
    ax.set_aspect("equal", adjustable="box")
    for a, b, c in SMALL_EDGES:
        active = (a, b) in highlight
        edge(ax, SMALL_POS[a], SMALL_POS[b], str(c), GREEN if active else MUTED,
             3.4 if active else 1.5)
    for name, xy in SMALL_POS.items():
        node(ax, xy, name, face="#e6f3ed" if name in ("S", "G") else "white")


def fig01(mobile):
    f, axes = panels(mobile, height=3.5 if mobile else 4.0)
    small_graph(axes[0], (("S", "B"), ("B", "C"), ("C", "G")))
    axes[0].set_title("(a) A directed, weighted graph")
    a = axes[1]
    bare(a, (0, 1), (0, 1))
    a.set_title("(b) A path is a sequence of states")
    a.text(.5, .78, r"$P=(S,B,C,G)$", ha="center", fontsize=25 if not mobile else 22)
    a.text(.5, .48, r"$C(P)=3+1+1=5$", ha="center", fontsize=25 if not mobile else 22, color=GREEN)
    a.text(.5, .14, "Nodes = states\nArrows = allowed moves\nEdge labels = move costs", ha="center", va="center", linespacing=1.6)
    return f


def tree(ax, method):
    pos = {"S": (2, 2.8), "A": (.7, 1.5), "B": (3.3, 1.5),
           "C": (0, .2), "D": (1.4, .2), "E": (2.6, .2), "G": (4, .2)}
    edges = [("S", "A"), ("S", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("B", "G")]
    order = ["S", "A", "C", "D", "B", "E", "G"] if method == "DFS" else ["S", "A", "B", "C", "D", "E", "G"]
    bare(ax, (-.5, 4.5), (-1.45, 3.5))
    ax.set_aspect("equal", adjustable="box")
    for a, b in edges:
        edge(ax, pos[a], pos[b])
    for name, xy in pos.items():
        node(ax, xy, name)
        # An upper-right label lies directly on the incoming arrow for A, C
        # and E. Keep every pop-order badge level with its node instead.
        # G's badge sits to the left so it stays inside the existing canvas.
        badge = ax.text(xy[0]+(-.43 if name == "G" else .43), xy[1],
                        str(order.index(name)+1), color=RED, ha="center", va="center",
                        weight="bold", zorder=6,
                        bbox={"boxstyle": "round,pad=.12", "facecolor": "#fff1f0",
                              "edgecolor": "none"})
        badge.set_gid("pop-order-badge")
    ax.text(2, -.5, "Red badges: pop order", ha="center", color=MUTED)
    ax.text(2, -.99, "Stack: last in, first out" if method == "DFS" else "Queue: first in, first out", ha="center", color=BLUE)
    ax.set_title(f"{method}: " + ("finish one branch first" if method == "DFS" else "visit each depth in turn"))


def fig02(mobile):
    f, axes = panels(mobile, height=4.0 if mobile else 4.7)
    tree(axes[0], "DFS")
    tree(axes[1], "BFS")
    return f


def fig03(mobile):
    f, axes = panels(mobile, height=3.5 if mobile else 4.1)
    small_graph(axes[0], (("S", "A"), ("A", "G")))
    small_graph(axes[1], (("S", "B"), ("B", "C"), ("C", "G")))
    axes[0].set_title("(a) BFS: 2 edges, cost 11")
    axes[1].set_title("(b) Cheapest: 3 edges, cost 5")
    for a, text in zip(axes, (r"$2+9=11$", r"$3+1+1=5$")):
        a.text(1.95, -.68, text, color=GREEN, ha="center", fontsize=24 if not mobile else 21)
        a.set_ylim(-1.05, 2.0)
    return f


def fig04(mobile):
    f, axes = panels(mobile, n=4, height=3.0 if mobile else 3.7)
    rows = [("Dijkstra", r"$p=g$", [2, 3, 4], "A"),
            ("Greedy best-first", r"$p=h$", [7, 2, 1], "C"),
            ("A*", r"$p=g+h$", [9, 5, 5], "C"),
            ("Same frontier", r"$g: 2,3,4$", [7, 2, 1], "")]
    for ax, (name, formula, vals, winner) in zip(axes, rows):
        bare(ax, (-.6, 2.6), (-.6, 1.35))
        ax.set_title(name + ("  " + formula if winner else ""))
        for j, (n, v) in enumerate(zip("ABC", vals)):
            node(ax, (j, .55), n, face="#e6f3ed" if n == winner else "white", radius=.24)
            ax.text(j, -.03, str(v), ha="center", color=GREEN if n == winner else INK,
                    fontsize=25 if not mobile else 22)
        ax.text(1, -.48, f"Next: {winner}" + (" (tie: smaller h)" if name == "A*" else "") if winner else
                r"$g=(2,3,4),\quad h=(7,2,1)$", ha="center", fontsize=14 if mobile else 18)
    return f


def fig05(mobile):
    # Explicit rows keep each grid, its formula and its explanatory caption
    # together. In an equal-aspect axes, off-axes text otherwise escapes the
    # allotted panel and crowds the next mobile panel's title.
    cols = 1 if mobile else 2
    f = plt.figure(figsize=(5 if mobile else 10.8, 10.8 if mobile else 5.4),
                   layout="constrained")
    gs = f.add_gridspec(6 if mobile else 3, cols,
                       height_ratios=[4, .6, .5] * (2 if mobile else 1),
                       hspace=.09 if mobile else .02)
    axes = [f.add_subplot(gs[j*3 if mobile else 0, 0 if mobile else j])
            for j in range(2)]
    for j, (ax, diagonal) in enumerate(zip(axes, (False, True))):
        ax.set(xlim=(-.5, 4.5), ylim=(-.5, 3.5), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, 5), minor=True)
        ax.set_yticks(np.arange(-.5, 4), minor=True)
        ax.grid(which="minor", color="#b8c6d0", lw=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3)] if diagonal else [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
        ax.plot(*zip(*path), color=GREEN, lw=3, marker="o", ms=5)
        for p, label in (((0, 0), "n"), ((4, 3), "G")):
            ax.text(*p, label, color="white", ha="center", va="center", weight="bold",
                    bbox={"boxstyle": "circle,pad=.2", "fc": BLUE, "ec": "none"})
        ax.set_title("8-way: diagonal cost " + r"$\sqrt{2}$" if diagonal else "4-way: each step costs 1")
        row, col = (3*j, 0) if mobile else (0, j)
        formula = f.add_subplot(gs[row+1, col]); bare(formula, (0, 1), (0, 1))
        formula.text(.5, .5, r"$h_{\rm oct}=3\sqrt{2}+1$" if diagonal else r"$h_{\rm Man}=4+3=7$",
                     ha="center", va="center", fontsize=24 if not mobile else 20)
        caption = f.add_subplot(gs[row+2, col]); bare(caption, (0, 1), (0, 1))
        caption.text(.5, .5, "Remove obstacles; keep move costs.",
                     ha="center", va="center", fontsize=16 if not mobile else 14, color=MUTED)
    return f


def fig06(mobile):
    f, axes = panels(mobile, height=4.8 if mobile else 5.9)
    ax = axes[0]
    pos = {"S": (0, 1), "A": (2, 1.8), "B": (2, .2), "G": (4, 1)}
    bare(ax, (-.5, 4.5), (-.5, 2.6))
    ax.set_aspect("equal", adjustable="box")
    for a, b, c in [("S", "A", 3), ("S", "B", 1), ("B", "A", 1), ("A", "G", 3), ("B", "G", 100)]:
        edge(ax, pos[a], pos[b], str(c), GREEN if (a, b) in [("S", "B"), ("B", "A"), ("A", "G")] else MUTED)
    h = {"S": 5, "A": 0, "B": 4, "G": 0}
    for n, p in pos.items():
        node(ax, p, n)
        # Put A's heuristic above its node, away from the vertical B -> A edge.
        hy = p[1] + .55 if n == "A" else p[1] - .43
        ax.text(p[0], hy, f"h={h[n]}", ha="center", fontsize=14 if mobile else 18)
    ax.set_title("(a) Admissible, but inconsistent")
    ax.text(2, -.75, r"$h(B)=4>1+h(A)=1$", ha="center", color=RED, fontsize=20 if not mobile else 17)
    ax.set_ylim(-1, 2.6)
    ax = axes[1]
    bare(ax, (0, 1), (0, 1))
    ax.set_title("(b) A better route must reopen A")
    lines = [("Pop S", "A: g=3, f=3\nB: g=1, f=5"),
             ("Pop A", "G: g=6, f=6"),
             ("Pop B", "A improves: g=2, f=2"),
             ("Reopen A", "G improves: g=5, f=5")]
    for j, (label, val) in enumerate(lines):
        y = .91-j*.235
        ax.text(.03, y, label, va="top", color=GREEN if j == 3 else BLUE, weight="bold")
        ax.text(.98, y, val, va="top", ha="right", fontsize=14 if mobile else 18)
    return f


def draw_grid(ax, result=None, title=None, small=False, terrain_numbers=True):
    data = np.array([[0 if c == "#" else 2 if c == "5" else 1 for c in row] for row in MAP])
    ax.imshow(data, cmap=ListedColormap(["#273b4c", "#f3f6f8", "#f6d9a1"]), vmin=0, vmax=2, interpolation="nearest")
    if result:
        for x, y in set(map(tuple, result["expanded_nodes"])):
            ax.add_patch(Rectangle((x-.46, y-.46), .92, .92, fc="#6e9fc1", alpha=.42, ec="none"))
        ax.plot(*zip(*result["path"]), color=GREEN, lw=3.3, zorder=5)
    for y, row in enumerate(MAP):
        for x, c in enumerate(row):
            if c == "5" and terrain_numbers:
                # The horizontal BFS/greedy route crosses the cell center.
                # Keep the entry-cost label above it, not struck through by it.
                label = ax.text(x, y-.27, c, ha="center", va="center", color="#7e4708",
                                fontsize=plt.rcParams["font.size"]*.83, zorder=3)
                label.set_gid("terrain-cost-label")
    for p, name in ((START, "S"), (GOAL, "G")):
        ax.text(*p, name, ha="center", va="center", color="white", weight="bold", zorder=6,
                fontsize=plt.rcParams["font.size"]*.9 if small else 17,
                bbox={"boxstyle": "circle,pad=.15", "fc": BLUE, "ec": "white", "lw": 1})
    ax.set(xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def fig07(mobile):
    f, axes = panels(mobile, n=4, height=4.7)
    for ax, method in zip(axes, ("bfs", "dijkstra", "greedy", "astar")):
        r = search(method)
        label = {"bfs": "BFS", "dijkstra": "Dijkstra", "greedy": "Greedy", "astar": "A*"}[method]
        draw_grid(ax, r, f"{label}: cost {r['cost']}, {r['expanded']} expansions", small=True)
        ax.text(.5, -.07, f"{r['steps']} moves · green: path · blue: expanded", transform=ax.transAxes,
                ha="center", fontsize=13 if mobile else 14, color=MUTED)
    return f


def fig08(mobile):
    f, axes = panels(mobile, height=4.8)
    weights = [1, 1.25, 1.5, 2, 3, 5]
    results = [search("astar", w) for w in weights]
    for ax, metric, label, color in zip(axes, ("cost", "expanded"), ("Returned path cost", "Expansions"), (GREEN, BLUE)):
        positions = np.arange(len(weights))
        ax.bar(positions, [r[metric] for r in results], color=color, width=.68)
        ax.set_xticks(positions, [f"{w:g}" for w in weights])
        for x, r in zip(positions, results):
            ax.annotate(str(r[metric]), (x, r[metric]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=12 if mobile else 17)
        ax.set(xlabel=r"Weight $w$ in $g+wh$", ylabel=label)
        ax.margins(y=.25, x=.06)
        ax.grid(alpha=.2)
    axes[0].axhline(results[0]["cost"], ls="--", lw=1, color=MUTED)
    axes[0].set_title("(a) Solution quality")
    axes[1].set_title("(b) Search work")
    return f


def cover():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "mathtext.fontset": "stix", "svg.fonttype": "path"})
    f = plt.figure(figsize=(14, 8), facecolor="#f8fafc")
    gs = f.add_gridspec(1, 2, width_ratios=[1, 1.45], left=.04, right=.98, bottom=.10, top=.90, wspace=.04)
    ax = f.add_subplot(gs[0]); bare(ax, (0, 1), (0, 1))
    ax.text(.04, .77, "A*", fontsize=110, weight="bold", color=BLUE)
    ax.text(.04, .53, r"$f=g+h$", fontsize=53, color=GREEN)
    ax.text(.04, .29, "Cost so far\n+ estimated rest", fontsize=32, linespacing=1.5)
    ax = f.add_subplot(gs[1])
    draw_grid(ax, search("astar"), terrain_numbers=False)
    return f


def verify_layout(f, stem):
    """Reject clipped labels, text collisions and labels on route lines.

    The check uses renderer extents after constrained layout, not estimated
    character widths. Cost labels intentionally cover their own edge with a
    white background. Traversal-order badges avoid graph arrows, and terrain
    labels avoid the green route lines.
    Browser checks are still needed for the article's final rendered size.
    """
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    texts = list(f.texts)
    for ax in f.axes:
        texts.extend(ax.texts)
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        if ax.axison:
            for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lower, upper = sorted(limits)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    # Locators also create ticks outside the view interval;
                    # Matplotlib does not draw those labels.
                    if lower <= tick.get_loc() <= upper:
                        texts.extend([tick.label1, tick.label2])
    entries = [(t, t.get_window_extent(renderer)) for t in texts
               if t.get_visible() and t.get_text().strip()]
    failures = []
    canvas = f.bbox
    for t, box in entries:
        if (box.x0 < canvas.x0-.5 or box.x1 > canvas.x1+.5 or
                box.y0 < canvas.y0-.5 or box.y1 > canvas.y1+.5):
            failures.append(f"clipped text {t.get_text()!r}")
    for j, (a, box_a) in enumerate(entries):
        for b, box_b in entries[j+1:]:
            overlap_x = min(box_a.x1, box_b.x1)-max(box_a.x0, box_b.x0)
            overlap_y = min(box_a.y1, box_b.y1)-max(box_a.y0, box_b.y0)
            if overlap_x > 1 and overlap_y > 1:
                failures.append(f"text collision {a.get_text()!r} / {b.get_text()!r}")
    for ax in f.axes:
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        for t in ax.texts:
            if t.get_gid() == "pop-order-badge":
                box = t.get_window_extent(renderer).expanded(1.15, 1.15)
                # to_polygons removes the unused (0, 0) CLOSEPOLY vertex;
                # testing that dummy vertex creates spurious intersections.
                if any(PlotPath(poly).intersects_bbox(box, filled=False)
                       for p in arrows
                       for poly in p.get_path().transformed(p.get_transform()).to_polygons(closed_only=False)):
                    failures.append(f"pop badge {t.get_text()!r} intersects an arrow")
            if t.get_gid() == "terrain-cost-label":
                box = t.get_window_extent(renderer).expanded(1.05, 1.05)
                if any(line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False)
                       for line in ax.lines):
                    failures.append(f"terrain cost {t.get_text()!r} intersects a route")
    if failures:
        raise AssertionError(stem + ": " + "; ".join(failures))
    return {"stem": stem, "checked_texts": len(entries),
            "canvas_bounds": "PASS", "text_collisions": "PASS",
            "pop_badges_avoid_arrows": "PASS", "terrain_labels_avoid_routes": "PASS"}


def save(f, stem):
    verify_layout(f, stem)
    svg_path = OUT / f"{stem}.svg"
    f.savefig(svg_path, metadata={"Date": None})
    # Matplotlib emits whitespace-only SVG lines; normalize generated text
    # without changing path coordinates, font outlines, or scientific values.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    f.savefig(OUT / f"{stem}.png", dpi=140, metadata={"Software": "Matplotlib"})
    plt.close(f)
    with Image.open(OUT / f"{stem}.png") as im:
        return {"stem": stem, "width": im.width, "height": im.height}


def verify(results):
    """Check path identity/cost and A* against independent Bellman relaxation."""
    nodes = [(x, y) for y, row in enumerate(MAP) for x, c in enumerate(row) if c != "#"]
    d = {p: float("inf") for p in nodes}
    d[START] = 0
    passes = 0
    for _ in range(len(nodes)-1):
        passes += 1
        changed = False
        for p in nodes:
            for q in neighbors(p):
                if d[p] + cost_at(q) < d[q]:
                    d[q] = d[p] + cost_at(q)
                    changed = True
        if not changed:
            break
    for r in results:
        path = list(map(tuple, r["path"]))
        assert path[0] == START and path[-1] == GOAL
        assert all(b in list(neighbors(a)) for a, b in zip(path, path[1:]))
        assert sum(cost_at(p) for p in path[1:]) == r["cost"]
    assert search("astar")["cost"] == d[GOAL] == search("dijkstra")["cost"] == 16
    assert all(heuristic(p) <= cost_at(q)+heuristic(q) for p in nodes for q in neighbors(p))
    return {"legal_path_and_cost_checks": "PASS", "bellman_optimal_cost": d[GOAL],
            "bellman_passes": passes, "heuristic_consistency_every_edge": "PASS"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    makers = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08]
    names = ["search-problem", "dfs-bfs", "weighted-detour", "priority-functions",
             "grid-heuristics", "reopening", "search-comparison", "weighted-astar"]
    assets = []
    for mobile in (False, True):
        style(mobile)
        for j, (make, name) in enumerate(zip(makers, names), 1):
            assets.append(save(make(mobile), f"fig-{j:02}-{name}" + ("-mobile" if mobile else "")))
    assets.append(save(cover(), "cover-astar"))
    results = {m: search(m) for m in ("bfs", "dfs", "dijkstra", "greedy", "astar")}
    record = {"grid": MAP, "start": START, "goal": GOAL,
              "coordinates": "(x,y), x rightwards, y downwards, zero-based",
              "movement": "east, south, west, north; no diagonals",
              "cost": "entering target cell; .=1, S=1, G=1, 5=5; start not charged",
              "priority_ties": "(priority, h, FIFO insertion sequence)",
              "expansion_definition": "non-stale non-goal pop whose neighbors are examined",
              "results": results,
              "weighted_astar": [search("astar", w) for w in (1, 1.25, 1.5, 2, 3, 5)]}
    record["verification"] = verify(list(results.values()) + record["weighted_astar"])
    record["assets"] = assets
    (OUT / "figure-data.json").write_text(json.dumps(record, indent=2) + "\n")
    for m, r in results.items():
        print(m, {k: r[k] for k in ("cost", "steps", "expanded")})
    print("weighted", [(r["weight"], r["cost"], r["expanded"]) for r in record["weighted_astar"]])


if __name__ == "__main__":
    main()
