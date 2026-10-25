"""Editorial cover and quantitative plates for the Galois article."""
from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np

from galois_visual_style import (
    BLUE, GOLD, GRID, INK, MUTED, RED, TEAL, arrow, canvas, clean, label, panel,
)


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def a5_graph():
    """A Cayley graph of A5, with an exact truncated-icosahedron embedding."""
    a, b, identity = (1, 2, 3, 4, 0), (0, 2, 1, 4, 3), tuple(range(5))
    elements, queue = {identity}, [identity]
    while queue:
        p = queue.pop()
        for g in (a, b):
            q = compose(p, g)
            if q not in elements:
                elements.add(q); queue.append(q)
    graph = nx.Graph()
    for p in sorted(elements):
        for g, kind in ((a, "a"), (b, "b")):
            graph.add_edge(p, compose(p, g), kind=kind)
    assert len(graph) == 60 and graph.number_of_edges() == 90
    assert set(dict(graph.degree()).values()) == {3}

    phi = (1 + math.sqrt(5)) / 2
    vertices = np.array([(0, s, t * phi) for s in (-1, 1) for t in (-1, 1)]
                        + [(s, t * phi, 0) for s in (-1, 1) for t in (-1, 1)]
                        + [(t * phi, 0, s) for s in (-1, 1) for t in (-1, 1)])
    edges = {(i, j) for i, j in itertools.combinations(range(12), 2)
             if np.isclose(np.linalg.norm(vertices[i] - vertices[j]), 2)}
    adjacency = {i: {j for u, v in edges for j in ([v] if u == i else [u] if v == i else [])}
                 for i in range(12)}
    coordinates = {(i, j): (2 * vertices[i] + vertices[j]) / 3
                   for i in range(12) for j in sorted(adjacency[i])}
    polyhedron = nx.Graph()
    for i, j in sorted(coordinates):
        polyhedron.add_edge((i, j), (j, i))
        for k in sorted(adjacency[i]):
            if k != j and tuple(sorted((j, k))) in edges:
                polyhedron.add_edge((i, j), (i, k))
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, polyhedron)
    assert matcher.is_isomorphic(), "The cover embedding must preserve Cayley adjacency"
    mapping = matcher.mapping
    points = {p: coordinates[mapping[p]] for p in sorted(elements)}
    lengths = [np.linalg.norm(points[u] - points[v]) for u, v in graph.edges()]
    assert np.allclose(lengths, lengths[0]), "The regular embedding must retain equal edge lengths"
    return graph, points, a, b


def cover():
    fig = canvas(False, 900, width=1600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1600), ylim=(900, 0)); ax.axis("off")
    graph, points, a, b = a5_graph()
    # A genuine graph embedding, not an inferred trajectory of the five roots.
    angle, tilt = .41, -.28
    rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    rotation = rx @ rz
    points = {p: rotation @ xyz for p, xyz in points.items()}
    radius = max(np.linalg.norm(xyz) for xyz in points.values())
    xy = {p: np.array([1118 + 335 * xyz[0] / radius,
                       425 - 335 * xyz[1] / radius]) for p, xyz in points.items()}
    for u, v in sorted(graph.edges(), key=lambda edge: (points[edge[0]][2] + points[edge[1]][2]) / 2):
        depth = (points[u][2] + points[v][2]) / (2 * radius)
        color = TEAL if graph.edges[u, v]["kind"] == "a" else BLUE
        alpha = .19 + .60 * (depth + 1) / 2
        ax.plot([xy[u][0], xy[v][0]], [xy[u][1], xy[v][1]],
                color=color, alpha=alpha, lw=2.0 + 1.0 * (depth + 1) / 2, zorder=2 + depth)
    for p, xyz in sorted(points.items(), key=lambda item: item[1][2]):
        depth = xyz[2] / radius
        ax.scatter(*xy[p], s=36 + 30 * (depth + 1) / 2,
                   color=TEAL if depth > 0 else "#c3d4d7", edgecolors="white",
                   linewidths=1, zorder=4 + depth)
    label(ax, 95, 250, "Galois", size=112, fontfamily="STIXGeneral", weight="normal")
    label(ax, 95, 418, "Theory", size=112, fontfamily="STIXGeneral", weight="normal")
    ax.plot([102, 536], [554, 554], color=TEAL, lw=3)
    label(ax, 100, 645, r"$x^5-5x+1$", size=60, color=INK)
    label(ax, 100, 770, r"$S_5\;\supset\;A_5=A_5'$", size=59, color=MUTED)
    label(ax, 1118, 828, r"$A_5$  ·  60 elements", size=58, ha="center", color=TEAL)
    fig._cover_graph = {"vertices": 60, "edges": 90, "degree": 3,
                        "a": list(a), "b": list(b), "embedding": "adjacency-preserving"}
    return fig


def order_plot(ax, mobile, quiz=False):
    rows = [(r"$S_3$", [6, 3, 1], BLUE)]
    if not quiz:
        rows.append((r"$S_4$", [24, 12, 4, 1], TEAL))
    rows.append((r"$S_5$", [120, 60, 60], RED))
    for name, values, color in rows:
        steps = np.arange(len(values))
        ax.plot(steps, values, "o-", color=color, lw=2.1, ms=7, label=name, zorder=4)
        for j, value in enumerate(values):
            dy = 10
            ax.annotate(str(value), (j, value), xytext=(0, dy),
                        textcoords="offset points", fontsize=20 if mobile else 18,
                        ha="center", va="bottom" if dy > 0 else "top", color=color)
    ax.plot([2, 3.35], [60, 60], color=RED, lw=2, ls=(0, (3, 3)))
    ax.set_yscale("log")
    ax.set(xlim=(-.24, 3.6), ylim=(.7, 340 if quiz else 230),
           xticks=[0, 1, 2, 3], yticks=[1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.minorticks_off()
    clean(ax)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.axhline(1, color=GRID, lw=1, zorder=1)
    ax.set_xlabel(r"Derived step $j$", fontsize=20 if mobile else 18, labelpad=12)
    ax.set_ylabel("Group order (log)", fontsize=20 if mobile else 18, labelpad=12)
    if quiz:
        label(ax, 2.8, 1.7, r"$S_3$", size=22 if mobile else 20, color=BLUE)
        label(ax, 2.8, 95, r"$S_5$", size=22 if mobile else 20, color=RED)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.19), ncol=len(rows),
                  frameon=False, fontsize=21 if mobile else 19,
                  handlelength=1.3, columnspacing=.9)
        label(ax, 2.26, 87, r"$A_5'=A_5$", size=22 if mobile else 20, color=RED)


def fig07(mobile):
    fig = canvas(mobile, 600 if mobile else 520)
    ax = fig.add_axes([.25, .18, .70, .65] if mobile else [.12, .19, .81, .65])
    order_plot(ax, mobile)
    return fig


def expression_tree(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    locations = {"root": (.46, .86), "plus": (.46, .57), "two": (.18, .21),
                 "sqrt": (.78, .37), "three": (.78, .07)}
    for child, parent in (("plus", "root"), ("two", "plus"), ("sqrt", "plus"), ("three", "sqrt")):
        x, y = locations[child]; xx, yy = locations[parent]
        arrow(ax, (x, y + .055), (xx, yy - .075), color=GRID, scale=12)
    for key, text in [("root", "√"), ("plus", "+"), ("two", "2"), ("sqrt", "√"), ("three", "3")]:
        x, y = locations[key]
        label(ax, x, y, text, size=28 if ax.figure._mobile else 25,
              color=TEAL if key in ("root", "sqrt") else INK, ha="center")
    label(ax, .02, .86, r"$\sqrt{2+\sqrt3}$", ha="left",
          size=21 if ax.figure._mobile else 18, color=MUTED)


def linear_quiz(ax):
    ax.set(xlim=(-.2, 2.3), ylim=(-.2, 2.55), xticks=[0, 1, 2], yticks=[0, 1, 2])
    clean(ax, grid=True)
    arrow(ax, (0, 0), (1, 1), color=BLUE, lw=2)
    arrow(ax, (0, 0), (1, 2), color=TEAL, lw=2)
    label(ax, 1.1, 1, r"$(a,b)$", color=BLUE)
    label(ax, 1.1, 2.13, r"$(a,2b)$", color=TEAL)
    label(ax, 2.17, .17, r"$a$", ha="center", size=22)
    label(ax, .13, 2.35, r"$b$", ha="center", size=22)


def permutation_quiz(ax):
    ax.set(xlim=(-.55, 1.55), ylim=(-.5, 3.5)); ax.axis("off")
    labels = [r"$\sqrt2$", r"$-\sqrt2$", r"$\sqrt3$", r"$-\sqrt3$"]
    ys, permutation = [3, 2, 1, 0], [2, 3, 0, 1]
    for i, y in enumerate(ys):
        color = BLUE if i < 2 else GOLD
        ax.scatter([0, 1], [y, y], s=45, color=color, zorder=5)
        label(ax, -.13, y, labels[i], ha="right", color=color, size=20 if ax.figure._mobile else 18)
        label(ax, 1.13, y, labels[i], ha="left", color=color, size=20 if ax.figure._mobile else 18)
        arrow(ax, (.04, y), (.94, ys[permutation[i]]), color=color, lw=1.4, scale=12)


def fixed_field_quiz(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    for y, value in ((.85, r"$L=\mathbb Q(\sqrt2,\sqrt3)$"),
                     (.52, r"$E=\mathbb Q(\sqrt3)$"), (.19, r"$\mathbb Q$")):
        ax.scatter([.11], [y], s=55, color=TEAL, zorder=5)
        label(ax, .20, y, value, size=20 if ax.figure._mobile else 18)
    ax.plot([.11, .11], [.19, .85], color=GRID, lw=2)
    label(ax, .5, .02, r"$\sigma:\sqrt2\mapsto-\sqrt2,\ \sqrt3\mapsto\sqrt3$",
          ha="center", size=20 if ax.figure._mobile else 18, color=MUTED)


def fig10(mobile):
    fig = canvas(mobile, 1740 if mobile else 1180)
    if mobile:
        rects = [[.12, y, .78, .135] for y in (.795, .607, .419, .231, .043)]
        rects[4] = [.25, .063, .66, .13]
    else:
        rects = [[.08, .70, .36, .235], [.59, .70, .33, .235],
                 [.08, .39, .36, .235], [.58, .39, .34, .235],
                 [.14, .075, .78, .20]]
    names = ["1. Finite expression", "2. A linear map", "3. Root-pair exchange",
             "4. Fix E pointwise", "5. Derived-series orders"]
    makers = [expression_tree, linear_quiz, permutation_quiz, fixed_field_quiz]
    for rect, name, maker in zip(rects, names, makers):
        ax = panel(fig, rect, name)
        maker(ax)
    ax = fig.add_axes(rects[4])
    order_plot(ax, mobile, quiz=True)
    # The group names are the legend; the condition, not its answer, is drawn.
    label(ax, 0, 1.12, names[4],
          transform=ax.transAxes, size=21 if mobile else 19)
    return fig
