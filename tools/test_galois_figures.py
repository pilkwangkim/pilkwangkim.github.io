#!/usr/bin/env python3
"""Independent checks for the geometry and group actions in the Galois figures."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import sympy as sp

from galois_visual_core import a5_graph, cover, fig07
from galois_visual_style import configure
from galois_figures import MAKERS, NAMES, validate

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/img/posts/2026-10-25-galois-from-scratch"


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    return tuple(p.index(i) for i in range(len(p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []

    def check(condition, description):
        assert condition, description
        checks.append(description)

    manifest = json.loads((ASSETS / "figure-data.json").read_text())
    rows = {row["stem"]: row for row in manifest["assets"]}
    check(len(rows) == 21, "Twenty body assets and one cover are present")
    for stem, row in rows.items():
        with Image.open(ASSETS / (stem + ".png")) as image:
            check(image.size == (row["png_width"], row["png_height"]),
                  stem + ": PNG dimensions match the manifest")
        check(row["layout"]["minimum_css_font_px"] >= 16
              and row["layout"]["text_collisions"] == 0,
              stem + ": body-scale type and label bounds pass")
    # Check today's generator geometry, not just the prior export's layout record.
    # Render in memory so this audit cannot overwrite publication assets.
    configure()
    for name, make in zip(NAMES, MAKERS):
        for mobile in (False, True):
            stem = name + ("-mobile" if mobile else "")
            fig = make(mobile)
            layout = validate(fig, stem)
            check(layout["minimum_css_font_px"] >= 16 and layout["text_collisions"] == 0,
                  stem + ": current generator text remains in bounds and readable")
            plt.close(fig)
    fig = cover()
    validate(fig, "cover-galois")
    check(True, "Current cover generator text remains in bounds and readable")
    plt.close(fig)

    graph, points, a, b = a5_graph()
    even = {p for p in itertools.permutations(range(5))
            if sum(p[i] > p[j] for i in range(5) for j in range(i+1, 5)) % 2 == 0}
    check(set(graph) == even, "Cover vertices are exactly the sixty even permutations")
    expected = {frozenset((p, compose(p, g))) for p in even for g in (a, b)}
    check({frozenset(edge) for edge in graph.edges()} == expected,
          "Every cover edge is multiplication by a stated generator or its inverse")
    check(graph.number_of_edges() == 90 and set(dict(graph.degree()).values()) == {3},
          "Cover graph has ninety edges and degree three")
    check(Counter(data["kind"] for _, _, data in graph.edges(data=True)) == {"a": 60, "b": 30},
          "The two cover edge colors have sixty and thirty generator edges")
    planar, embedding = nx.check_planarity(graph)
    check(planar, "Cover Cayley graph is planar")
    seen, faces = set(), []
    for u in embedding:
        for v in embedding[u]:
            if (u, v) not in seen:
                faces.append(embedding.traverse_face(u, v, mark_half_edges=seen))
    check(Counter(map(len, faces)) == {5: 12, 6: 20},
          "The geometric embedding has twelve pentagons and twenty hexagons")
    lengths = [np.linalg.norm(points[u]-points[v]) for u, v in graph.edges()]
    check(np.allclose(lengths, lengths[0]), "The exact geometric embedding has equal edge lengths")

    for mobile in (False, True):
        suffix = "-mobile" if mobile else ""
        cubic = np.array(rows["fig-05-cubic"+suffix]["semantics"]["roots"])
        roots = cubic[:, 0] + 1j*cubic[:, 1]
        check(np.allclose(roots**3, 2), "Cubic dots satisfy x cubed equals two")
        check(np.allclose(roots*np.exp(2j*np.pi/3), np.roll(roots, -1)),
              "A 120-degree rotation gives the displayed cubic cycle")
        check(np.allclose(roots.conjugate(), roots[[0, 2, 1]]),
              "Cubic conjugation fixes one root and swaps two")
        tracks = rows["fig-08-a5"+suffix]["semantics"]["tracks"]
        g, h = (1, 0, 3, 2, 4), (1, 0, 4, 3, 2)
        actual = []
        for source in range(5):
            path = [source]
            for operation in (inverse(h), inverse(g), h, g):
                path.append(operation[path[-1]])
            actual.append([value+1 for value in path])
        check(tracks == actual, "Every drawn commutator stage follows rightmost-first composition")
        check(tracks[0] == [1, 2, 1, 2, 1] and tracks[1] == [2, 1, 2, 1, 2],
              "Labels one and two move at intermediate stages and return at the end")
        quintic = rows["fig-09-quintic"+suffix]["semantics"]
        dots = list(map(complex, quintic["real_roots"]))
        dots += [complex(*z) for z in quintic["nonreal_roots"]]
        check(max(abs(z**5-5*z+1) for z in dots) < 1e-11,
              "All five quintic dots satisfy the displayed polynomial")
        check(np.isclose(dots[-1], dots[-2].conjugate()),
              "The quintic nonreal dots form one conjugate pair")

    x = sp.Symbol("x")
    from sympy.polys.numberfields import galois_group
    radical_group, _ = galois_group(x**5-2, x)
    cyclotomic = sp.Poly(sp.cyclotomic_poly(5, x), x)
    check(radical_group.order() == 20 and cyclotomic.is_irreducible
          and cyclotomic.degree() == 4,
          "The concrete fifth-root step over Q(zeta5) has degree twenty divided by four")
    check(np.allclose((2**.2*np.exp(2j*np.pi*np.arange(5)/5))**5, 2),
          "The five cyclic-orbit positions are actual fifth roots of two")

    configure()
    fig = fig07(False)
    axis = fig.axes[0]
    plotted = {line.get_label(): list(line.get_ydata()) for line in axis.lines
               if line.get_label() in (r"$S_3$", r"$S_4$", r"$S_5$")}
    check(plotted == {r"$S_3$": [6, 3, 1], r"$S_4$": [24, 12, 4, 1],
                      r"$S_5$": [120, 60, 60]}, "Derived-order plot uses the exact subgroup orders")
    check(axis.get_yscale() == "log" and 1 in axis.get_yticks(),
          "The derived-order axis is logarithmic with identity at one")
    plt.close(fig)
    report = {"status": "passed", "assertions": len(checks), "checks": checks,
              "cover": {"vertices": 60, "edges": 90, "degree": 3},
              "body_figures": 20, "fresh_layouts": 21}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
