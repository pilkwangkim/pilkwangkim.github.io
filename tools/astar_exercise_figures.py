#!/usr/bin/env python3
"""Generate answer-free A* exercise inputs at desktop and mobile sizes.

Every arrow is a directed edge; its number is the complete move cost. Graph
heuristics are adjacent to nodes, never mixed with edge costs. SVG text is
outlined so the notebook and website do not depend on installed fonts.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-10-24-astar-from-scratch"
BLUE, GREEN, INK, MUTED = "#235789", "#18794e", "#1c3045", "#576879"
GOLD, PALE, GRID = "#be7a10", "#edf4f8", "#c8d4dd"
INPUTS = {
    "01-goal": {"edges": [["S", "G", 10], ["S", "A", 1], ["A", "G", 1]],
                "heuristic": {"S": 0, "A": 0, "G": 0}},
    "02-reopening": {"edges": [["S", "A", 3], ["S", "B", 1], ["B", "A", 1],
                                    ["A", "G", 3], ["B", "G", 100]],
                     "heuristic": {"S": 5, "A": 0, "B": 4, "G": 0}},
    "03-moves": {"start": [0, 0], "goal": [3, 2], "obstacles": [],
                 "models": [{"directions": 4, "orthogonal_cost": 1},
                            {"directions": 8, "orthogonal_cost": 1,
                             "diagonal_cost": "sqrt(2)"},
                            {"directions": 8, "orthogonal_cost": 1,
                             "diagonal_cost": 1}]},
    "04-boundaries": {"cases": [
        {"name": "start-is-goal", "start": "S", "goal": "S", "edges": []},
        {"name": "unreachable", "start": "S", "goal": "G", "edges": [["S", "A", 1]]},
        {"name": "zero-cycle", "start": "S", "goal": "G",
         "edges": [["S", "A", 0], ["A", "S", 0], ["A", "G", 2]]}]},
    "05-teleport": {"straight_line_km": 60, "road_length_km": 90,
                    "maximum_road_speed_kmh": 120, "teleport_cost_hours": 0},
    "06-code": {"edges": [["S", "A", 1], ["A", "G", 20],
                           ["S", "B", 4], ["B", "G", 1]],
                "heuristic": {"S": 5, "A": 0, "B": 1, "G": 0}},
}


def style(mobile):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 19 if mobile else 23,
        "axes.titlesize": 20 if mobile else 24, "mathtext.fontset": "stix",
        "svg.fonttype": "path", "svg.hashsalt": "astar-exercise-inputs-v1",
        "text.color": INK, "axes.titlepad": 17,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def canvas(mobile, panels=1, height=4.8):
    cols, rows = (1, panels) if mobile else (panels, 1)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(5 if mobile else 10.8, height*rows),
                             layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.08, h_pad=.10,
                                     wspace=.10 if panels > 1 else 0, hspace=.15)
    return fig, list(axes.flat)


def bare(ax, xlim=(-.65, 4.65), ylim=(-2.1, 2.1)):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


def text(ax, x, y, value, **kw):
    defaults = {"ha": "center", "va": "center", "zorder": 8}
    defaults.update(kw)
    artist = ax.text(x, y, value, **defaults)
    return artist


def nodes(ax, positions, heuristic=None, hpositions=None, radius=.28):
    patches = {}
    for name, pos in positions.items():
        patches[name] = Circle(pos, radius, fc=PALE if name in ("S", "G") else "white",
                               ec=BLUE, lw=2.4, zorder=5)
        ax.add_patch(patches[name])
        label = text(ax, *pos, name, color=BLUE, weight="bold")
        label.set_gid("node-label")
        if heuristic is not None:
            hx, hy = hpositions[name]
            label = text(ax, hx, hy, f"h={heuristic[name]}", color=MUTED,
                         fontsize=plt.rcParams["font.size"]*.96)
            label.set_gid("node-heuristic")
    return patches


def arrow(ax, start, end, patches=None, names=None, label=None,
          labelpos=None, radius=0, color=MUTED, dashed=False):
    kw = {"patchA": patches[names[0]], "patchB": patches[names[1]]} if patches else {}
    item = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=21,
                           linewidth=2.0, color=color, shrinkA=4, shrinkB=5,
                           connectionstyle=f"arc3,rad={radius}",
                           linestyle="--" if dashed else "-", zorder=2, **kw)
    ax.add_patch(item)
    if label is not None:
        labelpos = labelpos or ((start[0]+end[0])/2, (start[1]+end[1])/2)
        cost = text(ax, *labelpos, str(label), color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        cost.set_gid("edge-cost")
    return item


def graph01(mobile):
    fig, (ax,) = canvas(mobile, height=4.5)
    bare(ax, ylim=(-1.8, 1.6))
    p = {"S": (0, .7), "G": (4, .7), "A": (2, -.85)}
    n = nodes(ax, p)
    arrow(ax, p["S"], p["G"], n, ("S", "G"), 10, (2, 1.02))
    arrow(ax, p["S"], p["A"], n, ("S", "A"), 1, (.85, -.2))
    arrow(ax, p["A"], p["G"], n, ("A", "G"), 1, (3.15, -.2))
    text(ax, 2, -1.55, "h(n) = 0 at every node", color=MUTED)
    ax.set_title("When may the search stop?")
    return fig


def diamond(mobile, key, title):
    fig, (ax,) = canvas(mobile, height=5.4)
    bare(ax, ylim=(-2.1, 2.05))
    p = {"S": (0, 0), "A": (2, 1.25), "B": (2, -1.25), "G": (4, 0)}
    hp = {"S": (0, -.7), "A": (2, 1.85), "B": (2, -1.88), "G": (4, -.7)}
    n = nodes(ax, p, INPUTS[key]["heuristic"], hp)
    lp = {("S", "A"): (.8, .85), ("S", "B"): (.8, -.85),
          ("B", "A"): (2.33, 0), ("A", "G"): (3.2, .85),
          ("B", "G"): (3.2, -.85)}
    for a, b, c in INPUTS[key]["edges"]:
        arrow(ax, p[a], p[b], n, (a, b), c, lp[a, b])
    ax.set_title(title)
    return fig


def graph02(mobile):
    return diamond(mobile, "02-reopening", "Can a processed node improve?")


def graph03(mobile):
    fig, axes = canvas(mobile, panels=3, height=5.15 if mobile else 5.3)
    titles = ["(a) 4-way moves", "(b) 8-way moves", "(c) 8-way moves"]
    rules = ["Every move: 1", "Orthogonal: 1\nDiagonal: " + r"$\sqrt{2}$", "Every move: 1"]
    for j, ax in enumerate(axes):
        bare(ax, (-.65, 3.65), (-2.5, 3.25))
        for x in np.arange(-.5, 4, 1):
            ax.plot([x, x], [-.5, 2.5], color=GRID, lw=1.1, zorder=0)
        for y in np.arange(-.5, 3, 1):
            ax.plot([-.5, 3.5], [y, y], color=GRID, lw=1.1, zorder=0)
        direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if j:
            direction += [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for dx, dy in direction:
            arrow(ax, (1, 1), (1+.82*dx, 1+.82*dy), color="#91a6b7")
        ax.plot(1, 1, "o", color=MUTED, ms=4)
        n = nodes(ax, {"S": (0, 0), "G": (3, 2)}, radius=.29)
        text(ax, 0, -.85, "(0, 0)", fontsize=18 if mobile else 19, color=MUTED)
        text(ax, 3, 2.9, "(3, 2)", fontsize=18 if mobile else 19, color=MUTED)
        text(ax, 1.5, -1.95, rules[j], fontsize=18 if mobile else 22, linespacing=1.35)
        ax.set_title(titles[j], fontsize=20 if mobile else 24)
    return fig


def graph04(mobile):
    fig, axes = canvas(mobile, panels=3, height=3.9 if mobile else 4.8)
    ax = axes[0]
    bare(ax, (-.7, 4.7), (-2.3, 1.4))
    n = nodes(ax, {"S = G": (2, 0)}, radius=.9)
    text(ax, 2, -1.7, "One state", color=MUTED)
    ax.set_title("(a) S is G", fontsize=20 if mobile else 24)
    ax = axes[1]
    bare(ax, (-.7, 4.7), (-2.3, 1.4))
    p = {"S": (0, 0), "A": (2, 0), "G": (4, 0)}
    n = nodes(ax, p)
    arrow(ax, p["S"], p["A"], n, ("S", "A"), 1, (1, .42))
    text(ax, 2, -1.7, "No other edges", color=MUTED)
    ax.set_title("(b) Unreachable G", fontsize=20 if mobile else 24)
    ax = axes[2]
    bare(ax, (-.7, 4.7), (-2.3, 1.4))
    p = {"S": (0, 0), "A": (2, 0), "G": (4, 0)}
    n = nodes(ax, p)
    # Opposite directed arcs have their own arrowheads and cost labels.
    arrow(ax, p["S"], p["A"], n, ("S", "A"), 0, (1, .62), radius=-.35)
    arrow(ax, p["A"], p["S"], n, ("A", "S"), 0, (1, -.62), radius=-.35)
    arrow(ax, p["A"], p["G"], n, ("A", "G"), 2, (3, .42))
    text(ax, 2, -1.7, "0 in both directions", color=MUTED,
         fontsize=18 if mobile else 21)
    ax.set_title("(c) Zero-cost cycle", fontsize=20 if mobile else 24)
    return fig


def road_panel(ax, mobile, teleport):
    bare(ax, (-.65, 4.65), (-2.55, 2.9))
    p = {"S": (0, 0), "G": (4, 0)}
    n = nodes(ax, p)
    # The curved directed road and straight geometric separation are distinct.
    arrow(ax, p["S"], p["G"], n, ("S", "G"), radius=-.58)
    text(ax, 2, 2.0, "Road length: 90 km\nSpeed ≤ 120 km/h",
         fontsize=18 if mobile else 22, linespacing=1.35)
    ax.plot([.35, 3.65], [-.06, -.06], color=GOLD, lw=1.7, linestyle="--", zorder=1)
    text(ax, 2, -.02, "Straight line:\n60 km", color=GOLD,
         fontsize=18 if mobile else 22, linespacing=1.2,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    if teleport:
        arrow(ax, p["S"], p["G"], n, ("S", "G"), radius=.8,
              color=GREEN, dashed=True)
        text(ax, 2, -2.12, "Teleport: 0 h", color=GREEN)
    else:
        text(ax, 2, -2.12, "Only road travel", fontsize=18 if mobile else 22)
    ax.set_title("(b) Add a teleport edge" if teleport else "(a) Travel by road only",
                 fontsize=20 if mobile else 24)


def graph05(mobile):
    fig, axes = canvas(mobile, panels=2, height=4.9)
    road_panel(axes[0], mobile, False)
    road_panel(axes[1], mobile, True)
    return fig


def graph06(mobile):
    return diamond(mobile, "06-code", "Coding challenge: directed input")


def validate_layout(fig, name):
    """Reject clipping, text collisions and non-cost labels on graph arrows.

    Edge labels intentionally have opaque backgrounds and may interrupt their
    own edge. All other text must avoid arrows. Node labels must fit within the
    circle's bounding square, preventing the one-state S=G label from escaping.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for ax in fig.axes for t in [ax.title, *ax.texts]
              if t.get_visible() and t.get_text().strip()]
    items = [(t, t.get_window_extent(renderer)) for t in labels]
    errors = []
    for t, box in items:
        if (box.x0 < -.5 or box.x1 > fig.bbox.x1+.5 or box.y0 < -.5 or box.y1 > fig.bbox.y1+.5):
            errors.append(f"clipped {t.get_text()!r}")
    for j, (a, ba) in enumerate(items):
        for b, bb in items[j+1:]:
            if min(ba.x1, bb.x1)-max(ba.x0, bb.x0) > 1 and min(ba.y1, bb.y1)-max(ba.y0, bb.y0) > 1:
                errors.append(f"overlapping {a.get_text()!r} and {b.get_text()!r}")
    arrow_labels_checked = 0
    for ax in fig.axes:
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        circles = [p for p in ax.patches if isinstance(p, Circle)]
        for label in ax.texts:
            box = label.get_window_extent(renderer)
            if label.get_gid() == "node-label":
                circle = next(p for p in circles if tuple(p.center) == label.get_position())
                nodebox = circle.get_window_extent(renderer)
                if not (nodebox.x0 <= box.x0 and box.x1 <= nodebox.x1
                        and nodebox.y0 <= box.y0 and box.y1 <= nodebox.y1):
                    errors.append(f"node label does not fit {label.get_text()!r}")
                continue
            if label.get_gid() == "edge-cost":
                continue
            arrow_labels_checked += 1
            for arrow_item in arrows:
                polygons = arrow_item.get_path().transformed(arrow_item.get_transform()).to_polygons(closed_only=False)
                if any(PlotPath(poly).intersects_bbox(box.expanded(1.03, 1.03), filled=False)
                       for poly in polygons):
                    errors.append(f"arrow intersects {label.get_text()!r}")
    if errors:
        raise AssertionError(name + ": " + "; ".join(errors))
    return {"labels_checked": len(items), "text_bounds": "PASS", "text_collisions": "PASS",
            "non_cost_labels_checked_against_arrows": arrow_labels_checked,
            "arrows_avoid_non_cost_labels": "PASS", "labels_fit_nodes": "PASS"}


def shortest(edges, source="S", goal="G"):
    """Independent heap shortest-path check; the diagrams disclose only input."""
    dist, queue = {source: 0}, [(0, source)]
    while queue:
        g, n = heapq.heappop(queue)
        if g != dist[n]:
            continue
        if n == goal:
            return g
        for a, b, c in edges:
            if a == n and g+c < dist.get(b, float("inf")):
                dist[b] = g+c
                heapq.heappush(queue, (g+c, b))
    return None


def sanity():
    checks = {}
    for key, expected in [("01-goal", 2), ("02-reopening", 5), ("06-code", 5)]:
        item = INPUTS[key]
        assert shortest(item["edges"]) == expected
        for node, h in item["heuristic"].items():
            assert 0 <= h <= shortest(item["edges"], node)
        checks[key] = {"shortest_cost": expected, "heuristic_admissible": True}
    cases = INPUTS["04-boundaries"]["cases"]
    assert [shortest(c["edges"], c["start"], c["goal"]) for c in cases] == [0, None, 2]
    # Enumerate the three finite obstacle-free grid models, rather than relying
    # on formulas that students are asked to identify.
    for model, expected in zip(INPUTS["03-moves"]["models"], [5, 1+2*np.sqrt(2), 3]):
        edges = []
        for x in range(4):
            for y in range(3):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if (dx == dy == 0 or not (0 <= x+dx < 4 and 0 <= y+dy < 3)
                                or (dx and dy and model["directions"] == 4)):
                            continue
                        c = model.get("diagonal_cost", 1) if dx and dy else 1
                        c = np.sqrt(2) if c == "sqrt(2)" else c
                        edges.append(((x, y), (x+dx, y+dy), c))
        assert abs(shortest(edges, (0, 0), (3, 2))-expected) < 1e-12
    checks["03-moves"] = "All three movement costs verified by enumerated graph"
    checks["04-boundaries"] = "Start=goal, no-route and zero-cycle inputs verified"
    checks["05-teleport"] = "km / (km/h) has hours; zero-time edge changes the graph"
    return checks


def contact_sheet(mobile):
    """Inspection overview at actual 691/328-pixel article widths, not browser QA."""
    target_width = 328 if mobile else 691
    columns = 3 if mobile else 2
    suffix = "-mobile" if mobile else ""
    thumbnails = []
    for key in INPUTS:
        with Image.open(OUT / f"exercise-{key}{suffix}.png") as source:
            thumbnails.append(source.convert("RGB").resize(
                (target_width, round(source.height*target_width/source.width)), Image.Resampling.LANCZOS))
    row_heights = [max(im.height for im in thumbnails[j:j+columns])
                   for j in range(0, len(thumbnails), columns)]
    gutter = 18
    sheet = Image.new("RGB", (columns*target_width+(columns-1)*gutter,
                              sum(row_heights)+(len(row_heights)-1)*gutter), "#edf1f5")
    y = 0
    for row, height in enumerate(row_heights):
        for column, im in enumerate(thumbnails[row*columns:(row+1)*columns]):
            sheet.paste(im, (column*(target_width+gutter), y))
        y += height+gutter
    stem = "exercise-contactsheet" + ("-mobile" if mobile else "-desktop")
    sheet.save(OUT / (stem+".png"))
    return {"stem": stem, "width": sheet.width, "height": sheet.height,
            "individual_image_width": target_width,
            "scope": "Resized native figures only; final browser rendering must be checked separately."}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_path = OUT / "figure-data.json"
    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    makers = [graph01, graph02, graph03, graph04, graph05, graph06]
    assets = []
    for mobile in (False, True):
        style(mobile)
        for key, maker in zip(INPUTS, makers):
            stem = "exercise-" + key + ("-mobile" if mobile else "")
            fig = maker(mobile)
            layout = validate_layout(fig, stem)
            svg = OUT / (stem + ".svg")
            fig.savefig(svg, metadata={"Date": None})
            # Mechanical SVG normalization only; input/code sources use patches.
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
            fig.savefig(OUT / (stem + ".png"), dpi=140,
                        metadata={"Software": "Matplotlib"})
            plt.close(fig)
            with Image.open(OUT / (stem + ".png")) as image:
                width, height = image.size
            assets.append({"stem": stem, "width": width, "height": height,
                           "svg_width_pt": 360 if mobile else 777.6,
                           "svg_height_pt": round(height/140*72, 3),
                           "layout": layout})
    assert old_hash == hashlib.sha256(old_path.read_bytes()).hexdigest()
    record = {"inputs": INPUTS, "verification": sanity(),
              "original_figure_data_sha256": old_hash, "assets": assets,
              "inspection_contact_sheets": [contact_sheet(False), contact_sheet(True)],
              "answer_visibility": "Figures show given inputs only; no solution route or cost is marked.",
              "coordinate_convention": "Exercise 03 uses x rightwards and y upwards; (0,0) is S."}
    (OUT / "exercise-figure-data.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"assets": assets, "verification": record["verification"]}, indent=2))


if __name__ == "__main__":
    main()
