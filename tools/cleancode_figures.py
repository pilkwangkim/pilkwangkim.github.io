#!/usr/bin/env python3
"""Exact state, test-contract and boundary diagrams for Clean Code / TDD.

Coordinates and type sizes use source pixels. SVG glyphs are outlined; separate
mobile compositions are audited at the real 328px article width. No measured
quality, speed or coverage score is implied by any diagram.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-11-04-clean-code-tdd"
INK, MUTED, GRID = "#29343D", "#6C7780", "#D9DFE3"
TEAL, RED, GOLD = "#536F7A", "#916865", "#847660"
PALE, TP, RP, GP = "#F5F7F8", "#F0F4F5", "#F7F3F2", "#F6F4F0"
NAMES = ["fig-01-state", "fig-02-test", "fig-03-cycle", "fig-04-refactor",
         "fig-05-boundaries"]


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "cleancode-tdd-exact-diagrams-v1",
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": INK, "lines.solid_capstyle": "round"})


class Canvas:
    def __init__(self, mobile, height, width=None):
        self.w, self.h = width or (480 if mobile else 960), height
        # Agg truncates floating canvas sizes: 10.2 * 100 may become 1019.9999.
        # One representable step upward preserves the requested integer pixels.
        self.fig = plt.figure(figsize=(math.nextafter(self.w / 100, math.inf),
                                       math.nextafter(self.h / 100, math.inf)), dpi=100)
        self.fig._mobile, self.fig._width, self.fig._height = mobile, self.w, self.h
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, self.w), ylim=(self.h, 0))
        self.ax.axis("off")
        self.fig._semantics = {}

    def text(self, x, y, value, px=28, color=INK, ha="left", weight="normal", mono=False):
        return self.ax.text(x, y, value, fontsize=px * .72, color=color,
                            ha=ha, va="center", weight="medium" if weight == "bold" else weight,
                            fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans")

    def box(self, x, y, w, h, fill=PALE, edge=GRID, radius=12, lw=1.3):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=edge, linewidth=lw * .9)
        self.ax.add_patch(patch)
        return patch

    def line(self, a, b, color=GRID, lw=1.5, dash=None):
        self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw * .9,
                     linestyle=dash or "-", zorder=1)

    def arrow(self, a, b, color=INK, lw=1.8, curve=0):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                                         linewidth=lw * .9, color=color,
                                         connectionstyle=f"arc3,rad={curve}"))

    def token(self, x, y, name, color=INK, radius=29, fill=None, px=32):
        self.ax.add_patch(Circle((x, y), radius, facecolor=fill or "white",
                                 edgecolor=color, linewidth=2))
        self.text(x, y, name, px, color, ha="center", weight="bold")

    def seat(self, x, y, name, color=INK, size=80, px=36, canceled=False):
        self.box(x - size / 2, y - size / 2, size, size, "white", color, 10, 2)
        self.line((x-size*.38, y+size*.28), (x+size*.38, y+size*.28), color, 3)
        self.text(x, y-4, name, px, color, ha="center", weight="bold")
        if canceled:
            self.line((x-size*.38, y-size*.38), (x+size*.38, y+size*.38), RED, 2)

    def queue(self, x, y, users, spacing=95, radius=29, px=32):
        for i, user in enumerate(users):
            if i:
                self.arrow((x+(i-1)*spacing+radius+4, y),
                           (x+i*spacing-radius-4, y), MUTED, lw=1.4)
            self.token(x+i*spacing, y, user, TEAL, radius, TP, px)

    def code(self, x, y, lines, px=26, step=43, colors=None):
        for i, value in enumerate(lines):
            self.text(x, y+i*step, value, px, (colors or {}).get(i, INK), mono=True)


def state(capacity=2, confirmed=(), waiting=()):
    return {"capacity": capacity, "confirmed": list(confirmed), "waiting": list(waiting)}


def register(source, user, bug=None):
    result = source if bug == "source-mutation" else deepcopy(source)
    known = user in result["confirmed"] + result["waiting"]
    if known and bug != "duplicate-allowed":
        return result
    has_seat = (len(result["confirmed"]) <= result["capacity"]
                if bug == "overfull-boundary"
                else len(result["confirmed"]) < result["capacity"])
    result["confirmed" if has_seat else "waiting"].append(user)
    return result


def cancel(source, user, bug=None):
    result = source if bug == "source-mutation" else deepcopy(source)
    if user in result["confirmed"]:
        result["confirmed"].remove(user)
        if result["waiting"]:
            result["confirmed"].append(result["waiting"].pop(-1 if bug == "lifo-promotion" else 0))
    elif user in result["waiting"]:
        result["waiting"].remove(user)
    return result


BUGS = ["overfull-boundary", "duplicate-allowed", "lifo-promotion", "source-mutation"]
TESTS = ["boundary", "duplicate", "fifo", "nonmutation"]


def run_case(test, bug=None):
    """Return whether a focused assertion passes for an executable mutant."""
    if test == "boundary":
        return register(state(1, ["A"]), "B", bug) == state(1, ["A"], ["B"])
    if test == "duplicate":
        return register(state(1, ["A"], ["B"]), "B", bug) == state(1, ["A"], ["B"])
    if test == "fifo":
        return cancel(state(1, ["A"], ["B", "C"]), "A", bug) == state(1, ["B"], ["C"])
    if test == "nonmutation":
        source = state(2)
        before = deepcopy(source)
        register(source, "A", bug)
        return source == before
    raise ValueError(test)


def domain_examples():
    initial = state(2, ["A", "B"], ["C", "D"])
    after = cancel(initial, "A")
    assert after == state(2, ["B", "C"], ["D"])
    assert initial == state(2, ["A", "B"], ["C", "D"])
    matrix = [[not run_case(test, bug) for bug in BUGS] for test in TESTS]
    assert all(run_case(test) for test in TESTS)
    return {"cancellation": {"initial": initial, "operation": "cancel(A)", "result": after},
            "independent_expectation": {"input": state(2), "operation": "register(A)",
                                        "expected": state(2, ["A"]),
                                        "actual": register(state(2), "A")},
            "assertion_counterexample": {"input": initial, "operation": "cancel(A)",
                                         "expected": state(2, ["B", "C"], ["D"]),
                                         "faulty_actual": cancel(initial, "A", "lifo-promotion"),
                                         "bug": "lifo-promotion"},
            "mutation_test_matrix": {"tests": TESTS, "bugs": BUGS,
                                     "detected": matrix,
                                     "basis": "Four focused executable Python fixtures; JS parity checked separately."}}


# Each plate has its own visual grammar: state trajectories, value comparison,
# an iterative cycle, extraction dependencies, and file/callback boundaries.
from cleancode_visual_flow import fig01, fig03, fig04
from cleancode_visual_contract import fig02, fig05
from cleancode_cover_flat import cover

MAKERS = [fig01, fig02, fig03, fig04, fig05]


def cover_record():
    """Render the native cover and preserve its reproducible source provenance."""
    record = export(cover(), "cover-flat")
    digest = hashlib.sha256((OUT / "cover-flat.png").read_bytes()).hexdigest()
    record.update({"kind": "code-native-flat", "sha256": digest,
                   "provenance": "cover-provenance.json"})
    provenance = {"tool": "matplotlib", "mode": "code-native-svg",
                  "asset": "cover-flat.png", "svg": "cover-flat.svg",
                  "renderer": "tools/cleancode_cover_flat.py",
                  "width": record["width"], "height": record["height"],
                  "sha256": digest}
    (OUT / "cover-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return record


def validate(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    labels = [(t, Text.get_window_extent(t, renderer=renderer))
              for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    errors, records = [], []
    css_width = 328 if fig._mobile or stem.startswith("cover") else 649
    for label, box in labels:
        if box.x0 < 8 or box.y0 < 8 or box.x1 > width-8 or box.y1 > height-8:
            errors.append(f"out of bounds: {label.get_text()}: {list(box.extents)}")
        css_font = label.get_fontsize()*100/72 * css_width/width
        if css_font < 16:
            errors.append(f"font below 16 CSS px: {label.get_text()}: {css_font:.2f}")
        records.append({"text": label.get_text(), "bbox_source_px": [round(float(v), 2) for v in box.extents],
                        "font_source_px": round(label.get_fontsize()*100/72, 2),
                        "font_css_px": round(css_font, 2)})
    for (a, aa), (b, bb) in itertools.combinations(labels, 2):
        if min(aa.x1, bb.x1)-max(aa.x0, bb.x0) > .5 and min(aa.y1, bb.y1)-max(aa.y0, bb.y0) > .5:
            errors.append(f"text collision: {a.get_text()} <> {b.get_text()}")
    if errors:
        raise AssertionError(stem + "\n" + "\n".join(errors))
    return {"review_width_css_px": css_width,
            "minimum_css_font_px": min(record["font_css_px"] for record in records),
            "labels": records, "text_collisions": 0, "text_out_of_bounds": 0}


def export(fig, stem):
    record = {"stem": stem, "width": fig._width, "height": fig._height,
              "layout": validate(fig, stem), "semantics": fig._semantics}
    fig.savefig(OUT / (stem+".svg"), metadata={"Date": None})
    fig.savefig(OUT / (stem+".png"), dpi=100, metadata={"Software": "Matplotlib"})
    svg = OUT / (stem+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    return record


def contact_sheet(mobile):
    images = []
    width = 328 if mobile else 649
    for name in NAMES:
        with Image.open(OUT / (name+("-mobile" if mobile else "")+".png")) as image:
            images.append(image.convert("RGB").resize((width, round(image.height*width/image.width)), Image.Resampling.LANCZOS))
    columns = 3 if mobile else 2
    rows = [images[i:i+columns] for i in range(0, len(images), columns)]
    row_heights = [max(image.height for image in row) for row in rows]
    sheet = Image.new("RGB", (columns*width+(columns-1)*20, sum(row_heights)+(len(rows)-1)*20), "#dce4e8")
    y = 0
    for row, height in zip(rows, row_heights):
        for j, image in enumerate(row):
            sheet.paste(image, (j*(width+20), y))
        y += height+20
    sheet.save(OUT / ("contact-sheet-mobile.png" if mobile else "contact-sheet-desktop.png"))


def main():
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    records = [export(maker(mobile), name+("-mobile" if mobile else ""))
               for mobile in (False, True) for maker, name in zip(MAKERS, NAMES)]
    records.append(cover_record())
    for mobile in (False, True):
        contact_sheet(mobile)
    manifest = {"generator": "tools/cleancode_figures.py", "assets": records,
                "domain_examples": domain_examples(),
                "design": {"svg_text": "outlined", "desktop_review_css_px": 649,
                           "mobile_review_css_px": 328, "body_minimum_source_font_px": 26,
                           "matrix": "native semantic HTML table in both articles",
                           "cover": "flat native SVG/PNG; actual text bounds audited at 328 CSS px",
                           "palette": {"ink": INK, "action": TEAL, "failure": RED, "refactor": GOLD}}}
    (OUT / "figure-data.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "passed", "assets": [{key: row[key] for key in ("stem", "width", "height")} for row in records],
                      "minimum_body_css_font_px": min(row["layout"]["minimum_css_font_px"] for row in records if "layout" in row),
                      "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(OUT.glob("*.svg"))}}, indent=2))


if __name__ == "__main__":
    main()
