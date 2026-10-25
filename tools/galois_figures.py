#!/usr/bin/env python3
"""Generate the Galois article's data-driven scientific figures and editorial cover.

Figure modules own geometry; article captions own explanations. SVG text is
outlined for portable rendering, with separate mobile compositions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from PIL import Image

from galois_visual_style import BLUE, TEAL, GOLD, RED, configure
from galois_visual_early import fig01, fig02, fig03, fig04
from galois_visual_late import fig05, fig06, fig08, fig09
from galois_visual_core import cover, fig07, fig10

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-10-25-galois-from-scratch"
NAMES = ["fig-01-formula", "fig-02-quadratic-field", "fig-03-permutations",
         "fig-04-correspondence", "fig-05-cubic", "fig-06-radical-tower",
         "fig-07-derived-series", "fig-08-a5", "fig-09-quintic", "fig-10-exercises"]
MAKERS = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10]


def validate(fig, stem):
    """Check actual rendered text bounds, not just the intended label positions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    labels = [(t, Text.get_window_extent(t, renderer=renderer))
              for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    errors = []
    for text, bounds in labels:
        if bounds.x0 < 4 or bounds.y0 < 4 or bounds.x1 > width-4 or bounds.y1 > height-4:
            errors.append(f"out of bounds: {text.get_text()}: {tuple(round(v,1) for v in bounds.extents)}")
    for (a, aa), (b, bb) in itertools.combinations(labels, 2):
        if min(aa.x1, bb.x1)-max(aa.x0, bb.x0) > 1 and min(aa.y1, bb.y1)-max(aa.y0, bb.y0) > 1:
            errors.append(f"overlap: {a.get_text()} <> {b.get_text()}")
    article_width = 328 if fig._mobile or stem == "cover-galois" else 691
    sizes = [t.get_fontsize()*100/72*article_width/width for t, _ in labels]
    if min(sizes) < 16:
        errors.append(f"body-scale type below 16px: {min(sizes):.2f}")
    if errors:
        raise AssertionError(stem + "\n" + "\n".join(errors))
    return {"labels": len(labels), "text_bounds": "passed", "text_collisions": 0,
            "minimum_css_font_px": round(min(sizes), 2),
            "review_width_css_px": article_width}


def mathematics():
    import sympy as sp
    x = sp.symbols("x")
    f = sp.Poly(x**5 - 5*x + 1, x)
    shifted = sp.Poly(f.as_expr().subs(x, x-1), x)
    assert shifted.all_coeffs() == [1, -5, 10, -10, 0, 5]
    assert all(int(c) % 5 == 0 for c in shifted.all_coeffs()[1:])
    assert int(shifted.TC()) % 25 != 0
    assert sp.diff(f.as_expr(), x) == 5*(x**4-1)
    assert f.eval(-1) == 5 and f.eval(1) == -3
    def compose(a, b): return tuple(a[b[i]] for i in range(5))
    g, h = (1, 0, 3, 2, 4), (1, 0, 4, 3, 2)
    commutator = compose(compose(compose(g, h), g), h)
    assert commutator == (0, 1, 3, 4, 2)
    return {"quintic": {"polynomial": "x^5-5x+1", "shifted_coefficients": [int(c) for c in shifted.all_coeffs()],
                         "eisenstein_prime": 5, "real_root_count": 3, "nonreal_conjugate_pairs": 1,
                         "numerical_roots": [[float(z.real), float(z.imag)] for z in np.roots([1, 0, 0, 0, -5, 1])]},
            "A5_commutator": {"g": "(12)(34)", "h": "(12)(35)", "convention": "rightmost first",
                               "commutator": "(345)", "image_list_zero_based": list(commutator)},
            "derived_orders": {"S3": [6, 3, 1], "S4": [24, 12, 4, 1], "S5": [120, 60, 60]},
            "quiz_figure_inputs": ["finite arithmetic and radical operations", "T(a+b*sqrt2)=a+2b*sqrt2",
                                    "exchange the sqrt2 and sqrt3 root pairs, fixing 2 and 3",
                                    "E=Q(sqrt3), sigma flips sqrt2 and fixes sqrt3",
                                    "derived orders S3:6,3,1 and S5:120,60,60"],
            "biquadratic_field": {"degree": 4, "group": "C2 x C2", "sigma": "flip sqrt2", "tau": "flip sqrt3",
                                    "fixed_fields": {"e": "L", "sigma": "Q(sqrt3)", "tau": "Q(sqrt2)",
                                                     "sigma*tau": "Q(sqrt6)", "G": "Q"}}}



def contact_sheet(mobile):
    width = 328 if mobile else 691
    images = []
    for name in NAMES:
        path = OUT / (name + ("-mobile" if mobile else "") + ".png")
        with Image.open(path) as source:
            images.append(source.convert("RGB").resize(
                (width, round(source.height*width/source.width)), Image.Resampling.LANCZOS))
    columns = 5 if mobile else 2
    heights = [max(image.height for image in images[i:i+columns])
               for i in range(0, len(images), columns)]
    result = Image.new("RGB", (width*columns+18*(columns-1),
                              sum(heights)+18*(len(heights)-1)), "#e6ecef")
    top = 0
    for row, height in enumerate(heights):
        for column, image in enumerate(images[row*columns:(row+1)*columns]):
            result.paste(image, (column*(width+18), top))
        top += height+18
    result.save(OUT / ("contact-sheet-mobile.png" if mobile else "contact-sheet-desktop.png"))


def export(fig, stem, cover_asset=False):
    layout = validate(fig, stem)
    svg_path = OUT / (stem + ".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    # Matplotlib leaves trailing blanks inside multiline SVG path attributes.
    # Newlines already separate those tokens; trim blanks for clean Git patches.
    svg_path.write_text("\n".join(line.rstrip(" \t") for line in
                                  svg_path.read_text().splitlines()) + "\n")
    dpi = 100 if cover_asset else 150
    fig.savefig(OUT / (stem + ".png"), dpi=dpi, metadata={"Software": "Matplotlib"})
    record = {"stem": stem, "source_width": fig._width, "source_height": fig._height,
              "png_width": round(fig.get_figwidth()*dpi),
              "png_height": round(fig.get_figheight()*dpi), "layout": layout}
    if hasattr(fig, "_visual_semantics"):
        record["semantics"] = fig._visual_semantics
    if hasattr(fig, "_cover_graph"):
        record["semantics"] = fig._cover_graph
    plt.close(fig)
    return record


def main():
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    records = [export(maker(mobile), name+("-mobile" if mobile else ""))
               for mobile in (False, True) for name, maker in zip(NAMES, MAKERS)]
    records.append(export(cover(), "cover-galois", cover_asset=True))
    for mobile in (False, True):
        contact_sheet(mobile)
    data = mathematics()
    manifest = {"generator": "tools/galois_figures.py", "assets": records,
                "mathematics": data,
                "design": {"version": 2, "svg_text": "outlined",
                           "desktop_article_width": 691, "mobile_article_width": 328,
                           "type": "data plots, coordinate transformations, group actions and Hasse diagrams",
                           "palette": {"first_family": BLUE, "action": TEAL,
                                       "second_family": GOLD, "obstruction": RED}},
                "scope": "Computed mathematical geometry. Root locations illustrate the exact algebra; graph shape is not a solvability proof."}
    (OUT / "figure-data.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.iterdir()) if p.is_file()}
    print(json.dumps({"status": "passed", "figures": 20, "cover": 1,
                      "assets": records, "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
