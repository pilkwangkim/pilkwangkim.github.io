#!/usr/bin/env python3
"""Analytic, accessible SVG/PNG figures for the paired Drude-model article.

Run ``python3 tools/drude_figures.py --png`` to rebuild all 84 assets.
Equations use local LaTeX/dvisvgm; analytic plot data use Python's standard
library. PNG output uses rsvg-convert. No external data or fonts.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from html import escape
from math import atan, cos, exp, log1p, pi, sin, sqrt
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-11-07-drude-model"
INK, MUTED = "#203649", "#607482"
BLUE, TEAL, GOLD = "#2b6090", "#087e81", "#a46e26"
GRID, PAPER, WHITE = "#dce5e8", "#fbfcfb", "#ffffff"
COLORS = (BLUE, TEAL, GOLD)
STEMS = ("fig-01-fields", "fig-02-drude", "fig-03-ac", "fig-04-band",
         "fig-05-matrix-element", "fig-06-scattering", "fig-07-charge")


def section_modules():
    # Support both direct CLI execution and importing tools.drude_figures in QA.
    if __package__:
        from . import drude_section_foundations, drude_section_quantum, drude_section_scattering
    else:
        import drude_section_foundations, drude_section_quantum, drude_section_scattering
    return (drude_section_foundations, drude_section_quantum, drude_section_scattering)


def section_figures():
    """One scientific figure per numbered section; retain all legacy image URLs."""
    result = {section: (renderer, stem, False) for section, renderer, stem in zip(
        (2, 5, 6, 9, 12, 16, 19),
        (fields, drude, ac, band, matrix_element, scattering, charge), STEMS)}
    for module in section_modules():
        for section, (renderer, stem) in module.FIGURES.items():
            assert section not in result, f"Duplicate section figure: {section}"
            result[section] = (renderer, stem, True)
    assert set(result) == set(range(1, 21)), "Every numbered section needs its own figure"
    assert len({stem for renderer, stem, external in result.values()}) == 20
    return dict(sorted(result.items()))


class Figure:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.nodes = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 L11 6 L1 11 Z" fill="context-stroke"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        ]

    def text(self, x, y, value, size=28, color=INK, anchor="start", weight=400):
        # Some scientific Unicode subscripts are absent in rasterizer fonts.
        # Render every sub/superscript with ordinary, universally present glyphs.
        sub = dict(zip("₀₁₂₃₄₅₆₇₈₉ₙₑₜᵣᵢⱼₓₚ", "0123456789netrijxp"))
        sup = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+"))
        label = ""
        for char in str(value):
            if char in sub or char in sup:
                shift, glyph = ("sub", sub[char]) if char in sub else ("super", sup[char])
                label += f'<tspan baseline-shift="{shift}" font-size="72%">{glyph}</tspan>'
            else:
                label += escape(char)
        label = re.sub(r"_([A-Za-z]+)", r'<tspan baseline-shift="sub" font-size="72%">\1</tspan>', label)
        self.nodes.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{label}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=2, arrow=False, dash=False):
        more = ' marker-end="url(#arrow)"' if arrow else ''
        more += ' stroke-dasharray="10 8"' if dash else ''
        self.nodes.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{more}/>')

    def path(self, points, color=BLUE, width=4.2, dash=False):
        data = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        more = ' stroke-dasharray="10 8"' if dash else ''
        self.nodes.append(f'<path d="{data}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{more}/>')

    def rect(self, x, y, width, height, fill=WHITE, stroke=GRID):
        self.nodes.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')

    def circle(self, x, y, radius=6, color=BLUE):
        self.nodes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def save(self, stem):
        OUT.mkdir(parents=True, exist_ok=True)
        data = "\n".join(self.nodes + ["</svg>"]) + "\n"
        ET.fromstring(data)
        (OUT / (stem + ".svg")).write_text(data, encoding="utf-8")
        print(f"{stem}.svg: {self.width}×{self.height}")


def axes(f, box, xlim, ylim, xticks, yticks, xlabel, ylabel, size=27):
    x, y, w, h = box
    X = lambda v: x + w * (v - xlim[0]) / (xlim[1] - xlim[0])
    Y = lambda v: y + h - h * (v - ylim[0]) / (ylim[1] - ylim[0])
    for tick in yticks:
        value, label = tick if isinstance(tick, tuple) else (tick, f"{tick:g}")
        f.line(x, Y(value), x + w, Y(value), GRID, 1.2)
        f.text(x - 13, Y(value) + size * .32, label, size, MUTED, "end")
    for tick in xticks:
        value, label = tick if isinstance(tick, tuple) else (tick, f"{tick:g}")
        f.line(X(value), y + h, X(value), y + h + 7, MUTED, 1.7)
        f.text(X(value), y + h + size + 13, label, size, MUTED, "middle")
    f.line(x, y, x, y + h, MUTED, 1.8)
    f.line(x, y + h, x + w, y + h, MUTED, 1.8)
    if ylim[0] < 0 < ylim[1]:
        f.line(x, Y(0), x + w, Y(0), MUTED, 1.3)
    f.text(x, y - 24, ylabel, size)
    f.text(x + w / 2, y + h + 2 * size + 28, xlabel, size, INK, "middle")
    return X, Y


def legend(f, labels, positions, y, size=26):
    for (label, color), x in zip(labels, positions):
        f.line(x, y - 9, x + 35, y - 9, color, 4.2)
        f.text(x + 48, y, label, size, color)


def header(f, desktop, mobile, compact):
    if compact:
        for index, line in enumerate(mobile):
            f.text(30, 48 + 42 * index, line, 31, weight=600)
    else:
        f.text(35, 53, desktop, 36, weight=600)


def points(start, stop, count=401):
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


@lru_cache(maxsize=None)
def equation_svg(formula):
    latex, dvisvgm = shutil.which("latex"), shutil.which("dvisvgm")
    if not latex or not dvisvgm:
        raise RuntimeError("Equation generation requires latex and dvisvgm.")
    tex = (r"\documentclass{article}\usepackage{amsmath,xcolor}"
           r"\definecolor{ink}{HTML}{203649}\pagestyle{empty}\begin{document}"
           r"\(\displaystyle\color{ink}" + formula + r"\)\end{document}")
    with tempfile.TemporaryDirectory(prefix="drude-equation-") as directory:
        path = Path(directory)
        (path / "eq.tex").write_text(tex, encoding="utf-8")
        for command in ([latex, "-interaction=nonstopmode", "-halt-on-error", "eq.tex"],
                        [dvisvgm, "--no-fonts", "--bbox=min", "--exact", "-o", "eq.svg", "eq.dvi"]):
            subprocess.run(command, cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return (path / "eq.svg").read_text(encoding="utf-8")


def math(f, key, formula, cx, top, height, max_width, *, font_size=None):
    svg = equation_svg(formula)
    svg = re.sub(r"(id=|(?:xlink:)?href=)(['\"])(#?)([^'\"]+)\2",
                 lambda match: match[1] + match[2] + match[3] + key + "-" + match[4] + match[2], svg)
    root = ET.fromstring(svg)
    _, _, natural_w, natural_h = map(float, root.attrib["viewBox"].split())
    # Hero equations use a target ink height. Scientific labels instead use a
    # fixed TeX font size: a fraction must be taller than a lone variable, not
    # have smaller glyphs merely because its bounding box is taller.
    scale = min(height / natural_h if font_size is None else font_size / 10,
                max_width / natural_w)
    w, h = natural_w * scale, natural_h * scale
    root.attrib.update(x=f"{cx - w / 2:.3f}", y=f"{top:.3f}", width=f"{w:.3f}",
                       height=f"{h:.3f}", preserveAspectRatio="xMidYMin meet")
    if font_size is not None:
        root.attrib.update({"data-latex": formula, "data-font-size": f"{10*scale:.3f}"})
    f.nodes.append(ET.tostring(root, encoding="unicode"))


def math_label(f, key, formula, cx, top, font_size=32, max_width=500):
    """Typeset a scientific label at a consistent 10pt-TeX-relative font size."""
    math(f, key, formula, cx, top, None, max_width, font_size=font_size)


def cover(compact=False):
    w, h = (960, 1320) if compact else (1600, 1000)
    f = Figure(w, h, "Drude model — electric drive, scattering and steady current",
               "DC conductivity sigma0=ne squared tau/m*. A single teal curve shows "
               "J(t)/(sigma0 E0)=1-exp(-t/tau): the current density parallel to a "
               "uniform positive field switched on at time zero, starting from J(0)=0. "
               "A gold point marks 1-exp(-1), about 63 percent, at t=tau. The dashed "
               "line at one is the steady current. Low-field isotropic parabolic-band "
               "Drude response with constant carrier density, positive effective mass "
               "and transport relaxation time; this is not an individual electron trajectory.")
    margin = 62 if compact else 80
    f.line(margin, 70 if compact else 84, margin+70, 70 if compact else 84, TEAL, 6)
    f.text(margin+96, 80 if compact else 94, "ELECTRON TRANSPORT", 26, MUTED, weight=500)
    f.text(margin, 207 if compact else 220, "Drude model", 104 if compact else 112, weight=600)
    if compact:
        f.text(margin, 283, "From electric fields", 45, MUTED)
        f.text(margin, 339, "to steady current", 45, MUTED)
        math(f, "dc", r"\sigma_0=\frac{ne^2\tau}{m^{\ast}}", w/2, 401, 150, 810)
        f.line(margin, 595, w-margin, 595)
        f.text(margin, 661, "Current after switch-on", 40, weight=500)
        box = (138, 770, 712, 285)
        math(f, "current-axis", r"J(t)/(\sigma_0E_0)", 288, 700, 44, 360)
        f.text(850, 748, "steady current", 35, MUTED, "end")
        footer = 1190
    else:
        f.text(margin, 288, "From electric fields to steady current", 44, MUTED)
        f.text(margin, 400, "DC conductivity", 35, weight=500)
        math(f, "dc", r"\sigma_0=\frac{ne^2\tau}{m^{\ast}}", 388, 445, 178, 615)
        f.text(margin, 703, "Carrier density · effective mass", 29, MUTED)
        f.text(margin, 747, "Transport relaxation time", 29, MUTED)
        f.line(731, 365, 731, 795, GRID, 1.5)
        f.text(847, 388, "Current after switch-on", 34, weight=500)
        box = (850, 499, 650, 274)
        math(f, "current-axis", r"J(t)/(\sigma_0E_0)", 995, 433, 39, 295)
        f.text(1500, 476, "steady current", 28, MUTED, "end")
        footer = 901

    # This is the analytic ensemble-current response, not a particle path.
    # Both variants use exactly the same dimensionless axes and data.
    x, y, plot_w, plot_h = box
    tick_size = 37 if compact else 29
    X = lambda s: x+plot_w*s/4
    Y = lambda j: y+plot_h*(1-j/1.1)
    f.line(x, Y(1), x+plot_w, Y(1), MUTED, 1.8, dash=True)
    f.line(x, y, x, y+plot_h, MUTED, 2)
    f.line(x, y+plot_h, x+plot_w, y+plot_h, MUTED, 2)
    for j in (0, 1):
        f.text(x-20, Y(j)+12, str(j), tick_size, MUTED, "end")
    for s in range(5):
        f.line(X(s), y+plot_h, X(s), y+plot_h+8, MUTED, 1.8)
        f.text(X(s), y+plot_h+45, str(s), tick_size, MUTED, "middle")
    f.path([(X(s), Y(1-exp(-s))) for s in points(0, 4)], TEAL, 6.5)
    f.line(X(1), Y(0), X(1), Y(1-exp(-1)), GOLD, 1.8, dash=True)
    f.circle(X(1), Y(1-exp(-1)), 7.5, GOLD)
    f.text(X(1)+30, Y(.32), "63% at t = τ", 36 if compact else 29, GOLD)
    math(f, "time-axis", r"t/\tau", x+plot_w/2, y+plot_h+65, 45 if compact else 39, 180)
    f.line(margin, footer, w-margin, footer, GRID, 1.5)
    if compact:
        f.text(margin, footer+53, "Low field · isotropic parabolic band", 32, MUTED)
        f.text(margin, footer+100, "Constant n, m* and τ · E₀ > 0 · J(0) = 0", 32, MUTED)
    else:
        f.text(margin, footer+53, "Low field · isotropic parabolic band · constant n, m* and τ", 27, MUTED)
        f.text(w-margin, footer+53, "E₀ > 0 · J(0) = 0", 27, MUTED, "end")
    f.save("cover-mobile" if compact else "cover")


def fields(compact=False):
    f = Figure(600 if compact else 1200, 1490 if compact else 910,
               "A charge-neutral bulk with a boundary-imposed electric field",
               "A one-dimensional boundary-value problem in a homogeneous conductor. "
               "Panel a is a continuum bulk region between prescribed boundaries phi(0)=V "
               "and phi(L)=0, with V positive and constant permittivity. There are no "
               "particle dots or particle trajectories. Its only interior arrow is the "
               "electric field, in the positive x direction. Carrier and background "
               "densities are equal: n=N_D, rho=e(N_D-n)=0. Panel b shows phi/V=1-x/L. "
               "Panel c shows E_x/(V/L)=1 on the same spatial interval. The field is "
               "minus the potential slope; zero field divergence does not imply zero field. "
               "No conductivity or current-voltage law is assumed in this electrostatic figure.")
    header(f, "Boundary voltage and the electric field",
           ("Charge-neutral bulk", "with a finite electric field"), compact)
    if compact:
        left, right, bar_y, bar_h = 110, 520, 360, 100
        math_label(f, "bulk", r"n=N_D,\qquad\rho=e(N_D-n)=0", 300, 127, 32, 540)
        math_label(f, "epsilon", r"V>0,\qquad\varepsilon=\mathrm{constant}", 300, 191, 30, 540)
        f.text(30, 271, "(a) Prescribed boundary potentials", 28, weight=500)
        phi_y, ruler_y = 305, 493
        label_left, label_right = 110, 500
        potential_box, field_box = (110, 733, 410, 180), (110, 1217, 410, 130)
        heads = ((30, 589), (30, 1092))
        formula_top = (626, 1127)
    else:
        left, right, bar_y, bar_h = 190, 1080, 270, 80
        math_label(f, "bulk", r"n=N_D,\qquad\rho=e(N_D-n)=0,\qquad\varepsilon=\mathrm{constant}", 600, 96, 32, 1110)
        f.text(45, 185, "(a) Prescribed boundary potentials", 29, weight=500)
        phi_y, ruler_y = 212, 388
        label_left, label_right = left, right-15
        potential_box, field_box = (190, 517, 890, 110), (190, 718, 890, 90)
        heads = ((45, 480), (45, 678))
        formula_top = (446, 645)

    # End faces are mathematical potential boundaries. The empty interior is a
    # homogeneous continuum, not a row of fixed electron/ion locations.
    f.rect(left, bar_y, right-left, bar_h, "#eef5f6", BLUE)
    f.rect(left-9, bar_y-6, 18, bar_h+12, "#b9cddc", BLUE)
    f.rect(right-9, bar_y-6, 18, bar_h+12, "#b9cddc", BLUE)
    math_label(f, "left-boundary", r"\varphi(0)=V>0", label_left, phi_y, 32, 208 if not compact else 183)
    math_label(f, "right-boundary", r"\varphi(L)=0", label_right, phi_y, 32, 180)
    mid = (left+right)/2
    math_label(f, "field-arrow", r"E_x=V/L>0", mid, bar_y+15, 32, right-left-55)
    f.line(mid-100, bar_y+bar_h-22, mid+100, bar_y+bar_h-22, TEAL, 3.3, arrow=True)
    f.line(left, ruler_y, right+24, ruler_y, MUTED, 1.6, arrow=True)
    for x in (left, right):
        f.line(x, ruler_y-7, x, ruler_y+7, MUTED, 1.5)
    math_label(f, "x-zero", "0", left, ruler_y+17, 29, 45)
    math_label(f, "x-length", "L", right, ruler_y+17, 29, 45)
    math_label(f, "x-direction", "x", right+42, ruler_y-7, 30, 35)

    f.text(*heads[0], "(b) Potential profile", 29, weight=500)
    math_label(f, "potential-law", r"\varphi(x)=V\!\left(1-\frac{x}{L}\right)",
               315 if compact else 795, formula_top[0], 32, 500)
    X, Y = axes(f, potential_box, (0, 1), (0, 1.1), (0, .5, 1) if compact else (), (0, 1), "", "")
    f.path([(X(0), Y(1)), (X(1), Y(0))], TEAL)
    math_label(f, "potential-axis", r"\varphi/V", 51 if compact else 100,
               potential_box[1]+potential_box[3]*.43, 30, 92 if compact else 140)
    if compact:
        math_label(f, "potential-position", r"x/L", 315, potential_box[1]+potential_box[3]+73, 30, 110)
    f.text(*heads[1], "(c) Electric-field profile", 29, weight=500)
    math_label(f, "field-law", r"E_x=-\frac{d\varphi}{dx}=\frac{V}{L}",
               315 if compact else 795, formula_top[1], 32, 500)
    X, Y = axes(f, field_box, (0, 1), (0, 1.2), (0, .5, 1), (0, 1), "", "")
    f.path([(X(0), Y(1)), (X(1), Y(1))], BLUE)
    math_label(f, "field-axis", r"\frac{E_x}{V/L}", 48 if compact else 100,
               field_box[1]+field_box[3]*.32, 30, 90 if compact else 155)
    math_label(f, "field-position", r"x/L", 315 if compact else 635,
               field_box[1]+field_box[3]+67 if compact else 863, 30, 110)
    f.save("fig-01-fields" + ("-mobile" if compact else ""))


def drude(compact=False):
    f = Figure(600 if compact else 1200, 1450 if compact else 830,
               "Drude current approaches a steady value and relaxes after switch-off",
               "A field E0 switched on at time zero gives J/(sigma0 E0)=1-exp(-t/tau) from zero initial drift. "
               "A different switch-off experiment starts at the steady current and gives exp(-t/tau). "
               "After switch-off the field is zero; E0 is the field before switch-off. "
               "In both panels time zero is that panel's switching event, E0 is positive and n, mass and transport time are constant.")
    header(f, "Current builds up and relaxes on the transport time",
           ("Current build-up and relaxation",), compact)
    math_label(f, "drude-normalization", r"j(t)\equiv J(t)/(\sigma_0E_0),\qquad E_0>0",
         f.width/2, 95, 32, f.width-70)
    if compact:
        panels = ((95, 350, 445, 280), (95, 950, 445, 280))
        heads = ((30, 183), (30, 783))
    else:
        panels = ((100, 350, 445, 260), (700, 350, 445, 260))
        heads = ((100, 185), (700, 185))
    for index, ((x, y), box) in enumerate(zip(heads, panels)):
        cx = box[0]+box[2]/2
        f.text(x, y, "(a) Switch on: zero initial drift" if index == 0 else "(b) Switch off: initial steady current", 27, weight=500)
        protocol = (r"E:0\longrightarrow E_0,\quad J(0)=0" if index == 0 else
                    r"E:E_0\longrightarrow0,\quad J(0)=\sigma_0E_0")
        response = (r"j(t)=1-e^{-t/\tau}" if index == 0 else
                    r"j(t)=e^{-t/\tau}")
        math_label(f, f"drude-protocol-{index}", protocol, cx, y+32, 31, 445)
        math_label(f, f"drude-response-{index}", response, cx, y+92, 32, 445)
        math_label(f, f"drude-current-axis-{index}", r"j(t)", box[0]+28, box[1]-31, 29, 95)
        X, Y = axes(f, box, (0, 5), (0, 1.05), (0, 1, 3, 5), (0, .5, 1), "", "")
        fun = (lambda s: 1-exp(-s)) if index == 0 else (lambda s: exp(-s))
        color = TEAL if index == 0 else GOLD
        f.line(X(1), Y(0), X(1), Y(fun(1)), MUTED, 1.7, dash=True)
        f.path([(X(s), Y(fun(s))) for s in points(0, 5)], color)
        f.circle(X(1), Y(fun(1)), 6, color)
        f.text(X(1.25), Y(.30 if index == 0 else .70), "63% at one τ" if index == 0 else "37% at one τ", 25, color)
        math_label(f, f"drude-time-axis-{index}", r"t/\tau", cx, box[1]+box[3]+65, 31, 150)
    base = 1345 if compact else 741
    f.line(35, base, f.width-35, base)
    if compact:
        f.text(35, base+44, "Time starts at each switching event.", 26, MUTED)
        f.text(35, base+87, "In (b), E₀ is the field before switch-off.", 25, MUTED)
    else:
        f.text(35, base+49, "Time starts at each switch; in (b), E₀ is the field before switch-off.", 27, MUTED)
    f.save("fig-02-drude" + ("-mobile" if compact else ""))


def ac(compact=False):
    f = Figure(600 if compact else 1200, 1800 if compact else 1180,
               "Complex Drude conductivity with the exp(-i omega t) convention",
               "Real conductivity is 1/(1+x squared), imaginary conductivity is x/(1+x squared), x=omega tau. "
               "The current lags the real field by arctan(x). At x=1 the normalized field is cos(omega t), "
               "and normalized current is cos(omega t-pi/4)/sqrt(2). Positive imaginary conductivity therefore describes a lag for exp(-i omega t).")
    header(f, "AC conductivity sets the current amplitude and phase",
           ("AC conductivity and phase lag",), compact)
    if compact:
        math_label(f, "ac-convention", r"E(t)=\operatorname{Re}[E_\omega e^{-i\omega t}]", 300, 96, 32, 540)
        math_label(f, "ac-response", r"\frac{\sigma(\omega)}{\sigma_0}=\frac{1}{1-i\omega\tau}", 300, 150, 31, 540)
    else:
        math_label(f, "ac-definition", r"E(t)=\operatorname{Re}[E_\omega e^{-i\omega t}],\qquad\frac{\sigma(\omega)}{\sigma_0}=\frac{1}{1-i\omega\tau}", 600, 94, 31, 1100)
    if compact:
        boxes = ((95, 335, 445, 260), (95, 895, 445, 240), (95, 1410, 445, 240))
        hs = ((30, 252), (30, 802), (30, 1310))
    else:
        boxes = ((100, 275, 445, 245), (700, 275, 445, 245), (100, 805, 1045, 220))
        hs = ((100, 196), (700, 196), (100, 715))
    f.text(*hs[0], "(a) Conductivity spectrum", 28, weight=500)
    math_label(f, "ac-spectrum-axis", r"\sigma/\sigma_0", boxes[0][0]+72, boxes[0][1]-47, 31, 180)
    X, Y = axes(f, boxes[0], (0, 5), (0, 1.05), (0, 1, 3, 5), (0, .5, 1), "", "")
    for fun, color in ((lambda x: 1/(1+x*x), BLUE), (lambda x: x/(1+x*x), TEAL)):
        f.path([(X(x), Y(fun(x))) for x in points(0, 5)], color)
    math_label(f, "ac-spectrum-frequency", r"\omega\tau", boxes[0][0]+boxes[0][2]/2, boxes[0][1]+boxes[0][3]+65, 32, 160)
    ly = 733 if compact else 650
    for index, (color, formula) in enumerate(((BLUE, r"\operatorname{Re}\sigma"), (TEAL, r"\operatorname{Im}\sigma"))):
        lx = (95 if compact else 100)+225*index
        f.line(lx, ly-9, lx+35, ly-9, color, 4.2)
        math_label(f, f"ac-spectrum-legend-{index}", formula, lx+105, ly-22, 31, 130)
    f.text(*hs[1], "(b) Current phase lag", 28, weight=500)
    math_label(f, "ac-lag-definition", r"\theta=\tan^{-1}(\omega\tau)\quad[{}^{\circ}]", boxes[1][0]+boxes[1][2]/2, boxes[1][1]-51, 30, 445)
    X, Y = axes(f, boxes[1], (0, 5), (0, 90), (0, 1, 3, 5), (0, 45, 90), "", "")
    f.path([(X(x), Y(atan(x)*180/pi)) for x in points(0, 5)], GOLD)
    f.circle(X(1), Y(45), 6, GOLD)
    f.text(X(1.35), Y(29), "45° at ωτ = 1", 25, GOLD)
    math_label(f, "ac-lag-frequency", r"\omega\tau", boxes[1][0]+boxes[1][2]/2, boxes[1][1]+boxes[1][3]+65, 32, 160)
    f.text(*hs[2], "(c) Field and current at ωτ = 1", 28, weight=500)
    f.text(boxes[2][0], boxes[2][1]-35, "Normalized field and current", 26, MUTED)
    X, Y = axes(f, boxes[2], (0, 2*pi), (-1.12, 1.4), ((0, "0"), (pi, "π"), (2*pi, "2π")), (-1, 0, 1), "", "")
    f.path([(X(q), Y(cos(q))) for q in points(0, 2*pi)], BLUE)
    f.path([(X(q), Y(cos(q-pi/4)/sqrt(2))) for q in points(0, 2*pi)], GOLD)
    # The two dots are maxima of the plotted waveforms, not added particles.
    # Their horizontal separation is the phase lag for this common frequency.
    f.circle(X(0), Y(1), 5, BLUE)
    f.circle(X(pi/4), Y(1/sqrt(2)), 5, GOLD)
    f.line(X(0), Y(1), X(0), Y(1.2), BLUE, 1.5, dash=True)
    f.line(X(pi/4), Y(1/sqrt(2)), X(pi/4), Y(1.2), GOLD, 1.5, dash=True)
    f.line(X(0), Y(1.2), X(pi/4), Y(1.2), GOLD, 2, arrow=True)
    math_label(f, "ac-waveform-lag", r"\pi/4", (X(0)+X(pi/4))/2, Y(1.2)-34, 27, 85)
    math_label(f, "ac-waveform-phase", r"\omega t", boxes[2][0]+boxes[2][2]/2, boxes[2][1]+boxes[2][3]+65, 32, 160)
    ly = 1770 if compact else 1148
    for index, (color, formula) in enumerate(((BLUE, r"E/E_0"), (GOLD, r"J/(\sigma_0E_0)"))):
        lx = (95+225*index) if compact else (390+310*index)
        f.line(lx, ly-9, lx+35, ly-9, color, 4.2)
        math_label(f, f"ac-waveform-legend-{index}", formula, lx+(110 if index==0 else 132), ly-27, 31, 180)
    f.save("fig-03-ac" + ("-mobile" if compact else ""))


def band(compact=False):
    f = Figure(600 if compact else 1200, 1930 if compact else 1180,
               "A tight-binding band has wavevector-dependent curvature mass",
               "For epsilon=2t_h[1-cos(ka)], the normalized energy is 1-cos(ka), group velocity divided by 2t_h a/hbar is sin(ka), "
               "and m_edge/m_curv=cos(ka), where m_edge=hbar squared/(2t_h a squared). "
               "A near-bottom parabola has normalized energy (ka) squared/2. "
               "Inverse curvature mass changes sign at plus or minus pi/2. No divergent mass pole is plotted.")
    header(f, "Tight-binding dispersion, velocity and curvature",
           ("Tight-binding dispersion,", "velocity and curvature"), compact)
    math_label(f, "band-definition", r"\epsilon(k)=2t_h[1-\cos(ka)],\qquad t_h>0", f.width/2,
         142 if compact else 96, 32, f.width-70)
    ticks = ((-pi, "−π"), (-pi/2, "−π/2"), (0, "0"), (pi/2, "π/2"), (pi, "π"))
    if compact:
        boxes = ((95, 330, 445, 245), (95, 935, 445, 240), (95, 1505, 445, 240))
        hs = ((30, 230), (30, 780), (30, 1350))
    else:
        boxes = ((100, 285, 1045, 210), (100, 865, 445, 170), (700, 865, 445, 170))
        hs = ((100, 200), (100, 716), (700, 716))
    f.text(*hs[0], "(a) Band and edge approximation", 28, weight=500)
    math_label(f, "band-energy-axis", r"\epsilon/(2t_h)", boxes[0][0]+90, boxes[0][1]-47, 31, 220)
    X, Y = axes(f, boxes[0], (-pi, pi), (0, 2.1), ticks, (0, 1, 2), "", "")
    f.path([(X(q), Y(1-cos(q))) for q in points(-pi, pi)], BLUE)
    f.path([(X(q), Y(q*q/2)) for q in points(-1.35, 1.35)], GOLD, dash=True)
    math_label(f, "band-energy-wavevector", r"ka", boxes[0][0]+boxes[0][2]/2, boxes[0][1]+boxes[0][3]+65, 32, 180)
    ly = 715 if compact else 631
    for index, (color, formula) in enumerate(((BLUE, r"1-\cos(ka)"), (GOLD, r"(ka)^2/2"))):
        lx = (35+285*index) if compact else (285+400*index)
        f.line(lx, ly-9, lx+35, ly-9, color, 4.2, dash=index==1)
        math_label(f, f"band-energy-legend-{index}", formula, lx+140, ly-26, 31, 205)
    f.text(*hs[1], "(b) Group velocity", 28, weight=500)
    math_label(f, "band-velocity-definition", r"v=\frac{1}{\hbar}\frac{d\epsilon}{dk}", boxes[1][0]+boxes[1][2]/2, hs[1][1]+25, 31, 420)
    math_label(f, "band-velocity-axis", r"v/v_0", boxes[1][0]+60, boxes[1][1]-38, 31, 190)
    X, Y = axes(f, boxes[1], (-pi, pi), (-1.1, 1.1), ticks, (-1, 0, 1), "", "")
    f.path([(X(q), Y(sin(q))) for q in points(-pi, pi)], TEAL)
    math_label(f, "band-velocity-wavevector", r"ka", boxes[1][0]+boxes[1][2]/2, boxes[1][1]+boxes[1][3]+65, 32, 180)
    f.text(*hs[2], "(c) Inverse curvature mass", 28, weight=500)
    math_label(f, "band-mass-definition", r"m_{\rm curv}^{-1}=\frac{1}{\hbar^2}\frac{d^2\epsilon}{dk^2}", boxes[2][0]+boxes[2][2]/2, hs[2][1]+25, 31, 430)
    math_label(f, "band-mass-axis", r"m_{\rm edge}/m_{\rm curv}", boxes[2][0]+130, boxes[2][1]-38, 30, 300)
    X, Y = axes(f, boxes[2], (-pi, pi), (-1.1, 1.1), ticks, (-1, 0, 1), "", "")
    f.path([(X(q), Y(cos(q))) for q in points(-pi, pi)], GOLD)
    for q in (-pi/2, pi/2):
        f.line(X(q), Y(-1.1), X(q), Y(0), GRID, 1.5, dash=True)
        f.circle(X(q), Y(0), 5, GOLD)
    math_label(f, "band-zero-curvature", r"ka=\pm\pi/2", boxes[2][0]+boxes[2][2]/2, Y(-.52), 27, 210)
    math_label(f, "band-mass-wavevector", r"ka", boxes[2][0]+boxes[2][2]/2, boxes[2][1]+boxes[2][3]+65, 32, 180)
    if compact:
        math_label(f, "band-scales", r"v_0=2t_h a/\hbar,\qquad m_{\rm edge}=\hbar^2/(2t_h a^2)", 300, 1871, 29, 540)
    else:
        math_label(f, "band-scales", r"v_0=2t_h a/\hbar,\qquad m_{\rm edge}=\hbar^2/(2t_h a^2)", 600, 1140, 29, 1100)
    f.save("fig-04-band" + ("-mobile" if compact else ""))


def matrix_element(compact=False):
    f = Figure(600 if compact else 1200, 1510 if compact else 900,
               "A wider Gaussian potential suppresses large wavevector transfer",
               "A one-dimensional Gaussian perturbation has fixed integrated strength g. "
               "Widths are ell0 and twice ell0. Real-space U ell0/g=(ell0/ell)/sqrt(2pi) exp[-x squared/(2ell squared)]. "
               "The Fourier intensity |U tilde/g| squared=exp[-q squared ell squared]. "
               "The transfer is q=k prime-k; physical momentum transfer is hbar q. "
               "The box-normalized one-dimensional plane-wave matrix element is M=U tilde(q)/mathcal L. "
               "These one-dimensional quantities are not a three-dimensional differential cross section.")
    header(f, "A wider potential suppresses large wavevector transfer",
           ("Potential width controls", "wavevector transfer"), compact)
    if compact:
        math_label(f, "gaussian-strength", r"\textstyle g=\int U_\ell(x)\,dx\quad\text{fixed}", 300, 132, 30, 535)
        boxes = ((95, 384, 445, 250), (95, 1006, 445, 250))
        hs = ((30, 217), (30, 850))
        formula_tops, label_tops = (250, 886), (338, 960)
        legend_tops = (746, 1368)
    else:
        f.text(35, 111, "1D Gaussian perturbations with equal integrated strength", 27, MUTED)
        math_label(f, "gaussian-strength", r"\textstyle g=\int U_\ell(x)\,dx", 994, 83, 30, 340)
        boxes = ((100, 350, 445, 275), (700, 350, 445, 275))
        hs = ((100, 184), (700, 184))
        formula_tops, label_tops = (219, 226), (305, 305)
        legend_tops = (741, 741)
    f.text(*hs[0], "(a) Potential in real space", 28, weight=500)
    math_label(f, "gaussian-model", r"U_\ell(x)=\frac{g}{\sqrt{2\pi}\ell}\exp\!\left[-\frac{x^2}{2\ell^2}\right]",
         boxes[0][0]+boxes[0][2]/2, formula_tops[0], 30, 445 if not compact else 535)
    math_label(f, "potential-axis", r"U_\ell(x)\ell_0/g", boxes[0][0]+113, label_tops[0], 30, 226)
    X, Y = axes(f, boxes[0], (-6, 6), (0, .43), (-6, -3, 0, 3, 6), (0, .2, .4), "", "")
    for ell, color in ((1, BLUE), (2, GOLD)):
        f.path([(X(x), Y(exp(-x*x/(2*ell*ell))/(sqrt(2*pi)*ell))) for x in points(-6, 6)], color)
    math_label(f, "position-axis", r"x/\ell_0", boxes[0][0]+boxes[0][2]/2, boxes[0][1]+boxes[0][3]+66, 30, 180)
    f.text(*hs[1], "(b) Fourier power", 28, weight=500)
    math_label(f, "fourier-model", r"\left|\widetilde U_\ell(q)/g\right|^2=e^{-q^2\ell^2}",
         boxes[1][0]+boxes[1][2]/2, formula_tops[1], 30, 445 if not compact else 535)
    math_label(f, "fourier-axis", r"|\widetilde U_\ell(q)/g|^2", boxes[1][0]+121, label_tops[1], 30, 242)
    X, Y = axes(f, boxes[1], (0, 4), (0, 1.05), (0, 1, 2, 3, 4), (0, .5, 1), "", "")
    for ell, color in ((1, BLUE), (2, GOLD)):
        f.path([(X(q), Y(exp(-q*q*ell*ell))) for q in points(0, 4)], color)
    math_label(f, "transfer-axis", r"q\ell_0\quad(q=k^{\prime}-k)", boxes[1][0]+boxes[1][2]/2,
         boxes[1][1]+boxes[1][3]+66, 30, 410)
    for index, (box, top) in enumerate(zip(boxes, legend_tops)):
        for j, (shift, color) in enumerate(((0, BLUE), (225, GOLD))):
            f.line(box[0]+shift, top+15, box[0]+shift+35, top+15, color, 4.2)
            math_label(f, f"width-{index}-{j}", r"\ell=\ell_0" if j == 0 else r"\ell=2\ell_0",
                 box[0]+shift+108, top, 30, 140)
    if compact:
        f.line(30, 1421, 570, 1421)
        math_label(f, "box-matrix", r"M_{k^{\prime}k}=\widetilde U_\ell(k^{\prime}-k)/\mathcal L",
             300, 1438, 30, 530)
    else:
        f.line(35, 799, 1165, 799)
        f.text(35, 855, "Box-normalized plane waves", 27, MUTED)
        math_label(f, "box-matrix", r"M_{k^{\prime}k}=\widetilde U_\ell(k^{\prime}-k)/\mathcal L",
             852, 822, 30, 595)
    f.save("fig-05-matrix-element" + ("-mobile" if compact else ""))


def angular_probability(theta, x):
    return (1+x)/2 * sin(theta) / (1+x*sin(theta/2)**2)**2


def mean_loss(x):
    return 1.0 if x == 0 else 2*(1+x)/(x*x)*(log1p(x)-x/(1+x))


def scattering(compact=False):
    f = Figure(600 if compact else 1200, 1600 if compact else 960,
               "Forward scattering changes direction less per event",
               "Three normalized 3D angular probability densities p_theta=(1+x_s) sin(theta)/(2[1+x_s sin squared(theta/2)] squared), "
               "for x_s=0,4,100 have integral one and equal total event rate. "
               "Transport weights each density by one minus cosine theta. "
               "The transport-to-event time ratio is the inverse mean angular loss, respectively 1, about 1.98, and about 13.66. "
               "This comparison fixes total rate, not microscopic potential strength.")
    header(f, "Forward scattering relaxes less momentum per event",
           ("Forward scattering relaxes", "less momentum per event"), compact)
    f.text(30 if compact else 35, 151 if compact else 110,
           "3D angles · equal total event rate" if compact else "3D angular distributions · equal total event rate",
           25 if compact else 27, MUTED)
    if compact:
        boxes = ((95, 384, 445, 266), (95, 987, 445, 266))
        hs = ((30, 220), (30, 873))
        formula_tops = (252, 920)
        label_top = 340
        legend_tops = (767, 1370)
    else:
        boxes = ((100, 350, 445, 280), (700, 350, 445, 280))
        hs = ((100, 186), (700, 186))
        formula_tops = (219, 263)
        label_top = 307
        legend_tops = (741, 741)
    f.text(*hs[0], "(a) Polar-angle probability", 28, weight=500)
    math_label(f, "angle-model", r"p(\theta)=\frac{(1+x_s)\sin\theta}{2[1+x_s\sin^2(\theta/2)]^2}",
         boxes[0][0]+boxes[0][2]/2, formula_tops[0], 30, 535 if compact else 445)
    math_label(f, "probability-axis", r"p(\theta)\;[\mathrm{rad}^{-1}]", boxes[0][0]+117, label_top, 30, 234)
    ticks = ((0, "0"), (pi/2, "π/2"), (pi, "π"))
    X, Y = axes(f, boxes[0], (0, pi), (0, 3.5), ticks, (0, 1, 2, 3), "", "")
    for x, color in zip((0, 4, 100), COLORS):
        f.path([(X(th), Y(angular_probability(th, x))) for th in points(0, pi, 1001)], color)
    f.text(*hs[1], "(b) Transport-rate weight", 28, weight=500)
    math_label(f, "transport-axis", r"p(\theta)(1-\cos\theta)", boxes[1][0]+boxes[1][2]/2,
         formula_tops[1], 30, 435)
    if not compact:
        f.text(boxes[1][0], 238, "Contribution per polar angle", 25, MUTED)
    X, Y = axes(f, boxes[1], (0, pi), (0, .7), ticks, (0, .3, .6), "", "")
    for x, color in zip((0, 4, 100), COLORS):
        f.path([(X(th), Y(angular_probability(th, x)*(1-cos(th)))) for th in points(0, pi, 1001)], color)
    for index, (box, top) in enumerate(zip(boxes, legend_tops)):
        math_label(f, f"angle-x-{index}", r"\theta\quad\text{(deflection angle)}", box[0]+box[2]/2,
             box[1]+box[3]+66, 30, 420)
        for j, (shape, color) in enumerate(zip((0, 4, 100), COLORS)):
            xx = (40 if compact else box[0]-30)+j*165
            f.line(xx, top+15, xx+30, top+15, color, 4.2)
            math_label(f, f"shape-{index}-{j}", rf"x_s={shape}", xx+102, top, 30, 112)
    if compact:
        f.line(30, 1421, 570, 1421)
        math_label(f, "lifetime-ratio", r"\frac{\tau_{\rm tr}}{\tau_q}=\frac{1}{\langle1-\cos\theta\rangle_p}",
             300, 1429, 30, 520)
        for index, (x, color) in enumerate(zip((0, 4, 100), COLORS)):
            math_label(f, f"ratio-shape-{index}", rf"x_s={x}", 120+index*180, 1520, 30, 145)
            f.text(120+index*180, 1582, f"{1/mean_loss(x):.2f}", 31, color, "middle", weight=500)
    else:
        f.line(35, 800, 1165, 800)
        math_label(f, "lifetime-ratio", r"\frac{\tau_{\rm tr}}{\tau_q}=\frac{1}{\langle1-\cos\theta\rangle_p}",
             278, 822, 30, 485)
        for index, (x, color) in enumerate(zip((0, 4, 100), COLORS)):
            math_label(f, f"ratio-shape-{index}", rf"x_s={x}", 680+index*185, 830, 30, 160)
            f.text(680+index*185, 913, f"{1/mean_loss(x):.2f}", 35, color, "middle", weight=500)
        f.text(35, 927, "Shape comparison at fixed total rate", 25, MUTED)
    f.save("fig-06-scattering" + ("-mobile" if compact else ""))


def charge_amplitude(s, omega):
    if omega == .5:
        return exp(-s/2)*(1+s/2), -.25*s*exp(-s/2)
    if omega < .5:
        d = sqrt(1-4*omega*omega)
        r1, r2 = (-1+d)/2, (-1-d)/2
        a = (-r2*exp(r1*s)+r1*exp(r2*s))/(r1-r2)
        ap = r1*r2*(-exp(r1*s)+exp(r2*s))/(r1-r2)
    else:
        beta = sqrt(omega*omega-.25)
        a = exp(-s/2)*(cos(beta*s)+sin(beta*s)/(2*beta))
        ap = -omega*omega/beta*exp(-s/2)*sin(beta*s)
    return a, ap


def charge(compact=False):
    a, ap = charge_amplitude(1, 1.2)
    f = Figure(600 if compact else 1200, 1820 if compact else 1285,
               "A self-consistent periodic charge mode can decay or oscillate",
               "A periodic electron-density perturbation delta n=A0 a(s) cos(qx) obeys a double prime+a prime+Omega squared a=0, "
               "s=t/tau, a(0)=1, a prime(0)=0. Curves use Omega=0.2 and 1.2. "
               f"The spatial snapshot uses Omega=1.2 and s=1, a={a:.8f}, a prime={ap:.8f}. "
               "Normalized density equals a cos(qx), normalized electric field equals minus a sin(qx), "
               "and normalized conventional current equals a prime sin(qx). "
               "This is a periodic collective electron-fluid density mode, not a quantum wavefunction or an electrode-biased uniform wire.")
    header(f, "A periodic density perturbation relaxes or oscillates",
           ("Periodic density relaxation",), compact)
    math_label(f, "density-ode", r"a^{\prime\prime}+a^{\prime}+\Omega^2a=0",
         300 if compact else 260, 87, 30, 535 if compact else 450)
    math_label(f, "density-initial", r"s=t/\tau,\quad a(0)=1,\quad a^{\prime}(0)=0",
         300 if compact else 318, 143 if compact else 140, 30, 535 if compact else 565)
    if not compact:
        math_label(f, "plasma-frequency", r"\Omega=\omega_p\tau,\quad\omega_p^2=\frac{n_0e^2}{\varepsilon m^{\ast}}",
             919, 84, 30, 490)
        f.text(1165, 183, "Periodic bulk · no contacts", 25, MUTED, "end")
    if compact:
        boxes = ((95, 312, 445, 300), (95, 1095, 445, 290))
        hs = ((30, 218), (30, 958))
    else:
        boxes = ((100, 275, 1045, 275), (100, 870, 1045, 225))
        hs = ((100, 209), (100, 750))
    f.text(*hs[0], "(a) Density-mode amplitude", 29, weight=500)
    math_label(f, "density-axis", r"a(s)=\mathcal A(t)/A_0", boxes[0][0]+165,
         261 if compact else 235, 30, 330)
    X, Y = axes(f, boxes[0], (0, 15), (-.35, 1.05), (0, 5, 10, 15), (-.25, 0, .5, 1), "", "")
    # The labeled guide selects the same s=1 used in the spatial panel.
    f.line(X(1), boxes[0][1], X(1), boxes[0][1]+boxes[0][3], GOLD, 1.6, dash=True)
    for omega, color in ((.2, BLUE), (1.2, GOLD)):
        f.path([(X(s), Y(charge_amplitude(s, omega)[0])) for s in points(0, 15, 601)], color)
    f.text(X(1)+12, boxes[0][1]+72, "s = 1", 25, GOLD)
    math_label(f, "density-time", r"s=t/\tau", boxes[0][0]+boxes[0][2]/2,
         boxes[0][1]+boxes[0][3]+66, 30, 250)
    if compact:
        legend(f, (("Ω = 0.2: overdamped", BLUE),), (50,), 757, 27)
        legend(f, (("Ω = 1.2: oscillatory", GOLD),), (50,), 811, 27)
        f.line(30, 863, 570, 863)
    else:
        legend(f, (("Ω = 0.2: overdamped", BLUE), ("Ω = 1.2: oscillatory", GOLD)), (150, 700), 685, 27)
    f.text(*hs[1], "(b) One spatial snapshot", 29, weight=500)
    math_label(f, "snapshot-values", rf"\Omega=1.2,\quad s=1,\quad a={a:.3f},\quad a^{{\prime}}={ap:.3f}",
         300 if compact else 480, hs[1][1]+25, 30, 535 if compact else 760)
    f.text(boxes[1][0], boxes[1][1]-24, "Separately normalized quantities", 25 if compact else 27, MUTED)
    X, Y = axes(f, boxes[1], (0, 2*pi), (-.8, .8), ((0, "0"), (pi, "π"), (2*pi, "2π")), (-.5, 0, .5), "", "")
    for fun, color in ((lambda z: a*cos(z), BLUE), (lambda z: -a*sin(z), TEAL), (lambda z: ap*sin(z), GOLD)):
        f.path([(X(z), Y(fun(z))) for z in points(0, 2*pi)], color)
    math_label(f, "snapshot-position", r"qx\quad\text{(one periodic cell)}", boxes[1][0]+boxes[1][2]/2,
         boxes[1][1]+boxes[1][3]+66, 30, 500)
    fields = ((r"\delta n/A_0=a\cos(qx)", BLUE),
              (r"\varepsilon qE/(eA_0)=-a\sin(qx)", TEAL),
              (r"q\tau J/(eA_0)=a^{\prime}\sin(qx)", GOLD))
    if compact:
        for index, (formula, color) in enumerate(fields):
            top = 1524+index*66
            f.line(40, top+16, 75, top+16, color, 4.2)
            math_label(f, f"snapshot-field-{index}", formula, 324, top, 30, 470)
        f.text(30, 1719, "Periodic bulk · no contacts", 26, MUTED)
        math_label(f, "frequency-ratio", r"\Omega=\omega_p\tau", 150, 1758, 30, 240)
        math_label(f, "plasma-frequency", r"\omega_p^2=\frac{n_0e^2}{\varepsilon m^{\ast}}",
             429, 1740, 30, 280)
    else:
        for index, (formula, color) in enumerate(fields):
            xx = 45+index*385
            f.line(xx, 1238, xx+30, 1238, color, 4.2)
            math_label(f, f"snapshot-field-{index}", formula, xx+206, 1222, 30, 335)
    f.save("fig-07-charge" + ("-mobile" if compact else ""))


def integrate(fun, start, stop, intervals=20000):
    step = (stop-start)/intervals
    return step*(.5*fun(start)+sum(fun(start+i*step) for i in range(1, intervals))+.5*fun(stop))


def verify_physics():
    for x in (0, 4, 100):
        assert abs(integrate(lambda th: angular_probability(th, x), 0, pi)-1) < 3e-7
        numerical = integrate(lambda th: angular_probability(th, x)*(1-cos(th)), 0, pi)
        assert abs(numerical-mean_loss(x)) < 3e-9
    for ell in (1, 2):
        area = integrate(lambda x: exp(-x*x/(2*ell*ell))/(sqrt(2*pi)*ell), -20, 20)
        assert abs(area-1) < 1e-12
        assert exp(-0.0*ell*ell) == 1
    step = 1e-4
    for omega in (.2, 1.2):
        assert charge_amplitude(0, omega) == (1, 0)
        for s in (.1, 1, 4, 10):
            a, ap = charge_amplitude(s, omega)
            before, _ = charge_amplitude(s-step, omega)
            after, _ = charge_amplitude(s+step, omega)
            assert abs((after-before)/(2*step)-ap) < 1e-8
            assert abs((after-2*a+before)/step**2+ap+omega*omega*a) < 1e-7
    for q in (-2, -.4, 0, .4, 2):
        assert abs(((1-cos(q+step))-(1-cos(q-step)))/(2*step)-sin(q)) < 2e-9
    assert abs(atan(1)-pi/4) < 1e-14
    assert abs(cos(-pi/4)/sqrt(2)-.5) < 1e-14
    print("Physics assertions: angular normalization, mean loss, Gaussian area, charge ODE, band derivative, AC phase PASS")
    for module in section_modules():
        module.verify()


def validate_assets(stems, png=False):
    for stem in stems:
        path = OUT / (stem+".svg")
        root = ET.parse(path).getroot()
        ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
        assert len(ids) == len(set(ids)), f"Duplicate SVG IDs: {path}"
        for node in root.iter():
            if "data-latex" in node.attrib:
                x, y, width, height = (float(node.attrib[key]) for key in ("x", "y", "width", "height"))
                assert x >= 0 and y >= 0, f"Equation starts outside canvas: {path}: {node.attrib['data-latex']}"
                assert x+width <= float(root.attrib["width"])+.01, f"Equation exceeds right edge: {path}"
                assert y+height <= float(root.attrib["height"])+.01, f"Equation exceeds bottom edge: {path}"
            for name, value in node.attrib.items():
                if name.endswith("href") and value.startswith("#"):
                    assert value[1:] in ids, f"Broken SVG reference: {value}"
                for target in re.findall(r"url\(#([^)]*)\)", value):
                    assert target in ids, f"Broken SVG paint/marker reference: {target}"
            if node.tag.endswith("}text"):
                assert not any("\u2070" <= char <= "\u209f" for char in "".join(node.itertext())), \
                    f"Untypeset Unicode super/subscript: {path}"
        assert root.attrib.get("role") == "img" and root.attrib.get("aria-labelledby") == "title desc"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        if stem.startswith("fig-01-fields"):
            labels = {node.attrib["data-latex"] for node in root.iter() if "data-latex" in node.attrib}
            assert r"\varphi(x)=V\!\left(1-\frac{x}{L}\right)" in labels
            assert r"E_x=-\frac{d\varphi}{dx}=\frac{V}{L}" in labels
            assert not root.findall("{http://www.w3.org/2000/svg}circle"), "No arbitrary particle points in the continuum schematic"
        if png:
            data = path.with_suffix(".png").read_bytes()
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack(">II", data[16:24])
            assert (width, height) == (int(root.attrib["width"]), int(root.attrib["height"]))
    print(f"Asset validation: {len(stems)} accessible SVGs" + (f" + {len(stems)} dimension-matched PNGs" if png else "") + " PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cover-only", action="store_true")
    selection.add_argument("--body-only", action="store_true")
    parser.add_argument("--sections", nargs="+", type=int, choices=range(1, 21),
                        help="Regenerate only these numbered-section figures; implies --body-only.")
    parser.add_argument("--png", action="store_true", help="Rasterize selected SVG assets with rsvg-convert.")
    args = parser.parse_args()
    if args.sections and args.cover_only:
        parser.error("--sections cannot be combined with --cover-only")
    verify_physics()
    generated = []
    if not args.body_only and not args.sections:
        for compact in (False, True):
            cover(compact)
            generated.append("cover-mobile" if compact else "cover")
    if not args.cover_only:
        for section, (renderer, stem, external) in section_figures().items():
            if args.sections and section not in args.sections:
                continue
            for compact in (False, True):
                if external:
                    renderer(sys.modules[__name__], compact)
                else:
                    renderer(compact)
                generated.append(stem+("-mobile" if compact else ""))
    if args.png:
        raster = shutil.which("rsvg-convert")
        if not raster:
            raise SystemExit("--png requires rsvg-convert")
        for stem in generated:
            source = OUT/(stem+".svg")
            subprocess.run([raster, str(source), "-o", str(source.with_suffix(".png"))], check=True)
    validate_assets(generated, args.png)


if __name__ == "__main__":
    main()
