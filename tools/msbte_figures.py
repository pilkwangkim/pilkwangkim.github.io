#!/usr/bin/env python3
"""Reproducible SVG/PNG figures for the multi-subband BTE tutorial.

Physical data use a parabolic effective-mass hard-wall rectangular wire.
No sampled transport data, external artwork, or font downloads are required.
Run ``python3 tools/msbte_figures.py --png`` to rebuild all assets.
Cover equations require LaTeX (amsmath/xcolor) and dvisvgm. Use --body-only
to rebuild the body figures without those cover-only dependencies.
"""

from __future__ import annotations

import argparse
from html import escape
from math import pi, sin, sqrt
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-11-06-multi-subband-bte"
INK = "#203649"
MUTED = "#657887"
BLUE = "#2b6090"
TEAL = "#087e81"
GOLD = "#a46e26"
RED = "#bc5747"
GRID = "#d9e2e6"
PALE = "#f2f6f6"
PAPER = "#fbfcfb"
WHITE = "#ffffff"
COLORS = (BLUE, TEAL, GOLD)
MODES = ((1, 1), (2, 1), (1, 2))
# hbar^2/(2 m0) in meV nm^2, CODATA 2022 constants.
KINETIC = 38.0998212 / 0.19
W, H = 10.0, 8.0


def edge(mode):
    ny, nz = mode
    return KINETIC * pi**2 * ((ny / W)**2 + (nz / H)**2)


class Figure:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.nodes = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 1 1 L 11 6 L 1 11 Z" fill="context-stroke"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        ]

    def text(self, x, y, value, size=28, color=INK, anchor="start", weight=400, rotate=None):
        # U+2099 (subscript n) is absent from common SVG rasterizer fonts.
        # Explicit tspans preserve the notation without tofu or font downloads.
        label=escape(str(value)).replace("ₙ", '<tspan baseline-shift="sub" font-size="72%">n</tspan>')
        transform=f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ''
        self.nodes.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{transform}>{label}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=2, arrow=False, dash=False):
        extra = ' marker-end="url(#arrow)"' if arrow else ''
        extra += ' stroke-dasharray="9 7"' if dash else ''
        self.nodes.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def path(self, points, color=BLUE, width=3.5, dash=False, arrow=False):
        data = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        extra = ' stroke-dasharray="9 7"' if dash else ''
        extra += ' marker-end="url(#arrow)"' if arrow else ''
        self.nodes.append(f'<path d="{data}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def rect(self, x, y, w, h, fill=WHITE, stroke=GRID, radius=0, width=1.5):
        self.nodes.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def polygon(self, points, fill, stroke=GRID, width=1.8):
        self.nodes.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in points)}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def circle(self, x, y, r=6, color=BLUE, fill=None):
        self.nodes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill or color}" stroke="{color}" stroke-width="2"/>')

    def save(self, name):
        data = "\n".join(self.nodes + ["</svg>"]) + "\n"
        ET.fromstring(data)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(data, encoding="utf-8")
        print(f"{name}: {self.width}x{self.height}")


def mix_color(q):
    """Map 0..1 relative probability density to one fixed sequential scale."""
    a, b = (244, 248, 247), (8, 126, 129)
    q = max(0.0, min(1.0, q))
    return "#" + "".join(f"{round(lo + q*(hi-lo)):02x}" for lo, hi in zip(a, b))


def heatmap(f, x, y, w, h, mode, cells=30):
    ny, nz = mode
    for j in range(cells):
        for i in range(cells):
            q = sin(ny*pi*(i+.5)/cells)**2 * sin(nz*pi*(j+.5)/cells)**2
            f.rect(x+i*w/cells, y+j*h/cells, w/cells+.12, h/cells+.12,
                   mix_color(q), "none", width=0)
    f.rect(x, y, w, h, "none", INK, width=1.8)


def axes(f, box, xlim, ylim, xticks, yticks, xlabel, ylabel, size=26):
    x, y, w, h = box
    def X(v):
        return x + w*(v-xlim[0])/(xlim[1]-xlim[0])
    def Y(v):
        return y+h - h*(v-ylim[0])/(ylim[1]-ylim[0])
    for v in yticks:
        f.line(x, Y(v), x+w, Y(v), GRID, 1)
        f.text(x-12, Y(v)+size*.32, f"{v:g}", size, MUTED, "end")
    for v in xticks:
        f.line(X(v), y+h, X(v), y+h+7, MUTED, 1.4)
        f.text(X(v), y+h+size+13, f"{v:g}", size, MUTED, "middle")
    f.line(x, y, x, y+h, MUTED, 1.7)
    f.line(x, y+h, x+w, y+h, MUTED, 1.7)
    f.text(x, y-22, ylabel, size, INK)
    f.text(x+w/2, y+h+2*size+29, xlabel, size, INK, "middle")
    return X, Y


def parabolas(f, box, emax=240, modes=MODES, legend=True, size=26):
    X, Y = axes(f, box, (-.6, .6), (0, emax), (-.6, 0, .6), (0, 80, 160, 240),
                "kₓ (nm⁻¹)", "Energy (meV)", size)
    for mode, col in zip(modes, COLORS):
        f.path([(X(k), Y(edge(mode)+KINETIC*k*k)) for k in [i*.006 for i in range(-100, 101)]], col, 4)
        if legend:
            f.text(X(0), Y(edge(mode))-12, f"{mode}", size, col,"middle")
    return X, Y


COVER_FORMULAS = {
    "energy": r"E_n(x,k)=\epsilon_n(x)+\frac{\hbar^2 k^2}{2m^{\ast}}",
    "balance": (r"\frac{\partial f_n}{\partial t}"
                r"+v_n\frac{\partial f_n}{\partial x}"
                r"+\frac{F_n}{\hbar}\frac{\partial f_n}{\partial k}"
                r"=\color{collision}C_n[f]"),
    "balance-mobile": (r"\begin{aligned}"
                       r"\frac{\partial f_n}{\partial t}"
                       r"&+v_n\frac{\partial f_n}{\partial x}\\[0.25em]"
                       r"&+\frac{F_n}{\hbar}\frac{\partial f_n}{\partial k}"
                       r"=\color{collision}C_n[f]"
                       r"\end{aligned}"),
    "motion": r"v_n=\frac{1}{\hbar}\partial_k E_n,\qquad F_n=-\partial_x E_n",
    "current": (r"I_e(x,t)=\frac{g_s e}{h}\sum_n"
                r"\int_{\epsilon_n(x)}^{\infty}"
                r"\left[f_{n,+}-f_{n,-}\right]\,dE"),
    "sign": r"g_s=2,\qquad I_{\mathrm{charge}}=-I_e",
    "spin-mobile": r"g_s=2",
    "sign-mobile": r"I_{\mathrm{charge}}=-I_e",
}


def cover_math(f, key, cx, top, height, max_width):
    """Typeset exact LaTeX as SVG paths; no font or MathJax dependency in covers."""
    import re
    import tempfile
    latex, dvisvgm = shutil.which("latex"), shutil.which("dvisvgm")
    if not latex or not dvisvgm:
        raise RuntimeError("Cover generation requires LaTeX and dvisvgm; body-only does not.")
    source = (r"\documentclass{article}" "\n"
              r"\usepackage{amsmath,xcolor}" "\n"
              r"\definecolor{ink}{HTML}{203649}" "\n"
              r"\definecolor{collision}{HTML}{087E81}" "\n"
              r"\pagestyle{empty}\begin{document}" "\n"
              r"\(\displaystyle\color{ink}" + COVER_FORMULAS[key] +
              r"\)\end{document}" "\n")
    with tempfile.TemporaryDirectory(prefix="msbte-cover-tex-") as directory:
        path = Path(directory)
        (path/"equation.tex").write_text(source, encoding="utf-8")
        subprocess.run([latex, "-interaction=nonstopmode", "-halt-on-error", "equation.tex"],
                       cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run([dvisvgm, "--no-fonts", "--bbox=min", "--exact",
                        "-o", "equation.svg", "equation.dvi"],
                       cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        svg = (path/"equation.svg").read_text(encoding="utf-8")
    # Every independently generated equation starts its glyph IDs at zero.
    # Give each fragment its own namespace before embedding it.
    svg = re.sub(r"(id=|(?:xlink:)?href=)(['\"])(#?)([^'\"]+)\2",
                 lambda m: m[1]+m[2]+m[3]+key+"-"+m[4]+m[2], svg)
    root = ET.fromstring(svg)
    _, _, natural_w, natural_h = map(float, root.attrib["viewBox"].split())
    scale = min(height/natural_h, max_width/natural_w)
    width, rendered_height = natural_w*scale, natural_h*scale
    root.attrib.update(x=f"{cx-width/2:.2f}", y=f"{top:.2f}",
                       width=f"{width:.2f}", height=f"{rendered_height:.2f}",
                       preserveAspectRatio="xMidYMin meet")
    f.nodes.append(ET.tostring(root, encoding="unicode"))
    return rendered_height


def cover(mobile=False):
    """Equation-led cover; no reduction arrow or independent bulk-BTE plot."""
    w,h=(960,1320) if mobile else (1600,1000)
    margin=56 if mobile else 80
    f=Figure(w,h,"MS-BTE: subband energies, occupation balance, and electron current",
             "An equation-led cover for scalar multi-subband Boltzmann transport. "
             "Parabolic subband energy E_n(x,k)=epsilon_n(x)+hbar squared k squared/(2m*) "
             "sets velocity v_n=partial_k E_n/hbar and force F_n=-partial_x E_n. "
             "The occupation balance is partial_t f_n+v_n partial_x f_n+"
             "(F_n/hbar)partial_k f_n=C_n[f]. "
             "The local electron-flow current is g_s e/h times the sum of the "
             "energy integrals of f_n,+ minus f_n,- above epsilon_n(x). "
             "Both directional occupations are evaluated at the same x,E,t. "
             "The spin factor g_s=2 assumes identical spin populations, and "
             "conventional charge current has the opposite sign.")
    f.text(margin,140 if mobile else 137,"MS-BTE",94 if mobile else 102,weight=600)
    f.text(margin,214 if mobile else 211,
           "Multi-subband Boltzmann transport",42 if mobile else 47,MUTED)
    if mobile:
        sections=((290,"Subband energy"),(510,"Occupation balance"),
                  (938,"Electron-flow current"))
        for y,label in sections:
            f.text(margin,y,label,34,weight=500)
        cover_math(f,"energy",w/2,326,113,848)
        f.line(margin,468,w-margin,468,GRID,1.5)
        cover_math(f,"balance-mobile",w/2,558,211,848)
        cover_math(f,"motion",w/2,806,67,848)
        f.line(margin,894,w-margin,894,GRID,1.5)
        cover_math(f,"current",w/2,997,151,848)
        f.text(margin,1196,"Directional occupations at the same (x, E, t)",30,MUTED)
        cover_math(f,"spin-mobile",123,1240,42,134)
        cover_math(f,"sign-mobile",756,1240,42,296)
    else:
        for y,label in ((292,"Subband energy"),(481,"Occupation balance"),
                        (771,"Electron-flow current")):
            f.text(margin,y,label,34,weight=500)
        cover_math(f,"energy",w/2,319,101,1350)
        f.line(margin,442,w-margin,442,GRID,1.5)
        cover_math(f,"balance",w/2,516,106,1400)
        cover_math(f,"motion",w/2,655,61,1200)
        f.line(margin,731,w-margin,731,GRID,1.5)
        cover_math(f,"current",w/2,808,111,1400)
        f.text(margin,963,"Directional occupations at the same (x, E, t)",28,MUTED)
        cover_math(f,"sign",1283,938,38,474)
    f.save("cover-mobile.svg" if mobile else "cover.svg")


BODY_MODE_COLORS = (BLUE, GOLD, RED)


def body_axes(f, box, xlim, ylim, xticks, yticks, xlabel, ylabel, size=27):
    """Body-only plot grammar; cover geometry and typography stay unchanged."""
    x,y,w,h=box
    X=lambda v:x+w*(v-xlim[0])/(xlim[1]-xlim[0])
    Y=lambda v:y+h-h*(v-ylim[0])/(ylim[1]-ylim[0])
    for value in yticks:
        f.line(x,Y(value),x+w,Y(value),"#e3e9ec",1.2)
        f.text(x-12,Y(value)+size*.32,f"{value:g}",size,MUTED,"end")
    for value in xticks:
        f.line(X(value),y+h,X(value),y+h+7,MUTED,1.6)
        f.text(X(value),y+h+size+13,f"{value:g}",size,MUTED,"middle")
    f.line(x,y,x,y+h,MUTED,1.8)
    f.line(x,y+h,x+w,y+h,MUTED,1.8)
    if ylabel:
        f.text(x,y-22,ylabel,size,INK)
    f.text(x+w/2,y+h+2*size+25,xlabel,size,INK,"middle")
    return X,Y


def body_legend(f, items, positions, y, size=26):
    for (label,color,dash), x in zip(items,positions):
        f.line(x,y-8,x+32,y-8,color,3.8,dash=dash)
        f.text(x+44,y,label,size,color)


def body_density(f, box, mode):
    """A common relative-density scale with seam-free cells and a closed wall."""
    x,y,w,h=box
    f.nodes.append('<g shape-rendering="crispEdges">')
    heatmap(f,x,y,w,h,mode,cells=60)
    f.nodes.append('</g>')
    f.rect(x,y,w,h,"none",INK,width=1.8)


def body_density_scale(f, x, y, width, size=26):
    # The field is |phi|^2=4/(W H) times the dimensionless color argument.
    f.text(x,y-23,"|φₙ|² (nm⁻²)",size,INK)
    f.nodes.append(f'<defs><linearGradient id="body-density-scale" x1="0%" x2="100%" y1="0%" y2="0%"><stop offset="0%" stop-color="{mix_color(0)}"/><stop offset="100%" stop-color="{mix_color(1)}"/></linearGradient></defs>')
    f.rect(x,y,width,22,"url(#body-density-scale)",GRID,width=1)
    for fraction,label in ((0,"0"),(.5,"0.025"),(1,"0.050")):
        xx=x+fraction*width
        f.line(xx,y+22,xx,y+29,MUTED,1.2)
        f.text(xx,y+size+43,label,size,MUTED,"middle")


def confinement(mobile=False):
    w,h=(600,1280) if mobile else (1200,760)
    f=Figure(w,h,"Transverse probability densities and longitudinal subbands",
             "Panel a shows the first three energy-ordered hard-wall modes n=1,2,3, corresponding to (ny,nz)=(1,1),(2,1),(1,2), in a 10 by 8 nanometre section. Horizontal is y and vertical is z. All density maps share the scale 0 to 4/(W H)=0.05 per square nanometre and integrate to one. Panel b plots the associated parabolic dispersions, using blue, gold, and red consistently for modes one, two, and three. Their edges are 50.7,110.1,143.5 meV for mass 0.19 m0. Legends are outside the curves.")
    f.text(30,46,"Confinement and dispersion",31,weight=500)
    f.text(30,88,"W = 10 nm · H = 8 nm · m* = 0.19m₀",25,MUTED)
    f.text(30,141,"(a) Transverse probability density",27 if mobile else 28,weight=500)
    if mobile:
        maps=[(40+180*i,200,140,112) for i in range(3)]
        mode_y,pair_y,energy_y=181,366,408
        scale=(83,476,434)
        f.text(30,600,"4/(WH) = 0.050 nm⁻²",26,MUTED)
        f.text(30,645,"∫ |φₙ|² dy dz = 1",28,MUTED)
        f.line(30,676,570,676,GRID,1.4)
        f.text(30,724,"(b) Longitudinal subbands",29,weight=500)
        plot=(90,845,462,300)
        legend_y,legend_x=773,(80,250,420)
    else:
        maps=[(40+185*i,215,150,120) for i in range(3)]
        mode_y,pair_y,energy_y=193,389,433
        scale=(65,505,470)
        f.text(40,625,"4/(WH) = 0.050 nm⁻²",26,MUTED)
        f.text(40,676,"∫ |φₙ|² dy dz = 1",28,MUTED)
        f.line(618,128,618,695,GRID,1.4)
        f.text(660,141,"(b) Longitudinal subbands",28,weight=500)
        plot=(714,225,438,335)
        legend_y,legend_x=690,(694,853,1012)
    for index,(box,mode,color) in enumerate(zip(maps,MODES,BODY_MODE_COLORS),1):
        x,y,mw,mh=box
        f.text(x+mw/2,mode_y,f"n = {index}",28,color,"middle",500)
        body_density(f,box,mode)
        f.text(x+mw/2,pair_y,f"({mode[0]}, {mode[1]})",28,color,"middle")
        f.text(x+mw/2,energy_y,f"{edge(mode):.1f} meV",27,INK,"middle")
    x,y,mw,mh=maps[0]
    f.line(x-18,y+mh,x-18,y+17,MUTED,1.6,arrow=True)
    f.text(x-18,y+1,"z",24,MUTED,"middle")
    f.line(x,y+mh+17,x+mw-12,y+mh+17,MUTED,1.6,arrow=True)
    f.text(x+mw+8,y+mh+24,"y",24,MUTED)
    body_density_scale(f,*scale,size=26)
    X,Y=body_axes(f,plot,(-.6,.6),(0,240),(-.6,0,.6),(0,80,160,240),
                  "kₓ (nm⁻¹)","Energy E (meV)",27)
    for mode,color in zip(MODES,BODY_MODE_COLORS):
        f.path([(X(k),Y(edge(mode)+KINETIC*k*k)) for k in [i*.006 for i in range(-100,101)]],color,4)
        f.circle(X(0),Y(edge(mode)),4,color)
    body_legend(f,[(f"n = {i}",color,False) for i,color in enumerate(BODY_MODE_COLORS,1)],
                legend_x,legend_y,25)
    if not mobile:
        f.text(660,732,"Eₙ(kₓ) = εₙ + ℏ²kₓ²/(2m*)",27,INK)
    f.save("fig-01-confinement-mobile.svg" if mobile else "fig-01-confinement.svg")


def contacts(mobile=False):
    w,h=(600,895) if mobile else (1200,570)
    f=Figure(w,h,"Half-range reservoir boundary conditions",
             "Solid arrows cross each incoming boundary from a reservoir; dashed arrows carry outgoing occupations to the reservoir. Blue is positive velocity and gold is negative velocity. Only f-plus at x=0 and f-minus at x=L are fixed. The interior occupations and the two outgoing halves are transport unknowns. Mobile stacks the reservoir labels but keeps the physical x direction horizontal.")
    f.text(30,47,"Reservoir boundary conditions",30 if mobile else 32,weight=500)
    body_legend(f,[("Inflow fixed",MUTED,False),("Outflow solved",MUTED,True)],
                (40,322) if mobile else (45,365),101,26)
    if mobile:
        f.rect(30,140,540,112,PALE,GRID,8)
        f.text(53,183,"Left reservoir",28,BLUE,weight=500)
        f.text(53,225,"fₙ⁺(0,E) = fL(E)",30,BLUE)
        f.rect(30,696,540,112,PALE,GRID,8)
        f.text(53,740,"Right reservoir",28,GOLD,weight=500)
        f.text(53,783,"fₙ⁻(L,E) = fR(E)",30,GOLD)
        f.text(300,290,"Uniform wire · subband n",27,INK,"middle")
        f.rect(90,315,420,238,WHITE,GRID,4)
        f.path([(35,252),(35,375),(110,375)],BLUE,3,arrow=True)
        f.path([(565,696),(565,485),(490,485)],GOLD,3,arrow=True)
        f.line(490,375,565,375,BLUE,3,arrow=True,dash=True)
        f.line(110,485,35,485,GOLD,3,arrow=True,dash=True)
        f.text(300,387,"fₙ⁺(x,E)",31,BLUE,"middle")
        f.text(300,347,"vₙ > 0",25,BLUE,"middle")
        f.text(300,497,"fₙ⁻(x,E)",31,GOLD,"middle")
        f.text(300,536,"vₙ < 0",25,GOLD,"middle")
        f.text(90,597,"x = 0",27,MUTED,"middle")
        f.text(510,597,"x = L",27,MUTED,"middle")
        f.text(300,651,"Outgoing values are not fixed.",26,MUTED,"middle")
        f.text(300,860,"Only incoming halves are prescribed.",27,INK,"middle")
    else:
        f.rect(40,176,205,205,PALE,GRID,8)
        f.rect(955,176,205,205,PALE,GRID,8)
        f.text(142,216,"Left reservoir",25,BLUE,"middle",500)
        f.text(142,282,"fL(E)",31,BLUE,"middle")
        f.text(1057,216,"Right reservoir",25,GOLD,"middle",500)
        f.text(1057,282,"fR(E)",31,GOLD,"middle")
        f.text(600,163,"Uniform wire · subband n",28,INK,"middle")
        f.rect(330,200,540,155,WHITE,GRID,4)
        f.line(225,238,350,238,BLUE,3.7,arrow=True)
        f.line(850,238,975,238,BLUE,3.2,arrow=True,dash=True)
        f.line(975,313,850,313,GOLD,3.7,arrow=True)
        f.line(350,313,225,313,GOLD,3.2,arrow=True,dash=True)
        f.text(584,250,"fₙ⁺(x,E)",30,BLUE,"middle")
        f.text(584,325,"fₙ⁻(x,E)",30,GOLD,"middle")
        f.text(748,250,"vₙ > 0",24,BLUE)
        f.text(748,325,"vₙ < 0",24,GOLD)
        f.text(330,407,"x = 0",27,MUTED,"middle")
        f.text(870,407,"x = L",27,MUTED,"middle")
        f.text(45,466,"fₙ⁺(0,E) = fL(E)",29,BLUE)
        f.text(1155,466,"fₙ⁻(L,E) = fR(E)",29,GOLD,"end")
        f.text(600,536,"The interior and outgoing occupations are solved.",29,INK,"middle")
    f.save("fig-02-contacts-mobile.svg" if mobile else "fig-02-contacts.svg")


def body_energy_budget(f, box, mobile=False):
    x,y,w,h=box
    X,Y=body_axes(f,box,(0,3),(0,150),(),(0,50,100,150),"","Energy (meV)",26)
    centers=(X(.8),X(2.2))
    total=130
    for i,(cx,mode,color) in enumerate(zip(centers,MODES[:2],BODY_MODE_COLORS[:2]),1):
        eps=edge(mode)
        kinetic=total-eps
        bw=88 if not mobile else 94
        f.rect(cx-bw/2,Y(eps),bw,Y(0)-Y(eps),"#edf1f4",color,width=2)
        f.rect(cx-bw/2,Y(total),bw,Y(eps)-Y(total),"#dcefed",TEAL,width=2)
        f.text(cx,(Y(eps)+Y(0))/2+9,f"{eps:.1f}",25,INK,"middle")
        f.text(cx,(Y(total)+Y(eps))/2+9,f"{kinetic:.1f}",24,INK,"middle")
        f.text(cx,Y(0)+43,f"n = {i}",27,color,"middle")
    f.line(x,Y(total),x+w,Y(total),MUTED,1.5,dash=True)
    f.text(x+w,Y(total)-12,"130",25,MUTED,"end")
    ly=y+h+93
    for xx,label,fill,col in ((x+15,"εₙ","#edf1f4",INK),(x+w*.59,"Kₙ","#dcefed",TEAL)):
        f.rect(xx,ly-18,24,19,fill,col,width=1.5)
        f.text(xx+35,ly,label,27,INK)


def elastic(mobile=False):
    w,h=(600,1270) if mobile else (1200,760)
    f=Figure(w,h,"Elastic transitions conserve total energy, not longitudinal kinetic energy",
             "Panel a shows an elastic same-direction transition at E=130 meV from mode one (blue) to mode two (gold). Their positive wavenumbers are 0.629 and 0.315 per nanometre. The teal arrow is a transition between states, not a real-space trajectory. Mode three (red dashed) is closed because its edge 143.5 meV exceeds E. Panel b accounts for the same total energy: mode one has transverse and axial kinetic energies 50.7 and 79.3 meV; mode two has 110.1 and 19.9 meV. In each stack the sum is 130 meV. No new scattering rate is inferred from this kinematic diagram.")
    f.text(30,46,"Elastic inter-subband scattering",30 if mobile else 32,weight=500)
    f.text(30,88,"E = 130 meV · W = 10 nm · H = 8 nm",25,MUTED)
    f.text(30 if mobile else 80,145,"(a) Equal-total-energy states",28,weight=500)
    if mobile:
        plot=(85,220,465,330)
        legend_x,legend_y=(48,210,372),681
        f.line(30,714,570,714,GRID,1.4)
        f.text(30,759,"(b) Energy accounting",29,weight=500)
        budget=(105,830,390,240)
    else:
        plot=(95,220,665,330)
        legend_x,legend_y=(100,310,520),681
        f.text(835,145,"(b) Energy accounting",28,weight=500)
        budget=(850,240,295,310)
    X,Y=body_axes(f,plot,(-.9,.9),(0,210),(-.8,0,.8),(0,50,100,150,200),
                  "kₓ (nm⁻¹)","Energy E (meV)",27)
    for mode,color in zip(MODES,BODY_MODE_COLORS):
        points=[]
        for i in range(-150,151):
            k=i*.006
            e=edge(mode)+KINETIC*k*k
            if e<=210:
                points.append((X(k),Y(e)))
        f.path(points,color,3.6,dash=mode==MODES[2])
    total=130
    f.line(X(-.9),Y(total),X(.9),Y(total),MUTED,1.8,dash=True)
    f.text(X(.85),Y(total)-13,"E",25,MUTED,"end")
    momenta=[sqrt((total-edge(mode))/KINETIC) for mode in MODES[:2]]
    f.line(X(momenta[0])-11,Y(total),X(momenta[1])+11,Y(total),TEAL,3.5,arrow=True)
    for k,color in zip(momenta,BODY_MODE_COLORS[:2]):
        f.circle(X(k),Y(total),6,color,WHITE)
        f.line(X(k),Y(total)+10,X(k),Y(0),color,1.4,dash=True)
        f.circle(X(-k),Y(total),4.5,color,WHITE)
    body_legend(f,[("n = 1",BLUE,False),("n = 2",GOLD,False),("n = 3 (closed)",RED,True)],
                legend_x,legend_y,26 if mobile else 25)
    body_energy_budget(f,budget,mobile)
    f.text(w/2,1233 if mobile else 737,"E = εₙ + Kₙ = 130 meV",29,INK,"middle")
    f.save("fig-03-elastic-mobile.svg" if mobile else "fig-03-elastic.svg")


def profile_panel(f, box, b, size=27):
    X,Y=body_axes(f,box,(0,1),(0,1),(0,.5,1),(0,.5,1),"x / L","Occupation f",size)
    fL,fR=.9,.2
    delta=(fL-fR)/(1+b)
    plus=lambda x:fL-b*x*delta
    minus=lambda x:fR+b*(1-x)*delta
    f.path([(X(x),Y(plus(x))) for x in (0,1)],BLUE,4)
    f.path([(X(x),Y(minus(x))) for x in (0,1)],GOLD,4,dash=True)
    f.circle(X(0),Y(fL),6,BLUE)
    f.circle(X(1),Y(fR),6,GOLD)
    xp=.55
    f.line(X(xp),Y(minus(xp)),X(xp),Y(plus(xp)),TEAL,2.4)
    for value in (minus(xp),plus(xp)):
        f.line(X(xp)-7,Y(value),X(xp)+7,Y(value),TEAL,2.4)
    f.text(X(.5),box[1]+box[3]+130,f"f⁺ − f⁻ = {delta:.3f}",29,TEAL,"middle")


def profiles(mobile=False):
    w,h=(600,1310) if mobile else (1200,745)
    f=Figure(w,h,"Elastic backscattering and directional imbalance",
             "Exact continuum single-subband solutions use incoming occupations fL=0.9 and fR=0.2. Panel a has beta L=0; panel b has beta L=2. Positive-direction occupation is blue solid and negative-direction occupation is gold dashed. Both spatial derivatives in panel b are negative and equal, leaving a constant difference 0.7/3 instead of the ballistic 0.7. Filled points identify only the prescribed incoming values at the left positive and right negative boundary.")
    f.text(30,46,"Backscattering and net flow" if mobile else "Backscattering and directional imbalance",31 if mobile else 32,weight=500)
    f.text(30,89,"Fixed inflow: fL = 0.9, fR = 0.2",26,MUTED)
    body_legend(f,[("f⁺  (v > 0)",BLUE,False),("f⁻  (v < 0)",GOLD,True)],
                (65,320) if mobile else (650,925),141 if mobile else 90,25)
    if mobile:
        f.text(30,205,"(a) Ballistic · βL = 0",29,weight=500)
        profile_panel(f,(85,267,465,260),0,28)
        f.line(30,709,570,709,GRID,1.4)
        f.text(30,756,"(b) Elastic reversal · βL = 2",29,weight=500)
        profile_panel(f,(85,818,465,260),2,28)
        f.circle(80,1264,5,MUTED)
        f.text(100,1273,"Fixed incoming occupations",26,MUTED)
    else:
        f.text(85,157,"(a) Ballistic · βL = 0",29,weight=500)
        f.text(690,157,"(b) Elastic reversal · βL = 2",29,weight=500)
        profile_panel(f,(85,225,465,300),0)
        profile_panel(f,(690,225,465,300),2)
        f.circle(393,709,5,MUTED)
        f.text(414,718,"Fixed incoming occupations",26,MUTED)
    f.save("fig-04-profiles-mobile.svg" if mobile else "fig-04-profiles.svg")


def force(mobile=False):
    w,h=(600,1200) if mobile else (1200,750)
    f=Figure(w,h,"A rising subband edge converts axial kinetic energy into potential energy",
             "The prescribed subband edge epsilon(x)=50+70x/L meV rises from 50 to 120 meV while total energy is 170 meV. Panel a shows the energy gap K_n=E-epsilon_n. Panel b shows k_x(x)=sqrt(2 m* (E-epsilon_n(x)))/hbar for mass 0.19m0. A right mover slows from positive k about 0.774 to 0.499 per nanometre without turning. The arrow follows increasing x, not a changing quantum phase. The force is minus the derivative of the included subband energy.")
    f.text(30,46,"A rising edge slows the electron",30 if mobile else 32,weight=500)
    f.text(30,89,"εₙ(x) = (50 + 70x/L) meV · E = 170 meV",24 if mobile else 26,MUTED)
    if mobile:
        boxes=((85,220,465,260),(85,790,465,240))
        f.text(30,151,"(a) Energy conversion",29,weight=500)
        f.text(30,721,"(b) Semiclassical trajectory",29,weight=500)
        legend_y=622
        f.line(30,669,570,669,GRID,1.4)
    else:
        boxes=((85,235,465,285),(690,235,465,285))
        f.text(85,151,"(a) Energy conversion",29,weight=500)
        f.text(690,151,"(b) Semiclassical trajectory",29,weight=500)
        legend_y=651
    X,Y=body_axes(f,boxes[0],(0,1),(0,200),(0,.5,1),(0,100,200),
                  "x / L","Energy (meV)",27)
    f.path([(X(x),Y(50+70*x)) for x in (0,1)],BLUE,4)
    f.line(X(0),Y(170),X(1),Y(170),MUTED,1.9,dash=True)
    xp=.38
    f.line(X(xp),Y(50+70*xp),X(xp),Y(170),TEAL,2.5)
    for value in (50+70*xp,170):
        f.line(X(xp)-6,Y(value),X(xp)+6,Y(value),TEAL,2.5)
    f.text(X(xp)+12,(Y(50+70*xp)+Y(170))/2+9,"Kₙ",28,TEAL)
    body_legend(f,[("εₙ(x)",BLUE,False),("E = 170",MUTED,True)],(85,335),legend_y,26)
    X,Y=body_axes(f,boxes[1],(0,1),(0,.9),(0,.5,1),(0,.4,.8),
                  "x / L","Positive kₓ (nm⁻¹)",27)
    vals=[i/100 for i in range(101)]
    f.path([(X(x),Y(sqrt((170-(50+70*x))/KINETIC))) for x in vals],TEAL,4)
    a,b=.4,.63
    f.path([(X(x),Y(sqrt((170-(50+70*x))/KINETIC))) for x in [a+(b-a)*i/20 for i in range(21)]],TEAL,4,arrow=True)
    if not mobile:
        f.text(690,651,"kₓ(x) = √[2m*(E − εₙ(x))] / ℏ",25,TEAL)
    f.text(w/2,1165 if mobile else 716,"Fₙ = −∂εₙ/∂x",31,INK,"middle")
    f.save("fig-05-force-mobile.svg" if mobile else "fig-05-force.svg")


def verify_physics():
    assert abs(edge((1,1))-50.714)<.03
    assert abs(edge((2,1))-110.087)<.03
    assert abs(edge((1,2))-143.48)<.04
    for mode in MODES[:2]:
        k=sqrt((130-edge(mode))/KINETIC)
        assert abs(edge(mode)+KINETIC*k*k-130)<1e-11
    d=(.9-.2)/3
    assert abs((.9-2*d)-.2-d)<1e-12
    assert abs((.2+2*d)-(.9-d))<1e-12
    assert abs(4/(W*H)-.05)<1e-14
    for mode in MODES:
        ny,nz=mode
        cells=60
        sampled_integral=sum(4/(W*H)*sin(ny*pi*(i+.5)/cells)**2
                             *sin(nz*pi*(j+.5)/cells)**2
                             *W*H/cells**2
                             for i in range(cells) for j in range(cells))
        assert abs(sampled_integral-1)<1e-13
    for mode in MODES[:2]:
        assert abs(edge(mode)+(130-edge(mode))-130)<1e-14


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png",action="store_true",help="Rasterize every SVG to matching-size PNG.")
    parser.add_argument("--cover-only",action="store_true",help="Rebuild only the paired cover assets.")
    parser.add_argument("--body-only",action="store_true",help="Rebuild only fig-01 through fig-05, preserving covers and other figures.")
    args=parser.parse_args()
    if args.cover_only and args.body_only:
        parser.error("--cover-only and --body-only are mutually exclusive")
    verify_physics()
    body_renderers=(confinement,contacts,elastic,profiles,force)
    renderers=(cover,) if args.cover_only else body_renderers if args.body_only else (cover,)+body_renderers
    for render in renderers:
        for mobile in (False,True):
            render(mobile)
    if args.png:
        raster=shutil.which("rsvg-convert")
        if not raster:
            raise SystemExit("--png requires rsvg-convert")
        body_stems=("fig-01-confinement","fig-02-contacts","fig-03-elastic","fig-04-profiles","fig-05-force")
        stems=("cover",) if args.cover_only else body_stems if args.body_only else ("cover",)+body_stems
        targets=[OUT/(stem+suffix+".svg") for stem in stems for suffix in ("","-mobile")]
        for svg in targets:
            subprocess.run([raster,str(svg),"-o",str(svg.with_suffix(".png"))],check=True)
            print(svg.with_suffix(".png").name)
