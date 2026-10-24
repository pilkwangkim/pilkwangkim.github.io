#!/usr/bin/env python3
"""Draw the NEGF article's shared desktop and mobile calculation maps.

Requires matplotlib. Run from any directory:
    python tools/negf_flow_figure.py

The figure separates Hamiltonian/boundary data from reservoir populations.
Its arrows express dependencies, not a time evolution or a Poisson iteration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets/img/posts/2026-07-22-negf-from-scratch"
INK = "#233443"
MUTED = "#526574"
BLUE = "#456E98"
TEAL = "#367F7B"
AMBER = "#A97524"
EDGE = "#AABBC5"
BLUE_FILL = "#F4F8FC"
TEAL_FILL = "#F2F9F8"
AMBER_FILL = "#FFF8EA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "mathtext.fontset": "stix",
    # Outline the actual math/text glyphs: STIX/DejaVu are not installed in
    # every reader's browser, and font substitution changes both spacing
    # and legibility. The article's image alt text supplies the description.
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


def canvas(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, content, size=14, color=INK, weight="normal", **kwargs):
    weight = "bold" if weight == "semibold" else weight
    return ax.text(x, y, content, fontsize=size, color=color, weight=weight,
                   ha="center", va="center", zorder=4, **kwargs)


def box(ax, x, y, width, height, color=BLUE, fill=BLUE_FILL):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.085",
        facecolor=fill, edgecolor=color, linewidth=1.05, zorder=2,
    ))


def arrow(ax, points, color=BLUE, width=1.4, head=10):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", color=color, linewidth=width,
        mutation_scale=head, capstyle="round", joinstyle="round", zorder=1,
    ))


def note(ax, x, y, content, color=MUTED, size=11.5, **kwargs):
    return text(ax, x, y, content, size=size, color=color, **kwargs)


CONTINUUM = r"$H=-\frac{\hbar^2}{2m^*}\frac{d^2}{dx^2}+U(x)$"
FINITE = r"$(H_D)_{jj}=2t_0+U_j,\quad(H_D)_{j,j+1}=-t_0$"
KINETIC = r"$t_0=\hbar^2/(2m^*a^2)$"
LEAD = r"$H_\alpha,\quad\tau_\alpha$"
SURFACE = r"$g_{s,\alpha}^{R}=\langle s_\alpha|g_\alpha^{R}|s_\alpha\rangle$"
LEAD_GREEN = r"$g_\alpha^{R}=[(E+i0^+)I-H_\alpha]^{-1}$"
SIGMA = r"$\Sigma_\alpha^{R}=\tau_\alpha g_{s,\alpha}^{R}\tau_\alpha^\dagger$"
GAMMA = r"$\Gamma_\alpha=i(\Sigma_\alpha^{R}-\Sigma_\alpha^{R\dagger})$"
RETARDED = r"$G^{R}=[(E+i0^+)I-H_D-\sum_\alpha\Sigma_\alpha^{R}]^{-1}$"
PARTIAL = r"$A_\alpha=G^{R}\Gamma_\alpha G^{A}$"
TRANSMISSION = r"$T(E)=\mathrm{Tr}[\Gamma_LG^{R}\Gamma_RG^{A}]$"
FERMI = r"$(\mu_\alpha,T_\alpha)\ \longrightarrow\ f_\alpha(E)$"
OCCUPIED = r"$G^n=\sum_\alpha A_\alpha f_\alpha$"
CURRENT = r"$I=\frac{2e}{h}\int dE\,T(E)[f_L(E)-f_R(E)]$"


def desktop():
    # At the article's 691 px content width, 21 pt labels become 17.7 px
    # and 17 pt annotations become 14.3 px. Small outer margins reserve
    # almost the full width for equations, rather than empty canvas.
    fig, ax = canvas(11.4, 11.8)
    left, right, width = 0.18, 5.90, 5.32
    lc, rc, center = 2.84, 8.56, 5.70
    text(ax, center, 11.46, "NEGF: dynamics and occupations", 26, weight="semibold")
    note(ax, center, 11.06, "Coherent, elastic transport • prescribed device potential", size=17)
    text(ax, lc, 10.63, "DEVICE DYNAMICS", 18, BLUE, "semibold")
    text(ax, rc, 10.63, "LEAD BOUNDARIES", 18, TEAL, "semibold")

    for x, color, fill in [(left, BLUE, BLUE_FILL), (right, TEAL, TEAL_FILL)]:
        box(ax, x, 9.10, width, 1.25, color, fill)
        box(ax, x, 7.27, width, 1.52, color, fill)
        box(ax, x, 5.37, width, 1.54, color, fill)

    text(ax, lc, 10.06, "Continuum Hamiltonian", 21, weight="semibold")
    text(ax, lc, 9.51, CONTINUUM, 27)
    text(ax, lc, 8.47, "Finite device Hamiltonian", 21, weight="semibold")
    text(ax, lc, 7.99, FINITE, 23)
    text(ax, lc, 7.49, KINETIC, 22)
    text(ax, lc, 6.57, "Open-device propagation", 21, weight="semibold")
    text(ax, lc, 5.91, RETARDED, 23)

    text(ax, rc, 10.06, "Isolated semi-infinite leads", 21, weight="semibold")
    text(ax, rc, 9.62, LEAD, 26)
    note(ax, rc, 9.28, "device–surface coupling", color=TEAL, size=17)
    text(ax, rc, 8.47, "Surface response", 21, weight="semibold")
    text(ax, rc, 7.99, SURFACE, 27)
    text(ax, rc, 7.48, LEAD_GREEN, 23)
    text(ax, rc, 6.65, "Device contact operators", 21, weight="semibold")
    text(ax, rc, 6.16, SIGMA, 27)
    text(ax, rc, 5.62, GAMMA, 25)

    arrow(ax, [(lc, 9.10), (lc, 8.79)])
    note(ax, 4.14, 8.95, "finite difference", color=BLUE, size=17)
    arrow(ax, [(rc, 9.10), (rc, 8.79)], TEAL)
    note(ax, 9.93, 8.95, "surface projection", color=TEAL, size=17)
    arrow(ax, [(lc, 7.27), (lc, 6.91)], head=8)
    arrow(ax, [(rc, 7.27), (rc, 6.91)], TEAL, head=8)
    note(ax, 9.93, 7.09, "exact elimination", color=TEAL, size=17)
    arrow(ax, [(right, 6.14), (left + width, 6.14)], TEAL)

    # Both G and Gamma feed this shared spectral stage. The grouping avoids
    # crossed arrows while retaining both dependencies for A_alpha and T.
    box(ax, left, 3.61, 11.04, 1.30, EDGE, "#F8FAFC")
    ax.plot([center, center], [3.83, 4.70], color="#D1DDE4", lw=0.8, zorder=3)
    text(ax, lc, 4.58, "Contact-resolved spectrum", 20.5, weight="semibold")
    text(ax, lc, 4.06, PARTIAL, 28)
    text(ax, rc, 4.58, "Transmission", 21, weight="semibold")
    text(ax, rc, 4.06, TRANSMISSION, 26)
    ax.plot([lc, lc, center], [5.37, 5.15, 5.15], color=BLUE, lw=1.4, zorder=1)
    note(ax, 2.23, 5.15, r"$G^R$", color=BLUE, size=18)
    ax.plot([rc, rc, center], [5.37, 5.15, 5.15], color=TEAL, lw=1.4, zorder=1)
    arrow(ax, [(center, 5.15), (center, 4.91)], MUTED)
    note(ax, 9.42, 5.15, r"$\Gamma_L,\Gamma_R$", color=TEAL, size=18)

    # Reservoir data is an independent input, with no arrow from the
    # Hamiltonian/spectral stage into this amber box.
    box(ax, 3.20, 2.27, 5.0, 1.14, AMBER, AMBER_FILL)
    text(ax, center, 3.14, "Independent reservoir input", 20, AMBER, "semibold")
    text(ax, center, 2.79, FERMI, 25)
    note(ax, center, 2.43, "Fermi–Dirac occupation", color=AMBER, size=17)

    box(ax, left, 0.73, width, 1.10)
    text(ax, lc, 1.58, "Contact-fed correlation", 21, weight="semibold")
    text(ax, lc, 1.08, OCCUPIED, 28)
    box(ax, right, 0.73, width, 1.10)
    text(ax, rc, 1.55, "Landauer current", 21, weight="semibold")
    text(ax, rc, 1.07, CURRENT, 25)
    arrow(ax, [(1.13, 3.61), (1.13, 1.83)])
    note(ax, 0.65, 2.69, r"$A_\alpha$", color=BLUE, size=20)
    arrow(ax, [(10.27, 3.61), (10.27, 1.83)])
    note(ax, 10.79, 2.69, r"$T(E)$", color=BLUE, size=20)
    arrow(ax, [(center, 2.27), (center, 1.99), (4.28, 1.99), (4.28, 1.83)], AMBER, head=8)
    arrow(ax, [(center, 1.99), (7.12, 1.99), (7.12, 1.83)], AMBER, head=8)
    note(ax, 4.81, 2.01, r"$f_\alpha$", color=AMBER, size=18,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    note(ax, 6.53, 2.01, r"$f_L-f_R$", color=AMBER, size=18,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    arrow(ax, [(lc, 0.73), (lc, 0.55)], head=8)
    box(ax, left, 0.06, width, 0.49)
    text(ax, lc, 0.31, "Density and bond current", 21, weight="semibold")
    note(ax, rc, 0.40, r"$\alpha=L,R\qquad G^A=(G^R)^\dagger$", size=18)
    note(ax, rc, 0.12, "Spin degeneracy: 2", size=17)
    return fig


def mobile():
    # 18 pt labels are about 18 px at a 358 px mobile article width.
    fig, ax = canvas(5.0, 19.28)
    ax.set_ylim(-1.28, 18.0)
    x, width, center = 0.40, 4.20, 2.50
    text(ax, center, 17.69, "NEGF calculation map", 22, weight="semibold")
    note(ax, center, 17.32, "Coherent, elastic transport", size=15)
    text(ax, center, 16.92, "DEVICE DYNAMICS", 16, BLUE, "semibold")

    box(ax, x, 15.57, width, 1.08)
    text(ax, center, 16.36, "Continuum Hamiltonian", 18, weight="semibold")
    text(ax, center, 15.87, CONTINUUM, 24)
    box(ax, x, 13.94, width, 1.31)
    text(ax, center, 14.96, "Finite device Hamiltonian", 18, weight="semibold")
    text(ax, center, 14.52, FINITE, 20.5)
    text(ax, center, 14.13, KINETIC, 20)
    arrow(ax, [(center, 15.57), (center, 15.25)], head=8)
    note(ax, 3.64, 15.41, "finite difference", color=BLUE, size=14.5)

    text(ax, center, 13.53, "LEAD BOUNDARIES", 16, TEAL, "semibold")
    box(ax, x, 12.05, width, 1.16, TEAL, TEAL_FILL)
    text(ax, center, 12.93, "Isolated semi-infinite leads", 17.5, weight="semibold")
    text(ax, center, 12.52, LEAD, 23)
    note(ax, center, 12.23, "device–surface coupling", color=TEAL, size=15)
    box(ax, x, 10.41, width, 1.32, TEAL, TEAL_FILL)
    text(ax, center, 11.44, "Surface response", 18, weight="semibold")
    text(ax, center, 11.02, SURFACE, 24)
    text(ax, center, 10.61, LEAD_GREEN, 20)
    arrow(ax, [(center, 12.05), (center, 11.73)], TEAL, head=8)
    note(ax, 3.65, 11.89, "surface projection", color=TEAL, size=14.5)
    box(ax, x, 8.76, width, 1.45, TEAL, TEAL_FILL)
    text(ax, center, 9.96, "Device contact operators", 18, weight="semibold")
    text(ax, center, 9.50, SIGMA, 24)
    text(ax, center, 8.98, GAMMA, 23)
    arrow(ax, [(center, 10.41), (center, 10.21)], TEAL, head=8)
    note(ax, 3.65, 10.31, "exact elimination", color=TEAL, size=14.5)

    box(ax, x, 7.16, width, 1.13)
    text(ax, center, 7.99, "Open-device propagation", 18, weight="semibold")
    text(ax, center, 7.49, RETARDED, 20.5)
    arrow(ax, [(center, 8.76), (center, 8.29)], TEAL)
    note(ax, 3.45, 8.53, r"$\Sigma_L^R+\Sigma_R^R$", color=TEAL, size=17)
    # H_D bypasses the independent lead construction and enters G directly.
    arrow(ax, [(x, 14.52), (0.11, 14.52), (0.11, 7.72), (x, 7.72)], head=8)
    note(ax, 0.17, 12.9, r"$H_D$", color=BLUE, size=17, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    box(ax, x, 4.91, width, 1.82, EDGE, "#F8FAFC")
    text(ax, center, 6.44, "Contact-resolved spectrum", 17.5, weight="semibold")
    text(ax, center, 6.02, PARTIAL, 25)
    ax.plot([0.65, 4.35], [5.79, 5.79], color="#D1DDE4", lw=0.8, zorder=3)
    text(ax, center, 5.54, "Transmission", 18, weight="semibold")
    text(ax, center, 5.15, TRANSMISSION, 22)
    ax.plot([center, center], [7.16, 6.94], color=BLUE, lw=1.4, zorder=1)
    arrow(ax, [(center, 6.94), (center, 6.73)], MUTED, head=8)
    note(ax, 1.94, 6.94, r"$G^R$", color=BLUE, size=17)
    ax.plot([4.60, 4.89, 4.89, center], [9.38, 9.38, 6.94, 6.94], color=TEAL, lw=1.4, zorder=1)
    note(ax, 4.88, 8.0, r"$\Gamma_L,\Gamma_R$", color=TEAL, size=16,
         rotation=90, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    box(ax, x, 3.45, width, 1.12, AMBER, AMBER_FILL)
    text(ax, center, 4.30, "Independent reservoir input", 17.5, AMBER, "semibold")
    text(ax, center, 3.90, FERMI, 23)
    note(ax, center, 3.62, "Fermi–Dirac occupation", color=AMBER, size=15)

    box(ax, x, 1.96, width, 1.10)
    text(ax, center, 2.77, "Contact-fed correlation", 18, weight="semibold")
    text(ax, center, 2.33, OCCUPIED, 25)
    arrow(ax, [(center, 3.45), (center, 3.06)], AMBER, head=8)
    note(ax, 3.05, 3.26, r"$f_\alpha$", color=AMBER, size=17)
    arrow(ax, [(x, 6.02), (0.11, 6.02), (0.11, 2.51), (x, 2.51)], head=8)
    note(ax, 0.17, 4.21, r"$A_\alpha$", color=BLUE, size=17, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    box(ax, x, 1.12, width, 0.56)
    text(ax, center, 1.40, "Density and bond current", 18, weight="semibold")
    arrow(ax, [(center, 1.96), (center, 1.68)], head=8)

    # The last current box sits below the density branch but has no arrow
    # from it: only T(E) and the independent reservoir imbalance feed I.
    box(ax, x, -0.42, width, 1.13)
    text(ax, center, 0.43, "Landauer current", 18, weight="semibold")
    text(ax, center, -0.05, CURRENT, 21)
    # Separate right-hand buses bring T(E) and reservoir imbalance to I;
    # they terminate at different ports and never join the Gn/density path.
    arrow(ax, [(4.60, 5.15), (4.89, 5.15), (4.89, -0.08), (4.60, -0.08)], head=8)
    note(ax, 4.86, 1.05, r"$T(E)$", color=BLUE, size=16.5, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 0})
    arrow(ax, [(4.60, 3.90), (4.73, 3.90), (4.73, 0.42), (4.60, 0.42)], AMBER, width=1.25, head=8)
    note(ax, 4.72, 1.83, r"$f_L-f_R$", color=AMBER, size=16, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    note(ax, center, -0.77, r"$\alpha=L,R\qquad G^A=(G^R)^\dagger$", size=17)
    note(ax, center, -1.08, "Spin degeneracy: 2", size=15)
    return fig


def save(fig, directory: Path, stem: str):
    for extension in ("svg", "png"):
        filename = directory / f"{stem}.{extension}"
        fig.savefig(filename, dpi=200, facecolor="white", metadata={
            "Title": "NEGF: dynamics and occupations",
            "Description": "Hamiltonian and lead inputs determine spectral quantities; independent reservoir occupations determine occupied correlations and current.",
        } if extension == "svg" else {"Title": "NEGF: dynamics and occupations"})
        if extension == "svg":
            # Matplotlib pads path-coordinate lines with spaces. Normalize
            # generated XML formatting so figure updates pass diff checks.
            svg = filename.read_text(encoding="utf-8")
            filename.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                                encoding="utf-8")
        print(f"{filename}: {filename.stat().st_size:,} bytes")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(desktop(), args.output_dir, "fig-00-negf-workflow")
    save(mobile(), args.output_dir, "fig-00-negf-workflow-mobile")


if __name__ == "__main__":
    main()
