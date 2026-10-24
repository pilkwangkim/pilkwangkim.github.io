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
    "svg.fonttype": "none",
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
LEAD = r"$H_\alpha,\quad\tau_\alpha$" + "  (device–surface coupling)"
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
    fig, ax = canvas(12.4, 12.2)
    text(ax, 6.2, 11.82, "NEGF: dynamics and occupations", 21, weight="semibold")
    note(ax, 6.2, 11.41, "Coherent, elastic transport with a prescribed device potential", size=12.3)
    text(ax, 3.15, 10.97, "DEVICE DYNAMICS", 12.4, BLUE, "semibold")
    text(ax, 9.25, 10.97, "LEAD BOUNDARIES", 12.4, TEAL, "semibold")

    for x, color, fill in [(0.5, BLUE, BLUE_FILL), (6.6, TEAL, TEAL_FILL)]:
        box(ax, x, 9.75, 5.3, 0.9, color, fill)
        box(ax, x, 8.2, 5.3, 1.05, color, fill)
        box(ax, x, 6.45, 5.3, 1.2, color, fill)

    text(ax, 3.15, 10.38, "Continuum Hamiltonian", 14, weight="semibold")
    text(ax, 3.15, 10.03, CONTINUUM, 17)
    text(ax, 3.15, 8.98, "Finite device Hamiltonian", 14, weight="semibold")
    text(ax, 3.15, 8.64, FINITE, 15)
    text(ax, 3.15, 8.36, KINETIC, 14)
    text(ax, 3.15, 7.36, "Open-device propagation", 14, weight="semibold")
    text(ax, 3.15, 6.89, RETARDED, 15.5)

    text(ax, 9.25, 10.38, "Isolated semi-infinite leads", 14, weight="semibold")
    text(ax, 9.25, 10.02, LEAD, 16)
    text(ax, 9.25, 8.98, "Surface response", 14, weight="semibold")
    text(ax, 9.25, 8.64, SURFACE, 16.5)
    text(ax, 9.25, 8.34, LEAD_GREEN, 14)
    text(ax, 9.25, 7.36, "Contact operators on the device", 14, weight="semibold")
    text(ax, 9.25, 6.99, SIGMA, 17)
    text(ax, 9.25, 6.66, GAMMA, 16.5)

    arrow(ax, [(3.15, 9.75), (3.15, 9.25)])
    note(ax, 4.05, 9.5, "finite difference", color=BLUE)
    arrow(ax, [(9.25, 9.75), (9.25, 9.25)], TEAL)
    note(ax, 10.25, 9.5, "surface projection", color=TEAL)
    arrow(ax, [(3.15, 8.2), (3.15, 7.65)])
    arrow(ax, [(9.25, 8.2), (9.25, 7.65)], TEAL)
    note(ax, 10.34, 7.92, "exact elimination", color=TEAL)
    arrow(ax, [(6.6, 7.05), (5.8, 7.05)], TEAL)

    # Both G and Gamma feed this shared spectral stage. The grouping avoids
    # crossed arrows while retaining both dependencies for A_alpha and T.
    box(ax, 0.5, 4.45, 11.4, 1.45, EDGE, "#F8FAFC")
    ax.plot([6.2, 6.2], [4.68, 5.67], color="#D1DDE4", lw=0.8, zorder=3)
    text(ax, 3.35, 5.54, "Contact-resolved spectrum", 14, weight="semibold")
    text(ax, 3.35, 5.04, PARTIAL, 19)
    text(ax, 9.05, 5.54, "Transmission", 14, weight="semibold")
    text(ax, 9.05, 5.04, TRANSMISSION, 18)
    ax.plot([3.15, 3.15, 6.2], [6.45, 6.17, 6.17], color=BLUE, lw=1.4, zorder=1)
    note(ax, 2.52, 6.18, r"$G^R$", color=BLUE, size=13)
    ax.plot([9.25, 9.25, 6.2], [6.45, 6.17, 6.17], color=TEAL, lw=1.4, zorder=1)
    arrow(ax, [(6.2, 6.17), (6.2, 5.9)], MUTED)
    note(ax, 10.04, 6.18, r"$\Gamma_L,\Gamma_R$", color=TEAL, size=13)

    # Reservoir data is an independent input, with no arrow from the
    # Hamiltonian/spectral stage into this amber box.
    box(ax, 4.2, 2.8, 4.0, 1.13, AMBER, AMBER_FILL)
    text(ax, 6.2, 3.66, "Independent reservoir input", 13.4, AMBER, "semibold")
    text(ax, 6.2, 3.28, FERMI, 17)
    note(ax, 6.2, 3.01, "Fermi–Dirac occupation", color=AMBER)

    box(ax, 0.5, 1.1, 5.3, 1.15)
    text(ax, 3.15, 1.97, "Contact-fed correlation", 14, weight="semibold")
    text(ax, 3.15, 1.53, OCCUPIED, 19)
    box(ax, 6.6, 1.1, 5.3, 1.15)
    text(ax, 9.25, 1.97, "Landauer current", 14, weight="semibold")
    text(ax, 9.25, 1.5, CURRENT, 16)
    arrow(ax, [(2.1, 4.45), (2.1, 2.25)])
    note(ax, 1.66, 3.37, r"$A_\alpha$", color=BLUE, size=14)
    arrow(ax, [(10.3, 4.45), (10.3, 2.25)])
    note(ax, 10.83, 3.37, r"$T(E)$", color=BLUE, size=14)
    arrow(ax, [(6.2, 2.8), (6.2, 2.6), (4.5, 2.6), (4.5, 2.25)], AMBER)
    arrow(ax, [(6.2, 2.6), (7.9, 2.6), (7.9, 2.25)], AMBER)
    note(ax, 4.92, 2.42, r"$f_\alpha$", color=AMBER, size=13)
    note(ax, 7.46, 2.42, r"$f_L-f_R$", color=AMBER, size=13)
    arrow(ax, [(3.15, 1.1), (3.15, 0.7)])
    box(ax, 0.5, 0.1, 5.3, 0.6)
    text(ax, 3.15, 0.4, "Density and bond current", 14, weight="semibold")
    note(ax, 9.25, 0.53, r"$\alpha=L,R\qquad G^A=(G^R)^\dagger$", size=13)
    note(ax, 9.25, 0.2, "Spin degeneracy: 2", size=11.5)
    return fig


def mobile():
    fig, ax = canvas(5.4, 20.0)
    text(ax, 2.7, 19.61, "NEGF calculation map", 18.5, weight="semibold")
    note(ax, 2.7, 19.2, "Coherent, elastic transport • prescribed potential", size=11.3)
    text(ax, 2.7, 18.75, "DEVICE DYNAMICS", 12.3, BLUE, "semibold")

    x, width, center = 0.52, 4.36, 2.7
    box(ax, x, 17.55, width, 0.9)
    text(ax, center, 18.2, "Continuum Hamiltonian", 13.3, weight="semibold")
    text(ax, center, 17.83, CONTINUUM, 16)
    box(ax, x, 16.15, width, 1.05)
    text(ax, center, 16.94, "Finite device Hamiltonian", 13.3, weight="semibold")
    text(ax, center, 16.59, FINITE, 13.3)
    text(ax, center, 16.31, KINETIC, 13.3)
    arrow(ax, [(center, 17.55), (center, 17.2)])
    note(ax, 3.61, 17.39, "finite difference", color=BLUE, size=11.2)

    text(ax, center, 15.68, "LEAD BOUNDARIES", 12.3, TEAL, "semibold")
    box(ax, x, 14.55, width, 0.85, TEAL, TEAL_FILL)
    text(ax, center, 15.15, "Isolated semi-infinite leads", 13.3, weight="semibold")
    text(ax, center, 14.81, LEAD, 14.5)
    box(ax, x, 13.05, width, 1.05, TEAL, TEAL_FILL)
    text(ax, center, 13.86, "Surface response", 13.3, weight="semibold")
    text(ax, center, 13.51, SURFACE, 16)
    text(ax, center, 13.2, LEAD_GREEN, 13.1)
    arrow(ax, [(center, 14.55), (center, 14.1)], TEAL)
    note(ax, 3.72, 14.33, "surface projection", color=TEAL, size=11.2)
    box(ax, x, 11.45, width, 1.15, TEAL, TEAL_FILL)
    text(ax, center, 12.33, "Contact operators on the device", 12.9, weight="semibold")
    text(ax, center, 11.97, SIGMA, 16.5)
    text(ax, center, 11.63, GAMMA, 16)
    arrow(ax, [(center, 13.05), (center, 12.6)], TEAL)
    note(ax, 3.73, 12.83, "exact elimination", color=TEAL, size=11.2)

    box(ax, x, 9.8, width, 1.1)
    text(ax, center, 10.6, "Open-device propagation", 13.3, weight="semibold")
    text(ax, center, 10.15, RETARDED, 13.3)
    arrow(ax, [(center, 11.45), (center, 10.9)], TEAL)
    note(ax, 3.55, 11.18, r"$\Sigma_L^R+\Sigma_R^R$", color=TEAL, size=12.5)
    # H_D bypasses the independent lead construction and enters G directly.
    arrow(ax, [(x, 16.66), (0.15, 16.66), (0.15, 10.35), (x, 10.35)])
    note(ax, 0.12, 13.3, r"$H_D$", color=BLUE, size=13, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    box(ax, x, 7.6, width, 1.7, EDGE, "#F8FAFC")
    text(ax, center, 9.02, "Contact-resolved spectrum", 13.1, weight="semibold")
    text(ax, center, 8.65, PARTIAL, 18)
    ax.plot([0.8, 4.6], [8.39, 8.39], color="#D1DDE4", lw=0.8, zorder=3)
    text(ax, center, 8.12, "Transmission", 13.1, weight="semibold")
    text(ax, center, 7.83, TRANSMISSION, 15.6)
    ax.plot([center, center], [9.8, 9.55], color=BLUE, lw=1.4, zorder=1)
    arrow(ax, [(center, 9.55), (center, 9.3)], MUTED)
    note(ax, 2.15, 9.55, r"$G^R$", color=BLUE, size=13)
    ax.plot([4.88, 5.24, 5.24, center], [12.0, 12.0, 9.55, 9.55], color=TEAL, lw=1.4, zorder=1)
    note(ax, 5.23, 10.08, r"$\Gamma_L,\Gamma_R$", color=TEAL, size=12,
         rotation=90, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    box(ax, x, 5.85, width, 1.15, AMBER, AMBER_FILL)
    text(ax, center, 6.72, "Independent reservoir input", 12.8, AMBER, "semibold")
    text(ax, center, 6.34, FERMI, 16.5)
    note(ax, center, 6.06, "Fermi–Dirac occupation", color=AMBER, size=11.3)

    box(ax, x, 4.2, width, 1.1)
    text(ax, center, 5.02, "Contact-fed correlation", 13.3, weight="semibold")
    text(ax, center, 4.58, OCCUPIED, 18)
    arrow(ax, [(center, 5.85), (center, 5.3)], AMBER)
    note(ax, 3.22, 5.58, r"$f_\alpha$", color=AMBER, size=13)
    arrow(ax, [(x, 8.65), (0.15, 8.65), (0.15, 4.75), (x, 4.75)])
    note(ax, 0.13, 6.56, r"$A_\alpha$", color=BLUE, size=13, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    box(ax, x, 2.95, width, 0.65)
    text(ax, center, 3.28, "Density and bond current", 13.3, weight="semibold")
    arrow(ax, [(center, 4.2), (center, 3.6)])

    box(ax, x, 0.95, width, 1.35)
    text(ax, center, 1.98, "Landauer current", 13.3, weight="semibold")
    text(ax, center, 1.49, CURRENT, 14)
    # Separate right-hand buses bring T(E) and reservoir imbalance to I;
    # they terminate at different ports and never join the Gn/density path.
    arrow(ax, [(4.88, 7.83), (5.24, 7.83), (5.24, 1.35), (4.88, 1.35)])
    note(ax, 5.23, 3.1, r"$T(E)$", color=BLUE, size=12.5, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    arrow(ax, [(4.88, 6.34), (5.02, 6.34), (5.02, 1.87), (4.88, 1.87)], AMBER, width=1.25, head=8)
    note(ax, 4.99, 4.0, r"$f_L-f_R$", color=AMBER, size=12, rotation=90,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    note(ax, center, 0.57, r"$\alpha=L,R\qquad G^A=(G^R)^\dagger$", size=12.4)
    note(ax, center, 0.23, "Spin degeneracy: 2", size=11.3)
    return fig


def save(fig, directory: Path, stem: str):
    for extension in ("svg", "png"):
        filename = directory / f"{stem}.{extension}"
        fig.savefig(filename, dpi=200, facecolor="white", metadata={
            "Title": "NEGF: dynamics and occupations",
            "Description": "Hamiltonian and lead inputs determine spectral quantities; independent reservoir occupations determine occupied correlations and current.",
        } if extension == "svg" else {"Title": "NEGF: dynamics and occupations"})
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
