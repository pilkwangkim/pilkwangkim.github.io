#!/usr/bin/env python3
"""Reproduce NEGF figures 1–4 with article-size desktop/mobile typography.

Data and equations follow cells 2, 4, 6, 10, 17 and 19 of:
https://github.com/pilkwangkim/Physics/blob/7fd9f32c390e117ab6b04bb7f787fc5b43c959e2/negf_from_scratch.ipynb

Requires NumPy and Matplotlib. No notebook execution or modification is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/img/posts/2026-07-22-negf-from-scratch"
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
J_PER_EV = 1.602176634e-19
T0 = HBAR**2 / (2 * 0.067 * M0 * (1.0e-9)**2) / J_PER_EV
EPS = 2 * T0
N = 40
BLUE = "#1F4E79"
DEVICE = "#2B6F9F"
LEAD = "#929BA3"
AMBER = "#B17B12"
INK = "#253745"
MUTED = "#586772"
RED = "#C0392B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "mathtext.fontset": "stix",
    "svg.fonttype": "path",
    "axes.labelcolor": INK,
    "axes.edgecolor": "#71808C",
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.4,
    "savefig.facecolor": "white",
})


def hamiltonian(count=N):
    result = np.diag(np.full(count, 2 * T0)).astype(complex)
    result += np.diag(np.full(count - 1, -T0), 1)
    result += np.diag(np.full(count - 1, -T0), -1)
    return result


def surface_green(energies, eta=1e-9):
    b = np.asarray(energies) + 1j * eta - EPS
    discriminant = np.sqrt(b * b - 4 * T0**2 + 0j)
    g1 = (b + discriminant) / (2 * T0**2)
    g2 = (b - discriminant) / (2 * T0**2)
    fallback = np.where(abs(T0 * g1) < abs(T0 * g2), g1, g2)
    return np.where(g1.imag < 0, g1, np.where(g2.imag < 0, g2, fallback))


def style_axes(ax, mobile=False):
    ax.tick_params(labelsize=15.5 if mobile else 16, length=4.5, pad=5)
    ax.xaxis.label.set_size(18 if mobile else 18.5)
    ax.yaxis.label.set_size(18 if mobile else 18.5)
    ax.grid(alpha=0.2, color="#8B9BA7", linewidth=0.75)
    ax.set_axisbelow(True)


def finish(fig, stem, output):
    fig.canvas.draw()
    extent = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    if (extent.x0 < -0.01 or extent.y0 < -0.01
            or extent.x1 > width + 0.01 or extent.y1 > height + 0.01):
        raise RuntimeError(f"Content outside {stem}: {extent.bounds}; canvas={(width, height)}")
    for extension in ("svg", "png"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    plt.close(fig)


def lattice_dispersion(mobile=False):
    ka = np.linspace(-np.pi, np.pi, 601)
    energy_tb = 2 * T0 * (1 - np.cos(ka))
    energy_parabola = T0 * ka**2
    n = np.arange(1, N + 1)
    ka_n = n * np.pi / (N + 1)
    energies_n = np.linalg.eigvalsh(hamiltonian().real)
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5, 11.4))
        fig.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.13, hspace=1.0)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.6))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.81, bottom=0.28, wspace=0.37)
    a, b = axes
    a.plot(ka / np.pi, energy_tb, color=BLUE, lw=2.8,
           label=r"$E(k)=2t_0(1-\cos ka)$")
    a.plot(ka / np.pi, energy_parabola, "--", color="crimson", lw=2.2,
           label=r"$\hbar^2k^2/(2m^*)$")
    a.axhline(4 * T0, color="#7F8B93", ls=":", lw=1.25)
    a.text(0, 4 * T0 * 1.035, r"Bandwidth $4t_0$", fontsize=15.5 if mobile else 16,
           color=MUTED, ha="center", va="bottom")
    a.set(xlim=(-1, 1), ylim=(-0.12 * T0, 4.45 * T0), xlabel=r"$ka/\pi$", ylabel="Energy (eV)")
    a.set_title("(a) Finite-difference\nlattice dispersion", fontsize=18 if mobile else 19.5, pad=15)
    a.set_xticks([-1, -0.5, 0, 0.5, 1])
    mask = ka >= 0
    b.plot(ka[mask] / np.pi, energy_tb[mask], color="#A4ADB4", lw=2.1, label="Analytic band")
    b.scatter(ka_n / np.pi, energies_n, s=37 if mobile else 43,
              color=RED, zorder=3, label=r"Diagonalized ($N=40$)")
    b.set(xlim=(0, 1.02), ylim=(-0.12 * T0, 4.45 * T0), xlabel=r"$ka/\pi$", ylabel="Energy (eV)")
    b.set_xticks([0, 0.25, 0.5, 0.75, 1])
    b.set_title("(b) Finite chain\nsamples the band", fontsize=18 if mobile else 19.5, pad=15)
    for ax in axes:
        style_axes(ax, mobile)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23),
                  fontsize=15.5 if mobile else 16, frameon=False,
                  handlelength=2.2, borderaxespad=0, labelspacing=0.65)
    return fig


def eigenstates(mobile=False):
    n = np.arange(1, N + 1)
    ka_n = n * np.pi / (N + 1)
    energies = 2 * T0 * (1 - np.cos(ka_n))
    j = np.arange(1, N + 1)
    modes = np.sqrt(2 / (N + 1)) * np.sin(np.outer(j, ka_n[:5]))
    amplitude = 0.30 * np.min(np.diff(energies[:5]))
    colors = plt.cm.viridis(np.linspace(0.16, 0.88, 5))
    if mobile:
        fig, ax = plt.subplots(figsize=(5, 10.8))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.77, bottom=0.35)
        fig.text(0.5, 0.96, "Closed tight-binding chain", fontsize=18, ha="center", color=INK)
        fig.text(0.5, 0.91, r"$U_j=0,\quad\psi_0=\psi_{N+1}=0$", fontsize=17, ha="center", color=INK)
        fig.text(0.5, 0.85, r"Solid: $E_n+s\widetilde{\psi}_n(j)$" + "\n"
                 + rf"$s={amplitude:.4f}$ eV (display only)", fontsize=16,
                 ha="center", va="center", color=MUTED, linespacing=1.45)
    else:
        fig, ax = plt.subplots(figsize=(10.8, 9.2))
        fig.subplots_adjust(left=0.11, right=0.98, top=0.79, bottom=0.32)
        fig.text(0.5, 0.95, "Closed tight-binding chain", fontsize=21, ha="center", color=INK)
        fig.text(0.5, 0.9, r"$U_j=0,\quad\psi_0=\psi_{N+1}=0$", fontsize=18, ha="center", color=INK)
        fig.text(0.5, 0.85, r"Solid: $E_n+s\widetilde{\psi}_n(j)$; "
                 + rf"$s={amplitude:.4f}$ eV (display only)", fontsize=17,
                 ha="center", color=MUTED)
    ax.plot(j, np.zeros(N), color="#3E4850", lw=2.6)
    ax.axvline(0, color="#677079", lw=2.6)
    ax.axvline(N + 1, color="#677079", lw=2.6)
    handles = []
    for index, (energy, color) in enumerate(zip(energies[:5], colors), start=1):
        psi = np.r_[0.0, modes[:, index - 1], 0.0]
        psi /= np.max(abs(psi))
        ax.hlines(energy, 0, N + 1, color=color, ls=":", lw=1.4)
        handle, = ax.plot(np.arange(N + 2), energy + amplitude * psi, color=color, lw=2.8,
                          label=rf"$n={index},\ E_n={energy:.4f}$ eV")
        handles.append(handle)
        ax.text(N + 1.5, energy, rf"$n={index}$", fontsize=16 if mobile else 18,
                color=INK, va="center")
    ax.set(xlim=(-0.8, N + 7.2), ylim=(-0.0085, energies[4] + 0.012),
           xlabel=r"Site index $j$", ylabel="Energy with signed state offset (eV)")
    if mobile:
        ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
        ax.set_xticks([0, 10, 20, 30, 40])
    style_axes(ax, mobile)
    ax.grid(axis="x", visible=False)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035 if mobile else 0.045),
               ncol=1 if mobile else 2, fontsize=15.5 if mobile else 16.5,
               frameon=False, labelspacing=0.7, columnspacing=2.0, handlelength=2.5)
    return fig


def diagram_axes(fig, rectangle, width, ymax=4.4, ymin=-0.85):
    ax = fig.add_axes(rectangle)
    ax.set(xlim=(0, width), ylim=(ymin, ymax))
    ax.axis("off")
    return ax


def diagram_text(ax, x, y, content, size=18, color=INK, **kwargs):
    return ax.text(x, y, content, ha=kwargs.pop("ha", "center"),
                   va=kwargs.pop("va", "center"), fontsize=size,
                   color=color, zorder=4, **kwargs)


def diagram_arrow(ax, start, end, color=MUTED, width=1.7, head=11):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                color=color, linewidth=width, mutation_scale=head,
                                shrinkA=0, shrinkB=0, zorder=3))


def chain(ax, xs, y, color, labels, radius=0.13, size=18):
    ax.plot(xs, np.full(len(xs), y), color=color, lw=2.6, zorder=1)
    for x, label in zip(xs, labels):
        ax.add_patch(Circle((x, y), radius, facecolor=color,
                            edgecolor="white", lw=1.1, zorder=2))
        diagram_text(ax, x, y - 0.42, label, size=size)


def before_elimination(ax, width, small=False):
    size = 16 if small else 18
    diagram_text(ax, width / 2, 4.02, "(a) Before elimination", size=18 if small else 21, weight="bold")
    diagram_text(ax, width / 2, 3.57, "Device + all lead coordinates", size=16.5 if small else 18)
    xs = np.linspace(0.65 if small else 1.05, width - (0.45 if small else 1.05), 8)
    lead_x, device_x = xs[:4], xs[4:]
    chain(ax, lead_x, 1.1, LEAD, [r"$\ell_3$", r"$\ell_2$", r"$\ell_1$", r"$\ell_0$"], size=size)
    chain(ax, device_x, 1.1, DEVICE, [r"$d_0$", r"$d_1$", r"$d_2$", r"$d_3$"], size=size)
    ax.plot([lead_x[-1], device_x[0]], [1.1, 1.1], color=AMBER, lw=4.1, zorder=1)
    diagram_text(ax, (lead_x[-1] + device_x[0]) / 2, 1.65, r"Interface $\tau$", size=size, color=AMBER)
    diagram_text(ax, lead_x[0] - 0.34, 1.1, r"$\cdots$", size=22)
    diagram_text(ax, np.mean(lead_x), 2.91, "Semi-infinite\nlead $L$", size=size, color=MUTED, linespacing=1.3)
    diagram_text(ax, np.mean(device_x), 2.91, "Retained\ndevice $D$", size=size, color=DEVICE, linespacing=1.3)
    source_x = device_x[1]
    diagram_text(ax, source_x, 2.13, "Device source", size=15.5 if small else 18)
    diagram_arrow(ax, (source_x, 1.87), (source_x, 1.24))


def after_elimination(ax, width, small=False):
    size = 16 if small else 18
    diagram_text(ax, width / 2, 4.02, "(b) After elimination", size=18 if small else 21, weight="bold")
    d0 = 2.0 if small else 4.2
    spacing = 0.83 if small else 1.5
    xs = d0 + spacing * np.arange(4)
    y_device = 1.05
    chain(ax, xs, y_device, DEVICE, [r"$d_0$", r"$d_1$", r"$d_2$", r"$d_3$"],
          radius=0.15, size=size)
    response_width = 3.65 if small else 5.0
    left, bottom, height = d0 - response_width / 2, 2.65, 0.8
    ax.add_patch(FancyBboxPatch((left, bottom), response_width, height,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#F2F4F5", edgecolor=LEAD, lw=1.4, zorder=2))
    diagram_text(ax, d0, 3.07, "Eliminated lead response\n$g_L^R(E)$",
                 size=16 if small else 19, linespacing=1.15)
    # Arrows end exactly on the response-box edge and the contacted site.
    x_offset = 0.11 if small else 0.15
    site_edge_y = y_device + np.sqrt(0.15**2 - x_offset**2) if small else y_device
    if not small:
        x_offset = 0.09
        site_edge_y = y_device + np.sqrt(0.15**2 - x_offset**2)
    diagram_arrow(ax, (d0 - x_offset, site_edge_y), (d0 - x_offset, bottom), AMBER, head=10 if small else 12)
    diagram_arrow(ax, (d0 + x_offset, bottom), (d0 + x_offset, site_edge_y), AMBER, head=10 if small else 12)
    diagram_text(ax, d0 - (0.28 if small else 0.4), 1.98,
                 r"$D\to L:\ \tau^\dagger$", size=size, color=AMBER, ha="right")
    diagram_text(ax, d0 + (0.28 if small else 0.4), 1.98,
                 r"$L\to D:\ \tau$", size=size, color=AMBER, ha="left")
    formula = r"$\Sigma_L^R(E)=\tau g_L^R(E)\tau^\dagger$"
    diagram_text(ax, width / 2, -0.06, formula, size=18 if small else 22)
    diagram_text(ax, width / 2, -0.6, "Contacted device subspace only",
                 size=15.5 if small else 18, color=MUTED)


def recursive_surface(ax, width, small=False):
    size = 16 if small else 18
    diagram_text(ax, width / 2, 4.22, "(c) Surface self-similarity", size=18 if small else 21, weight="bold")
    diagram_text(ax, width / 2, 3.47,
                 r"$g_s^R=[E^+-\varepsilon-t_0^2g_s^R]^{-1}$",
                 size=18 if small else 23,
                 bbox={"boxstyle": "round,pad=0.32", "facecolor": "#F8FAFB", "edgecolor": LEAD})
    xs = np.array([0.75, 1.75, 2.75, 3.75]) if small else np.array([1.7, 4.0, 6.3, 8.6])
    chain(ax, xs, 2.12, LEAD, [r"$\ell_0$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$"], size=size)
    ax.plot(xs[:2], [2.12, 2.12], color=AMBER, lw=4.0, zorder=1)
    midpoint = np.mean(xs[:2])
    diagram_text(ax, midpoint, 1.08, r"Separate $\ell_0$", size=size)
    diagram_arrow(ax, (midpoint, 1.36), (midpoint, 2.12))
    chain(ax, xs[1:], 0.38, LEAD, [r"$\ell_1$", r"$\ell_2$", r"$\ell_3$"], size=size)
    for y in [2.12, 0.38]:
        diagram_text(ax, xs[-1] + (0.56 if small else 0.9), y, r"$-\cdots$", size=20 if small else 24)
    diagram_text(ax, width / 2, -0.61, "Remaining tail: the same\nsemi-infinite lead" if small
                 else "Remaining tail: the same semi-infinite lead", size=16.5 if small else 18,
                 color=MUTED, linespacing=1.3)


def elimination(mobile=False, cover=False):
    if cover:
        # The 358 px cover keeps only the before/after topology and equation;
        # node indices and the detailed response explanation are in figure 3.
        fig = plt.figure(figsize=(7, 5.2), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 7), ylim=(0, 5.2))
        ax.axis("off")
        diagram_text(ax, 1.75, 4.72, "Before", size=28, weight="bold")
        diagram_text(ax, 5.25, 4.72, "After", size=28, weight="bold")
        ax.plot([3.5, 3.5], [1.32, 4.42], color="#DCE3E8", lw=0.9)
        lead_x = np.array([0.65, 1.15, 1.65])
        device_x = np.array([2.2, 2.7, 3.2])
        y_device = 2.05
        for xs, color in [(lead_x, LEAD), (device_x, DEVICE)]:
            ax.plot(xs, np.full(len(xs), y_device), color=color, lw=3.0, zorder=1)
            for x in xs:
                ax.add_patch(Circle((x, y_device), 0.105, facecolor=color,
                                    edgecolor="white", lw=1.0, zorder=2))
        ax.plot([lead_x[-1], device_x[0]], [y_device, y_device], color=AMBER, lw=4, zorder=1)
        diagram_text(ax, 0.2, y_device, r"$\cdots$", size=26)
        diagram_text(ax, 1.15, 3.35, "Lead", size=24, color=MUTED)
        diagram_text(ax, 2.7, 3.35, "Device", size=24, color=DEVICE)
        diagram_text(ax, 1.925, 2.55, r"$\tau$", size=28, color=AMBER)

        d0 = 4.65
        after_x = np.array([d0, 5.45, 6.25])
        ax.plot(after_x, np.full(3, y_device), color=DEVICE, lw=3.0, zorder=1)
        for x in after_x:
            ax.add_patch(Circle((x, y_device), 0.12, facecolor=DEVICE,
                                edgecolor="white", lw=1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((3.8, 3.0), 1.7, 1.15,
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor="#F2F4F5", edgecolor=LEAD, lw=1.4, zorder=2))
        diagram_text(ax, d0, 3.85, "Lead", size=24, color=MUTED)
        diagram_text(ax, d0, 3.3, r"$g_L^R(E)$", size=28)
        offset = 0.075
        site_edge = y_device + np.sqrt(0.12**2 - offset**2)
        diagram_arrow(ax, (d0 - offset, site_edge), (d0 - offset, 3.0), AMBER, head=10)
        diagram_arrow(ax, (d0 + offset, 3.0), (d0 + offset, site_edge), AMBER, head=10)
        diagram_text(ax, 4.3, 2.55, r"$\tau^\dagger$", size=27, color=AMBER, ha="right")
        diagram_text(ax, 5.0, 2.55, r"$\tau$", size=27, color=AMBER, ha="left")
        diagram_text(ax, 5.45, 1.6, "Device", size=24, color=DEVICE)
        ax.plot([0.4, 6.6], [1.1, 1.1], color="#DCE3E8", lw=0.9)
        diagram_text(ax, 3.5, 0.55, r"$\Sigma_L^R(E)=\tau g_L^R(E)\tau^\dagger$", size=30)
        return fig
    if mobile:
        height = 15.7
        fig = plt.figure(figsize=(5, height), facecolor="white")
        rectangles = [[0.02, 11.3 / height, 0.96, 4.2 / height],
                      [0.02, 5.65 / height, 0.96, 5.3 / height],
                      [0.02, 0.2 / height, 0.96, 5.1 / height]]
        width = 5.0
    else:
        height = 14.6
        fig = plt.figure(figsize=(10.8, height), facecolor="white")
        rectangles = [[0.03, 10.3 / height, 0.94, 4.2 / height],
                      [0.03, 5.2 / height, 0.94, 4.75 / height],
                      [0.03, 0.2 / height, 0.94, 4.65 / height]]
        width = 10.8
    for function, rectangle in zip([before_elimination, after_elimination, recursive_surface], rectangles):
        ax = diagram_axes(fig, rectangle, width, ymax=4.6 if function is recursive_surface else 4.4,
                          ymin=0.25 if function is before_elimination else
                          (-1.05 if function is recursive_surface else -0.85))
        function(ax, width, small=mobile)
    return fig


def contact_data():
    energies = np.linspace(-0.25, 4 * T0 + 0.25, 700)
    sigma = T0**2 * surface_green(energies)
    broadening = -2 * sigma.imag
    dos_open = np.empty_like(energies)
    dos_isolated = np.empty_like(energies)
    identity = np.eye(N)
    h_device = hamiltonian()
    for index, (energy, contact) in enumerate(zip(energies, sigma)):
        contacted = (energy + 1e-9j) * identity - h_device
        contacted[0, 0] -= contact
        contacted[-1, -1] -= contact
        isolated = (energy + 0.012j) * identity - h_device
        dos_open[index] = -np.trace(np.linalg.inv(contacted)).imag / np.pi
        dos_isolated[index] = -np.trace(np.linalg.inv(isolated)).imag / np.pi
    q_segments = [-T0 * surface_green(values, eta=0) for values in [
        np.linspace(energies[0], 0, 160),
        np.linspace(0, 4 * T0, 360),
        np.linspace(4 * T0, energies[-1], 160),
    ]]
    return energies, sigma.real, broadening, dos_open, dos_isolated, q_segments


def surface_response(data, mobile=False):
    energies, re_sigma, gamma, dos_open, dos_isolated, q_segments = data
    fig, axes = plt.subplots(3, 1, figsize=(5, 20.2) if mobile else (10.8, 19.5))
    fig.subplots_adjust(left=0.23 if mobile else 0.14, right=0.97 if mobile else 0.96,
                        top=0.95, bottom=0.12, hspace=0.70 if mobile else 0.58)
    a, b, c = axes
    a.fill_between(energies, np.maximum(dos_open, 1e-2), 1e-2, color=BLUE, alpha=0.12)
    a.plot(energies, dos_isolated, color="#8A949D", lw=1.5, label="Isolated (hard wall)")
    a.plot(energies, dos_open, color=BLUE, lw=2.7, label=r"Contacted ($\Sigma_L,\Sigma_R$)")
    a.set(xlim=(energies[0], energies[-1]), ylim=(1e-2, 1.25 * max(dos_open.max(), dos_isolated.max())),
          xlabel="Energy E (eV)", ylabel="Projected DOS per spin (1/eV)" if mobile else
          "Device-projected DOS\nper spin (1/eV)", yscale="log")
    a.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    a.set_title("(a) Hard-wall levels and\ncontacted continuum" if mobile else
                "(a) Hard-wall levels versus contacted continuum", fontsize=18 if mobile else 19.5, pad=15)
    a.legend(fontsize=15.5 if mobile else 16, loc="upper center", bbox_to_anchor=(0.5, -0.22),
             ncol=1 if mobile else 2, frameon=False, borderaxespad=0)
    b.axvspan(0, 4 * T0, color="gold", alpha=0.13)
    b.plot(energies, gamma, color=RED, lw=2.8, label=r"$\gamma=-2\,\mathrm{Im}\,\sigma$")
    b.plot(energies, re_sigma, "--", color="#2E7D32", lw=2.3,
           label=r"$\mathrm{Re}\,\sigma$ (level shift)")
    b.axhline(0, color="#9AA5AD", lw=0.9)
    b.text(2.7 * T0, -0.36, "Lead band\n(propagating)", fontsize=15.5 if mobile else 16,
           ha="center", va="center", color=MUTED, linespacing=1.15)
    span = energies[-1] - energies[0]
    y_min = min(gamma.min(), re_sigma.min(), 0)
    y_max = max(gamma.max(), re_sigma.max(), 0)
    b.set(xlim=(energies[0] - 0.05 * span, energies[-1] + 0.05 * span),
          ylim=(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)),
          xlabel="Energy E (eV)", ylabel="Self-energy (eV)")
    b.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    b.set_title(r"(b) Boundary self-energy" + "\n" + r"$\sigma=-t_0e^{ika}$",
                fontsize=18 if mobile else 19.5, pad=15)
    b.legend(fontsize=15.5 if mobile else 16, loc="upper center", bbox_to_anchor=(0.5, -0.22),
             ncol=1 if mobile else 2, frameon=False, borderaxespad=0)
    angle = np.linspace(0, 2 * np.pi, 400)
    c.plot(np.cos(angle), np.sin(angle), color="#B7C0C6", ls=":", lw=1.4, label=r"$|q|=1$")
    for q, color, label in zip(q_segments, ["#2E7D32", BLUE, "#7B1FA2"],
                               ["Below band: decaying", "Inside band: outgoing", "Above band: decaying"]):
        c.plot(q.real, q.imag, color=color, lw=3, label=label)
    for index, q in enumerate([q_segments[1], q_segments[0], q_segments[2]]):
        reciprocal = 1 / q
        mask = abs(reciprocal.real) <= 2.05
        c.plot(reciprocal.real[mask], reciprocal.imag[mask], color=RED, ls="--", lw=1.7, alpha=0.8,
               label=r"Other root $q^{-1}$" if index == 0 else None)
    q_landmarks = -T0 * surface_green(np.array([0, 2 * T0, 4 * T0]), eta=0)
    c.scatter(q_landmarks.real, q_landmarks.imag, s=42, color=BLUE, zorder=4)
    c.text(1.08, 0.11, r"$E=0$", fontsize=15.5 if mobile else 17)
    c.text(0, 0.63, r"$E=2t_0$", fontsize=15.5 if mobile else 17, ha="center", va="center")
    c.text(-1.88 if mobile else -1.78, 0.11, r"$E=4t_0$", fontsize=15.5 if mobile else 17)
    c.axhline(0, color="#A9B3BB", lw=0.8)
    c.axvline(0, color="#A9B3BB", lw=0.8)
    c.set(xlim=(-2.05, 2.05), ylim=(-1.18, 1.18), xlabel=r"$\mathrm{Re}\,q$", ylabel=r"$\mathrm{Im}\,q$")
    c.set_xticks([-2, -1, 0, 1, 2])
    c.set_yticks([-1, 0, 1])
    c.set_aspect("equal", adjustable="box")
    c.set_title(r"(c) Retarded root of $q=-t_0g_s^R$", fontsize=17.5 if mobile else 19.5, pad=15)
    c.legend(fontsize=15.2 if mobile else 16, loc="upper center", bbox_to_anchor=(0.5, -0.23),
             ncol=1 if mobile else 2, frameon=False, borderaxespad=0, labelspacing=0.6)
    for ax in axes:
        style_axes(ax, mobile)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = contact_data()
    plots = [
        ("fig-01-lattice-dispersion", lattice_dispersion),
        ("fig-02-closed-chain-eigenstates", eigenstates),
        ("fig-03-exact-lead-elimination", elimination),
        ("fig-04-surface-green-function", lambda mobile: surface_response(data, mobile)),
    ]
    for stem, function in plots:
        finish(function(False), stem, args.output_dir)
        finish(function(True), stem + "-mobile", args.output_dir)
    finish(elimination(cover=True), "cover-exact-lead-elimination", args.output_dir)
    n = np.arange(1, N + 1)
    exact = 2 * T0 * (1 - np.cos(n * np.pi / (N + 1)))
    print(f"t0={T0:.16f} eV; eigenvalue max error={np.max(abs(np.linalg.eigvalsh(hamiltonian().real) - exact)):.3e} eV")
    print(f"Contact DOS range={data[3].min():.6e}..{data[3].max():.6e} 1/eV; gamma_max={data[2].max():.9f} eV")


if __name__ == "__main__":
    main()
