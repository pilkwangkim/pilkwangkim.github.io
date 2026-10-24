#!/usr/bin/env python3
"""Replot NEGF Figures 5–8 at article-readable desktop and mobile sizes.

Requires NumPy and Matplotlib. The notebook is read-only; only the required
function/geometry/sweep portions of cells 2,17,19,23,24,27,32 are executed.
Example: python tools/negf_transport_figures.py --notebook negf_from_scratch.ipynb
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/img/posts/2026-07-22-negf-from-scratch"
SOURCE_SHA256 = "2918fa2f8a8e0b20762fffa1b4b87d96a4cab67800e06fc0b94a79ce8b607cfa"
BLUE, RED, PURPLE = "#1f4e79", "#c0392b", "#7d3c98"


def notebook_data(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Notebook differs from the audited 7fd9f32 source")
    notebook = json.loads(raw)
    source = lambda i: "".join(notebook["cells"][i]["source"])
    scope = {"np": np, "plt": plt, "LogNorm": LogNorm}
    pieces = [
        "\n".join(line for line in source(2).splitlines()
                  if not line.startswith(("%", "from scipy"))),
        source(17), source(19).split("# ---- same finite segment")[0],
        source(23).split("fig_geometry_transport,")[0],
        source(24).split("fig_transport,")[0],
        source(27).split("fig_tunnel,")[0],
        source(32).split("# Plot U_j")[0],
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        for piece in pieces:
            exec(compile(piece, str(path), "exec"), scope)
    return scope


def style(mobile):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15 if mobile else 16,
        "axes.titlesize": 17 if mobile else 19,
        "axes.labelsize": 17 if mobile else 19,
        "xtick.labelsize": 15 if mobile else 16,
        "ytick.labelsize": 15 if mobile else 16,
        "legend.fontsize": 15 if mobile else 16,
        "mathtext.fontset": "stix", "svg.fonttype": "path",
        "svg.hashsalt": "negf-transport-figures", "savefig.facecolor": "white",
        "axes.titlepad": 14, "axes.labelpad": 9,
        "figure.constrained_layout.h_pad": .12,
        "figure.constrained_layout.w_pad": .08,
    })


def pair(mobile, heights=None):
    if mobile:
        fig = plt.figure(figsize=(5, 12.6), layout="constrained")
        grid = fig.add_gridspec(4, 1, height_ratios=heights or [3.5, 1.9, 3.5, 1.5])
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[2])]
        notes = [fig.add_subplot(grid[1]), fig.add_subplot(grid[3])]
    else:
        fig = plt.figure(figsize=(10.8, 7.5), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=heights or [4.6, 1.7])
        axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
        notes = [fig.add_subplot(grid[1, i]) for i in range(2)]
    for note in notes:
        note.axis("off")
    return fig, axes, notes


def key(ax, note, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    note.legend(handles, labels, loc="center", frameon=False, **kwargs)


def fig05(d, mobile):
    fig, (a, b), (na, nb) = pair(mobile)
    sites, u = d["sites_transport"], d["U_barrier_transport"]
    a.step(sites, d["U_clean_transport"], where="mid", color="0.45", ls="--", lw=2,
           label="Clean device / leads: $U=0$")
    a.step(sites, u, where="mid", color=RED, lw=2.4, label=r"Barrier: $V_b=0.60$ eV")
    a.axhline(.4, color=BLUE, ls="-.", lw=2, label=r"Probe: $E_{\mathrm{probe}}=0.40$ eV")
    a.fill_between([26.5, 32.5], [.4, .4], [.6, .6], color="#f39c12", alpha=.28,
                   label=r"$E_{\mathrm{probe}}<U$: no real local $k$")
    a.axvspan(26.5, 32.5, color="crimson", alpha=.07)
    a.annotate("Evanescent\nregion", xy=(29.5, .50), xytext=(5, .65),
               arrowprops={"arrowstyle": "->", "color": "0.25"}, fontsize=15 if mobile else 16)
    a.set(xlabel="Device site $j$", ylabel=r"$U_j$ and $E_{\mathrm{probe}}$ (eV)",
          title="(a) Sub-barrier probe energy", xlim=(-.5, 59.5), ylim=(-.06, .75))
    a.grid(alpha=.20)
    key(a, na)
    matrix = d["H_barrier_transport"].real
    scale = abs(matrix).max()
    image = b.imshow(matrix, origin="lower", interpolation="nearest", cmap="RdBu_r",
                      vmin=-scale, vmax=scale, aspect="equal", rasterized=True)
    b.plot([26.5, 32.5, 32.5, 26.5, 26.5], [26.5, 26.5, 32.5, 32.5, 26.5],
            color="#f1c40f", lw=2.2)
    b.annotate("Six raised\n$H_{jj}$ entries", xy=(32, 32), xytext=(4, 49),
               arrowprops={"arrowstyle": "->", "color": "#7d6608"}, color="#5d4e00",
               fontsize=15 if mobile else 16)
    b.set(xlabel="Column site $j$", ylabel="Row site $i$", title=r"(b) Bare device matrix $[H_D]_{ij}$")
    bar = fig.colorbar(image, ax=b, fraction=.055, pad=.04)
    bar.set_label(r"$H_{ij}$ (eV)")
    nb.text(.5, .5, (r"Main diagonal: $2t_0+U_j$" "\n"
                      "Adjacent diagonals:\n" rf"$-t_0={-d['t0']:.3f}$ eV"),
             transform=nb.transAxes, ha="center", va="center", linespacing=1.45)
    return fig


def fig06(d, mobile):
    fig, (a, b), (na, nb) = pair(mobile)
    energy = d["E_grid_transport"]
    a.axhline(1, color=".7", ls=":", lw=1)
    a.plot(energy, d["T_clean_transport"], color=BLUE, lw=2.4, label="Clean wire ($U=0$)")
    a.plot(energy, d["T_barrier_transport"], color=RED, lw=2,
            label="Single barrier\n(0.60 eV, 6 sites)")
    a.axvline(.4, color=PURPLE, ls="--", lw=1.3, label=r"Probe: $E_{\mathrm{probe}}=0.400$ eV")
    a.scatter(.4, d["T_tunnel"], color=PURPLE, zorder=4)
    a.set(xlabel="E (eV)", ylabel="T(E)", xlim=(0, 4*d["t0"]), ylim=(0, 1.12),
          title="(a) Matched wire versus\na single barrier" if mobile else "(a) Matched wire versus a barrier")
    a.grid(alpha=.25)
    key(a, na)
    rho = d["rho_L_barrier_transport"]
    positive = rho[rho > 0]
    vmax = np.percentile(positive, 99.5)
    vmin = max(np.percentile(positive, 1), 1e-5*vmax)
    image = b.pcolormesh(d["sites_transport"], energy, np.ma.masked_less_equal(rho.T, 0),
                         shading="auto", cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
                         rasterized=True)
    for edge in (26.5, 32.5):
        b.axvline(edge, color="cyan", lw=1, alpha=.85)
    b.axhline(.4, color="#b084cc", ls="--", lw=1.6, label=r"Probe slice: $E_{\mathrm{probe}}=0.400$ eV")
    b.plot([26.5, 32.5], [.6, .6], color="#ff6b6b", ls=":", lw=1.5,
            label="Barrier-local band bottom\n$V_b=0.60$ eV")
    b.set(xlabel="Device site $j$", ylabel="E (eV)", xlim=(-.5, 59.5),
          ylim=(energy[0], energy[-1]),
          title=r"(b) Left-injected partial LDOS" + "\n" + r"$\rho_{L,j}(E)$")
    bar = fig.colorbar(image, ax=b, fraction=.055, pad=.04)
    bar.set_label(r"$\rho_{L,j}$ per spin (1/eV)")
    key(b, nb)
    return fig


def fig07(d, mobile):
    fig, (a, b), (na, nb) = pair(mobile)
    rho = d["rho_L_tunnel"]
    a.semilogy(d["sites_transport"], rho, color=BLUE, lw=2.2)
    a.axvspan(26.5, 32.5, color="crimson", alpha=.14,
               label=r"Barrier: $U=0.60$ eV" + "\n" + r"$>E_{\mathrm{probe}}=0.40$ eV")
    for edge in (26.5, 32.5):
        a.axvline(edge, color="crimson", lw=.8)
    fs = 15 if mobile else 16
    mid = np.sqrt(rho[27]*rho[32])
    a.annotate("Evanescent\nsuppression", xy=(29.5, mid), xytext=(35, rho.max()*.09),
               arrowprops={"arrowstyle": "->", "color": ".25"}, fontsize=fs)
    a.annotate("Small right-going\ntransmitted\ncomponent", xy=(48, np.median(rho[36:])),
               xytext=(59, 2.5*np.median(rho[36:])), ha="right", va="bottom", fontsize=fs,
               arrowprops={"arrowstyle": "->", "color": ".25"})
    a.set(xlabel="Device site $j$", ylabel=r"$|\psi_{L,j}|^2/(2\pi)$ per spin (1/eV)",
          title="(a) Spectral weight\n" + r"$E_{\mathrm{probe}}=0.400$ eV", xlim=(-.5, 59.5))
    a.grid(alpha=.20, which="both")
    handles, labels = a.get_legend_handles_labels()
    handles.insert(0, Line2D([], [], color=BLUE, lw=2.2))
    labels.insert(0, "Left of barrier: incident +\nreflected (interference)")
    na.legend(handles, labels, loc="center", frameon=False)
    flux, transmission = d["bond_flux_L_tunnel"], d["T_tunnel"]
    b.plot(d["bond_positions_tunnel"], flux, color="#2e8b57", lw=2.2, marker=".", ms=4,
            label="Left-injected spectral\nbond flux")
    b.axhline(transmission, color=PURPLE, ls="--", lw=1.8,
               label=(r"Caroli $T(E_{\mathrm{probe}})$" "\n"
                      rf"$={transmission*1e3:.4f}\times10^{{-3}}$"))
    b.axvspan(26.5, 32.5, color="crimson", alpha=.14, label="Barrier span")
    b.text(29.5, .18*transmission, "Same flux\nthrough barrier", ha="center", fontsize=fs)
    b.set(xlabel=r"Bond centre $j+1/2$", ylabel=r"Dimensionless $\mathcal{J}_{j\to j+1}^{(L)}(E_{\mathrm{probe}})$",
          title="(b) Conserved tunneling flux\n(not an ampere current)", xlim=(0, 59), ylim=(0, 1.25*transmission))
    b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    b.grid(alpha=.22)
    key(b, nb)
    return fig


def fig08(d, mobile):
    if mobile:
        fig = plt.figure(figsize=(5, 5.5), layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=[4.5, .7])
        ax, note = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        note.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(10.8, 4.0), layout="constrained")
    ax.step(np.arange(d["N_spec"]), d["U_spec"], where="mid", color=BLUE, lw=2.1)
    for lo, hi in ((39, 42), (58, 61)):
        ax.axvspan(lo-.5, hi-.5, color="crimson", alpha=.15,
                    label="Barriers" if lo == 39 else None)
    ax.axvspan(41.5, 57.5, color="gold", alpha=.15, label="Well")
    ax.set(xlabel="Site $j$", ylabel="$U_j$ (eV)",
           title="Example A: fixed\ndouble-barrier potential" if mobile else "Example A: fixed double-barrier potential",
           xlim=(-.5, 99.5), ylim=(-.03, .47))
    if mobile:
        key(ax, note, ncol=2)
    else:
        ax.legend(loc="upper right", fontsize=16)
    ax.grid(alpha=.20)
    return fig


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".svg"), dpi=200, metadata={"Date": None})
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = notebook_data(args.notebook)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = [(fig05, "fig-05-potential-and-hamiltonian"),
               (fig06, "fig-06-clean-wire-and-barrier-transport"),
               (fig07, "fig-07-sub-barrier-scattering-state"),
               (fig08, "fig-08-example-a-double-barrier")]
    for mobile in (False, True):
        style(mobile)
        for draw, stem in figures:
            suffix = "-mobile" if mobile else ""
            save(draw(data, mobile), args.output_dir / (stem+suffix))
    print(json.dumps({"T_probe": float(data["T_tunnel"]),
                      "rho_slice_error": float(data["rho_slice_error_transport"]),
                      "flux_relative_spread": float(data["flux_relative_spread_tunnel"]),
                      "source_sha256": SOURCE_SHA256}, indent=2))


if __name__ == "__main__":
    main()
