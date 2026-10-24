#!/usr/bin/env python3
"""Reproduce the two Example A figures with resonance-resolved quadrature.

Run with a Python environment containing NumPy and Matplotlib.  The numerical
model and quadrature are shared with check_negf_occupation.py; the plotting
mesh does not determine the integrated density or current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
import numpy as np

import check_negf_occupation as physics


IMAGE_DIR = Path(__file__).resolve().parents[1] / "assets/img/posts/2026-07-22-negf-from-scratch"


def ldos_diagonal(energies, eta):
    """Evaluate -Im G_jj/pi with left/right continued fractions.

    At finite eta the complete LDOS also contains the regulator's spectral
    term, so it must not be silently replaced by (A_L+A_R)/(2*pi).
    """
    z = np.asarray(energies) + 1j * eta
    b = z - 2 * physics.T0
    radical = np.sqrt(b * b - 4 * physics.T0**2)
    first, second = (b - radical) / 2, (b + radical) / 2
    sigma = np.where(first.imag < 0, first, second)
    diagonal = z[None, :] - 2 * physics.T0 - physics.POTENTIAL[:, None]
    left = np.zeros_like(diagonal)
    right = np.zeros_like(diagonal)
    left[0], right[-1] = sigma, sigma
    for site in range(1, physics.N):
        left[site] = physics.T0**2 / (diagonal[site - 1] - left[site - 1])
    for site in range(physics.N - 2, -1, -1):
        right[site] = physics.T0**2 / (diagonal[site + 1] - right[site + 1])
    return (-np.imag(1 / (diagonal - left - right)) / np.pi).T


def calculate(refinement=16, eta=1e-9, tail_kT=18):
    peaks = physics.resonance_data()
    result = physics.integrate(physics.energy_grid(peaks, refinement, tail_kT), eta)
    data = {
        "energy": result["energies"],
        "rho_left": result["rho_well_L"], "rho_right": result["rho_well_R"],
        "fermi_left": result["fermi_L"], "fermi_right": result["fermi_R"],
        "mu_left": physics.MU_L, "mu_right": physics.MU_R,
        "selected_energy": physics.SELECTED_RESONANCE_EV,
        "occupation_left": result["site_occupation_L"],
        "occupation_right": result["site_occupation_R"],
        "current_spectrum": result["current_spectrum_nA_per_eV"],
        "current_nA": result["landauer_current_nA"],
        "bond_relative_spread": result["relative_bond_spread"],
    }
    # Dense quadrature is essential for the integrated numbers, but a filled
    # SVG polygon need not contain every quadrature node.  Keep every peak
    # center and 100 samples per FWHM in each local interval after thinning.
    display_indices = np.unique(np.concatenate((
        np.arange(0, len(result["energies"]), 16),
        np.array([len(result["energies"]) - 1]),
        np.searchsorted(result["energies"], [p["energyEV"] for p in peaks
                                             if p["energyEV"] <= result["energies"][-1]]),
    )))
    data["display_indices"] = display_indices

    # Preserve the original visualization range while explicitly including
    # all six maxima and many samples on each side of every narrow peak.
    uniform = np.linspace(1e-3, 0.55, 1400)
    parts = [uniform]
    for peak in peaks:
        local = peak["energyEV"] + peak["widthEV"] * np.linspace(-8, 8, 801)
        parts.append(local[(local >= 1e-3) & (local <= 0.55)])
        parts.append(np.array([peak["energyEV"]]))
    energies = np.unique(np.concatenate(parts))
    gl, _, gamma = physics.endpoint_columns(energies, eta)
    peak_energy = np.array([peak["energyEV"] for peak in peaks])
    peak_columns, _, peak_gamma = physics.endpoint_columns(peak_energy, eta)
    visual = {
        "energy": energies,
        "transmission": gamma**2 * abs(gl[-1])**2,
        "peak_energy": peak_energy,
        "peak_transmission": peak_gamma**2 * abs(peak_columns[-1])**2,
        "ldos": ldos_diagonal(energies, eta),
        "ldos_cap": np.percentile(ldos_diagonal(uniform, eta), 99.3),
    }
    assert np.all(visual["ldos"] >= 0)
    assert len(peak_energy) == 6
    assert abs(result["well_population_sites"] - result["well_population_spectrum"]) < 1e-10
    report = {
        "quadrature": {"refinement": refinement, "tail_kT": tail_kT, "eta_ev": eta},
        "results": physics.compact(result),
        "spectroscopy_samples": len(energies),
        "resonance_energies_ev": peak_energy.tolist(),
        "resonance_transmissions": visual["peak_transmission"].tolist(),
        "ldos_uniform_mesh_percentile_99_3": float(visual["ldos_cap"]),
    }
    return data, visual, report


def figure_style(mobile=False):
    """Keep physical text sizes readable at 691/358 CSS-pixel article widths."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.labelsize": 18 if mobile else 19,
        "axes.titlesize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "axes.titlepad": 13,
        "axes.labelpad": 9, "axes.linewidth": 1.0,
        "svg.fonttype": "path", "savefig.facecolor": "white",
        "figure.constrained_layout.w_pad": 0.12,
        "figure.constrained_layout.h_pad": 0.16,
    })


def save_figure(fig, destination):
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.12)
    svg = destination.with_suffix(".svg")
    fig.savefig(svg, dpi=180, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_bookkeeping(data, destination, mobile=False):
    """Keep the available/occupied comparison on the same energy samples."""
    energy = data["energy"]
    rho_left, rho_right = data["rho_left"], data["rho_right"]
    occupied_left = data["fermi_left"] * rho_left
    occupied_right = data["fermi_right"] * rho_right
    total = rho_left + rho_right
    occupied_total = occupied_left + occupied_right
    positive = np.concatenate((total[total > 0], occupied_total[occupied_total > 0]))
    ymax = 1.15 * positive.max()
    ymin = max(positive.min(), ymax * 1e-7)

    figure_style(mobile)
    shape = (2, 1) if mobile else (1, 2)
    fig, axes = plt.subplots(*shape, figsize=(5, 12.6) if mobile else (10.8, 7.3),
                             sharex=True, sharey=True, layout="constrained")
    available, occupied = axes.flat
    available.semilogy(energy, rho_left, color="#1f77b4", lw=2.2,
                      label=r"$\rho_{W,L}=\rho_{W,R}$")
    available.semilogy(energy, total, color="0.12", lw=2.5, ls=":",
                      label=r"$\rho_{W,L}+\rho_{W,R}$")
    available.set_title("(a) Available well spectrum", loc="left")
    available.set_ylabel("Well spectral density\nper spin (eV$^{-1}$)")
    occupied.semilogy(energy, occupied_left, color="#1f77b4", lw=2.2,
                     label=r"$f_L\rho_{W,L}$")
    occupied.semilogy(energy, occupied_right, color="#e67e22", lw=2.2, ls="--",
                     label=r"$f_R\rho_{W,R}$")
    occupied.semilogy(energy, occupied_total, color="0.12", lw=2.5, ls=":",
                     label=r"$f_L\rho_{W,L}+f_R\rho_{W,R}$")
    occupied.set_title("(b) Occupied well spectrum", loc="left")
    if mobile:
        occupied.set_ylabel("Well spectral density\nper spin (eV$^{-1}$)")
    for ax in (available, occupied):
        ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.18,
                   label=r"$[\mu_R,\mu_L]$" if ax is available else "_nolegend_")
        ax.axvline(data["selected_energy"], color="#7d3c98", ls="--", lw=1.5,
                   label=r"Selected $E_\star$" if ax is available else "_nolegend_")
        ax.set_xlabel("Energy E (eV)")
        ax.set_xlim(0, data["energy"][-1])
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(LogLocator(numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelleft=True, labelbottom=True)
        ax.grid(alpha=0.22, which="major")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), frameon=False,
                  borderaxespad=0, handlelength=1.6, labelspacing=0.45)
    save_figure(fig, destination)


def plot_summary(data, visual, destination, mobile=False):
    """Use resolved spectra for (a,b,d), and integrated populations for (c)."""
    sites = np.arange(100)
    figure_style(mobile)
    fig, axes = plt.subplots(4, 1, figsize=(5, 23.4), layout="constrained") if mobile else plt.subplots(
        2, 2, figsize=(10.8, 12.2), layout="constrained")
    transmission_ax, ldos_ax, density_ax, current_ax = axes.flat

    transmission_ax.semilogy(visual["energy"], visual["transmission"], color="#1f4e79", lw=1.8)
    for energy in visual["peak_energy"]:
        transmission_ax.axvline(energy, color="#c0392b", ls="--", lw=0.9, alpha=0.7)
    transmission_ax.plot(visual["peak_energy"], visual["peak_transmission"], "o",
                         color="#c0392b", ms=5.5, label="Resonances")
    transmission_ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.25,
                            label=r"$[\mu_R,\mu_L]$")
    transmission_ax.axhline(1.0, color="0.6", ls=":", lw=1)
    transmission_ax.set_xlabel("Energy E (eV)")
    transmission_ax.set_ylabel("T(E)")
    transmission_ax.set_title("(a) Resonant transmission", loc="left")
    transmission_ax.set_ylim(1e-7, 3)
    transmission_ax.set_xlim(0, 0.55)
    transmission_ax.legend(loc="lower right", handlelength=1.3)
    transmission_ax.xaxis.set_major_locator(MaxNLocator(4))
    transmission_ax.yaxis.set_major_locator(LogLocator(numticks=5))
    transmission_ax.yaxis.set_minor_formatter(NullFormatter())
    transmission_ax.grid(alpha=0.25, which="major")

    # The percentile is measured on a uniform energy mesh, so adding many
    # samples near a resonance does not silently change the color scale.
    image = ldos_ax.pcolormesh(sites, visual["energy"], visual["ldos"], shading="auto",
                               cmap="inferno", vmin=0, vmax=visual["ldos_cap"], rasterized=True)
    for boundary in (39, 42, 58, 61):
        ldos_ax.axvline(boundary - 0.5, color="cyan", lw=0.7, alpha=0.5)
    ldos_ax.set_xlabel("Site $j$")
    ldos_ax.set_ylabel("Energy E (eV)")
    ldos_ax.set_title("(b) Well modes in the LDOS", loc="left")
    ldos_ax.set_ylim(0, 0.55)
    ldos_ax.xaxis.set_major_locator(MaxNLocator(4))
    colorbar = fig.colorbar(image, ax=ldos_ax, orientation="horizontal" if mobile else "vertical",
                           pad=0.11 if mobile else 0.035, shrink=0.94)
    colorbar.set_label("LDOS per spin (eV$^{-1}$)", fontsize=16)
    colorbar.ax.tick_params(labelsize=16)
    colorbar.locator = MaxNLocator(4)
    colorbar.update_ticks()

    density_ax.stackplot(sites, data["occupation_left"], data["occupation_right"],
                         colors=["#4c9ed9", "#f0a35e"], alpha=0.58,
                         labels=[r"$n_j^{(L)}$", r"$n_j^{(R)}$"])
    density_ax.plot(sites, data["occupation_left"] + data["occupation_right"],
                    color="0.15", lw=2.1, label=r"$n_j$")
    density_ax.axvspan(38.5, 41.5, color="crimson", alpha=0.12, label="Barriers")
    density_ax.axvspan(57.5, 60.5, color="crimson", alpha=0.12)
    density_ax.axvspan(41.5, 57.5, color="gold", alpha=0.12, label="Well")
    density_ax.set_xlabel("Site $j$")
    density_ax.set_ylabel("$n_j$ (electrons/site)")
    density_ax.set_title("(c) Site occupation" if mobile else "(c) Integrated occupation", loc="left")
    density_ax.legend(loc="upper left", bbox_to_anchor=(0, -0.23), frameon=False,
                     ncol=2, columnspacing=0.8, handlelength=1.1, borderaxespad=0)
    density_ax.xaxis.set_major_locator(MaxNLocator(4))
    density_ax.yaxis.set_major_locator(MaxNLocator(5))
    density_ax.set_ylim(0, 0.63)
    density_ax.grid(alpha=0.25)

    display = data["display_indices"]
    current_ax.fill_between(data["energy"][display], data["current_spectrum"][display], 0,
                            color="#2e8b57", alpha=0.30)
    current_ax.plot(data["energy"], data["current_spectrum"], color="#1f6f4a", lw=2.0,
                    label=r"$dI/dE$")
    current_ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.15,
                       label=r"$[\mu_R,\mu_L]$")
    current_ax.axvline(data["selected_energy"], color="#7d3c98", ls="--", lw=1.2,
                       label=r"Selected $E_\star$")
    current_ax.text(0.97, 0.97,
                    (rf"area $=I={data['current_nA']:.4f}$ nA" "\n"
                     rf"$\Delta I_{{\rm bond}}/\bar I={data['bond_relative_spread']:.2e}$"),
                    transform=current_ax.transAxes, ha="right", va="top", fontsize=16,
                    bbox=dict(boxstyle="round", fc="white", ec="#2e8b57", alpha=0.92))
    current_ax.set_xlabel("Energy E (eV)")
    current_ax.set_ylabel("Spectral current\n$dI/dE$ (nA/eV)")
    current_ax.set_title("(d) Spectral current" if mobile else "(d) Current from spectral area", loc="left")
    current_ax.set_xlim(0, data["energy"][-1])
    current_ax.set_ylim(0, data["current_spectrum"].max() * 1.45)
    current_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    current_ax.yaxis.set_major_locator(MaxNLocator(5))
    current_ax.xaxis.set_major_locator(MaxNLocator(4))
    current_ax.legend(loc="upper left", bbox_to_anchor=(0, -0.23), frameon=False,
                      handlelength=1.5, borderaxespad=0)
    current_ax.grid(alpha=0.25)

    save_figure(fig, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--refinement", type=int, default=16)
    parser.add_argument("--eta", type=float, default=1e-9)
    parser.add_argument("--tail-kT", type=int, default=18)
    args = parser.parse_args()
    data, visual, report = calculate(args.refinement, args.eta, args.tail_kT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_bookkeeping(data, args.output_dir / "fig-09-example-a-spectral-bookkeeping.png")
    plot_summary(data, visual, args.output_dir / "fig-10-example-a-transport-summary.png")
    plot_bookkeeping(data, args.output_dir / "fig-09-example-a-spectral-bookkeeping-mobile.png", mobile=True)
    plot_summary(data, visual, args.output_dir / "fig-10-example-a-transport-summary-mobile.png", mobile=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
