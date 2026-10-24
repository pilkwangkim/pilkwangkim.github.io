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


def plot_bookkeeping(data, destination):
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

    fig, (available, occupied) = plt.subplots(1, 2, figsize=(11.8, 4.0), sharex=True, sharey=True)
    available.semilogy(energy, rho_left, color="#1f77b4", lw=1.5,
                      label=r"$\rho_{W,L}=\rho_{W,R}$ (symmetry)")
    available.semilogy(energy, total, color="0.12", lw=2.0, ls=":",
                      label=r"$\rho_{W,L}+\rho_{W,R}$")
    available.set_title("(a) available contact-resolved well spectrum")
    available.set_ylabel("well-projected spectral density per spin (1/eV)")
    occupied.semilogy(energy, occupied_left, color="#1f77b4", lw=1.5,
                     label=r"$f_L\rho_{W,L}$")
    occupied.semilogy(energy, occupied_right, color="#e67e22", lw=1.5, ls="--",
                     label=r"$f_R\rho_{W,R}$")
    occupied.semilogy(energy, occupied_total, color="0.12", lw=2.0, ls=":",
                     label=r"$f_L\rho_{W,L}+f_R\rho_{W,R}$")
    occupied.set_title("(b) the same states after reservoir filling")
    for ax in (available, occupied):
        ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.18,
                   label=r"$[\mu_R,\mu_L]$")
        ax.axvline(data["selected_energy"], color="#7d3c98", ls="--", lw=1.2,
                   label=r"selected $E_\star$")
        ax.set_xlabel("E (eV)")
        ax.set_xlim(0, data["energy"][-1])
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.22, which="both")
        ax.legend(fontsize=8.0, loc="lower left")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), dpi=200, bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)


def plot_summary(data, visual, destination):
    """Use resolved spectra for (a,b,d), and integrated populations for (c)."""
    sites = np.arange(100)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    transmission_ax, ldos_ax, density_ax, current_ax = axes.flat

    transmission_ax.semilogy(visual["energy"], visual["transmission"], color="#1f4e79", lw=1.8)
    for energy in visual["peak_energy"]:
        transmission_ax.axvline(energy, color="#c0392b", ls="--", lw=0.9, alpha=0.7)
    transmission_ax.plot(visual["peak_energy"], visual["peak_transmission"], "o",
                         color="#c0392b", ms=4.5, label="refined resonance maxima")
    transmission_ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.25,
                            label="main population-difference window")
    transmission_ax.axhline(1.0, color="0.6", ls=":", lw=1)
    transmission_ax.set_xlabel("E (eV)")
    transmission_ax.set_ylabel("T(E)")
    transmission_ax.set_title(r"(a) resonant tunneling: $T_{\rm peak}\to1$")
    transmission_ax.set_ylim(1e-7, 3)
    transmission_ax.set_xlim(0, 0.55)
    transmission_ax.legend(fontsize=8.5, loc="lower right")
    transmission_ax.grid(alpha=0.25, which="both")

    # The percentile is measured on a uniform energy mesh, so adding many
    # samples near a resonance does not silently change the color scale.
    image = ldos_ax.pcolormesh(sites, visual["energy"], visual["ldos"], shading="auto",
                               cmap="inferno", vmin=0, vmax=visual["ldos_cap"], rasterized=True)
    for boundary in (39, 42, 58, 61):
        ldos_ax.axvline(boundary - 0.5, color="cyan", lw=0.7, alpha=0.5)
    ldos_ax.set_xlabel("site $j$")
    ldos_ax.set_ylabel("E (eV)")
    ldos_ax.set_title(r"(b) LDOS$_j(E)$: quasi-bound well states")
    ldos_ax.set_ylim(0, 0.55)
    fig.colorbar(image, ax=ldos_ax, label="LDOS per spin (1/eV)")

    density_ax.stackplot(sites, data["occupation_left"], data["occupation_right"],
                         colors=["#4c9ed9", "#f0a35e"], alpha=0.58,
                         labels=[r"left-fed $n_j^{(L)}$", r"right-fed $n_j^{(R)}$"])
    density_ax.plot(sites, data["occupation_left"] + data["occupation_right"],
                    color="0.15", lw=1.7, label=r"total $n_j$")
    density_ax.axvspan(38.5, 41.5, color="crimson", alpha=0.12, label="barriers")
    density_ax.axvspan(57.5, 60.5, color="crimson", alpha=0.12)
    density_ax.axvspan(41.5, 57.5, color="gold", alpha=0.12, label="well")
    density_ax.set_xlabel("site $j$")
    density_ax.set_ylabel("$n_j$ (electrons/site)")
    density_ax.set_title(r"(c) occupied population: $n_j^{(L)}+n_j^{(R)}$")
    density_ax.legend(fontsize=7.8, loc="upper right", ncol=2)
    density_ax.grid(alpha=0.25)

    display = data["display_indices"]
    current_ax.fill_between(data["energy"][display], data["current_spectrum"][display], 0,
                            color="#2e8b57", alpha=0.30)
    current_ax.plot(data["energy"], data["current_spectrum"], color="#1f6f4a", lw=2.0,
                    label=(r"$dI/dE_{\rm eV}=(2e/h)J_{\rm per\,eV}$"
                           r"$\,T(f_L-f_R)$"))
    current_ax.axvspan(data["mu_right"], data["mu_left"], color="gold", alpha=0.15,
                       label=r"main $f_L-f_R$ window")
    current_ax.axvline(data["selected_energy"], color="#7d3c98", ls="--", lw=1.2,
                       label=r"selected $E_\star$")
    current_ax.text(0.97, 0.94,
                    (rf"area $=I={data['current_nA']:.4f}$ nA" "\n"
                     rf"$\Delta I_{{\rm bond}}/\bar I={data['bond_relative_spread']:.2e}$"),
                    transform=current_ax.transAxes, ha="right", va="top", fontsize=8.7,
                    bbox=dict(boxstyle="round", fc="white", ec="#2e8b57", alpha=0.92))
    current_ax.set_xlabel("E (eV)")
    current_ax.set_ylabel(r"spectral current density $dI/dE$ (nA/eV)")
    current_ax.set_title("(d) current is the area of the weighted spectrum")
    current_ax.set_xlim(0, data["energy"][-1])
    current_ax.set_ylim(bottom=0)
    current_ax.legend(fontsize=7.8, loc="upper left")
    current_ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), dpi=200, bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)


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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
