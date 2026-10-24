#!/usr/bin/env python3
"""Reproduce and converge the NEGF article's Example A occupation.

Only NumPy and Node.js are required.  The browser solver supplies resonance
locations and widths for placing quadrature points; an independent, batched
Thomas solve computes both Green-function endpoint columns.  Density and
bond/terminal currents then follow directly from those columns, not from
the browser determinant recurrence.

Example:
  python tools/check_negf_occupation.py --output /tmp/negf-occupation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
J_PER_EV = 1.602176634e-19
CHARGE = 1.602176634e-19
H = 2 * np.pi * HBAR
T0 = HBAR**2 / (2 * 0.067 * M0 * 1e-18) / J_PER_EV
CURRENT_PREFACTOR = 2 * CHARGE / H * J_PER_EV
N = 100
WELL = slice(42, 58)
POTENTIAL = np.zeros(N)
POTENTIAL[39:42] = POTENTIAL[58:61] = 0.4
# Full precision matters: rounding mu before placing a coarse mesh can move
# its samples relative to a narrow occupied resonance.  This is the selected
# notebook resonance independently refined by the dense reference check.
SELECTED_RESONANCE_EV = 0.141312760965677
MU_L = SELECTED_RESONANCE_EV + 0.06
MU_R = SELECTED_RESONANCE_EV - 0.06
KT = 0.005
NOTEBOOK_COMMIT = "7fd9f32c390e117ab6b04bb7f787fc5b43c959e2"
NOTEBOOK_SHA256 = "2918fa2f8a8e0b20762fffa1b4b87d96a4cab67800e06fc0b94a79ce8b607cfa"


def resonance_data(node: str | None = None) -> list[dict]:
    """Get peak centers/FWHM from the separately checked browser spectrum."""
    js = "const c=require(process.argv[1]);console.log(JSON.stringify(c.spectrum(c.profile()).peaks));"
    completed = subprocess.run(
        [node or shutil.which("node") or "node", "-e", js, str(ROOT / "assets/js/negf-core.js")],
        text=True, capture_output=True, check=True,
    )
    return json.loads(completed.stdout)


def energy_grid(peaks: list[dict], refinement: int = 16, tail_kT: int = 18) -> np.ndarray:
    """A global mesh plus linewidth-aware local meshes, all merged once.

    Each local mesh spans +/-20 FWHM with a step of FWHM/100.  The global
    mesh covers smooth backgrounds and the geometric part covers the band
    threshold.  Refinement doubles every resolution rather than merely
    shifting a uniform mesh relative to a narrow peak.
    """
    upper = MU_L + tail_kT * KT
    parts = [
        np.geomspace(1e-10, 1e-3, 400 * refinement, endpoint=False),
        np.linspace(1e-3, upper, 3000 * refinement + 1),
    ]
    for peak in peaks:
        center, width = peak["energyEV"], peak["widthEV"]
        if width is None or not 1e-10 < center < upper:
            continue
        local = center + width * np.linspace(-20, 20, 4000 * refinement + 1)
        parts.append(local[(local > 1e-10) & (local < upper)])
        parts.append(np.array([center]))
    return np.unique(np.concatenate(parts))


def trapezoid_weights(energies: np.ndarray) -> np.ndarray:
    weights = np.empty_like(energies)
    weights[0] = (energies[1] - energies[0]) / 2
    weights[-1] = (energies[-1] - energies[-2]) / 2
    weights[1:-1] = (energies[2:] - energies[:-2]) / 2
    return weights


def endpoint_columns(energies: np.ndarray, eta: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent solve of A G[:,0/-1] = e_0/e_(N-1), vectorized in E."""
    z = np.asarray(energies) + 1j * eta
    b = z - 2 * T0
    radical = np.sqrt(b * b - 4 * T0 * T0)
    first, second = (b - radical) / 2, (b + radical) / 2
    sigma = np.where(first.imag < 0, first, second)
    gamma = -2 * sigma.imag
    diagonal = z[None, :] - 2 * T0 - POTENTIAL[:, None]
    diagonal[0] -= sigma
    diagonal[-1] -= sigma
    left = np.zeros_like(diagonal)
    left[0] = 1
    # A has +t0 off-diagonals, since A=zI-H-Sigma and H_j,j+1=-t0.
    for j in range(1, N):
        multiplier = T0 / diagonal[j - 1]
        diagonal[j] -= T0 * multiplier
        left[j] = -multiplier * left[j - 1]
    right = np.empty_like(left)
    left[-1] /= diagonal[-1]
    right[-1] = 1 / diagonal[-1]
    for j in range(N - 2, -1, -1):
        left[j] = (left[j] - T0 * left[j + 1]) / diagonal[j]
        right[j] = -T0 * right[j + 1] / diagonal[j]
    return left, right, gamma


def integrate(energies: np.ndarray, eta: float = 1e-9, batch_size: int = 4096) -> dict:
    """Site/contact occupation, bond current, and energy-resolved well data.

    A_alpha and rho_W,alpha are per spin.  Site occupations include two
    spins, as does CURRENT_PREFACTOR.  Applying either spin factor again
    would double count.  Batching bounds memory without changing weights.
    """
    weights = trapezoid_weights(energies)
    site_left, site_right = np.zeros(N), np.zeros(N)
    bond = np.zeros(N - 1)
    keys = ("rho_well_L", "rho_well_R", "occupied_well_L", "occupied_well_R",
            "fermi_L", "fermi_R", "transmission", "current_spectrum_nA_per_eV")
    out = {key: np.empty(len(energies)) for key in keys}
    for start in range(0, len(energies), batch_size):
        sl = slice(start, min(start + batch_size, len(energies)))
        e, w = energies[sl], weights[sl]
        gl, gr, gamma = endpoint_columns(e, eta)
        fl = 1 / (1 + np.exp((e - MU_L) / KT))
        fr = 1 / (1 + np.exp((e - MU_R) / KT))
        al, ar = gamma * abs(gl)**2, gamma * abs(gr)**2
        site_left += (al * fl) @ w / np.pi
        site_right += (ar * fr) @ w / np.pi
        rho_l, rho_r = al[WELL].sum(axis=0) / (2 * np.pi), ar[WELL].sum(axis=0) / (2 * np.pi)
        gn_next_current = gamma * (
            fl * gl[1:] * gl[:-1].conj() + fr * gr[1:] * gr[:-1].conj()
        )
        bond += (2 * T0 * gn_next_current.imag) @ w * CURRENT_PREFACTOR
        transmission = gamma**2 * abs(gl[-1])**2
        values = (rho_l, rho_r, fl * rho_l, fr * rho_r, fl, fr, transmission,
                  1e9 * CURRENT_PREFACTOR * transmission * (fl - fr))
        for key, value in zip(keys, values, strict=True):
            out[key][sl] = value
    site_total = site_left + site_right
    out.update({
        "energies": energies, "weights": weights, "eta_ev": eta,
        "site_occupation": site_total, "site_occupation_L": site_left,
        "site_occupation_R": site_right, "bond_current_A": bond,
        "well_population_sites": float(site_total[WELL].sum()),
        "well_population_spectrum": float(2 * np.dot(weights, out["occupied_well_L"] + out["occupied_well_R"])),
        "landauer_current_nA": float(np.dot(weights, out["current_spectrum_nA_per_eV"])),
        "mean_bond_current_nA": float(bond.mean() * 1e9),
        "relative_bond_spread": float(np.ptp(bond) / abs(bond.mean())),
    })
    return out


def compact(result: dict) -> dict:
    return {
        "samples": len(result["energies"]), "eta_ev": result["eta_ev"],
        "energy_upper_ev": float(result["energies"][-1]),
        **{key: result[key] for key in (
            "well_population_sites", "well_population_spectrum", "landauer_current_nA",
            "mean_bond_current_nA", "relative_bond_spread")},
        "well_spectrum_sites_abs_error": abs(result["well_population_sites"] - result["well_population_spectrum"]),
    }


def dense_checks() -> list[dict]:
    """Check both endpoint columns against a dense 100x100 solve."""
    checks = []
    for e in (0.016126494081925, 0.01615, SELECTED_RESONANCE_EV, 0.2):
        gl, gr, _ = endpoint_columns(np.array([e]))
        z = e + 1e-9j
        b = z - 2 * T0
        roots = ((b - np.sqrt(b*b - 4*T0*T0))/2, (b + np.sqrt(b*b - 4*T0*T0))/2)
        sigma = next(s for s in roots if s.imag < 0)
        matrix = np.diag(z - 2*T0 - POTENTIAL)
        matrix += np.diag(np.full(N - 1, T0), 1) + np.diag(np.full(N - 1, T0), -1)
        matrix[0, 0] -= sigma
        matrix[-1, -1] -= sigma
        rhs = np.zeros((N, 2), complex)
        rhs[0, 0] = rhs[-1, 1] = 1
        reference = np.linalg.solve(matrix, rhs)
        columns = np.column_stack((gl[:, 0], gr[:, 0]))
        checks.append({"energy_ev": e, "relative_column_error": float(np.max(abs(columns-reference))/np.max(abs(reference)))})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebook", type=Path, help="Optional downloaded original notebook for SHA verification")
    args = parser.parse_args()
    peaks = resonance_data(args.node)
    original = np.unique(np.r_[np.geomspace(1e-10, 1e-3, 400, endpoint=False), np.linspace(1e-3, MU_L + 0.05, 3000)])
    historical = compact(integrate(original))
    refinement = [compact(integrate(energy_grid(peaks, r))) | {"refinement": r} for r in (4, 8, 16)]
    final_grid = energy_grid(peaks, 16)
    eta_results = [compact(integrate(final_grid, eta)) for eta in (1e-9, 1e-10, 1e-11, 0.0)]
    cutoff_results = [compact(integrate(energy_grid(peaks, 16, tail))) | {"tail_kT": tail} for tail in (10, 14, 18, 22)]
    dense = dense_checks()
    errors = []
    if max(c["relative_column_error"] for c in dense) > 1e-9:
        errors.append("Tridiagonal endpoint columns differ from dense solve")
    if abs(refinement[-1]["well_population_sites"] - refinement[-2]["well_population_sites"]) > 2e-6:
        errors.append("Resonance-local occupation grid has not converged")
    if abs(refinement[-1]["landauer_current_nA"] - refinement[-2]["landauer_current_nA"]) > 1e-4:
        errors.append("Resonance-local current grid has not converged")
    if abs(cutoff_results[-1]["well_population_sites"] - cutoff_results[-2]["well_population_sites"]) > 2e-6:
        errors.append("Occupation upper cutoff has not converged")
    if abs(cutoff_results[-1]["landauer_current_nA"] - cutoff_results[-2]["landauer_current_nA"]) > 1e-4:
        errors.append("Current upper cutoff has not converged")
    if abs(eta_results[-1]["well_population_sites"] - eta_results[-2]["well_population_sites"]) > 2e-6:
        errors.append("Small-eta occupation has not approached its real-axis limit")
    if abs(eta_results[-1]["landauer_current_nA"] - eta_results[-2]["landauer_current_nA"]) > 1e-4:
        errors.append("Small-eta current has not approached its real-axis limit")
    if eta_results[-1]["relative_bond_spread"] > 1e-8:
        errors.append("Zero-eta bond current is not conserved")
    if abs(eta_results[-1]["landauer_current_nA"] - eta_results[-1]["mean_bond_current_nA"]) > 1e-6:
        errors.append("Zero-eta Landauer and bond currents differ")
    if args.notebook and hashlib.sha256(args.notebook.read_bytes()).hexdigest() != NOTEBOOK_SHA256:
        errors.append("Original notebook hash differs")
    report = {
        "pass": not errors, "errors": errors, "numpy": np.__version__,
        "method": "Independent batched tridiagonal solve; all integration grids use explicit nonuniform trapezoid weights",
        "tolerances": {"density_change_electrons": 2e-6, "current_change_nA": 1e-4,
                       "dense_relative_error": 1e-9, "zero_eta_relative_bond_spread": 1e-8},
        "parameters": {"t0_ev": T0, "sites": N, "barriers": [[39, 42], [58, 61]],
                       "well": [42, 58], "selected_resonance_ev": SELECTED_RESONANCE_EV,
                       "mu_L_ev": MU_L, "mu_R_ev": MU_R, "kT_ev": KT},
        "notebook": {"commit": NOTEBOOK_COMMIT, "sha256": NOTEBOOK_SHA256},
        "resonances": peaks, "historical_3000_grid": historical,
        "local_grid_refinement": refinement, "eta_convergence": eta_results,
        "cutoff_convergence": cutoff_results, "dense_column_checks": dense,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
