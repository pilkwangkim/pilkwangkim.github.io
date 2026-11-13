#!/usr/bin/env python3
"""Check the elementary derivation bridges in the bilingual NEGF article.

Requires only NumPy and the standard library. These small, explicitly
constructed models check normalization and prefactors; the separate reference
and occupation scripts validate the article's 100-site numerical examples.
All energies below use one consistent arbitrary unit, not material data.
"""

from __future__ import annotations

import json
import math

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def surface_green(energy: float, hopping: float, onsite: float) -> complex:
    """Retarded root at eta=0, including the decaying real root outside band."""
    b = complex(energy - onsite)
    disc = np.sqrt(b * b - 4 * hopping**2 + 0j)
    roots = ((b + disc) / (2 * hopping**2), (b - disc) / (2 * hopping**2))
    for root in roots:
        if root.imag < 0:
            return complex(root)
    return complex(min(roots, key=lambda root: abs(hopping * root)))


def spectral_normalization() -> dict[str, float]:
    """A closed single level has one state per spin, independent of plot eta."""
    level, eta = 0.37, 0.08
    radius = 1000 * eta
    energy = np.linspace(level - radius, level + radius, 200001)
    green = 1 / (energy - level + 1j * eta)
    rho = -green.imag / math.pi
    widths = np.diff(energy)
    integral = float(np.sum(0.5 * (rho[1:] + rho[:-1]) * widths))
    # The finite window omits a known tail. Do not mistake it for grid error.
    finite_exact = 2 * math.atan(radius / eta) / math.pi
    require(abs(integral - finite_exact) < 1e-10, "Lorentzian quadrature mismatch")
    require(abs(1 - integral) < 1e-3, "Lorentzian state count is not near one")

    rng = np.random.default_rng(20261113)
    raw = rng.normal(size=(5, 5))
    hamiltonian = (raw + raw.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    error = 0.0
    for probe in (-1.3, 0.2, 1.7):
        resolvent = np.linalg.inv((probe + 1j * eta) * np.eye(5) - hamiltonian)
        spectral = 1j * (resolvent - resolvent.conj().T)
        lorentzians = 2 * eta / ((probe - eigenvalues) ** 2 + eta**2)
        expected_diagonal = np.abs(eigenvectors) ** 2 @ lorentzians
        error = max(error, max_abs(np.diag(spectral) - expected_diagonal))
    require(error < 1e-12, "Eigenstate weights do not reproduce diagonal A")
    return {
        "one_level_integral_on_finite_window": integral,
        "known_omitted_tail": 1 - finite_exact,
        "finite_window_quadrature_error": abs(integral - finite_exact),
        "eigenstate_spectral_diagonal_max_error": error,
    }


def surface_and_flux() -> dict[str, float | int]:
    """Check root choice, velocity factors, scattering unitarity and bond flux."""
    hopping, onsite, spacing, hbar = 0.8, 1.6, 0.7, 0.41
    root_error = velocity_error = quadratic_error = 0.0
    for ka in (0.1, 0.5, 1.3, 2.4, math.pi - 0.1):
        energy = onsite - 2 * hopping * math.cos(ka)
        root = surface_green(energy, hopping, onsite)
        root_error = max(root_error, abs(root + np.exp(1j * ka) / hopping))
        gamma = -2 * (hopping**2 * root).imag
        velocity = 2 * hopping * spacing * math.sin(ka) / hbar
        velocity_error = max(velocity_error, abs(gamma - hbar * velocity / spacing))
    for energy in (onsite - 3 * hopping, onsite + 3 * hopping):
        root = surface_green(energy, hopping, onsite)
        quadratic_error = max(
            quadratic_error,
            abs(hopping**2 * root**2 - (energy - onsite) * root + 1),
        )
        require(abs(hopping * root) < 1, "Exterior root grows into the lead")
        require(abs(root.imag) < 1e-14, "Exterior eta=0 broadening must vanish")
    require(root_error < 1e-12, "Surface root differs from outgoing Bloch form")
    require(velocity_error < 1e-12, "Matched contact gamma/velocity mismatch")
    require(quadratic_error < 1e-12, "Surface root fails fixed-point equation")

    rng = np.random.default_rng(20261113)
    identity_error = unitarity_error = flux_error = occupation_flux_error = 0.0
    # Both equal and unequal lead band bottoms are tested. H is real and
    # Hermitian, eta=0, and every chosen energy is inside both lead bands.
    cases = 0
    for right_onsite in (onsite, onsite + 0.2):
        for _ in range(4):
            n = 7
            potential = rng.uniform(-0.2, 0.9, n)
            hamiltonian = np.diag(onsite + potential)
            hamiltonian += np.diag(np.full(n - 1, -hopping), 1)
            hamiltonian += np.diag(np.full(n - 1, -hopping), -1)
            for energy in (0.3, 0.7, 1.4, 2.8):
                sigma_l = hopping**2 * surface_green(energy, hopping, onsite)
                sigma_r = hopping**2 * surface_green(energy, hopping, right_onsite)
                gamma_l, gamma_r = -2 * sigma_l.imag, -2 * sigma_r.imag
                operator = energy * np.eye(n, dtype=complex) - hamiltonian
                operator[0, 0] -= sigma_l
                operator[-1, -1] -= sigma_r
                green = np.linalg.inv(operator)
                spectral = 1j * (green - green.conj().T)
                psi_l = green[:, 0] * math.sqrt(gamma_l)
                psi_r = green[:, -1] * math.sqrt(gamma_r)
                a_l = np.outer(psi_l, psi_l.conj())
                a_r = np.outer(psi_r, psi_r.conj())
                identity_error = max(identity_error, max_abs(spectral - a_l - a_r))

                reflection = abs(1 - 1j * gamma_l * green[0, 0]) ** 2
                transmission = gamma_l * gamma_r * abs(green[-1, 0]) ** 2
                unitarity_error = max(unitarity_error, abs(reflection + transmission - 1))
                require(-1e-12 <= transmission <= 1 + 1e-12, "Single-channel T outside [0,1]")
                directed_flux = -2 * np.imag(
                    np.diag(hamiltonian, 1) * psi_l[1:] * psi_l[:-1].conj()
                )
                flux_error = max(flux_error, max_abs(directed_flux - transmission))

                # Occupations here are specified at this energy; no constant-
                # occupation assumption over a full Fermi distribution is made.
                f_l, f_r = 0.83, 0.21
                correlation = f_l * a_l + f_r * a_r
                occupied_flux = -2 * np.imag(
                    np.diag(hamiltonian, 1) * np.diag(correlation, -1)
                )
                occupation_flux_error = max(
                    occupation_flux_error,
                    max_abs(occupied_flux - (f_l - f_r) * transmission),
                )
                cases += 1
    for name, error in (
        ("spectral matrix identity", identity_error),
        ("R+T=1", unitarity_error),
        ("every-bond spectral flux=T", flux_error),
        ("occupied flux=T(f_L-f_R)", occupation_flux_error),
    ):
        require(error < 1e-11, f"Failed {name}: {error}")
    return {
        "dense_scattering_cases": cases,
        "outgoing_surface_root_max_error": root_error,
        "matched_velocity_factor_max_error": velocity_error,
        "exterior_quadratic_max_error": quadratic_error,
        "spectral_matrix_identity_max_error": identity_error,
        "rank_one_unitarity_max_error": unitarity_error,
        "per_bond_spectral_flux_max_error": flux_error,
        "occupied_spectral_flux_max_error": occupation_flux_error,
    }


def occupation_continuity_and_lifetime() -> dict[str, float]:
    """Check weighted filling, density-matrix continuity and probability decay."""
    level, gamma_l, gamma_r, f_l, f_r = 0.6, 0.08, 0.2, 0.9, 0.1
    gamma = gamma_l + gamma_r
    green = 1 / (0.55 - level + 0.5j * gamma)
    spectral = gamma * abs(green) ** 2
    correlation = (gamma_l * f_l + gamma_r * f_r) * abs(green) ** 2
    effective_filling = (gamma_l * f_l + gamma_r * f_r) / gamma
    filling_error = abs(correlation / spectral - effective_filling)
    require(filling_error < 1e-14, "Single-level occupation weighting mismatch")
    require(0 <= effective_filling <= 1, "Invalid single-level effective filling")

    rng = np.random.default_rng(7)
    n, hbar, electron_charge_magnitude = 6, 0.41, 0.73
    raw = rng.normal(size=n - 1) + 1j * rng.normal(size=n - 1)
    hamiltonian = np.diag(rng.normal(size=n)).astype(complex)
    hamiltonian += np.diag(raw, 1) + np.diag(raw.conj(), -1)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi /= np.linalg.norm(psi)
    density_matrix = np.outer(psi, psi.conj())
    derivative = -1j * (hamiltonian @ density_matrix - density_matrix @ hamiltonian) / hbar
    particle_flux = -2 * np.imag(raw * np.diag(density_matrix, -1)) / hbar
    # There are no external currents at the two ends of this closed test chain.
    padded = np.r_[0.0, particle_flux, 0.0]
    continuity_error = max_abs(np.diag(derivative).real + np.diff(padded))
    require(continuity_error < 1e-12, "Density-matrix bond-current sign mismatch")

    # Gn/(2*pi) is the occupied density matrix per unit energy per spin.
    # Hence physical particle flux = J_Gn/h, and spin-degenerate electron-flow
    # current = (2*e/h)*J_Gn. This checks the factors, not a charge-sign choice.
    gn = 2 * math.pi * density_matrix
    j_gn = -2 * np.imag(raw * np.diag(gn, -1))
    planck = 2 * math.pi * hbar
    current_from_density = 2 * electron_charge_magnitude * particle_flux
    current_from_gn = 2 * electron_charge_magnitude * j_gn / planck
    prefactor_error = max_abs(current_from_density - current_from_gn)
    require(prefactor_error < 1e-12, "Current 2*pi/hbar/spin prefactor mismatch")

    # A physical, energy-independent escape width gamma produces probability
    # decay gamma/hbar. A finite numerical plotting eta is not that width.
    lifetime = hbar / gamma
    time = np.linspace(0, 5 * lifetime, 101)
    amplitude = np.exp(-1j * level * time / hbar - gamma * time / (2 * hbar))
    probability_error = max_abs(abs(amplitude) ** 2 - np.exp(-time / lifetime))
    e_fold_error = abs(abs(np.exp(-1j * level * lifetime / hbar - 0.5)) ** 2 - math.exp(-1))
    require(probability_error < 1e-14 and e_fold_error < 1e-14, "Probability lifetime mismatch")
    return {
        "single_level_effective_filling": effective_filling,
        "single_level_filling_error": filling_error,
        "density_matrix_continuity_max_error": continuity_error,
        "current_prefactor_max_error": prefactor_error,
        "probability_lifetime_max_error": probability_error,
        "probability_at_one_lifetime_error": e_fold_error,
    }


def main() -> None:
    results = {
        "status": "passed",
        "spectral_normalization": spectral_normalization(),
        "surface_and_flux": surface_and_flux(),
        "occupation_continuity_and_lifetime": occupation_continuity_and_lifetime(),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
