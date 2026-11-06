#!/usr/bin/env python3
"""A conservative, educational elastic multi-subband Boltzmann solver.

Requires Python >= 3.9 and NumPy. All energies are in eV unless explicitly
labelled J, lengths in nm, effective masses in units of the free-electron mass.
The transport geometry is a uniform wire: two confined transverse directions,
one semiclassical transport direction, prescribed subband edges, no axial force.

At a fixed TOTAL electron energy E, each open parabolic subband supplies two
states (n,+) and (n,-), in that order. The occupation of each state is f in [0,1].
Choose a symmetric nonnegative pair conductance kappa_ab, in nm^-1, and set

    C_a = |v_a| sum_b kappa_ab (f_b - f_a).

Use |v_a| in nm/s to obtain C in s^-1. The directional 1D density-of-states
weight is proportional to 1/|v_a|, so collisions conserve particles at each E.
For symmetric elastic microscopic transitions the Pauli factors cancel pairwise:
f_b(1-f_a)-f_a(1-f_b)=f_b-f_a. This does NOT justify dropping Pauli factors from
general inelastic or electron-electron collisions.

After division by speed, the stationary equations are

    s_a df_a/dx = sum_b kappa_ab (f_b - f_a),  s_a = +1 or -1.

This phenomenological conductance model is not a calibrated semiconductor
scattering model. It omits longitudinal coherence, tunneling, phonons, roughness
parameters and a self-consistent electrostatic potential.

Discretization: N cell averages, first-order upwind face occupations. Inflow is
prescribed only at the left (+ states) and right (- states) contact. The signed
FACE flux, not the same-cell signed occupation sum, is exactly conservative.
For one mode with kappa_(+,-)=beta the continuum transmission is 1/(1+beta L),
while this finite grid gives (1+beta dx)/(1+beta L+beta dx). Refinement is needed.

Framework references:
https://doi.org/10.1007/s10825-020-01488-4
https://doi.org/10.1016/j.jcp.2009.06.001
https://doi.org/10.3390/electronics10202472
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

E_CHARGE = 1.602176634e-19  # Positive elementary charge, C; electron charge=-e.
PLANCK = 6.62607015e-34  # J s, exact SI definition.
HBAR = PLANCK / (2.0 * np.pi)
ELECTRON_MASS = 9.1093837139e-31  # kg, CODATA 2022.
KB_EV = 8.617333262145e-5  # eV/K.
SPIN = 2  # Spin degeneracy; no extra valley degeneracy in this model.
ArrayInput = Union[float, Sequence[float], np.ndarray]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive.")
    return value


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer.")
    return int(value)


def well_energy_ev(n: int, width_nm: float, mass_rel: float = 0.19) -> float:
    """nth hard-wall transverse energy, hbar^2 pi^2 n^2/(2 m W^2)."""
    n = _integer(n, "n")
    width_m = 1e-9 * _positive(width_nm, "width_nm")
    mass_kg = ELECTRON_MASS * _positive(mass_rel, "mass_rel")
    return float(HBAR**2 * np.pi**2 * n**2 / (2 * mass_kg * width_m**2 * E_CHARGE))


def subband_edges(
    width_y_nm: float,
    width_z_nm: float,
    modes: Sequence[tuple[int, int]] = ((1, 1), (2, 1), (1, 2)),
    mass_y: float = 0.19,
    mass_z: float = 0.19,
    offset_ev: float = 0.0,
) -> np.ndarray:
    """Rectangular-wire hard-wall edges in the supplied mode order.

    These three default modes need not be the three lowest modes for an
    arbitrary aspect ratio. Enumerate/sort enough (ny,nz) pairs when choosing
    a spectral cutoff; this function deliberately does not silently reorder.
    """
    if not np.isfinite(offset_ev) or not modes:
        raise ValueError("Supply a finite offset and at least one transverse mode.")
    if len(set(tuple(mode) for mode in modes)) != len(modes):
        raise ValueError("Transverse mode pairs must be distinct.")
    return np.array([
        offset_ev + well_energy_ev(ny, width_y_nm, mass_y)
        + well_energy_ev(nz, width_z_nm, mass_z)
        for ny, nz in modes
    ])


def rectangular_modes_below(
    cutoff_ev: float, width_y_nm: float, width_z_nm: float,
    mass_y: float = 0.19, mass_z: float = 0.19, offset_ev: float = 0.0,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Enumerate ALL hard-wall modes with edge strictly below a cutoff.

    Return mode pairs and sorted energies together so an index never silently
    changes its physical meaning. Raising this transverse cutoff tests mode
    truncation; it is separate from refining the transport energy quadrature.
    """
    available = float(cutoff_ev) - float(offset_ev)
    if not np.isfinite(available) or not np.isfinite(offset_ev):
        raise ValueError("cutoff_ev and offset_ev must be finite.")
    ey = well_energy_ev(1, width_y_nm, mass_y)
    ez = well_energy_ev(1, width_z_nm, mass_z)
    if available <= ey + ez:
        return [], np.array([], dtype=float)
    ny_max = int(np.floor(np.sqrt(max(0, (available - ez) / ey))))
    nz_max = int(np.floor(np.sqrt(max(0, (available - ey) / ez))))
    selected = sorted((offset_ev + ey * ny**2 + ez * nz**2, ny, nz)
                      for ny in range(1, ny_max + 1) for nz in range(1, nz_max + 1)
                      if offset_ev + ey * ny**2 + ez * nz**2 < cutoff_ev)
    return [(ny, nz) for _, ny, nz in selected], np.array([energy for energy, _, _ in selected])


def transverse_wavefunction(
    y_nm: ArrayInput, z_nm: ArrayInput, width_y_nm: float, width_z_nm: float,
    ny: int = 1, nz: int = 1,
) -> np.ndarray:
    """Normalized hard-wall phi(y,z), in nm^-1; zero outside the rectangle."""
    wy = _positive(width_y_nm, "width_y_nm")
    wz = _positive(width_z_nm, "width_z_nm")
    ny = _integer(ny, "ny")
    nz = _integer(nz, "nz")
    y, z = np.broadcast_arrays(np.asarray(y_nm, dtype=float), np.asarray(z_nm, dtype=float))
    inside = (y >= 0) & (y <= wy) & (z >= 0) & (z <= wz)
    return np.where(inside, 2 / np.sqrt(wy * wz) * np.sin(ny * np.pi * y / wy)
                    * np.sin(nz * np.pi * z / wz), 0.0)


def fermi(energy_ev: ArrayInput, mu_ev: float, temperature_k: float = 300.0) -> np.ndarray:
    """Fermi-Dirac state occupation. T>0; mu and energy share one energy zero."""
    temperature_k = _positive(temperature_k, "temperature_k")
    energy = np.asarray(energy_ev, dtype=float)
    if not np.all(np.isfinite(energy)) or not np.isfinite(mu_ev):
        raise ValueError("Energies and chemical potential must be finite.")
    # exp(-logaddexp(0,x)) equals 1/(1+exp(x)) without overflow.
    return np.exp(-np.logaddexp(0.0, (energy - mu_ev) / (KB_EV * temperature_k)))


def speed_m_s(energy_ev: float, edges_ev: ArrayInput, mass_rel: float = 0.19) -> np.ndarray:
    """Positive axial speeds of OPEN states only; E must exceed every edge."""
    kinetic_ev = float(energy_ev) - np.asarray(edges_ev, dtype=float)
    if not np.all(np.isfinite(kinetic_ev)) or np.any(kinetic_ev <= 0):
        raise ValueError("Every supplied subband must be open: E > edge.")
    return np.sqrt(2 * E_CHARGE * kinetic_ev / (ELECTRON_MASS * _positive(mass_rel, "mass_rel")))


def directional_dos_per_ev_m(
    energy_ev: float, edges_ev: ArrayInput, mass_rel: float = 0.19, spin: int = SPIN,
) -> np.ndarray:
    """One-direction DOS in states/(eV m), including the stated spin factor.

    D_s(E)=g_s/(h |v|) when E is measured in joules. Multiplication by e
    converts the integration variable to eV. The threshold itself is excluded.
    """
    spin = _integer(spin, "spin")
    return spin * E_CHARGE / (PLANCK * speed_m_s(energy_ev, edges_ev, mass_rel))


def _mode_values(value: ArrayInput, modes: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(modes, float(array))
    if array.shape != (modes,) or not np.all(np.isfinite(array)):
        raise ValueError(name + " must be a finite scalar or one value per mode.")
    return array.copy()


def build_kappa(
    modes: int, beta_nm_inv: ArrayInput = 0.01, gamma_nm_inv: float = 0.0,
) -> np.ndarray:
    """Symmetric pair conductances in nm^-1, state order (0+,0-,1+,1-,...).

    Intraband (+,-) pair n has conductance beta_n. Between distinct modes,
    EVERY directional pair has gamma/2. For two modes the total conductance
    from a state to the other mode is gamma, split equally between directions.
    gamma is not by itself a universal microscopic inter-subband event rate.
    """
    modes = _integer(modes, "modes")
    beta = _mode_values(beta_nm_inv, modes, "beta_nm_inv")
    gamma = float(gamma_nm_inv)
    if np.any(beta < 0) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Conductances must be finite and nonnegative.")
    kappa = np.zeros((2 * modes, 2 * modes))
    for n in range(modes):
        kappa[2 * n, 2 * n + 1] = kappa[2 * n + 1, 2 * n] = beta[n]
        for m in range(n):
            kappa[2 * n:2 * n + 2, 2 * m:2 * m + 2] = gamma / 2
            kappa[2 * m:2 * m + 2, 2 * n:2 * n + 2] = gamma / 2
    return kappa


def _validated_kappa(kappa_nm_inv: ArrayInput) -> np.ndarray:
    kappa = np.asarray(kappa_nm_inv, dtype=float)
    if (kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]
            or kappa.shape[0] < 2 or kappa.shape[0] % 2
            or not np.all(np.isfinite(kappa)) or np.any(kappa < 0)
            or not np.allclose(kappa, kappa.T, rtol=1e-12, atol=1e-14)
            or np.any(np.diag(kappa) != 0)):
        raise ValueError("kappa must be an even, symmetric, nonnegative square matrix with zero diagonal.")
    return kappa.copy()


def collision_term(
    occupations: ArrayInput, speeds_nm_s: ArrayInput, kappa_nm_inv: ArrayInput,
) -> np.ndarray:
    """C_a in s^-1 for one energy shell, with positive speeds for each state.

    Sum_a C_a/|v_a|=0 is the appropriate number-conservation check; sum_a C_a
    need not vanish because the available state density differs by subband.
    """
    kappa = _validated_kappa(kappa_nm_inv)
    f = np.asarray(occupations, dtype=float)
    v = np.asarray(speeds_nm_s, dtype=float)
    if (f.shape != (len(kappa),) or v.shape != f.shape or not np.all(np.isfinite(f))
            or not np.all(np.isfinite(v)) or np.any(v <= 0)
            or np.any(f < 0) or np.any(f > 1)):
        raise ValueError("Supply one occupation in [0,1] and one positive speed per state.")
    return v * (kappa @ f - kappa.sum(axis=1) * f)


@dataclass(frozen=True)
class ShellSolution:
    """Finite-volume solution and conservative signed particle-flux factors.

    f has shape (cells, 2*modes); x_nm are cell centers. face_flux_by_mode has
    shape (cells+1,modes). Its sum is constant in x even though a mode's flux
    may change by inter-subband scattering. Multiply total face_flux by g_s/h
    for particle flux per joule (or by g_s*e/h per eV), then integrate in E.
    """
    x_nm: np.ndarray
    f: np.ndarray
    face_x_nm: np.ndarray
    face_flux_by_mode: np.ndarray
    f_left: np.ndarray
    f_right: np.ndarray
    residual_nm_inv: float

    @property
    def face_flux(self) -> np.ndarray:
        return self.face_flux_by_mode.sum(axis=1)

    @property
    def left_outgoing(self) -> np.ndarray:
        return self.f[0, 1::2].copy()

    @property
    def right_outgoing(self) -> np.ndarray:
        return self.f[-1, 0::2].copy()


def _solve_block_tridiagonal(
    diagonal: np.ndarray, lower: np.ndarray, upper: np.ndarray, rhs: np.ndarray,
) -> np.ndarray:
    """Block Thomas elimination: O(N S^3), not a dense (N*S)^3 solve."""
    cells, states = rhs.shape
    modified_upper = np.zeros((max(cells - 1, 0), states, states))
    modified_rhs = np.empty_like(rhs)
    for j in range(cells):
        pivot = diagonal if j == 0 else diagonal - lower @ modified_upper[j - 1]
        right = rhs[j] if j == 0 else rhs[j] - lower @ modified_rhs[j - 1]
        if j < cells - 1:
            solved = np.linalg.solve(pivot, np.column_stack((upper, right)))
            modified_upper[j] = solved[:, :states]
            modified_rhs[j] = solved[:, states]
        else:
            modified_rhs[j] = np.linalg.solve(pivot, right)
    result = modified_rhs.copy()
    for j in range(cells - 2, -1, -1):
        result[j] -= modified_upper[j] @ result[j + 1]
    return result


def solve_shell(
    length_nm: float, kappa_nm_inv: ArrayInput, f_left: ArrayInput,
    f_right: ArrayInput, cells: int = 80,
) -> ShellSolution:
    """Solve a uniform-wire stationary elastic shell with half-range contacts.

    f_left/f_right are reservoir incoming occupations (scalar, or per mode).
    This routine accepts already-selected OPEN modes. It does not create a
    propagating state at/below a subband edge; energy selection belongs to the
    caller. No outgoing contact occupation is prescribed.
    """
    length_nm = _positive(length_nm, "length_nm")
    cells = _integer(cells, "cells")
    kappa = _validated_kappa(kappa_nm_inv)
    states = len(kappa)
    modes = states // 2
    left = _mode_values(f_left, modes, "f_left")
    right = _mode_values(f_right, modes, "f_right")
    if np.any(left < 0) or np.any(left > 1) or np.any(right < 0) or np.any(right > 1):
        raise ValueError("Incoming occupations must lie in [0,1].")
    dx = length_nm / cells
    positive = np.diag(np.tile([1.0, 0.0], modes))
    negative = np.eye(states) - positive
    # Streaming loss plus collision loss; off-diagonal entries transfer gain.
    diagonal = np.eye(states) / dx + np.diag(kappa.sum(axis=1)) - kappa
    lower, upper = -positive / dx, -negative / dx
    rhs = np.zeros((cells, states))
    rhs[0, 0::2] += left / dx
    rhs[-1, 1::2] += right / dx
    f = _solve_block_tridiagonal(diagonal, lower, upper, rhs)

    # Upwind face values use the cell on the side the carrier arrives FROM.
    plus = np.vstack((left, f[:, 0::2]))
    minus = np.vstack((f[:, 1::2], right))
    face_by_mode = plus - minus
    upwind = np.empty_like(f)
    upwind[0, 0::2] = left
    upwind[1:, 0::2] = f[:-1, 0::2]
    upwind[-1, 1::2] = right
    upwind[:-1, 1::2] = f[1:, 1::2]
    # Each directional upwind streaming term has positive magnitude 1/dx.
    residual = (f - upwind) / dx - (f @ kappa.T - f * kappa.sum(axis=1))
    return ShellSolution(
        (np.arange(cells) + 0.5) * dx, f, np.arange(cells + 1) * dx,
        face_by_mode, left, right, float(np.max(np.abs(residual))),
    )


def single_mode_exact(
    x_nm: ArrayInput, length_nm: float, beta_nm_inv: float,
    f_left: float, f_right: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Continuum one-mode occupations and constant f+−f−, not grid values."""
    length = _positive(length_nm, "length_nm")
    beta = float(beta_nm_inv)
    x = np.asarray(x_nm, dtype=float)
    if (not np.isfinite(beta) or beta < 0 or not np.all(np.isfinite(x))
            or np.any(x < 0) or np.any(x > length)):
        raise ValueError("beta >= 0 and 0 <= x <= L are required.")
    if not (np.isfinite(f_left) and np.isfinite(f_right) and 0 <= f_left <= 1 and 0 <= f_right <= 1):
        raise ValueError("Incoming occupations must lie in [0,1].")
    difference = (f_left - f_right) / (1 + beta * length)
    return f_left - beta * x * difference, f_right + beta * (length - x) * difference, difference


def fermi_integral_above_ev(edge_ev: ArrayInput, mu_ev: float, temperature_k: float) -> np.ndarray:
    """Integral_edge^infinity f(E)dE in eV, evaluated analytically."""
    kt = KB_EV * _positive(temperature_k, "temperature_k")
    edge = np.asarray(edge_ev, dtype=float)
    if not np.all(np.isfinite(edge)) or not np.isfinite(mu_ev):
        raise ValueError("Energies must be finite.")
    return kt * np.logaddexp(0.0, (mu_ev - edge) / kt)


def ballistic_current_a(
    edges_ev: ArrayInput, mu_left_ev: float, mu_right_ev: float,
    temperature_k: float = 300.0, spin: int = SPIN,
) -> float:
    """Conventional charge current along +x for uniform unit-transmission modes.

    With mu_left > mu_right, electrons flow +x and this current is NEGATIVE.
    The result includes contact injection resistance. It is not an ohmic
    field-driven current inside an infinite wire.
    """
    spin = _integer(spin, "spin")
    integral_ev = np.sum(fermi_integral_above_ev(edges_ev, mu_left_ev, temperature_k)
                         - fermi_integral_above_ev(edges_ev, mu_right_ev, temperature_k))
    return float(-spin * E_CHARGE**2 / PLANCK * integral_ev)


def integrate_current(
    edges_ev: ArrayInput, length_nm: float, beta_nm_inv: ArrayInput,
    gamma_nm_inv: float, mu_left_ev: float, mu_right_ev: float,
    temperature_k: float = 300.0, cells: int = 80, quadrature: int = 24,
    cutoff_ev: Optional[float] = None, spin: int = SPIN,
    integration: str = "analytic",
) -> dict:
    """Integrate conservative face flux across TOTAL energy, in SI amperes.

    kappa is energy-independent within each open-mode interval. By linearity,
    solve its unit-injection problem once, then integrate f_L(E)-f_R(E).
    The default integration='analytic' uses exact Fermi integrals over each
    interval, avoiding unresolved low-temperature contact steps. The optional
    integration='gauss' uses Gauss-Legendre quadrature, additionally split near
    both chemical potentials on the kBT scale. It never samples a threshold
    itself. All subband edges split the energy grid; closed modes never enter
    a collision graph. quadrature is validated for either method, but sets the
    node count only for 'gauss'.
    The default upper cutoff is max(mu_L,mu_R,max edge)+24 kBT. Callers should
    verify spatial grid, cutoff AND retained-subband convergence; when using
    'gauss', verify quadrature convergence as well.
    """
    edges = np.asarray(edges_ev, dtype=float)
    if edges.ndim != 1 or not len(edges) or not np.all(np.isfinite(edges)):
        raise ValueError("Supply a nonempty finite array of subband edges.")
    beta = _mode_values(beta_nm_inv, len(edges), "beta_nm_inv")
    if np.any(beta < 0):
        raise ValueError("beta must be nonnegative.")
    kt = KB_EV * _positive(temperature_k, "temperature_k")
    quadrature = _integer(quadrature, "quadrature")
    spin = _integer(spin, "spin")
    if integration not in ("analytic", "gauss"):
        raise ValueError("integration must be 'analytic' or 'gauss'.")
    if not np.isfinite(mu_left_ev) or not np.isfinite(mu_right_ev):
        raise ValueError("Chemical potentials must be finite.")
    upper = float(cutoff_ev) if cutoff_ev is not None else max(max(edges), mu_left_ev, mu_right_ev) + 24 * kt
    if not np.isfinite(upper) or upper <= min(edges):
        raise ValueError("cutoff_ev must exceed the lowest edge.")
    # The problem is uniform in x and has equal contact occupation per open
    # mode, so the unit-injection response factors out of the energy integral.
    boundaries = np.unique(np.r_[edges[edges < upper], upper])
    if integration == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    integral_ev = 0.0
    intervals = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        active = np.flatnonzero(edges < (lo + hi) / 2)
        kappa = build_kappa(len(active), beta[active], gamma_nm_inv)
        shell = solve_shell(length_nm, kappa, 1.0, 0.0, cells)
        transmission = float(np.mean(shell.face_flux))
        if integration == "analytic":
            # A_mu(a)=integral_a^infinity f_FD(E;mu,T)dE.  Subtraction gives
            # the finite interval integral, not an extrapolation past cutoff.
            left_integral = (fermi_integral_above_ev(lo, mu_left_ev, temperature_k)
                             - fermi_integral_above_ev(hi, mu_left_ev, temperature_k))
            right_integral = (fermi_integral_above_ev(lo, mu_right_ev, temperature_k)
                              - fermi_integral_above_ev(hi, mu_right_ev, temperature_k))
            injection_integral = float(left_integral - right_integral)
        else:
            # An edge-only grid can miss a sharp Fermi step inside a wide
            # interval. Resolve both contact steps even when kBT is tiny.
            thermal_offsets = np.array([-24, -12, -4, -1, 0, 1, 4, 12, 24]) * kt
            near_contacts = np.r_[mu_left_ev + thermal_offsets, mu_right_ev + thermal_offsets]
            breaks = np.unique(np.r_[lo, near_contacts[(near_contacts > lo) & (near_contacts < hi)], hi])
            injection_integral = 0.0
            for start, stop in zip(breaks[:-1], breaks[1:]):
                energy = (stop + start) / 2 + (stop - start) / 2 * nodes
                injection = (fermi(energy, mu_left_ev, temperature_k)
                             - fermi(energy, mu_right_ev, temperature_k))
                injection_integral += (stop - start) / 2 * float(weights @ injection)
        integral_ev += transmission * injection_integral
        intervals.append({"low_ev": float(lo), "high_ev": float(hi),
                          "open_modes": active.tolist(), "transmission": transmission})
    electron_rate = float(spin * E_CHARGE / PLANCK * integral_ev)
    return {"current_a": float(-E_CHARGE * electron_rate),
            "electron_rate_s": electron_rate, "integral_ev": integral_ev,
            "cutoff_ev": upper, "integration": integration, "intervals": intervals}


def demo() -> None:
    """Small reproducible benchmark; no external data or plot backend needed."""
    edges = subband_edges(10, 8, ((1, 1), (2, 1), (1, 2)))
    print("Rectangular-wire edges (eV):", np.round(edges, 6))
    exact = single_mode_exact([0, 50, 100], 100, 0.01, 0.8, 0.2)[2]
    print("One-mode continuum spectral flux:", exact)
    for cells in (20, 80, 320):
        result = solve_shell(100, build_kappa(1, 0.01), 0.8, 0.2, cells)
        print("cells=%3d: face flux=%.9f, variation=%.3g" % (
            cells, np.mean(result.face_flux), np.ptp(result.face_flux)))
    params = dict(edges_ev=edges, mu_left_ev=0.16, mu_right_ev=0.12, temperature_k=300)
    ballistic = ballistic_current_a(**params)
    scattered = integrate_current(**params, length_nm=100,
                                  beta_nm_inv=[0.01, 0.015, 0.02], gamma_nm_inv=0.004)
    print("Ballistic current along +x (A):", ballistic)
    print("Elastic-model current along +x (A):", scattered["current_a"])


if __name__ == "__main__":
    demo()
