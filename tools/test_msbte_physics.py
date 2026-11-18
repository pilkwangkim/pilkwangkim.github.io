#!/usr/bin/env python3
"""Analytical and conservation checks for the educational MS-BTE reference."""

from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("msbte_reference", ROOT / "assets/code/multi-subband-bte.py")
assert SPEC and SPEC.loader
bte = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bte
SPEC.loader.exec_module(bte)


class OpeningConceptChecks(unittest.TestCase):
    """Check the opening examples independently of their inline assertions."""

    @classmethod
    def setUpClass(cls):
        article = ROOT / "_posts/2026-4Q/2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch.md"
        snippets = re.findall(r"^```python\n(.*?)^```", article.read_text(encoding="utf-8"), re.M | re.S)
        cls.examples = {}
        for key, marker in (("kinetic", "def packet_occupation("),
                            ("storage", "full_values = 100**6"),
                            ("density", "rho = np.outer(c, c.conj())")):
            matches = [snippet for snippet in snippets if marker in snippet]
            if len(matches) != 1:
                raise AssertionError("Expected one opening code example: " + key)
            namespace = {"np": np, "__name__": "msbte_opening_" + key}
            with redirect_stdout(io.StringIO()):
                exec(compile(matches[0], str(article), "exec"), namespace)
            cls.examples[key] = namespace

    def test_gaussian_translation_preserves_peak_width_and_excess_area(self):
        packet = self.examples["kinetic"]["packet_occupation"]
        x = np.linspace(-300, 300, 12001)
        dx = x[1] - x[0]
        expected_excess_area = 0.8 * 12 * np.sqrt(2 * np.pi)
        for t, centre in ((0, -30), (1, -10), (2, 10), (4, 50)):
            self.assertAlmostEqual(float(packet(centre, t)), 0.9, places=14)
            self.assertAlmostEqual(float(packet(centre + 12, t)), 0.1 + 0.8 * np.exp(-0.5), places=14)
            np.testing.assert_allclose(packet(x, t), packet(x - 20 * t, 0), atol=1e-14)
            values = packet(x, t)
            self.assertGreaterEqual(float(values.min()), 0.1)
            self.assertLessEqual(float(values.max()), 0.9)
            # The conserved quantity here is area ABOVE the uniform 0.1
            # background, not normalization of f to one electron.
            excess_area = float(np.sum(values - 0.1) * dx)
            self.assertAlmostEqual(excess_area, expected_excess_area, delta=1e-10)
        self.assertNotAlmostEqual(expected_excess_area, 1.0)

    def test_gaussian_satisfies_fixed_velocity_bte_by_numerical_derivatives(self):
        packet = self.examples["kinetic"]["packet_occupation"]
        dx, dt, velocity = 1e-4, 1e-6, 20.0  # nm, ps, nm/ps.
        for x in (-50.0, -10.0, 15.0, 50.0):
            for t in (0.25, 1.25, 3.75):
                spatial = float(packet(x + dx, t) - packet(x - dx, t)) / (2 * dx)
                temporal = float(packet(x, t + dt) - packet(x, t - dt)) / (2 * dt)
                centre = -30 + velocity * t
                exact_spatial = -(x - centre) / 12**2 * (float(packet(x, t)) - 0.1)
                self.assertAlmostEqual(spatial, exact_spatial, delta=1e-9)
                self.assertAlmostEqual(temporal, -velocity * exact_spatial, delta=1e-8)
                self.assertAlmostEqual(temporal + velocity * spatial, 0, delta=2e-8)

    def test_homogeneous_reciprocal_pauli_gain_loss_is_conservative(self):
        # These equally weighted states conserve fa+fb for any initial sum;
        # the grid includes sums other than one without renormalization.
        for fa in (0.0, 0.1, 0.5, 0.9, 1.0):
            for fb in (0.0, 0.2, 0.5, 0.8, 1.0):
                for rate in (0.0, 0.4, 1.0):
                    gain = rate * fb * (1 - fa)
                    loss = rate * fa * (1 - fb)
                    ca = gain - loss
                    cb = loss - gain
                    self.assertAlmostEqual(ca, rate * (fb - fa), places=14)
                    self.assertAlmostEqual(ca + cb, 0, places=14)
    def test_homogeneous_pair_matches_ode_and_factor_two_decay(self):
        pair = self.examples["kinetic"]["relaxing_pair"]
        times = np.linspace(0, 4, 81)
        for rate in (0.0, 0.4, 1.0):
            fp, fm = pair(times, rate)
            np.testing.assert_allclose(fp + fm, 1, atol=1e-14)
            np.testing.assert_allclose(fp - fm, 0.6 * np.exp(-2 * rate * times), atol=1e-14)
            np.testing.assert_allclose((fp + fm) / 2, 0.5, atol=1e-14)
            self.assertGreaterEqual(float(fm.min()), 0.2)
            self.assertLessEqual(float(fp.max()), 0.8)
            dt, t = 1e-5, 1.25
            a, b = pair(t - dt, rate), pair(t + dt, rate)
            current = pair(t, rate)
            expected = rate * (current[1] - current[0])
            self.assertAlmostEqual(float((b[0] - a[0]) / (2 * dt)), expected, delta=1e-10)
            self.assertAlmostEqual(float((b[1] - a[1]) / (2 * dt)), -expected, delta=1e-10)
        initial = pair(0, 0.4)
        np.testing.assert_allclose(initial, [0.8, 0.2], atol=1e-14)
        at_one_decay_time = pair(1.25, 0.4)
        self.assertAlmostEqual(float(at_one_decay_time[0] - at_one_decay_time[1]), 0.6 / np.e, places=14)
        np.testing.assert_allclose(pair(times, 0)[0], 0.8, atol=1e-14)
        np.testing.assert_allclose(pair(times, 0)[1], 0.2, atol=1e-14)

    def test_complete_density_matrix_rotation_preserves_state_and_observables(self):
        example = self.examples["density"]
        rho, rotated, unitary = example["rho"], example["M"], example["U"]
        self.assertAlmostEqual(float(example["theta"]), np.pi / 6, places=14)
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(unitary @ rotated @ unitary.conj().T, rho, atol=1e-14)
        for matrix in (rho, rotated):
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)
            self.assertAlmostEqual(float(np.trace(matrix)), 1, places=14)
            self.assertAlmostEqual(float(np.trace(matrix @ matrix)), 1, places=14)
        observable = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
        rotated_observable = unitary.conj().T @ observable @ unitary
        expected_value = 2 * np.sqrt(0.6 * 0.1)
        self.assertAlmostEqual(float(np.trace(rho @ observable)), expected_value, places=14)
        self.assertAlmostEqual(float(np.trace(rotated @ rotated_observable)), expected_value, places=14)
        # Dropping coherence is not the same operation as rotating coordinates.
        self.assertAlmostEqual(float(np.trace(example["D"] @ rotated_observable)), 0, places=14)

    def test_diagonal_approximation_keeps_populations_but_reduces_purity(self):
        example = self.examples["density"]
        rotated, diagonal = example["M"], example["D"]
        # Closed-form probabilities for the specified theta=pi/6 rotation,
        # independent of the matrix multiplications performed by the snippet.
        p = np.array([0.525 + 0.15 * np.sqrt(6),
                      0.375 - 0.15 * np.sqrt(6), 0.1])
        np.testing.assert_allclose(np.diag(rotated), p, atol=1e-14)
        np.testing.assert_allclose(diagonal, np.diag(p), atol=1e-14)
        self.assertGreater(float(np.max(np.abs(rotated - diagonal))), 0.2)
        self.assertAlmostEqual(float(np.trace(diagonal)), 1, places=14)
        expected_purity = 0.69625 + 0.045 * np.sqrt(6)
        self.assertAlmostEqual(float(np.trace(diagonal @ diagonal)), expected_purity, places=14)
        self.assertLess(expected_purity, 1)

    def test_projection_exposes_omitted_weight_without_renormalizing(self):
        example = self.examples["density"]
        projection, diagonal, retained = example["P"], example["D"], example["R"]
        np.testing.assert_array_equal(projection, np.diag([1., 1., 0.]))
        np.testing.assert_array_equal(projection @ projection, projection)
        np.testing.assert_allclose(retained, projection @ diagonal @ projection, atol=1e-14)
        np.testing.assert_allclose(np.diag(retained)[:2], np.diag(diagonal)[:2], atol=1e-14)
        self.assertAlmostEqual(float(np.trace(retained)), 0.9, places=14)
        self.assertAlmostEqual(float(np.trace(diagonal - retained)), 0.1, places=14)
        self.assertAlmostEqual(float(np.trace(retained @ retained)),
                               0.68625 + 0.045 * np.sqrt(6), places=14)
        self.assertNotAlmostEqual(float(np.trace(retained)), 1)

    def test_representation_storage_count_uses_decimal_units_without_allocation(self):
        example = self.examples["storage"]
        self.assertEqual(example["full_values"], 10**12)
        self.assertEqual(example["subband_values"], 10**5)
        item_bytes = np.dtype(np.float64).itemsize
        self.assertEqual(item_bytes, 8)
        self.assertEqual(item_bytes * example["full_values"], 8 * 10**12)
        self.assertEqual(item_bytes * example["subband_values"], 800_000)
        self.assertAlmostEqual(item_bytes * example["full_values"] / 1e12, 8.0)
        self.assertAlmostEqual(item_bytes * example["subband_values"] / 1e6, 0.8)
        self.assertEqual(example["full_values"] // example["subband_values"], 10**7)


class SubbandChecks(unittest.TestCase):
    def test_hard_wall_scaling_and_ground_state(self):
        e = bte.well_energy_ev(1, 10, 0.19)
        self.assertAlmostEqual(e, 0.019791061154, delta=2e-9)
        self.assertAlmostEqual(bte.well_energy_ev(2, 10, 0.19), 4 * e, places=13)
        self.assertAlmostEqual(bte.well_energy_ev(1, 20, 0.19), e / 4, places=13)
        self.assertAlmostEqual(bte.well_energy_ev(1, 10, 0.38), e / 2, places=13)
        np.testing.assert_allclose(bte.subband_edges(10, 8, ((1, 1), (2, 1))),
                                   [e + e * (10 / 8)**2, 4 * e + e * (10 / 8)**2])

    def test_wavefunctions_normalized_and_orthogonal(self):
        y = np.linspace(0, 10, 301)
        z = np.linspace(0, 8, 241)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        phi11 = bte.transverse_wavefunction(yy, zz, 10, 8)
        phi21 = bte.transverse_wavefunction(yy, zz, 10, 8, ny=2)
        # Uniform-grid trapezoid with zero hard-wall endpoint values.
        area = (y[1] - y[0]) * (z[1] - z[0])
        self.assertAlmostEqual(float(np.sum(phi11**2) * area), 1, places=12)
        self.assertAlmostEqual(float(np.sum(phi21**2) * area), 1, places=12)
        self.assertAlmostEqual(float(np.sum(phi11 * phi21) * area), 0, places=12)

    def test_complete_mode_enumeration_preserves_labels(self):
        modes, edges = bte.rectangular_modes_below(0.25, 10, 8)
        expected = sorted((bte.well_energy_ev(ny, 10) + bte.well_energy_ev(nz, 8), ny, nz)
                          for ny in range(1, 10) for nz in range(1, 10)
                          if bte.well_energy_ev(ny, 10) + bte.well_energy_ev(nz, 8) < 0.25)
        self.assertEqual(modes, [(ny, nz) for _, ny, nz in expected])
        np.testing.assert_allclose(edges, [e for e, _, _ in expected])
        self.assertEqual(bte.rectangular_modes_below(0.01, 10, 8)[0], [])

    def test_dos_jacobian_and_threshold(self):
        speeds = bte.speed_m_s(0.15, [0.04, 0.12])
        dos = bte.directional_dos_per_ev_m(0.15, [0.04, 0.12])
        np.testing.assert_allclose(dos * speeds, bte.SPIN * bte.E_CHARGE / bte.PLANCK)
        self.assertGreater(dos[1], dos[0])
        with self.assertRaises(ValueError):
            bte.speed_m_s(0.12, [0.12])


class CollisionChecks(unittest.TestCase):
    def test_equilibrium_and_weighted_number_conservation(self):
        kappa = bte.build_kappa(2, [0.01, 0.03], 0.008)
        speeds = np.repeat(bte.speed_m_s(0.15, [0.04, 0.12]) * 1e9, 2)
        np.testing.assert_allclose(bte.collision_term([0.37] * 4, speeds, kappa), 0, atol=5e-4)
        f = np.array([0.82, 0.13, 0.49, 0.66])
        c = bte.collision_term(f, speeds, kappa)
        self.assertAlmostEqual(float(np.sum(c / speeds)), 0, places=14)
        # Unweighted occupation derivative need not conserve particle number.
        self.assertGreater(abs(float(np.sum(c))), 1e9)

    def test_elastic_pauli_cancellation(self):
        f = np.array([0, 0.2, 0.6, 1.0])
        for a in f:
            for b in f:
                self.assertAlmostEqual(b * (1 - a) - a * (1 - b), b - a)

    def test_closed_two_state_exact_relaxation(self):
        # g1 C1 + g2 C2=0; equilibrium occupations equal, not populations.
        v1, v2, kappa = 2.0, 1.0, 0.3
        f1, f2 = 0.9, 0.2
        equilibrium = (f1 / v1 + f2 / v2) / (1 / v1 + 1 / v2)
        for t in (0, 0.2, 2, 100):
            difference = (f1 - f2) * np.exp(-(v1 + v2) * kappa * t)
            a = equilibrium + v1 / (v1 + v2) * difference
            b = equilibrium - v2 / (v1 + v2) * difference
            self.assertAlmostEqual(a / v1 + b / v2, f1 / v1 + f2 / v2)
            self.assertGreaterEqual(min(a, b), 0)
            self.assertLessEqual(max(a, b), 1)


class TeachingExampleChecks(unittest.TestCase):
    def test_density_matching_rta_requires_common_relaxation_time(self):
        weights, f = np.array([2., 1.]), np.array([.8, .2])
        target = np.full(2, weights @ f / weights.sum())
        self.assertAlmostEqual(float(weights @ (target - f)), 0, places=14)
        self.assertAlmostEqual(float(weights @ ((target - f) / .4)), 0, places=14)
        times = np.array([.4, .8])
        self.assertAlmostEqual(float(weights @ ((target - f) / times)), -.5, places=14)
        # A state-dependent rate requires cancellation under the state measure
        # weighted by 1/tau, not just an unweighted density match.
        conserving_target = (weights / times) @ f / (weights / times).sum()
        self.assertAlmostEqual(float(weights @ ((conserving_target - f) / times)), 0, places=14)

    def test_coherent_and_mixed_states_share_populations_but_not_density(self):
        # Nonzero allowed periodic momentum makes the +/- waves orthogonal.
        length, count = 1.0, 4096
        x = np.arange(count) * length / count
        dx, k = length / count, 2 * np.pi * 3 / length
        plus = np.exp(1j * k * x) / np.sqrt(length)
        minus = np.exp(-1j * k * x) / np.sqrt(length)
        coherent = (plus + minus) / np.sqrt(2)
        density_coherent = np.abs(coherent)**2
        density_mixture = 0.5 * np.abs(plus)**2 + 0.5 * np.abs(minus)**2
        np.testing.assert_allclose(density_coherent, (1 + np.cos(2 * k * x)) / length, atol=2e-14)
        np.testing.assert_allclose(density_mixture, 1 / length, atol=1e-14)
        self.assertAlmostEqual(float(dx * density_coherent.sum()), 1, places=13)
        self.assertAlmostEqual(float(dx * density_mixture.sum()), 1, places=13)
        for basis in (plus, minus):
            self.assertAlmostEqual(abs(dx * np.vdot(basis, coherent))**2, 0.5, places=13)
        # psi is real here, so Im(psi* dpsi/dx), proportional to its current,
        # is zero; the mixture's equal opposite-momentum fluxes also cancel.
        derivative = (1j * k * plus - 1j * k * minus) / np.sqrt(2)
        np.testing.assert_allclose(np.imag(np.conj(coherent) * derivative), 0, atol=1e-13)
        self.assertGreater(float(np.ptp(density_coherent)), 1.9)

    def test_inelastic_detailed_balance_matches_traffic_not_bare_rates(self):
        mu, temperature, down = 0.1, 300.0, 4.2
        ea, eb = mu, mu + bte.KB_EV * temperature
        fa, fb = bte.fermi([ea, eb], mu, temperature)
        up = np.exp(-1) * down
        forward = up * fa * (1 - fb)
        reverse = down * fb * (1 - fa)
        expected = down / (2 * (1 + np.exp(1)))
        self.assertAlmostEqual(float(fa), 0.5, places=14)
        self.assertAlmostEqual(float(fb), 1 / (1 + np.exp(1)), places=14)
        self.assertAlmostEqual(float(forward), expected, places=14)
        self.assertAlmostEqual(float(reverse), expected, places=14)
        self.assertNotEqual(up, down)

    def test_one_cell_faces_differ_from_cell_center_difference(self):
        solution = bte.solve_shell(100, bte.build_kappa(1, 0.02), 0.9, 0.2, cells=1)
        np.testing.assert_allclose(solution.f[0], [0.62, 0.48], atol=1e-14)
        np.testing.assert_allclose(solution.face_flux, [0.42, 0.42], atol=1e-14)
        self.assertAlmostEqual(float(solution.f[0, 0] - solution.f[0, 1]), 0.14, places=14)

    def test_dos_window_and_spectral_current_units(self):
        edges = bte.subband_edges(10, 8, ((1, 1), (2, 1)))
        states_per_um = bte.directional_dos_per_ev_m(0.130, edges) * 0.001 * 1e-6
        # Narrow-window, constant-DOS approximation; spin two, ONE direction.
        np.testing.assert_allclose(states_per_um, [1.26, 2.52], rtol=0, atol=0.01)
        speeds = bte.speed_m_s(0.130, edges)
        np.testing.assert_allclose(speeds, [3.83e5, 1.92e5], rtol=0.003)
        electron_current_a = bte.SPIN * bte.E_CHARGE**2 / bte.PLANCK * 0.2 * 0.001
        self.assertAlmostEqual(electron_current_a * 1e9, 15.4961834597, delta=1e-8)

    def test_runnable_energy_bin_current_matches_reference_at_same_spatial_grid(self):
        # Execute the EN article's complete snippet chain without writing files.
        # A passing snippet's own assertion is not enough: compare its scattered
        # current against the reference's independent analytic energy integral.
        article = ROOT / "_posts/2026-4Q/2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch.md"
        snippets = re.findall(r"^```python\n(.*?)^```", article.read_text(encoding="utf-8"), re.M | re.S)
        namespace = {"__name__": "msbte_teaching_examples"}
        with redirect_stdout(io.StringIO()):
            for snippet in snippets:
                exec(compile(snippet, str(article), "exec"), namespace)
        teaching_current = namespace["current_from_shells_A"]
        edges = namespace["edges"]
        reference = bte.integrate_current(edges, 100, 0.02, 0.01, 0.150, 0.100,
                                          temperature_k=300, cells=24)
        exact_at_fixed_grid = -reference["current_a"]
        errors = []
        for bins in (32, 128, 512):
            result = teaching_current(edges, 0.150, 0.100, bins=bins)
            self.assertGreater(result, 0)
            errors.append(abs(result - exact_at_fixed_grid))
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1] / exact_at_fixed_grid, 3e-4)


class SolverChecks(unittest.TestCase):
    def test_ballistic_half_range_inflow(self):
        solution = bte.solve_shell(100, bte.build_kappa(2, 0), [0.8, 0.6], [0.2, 0.1], cells=17)
        np.testing.assert_allclose(solution.f, np.tile([0.8, 0.2, 0.6, 0.1], (17, 1)))
        np.testing.assert_allclose(solution.face_flux, 1.1)
        np.testing.assert_allclose(solution.left_outgoing, [0.2, 0.1])
        np.testing.assert_allclose(solution.right_outgoing, [0.8, 0.6])

    def test_equilibrium_zero_current_at_any_scattering(self):
        for cells in (1, 3, 80):
            solution = bte.solve_shell(100, bte.build_kappa(3, [0.01, 0.1, 1.0], 0.2), 0.43, 0.43, cells)
            np.testing.assert_allclose(solution.f, 0.43, atol=3e-12)
            np.testing.assert_allclose(solution.face_flux, 0, atol=3e-12)

    def test_positive_and_conservative_with_mode_redistribution(self):
        solution = bte.solve_shell(100, bte.build_kappa(2, [0.003, 0.08], 0.02), [0.9, 0.3], [0.1, 0.2], 93)
        self.assertGreaterEqual(float(solution.f.min()), 0.1 - 1e-12)
        self.assertLessEqual(float(solution.f.max()), 0.9 + 1e-12)
        self.assertLess(float(np.ptp(solution.face_flux)), 2e-12)
        self.assertLess(solution.residual_nm_inv, 2e-13)
        self.assertGreater(float(np.ptp(solution.face_flux_by_mode[:, 0])), 0.05)
        # Current continuity belongs to faces, not the cell-sum approximation.
        cell_sum = solution.f[:, 0::2].sum(1) - solution.f[:, 1::2].sum(1)
        self.assertGreater(float(np.ptp(cell_sum)), 1e-5)

    def test_one_mode_discrete_formula_and_first_order_convergence(self):
        exact = (0.8 - 0.2) / (1 + 0.02 * 100)
        errors = []
        for cells in (10, 20, 40, 80, 160):
            solution = bte.solve_shell(100, bte.build_kappa(1, 0.02), 0.8, 0.2, cells)
            dx = 100 / cells
            discrete = (0.8 - 0.2) * (1 + 0.02 * dx) / (1 + 0.02 * 100 + 0.02 * dx)
            np.testing.assert_allclose(solution.face_flux, discrete, atol=1e-13)
            errors.append(abs(float(solution.face_flux[0]) - exact))
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 1.9)
            self.assertLess(coarse / fine, 2.01)
        solution = bte.solve_shell(100, bte.build_kappa(1, 0.02), 0.8, 0.2, 160)
        fp, fm, _ = bte.single_mode_exact(solution.x_nm, 100, 0.02, 0.8, 0.2)
        self.assertLess(float(np.max(abs(solution.f[:, 0] - fp))), 0.002)
        self.assertLess(float(np.max(abs(solution.f[:, 1] - fm))), 0.002)

    def test_contact_exchange_reverses_current(self):
        kappa = bte.build_kappa(2, [0.01, 0.04], 0.005)
        forward = bte.solve_shell(100, kappa, 0.8, 0.2, 40)
        reverse = bte.solve_shell(100, kappa, 0.2, 0.8, 40)
        np.testing.assert_allclose(forward.face_flux, -reverse.face_flux, atol=1e-13)

    def test_two_mode_continuum_profiles_converge_to_independent_analytic_solution(self):
        # Independent continuum benchmark, not the discrete matrix assembled by
        # the solver. Equal beta permits decomposition into the mean of the
        # modes and half their difference. The mean obeys the one-mode system
        # with beta_eff=beta+gamma/2; the difference obeys dA/dx=M A.
        length, beta, gamma = 100.0, 0.005, 0.003
        left = np.array([0.9, 0.3])
        right = np.array([0.2, 0.2])
        beta_eff = beta + gamma / 2
        delta_mean = (left.mean() - right.mean()) / (1 + beta_eff * length)
        alpha, zeta = beta + 1.5 * gamma, beta - 0.5 * gamma
        matrix = np.array([[-alpha, zeta], [-zeta, alpha]])
        eigenvalue = np.sqrt(alpha**2 - zeta**2)

        def propagator(x):
            # M^2=lambda^2 I, so exp(Mx) is exact in this closed form.
            return (np.cosh(eigenvalue * x) * np.eye(2)
                    + np.sinh(eigenvalue * x) / eigenvalue * matrix)

        a_plus_left = (left[0] - left[1]) / 2
        a_minus_right = (right[0] - right[1]) / 2
        transfer = propagator(length)
        a_minus_left = (a_minus_right - transfer[1, 0] * a_plus_left) / transfer[1, 1]
        initial = np.array([a_plus_left, a_minus_left])

        def continuum(x):
            mean_plus = left.mean() - beta_eff * x * delta_mean
            mean_minus = right.mean() + beta_eff * (length - x) * delta_mean
            a_plus, a_minus = propagator(x) @ initial
            return [mean_plus + a_plus, mean_minus + a_minus,
                    mean_plus - a_plus, mean_minus - a_minus]

        # Check that the independent solution has the required incoming data.
        np.testing.assert_allclose(np.array(continuum(0))[0::2], left, atol=1e-14)
        np.testing.assert_allclose(np.array(continuum(length))[1::2], right, atol=1e-14)
        exact_flux = 2 * delta_mean
        profile_errors, flux_errors = [], []
        for cells in (20, 40, 80, 160, 320):
            solution = bte.solve_shell(length, bte.build_kappa(2, beta, gamma), left, right, cells)
            expected = np.array([continuum(x) for x in solution.x_nm])
            profile_errors.append(float(np.max(np.abs(solution.f - expected))))
            flux_errors.append(abs(float(solution.face_flux[0]) - exact_flux))
            self.assertLess(float(np.ptp(solution.face_flux)), 1e-12)
        # Both full profiles and integrated mode flux approach the continuum
        # answer at first order, rather than merely satisfying a residual.
        for errors in (profile_errors, flux_errors):
            for coarse, fine in zip(errors, errors[1:]):
                self.assertGreater(coarse / fine, 1.8)
                self.assertLess(coarse / fine, 2.05)
        self.assertLess(profile_errors[-1], 0.0007)
        self.assertLess(flux_errors[-1], 0.0004)

    def test_input_validation(self):
        for args in ((100, [[0, 1], [2, 0]], 0.8, 0.2),
                     (100, [[0, -1], [-1, 0]], 0.8, 0.2),
                     (100, [[0, 1], [1, 0]], 1.1, 0.2)):
            with self.assertRaises(ValueError):
                bte.solve_shell(*args)

    def test_random_positive_graphs_obey_maximum_principle(self):
        rng = np.random.default_rng(416)
        for states in (2, 4, 6, 8):
            raw = rng.uniform(0, 2, (states, states))
            kappa = (raw + raw.T) / 2
            np.fill_diagonal(kappa, 0)
            left = rng.uniform(0, 1, states // 2)
            right = rng.uniform(0, 1, states // 2)
            solution = bte.solve_shell(100, kappa, left, right, cells=20)
            self.assertGreaterEqual(float(solution.f.min()), min(min(left), min(right)) - 1e-11)
            self.assertLessEqual(float(solution.f.max()), max(max(left), max(right)) + 1e-11)
            self.assertLess(float(np.ptp(solution.face_flux)), 1e-11)


class CurrentChecks(unittest.TestCase):
    def test_ballistic_landauer_integration_and_sign(self):
        edges = [0.05, 0.12, 0.21]
        params = dict(edges_ev=edges, mu_left_ev=0.18, mu_right_ev=0.13, temperature_k=300)
        numerical = bte.integrate_current(**params, length_nm=100, beta_nm_inv=0,
                                         gamma_nm_inv=0, quadrature=48)
        exact = bte.ballistic_current_a(**params)
        self.assertLess(exact, 0)
        self.assertAlmostEqual(numerical["current_a"] / exact, 1, delta=2e-10)
        for interval in numerical["intervals"]:
            self.assertAlmostEqual(interval["transmission"], len(interval["open_modes"]))

    def test_closed_mode_never_scatters(self):
        # Upper mode is entirely outside the integration window: changing its
        # beta or gamma cannot affect a one-open-mode graph below that edge.
        params = dict(length_nm=100, mu_left_ev=0.10, mu_right_ev=0.08,
                      temperature_k=300, cutoff_ev=0.2, cells=30)
        a = bte.integrate_current([0.04], beta_nm_inv=0.01, gamma_nm_inv=0, **params)
        b = bte.integrate_current([0.04, 0.5], beta_nm_inv=[0.01, 100], gamma_nm_inv=100, **params)
        self.assertAlmostEqual(a["current_a"], b["current_a"], places=16)

    def test_quadrature_cutoff_and_grid_convergence(self):
        params = dict(edges_ev=[0.05, 0.12], length_nm=100,
                      beta_nm_inv=[0.01, 0.03], gamma_nm_inv=0.005,
                      mu_left_ev=0.15, mu_right_ev=0.12, temperature_k=300)
        a = bte.integrate_current(**params, cells=80, quadrature=8, cutoff_ev=0.9, integration="gauss")
        b = bte.integrate_current(**params, cells=80, quadrature=24, cutoff_ev=1.2, integration="gauss")
        self.assertAlmostEqual(a["current_a"] / b["current_a"], 1, delta=1e-8)
        values = [bte.integrate_current(**params, cells=n)["current_a"] for n in (40, 80, 160)]
        self.assertLess(abs(values[2] - values[1]), abs(values[1] - values[0]))

    def test_low_temperature_fermi_steps_have_analytic_and_resolved_gauss_integrals(self):
        length, beta, cells = 100, 0.01, 80
        dx = length / cells
        transmission = (1 + beta * dx) / (1 + beta * length + beta * dx)
        for temperature in (1, 10, 300):
            params = dict(edges_ev=[0.05], length_nm=length, beta_nm_inv=beta,
                          gamma_nm_inv=0, mu_left_ev=0.15, mu_right_ev=0.12,
                          temperature_k=temperature, cells=cells)
            expected = transmission * bte.ballistic_current_a([0.05], 0.15, 0.12, temperature)
            analytic = bte.integrate_current(**params)
            gauss = bte.integrate_current(**params, integration="gauss", quadrature=24)
            self.assertEqual(analytic["integration"], "analytic")
            self.assertEqual(gauss["integration"], "gauss")
            self.assertAlmostEqual(analytic["current_a"] / expected, 1, delta=2e-10)
            self.assertAlmostEqual(gauss["current_a"] / expected, 1, delta=2e-10)

    def test_finite_cutoff_error_matches_omitted_fermi_tail(self):
        edge, cutoff, mu_left, mu_right, temperature = 0.05, 0.16, 0.15, 0.12, 300
        length, beta, cells = 100, 0.01, 80
        dx = length / cells
        transmission = (1 + beta * dx) / (1 + beta * length + beta * dx)
        full = transmission * bte.ballistic_current_a([edge], mu_left, mu_right, temperature)
        omitted_tail = transmission * bte.ballistic_current_a([cutoff], mu_left, mu_right, temperature)
        truncated = bte.integrate_current([edge], length, beta, 0, mu_left, mu_right,
                                         temperature, cells=cells, cutoff_ev=cutoff)
        self.assertAlmostEqual(truncated["current_a"], full - omitted_tail, delta=1e-18)
        self.assertGreater(abs(omitted_tail), 0.1 * abs(full))
        extended = bte.integrate_current([edge], length, beta, 0, mu_left, mu_right,
                                        temperature, cells=cells, cutoff_ev=0.9)
        self.assertAlmostEqual(extended["current_a"], full, delta=1e-18)

    def test_integration_method_and_quadrature_are_validated(self):
        params = dict(edges_ev=[0.05], length_nm=100, beta_nm_inv=0.01,
                      gamma_nm_inv=0, mu_left_ev=0.15, mu_right_ev=0.12)
        with self.assertRaises(ValueError):
            bte.integrate_current(**params, integration="automatic")
        with self.assertRaises(ValueError):
            bte.integrate_current(**params, quadrature=0)


class NumericalAuditChecks(unittest.TestCase):
    def test_one_kelvin_edge_injection_and_extreme_bias_have_independent_limits(self):
        # At mu=edge, the one-sided Fermi integral is kBT*ln(2), not
        # zero and not an arbitrary convention for a single threshold point.
        edge, temperature = .11, 1.0
        thermal = bte.KB_EV * temperature
        expected = -4 * bte.E_CHARGE**2 / bte.PLANCK * thermal * np.log(2)
        self.assertAlmostEqual(bte.ballistic_current_a([edge, edge], edge, -1000., temperature)
                               / expected, 1, delta=2e-14)
        for method in ("analytic", "gauss"):
            result = bte.integrate_current([edge, edge], 100, 0, 0, edge, -1000.,
                                           temperature_k=temperature, cells=2,
                                           integration=method)
            # The reference ends at edge+24kBT, omitting exp(-24)'s tail.
            self.assertAlmostEqual(result["current_a"] / expected, 1, delta=2e-10)
            self.assertEqual(result["intervals"][0]["open_modes"], [0, 1])
        edges = np.array([.05, .05, .11, .30])
        # Every edge is hundreds of kBT from these contacts. The zero-T
        # overlap length is therefore an independent double-precision oracle.
        overlap = float(np.maximum(.20 - edges, 0).sum())
        expected_bias = -2 * bte.E_CHARGE**2 / bte.PLANCK * overlap
        forward = bte.ballistic_current_a(edges, .20, -1000., 1)
        reverse = bte.ballistic_current_a(edges, -1000., .20, 1)
        self.assertAlmostEqual(forward / expected_bias, 1, delta=2e-14)
        self.assertEqual(reverse, -forward)
        for temperature in (1., 300., 2000.):
            thermal = bte.KB_EV * temperature
            delta = thermal * 1e-4
            slope = (bte.fermi(edge, edge + delta, temperature)
                     - bte.fermi(edge, edge - delta, temperature)) / (2 * delta)
            self.assertAlmostEqual(float(slope) * 4 * thermal, 1, delta=2e-9)
            for offset in (-1000., -20., -1., 0., 1., 20., 1000.):
                a, c = bte.fermi(edge + thermal * np.array([offset, -offset]), edge, temperature)
                self.assertAlmostEqual(float(a + c), 1, delta=2e-14)

    def test_degenerate_edges_enter_together_without_threshold_dos_sampling(self):
        modes = ((1, 1), (1, 2), (2, 1))
        edges = bte.subband_edges(10, 10, modes)
        self.assertEqual(edges[1], edges[2])
        labels_at, energies_at = bte.rectangular_modes_below(float(edges[1]), 10, 10)
        labels_above, energies_above = bte.rectangular_modes_below(
            float(np.nextafter(edges[1], np.inf)), 10, 10)
        self.assertEqual(labels_at, [(1, 1)])
        self.assertEqual(labels_above, list(modes))
        np.testing.assert_array_equal(energies_above, edges)
        self.assertEqual(len(energies_at), 1)
        with self.assertRaises(ValueError):
            bte.directional_dos_per_ev_m(float(edges[1]), edges)
        immediately_above = float(np.nextafter(edges[1], np.inf))
        speed = bte.speed_m_s(immediately_above, edges)
        dos = bte.directional_dos_per_ev_m(immediately_above, edges)
        self.assertTrue(np.all(np.isfinite(dos)))
        np.testing.assert_allclose(dos * speed, 2 * bte.E_CHARGE / bte.PLANCK, rtol=2e-15)
        result = bte.integrate_current(edges, 100, .02, .01, .15, .10,
                                       temperature_k=1, cells=2)
        self.assertEqual([v["open_modes"] for v in result["intervals"]], [[0], [0, 1, 2]])
        self.assertTrue(np.isfinite(result["current_a"]))

    def test_collision_units_and_spatial_rescaling_preserve_the_same_physics(self):
        f = np.array([.9, .1, .3, .7])
        speeds_nm_ps = np.array([2., 2., 1., 1.])
        kappa_nm_inv = bte.build_kappa(2, [.003, .08], .02)
        # nm/ps -> m/s is *1e3; nm^-1 -> m^-1 is *1e9.
        per_ps = bte.collision_term(f, speeds_nm_ps, kappa_nm_inv)
        per_s = bte.collision_term(f, speeds_nm_ps * 1e3, kappa_nm_inv * 1e9)
        np.testing.assert_allclose(per_s, per_ps * 1e12, rtol=2e-14, atol=1e-4)
        self.assertAlmostEqual(float(np.sum(per_ps / speeds_nm_ps)), 0, delta=1e-16)
        baseline = bte.solve_shell(100., kappa_nm_inv, [.9, .3], [.1, .2], cells=7)
        for scale in (1e-3, 1e3):
            changed_units = bte.solve_shell(100. * scale, kappa_nm_inv / scale,
                                            [.9, .3], [.1, .2], cells=7)
            np.testing.assert_allclose(changed_units.x_nm / scale, baseline.x_nm, atol=1e-13)
            np.testing.assert_allclose(changed_units.f, baseline.f, atol=2e-14)
            np.testing.assert_allclose(changed_units.face_flux, baseline.face_flux, atol=2e-14)

    def test_default_three_mode_current_separates_spatial_and_energy_bin_errors(self):
        beta, gamma, length, temperature = .02, .01, 100., 300.
        mu_left, mu_right = .150, .100
        edges = bte.subband_edges(10, 8, ((1, 1), (2, 1), (1, 2)))
        thermal = bte.KB_EV * temperature

        def tail_integral(edge, mu):
            # Integral_edge^infinity f_FD dE in eV, derived directly from
            # its antiderivative rather than calling the reference helper.
            if np.isposinf(edge):
                return 0.0
            return float(thermal * np.logaddexp(0., (mu - edge) / thermal))

        def independent_current(cells=None):
            integral_ev = 0.0
            boundaries = np.r_[edges, np.inf]
            for modes, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
                alpha = beta + (modes - 1) * gamma / 2
                transmission = 1 / (1 + alpha * length)
                if cells is not None:
                    dx = length / cells
                    transmission = (1 + alpha * dx) / (1 + alpha * length + alpha * dx)
                incoming = (tail_integral(lo, mu_left) - tail_integral(hi, mu_left)
                            - tail_integral(lo, mu_right) + tail_integral(hi, mu_right))
                integral_ev += modes * transmission * incoming
            return 2 * bte.E_CHARGE**2 / bte.PLANCK * integral_ev

        continuum = independent_current()
        fixed_grid = independent_current(cells=24)
        reference = -bte.integrate_current(edges, length, beta, gamma, mu_left, mu_right,
                                           temperature_k=temperature, cells=24)["current_a"]
        # Run the published teaching implementation, not a second copy of its
        # midpoint loop. Its code uses the same three prescribed subbands.
        article = ROOT / "_posts/2026-4Q/2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch.md"
        snippets = re.findall(r"^```python\n(.*?)^```", article.read_text(encoding="utf-8"), re.M | re.S)
        namespace = {"__name__": "msbte_default_current_audit"}
        with redirect_stdout(io.StringIO()):
            for snippet in snippets:
                exec(compile(snippet, str(article), "exec"), namespace)
            teaching = namespace["current_from_shells_A"](edges, mu_left, mu_right)
        np.testing.assert_allclose(namespace["edges"], edges, atol=1e-14)
        self.assertAlmostEqual(continuum * 1e6, 2.029909021, delta=1e-9)
        self.assertAlmostEqual(reference * 1e6, 2.187730239, delta=1e-9)
        self.assertAlmostEqual(teaching * 1e6, 2.187451852, delta=1e-9)
        # The reference uses a finite high-energy cutoff, unlike the infinity
        # in the independent expression. The omitted thermal tail is tiny.
        self.assertAlmostEqual(reference, fixed_grid, delta=1e-15)
        self.assertGreater(fixed_grid - continuum, .15e-6)
        self.assertLess(abs(teaching - fixed_grid), .001e-6)
        errors = []
        for cells in (12, 24, 48, 96):
            expected = independent_current(cells)
            actual = -bte.integrate_current(edges, length, beta, gamma, mu_left, mu_right,
                                            temperature_k=temperature, cells=cells)["current_a"]
            self.assertAlmostEqual(actual, expected, delta=1e-15)
            errors.append(expected - continuum)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 1.9)
            self.assertLess(coarse / fine, 2.)

    def test_block_solver_matches_independent_dense_kronecker_assembly(self):
        rng = np.random.default_rng(691023)
        for states, cells in ((2, 1), (4, 3), (6, 17), (8, 9)):
            raw = rng.uniform(0, .1, (states, states))
            kappa = raw + raw.T
            np.fill_diagonal(kappa, 0)
            left, right = rng.random((2, states // 2))
            length = 137.0
            solution = bte.solve_shell(length, kappa, left, right, cells)
            dx = length / cells
            plus = np.diag(np.tile([1., 0.], states // 2))
            minus = np.eye(states) - plus
            backward = np.eye(cells) - np.eye(cells, k=-1)
            laplacian = np.diag(kappa.sum(1)) - kappa
            matrix = ((np.kron(backward, plus) + np.kron(backward.T, minus)) / dx
                      + np.kron(np.eye(cells), laplacian))
            rhs = np.zeros((cells, states))
            rhs[0, 0::2], rhs[-1, 1::2] = left / dx, right / dx
            dense = np.linalg.solve(matrix, rhs.ravel()).reshape(cells, states)
            np.testing.assert_allclose(solution.f, dense, atol=3e-13)

    def test_space_and_direction_reversal_covariance_for_general_graphs(self):
        rng = np.random.default_rng(12076)
        raw = rng.random((6, 6)) * .03
        kappa = raw + raw.T
        np.fill_diagonal(kappa, 0)
        left, right = rng.random((2, 3))
        reverse_branches = np.array([1, 0, 3, 2, 5, 4])
        forward = bte.solve_shell(123, kappa, left, right, 31)
        reverse = bte.solve_shell(123, kappa[np.ix_(reverse_branches, reverse_branches)],
                                  right, left, 31)
        np.testing.assert_allclose(reverse.f, forward.f[::-1, reverse_branches], atol=2e-13)
        np.testing.assert_allclose(reverse.face_flux, -forward.face_flux[::-1], atol=2e-13)

    def test_equal_mode_inflow_reduces_to_effective_one_pair_and_shows_grid_error(self):
        for modes in (1, 2, 3, 5):
            for cells in (12, 36):
                beta, gamma, length = .04, .04, 100.0
                alpha = beta + (modes - 1) * gamma / 2
                dx = length / cells
                continuum = 1 / (1 + alpha * length)
                discrete = (1 + alpha * dx) / (1 + alpha * length + alpha * dx)
                solved = bte.solve_shell(length, bte.build_kappa(modes, beta, gamma), .9, .2, cells)
                np.testing.assert_allclose(solved.face_flux, modes * .7 * discrete, atol=2e-13)
                np.testing.assert_allclose(solved.f[:, 0::2], np.repeat(solved.f[:, :1], modes, axis=1), atol=2e-13)
                self.assertGreater(discrete, continuum)
        # The allowed two-mode UI corner is conservative but 40% above the
        # continuum transmission on N=12. This is discretization, not leakage.
        self.assertAlmostEqual(100 * (.2 / (1 / 7) - 1), 40, places=12)
        self.assertAlmostEqual(100 * ((7 / 43) / (1 / 7) - 1), 600 / 43, places=12)

    def test_stiff_one_pair_retains_the_finite_grid_diffusion_floor(self):
        # Refining a grid and increasing collision strength are different
        # limits: at fixed N, T_N tends to 1/(N+1), not the continuum zero.
        cells, length, beta = 24, 100.0, 100.0
        dx = length / cells
        solution = bte.solve_shell(length, bte.build_kappa(1, beta), 1., 0., cells)
        exact_discrete = (1 + beta * dx) / (1 + beta * length + beta * dx)
        np.testing.assert_allclose(solution.face_flux, exact_discrete, rtol=1e-9, atol=1e-12)
        self.assertGreater(exact_discrete, 1 / (cells + 1))
        self.assertLess(abs(exact_discrete - 1 / (cells + 1)), .0001)
        self.assertGreater(exact_discrete, 100 * (1 / (1 + beta * length)))

    def test_density_and_directional_flux_agree_in_energy_and_wavevector_variables(self):
        nodes, weights = np.polynomial.legendre.leggauss(192)
        for edge in (.05, .12):
            mu, temperature, upper = .15, 300., edge + .8
            # E=edge+u^2 removes the integrable DOS threshold singularity.
            u_max = np.sqrt(upper - edge)
            u = (nodes + 1) * u_max / 2
            energy = edge + u**2
            occupied_dos = (np.array([float(bte.directional_dos_per_ev_m(float(e), edge))
                                      for e in energy])
                            * bte.fermi(energy, mu, temperature))
            density_from_energy = float(weights @ (occupied_dos * 2 * u)) * u_max / 2
            k_max = np.sqrt(2 * bte.ELECTRON_MASS * .19 * bte.E_CHARGE * (upper - edge)) / bte.HBAR
            k = (nodes + 1) * k_max / 2
            kinetic_ev = bte.HBAR**2 * k**2 / (2 * bte.ELECTRON_MASS * .19 * bte.E_CHARGE)
            occupations = bte.fermi(edge + kinetic_ev, mu, temperature)
            density_from_k = bte.SPIN / (2 * np.pi) * float(weights @ occupations) * k_max / 2
            self.assertAlmostEqual(density_from_energy / density_from_k, 1, delta=3e-12)
            velocity = bte.HBAR * k / (bte.ELECTRON_MASS * .19)
            flux_from_k = bte.SPIN / (2 * np.pi) * float(weights @ (velocity * occupations)) * k_max / 2
            integral_ev = float(bte.fermi_integral_above_ev(edge, mu, temperature)
                                - bte.fermi_integral_above_ev(upper, mu, temperature))
            flux_from_energy = bte.SPIN * bte.E_CHARGE / bte.PLANCK * integral_ev
            self.assertAlmostEqual(flux_from_k / flux_from_energy, 1, delta=3e-12)

    def test_mode_permutation_energy_zero_and_spin_conventions(self):
        edges = np.array([.14, .05, .11])
        beta = np.array([.007, .03, .011])
        args = dict(length_nm=100, gamma_nm_inv=.013, mu_left_ev=.17,
                    mu_right_ev=.095, temperature_k=50, cells=36)
        forward = bte.integrate_current(edges, beta_nm_inv=beta, **args)["current_a"]
        order = np.array([1, 2, 0])
        permuted = bte.integrate_current(edges[order], beta_nm_inv=beta[order], **args)["current_a"]
        self.assertAlmostEqual(forward / permuted, 1, delta=3e-13)
        shifted_args = {**args, "mu_left_ev": args["mu_left_ev"] - .7,
                        "mu_right_ev": args["mu_right_ev"] - .7}
        shifted = bte.integrate_current(edges - .7, beta_nm_inv=beta, **shifted_args)["current_a"]
        self.assertAlmostEqual(shifted / forward, 1, delta=3e-13)
        one_spin = bte.ballistic_current_a([.05], .15, .12, 1, spin=1)
        two_spins = bte.ballistic_current_a([.05], .15, .12, 1, spin=2)
        two_degenerate_modes = bte.ballistic_current_a([.05, .05], .15, .12, 1, spin=2)
        self.assertAlmostEqual(two_spins / one_spin, 2, places=14)
        self.assertAlmostEqual(two_degenerate_modes / two_spins, 2, places=14)
        delta = 1e-6  # eV energy difference, numerically an equivalent volt.
        small_bias = -bte.ballistic_current_a([.05], .15 + delta / 2, .15 - delta / 2, 1)
        self.assertAlmostEqual(small_bias / delta / (2 * bte.E_CHARGE**2 / bte.PLANCK), 1, delta=1e-10)

    @unittest.skipUnless(shutil.which("node"), "Cross-language audit requires Node.js")
    def test_python_and_javascript_match_over_geometry_contacts_and_stress_cases(self):
        rng = np.random.default_rng(270118)
        geometry = [[6, 8, .19, 8], [8, 8, .19, 8], [10, 8, .19, 8],
                    [20, 8, .19, 8], [100, 2, .19, 16], [2, 100, .19, 16]]
        edges_mev = bte.subband_edges(10, 8, [(ny, nz) for ny in range(1, 9) for nz in range(1, 9)])
        edges_mev = (1000 * np.sort(edges_mev)[:8]).tolist()
        contacts = [[edges_mev, left, right, temp] for temp in (1, 50, 300, 500, 2000)
                    for left, right in ((115, 95), (50, 180), (100, 100))]
        fd = [[energy, 100, temp] for temp in (1, 50, 300, 2000)
              for energy in (-1e6, 50, 100, 130, 1e6)]
        shells = []
        for modes in (1, 2):
            for i, length in enumerate((1, 100, 1000)):
                for beta, gamma in ((0, 0), (.005, .003), (.1, .1)):
                    shells.append(dict(modes=modes, length=length, beta=beta, gamma=gamma,
                                       fL=rng.random(modes).tolist(), fR=rng.random(modes).tolist(),
                                       cells=(2, 12, 36)[i]))
        script = """const fs=require('fs'),m=require(process.argv[1]);
const q=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify({
 constants:[m.A0,m.KB,m.G0],geometry:q.geometry.map(x=>m.subbands(...x)),
 currents:q.contacts.map(x=>m.ballisticCurrent(...x)),
 fd:q.fd.map(x=>m.fermi(...x)),shells:q.shells.map(x=>m.solveShell(x))}));"""
        process = subprocess.run([shutil.which("node"), "-e", script,
                                  str(ROOT / "assets/js/article-labs-msbte.js")],
                                 input=json.dumps(dict(geometry=geometry, contacts=contacts, fd=fd, shells=shells)),
                                 text=True, capture_output=True, check=True, timeout=30)
        result = json.loads(process.stdout)
        np.testing.assert_allclose(result["constants"],
                                   [bte.HBAR**2 / (2 * bte.ELECTRON_MASS * bte.E_CHARGE) * 1e21,
                                    1000 * bte.KB_EV, 2 * bte.E_CHARGE**2 / bte.PLANCK * 1e6],
                                   rtol=1e-12)
        for params, calculated in zip(geometry, result["geometry"]):
            width, height, mass, count = params
            independent = sorted((float(bte.subband_edges(width, height, [(ny, nz)], mass, mass)[0]), ny, nz)
                                 for ny in range(1, 33) for nz in range(1, 33))[:count]
            np.testing.assert_allclose([v["edge"] / 1000 for v in calculated],
                                       [v[0] for v in independent], rtol=2e-14)
            for value in calculated:
                exact = bte.subband_edges(width, height, [(value["ny"], value["nz"])], mass, mass)[0]
                self.assertAlmostEqual(value["edge"] / 1000, float(exact), delta=1e-13)
        for query, value in zip(contacts, result["currents"]):
            energy, left, right, temperature = query
            exact = bte.ballistic_current_a(np.array(energy) / 1000, left / 1000, right / 1000, temperature)
            self.assertAlmostEqual(value * 1e-6, exact, delta=2e-17)
        np.testing.assert_allclose(result["fd"], [float(bte.fermi(e / 1000, mu / 1000, t)) for e, mu, t in fd], rtol=1e-12, atol=1e-15)
        for query, calculated in zip(shells, result["shells"]):
            expected = bte.solve_shell(query["length"], bte.build_kappa(query["modes"], query["beta"], query["gamma"]),
                                       query["fL"], query["fR"], query["cells"])
            np.testing.assert_allclose(calculated["occupation"], expected.f, atol=3e-12)
            np.testing.assert_allclose(calculated["modeFlux"], expected.face_flux_by_mode, atol=3e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
