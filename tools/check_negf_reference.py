#!/usr/bin/env python3
"""Independently compare the browser NEGF solver with dense NumPy solves.

Run with a Python environment containing NumPy. No SciPy, browser, network,
post modification or damping parameter is required. The reference constructs
the complete Hamiltonian and solves a dense linear system; the JavaScript
implementation uses its own O(N) recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
J_PER_EV = 1.602176634e-19
T0 = HBAR**2 / (2 * 0.067 * M0 * 1e-18) / J_PER_EV
N = 100
NOTEBOOK_COMMIT = "7fd9f32c390e117ab6b04bb7f787fc5b43c959e2"
NOTEBOOK_SHA256 = "2918fa2f8a8e0b20762fffa1b4b87d96a4cab67800e06fc0b94a79ce8b607cfa"


def potential(height: float, well_sites: int) -> np.ndarray:
    """Exactly `well_sites` zero-potential sites between two three-site barriers."""
    left = (N - well_sites) // 2
    right = left + well_sites
    result = np.zeros(N)
    result[left - 3 : left] = height
    result[right : right + 3] = height
    return result


def dense_transmission(energy: float, profile: np.ndarray) -> float:
    """Caroli transmission from a dense solve and Bloch-form self-energy."""
    if not 0 < energy < 4 * T0:
        return 0.0
    ka = np.arccos(1 - energy / (2 * T0))
    sigma = -T0 * np.exp(1j * ka)
    gamma = 2 * T0 * np.sin(ka)
    hamiltonian = np.diag(2 * T0 + profile)
    hamiltonian += np.diag(np.full(N - 1, -T0), 1)
    hamiltonian += np.diag(np.full(N - 1, -T0), -1)
    matrix = energy * np.eye(N, dtype=complex) - hamiltonian
    matrix[0, 0] -= sigma
    matrix[-1, -1] -= sigma
    rhs = np.zeros(N, dtype=complex)
    rhs[-1] = 1
    propagator = np.linalg.solve(matrix, rhs)[0]
    return float(gamma**2 * abs(propagator) ** 2)


def refine_peak(profile: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Golden-section maximization over one known local-resonance bracket."""
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    y1 = dense_transmission(x1, profile)
    y2 = dense_transmission(x2, profile)
    for _ in range(95):
        if y1 < y2:
            lo, x1, y1 = x1, x2, y2
            x2 = lo + ratio * (hi - lo)
            y2 = dense_transmission(x2, profile)
        else:
            hi, x2, y2 = x2, x1, y1
            x1 = hi - ratio * (hi - lo)
            y1 = dense_transmission(x1, profile)
        if hi - lo < 2e-14:
            break
    energy = (lo + hi) / 2
    return energy, dense_transmission(energy, profile)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "assets/js/negf-core.js")
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    default_profile = potential(0.4, 16)
    article_rounded = [0.0639, 0.1413, 0.2453, 0.3713, 0.5146]
    peaks = [refine_peak(default_profile, 0.01, 0.025)]
    peaks.extend(refine_peak(default_profile, e - 0.002, e + 0.002) for e in article_rounded)
    rng = np.random.default_rng(20261024)
    cases = []
    for height, well_sites in [(0, 6), (0.15, 12), (0.15, 17), (0.4, 16), (0.65, 30)]:
        energies = [0.001, 0.016, 0.03, 0.0639, 0.1413, 0.2453, 0.3713, 0.5146, 0.6]
        energies.extend(rng.uniform(0.001, 0.6, 8).tolist())
        if height == 0.4:
            energies.extend(e for e, _ in peaks)
        profile = potential(height, well_sites)
        cases.append({
            "height": height, "wellSites": well_sites, "energies": energies,
            "potential": profile.tolist(),
            "expected": [dense_transmission(e, profile) for e in energies],
        })
    # Entire propagating lattice band, including both near-edge regimes.
    clean_energies = [1e-7, 0.001, 0.1, 0.6, T0, 2 * T0, 3 * T0, 4 * T0 - 1e-7]
    cases.append({
        "height": 0, "wellSites": 16, "energies": clean_energies,
        "potential": potential(0, 16).tolist(), "expected": [1.0] * len(clean_energies),
    })

    spectrum_cases = [(0.4, 16), (0.65, 30), (0.65, 6), (0.15, 17), (0.01, 29), (0, 16)]
    js = r"""
const fs = require('fs');
const core = require(process.argv[1]);
const {cases, spectrumCases} = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = cases.map(item => {
  const geometry = core.profile(item.height, item.wellSites);
  return {
    potential: Array.from(geometry.potential),
    values: item.energies.map(e => core.transmission(e, geometry)),
    outside: [-1, 0, 4 * core.constants.hoppingEV, 5].map(e => core.transmission(e, geometry))
  };
});
const spectra = spectrumCases.map(([height, wellSites]) => {
  const spectrum = core.spectrum(core.profile(height, wellSites));
  return {height, wellSites, peaks: spectrum.peaks,
    minTransmission: Math.min(...spectrum.transmissions),
    maxTransmission: Math.max(...spectrum.transmissions),
    sampleCount: spectrum.energies.length};
});
console.log(JSON.stringify({constants: core.constants, results, spectra}));
"""
    completed = subprocess.run(
        [args.node, "-e", js, str(args.engine.resolve())],
        input=json.dumps({"cases": cases, "spectrumCases": spectrum_cases}),
        text=True, capture_output=True, check=True,
    )
    observed = json.loads(completed.stdout)
    errors = []
    max_error = 0.0
    sample_count = 0
    for case, actual in zip(cases, observed["results"], strict=True):
        if actual["potential"] != case["potential"]:
            errors.append(f"Geometry differs at height={case['height']}, width={case['wellSites']}")
        for energy, expected, value in zip(case["energies"], case["expected"], actual["values"], strict=True):
            sample_count += 1
            error = abs(expected - value) if value is not None else math.inf
            max_error = max(max_error, error)
            if error > 2e-8 or value is None or not -1e-10 <= value <= 1 + 2e-8:
                errors.append(f"T({energy}, h={case['height']}, w={case['wellSites']}): JS={value}, dense={expected}")
        if actual["outside"] != [0, 0, 0, 0]:
            errors.append(f"Nonzero/out-of-domain outside-band result: {actual['outside']}")
    for target, (energy, transmission) in zip(article_rounded, peaks[1:], strict=True):
        if abs(target - energy) > 5.1e-5 or abs(1 - transmission) > 2e-8:
            errors.append(f"Article peak mismatch near {target}: E={energy}, T={transmission}")
    if abs(1 - peaks[0][1]) > 2e-8:
        errors.append(f"Unresolved-lowest peak reference did not reach unit transmission: {peaks[0]}")
    if abs(observed["constants"]["hoppingEV"] - T0) > 1e-14:
        errors.append("Kinetic coefficient does not match notebook constants")
    spectrum_peak_count = 0
    max_peak_error = 0.0
    for spectrum in observed["spectra"]:
        profile = potential(spectrum["height"], spectrum["wellSites"])
        for peak in spectrum["peaks"]:
            value = dense_transmission(peak["energyEV"], profile)
            error = abs(value - peak["transmission"])
            max_peak_error = max(max_peak_error, error)
            spectrum_peak_count += 1
            if abs(value - 1) > 2e-8 or error > 2e-8:
                errors.append(f"Seed is not a unit-transmission peak in dense solve: {spectrum}, {peak}")
        if not -1e-10 <= spectrum["minTransmission"] <= spectrum["maxTransmission"] <= 1 + 2e-8:
            errors.append(f"Spectrum violates unitarity: {spectrum}")
    if len(observed["spectra"][0]["peaks"]) != 6:
        errors.append("Default spectrum does not resolve all six independently confirmed peaks")
    else:
        for actual, (energy, _) in zip(observed["spectra"][0]["peaks"], peaks, strict=True):
            if abs(actual["energyEV"] - energy) > 1e-8:
                errors.append(f"Default resonance energy differs from dense refinement: {actual}")

    report = {
        "pass": not errors,
        "method": "Dense 100x100 NumPy solve, Bloch self-energy, zero interior eta; independent of JS recurrence",
        "numpy": np.__version__,
        "notebook": {
            "url": f"https://github.com/pilkwangkim/Physics/blob/{NOTEBOOK_COMMIT}/negf_from_scratch.ipynb",
            "sha256": NOTEBOOK_SHA256,
        },
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "hopping_ev": T0,
        "sample_count": sample_count,
        "max_abs_transmission_error": max_error,
        "spectrum_peak_checks": spectrum_peak_count,
        "max_abs_spectrum_peak_error": max_peak_error,
        "geometry": {"N": N, "a_nm": 1, "barrier_sites": 3, "well_sites": 16,
                     "barriers": [[39, 42], [58, 61]], "height_ev": 0.4},
        "dense_default_peaks": [{"energy_ev": e, "transmission": t} for e, t in peaks],
        "article_resolved_peaks_ev": article_rounded,
        "clean_wire": {"energies_ev": clean_energies, "js_transmission": observed["results"][-1]["values"]},
        "spectra": observed["spectra"],
        "errors": errors,
    }
    output = json.dumps(report, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
