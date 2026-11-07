"""Drude and weak-scattering reference calculations (NumPy; SI units).

e > 0; an electron has charge -e.  tau is momentum relaxation time,
not energy lifetime.  The scalar Drude model assumes an isotropic,
constant parabolic mass.  Born rates require weak, dilute, independent
scatterers; no strong-scattering or localization correction is included.
"""
from math import gamma, pi
import numpy as np

E_CHARGE = 1.602176634e-19
HBAR = 6.62607015e-34 / (2 * pi)
M0 = 9.1093837139e-31
KB = 1.380649e-23
EPS0 = 8.8541878188e-12


def _finite(value, name, positive=False, nonnegative=False):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or (positive and np.any(value <= 0)) or (nonnegative and np.any(value < 0)):
        raise ValueError(f"Invalid {name}")
    return value


def drude_step(t, density, mass, tau, field):
    """E is switched on at t=0; initially u=J=0. Returns SI quantities."""
    t = _finite(t, "time", nonnegative=True)
    n, mass, tau = (_finite(x, name, positive=True) for x, name in [(density, "density"), (mass, "mass"), (tau, "tau")])
    field = _finite(field, "field")
    mobility = E_CHARGE * tau / mass
    conductivity = n * E_CHARGE * mobility
    build = -np.expm1(-t / tau)
    drift = -mobility * field * build
    return dict(mobility=mobility, conductivity=conductivity,
                drift=drift, current_density=-E_CHARGE * n * drift)


def drude_ac(omega, density, mass, tau):
    """Complex conductivity with real fields Re[E0 exp(-i omega t)]."""
    omega = _finite(omega, "angular frequency", nonnegative=True)
    sigma = drude_step(0, density, mass, tau, 0)["conductivity"]
    return sigma / (1 - 1j * omega * tau)


def parabolic_band(k, mass):
    k, mass = _finite(k, "wave vector"), _finite(mass, "mass", positive=True)
    return dict(energy=HBAR**2 * k**2 / (2 * mass), velocity=HBAR * k / mass, inverse_mass=np.ones_like(k) / mass)


def tight_binding_band(k, hopping, lattice):
    """1D epsilon=2t[1-cos(ka)]; hopping is energy in joules."""
    k = _finite(k, "wave vector")
    hopping, lattice = _finite(hopping, "hopping", positive=True), _finite(lattice, "lattice", positive=True)
    ka = k * lattice
    return dict(energy=4 * hopping * np.sin(ka / 2)**2,
                velocity=2 * hopping * lattice / HBAR * np.sin(ka),
                inverse_mass=2 * hopping * lattice**2 / HBAR**2 * np.cos(ka))


def inverse_mass_tensor(energy_hessian):
    """(m^-1)_ij = (1/hbar^2) d²epsilon/(dk_i dk_j), including sign."""
    hessian = _finite(energy_hessian, "energy Hessian")
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or not hessian.size or not np.allclose(hessian, hessian.T, rtol=1e-12, atol=0):
        raise ValueError("A finite symmetric square Hessian is required")
    return hessian / HBAR**2


def fermi_dirac(energy, chemical_potential, temperature):
    energy, mu = _finite(energy, "energy"), _finite(chemical_potential, "chemical potential")
    temperature = _finite(temperature, "temperature", positive=True)
    z = (energy - mu) / (KB * temperature)
    small = np.exp(-np.abs(z))
    return np.where(z >= 0, small / (1 + small), 1 / (1 + small))


def first_order_transition(delta_energy, coupling, duration):
    """Finite-time c_f^(1) for two unperturbed states; coupling=<f|V|i>.

    delta_energy=Ef-Ei. This is perturbation theory, NOT an exact Rabi
    probability: |coupling|*t/hbar and the transition probability must be
    small for its use. No artificial clipping to one is applied.
    """
    delta_energy = _finite(delta_energy, "energy difference")
    duration = _finite(duration, "duration", nonnegative=True)
    coupling = np.asarray(coupling, dtype=complex)
    if not np.all(np.isfinite(coupling)):
        raise ValueError("Invalid coupling")
    phase = delta_energy * duration / (2 * HBAR)
    amplitude = -1j * coupling * duration / HBAR * np.exp(1j * phase) * np.sinc(phase / pi)
    return amplitude, np.abs(amplitude)**2


def dos_3d_per_spin(energy, mass):
    """Single-valley DOS per volume per joule for ONE spin, m*k/(2pi²hbar²)."""
    energy, mass = _finite(energy, "energy", nonnegative=True), _finite(mass, "mass", positive=True)
    k = np.sqrt(2 * mass * energy) / HBAR
    return mass * k / (2 * pi**2 * HBAR**2)


def golden_rule_continuum_rate(energy, mass, impurity_density, fourier_potential):
    """Short-range isotropic elastic rate: (2pi/hbar) ni |U~|² DOS_spin.

    Plane waves normalized in volume V give <k'|U|k>=U~(q)/V,
    U~(q)=integral d³r exp(-iq.r) U(r), in J m³. An incoherent ni V
    scatterers cancels the final-state volume. Spin is conserved: do not
    add another factor 2 to the accessible one-spin final DOS.
    """
    density = _finite(impurity_density, "impurity density", nonnegative=True)
    potential = np.asarray(fourier_potential, dtype=complex)
    if not np.all(np.isfinite(potential)):
        raise ValueError("Invalid Fourier potential")
    return 2 * pi / HBAR * density * np.abs(potential)**2 * dos_3d_per_spin(energy, mass)


def short_range_born_rate(energy, mass, impurity_density, born_length):
    """Isotropic amplitude f=-a_B, cross section 4pi*a_B², rate ni*v*sigma."""
    energy, mass = _finite(energy, "energy", nonnegative=True), _finite(mass, "mass", positive=True)
    density, length = _finite(impurity_density, "impurity density", nonnegative=True), _finite(born_length, "Born length")
    velocity = np.sqrt(2 * energy / mass)
    rate = density * velocity * 4 * pi * length**2
    return dict(rate_q=rate, rate_tr=rate, amplitude=-length, cross_section=4 * pi * length**2)


def screening_integral(x):
    """Stable [ln(1+x)-x/(1+x)]/x², with value 1/2 at x=0."""
    x = _finite(x, "screening shape", nonnegative=True)
    safe = np.where(x < 1e-3, 1, x)
    direct = (np.log1p(safe) - safe / (1 + safe)) / safe**2
    small = np.where(x < 1e-3, x, 0.)
    series = sum((-1)**j * (j + 1) / (j + 2) * small**j for j in range(9))
    return np.where(x < 1e-3, series, direct)


def screening_ratio(x):
    """tau_tr/tau_q for a 3D Born kernel (1+x sin²(theta/2))^-2."""
    x = _finite(x, "screening shape", nonnegative=True)
    return 1 / (2 * (1 + x) * screening_integral(x))


def yukawa_fourier(q, charge_number, permittivity, screening_wavevector):
    q, charge = _finite(q, "momentum-transfer wavevector", nonnegative=True), _finite(charge_number, "charge number")
    epsilon = _finite(permittivity, "permittivity", positive=True)
    qs = _finite(screening_wavevector, "screening wavevector", positive=True)
    return charge * E_CHARGE**2 / (epsilon * (q**2 + qs**2))


def screened_born_rates(energy, mass, impurity_density, charge_number, permittivity, screening_wavevector):
    """Analytic Yukawa rates; q=|k'-k|, x=4k²/qs², static screening."""
    energy, mass = _finite(energy, "energy", nonnegative=True), _finite(mass, "mass", positive=True)
    density = _finite(impurity_density, "impurity density", nonnegative=True)
    qs = _finite(screening_wavevector, "screening wavevector", positive=True)
    velocity, k = np.sqrt(2 * energy / mass), np.sqrt(2 * mass * energy) / HBAR
    x = 4 * k**2 / qs**2
    amplitude0 = -mass * yukawa_fourier(0, charge_number, permittivity, qs) / (2 * pi * HBAR**2)
    prefactor = 4 * pi * density * velocity * amplitude0**2
    return dict(rate_q=prefactor / (1 + x), rate_tr=prefactor * 2 * screening_integral(x), shape=x)


def angular_born_rates(energy, mass, impurity_density, fourier_potential, order=512):
    """Independent angular quadrature; callable U~(q), includes full dOmega.

    Integrate y=sin²(theta/2) on [0,1], so dOmega=4pi dy and
    transport weight 1-cos(theta)=2y. Increase order for a sharp kernel.
    """
    energy = float(_finite(energy, "energy", nonnegative=True))
    mass = float(_finite(mass, "mass", positive=True))
    density = float(_finite(impurity_density, "impurity density", nonnegative=True))
    if not isinstance(order, int) or not 8 <= order <= 4096:
        raise ValueError("Quadrature order must be 8..4096")
    points, weights = np.polynomial.legendre.leggauss(order)
    y, weights = (points + 1) / 2, weights / 2
    k, velocity = np.sqrt(2 * mass * energy) / HBAR, np.sqrt(2 * energy / mass)
    potential = np.asarray(fourier_potential(2 * k * np.sqrt(y)), dtype=complex)
    if not np.all(np.isfinite(potential)):
        raise ValueError("Invalid Fourier potential")
    differential = np.abs(-mass * potential / (2 * pi * HBAR**2))**2
    return dict(rate_q=density * velocity * 4 * pi * np.sum(weights * differential),
                rate_tr=density * velocity * 4 * pi * np.sum(weights * 2 * y * differential))


def acoustic_quasielastic_rate(energy, mass, deformation_potential, temperature, mass_density, sound_speed):
    """3D equipartition, elastic acoustic deformation-potential rate.

    Isotropic parabolic band, longitudinal acoustic branch, high phonon
    occupation at the relevant q; combines absorption and emission.
    D² kBT/(rho_mass vs²) has units J² m³, like ni|U~|². No spin 2.
    """
    deformation = _finite(deformation_potential, "deformation potential")
    temperature = _finite(temperature, "temperature", positive=True)
    density, speed = _finite(mass_density, "mass density", positive=True), _finite(sound_speed, "sound speed", positive=True)
    return 2 * pi / HBAR * deformation**2 * KB * temperature / (density * speed**2) * dos_3d_per_spin(energy, mass)


def thermal_mb_tau(tau_ref, energy_ref, exponent, temperature):
    """Conductivity-weighted 3D Maxwell–Boltzmann tau for tau(E)∝E^r.

    Weight E^(3/2) exp(-E/kBT); finite for r > -5/2. Not a Fermi-
    degenerate average, nor a number-weighted mean free time.
    """
    tau, energy, temperature = (_finite(v, name, positive=True) for v, name in [(tau_ref, "reference time"), (energy_ref, "reference energy"), (temperature, "temperature")])
    exponent = float(_finite(exponent, "exponent"))
    if exponent <= -2.5:
        raise ValueError("The conductivity-weighted integral requires r > -5/2")
    return tau * (KB * temperature / energy)**exponent * gamma(exponent + 2.5) / gamma(2.5)


def charge_mode(s, omega_tau):
    """a''+a'+Omega²a=0, a(0)=1,a'(0)=0; s=t/tau, Omega=omega_p*tau."""
    s = _finite(s, "dimensionless time", nonnegative=True)
    omega = float(_finite(omega_tau, "plasma parameter", nonnegative=True))
    if omega > .5:
        frequency = np.sqrt(omega**2 - .25)
        sinc = np.sinc(frequency * s / pi)
        decay = np.exp(-s / 2)
        a = decay * (np.cos(frequency * s) + .5 * s * sinc)
        derivative = -omega**2 * s * decay * sinc
        regime = "underdamped"
    elif omega == .5:
        a, derivative = np.exp(-s / 2) * (1 + s / 2), -s / 4 * np.exp(-s / 2)
        frequency, regime = 0., "critical"
    else:
        b = np.sqrt(.25 - omega**2)
        slow = -omega**2 / (.5 + b)
        divided = -np.expm1(-2 * b * s) / (2 * b)
        a = np.exp(slow * s) * (1 - slow * divided)
        derivative = -omega**2 * np.exp(slow * s) * divided
        frequency, regime = 0., "overdamped"
    return dict(a=a, derivative=derivative, second_derivative=-derivative-omega**2*a,
                damped_frequency_tau=frequency, regime=regime)


def charge_profile(x_over_length, s, omega_tau, initial_fraction=.05):
    """Periodic qL=2pi; fixed positive background; no diffusion or contacts.

    delta_n/n0=initial_fraction*a*cos(qx), E/Escale=-a*sin(qx),
    J/Jscale=a'*sin(qx), Jscale=e*A0/(q*tau). Charge continuity is
    -e*d_t(delta_n)+d_x J=0, not +e*d_t(delta_n)+d_x J=0.
    """
    x = _finite(x_over_length, "periodic position")
    fraction = float(_finite(initial_fraction, "initial fraction", positive=True))
    if fraction > .1:
        raise ValueError("Use a small perturbation (<= 0.1)")
    mode = charge_mode(s, omega_tau)
    c, sn = np.cos(2 * pi * x), np.sin(2 * pi * x)
    derivative_n = mode["derivative"] * c
    derivative_j = mode["derivative"] * c
    return dict(delta_density_fraction=fraction*mode["a"]*c, field_scaled=-mode["a"]*sn,
                current_scaled=mode["derivative"]*sn,
                continuity_residual=-derivative_n+derivative_j)


if __name__ == "__main__":
    result = drude_step(200e-15, 1e23, .2*M0, 100e-15, 1000)
    print({key: float(value) for key, value in result.items()})
