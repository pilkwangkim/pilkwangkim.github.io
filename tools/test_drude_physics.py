"""Retained independent checks for assets/code/drude-model.py; NumPy only."""
import importlib.util
import contextlib
import io
from pathlib import Path
import math
import re
import numpy as np

path = Path(__file__).resolve().parents[1] / "assets/code/drude-model.py"
spec = importlib.util.spec_from_file_location("drude_model", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = 0


def close(actual, expected, rtol=1e-10, atol=0):
    global checks
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    checks += np.asarray(actual).size


def check(condition):
    global checks
    assert bool(condition)
    checks += 1


def rejects(fn):
    global checks
    try:
        fn()
    except (ValueError, TypeError):
        checks += 1
    else:
        raise AssertionError("Invalid input was accepted")


e, mass, tau, n, field = m.E_CHARGE, .2*m.M0, 1e-13, 1e23, 1000.
close(m.HBAR*2*math.pi, 6.62607015e-34)
p = m.drude_step(100*tau, n, mass, tau, field)
close(p["mobility"]*1e4, 879.410004189, rtol=1e-11)
close(p["conductivity"], 1408.97016042, rtol=1e-11)
close(p["drift"], -87.941000419, rtol=1e-11)
close(p["current_density"]*1e-14*1e9, 14.0897016042, rtol=1e-11)
close(field*1e-6 / (p["current_density"]*1e-14), 70973.8238675, rtol=1e-11)
for s in np.linspace(0, 8, 81):
    for sign in [-1, 0, 1]:
        v = m.drude_step(s*tau, n, mass, tau, sign*field)
        close(v["current_density"], -e*n*v["drift"])
        check(v["current_density"]*sign >= 0)
        close(v["drift"], -e*tau/mass*sign*field*(1-math.exp(-s)), atol=1e-13)
        if s > 0:
            dt = tau*1e-4
            uplus = m.drude_step(s*tau+dt, n, mass, tau, sign*field)["drift"]
            uminus = m.drude_step(s*tau-dt, n, mass, tau, sign*field)["drift"]
            close((uplus-uminus)/(2*dt), -e*sign*field/mass-v["drift"]/tau, rtol=2e-8, atol=2)
for w in np.linspace(0, 10, 101):
    ac = m.drude_ac(w/tau, n, mass, tau)
    close(ac*(1-1j*w), p["conductivity"])
    close(abs(ac)/p["conductivity"], 1/math.hypot(1, w))
    check(ac.real >= 0 and ac.imag >= 0)
    phase = math.atan(w)
    close((ac*np.exp(-1j*phase)).imag, 0, atol=1e-12)
    # Independent real-valued steady waveform satisfies tau*dJ/dt+J=sigma E.
    theta, delta = .71, 1e-5
    waveform = lambda x: (ac*np.exp(-1j*x)).real
    close(w*(waveform(theta+delta)-waveform(theta-delta))/(2*delta)+waveform(theta), p["conductivity"]*math.cos(theta), rtol=1e-9)

# k-derivatives of the actual energy, not duplicated analytic derivatives.
for hopping_ev in [.1, 1, 3]:
    for lattice_nm in [.1, .5, 1]:
        lattice, hopping = lattice_nm*1e-9, hopping_ev*e
        for ka in np.linspace(-math.pi, math.pi, 31):
            k, dk = ka/lattice, 1e-4/lattice
            b = m.tight_binding_band(k, hopping, lattice)
            plus = m.tight_binding_band(k+dk, hopping, lattice)["energy"]
            minus = m.tight_binding_band(k-dk, hopping, lattice)["energy"]
            close((plus-minus)/(2*dk)/m.HBAR, b["velocity"], rtol=3e-8, atol=1e-5)
            close((plus-2*b["energy"]+minus)/dk**2/m.HBAR**2*m.M0, b["inverse_mass"]*m.M0, rtol=2e-6, atol=2e-5)
            check(b["energy"] >= 0)
        close(m.tight_binding_band(0, hopping, lattice)["inverse_mass"], 2*hopping*lattice**2/m.HBAR**2)
for k in [0, 1e8, -1e9]:
    b = m.parabolic_band(k, mass)
    close(b["velocity"], m.HBAR*k/mass)
    close(b["inverse_mass"], 1/mass)
tensor = np.array([[1/mass, .1/mass], [.1/mass, -2/mass]])
close(m.inverse_mass_tensor(tensor*m.HBAR**2), tensor)

for z in [-1e9, -1000, -10, 0, 10, 1000, 1e9]:
    f = m.fermi_dirac(z*m.KB*300, 0, 300)
    check(0 <= f <= 1)
    close(f+m.fermi_dirac(-z*m.KB*300, 0, 300), 1)
close(m.fermi_dirac(0, 0, 300), .5)

# Time integral checks sinc amplitude, including exact resonance and nodes.
coupling, duration = 1e-5*e, 1e-13
for phase in [0, 1e-10, .1, 1, math.pi, 5]:
    delta = phase*2*m.HBAR/duration
    amplitude, probability = m.first_order_transition(delta, coupling, duration)
    times = np.linspace(0, duration, 20001)
    integrand = np.exp(1j*delta*times/m.HBAR)
    integral = np.sum((integrand[1:]+integrand[:-1])/2*np.diff(times))
    close(amplitude, -1j*coupling/m.HBAR*integral, rtol=1e-7, atol=1e-12)
    close(probability, abs(amplitude)**2)
    check(probability < 1e-4)
close(m.first_order_transition(0, coupling, duration)[1], (coupling*duration/m.HBAR)**2)
close(m.first_order_transition(e, coupling, 0)[1], 0)

# DOS normalization: integrate one-spin occupied states to kF³/(6pi²).
ef, ni, born_length = .1*e, 1e22, .03e-9
kf = np.sqrt(2*mass*ef)/m.HBAR
points, weights = np.polynomial.legendre.leggauss(256)
energy = ef*((points+1)/2)**2
jacobian = ef*(points+1)/2
close(np.sum(weights*jacobian*m.dos_3d_per_spin(energy, mass)), kf**3/(6*math.pi**2))
for energy in [.001*e, .1*e, 1*e]:
    short = m.short_range_born_rate(energy, mass, ni, born_length)
    u0 = 2*math.pi*m.HBAR**2*born_length/mass
    close(short["rate_q"], m.golden_rule_continuum_rate(energy, mass, ni, u0))
    quad = m.angular_born_rates(energy, mass, ni, lambda q: np.full_like(q, u0), order=64)
    close(quad["rate_q"], short["rate_q"])
    close(quad["rate_tr"], short["rate_tr"])
    close(short["amplitude"], -born_length)
    for x in [.00001, .01, 1, 10, 400]:
        k = np.sqrt(2*mass*energy)/m.HBAR
        qs = 2*k/np.sqrt(x)
        analytic = m.screened_born_rates(energy, mass, ni, 1, 12*m.EPS0, qs)
        numerical = m.angular_born_rates(energy, mass, ni, lambda q: m.yukawa_fourier(q, 1, 12*m.EPS0, qs), order=256)
        close(analytic["rate_q"], numerical["rate_q"], rtol=2e-10)
        close(analytic["rate_tr"], numerical["rate_tr"], rtol=2e-10)
        close(analytic["rate_q"]/analytic["rate_tr"], m.screening_ratio(x))
for x in [0, 1e-12, 1e-8, .0001, .01, 1, 400]:
    y = (points+1)/2
    normalization = np.sum(weights/2/(1+x*y)**2)
    weighted = np.sum(weights*y/(1+x*y)**2)
    close(m.screening_ratio(x), normalization/weighted, rtol=3e-10)
    check(m.screening_ratio(x) >= 1-1e-12)
close(m.screening_ratio(0), 1)
close(m.screening_ratio(1e-8), 1+1e-8/3, rtol=1e-15)
close(1/2, .5)  # pure-reversal benchmark: event transport weight=2

acoustic = m.acoustic_quasielastic_rate(ef, mass, 5*e, 300, 2330, 9000)
check(acoustic > 0)
close(m.acoustic_quasielastic_rate(ef, mass, 10*e, 300, 2330, 9000), acoustic*4)
close(m.acoustic_quasielastic_rate(ef, mass, 5*e, 600, 2330, 9000), acoustic*2)
close(m.acoustic_quasielastic_rate(ef, mass, 5*e, 300, 2330, 18000), acoustic/4)
for r in [-1, -.5, 0, .5, 1, 1.5]:
    temperature = 300
    x = np.linspace(1e-9, 50, 300001)
    dx = np.diff(x)
    numerator = x**(r+1.5)*np.exp(-x)
    denominator = x**1.5*np.exp(-x)
    integral = lambda values: np.sum((values[:-1]+values[1:])/2*dx)
    independent = tau*(m.KB*temperature/ef)**r*integral(numerator)/integral(denominator)
    close(m.thermal_mb_tau(tau, ef, r, temperature), independent, rtol=1e-5)
close(m.thermal_mb_tau(tau, ef, 0, 300), tau)

# Full critical neighbourhood + long-time numerical stability.
for omega in [0, .1, .49, .5-1e-12, .5, .5+1e-12, .51, 1, 3]:
    start = m.charge_mode(0, omega)
    close(start["a"], 1); close(start["derivative"], 0)
    times = np.linspace(0, 30, 301)
    mode = m.charge_mode(times, omega)
    check(np.all(np.isfinite(mode["a"])) and np.all(np.abs(mode["a"]) <= 1+1e-12))
    for s in [.1, 1, 4, 12]:
        h = 1e-5
        p, plus, minus = m.charge_mode(s, omega), m.charge_mode(s+h, omega), m.charge_mode(s-h, omega)
        close((plus["a"]-minus["a"])/(2*h), p["derivative"], rtol=1e-5, atol=1e-8)
        close((plus["derivative"]-minus["derivative"])/(2*h)+p["derivative"]+omega**2*p["a"], 0, atol=5e-8)
        for position in [.13, .37, .8]:
            dp = m.charge_profile(position, s+h, omega)
            dm = m.charge_profile(position, s-h, omega)
            jp = m.charge_profile(position+h, s, omega)
            jm = m.charge_profile(position-h, s, omega)
            # Scale continuity by e*A0/tau: -d_s(delta_n/A0)+d_(qx)(J/Jscale).
            residual = -(dp["delta_density_fraction"]-dm["delta_density_fraction"])/(.05*2*h)+(jp["current_scaled"]-jm["current_scaled"])/(4*math.pi*h)
            close(residual, 0, atol=1e-7)
            close(m.charge_profile(position, s, omega)["continuity_residual"], 0)
    # Independent RK4 evolution does not use the analytic roots.
    state, h = np.array([1., 0.]), .002
    f = lambda a: np.array([a[1], -a[1]-omega**2*a[0]])
    for i in range(3000):
        k1 = f(state); k2 = f(state+h*k1/2); k3 = f(state+h*k2/2); k4 = f(state+h*k3)
        state += h*(k1+2*k2+2*k3+k4)/6
        if i % 100 == 99:
            exact = m.charge_mode((i+1)*h, omega)
            close(state, [exact["a"], exact["derivative"]], rtol=1e-7, atol=1e-10)
for s in [0, 1, 4, 100, 1e6]:
    critical = m.charge_mode(s, .5)
    close(critical["a"], np.exp(-s/2)*(1+s/2))
    close(critical["derivative"], -s/4*np.exp(-s/2))
    check(np.isfinite(m.charge_mode(s, .1)["a"]))

for call in [lambda: m.drude_step(-1, n, mass, tau, field), lambda: m.drude_step(0, 0, mass, tau, field), lambda: m.drude_step(0, n, -mass, tau, field), lambda: m.drude_ac(-1, n, mass, tau), lambda: m.fermi_dirac(0, 0, 0), lambda: m.tight_binding_band(0, -e, 1e-9), lambda: m.inverse_mass_tensor([[1, 2], [0, 1]]), lambda: m.inverse_mass_tensor([]), lambda: m.first_order_transition(0, np.inf, 1), lambda: m.dos_3d_per_spin(-e, mass), lambda: m.screening_ratio(-1), lambda: m.yukawa_fourier(0, 1, m.EPS0, 0), lambda: m.angular_born_rates(ef, mass, ni, lambda q: q, 1), lambda: m.thermal_mb_tau(tau, ef, -2.5, 300), lambda: m.charge_mode(-1, 1), lambda: m.charge_profile(0, 1, 1, .2)]:
    rejects(call)

# Execute the actual published snippets in their original order and namespace.
# These do not import the download module: compare their numerical outputs to it.
repo = Path(__file__).resolve().parents[1]
article = repo / "_posts/2026-4Q/2026-11-07-Drude-Model-From-Scratch.md"
blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", article.read_text(), flags=re.M | re.S)
check(len(blocks) == 16)
block_markers = {
    "dc": "E_CHARGE = ", "step": "j_on = ", "ac": "omega_tau = np.array",
    "four_site": "ring_size, hopping_four_eV", "band": "def cosine_band(",
    "tensor": "Raxes = ", "probability": "def first_order_probability(",
    "continuum": "continuum_detunings = ", "fourier": "ell0, box_length = ",
    "impurity": "impurity_mass, kinetic = ", "screening": "def screened_time_ratio(",
    "phonon": "deformation_J, lattice_density, sound_speed = ",
    "thermal": "def mb_transport_time(", "charge": "def density_mode(",
    "starter": "def angular_relaxation_rates(",
    "reference": "def angular_relaxation_rates_reference(",
}
seen_blocks = set()
published, emitted = {}, io.StringIO()
with contextlib.redirect_stdout(emitted):
    for index, source in enumerate(blocks, 1):
        matches = [name for name, marker in block_markers.items() if marker in source]
        check(len(matches) == 1)
        kind = matches[0]
        check(kind not in seen_blocks)
        seen_blocks.add(kind)
        exec(compile(source, f"{article.name}:python-block-{index}", "exec"), published)
        if kind == "dc":
            for constant in ["E_CHARGE", "HBAR", "M0", "KB", "EPS0"]:
                close(published[constant], getattr(m, constant))
            dc = m.drude_step(100*published["tau"], published["n"], published["mass"], published["tau"], published["field"])
            close(published["mobility"], dc["mobility"])
            close(published["sigma0"], dc["conductivity"])
            close(published["drift"], dc["drift"])
            close(published["current"], dc["current_density"]*published["area"])
            close(published["voltage"], published["field"]*published["length"])
        elif kind == "step":
            transient = m.drude_step(published["s"]*published["tau"], published["n"], published["mass"], published["tau"], published["field"])
            close(published["j_on"], transient["current_density"]/(published["sigma0"]*published["field"]))
            close(published["j_off"], np.exp(-published["s"]))
        elif kind == "ac":
            close(published["response"], m.drude_ac(published["omega_tau"]/published["tau"], published["n"], published["mass"], published["tau"])/published["sigma0"])
            close(published["waveform"], np.cos(published["phase"]-np.pi/4)/np.sqrt(2), atol=1e-15)
        elif kind == "four_site":
            size, hopping = published["ring_size"], published["hopping_four_eV"]
            check(size == 4)
            shift = np.roll(np.eye(size), 1, axis=0)
            expected_matrix = hopping*(2*np.eye(size)-shift-shift.T)
            close(published["H_four"], expected_matrix)
            close(published["matrix_four_eV"], hopping*np.array([0., 2., 2., 4.]), atol=1e-14)
            wavevectors = published["ring_ka"]
            bloch = np.exp(1j*np.outer(np.arange(size), wavevectors))/np.sqrt(size)
            close(bloch.conj().T@bloch, np.eye(size), atol=1e-15)
            band = m.tight_binding_band(wavevectors/1e-9, hopping*e, 1e-9)["energy"]/e
            close(published["H_four"]@bloch, bloch*band, atol=1e-14)
            close(published["band_four_eV"], band, atol=1e-14)
        elif kind == "band":
            for hopping in [.1, 1, 3]:
                for lattice in [.1, .5, 1]:
                    ka = np.linspace(-np.pi, np.pi, 101)
                    energy_ev, velocity, inverse = published["cosine_band"](ka, hopping, lattice)
                    band = m.tight_binding_band(ka/(lattice*1e-9), hopping*e, lattice*1e-9)
                    close(energy_ev, band["energy"]/e, atol=1e-14)
                    close(velocity, band["velocity"], atol=1e-8)
                    close(inverse*m.M0, band["inverse_mass"]*m.M0, atol=1e-13)
            for ka in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
                for signed in [ka, -ka]:
                    energy_ev = published["cosine_band"](signed)[0]
                    expected = m.tight_binding_band(signed/.5e-9, e, .5e-9)["energy"]/e
                    check(energy_ev > 0)
                    close(energy_ev, expected, rtol=1e-12)
                    close(energy_ev/signed**2, 1, rtol=1e-8)
        elif kind == "tensor":
            close(published["inverse_M"], m.inverse_mass_tensor(published["inverse_M"]*m.HBAR**2))
            close(published["acceleration"][1]/published["acceleration"][0], np.sqrt(3)/4)
        elif kind == "probability":
            _, exact_probability = m.first_order_transition(published["detunings"]*e, .001*e, 50e-15)
            close(published["probability"], exact_probability, atol=1e-16)
            check(0 < published["p_resonant"] < .01)
        elif kind == "continuum":
            coupling = published["continuum_coupling_eV"]*e
            dos_per_joule = published["continuum_dos_per_eV"]/e
            golden_rate = 2*np.pi/m.HBAR*abs(coupling)**2*dos_per_joule
            close(published["continuum_gamma"], golden_rate)
            detuning_joule = published["continuum_detunings"]*e
            ratios = []
            for duration in [50e-15, 100e-15, 200e-15]:
                _, probabilities = m.first_order_transition(detuning_joule, coupling, duration)
                total = dos_per_joule*np.sum((probabilities[1:]+probabilities[:-1])/2*np.diff(detuning_joule))
                source_probabilities = published["first_order_probability"](published["continuum_detunings"], published["continuum_coupling_eV"], duration)
                source_total = published["continuum_dos_per_eV"]*np.sum((source_probabilities[1:]+source_probabilities[:-1])/2*np.diff(published["continuum_detunings"]))
                close(source_total, total)
                check(0 < total < 1e-3)
                close(total/(golden_rate*duration), 1, rtol=.01)
                ratios.append(total/(golden_rate*duration))
            check(ratios[0] < ratios[1] < ratios[2] < 1)
            close(published["continuum_total"], source_total)
        elif kind == "fourier":
            # This is a 1D Fourier overlap (J m), not the 3D impurity U~ (J m³).
            for width in [published["ell0"], 2*published["ell0"]]:
                shape = published["g_1d"]/(np.sqrt(2*np.pi)*width)*np.exp(-published["x_grid"]**2/(2*width**2))
                computed = np.sum(shape*np.exp(-1j*published["q_probe"]*published["x_grid"]))*published["dx"]
                exact = published["g_1d"]*np.exp(-.5*(published["q_probe"]*width)**2)
                close(computed/published["g_1d"], exact/published["g_1d"], atol=1e-8)
        elif kind == "impurity":
            short = m.short_range_born_rate(published["kinetic"], published["impurity_mass"], published["impurity_density"], published["a_born"])
            close(published["rate_flux"], short["rate_tr"])
            close(published["rate_dos"], m.golden_rule_continuum_rate(published["kinetic"], published["impurity_mass"], published["impurity_density"], published["u0"]))
            close(published["dos_one_spin"], m.dos_3d_per_spin(published["kinetic"], published["impurity_mass"]))
            close(published["tau_impurity"], 1/short["rate_q"])
        elif kind == "screening":
            for shape in [1e-12, 1e-8, .01, 4., 100., 400.]:
                close(published["screened_time_ratio"](shape), m.screening_ratio(shape), rtol=1e-9)
            for bad_shape in [0., -1., 400.01, 1e4, 1e6, np.nan, np.inf]:
                rejects(lambda value=bad_shape: published["screened_time_ratio"](value))
        elif kind == "phonon":
            close(published["rate_acoustic"], m.acoustic_quasielastic_rate(published["kinetic"], published["impurity_mass"], published["deformation_J"], published["temperature"], published["lattice_density"], published["sound_speed"]))
            momentum_max = 2*np.sqrt(2*published["impurity_mass"]*published["kinetic"])/m.HBAR
            phonon_max = m.HBAR*published["sound_speed"]*momentum_max
            close(published["q_max_elastic"], momentum_max)
            close(published["phonon_energy_max"], phonon_max)
            close(phonon_max/e*1e3, 3.3721299237441)
            close(phonon_max/(m.KB*300), .13043980936103233)
            close(phonon_max/published["kinetic"], .06744259847488199)
            check(phonon_max/(m.KB*30) > 1)
        elif kind == "thermal":
            for exponent in [-2, -1, -.5, 0, 1, 2]:
                close(published["mb_transport_time"](tau, ef, 300, exponent), m.thermal_mb_tau(tau, ef, exponent, 300))
            close(published["eff_600"]/published["eff_300"], 2**(-1.5))
            nodes, weights = np.polynomial.legendre.leggauss(256)
            y, weights = (nodes+1)/2*np.sqrt(80), weights/2*np.sqrt(80)
            boltzmann = np.exp(-y**2)
            prefactor = published["tau_impurity"]*(m.KB*300/published["kinetic"])**(-.5)
            transport_tau = prefactor*np.sum(weights*2*y**3*boltzmann)/np.sum(weights*2*y**4*boltzmann)
            particle_tau = prefactor*np.sum(weights*2*y*boltzmann)/np.sum(weights*2*y**2*boltzmann)
            close(published["impurity_eff_300"], transport_tau)
            close(published["impurity_particle_tau_300"], particle_tau)
            close(published["impurity_mobility_300"], e*transport_tau/published["impurity_mass"])
            close(particle_tau/transport_tau, 1.5)
        elif kind == "charge":
            times = np.array([0, .1, 1, 4, 12, 100])
            for omega in [.01, .1, .49, .5-1e-8, np.nextafter(.5, 0), .5-1e-14, .5, np.nextafter(.5, 1), .5+1e-14, .5+1e-8, .51, 1.2, 3]:
                a, derivative = published["density_mode"](times, omega)
                mode = m.charge_mode(times, omega)
                close(a, mode["a"], atol=1e-14)
                close(derivative, mode["derivative"], atol=1e-14)
            for omega in [.1, .5, 1., 3.]:
                for s in np.linspace(.1, 12, 31):
                    h, theta = 1e-5, 2*np.pi*.17
                    a, ap = published["density_mode"](s, omega)
                    plus = published["density_mode"](s+h, omega)
                    minus = published["density_mode"](s-h, omega)
                    field, current = -a*np.sin(theta), ap*np.sin(theta)
                    current_derivative = (plus[1]-minus[1])/(2*h)*np.sin(theta)
                    close(current_derivative+current-omega**2*field, 0, atol=1e-8)
                    energy = lambda state: .5*(state[1]**2+omega**2*state[0]**2)
                    close((energy(plus)-energy(minus))/(2*h)+ap**2, 0, atol=1e-8)
        elif kind == "starter":
            try:
                published["angular_relaxation_rates"](np.array([4., 6.]), np.array([1., 0.]))
            except NotImplementedError:
                checks += 1
            else:
                raise AssertionError("Run All must preserve the learner's unimplemented starter")
        elif kind == "reference":
            check(published["check_angular_rates"](published["angular_relaxation_rates_reference"]) == "4 cases passed")
check(seen_blocks == set(block_markers))

# Direct Eq. (27) quadrature, independent of the Maxwell–Boltzmann gamma
# average: a constant tau recovers Drude at nondegenerate and degenerate eta.
nodes, weights = np.polynomial.legendre.leggauss(512)
y, weights = (nodes+1)/2*10, weights/2*10
for eta in [-8, 0, 8, 30]:
    occupation = m.fermi_dirac(y**2*m.KB*300, eta*m.KB*300, 300)
    number_integral = np.sum(weights*2*y**2*occupation)
    conductivity_integral = 2/3*np.sum(weights*2*y**4*occupation*(1-occupation))
    close(conductivity_integral/number_integral, 1, rtol=1e-12)

# A matrix element can be complex even for a static Hermitian perturbation.
# Probability must depend on |M|², not M², and remain real/nonnegative.
phase_detunings = np.array([-.02, 0., .03])
for coupling_ev in [.001j, .001*np.exp(1j*np.pi/4), .0003+.0004j]:
    actual = published["first_order_probability"](phase_detunings, coupling_ev, 50e-15)
    _, expected = m.first_order_transition(phase_detunings*e, coupling_ev*e, 50e-15)
    close(actual, expected, atol=1e-16)
    close(actual, published["first_order_probability"](phase_detunings, abs(coupling_ev), 50e-15), atol=1e-16)
    check(np.isrealobj(actual) and np.all(actual >= 0))

checker, reference = published["check_angular_rates"], published["angular_relaxation_rates_reference"]
check(checker(reference) == "4 cases passed")
check(published["angular_relaxation_rates"] is not reference)
try:
    published["angular_relaxation_rates"](np.array([8.]), np.array([-1.]))
except NotImplementedError:
    checks += 1
else:
    raise AssertionError("Reference execution replaced the learner's starter")


def mutate_rates(rates, cosines):
    result = reference(rates, cosines)
    rates[:] = 0
    return result


def mutate_cosines(rates, cosines):
    result = reference(rates, cosines)
    cosines[:] = 0
    return result


def alias_at_zero(rates, cosines):
    return rates.view() if not np.any(rates) else reference(rates, cosines)


wrong_answers = [
    lambda r, c: np.array([r.sum(), r.sum()]),
    lambda r, c: np.array([r.sum(), np.dot(r, 1+c)]),
    lambda r, c: np.array([1., np.dot(r, 1-c)/r.sum()]),
    lambda r, c: np.array([r.sum(), np.mean(1-c)]),
    lambda r, c: reference(r, c)*2,
    lambda r, c: reference(r, c)[:, None],
    lambda r, c: np.array([np.nan, np.nan]),
    mutate_rates, mutate_cosines, alias_at_zero,
]
for candidate in wrong_answers:
    try:
        checker(candidate)
    except AssertionError:
        checks += 1
    else:
        raise AssertionError("The published coding checker accepted a wrong/mutating/aliasing answer")
print(f"Drude physics: {checks} assertions passed; {len(blocks)} published Python blocks executed in order and cross-checked; {len(wrong_answers)} wrong exercise answers rejected; starter remains NotImplemented.")
