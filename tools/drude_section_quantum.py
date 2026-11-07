"""Section-specific quantum figures for the Drude tutorial.

Renderers receive the shared figure API as ``d``.  The Gaussian and sinc data
are analytic; the angular schematic uses phase fronts separated by one common
wavelength.  This module never changes the article or chooses output paths.
"""

from __future__ import annotations

import cmath
from math import cos, exp, pi, sin, sqrt
import xml.etree.ElementTree as ET


HBAR_EV_FS = 0.6582119569509067


def _left_math(d, f, key, formula, left, top, size=30, max_width=500):
    root = ET.fromstring(d.equation_svg(formula))
    width = float(root.attrib["viewBox"].split()[2])
    width = min(width * size / 10, max_width)
    d.math_label(f, key, formula, left + width / 2, top, size, max_width)


def _legend(d, f, key, formula, x, top, color, size=30, max_width=200):
    f.line(x, top + 14, x + 35, top + 14, color, 4.2)
    _left_math(d, f, key, formula, x + 52, top, size, max_width)


def packet_density(xi, s):
    """Delta_x0 times an envelope density, xi=x/Delta_x0, s=t/t0.

    Initial position variance is Delta_x0 squared, t0=2*m*Delta_x0^2/hbar,
    and F*t0^2/(2*m*Delta_x0)=1. Uniform force shifts the center by s squared;
    the quadratic dispersion broadens the variance to 1+s squared.
    """
    variance = 1 + s * s
    return exp(-(xi - s * s) ** 2 / (2 * variance)) / sqrt(2 * pi * variance)


def wave_packet(d, compact=False):
    f = d.Figure(600 if compact else 1200, 1220 if compact else 920,
                 "A localized parabolic-band envelope moves and spreads",
                 "A normalized Gaussian envelope evolves under the one-dimensional "
                 "Hamiltonian p squared/(2m*) minus F x, with positive constant m* "
                 "and external F. The initial center and mean velocity are zero. "
                 "The initial position standard deviation is Delta x0, t0=2m*Delta x0 "
                 "squared/hbar and F*t0 squared/(2m*Delta x0)=1. At s=t/t0=0,1,2, "
                 "the position centers x_c/Delta x0 are 0,1,4 and the widths are "
                 "sqrt(1+s squared). Dashed lines mark expectation-value centers, "
                 "not trajectories of individual electrons. The plotted density "
                 "is the coarse-grained envelope of an isotropic parabolic band.")
    d.header(f, "Parabolic-band motion: the packet moves and spreads",
             ("A packet moves and spreads", "in a parabolic band"), compact)
    d.math_label(f, "packet-hamiltonian", r"\mathcal H_{\rm env}=\hbar^2k^2/(2m^{\ast})-Fx",
                 f.width / 2, 130 if compact else 95, 30, f.width - 60)
    if compact:
        for key, formula, top in (
            ("time", r"t_0=2m^{\ast}(\Delta x_0)^2/\hbar", 190),
            ("force", r"Ft_0^2/(2m^{\ast}\Delta x_0)=1", 250),
            ("center", r"s=t/t_0,\quad x_c/\Delta x_0=s^2", 310),
            ("width", r"\Delta x(t)/\Delta x_0=\sqrt{1+s^2}", 370),
        ):
            d.math_label(f, "packet-" + key, formula, 300, top, 30, 540)
        f.text(30, 451, "Zero initial center and mean velocity", 25, d.MUTED)
        box = (95, 545, 445, 330)
        ylabel_top, legend_top, xlabel_top = 495, 1010, 941
        footer_top = 1100
    else:
        d.math_label(f, "packet-scales", r"t_0=2m^{\ast}(\Delta x_0)^2/\hbar,\qquad Ft_0^2/(2m^{\ast}\Delta x_0)=1",
                     600, 150, 31, 1110)
        d.math_label(f, "packet-laws", r"s=t/t_0,\qquad x_c/\Delta x_0=s^2,\qquad\Delta x(t)/\Delta x_0=\sqrt{1+s^2}",
                     600, 208, 31, 1110)
        f.text(35, 281, "Uniform external F > 0 · positive constant mass · zero initial mean velocity", 25, d.MUTED)
        box = (110, 363, 1025, 300)
        ylabel_top, legend_top, xlabel_top = 311, 783, 729
        footer_top = 851
    d.math_label(f, "packet-density-axis", r"\Delta x_0\,|\psi_{\rm env}(x,t)|^2",
                 box[0] + 160, ylabel_top, 30, 330)
    X, Y = d.axes(f, box, (-4, 10), (0, .43), (-4, 0, 4, 8, 10), (0, .2, .4), "", "")
    for s, color in zip((0, 1, 2), (d.BLUE, d.TEAL, d.GOLD)):
        f.path([(X(xi), Y(packet_density(xi, s))) for xi in d.points(-4, 10, 701)], color)
        f.line(X(s * s), Y(0), X(s * s), Y(packet_density(s * s, s)), color, 1.6, dash=True)
    d.math_label(f, "packet-position-axis", r"x/\Delta x_0", box[0] + box[2] / 2, xlabel_top, 31, 180)
    for index, (s, color) in enumerate(zip((0, 1, 2), (d.BLUE, d.TEAL, d.GOLD))):
        _legend(d, f, f"packet-time-{index}", rf"s={s}",
                (45 + 175 * index) if compact else (155 + 295 * index), legend_top, color, 30, 100)
    d.math_label(f, "packet-expectation", r"x_c=\langle x\rangle,\qquad\ddot x_c=F/m^{\ast}",
                 f.width / 2, footer_top, 30, f.width - 60)
    if compact:
        f.text(30, 1182, "Dashed lines: expectation-value centers", 23, d.MUTED)
    f.save("fig-sec08-wave-packet" + ("-mobile" if compact else ""))


def occupations(d, compact=False):
    f = d.Figure(600 if compact else 1200, 1160 if compact else 790,
                 "Opposite occupation changes carry current without adding carriers",
                 "Two equally weighted electron states have velocities minus v and "
                 "plus v, with v positive. Both equilibrium occupations are 0.500. "
                 "The perturbed occupations are f_minus=0.501 and f_plus=0.499. "
                 "The signed bars show delta f divided by 0.001: plus one for the "
                 "minus-v state and minus one for the plus-v state. The number "
                 "weight remains one, but the velocity weight is minus 0.002v. "
                 "Multiplying by electron charge minus e and the common positive "
                 "state-counting weight gives a positive current contribution.")
    d.header(f, "An occupation imbalance carries current at fixed density",
             ("An occupation imbalance", "carries current"), compact)
    d.math_label(f, "occupation-reference", r"f_+^{(0)}=f_-^{(0)}=0.500,\qquad v>0",
                 f.width / 2, 130 if compact else 103, 31, f.width - 60)
    box = (95, 270, 445, 340) if compact else (120, 235, 1010, 270)
    d.math_label(f, "occupation-change-axis", r"\delta f/10^{-3}", box[0] + 100,
                 box[1] - 54, 30, 210)
    X, Y = d.axes(f, box, (-1.8, 1.8), (-1.7, 1.6), ((-1, "−v"), (1, "+v")), (-1, 0, 1), "", "")
    for velocity, value, color, formula in (
        (-1, 1, d.TEAL, r"f_-=0.501"),
        (1, -1, d.GOLD, r"f_+=0.499"),
    ):
        left, right = X(velocity - .27), X(velocity + .27)
        f.rect(left, min(Y(0), Y(value)), right - left, abs(Y(value) - Y(0)), color, color)
        top = Y(value) - 45 if value > 0 else Y(value) + 12
        d.math_label(f, "occupation-value-" + ("minus" if velocity < 0 else "plus"),
                     formula, X(velocity), top, 29, 230 if not compact else 210)
    f.text(box[0] + box[2] / 2, box[1] + box[3] + 83, "Velocity state", 28, d.INK, "middle")
    if compact:
        f.text(30, 780, "Number weight: unchanged", 27, d.INK, weight=500)
        d.math_label(f, "occupation-number", r"f_++f_-=1.000", 300, 814, 31, 540)
        f.text(30, 913, "Velocity weight: no longer zero", 27, d.INK, weight=500)
        d.math_label(f, "occupation-velocity", r"vf_+-vf_-=-0.002v", 300, 949, 31, 540)
        f.text(30, 1071, "Electron charge −e:", 26, d.MUTED)
        f.text(30, 1113, "positive conventional current", 26, d.TEAL)
    else:
        f.text(80, 625, "Number weight: unchanged", 28, d.INK, weight=500)
        d.math_label(f, "occupation-number", r"f_++f_-=1.000", 320, 666, 32, 460)
        f.text(660, 625, "Velocity weight: no longer zero", 28, d.INK, weight=500)
        d.math_label(f, "occupation-velocity", r"vf_+-vf_-=-0.002v", 910, 666, 32, 480)
        f.text(35, 759, "Electron charge −e reverses the sign: the pair carries a positive conventional current.", 24, d.MUTED)
    f.save("fig-sec10-occupations" + ("-mobile" if compact else ""))


def sinc(z):
    return 1.0 if abs(z) < 1e-14 else sin(z) / z


def transition_peak(delta, s):
    """P/P_ref, delta=DeltaE*t0/hbar and s=t/t0."""
    return s * s * sinc(delta * s / 2) ** 2


def golden_rule(d, compact=False):
    pref = (.001 * 25 / HBAR_EV_FS) ** 2
    f = d.Figure(600 if compact else 1200, 1410 if compact else 915,
                 "A finite-time transition peak becomes taller and narrower",
                 "The weak-transition example has coupling magnitude one meV and "
                 "reference time t0=25 femtoseconds. P_ref=(M*t0/hbar) squared is "
                 f"{pref:.8f}. With delta=DeltaE*t0/hbar and s=t/t0, the curves "
                 "show P/P_ref=s squared sinc squared(s*delta/2) at s=1 and2. "
                 "The second graph gives the full detuning-axis area divided by "
                 "2pi P_ref; it equals s. The first graph displays only a finite "
                 "central window. A smooth, broad continuum of final states converts "
                 "this integrated area into a summed probability linear in time; "
                 "the probability for one exactly resonant state remains quadratic.")
    d.header(f, "A transition peak narrows while its integrated area grows",
             ("A transition peak narrows", "while its area grows"), compact)
    d.math_label(f, "golden-coordinates", r"s=t/t_0,\qquad\delta=\Delta E\,t_0/\hbar",
                 f.width / 2, 130 if compact else 103, 30, f.width - 60)
    d.math_label(f, "golden-reference", r"P_{\rm ref}=(|M|t_0/\hbar)^2\simeq1.44\times10^{-3}",
                 f.width / 2, 190 if compact else 155, 30, f.width - 60)
    f.text(30 if compact else 35, 253 if compact else 214, "Fixed |M| = 1 meV · t₀ = 25 fs", 25, d.MUTED)
    if compact:
        boxes = ((95, 450, 445, 240), (95, 1070, 445, 200))
        heads, formula_tops = ((30, 320), (30, 910)), (350, 942)
    else:
        boxes = ((100, 395, 445, 290), (700, 395, 445, 290))
        heads, formula_tops = ((100, 267), (700, 267)), (299, 289)
    f.text(*heads[0], "(a) Finite-time energy peak", 28, weight=500)
    d.math_label(f, "golden-finite-time", r"P/P_{\rm ref}=s^2\operatorname{sinc}^2(s\delta/2)",
                 boxes[0][0] + boxes[0][2] / 2, formula_tops[0], 30, 535 if compact else 445)
    d.math_label(f, "golden-peak-axis", r"P/P_{\rm ref}", boxes[0][0] + 83,
                 boxes[0][1] - 46, 30, 200)
    X, Y = d.axes(f, boxes[0], (-12, 12), (0, 4.2), (-12, -6, 0, 6, 12), (0, 2, 4), "", "")
    for s, color in ((1, d.BLUE), (2, d.TEAL)):
        f.path([(X(delta), Y(transition_peak(delta, s))) for delta in d.points(-12, 12, 1201)], color)
    d.math_label(f, "golden-detuning", r"\delta", boxes[0][0] + boxes[0][2] / 2,
                 boxes[0][1] + boxes[0][3] + 65, 31, 160)
    legend_top = 810 if compact else 806
    _legend(d, f, "golden-time-one", r"s=1", 100, legend_top, d.BLUE, 30, 140)
    _legend(d, f, "golden-time-two", r"s=2", 325, legend_top, d.TEAL, 30, 140)
    f.text(*heads[1], "(b) Full peak area", 28, weight=500)
    d.math_label(f, "golden-area-law", r"\mathrm{Area}/(2\pi)=s",
                 boxes[1][0] + boxes[1][2] / 2, formula_tops[1], 30, 535 if compact else 445)
    if compact:
        f.text(boxes[1][0], 1008, "All detunings", 25, d.MUTED)
    d.math_label(f, "golden-area-axis", r"\mathrm{Area}/(2\pi)", boxes[1][0] + 129,
                 boxes[1][1] - 45, 29, 270)
    X, Y = d.axes(f, boxes[1], (0, 2), (0, 2.1), (0, 1, 2), (0, 1, 2), "", "")
    f.path([(X(s), Y(s)) for s in d.points(0, 2)], d.GOLD)
    d.math_label(f, "golden-area-time", r"s=t/t_0", boxes[1][0] + boxes[1][2] / 2,
                 boxes[1][1] + boxes[1][3] + 65, 31, 210)
    if not compact:
        f.text(35, 862, "Area integrates P/P_ref over all detunings; (a) shows only the central window.", 24, d.MUTED)
        f.text(35, 902, "A flat final-state density converts the full peak area into a total transition probability.", 25, d.MUTED)
    f.save("fig-sec11-golden-rule" + ("-mobile" if compact else ""))


def born_cross_section(d, compact=False):
    f = d.Figure(600 if compact else 1200, 1480 if compact else 870,
                 "The outgoing scattering amplitude sets an effective cross section",
                 "A meridional section of an isotropic three-dimensional Born "
                 "scattering state. The incident direction is positive z. Blue "
                 "parallel lines are equal-phase incident plane-wave fronts. Teal "
                 "circles are equal-phase outgoing spherical-wave fronts with "
                 "spacing lambda, the same as the plane fronts. The scatterer is "
                 "at the labeled origin. A dashed radial coordinate and angle theta "
                 "are coordinates, not particle paths. The chosen amplitude is "
                 "f_sc=-a_B, so the differential cross section is a_B squared and "
                 "the total is4pi a_B squared. With impurity density10^22 per cubic "
                 "metre, speed10^5m/s and total cross section10^-18m squared, the "
                 "event rate is10^9 per second and the waiting time is1ns.")
    d.header(f, "Scattering amplitude, cross section and event rate",
             ("From scattering amplitude", "to cross section and rate"), compact)
    if compact:
        center, cy = 300, 450
        f.text(30, 178, "(a) Section through the incident axis", 27, weight=500)
        label_y, bx = 222, 300
        diagram_bottom, second_top = 756, 862
    else:
        center, cy = 330, 440
        f.text(35, 173, "(a) Section through the incident axis", 27, weight=500)
        label_y, bx = 214, 925
        diagram_bottom, second_top = 768, 175
    # Every front in each wave family has the same phase, with lambda=60
    # drawing units. At the incident fronts z=-3lambda,-2lambda,-lambda.
    # Since f=-a_B, outgoing maxima occur at r=(n+1/2)lambda:90,150,210.
    for radius in (90, 150, 210):
        f.nodes.append(f'<circle cx="{center}" cy="{cy}" r="{radius}" fill="none" stroke="{d.TEAL}" stroke-width="2"/>')
    for z in (-180, -120, -60):
        f.line(center + z, cy - 132, center + z, cy + 132, d.BLUE, 2.8)
    f.line(35, cy, 565 if compact else 632, cy, d.MUTED, 1.8, arrow=True)
    d.math_label(f, "born-z-coordinate", "z", 571 if compact else 646, cy - 15, 30, 35)
    d.math_label(f, "born-origin", "0", center, cy + 15, 28, 35)
    f.circle(center, cy, 5, d.INK)
    theta, radius = pi / 6, 210
    f.line(center, cy, center + radius * cos(theta), cy - radius * sin(theta), d.MUTED, 1.6, arrow=True, dash=True)
    f.path([(center + 60 * cos(q), cy - 60 * sin(q)) for q in d.points(0, theta, 41)], d.GOLD, 2.4)
    d.math_label(f, "born-angle", r"\theta", center + 67, cy - 39, 30, 45)
    d.math_label(f, "born-radius", "r", center + 100, cy - 80, 30, 45)
    d.math_label(f, "born-incident", r"e^{ikz}", center - 151, label_y, 31, 180)
    d.math_label(f, "born-outgoing", r"-a_Be^{ikr}/r", center + 148, label_y, 30, 250)
    f.text(35 if not compact else 30, cy + 241, "Localized scatterer at the origin", 25, d.MUTED)
    _legend(d, f, "born-plane-legend", r"\mathrm{incident\ phase\ fronts}", 35,
            diagram_bottom - 20 if compact else diagram_bottom - 60, d.BLUE, 26, 430)
    _legend(d, f, "born-spherical-legend", r"\mathrm{outgoing\ phase\ fronts}", 35,
            diagram_bottom + 25 if compact else diagram_bottom - 15, d.TEAL, 26, 430)
    if not compact:
        d.math_label(f, "born-wavelength", r"k=2\pi/\lambda", 329, 803, 29, 350)
    f.text(30 if compact else 700, second_top, "(b) Isotropic Born example", 28, weight=500)
    offsets = (34, 91, 191, 263) if compact else (43, 107, 206, 281)
    for key, formula, offset in (
        ("amplitude", r"f_{\rm sc}=-a_B,\quad a_B>0", offsets[0]),
        ("differential", r"\frac{d\sigma_{\rm sc}}{d\Omega}=a_B^2", offsets[1]),
        ("total", r"\sigma_{\rm sc}^{\rm total}=4\pi a_B^2", offsets[2]),
        ("rate", r"\Gamma_q=n_iv\,\sigma_{\rm sc}^{\rm total}", offsets[3]),
    ):
        d.math_label(f, "born-" + key, formula, bx, second_top + offset, 31, 540 if compact else 445)
    numeric_top = 1197 if compact else 558
    f.line(30 if compact else 700, numeric_top - 23, 570 if compact else 1165, numeric_top - 23)
    for index, formula in enumerate((r"n_i=10^{22}\,\mathrm{m}^{-3}", r"v=10^5\,\mathrm{m/s}",
                                    r"\sigma_{\rm sc}^{\rm total}=10^{-18}\,\mathrm{m}^2")):
        d.math_label(f, f"born-numeric-{index}", formula, bx, numeric_top + index * 49, 29, 540 if compact else 445)
    d.math_label(f, "born-numeric-rate", r"\Gamma_q=10^9\,\mathrm{s}^{-1},\quad\tau_q=1\,\mathrm{ns}",
                 bx, numeric_top + 165, 30, 540 if compact else 445)
    if compact:
        d.math_label(f, "born-wavelength", r"k=2\pi/\lambda", 300, 1444, 28, 400)
    f.save("fig-sec13-born-cross-section" + ("-mobile" if compact else ""))


def _integrate(fun, a, b, count):
    step = (b - a) / count
    return step * (.5 * fun(a) + sum(fun(a + i * step) for i in range(1, count)) + .5 * fun(b))


def _packet_wavefunction(x, t):
    """Exact envelope solution in units hbar=m=Delta_x0=1, F=1/2."""
    spreading = 1 + .5j * t
    center = .25 * t * t
    free = (2*pi) ** (-.25) / cmath.sqrt(spreading) * cmath.exp(-(x-center) ** 2 / (4*spreading))
    return cmath.exp(1j*(.5*t*x-t**3/24)) * free


def verify():
    """Independent moment, weight, line-shape and dimensional examples."""
    for s in (0, 1, 2):
        lo, hi = s*s - 12*sqrt(1+s*s), s*s + 12*sqrt(1+s*s)
        norm = _integrate(lambda q: packet_density(q, s), lo, hi, 12000)
        mean = _integrate(lambda q: q * packet_density(q, s), lo, hi, 12000)
        variance = _integrate(lambda q: (q-s*s)**2 * packet_density(q, s), lo, hi, 12000)
        assert abs(norm - 1) < 1e-12
        assert abs(mean - s*s) < 1e-11
        assert abs(variance - (1+s*s)) < 1e-11
    step = 1e-4
    for t in (.4, 2, 4):
        for x in (-2, 0, 3, 6):
            psi = _packet_wavefunction(x, t)
            assert abs(abs(psi)**2-packet_density(x, t/2)) < 1e-14
            derivative_t = (_packet_wavefunction(x,t+step)-_packet_wavefunction(x,t-step))/(2*step)
            derivative_xx = (_packet_wavefunction(x+step,t)-2*psi+_packet_wavefunction(x-step,t))/step**2
            assert abs(1j*derivative_t+.5*derivative_xx+.5*x*psi) < 2e-6
    assert abs((.501 + .499) - 1) < 1e-14
    assert abs((.499 - .501) + .002) < 1e-14
    pref = (.001 * 25 / HBAR_EV_FS) ** 2
    assert .0014 < pref < .0015 and 4 * pref < .006
    for s in (1, 2):
        assert abs(transition_peak(0, s) - s*s) < 1e-14
        assert transition_peak(2*pi/s, s) < 1e-28
        area = _integrate(lambda delta: transition_peak(delta, s), -400, 400, 160000)
        assert abs(area/(2*pi*s) - 1) < .003
    assert abs(1e22 * 1e5 * 1e-18 / 1e9 - 1) < 1e-14
    amplitude = sqrt(1e-18/(4*pi))
    assert abs(4*pi*amplitude*amplitude/1e-18 - 1) < 1e-14
    for z in (-180, -120, -60):
        assert abs(cos(2*pi*z/60) - 1) < 1e-14
    for radius in (90, 150, 210):
        assert abs(-cos(2*pi*radius/60) - 1) < 1e-14
    return {"packet_moments": 9, "packet_schrodinger_checks": 24, "occupation_weights": 2, "finite_time_peaks": 6,
            "cross_section_examples": 2, "phase_fronts": 6}


FIGURES = {
    8: (wave_packet, "fig-sec08-wave-packet"),
    10: (occupations, "fig-sec10-occupations"),
    11: (golden_rule, "fig-sec11-golden-rule"),
    13: (born_cross_section, "fig-sec13-born-cross-section"),
}
