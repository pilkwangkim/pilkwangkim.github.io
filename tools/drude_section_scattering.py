"""Additional exact-data figures for Drude sections 14, 15, 17, 18 and 20.

Renderers receive the main figure module as ``d`` so its output directory,
typesetter, palette and coordinate helpers stay authoritative. No renderer
samples random particle trajectories or imports a material dataset.
"""

from __future__ import annotations

from math import cos, exp, expm1, gamma, pi, sin, sqrt
import xml.etree.ElementTree as ET


def _axes(d, f, box, xlim, ylim, xticks, yticks, xtex, ytex):
    X, Y = d.axes(f, box, xlim, ylim, xticks, yticks, "", "")
    x, y, w, h = box
    d.math_label(f, f"axis-x-{x}-{y}", xtex, x+w/2, y+h+65, 30, w+40)
    if ytex:
        d.math_label(f, f"axis-y-{x}-{y}", ytex, x+min(w/2, 125), y-49, 30, w+10)
    return X, Y


def _legend(d, f, entries, x, top, width=500, gap=54):
    for index, (formula, color, dashed) in enumerate(entries):
        yy = top+index*gap
        f.line(x, yy+16, x+35, yy+16, color, 4.2, dash=dashed)
        root = ET.fromstring(d.equation_svg(formula))
        natural_width = float(root.attrib["viewBox"].split()[2])
        ink_width = min(natural_width*3, width-55)
        d.math_label(f, f"legend-{x}-{yy}", formula, x+55+ink_width/2,
                     yy, 30, width-55)


def angular_relaxation(d, compact=False):
    """Project one elastic deflection, then plot its exact angular loss."""
    f = d.Figure(600 if compact else 1200, 1430 if compact else 835,
                 "Scattering angle controls the loss of directional momentum",
                 "Wavevectors before and after one elastic event have the same length. "
                 "The example deflects through 60 degrees, leaving k cos(theta) in the "
                 "incoming direction and losing k[1-cos(theta)]. A separate curve shows "
                 "1-cos(theta) from zero through pi, with the same 60-degree example "
                 "marked. The physical momentum-component loss is hbar times this "
                 "wavevector-component loss. Vectors live in wavevector space.")
    d.header(f, "Scattering angle controls directional relaxation",
             ("Scattering angle and", "directional relaxation"), compact)
    d.math_label(f, "elastic-shell", r"|\mathbf k^{\prime}|=|\mathbf k|=k", f.width/2,
                 132 if compact else 100, 31, f.width-70)
    if compact:
        f.text(30, 225, "(a) One elastic deflection", 29, weight=500)
        origin, length = (125, 585), 345
        curve_box = (100, 990, 440, 270)
        f.text(30, 882, "(b) Lost forward component", 29, weight=500)
    else:
        f.text(50, 187, "(a) One elastic deflection", 29, weight=500)
        origin, length = (150, 520), 330
        curve_box = (735, 320, 410, 310)
        f.text(675, 187, "(b) Lost forward component", 29, weight=500)
    ox, oy = origin
    angle = pi/3
    px, py = ox+length*cos(angle), oy-length*sin(angle)
    f.line(ox-25, oy, ox+length+40, oy, d.GRID, 1.6)
    f.line(ox, oy, ox+length, oy, d.BLUE, 4.3, arrow=True)
    f.line(ox, oy, px, py, d.TEAL, 4.3, arrow=True)
    f.line(px, py, px, oy, d.MUTED, 1.7, dash=True)
    f.path([(ox+65*cos(t), oy-65*sin(t)) for t in d.points(0, angle, 51)], d.GOLD, 2.1)
    d.math_label(f, "incident-k", r"\mathbf k", ox+length+8, oy+18, 32, 70)
    d.math_label(f, "outgoing-k", r"\mathbf k^{\prime}", px+40, py+13, 32, 75)
    d.math_label(f, "theta-example", r"\theta=60^{\circ}", ox+100, oy-85, 30, 170)
    yy = oy+58
    f.line(ox, yy, px, yy, d.TEAL, 2.3)
    f.line(ox, yy-7, ox, yy+7, d.TEAL, 1.8)
    f.line(px, yy-7, px, yy+7, d.TEAL, 1.8)
    d.math_label(f, "retained-projection", r"k\cos\theta", (ox+px)/2, yy+19, 30, 200)
    yy += 91
    f.line(px, yy, ox+length, yy, d.GOLD, 2.3)
    f.line(px, yy-7, px, yy+7, d.GOLD, 1.8)
    f.line(ox+length, yy-7, ox+length, yy+7, d.GOLD, 1.8)
    d.math_label(f, "lost-projection", r"k-k_x^{\prime}=k(1-\cos\theta)",
                 (px+ox+length)/2, yy+20, 30, 410 if compact else 420)
    if compact:
        f.text(30, 826, "Vectors in wavevector space", 25, d.MUTED)
    else:
        f.text(50, 788, "Vectors in wavevector space", 25, d.MUTED)
    X, Y = _axes(d, f, curve_box, (0, pi), (0, 2.15),
                 ((0, "0"), (pi/2, "π/2"), (pi, "π")), (0, 1, 2),
                 r"\theta", r"1-\cos\theta")
    f.path([(X(t), Y(1-cos(t))) for t in d.points(0, pi)], d.GOLD)
    f.line(X(angle), Y(0), X(angle), Y(.5), d.MUTED, 1.6, dash=True)
    f.circle(X(angle), Y(.5), 6.2, d.GOLD)
    d.math_label(f, "loss-example", r"60^{\circ}\ \longrightarrow\ 1/2",
                 X(angle)+100, Y(.5)+16, 29, 260)
    f.save("fig-sec14-angular-relaxation"+ ("-mobile" if compact else ""))


def short_range(d, compact=False):
    """Use the same constant Born amplitude at every plotted positive energy."""
    f = d.Figure(600 if compact else 1200, 1340 if compact else 805,
                 "Short-range impurity rates change with the final-state density",
                 "In an isotropic 3D parabolic band with fixed mass, impurity density "
                 "and constant Born amplitude, the final-state DOS and scattering rate "
                 "are proportional to sqrt(K), while the relaxation time is proportional "
                 "to one over sqrt(K). Curves start at K/Kref=0.1, not at zero. "
                 "Kref=50 meV and tauref=0.67086 ps refer to the article's example.")
    d.header(f, "A constant matrix element does not give a constant rate",
             ("Constant matrix element,", "energy-dependent rate"), compact)
    if compact:
        f.text(30, 143, "3D parabolic band", 25, d.MUTED)
        f.text(30, 181, "Fixed mass and impurity model", 25, d.MUTED)
    else:
        f.text(35, 106, "3D parabolic band · fixed mass and impurity model", 27, d.MUTED)
    if compact:
        boxes = ((95, 330, 445, 270), (95, 910, 445, 270))
        heads = ((30, 225), (30, 805))
    else:
        boxes = ((100, 300, 445, 300), (700, 300, 445, 300))
        heads = ((100, 183), (700, 183))
    f.text(*heads[0], "(a) Final states set the rate", 29, weight=500)
    d.math_label(f, "rate-law", r"\Gamma/\Gamma_{\rm ref}=D_1/D_{1,\rm ref}=\sqrt{K/K_{\rm ref}}",
                 boxes[0][0]+boxes[0][2]/2, heads[0][1]+28, 30, 530 if compact else 500)
    X, Y = _axes(d, f, boxes[0], (.1, 4), (0, 2.15), (.1, 1, 2, 4), (0, 1, 2),
                 r"K/K_{\rm ref}", "")
    f.path([(X(x), Y(sqrt(x))) for x in d.points(.1, 4)], d.BLUE)
    f.line(X(1), Y(0), X(1), Y(1), d.MUTED, 1.6, dash=True)
    f.circle(X(1), Y(1), 5.5, d.BLUE)
    f.text(*heads[1], "(b) The reciprocal time", 29, weight=500)
    d.math_label(f, "time-law", r"\tau/\tau_{\rm ref}=\sqrt{K_{\rm ref}/K}",
                 boxes[1][0]+boxes[1][2]/2, heads[1][1]+28, 30, 445)
    X, Y = _axes(d, f, boxes[1], (.1, 4), (0, 3.4), (.1, 1, 2, 4), (0, 1, 2, 3),
                 r"K/K_{\rm ref}", "")
    f.path([(X(x), Y(1/sqrt(x))) for x in d.points(.1, 4)], d.GOLD)
    f.line(X(1), Y(0), X(1), Y(1), d.MUTED, 1.6, dash=True)
    f.circle(X(1), Y(1), 5.5, d.GOLD)
    top = 1282 if compact else 743
    d.math_label(f, "reference", r"K_{\rm ref}=50\,{\rm meV},\quad\tau_{\rm ref}=0.67086\,{\rm ps}",
                 f.width/2, top, 30, f.width-70)
    f.save("fig-sec15-short-range"+ ("-mobile" if compact else ""))


def acoustic_phonons(d, compact=False):
    """A classical mode snapshot exposes strain; Bose factors expose quanta."""
    f = d.Figure(600 if compact else 1200, 1620 if compact else 970,
                 "Longitudinal strain couples to electrons and phonon counts weight transitions",
                 "Panel a is a classical longitudinal normal-mode shape u_lat,x/u0=cos(qx), with q u0 much less than one. "
                 "Its strain and deformation potential are Uac/(D q u0)=-sin(qx), so "
                 "a displacement maximum need not be a band-edge-shift maximum. "
                 "Panel b shows equilibrium phonon factors Nq=1/[exp(z)-1] for absorption "
                 "and Nq+1 for emission, where z=hbar omegaq/(kBT). The dashed curve 1/z "
                 "is the high-occupation approximation to Nq, not an independent channel.")
    d.header(f, "Lattice strain and phonon occupation play different roles",
             ("Lattice strain and", "phonon occupation"), compact)
    d.math_label(f, "small-strain", r"q>0,\qquad qu_0\ll1", f.width/2,
                 132 if compact else 101, 30, f.width-70)
    if compact:
        boxes = ((95, 345, 445, 270), (95, 1040, 445, 270))
        heads = ((30, 198), (30, 875))
    else:
        boxes = ((100, 320, 445, 300), (700, 320, 445, 300))
        heads = ((100, 170), (700, 170))
    f.text(*heads[0], "(a) Classical mode shape", 29, weight=500)
    d.math_label(f, "displacement-model", r"\mathbf u_{\rm lat}=u_0\cos(qx)\,\widehat{\mathbf x}",
                 boxes[0][0]+boxes[0][2]/2, heads[0][1]+32, 30, 535 if compact else 490)
    d.math_label(f, "strain-model", r"U_{\rm ac}=D\,\partial_xu_{{\rm lat},x}",
                 boxes[0][0]+boxes[0][2]/2, heads[0][1]+85, 30, 500)
    X, Y = _axes(d, f, boxes[0], (0, 2*pi), (-1.12, 1.12),
                 ((0, "0"), (pi, "π"), (2*pi, "2π")), (-1, 0, 1), r"qx", "")
    f.path([(X(t), Y(cos(t))) for t in d.points(0, 2*pi)], d.BLUE)
    f.path([(X(t), Y(-sin(t))) for t in d.points(0, 2*pi)], d.TEAL)
    legend_top = 738 if compact else 747
    _legend(d, f, ((r"u_{{\rm lat},x}/u_0=\cos(qx)", d.BLUE, False),
                  (r"U_{\rm ac}/(Dqu_0)=-\sin(qx)", d.TEAL, False)),
            35 if compact else 75, legend_top, 525 if compact else 510, 55)
    f.text(*heads[1], "(b) Absorption and emission", 29, weight=500)
    d.math_label(f, "bose-model", r"N_q=\frac{1}{e^z-1}", boxes[1][0]+boxes[1][2]/2,
                 heads[1][1]+35, 30, 445)
    X, Y = _axes(d, f, boxes[1], (.2, 4), (0, 5.8), (.2, 1, 2, 4), (0, 1, 3, 5),
                 r"z=\hbar\omega_q/(k_BT)", r"\text{Occupation factor}")
    for fun, color, dashed in ((lambda z: 1/expm1(z), d.BLUE, False),
                               (lambda z: 1+1/expm1(z), d.GOLD, False),
                               (lambda z: 1/z, d.MUTED, True)):
        f.path([(X(z), Y(fun(z))) for z in d.points(.2, 4)], color, dash=dashed)
    _legend(d, f, ((r"N_q\quad\text{absorption}", d.BLUE, False),
                  (r"N_q+1\quad\text{emission}", d.GOLD, False),
                  (r"1/z\quad(z\ll1)", d.MUTED, True)),
            35 if compact else 675, 1425 if compact else 747,
            525 if compact else 480, 55)
    f.save("fig-sec17-acoustic-phonons"+ ("-mobile" if compact else ""))


def particle_weight(y):
    return sqrt(y)*exp(-y)/gamma(1.5) if y > 0 else 0.0


def transport_weight(y):
    return y**1.5*exp(-y)/gamma(2.5) if y > 0 else 0.0


def averaged_times():
    """ps, for the rounded reference inputs explicitly quoted in section 18."""
    kbt_mev = 1.380649e-23*300/1.602176634e-19*1000
    prefactor = .670864*sqrt(50/kbt_mev)
    return prefactor/gamma(2.5), prefactor/gamma(1.5)


def energy_average(d, compact=False):
    f = d.Figure(600 if compact else 1200, 1570 if compact else 950,
                 "Mobility uses a transport-weighted time rather than a particle average",
                 "Normalized Maxwell-Boltzmann particle and transport energy densities "
                 "are sqrt(y)exp(-y)/Gamma(3/2) and y^(3/2)exp(-y)/Gamma(5/2), "
                 "where y=K/(kBT); each integrates to one over zero to infinity. "
                 "For tau(K)=0.670864 ps sqrt(50 meV/K) at 300 K, the transport average "
                 "is approximately 0.702 ps and the particle average 1.053 ps, a ratio 1.5.")
    d.header(f, "Mobility uses a different energy weight from particle counting",
             ("Mobility uses a", "transport-weighted average"), compact)
    if compact:
        f.text(30, 143, "3D parabolic band", 25, d.MUTED)
        f.text(30, 181, "Maxwell–Boltzmann occupations", 25, d.MUTED)
    else:
        f.text(35, 106, "3D parabolic band · Maxwell–Boltzmann occupations", 27, d.MUTED)
    if compact:
        f.text(30, 217, "(a) Normalized energy weights", 29, weight=500)
        weight_box = (95, 410, 445, 260)
        f.text(30, 938, "(b) The same time, two averages", 29, weight=500)
        bar_box = (110, 1100, 430, 275)
        formula_top, legend_top = 249, 789
    else:
        f.text(75, 182, "(a) Normalized energy weights", 29, weight=500)
        weight_box = (100, 390, 610, 290)
        f.text(820, 182, "(b) Time averages", 29, weight=500)
        bar_box = (860, 390, 290, 290)
        formula_top, legend_top = 220, 801
    d.math_label(f, "particle-weight", r"w_{\rm p}(y)=\frac{y^{1/2}e^{-y}}{\Gamma(3/2)}",
                 weight_box[0]+weight_box[2]/2, formula_top, 30, 520 if compact else 610)
    d.math_label(f, "transport-weight", r"w_{\rm tr}(y)=\frac{y^{3/2}e^{-y}}{\Gamma(5/2)}",
                 weight_box[0]+weight_box[2]/2, formula_top+76, 30, 520 if compact else 610)
    X, Y = _axes(d, f, weight_box, (0, 8), (0, .53), (0, 2, 4, 6, 8), (0, .2, .4),
                 r"y=K/(k_BT)", "")
    f.path([(X(y), Y(particle_weight(y))) for y in d.points(0, 8, 801)], d.BLUE)
    f.path([(X(y), Y(transport_weight(y))) for y in d.points(0, 8, 801)], d.GOLD)
    _legend(d, f, ((r"w_{\rm p}\quad\text{particle count}", d.BLUE, False),
                  (r"w_{\rm tr}\quad\text{current response}", d.GOLD, False)),
            35 if compact else 75, legend_top, 525 if compact else 610, 55)
    d.math_label(f, "time-model", r"\tau(K)=\tau_{\rm ref}\sqrt{K_{\rm ref}/K}",
                 bar_box[0]+bar_box[2]/2, 973 if compact else 225, 30, 530 if compact else 380)
    if not compact:
        d.math_label(f, "bar-temperature", r"T=300\,{\rm K}", 1005, 294, 30, 280)
    Xb, Yb = d.axes(f, bar_box, (0, 2), (0, 1.2), (), (0, .5, 1), "", "")
    d.math_label(f, "bar-time-units", r"\langle\tau\rangle\ [{\rm ps}]", bar_box[0]+130,
                 bar_box[1]-49, 30, 290)
    av = averaged_times()
    for index, (value, color, label) in enumerate(zip(av, (d.GOLD, d.BLUE), ("Transport", "Particle"))):
        cx = Xb(.5+index)
        bw = 78 if compact else 65
        f.rect(cx-bw/2, Yb(value), bw, Yb(0)-Yb(value), color, color)
        f.text(cx, Yb(value)-18, f"{value:.3f}", 29, color, "middle", weight=500)
        f.text(cx, bar_box[1]+bar_box[3]+43, label, 25, d.INK, "middle")
    if compact:
        d.math_label(f, "bar-conditions", r"T=300\,{\rm K},\quad K_{\rm ref}=50\,{\rm meV}", 300, 1450, 29, 540)
        d.math_label(f, "bar-reference-time", r"\tau_{\rm ref}=0.670864\,{\rm ps}", 300, 1510, 29, 530)
    else:
        d.math_label(f, "bar-conditions", r"K_{\rm ref}=50\,{\rm meV},\quad\tau_{\rm ref}=0.670864\,{\rm ps}", 870, 850, 29, 580)
    f.save("fig-sec18-energy-average"+ ("-mobile" if compact else ""))


def exercise_input(d, compact=False):
    f = d.Figure(600 if compact else 1200, 1300 if compact else 800,
                 "Angular-bin input for the event-rate and transport-rate coding exercise",
                 "The incoming direction is positive x. Bin zero is forward, cos(theta0)=1, "
                 "and bin one is perpendicular, cos(theta1)=0. The supplied rate weights "
                 "are respectively 4 and 6 inverse seconds, already integrated over "
                 "their angular bins. The bars show these inputs, not the summed outputs. "
                 "The event rate sums w_i; the transport rate sums w_i[1-cos(thetai)].")
    d.header(f, "The coding input is a rate assigned to each outgoing bin",
             ("Coding input:", "directions and rate weights"), compact)
    f.text(30 if compact else 35, 143 if compact else 110,
           "Reference incoming direction: +x", 27, d.MUTED)
    if compact:
        f.text(30, 220, "(a) Two outgoing directions", 29, weight=500)
        ox, oy, length = 155, 590, 300
        f.text(30, 782, "(b) Rates already include bin measure", 27, weight=500)
        bar_box = (100, 904, 440, 245)
    else:
        f.text(55, 184, "(a) Two outgoing directions", 29, weight=500)
        ox, oy, length = 170, 520, 285
        f.text(690, 184, "(b) Supplied rate weights", 29, weight=500)
        bar_box = (760, 320, 375, 290)
    f.line(ox-30, oy, ox+length+30, oy, d.GRID, 1.7)
    f.line(ox, oy+20, ox, oy-length-30, d.GRID, 1.7)
    f.line(ox, oy, ox+length, oy, d.BLUE, 4.5, arrow=True)
    f.line(ox, oy, ox, oy-length, d.GOLD, 4.5, arrow=True)
    d.math_label(f, "forward-bin", r"i=0:\quad\cos\theta_0=1", ox+length-105, oy+45, 30, 370)
    d.math_label(f, "side-bin", r"i=1:\quad\cos\theta_1=0", ox+170, oy-length+28, 30, 370)
    d.math_label(f, "kx-axis", r"k_x^{\prime}", ox+length+36, oy-10, 29, 65)
    d.math_label(f, "ky-axis", r"k_y^{\prime}", ox-45, oy-length+10, 29, 65)
    f.path([(ox+42*cos(t), oy-42*sin(t)) for t in d.points(0, pi/2, 41)], d.MUTED, 1.8)
    if compact:
        f.text(30, 714, "Directions in wavevector space", 25, d.MUTED)
    else:
        f.text(55, 657, "Directions in wavevector space", 25, d.MUTED)
    X, Y = d.axes(f, bar_box, (0, 2), (0, 7.5), (), (0, 2, 4, 6), "", "")
    d.math_label(f, "rate-units", r"w_i\ [{\rm s}^{-1}]", bar_box[0]+130, bar_box[1]-49, 30, 275)
    for index, (value, color) in enumerate(((4, d.BLUE), (6, d.GOLD))):
        cx = X(.5+index)
        f.rect(cx-45, Y(value), 90, Y(0)-Y(value), color, color)
        f.text(cx, Y(value)-20, str(value), 31, color, "middle", weight=500)
        d.math_label(f, f"bar-bin-{index}", rf"i={index}", cx, bar_box[1]+bar_box[3]+32, 30, 100)
    top = 1230 if compact else 737
    d.math_label(f, "requested-sums", r"\textstyle\Gamma_q=\sum_iw_i,\qquad\Gamma_{\rm tr}=\sum_iw_i(1-\cos\theta_i)",
                 f.width/2, top, 30, f.width-65)
    f.save("fig-sec20-exercise-input"+ ("-mobile" if compact else ""))


def _simpson(fun, start, stop, n=4000):
    assert n % 2 == 0
    step = (stop-start)/n
    return step/3*(fun(start)+fun(stop)+sum((4 if i % 2 else 2)*fun(start+i*step) for i in range(1, n)))


def verify():
    """Independent conservation, scaling, Bose and normalized-weight checks."""
    for theta in (0, pi/18, pi/3, pi/2, pi):
        outgoing = (cos(theta), sin(theta))
        assert abs(sum(v*v for v in outgoing)-1) < 1e-14
        assert 0 <= 1-outgoing[0] <= 2
    assert abs((1-cos(pi/3))-.5) < 1e-14
    assert abs(1/(1-cos(pi/18))-65.8230478219) < 1e-8
    for u in (.1, .5, 1, 2, 4):
        assert abs(sqrt(u)*(1/sqrt(u))-1) < 1e-14
    for z in (.2, .5, 1, 2, 4):
        occupation = 1/expm1(z)
        assert occupation > 0 and abs((occupation+1)-occupation-1) < 1e-14
        assert abs((occupation+1)/occupation-exp(z)) < 1e-12
    for phase in (0, .3, 1, pi, 2*pi):
        h = 1e-5
        assert abs((cos(phase+h)-cos(phase-h))/(2*h)+sin(phase)) < 1e-10
    # y=t² removes the square-root endpoint when integrating normalized weights.
    for weight in (particle_weight, transport_weight):
        area = _simpson(lambda t: 2*t*weight(t*t), 0, sqrt(50))
        assert abs(area-1) < 1e-10
    transport, particle = averaged_times()
    assert abs(particle/transport-1.5) < 1e-14
    assert abs(transport-.701837) < 1e-6 and abs(particle-1.052755) < 1e-6
    rates, cosines = (4, 6), (1, 0)
    assert sum(rates) == 10 and sum(w*(1-c) for w, c in zip(rates, cosines)) == 6
    return {"sections": (14, 15, 17, 18, 20), "models": "elastic projection, DOS/rate scaling, phonon strain/Bose factors, normalized thermal weights, angular input"}


FIGURES = {
    14: (angular_relaxation, "fig-sec14-angular-relaxation"),
    15: (short_range, "fig-sec15-short-range"),
    17: (acoustic_phonons, "fig-sec17-acoustic-phonons"),
    18: (energy_average, "fig-sec18-energy-average"),
    20: (exercise_input, "fig-sec20-exercise-input"),
}
