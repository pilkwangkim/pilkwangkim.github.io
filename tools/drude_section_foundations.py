"""Section-local Drude figures: charge balance, mean drift, mobility and Bloch phase.

The parent figure module is supplied as ``d`` so these renderers share the
publication palette, fixed-font TeX labels and output directory without a
circular import. No material data or stochastic samples enter these figures.
"""

from __future__ import annotations

from cmath import exp as cexp
from math import cos, exp, pi, sin


def charge_conservation(d, compact=False):
    """A signed boundary schematic, not a redundant charge-versus-time line."""
    f = d.Figure(600 if compact else 1200, 1110 if compact else 815,
                 "More conventional current enters than leaves, so electrons are lost",
                 "A fixed control volume receives conventional current I_left=2 microampere "
                 "from the left and sends I_right=1 microampere to the right. The blue arrows "
                 "both point right; the gold net-electron-flow arrows both point left. "
                 "Individual electrons can move in either direction; these arrows show the mean imbalance. Arrow "
                 "lengths indicate direction only, not magnitudes. The enclosed budget after "
                 "one nanosecond is Delta Q=+1 femtocoulomb and Delta N_e about minus 6242. "
                 "The fixed positive background does not change. Hence an increase of net "
                 "positive charge is a loss of electrons, not a source of new charge. Boundary "
                 "currents are prescribed; this is not a self-consistent electrostatic solution.")
    d.header(f, "Conventional current and net electron flow have opposite signs",
             ("Conventional current", "versus net electron flow"), compact)
    d.math_label(f, "charge-budget", r"\frac{dQ}{dt}=I_{\rm left}-I_{\rm right}",
                 f.width/2, 135 if compact else 108, 33, f.width-70)
    if compact:
        left, right, top, bottom = 145, 455, 340, 650
        current_y, electron_y = 390, 590
        ends = ((35, 125), (475, 565))
        for side, x, value in (("left", 80, 2), ("right", 520, 1)):
            d.math_label(f, "charge-name-"+side, rf"I_{{\rm {side}}}", x, 252, 31, 140)
            d.math_label(f, "charge-value-"+side, rf"{value}\,\mu\mathrm A", x, 294, 32, 140)
        f.text(300, 383, "Control volume", 27, d.MUTED, "middle")
        d.math_label(f, "charge-elapsed", r"\Delta t=1\,\mathrm{ns}", 300, 423, 31, 280)
        result_tops = (485, 550)
        legend_rows = ((715, d.BLUE, "Conventional current"), (768, d.GOLD, "Net electron flow"))
        footer = 817
    else:
        left, right, top, bottom = 430, 770, 250, 540
        current_y, electron_y = 335, 445
        ends = ((75, 405), (795, 1125))
        d.math_label(f, "charge-inflow", r"I_{\rm left}=2\,\mu\mathrm A", 240, 258, 32, 330)
        d.math_label(f, "charge-outflow", r"I_{\rm right}=1\,\mu\mathrm A", 960, 258, 32, 330)
        f.text(600, 296, "Control volume", 27, d.MUTED, "middle")
        d.math_label(f, "charge-elapsed", r"\Delta t=1\,\mathrm{ns}", 600, 331, 31, 300)
        result_tops = (395, 465)
        legend_rows = ((597, d.BLUE, "Conventional current"), (597, d.GOLD, "Net electron flow"))
        footer = 641
    # Draw the enclosure before its text; arrows stop short of both boundaries.
    f.nodes.insert(5, f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#eef5f6" stroke="{d.BLUE}" stroke-width="2"/>')
    for x0, x1 in ends:
        f.line(x0, current_y, x1, current_y, d.BLUE, 4.2, arrow=True)
        f.line(x1, electron_y, x0, electron_y, d.GOLD, 4.2, arrow=True)
    d.math_label(f, "charge-enclosed-q", r"\Delta Q=+1.00\,\mathrm{fC}",
                 f.width/2, result_tops[0], 32, 300)
    d.math_label(f, "charge-enclosed-electrons", r"\Delta N_e\simeq-6.24\times10^3",
                 f.width/2, result_tops[1], 30, 300)
    for index, (y, color, label) in enumerate(legend_rows):
        x = 55 if compact else (155 if index == 0 else 675)
        if index == 0:
            f.line(x, y-10, x+42, y-10, color, 4.2, arrow=True)
        else:
            f.line(x+42, y-10, x, y-10, color, 4.2, arrow=True)
        f.text(x+58, y, label, 27, color)
    f.line(30 if compact else 35, footer, f.width-(30 if compact else 35), footer)
    d.math_label(f, "charge-numeric-budget", r"2\,\mu\mathrm A-1\,\mu\mathrm A=+1\,\mu\mathrm A",
                 f.width/2, footer+22, 32, f.width-70)
    if compact:
        d.math_label(f, "charge-integrated-budget", r"\Delta Q=(I_{\rm left}-I_{\rm right})\Delta t", 300, 904, 31, 535)
        d.math_label(f, "charge-electron-definition", r"\Delta N_e=-\Delta Q/e,\quad e>0", 300, 962, 31, 535)
        f.text(30, 1034, "Fixed positive background · mean values", 25, d.MUTED)
        f.text(30, 1070, "Arrows show direction, not magnitude.", 24, d.MUTED)
    else:
        d.math_label(f, "charge-integrated-budget", r"\Delta Q=(I_{\rm left}-I_{\rm right})\Delta t,\qquad\Delta N_e=-\Delta Q/e,\quad e>0",
                     600, 709, 31, 1110)
        f.text(35, 784, "Fixed positive background · mean values · arrows show direction, not magnitude", 25, d.MUTED)
    f.save("fig-sec01-charge-conservation"+("-mobile" if compact else ""))


def mean_drift(d, compact=False):
    """One common finite reference time compares two different motion laws."""
    f = d.Figure(600 if compact else 1200, 1090 if compact else 740,
                 "A constant field gives linear growth without scattering and saturation with relaxation",
                 "Both models have the same fixed density, positive response mass m*=m0 and "
                 "constant field E0>0, with zero initial mean drift. Define one finite positive "
                 "reference time tau_ref and J_ref=n e^2 E0 tau_ref/m*. Plot s=t/tau_ref. "
                 "The collisionless current is J/J_ref=s. For complete randomization with "
                 "relaxation time tau=tau_ref, the Drude current is J/J_ref=1-exp(-s). "
                 "The reference time is only a plotting unit for the collisionless model, "
                 "not its scattering time. Both initial slopes are equal in physical time.")
    d.header(f, "Scattering changes indefinite growth into a steady mean current",
             ("Without scattering,", "the mean current keeps growing"), compact)
    if compact:
        d.math_label(f, "drift-reference", r"J_{\rm ref}=\frac{ne^2E_0\tau_{\rm ref}}{m^{\ast}}",
                     300, 136, 32, 535)
        d.math_label(f, "drift-conditions", r"E_0>0,\quad m^{\ast}=m_0,\quad s=t/\tau_{\rm ref}",
                     300, 230, 30, 535)
        box = (95, 355, 445, 375)
        axis_top, legend_y = 305, 872
    else:
        d.math_label(f, "drift-reference", r"J_{\rm ref}=\frac{ne^2E_0\tau_{\rm ref}}{m^{\ast}},\qquad E_0>0,\quad m^{\ast}=m_0",
                     600, 100, 32, 1110)
        box = (100, 260, 1045, 290)
        axis_top, legend_y = 213, 668
    d.math_label(f, "drift-current-axis", r"J/J_{\rm ref}", box[0]+86, axis_top, 31, 250)
    X, Y = d.axes(f, box, (0, 4), (0, 4.2), (0, 1, 2, 4), (0, 1, 2, 4), "", "")
    f.path([(X(s), Y(s)) for s in d.points(0, 4)], d.BLUE, dash=True)
    f.path([(X(s), Y(1-exp(-s))) for s in d.points(0, 4)], d.TEAL)
    d.math_label(f, "drift-time-axis", r"s=t/\tau_{\rm ref}", box[0]+box[2]/2,
                 box[1]+box[3]+66, 31, 350)
    if compact:
        rows = [(legend_y, d.BLUE, r"\text{No scattering}:\quad J/J_{\rm ref}=s"),
                (legend_y+66, d.TEAL, r"\tau=\tau_{\rm ref}:\quad J/J_{\rm ref}=1-e^{-s}")]
        for index, (top, color, formula) in enumerate(rows):
            f.line(35, top+15, 70, top+15, color, 4.2, dash=index==0)
            d.math_label(f, f"drift-legend-{index}", formula, 324, top, 29, 470)
        f.text(30, 1030, "Same field, density and mass", 26, d.MUTED)
        f.text(30, 1070, "One finite reference time for both axes", 25, d.MUTED)
    else:
        f.line(55, legend_y+15, 90, legend_y+15, d.BLUE, 4.2, dash=True)
        d.math_label(f, "drift-legend-free", r"\text{No scattering}:\ J/J_{\rm ref}=s", 320, legend_y, 30, 440)
        f.line(650, legend_y+15, 685, legend_y+15, d.TEAL, 4.2)
        d.math_label(f, "drift-legend-drude", r"\tau=\tau_{\rm ref}:\ J/J_{\rm ref}=1-e^{-s}", 930, legend_y, 30, 465)
        f.text(35, 728, "The collisionless model uses τref as a unit, not as a scattering time.", 25, d.MUTED)
    f.save("fig-sec03-mean-drift"+("-mobile" if compact else ""))


def mobility(d, compact=False):
    """Compare density and mobility through physical drift/current directions."""
    f = d.Figure(600 if compact else 1200, 1510 if compact else 970,
                 "Higher mobility changes drift; more electrons can change current without changing drift",
                 "Three different uniform channels have identical geometry, the same prescribed "
                 "positive field E0, and a matching fixed positive background N_D=n in each. "
                 "All reference parameters are positive. Define mu_ref=e tau_ref/m_ref, "
                 "u_ref=mu_ref E0>0 and J_ref=e n_ref u_ref. In the reference channel, "
                 "n=n_ref, mu=mu_ref, u=-u_ref and J=J_ref. Doubling mobility at fixed "
                 "density doubles both the leftward electron drift and the rightward current. "
                 "Doubling density at fixed mobility leaves drift unchanged and doubles current. "
                 "Arrow lengths are quantitative only when comparing the same quantity across "
                 "channels, never current against velocity. These are separate low-field models, "
                 "not time evolution or a claim that doping leaves microscopic scattering fixed.")
    d.header(f, "Mobility and density change current in different ways",
             ("Mobility and density:", "different ways to change current"), compact)
    if compact:
        d.math_label(f, "mobility-reference", r"\mu_{\rm ref}=e\tau_{\rm ref}/m_{\rm ref}", 300, 136, 31, 535)
        d.math_label(f, "mobility-drift-reference", r"u_{\rm ref}=\mu_{\rm ref}E_0>0", 300, 208, 31, 535)
        d.math_label(f, "mobility-current-reference", r"J_{\rm ref}=en_{\rm ref}u_{\rm ref}", 300, 262, 31, 535)
        f.text(30, 329, "All reference parameters are positive.", 25, d.MUTED)
        f.text(30, 369, "Matching positive background: N_D = n", 25, d.MUTED)
        f.text(30, 411, "Same geometry", 25, d.MUTED)
        f.line(390, 400, 555, 400, d.GOLD, 4.2, arrow=True)
        d.math_label(f, "mobility-field", r"E_0", 325, 385, 30, 100)
        positions = ((30, 438), (30, 724), (30, 1010))
        footer = 1300
    else:
        d.math_label(f, "mobility-reference", r"\mu_{\rm ref}=e\tau_{\rm ref}/m_{\rm ref}", 600, 108, 32, 1110)
        d.math_label(f, "mobility-reference-scales", r"u_{\rm ref}=\mu_{\rm ref}E_0>0,\qquad J_{\rm ref}=en_{\rm ref}u_{\rm ref}", 600, 177, 32, 1110)
        f.text(35, 269, "Same geometry and prescribed field", 29, d.MUTED)
        f.line(850, 258, 1120, 258, d.GOLD, 4.2, arrow=True)
        d.math_label(f, "mobility-field", r"E_0", 790, 240, 31, 100)
        f.text(35, 310, "Separate channels · matching positive background N_D = n · positive references", 25, d.MUTED)
        positions = ((35, 345), (425, 345), (815, 345))
        footer = 820
    cases = (("(a) Reference", 1, 1), ("(b) Higher mobility", 1, 2), ("(c) More carriers", 2, 1))
    for index, ((x, y), (title, density_ratio, mu_ratio)) in enumerate(zip(positions, cases)):
        w, h = (540, 265) if compact else (350, 450)
        f.rect(x, y, w, h, d.WHITE, d.GRID)
        f.text(x+20, y+37, title, 28, weight=500)
        n_label = ("" if density_ratio == 1 else "2")+r"n_{\rm ref}"
        mu_label = ("" if mu_ratio == 1 else "2")+r"\mu_{\rm ref}"
        if compact:
            d.math_label(f, f"mobility-conditions-{index}", rf"n={n_label},\quad\mu={mu_label}", x+w/2, y+62, 31, 495)
            channel = (x+35, y+119, 470, 128)
            velocity_y, current_y = y+156, y+216
            velocity_start, current_start, unit_length = x+380, x+140, 100
            value_x, label_x = x+450, x+65
        else:
            d.math_label(f, f"mobility-density-{index}", rf"n={n_label}", x+w/2, y+65, 31, 315)
            d.math_label(f, f"mobility-mu-{index}", rf"\mu={mu_label}", x+w/2, y+115, 31, 315)
            channel = (x+25, y+175, 300, 150)
            velocity_y, current_y = y+219, y+289
            velocity_start, current_start, unit_length = x+280, x+80, 70
            label_x = x+48
        f.rect(*channel, "#f4f8f8", d.GRID)
        d.math_label(f, f"mobility-u-symbol-{index}", r"u", label_x, velocity_y-16, 31, 50)
        d.math_label(f, f"mobility-j-symbol-{index}", r"J", label_x, current_y-18, 31, 50)
        f.line(velocity_start, velocity_y, velocity_start-unit_length*mu_ratio, velocity_y, d.TEAL, 4.3, arrow=True)
        f.line(current_start, current_y, current_start+unit_length*density_ratio*mu_ratio, current_y, d.BLUE, 4.3, arrow=True)
        u_value = "-"+("" if mu_ratio == 1 else "2")+r"u_{\rm ref}"
        j_value = ("" if density_ratio*mu_ratio == 1 else "2")+r"J_{\rm ref}"
        if compact:
            d.math_label(f, f"mobility-u-value-{index}", u_value, value_x, velocity_y-16, 30, 105)
            d.math_label(f, f"mobility-j-value-{index}", j_value, value_x, current_y-18, 30, 105)
        else:
            d.math_label(f, f"mobility-u-value-{index}", rf"u={u_value}", x+w/2, y+353, 32, 315)
            d.math_label(f, f"mobility-j-value-{index}", rf"J={j_value}", x+w/2, y+405, 32, 315)
    f.line(30 if compact else 35, footer, f.width-(30 if compact else 35), footer)
    if compact:
        d.math_label(f, "mobility-drift-law", r"u=-\mu E_0", 300, 1325, 33, 535)
        d.math_label(f, "mobility-current-law", r"J=-enu=en\mu E_0", 300, 1380, 33, 535)
        f.text(30, 1450, "Compare u arrows only with u arrows;", 25, d.MUTED)
        f.text(30, 1488, "compare J arrows only with J arrows.", 25, d.MUTED)
    else:
        d.math_label(f, "mobility-response-laws", r"u=-\mu E_0,\qquad J=-enu=en\mu E_0", 600, 849, 33, 1110)
        f.text(35, 940, "Arrow lengths compare the same quantity across channels: u with u, J with J.", 25, d.MUTED)
    f.save("fig-sec04-mobility"+("-mobile" if compact else ""))


def _phasor(f, d, cx, cy, phase, index, key):
    """Arrow in a labeled complex plane for 2*c_j, not a spatial velocity."""
    radius = 53
    f.nodes.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{d.GRID}" stroke-width="1.5"/>')
    f.line(cx-74, cy, cx+74, cy, d.GRID, 1.5)
    f.line(cx, cy+68, cx, cy-68, d.GRID, 1.5)
    f.text(cx+81, cy+7, "Re", 24, d.MUTED)
    f.text(cx-10, cy-78, "Im", 24, d.MUTED)
    f.line(cx, cy, cx+(radius-2)*cos(phase), cy-(radius-2)*sin(phase), d.TEAL, 3.3, arrow=True)
    d.math_label(f, f"{key}-site-{index}", rf"j={index}", cx, cy+88, 29, 110)


def bloch_states(d, compact=False):
    """Four-site eigenstates distinguish translation phase from probability."""
    f = d.Figure(600 if compact else 1200, 1410 if compact else 880,
                 "Bloch phases differ while the site probabilities remain equal",
                 "Two normalized eigenvectors of a periodic four-site nearest-neighbor "
                 "tight-binding chain have coefficients c_j=exp(i k j a)/2. The site labels "
                 "are j=0,1,2,3. Each small coordinate frame is a complex plane, with the "
                 "real axis right and imaginary axis up. Its arrow is the complex number "
                 "2 c_j, not a position, momentum or particle trajectory. For ka=0 all "
                 "arrows point right and the energy is zero. For ka=pi/2 the arrows point "
                 "right, up, left and down and the energy is 2t_h. Every site has probability "
                 "one quarter in both states. The diagonal energy is epsilon0=2t_h.")
    d.header(f, "Bloch phase changes without changing the site probabilities",
             ("Bloch phases and", "equal site probabilities"), compact)
    d.math_label(f, "bloch-coefficients", r"N=4,\qquad c_j=\frac{e^{ikja}}{2},\qquad\epsilon_0=2t_h",
                 f.width/2, 137 if compact else 101, 31, f.width-70)
    if compact:
        centers = (((165, 390), (420, 390), (165, 610), (420, 610)),
                   ((165, 944), (420, 944), (165, 1164), (420, 1164)))
        d.math_label(f, "bloch-row-a", r"\text{(a)}\quad ka=0,\quad\epsilon=0", 300, 240, 30, 535)
        d.math_label(f, "bloch-row-b", r"\text{(b)}\quad ka=\pi/2,\quad\epsilon=2t_h", 300, 795, 30, 535)
        footer = 1300
    else:
        row_heads = (229, 536)
        centers = (tuple((x, 361) for x in (165, 450, 735, 1020)),
                   tuple((x, 668) for x in (165, 450, 735, 1020)))
        f.text(45, row_heads[0], "(a) Same phase on every site", 28, weight=500)
        f.text(45, row_heads[1], "(b) Phase advances by π/2 per site", 28, weight=500)
        d.math_label(f, "bloch-row-a", r"ka=0,\quad\epsilon=0", 935, 203, 30, 400)
        d.math_label(f, "bloch-row-b", r"ka=\pi/2,\quad\epsilon=2t_h", 935, 510, 30, 400)
        footer = 810
    for row, ka in enumerate((0, pi/2)):
        for j, (cx, cy) in enumerate(centers[row]):
            _phasor(f, d, cx, cy, j*ka, j, f"bloch-row-{row}")
    f.line(30 if compact else 35, footer, f.width-(30 if compact else 35), footer)
    if compact:
        d.math_label(f, "bloch-probabilities", r"|c_j|^2=1/4\quad\text{for every site}", 300, footer+20, 30, 535)
        f.text(30, footer+84, "Arrows: complex coefficients 2c_j", 26, d.MUTED)
    else:
        d.math_label(f, "bloch-probabilities", r"|c_j|^2=1/4\quad\text{for every site in both states}", 425, footer+21, 30, 750)
        f.text(1165, footer+46, "Arrows: complex numbers 2c_j", 25, d.MUTED, "end")
    f.save("fig-sec07-bloch-states"+("-mobile" if compact else ""))


FIGURES = {
    1: (charge_conservation, "fig-sec01-charge-conservation"),
    3: (mean_drift, "fig-sec03-mean-drift"),
    4: (mobility, "fig-sec04-mobility"),
    7: (bloch_states, "fig-sec07-bloch-states"),
}


def verify():
    """Check signs, reference normalizations and four-site eigenvectors."""
    elementary_charge = 1.602176634e-19
    for t_ns in (0, .2, .5, 1):
        delta_q = (2e-6-1e-6)*t_ns*1e-9
        assert abs(delta_q/1e-15-t_ns) < 2e-15
        assert -delta_q/elementary_charge <= 0
    assert abs(1e-15/elementary_charge-6241.509074460763) < 1e-9
    for s in (0, .1, 1, 2, 4):
        j = 1-exp(-s)
        assert 0 <= j <= s
        assert abs(exp(-s)+j-1) < 1e-15
    for mass_ratio in (.5, 1, 2):
        for time_ratio in (0, .3, 1, 2):
            mass_ref, time_ref = .2*9.1093837139e-31, 100e-15
            mu_ref = elementary_charge*time_ref/mass_ref
            mu = elementary_charge*(time_ratio*time_ref)/(mass_ratio*mass_ref)
            assert abs(mu/mu_ref-time_ratio/mass_ratio) < 1e-14
    for density_ratio in (0, .1, 1, 2):
        n_ref = 1e23
        sigma = elementary_charge*(density_ratio*n_ref)*mu_ref
        sigma_ref = elementary_charge*n_ref*mu_ref
        assert abs(sigma/sigma_ref-density_ratio) < 1e-14
    # The repeated channels are separate charge-neutral low-field models.
    # Matching fixed positive backgrounds make a prescribed uniform E0
    # compatible with charge density zero in each, even when n differs.
    field, n_ref = 1000.0, 1e23
    u_ref, j_ref = mu_ref*field, elementary_charge*n_ref*mu_ref*field
    for density_ratio, mu_ratio, expected_u, expected_j in (
            (1, 1, -1, 1), (1, 2, -2, 2), (2, 1, -1, 2)):
        density, background = density_ratio*n_ref, density_ratio*n_ref
        drift = -mu_ratio*mu_ref*field
        current = -elementary_charge*density*drift
        assert elementary_charge*(background-density) == 0
        assert abs(drift/u_ref-expected_u) < 1e-14
        assert abs(current/j_ref-expected_j) < 1e-14
        assert drift < 0 < current and field > 0
    assert abs(mu_ref*1e4-879.410010772163) < .01
    for ka in (-pi, -pi/2, 0, pi/2):
        vector = [cexp(1j*ka*j)/2 for j in range(4)]
        energy = 2*(1-cos(ka))
        assert abs(sum(abs(c)**2 for c in vector)-1) < 1e-14
        assert all(abs(abs(c)**2-.25) < 1e-14 for c in vector)
        for j in range(4):
            hc = 2*vector[j]-vector[(j-1)%4]-vector[(j+1)%4]
            assert abs(hc-energy*vector[j]) < 2e-14
    return {"sections": sorted(FIGURES), "models": "signed control-volume budget, shared-reference drift, independent charge-neutral channel comparisons, four-site Bloch eigenvectors"}
