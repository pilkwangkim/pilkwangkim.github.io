#!/usr/bin/env python3
"""Reproduce Example B's imposed-bias profile and first-cycle NDR figures.

The geometry, energy/voltage grids, Fermi level and eta match cells 42--51
of the cited notebook.  This targeted calculation does not execute or edit
the notebook.  A vectorized tridiagonal solve evaluates the same endpoint
Green-function element as the original scalar SciPy banded solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import check_negf_occupation as constants
from negf_example_a_figures import figure_style, save_figure


IMAGE_DIR = Path(__file__).resolve().parents[1] / "assets/img/posts/2026-07-22-negf-from-scratch"
N = 90
T0 = constants.T0
U0 = np.zeros(N)
U0[38:41] = U0[49:52] = 0.45
ETA = 1e-9
KT = 0.003
V_FOCUS = 0.20


def potential(voltage):
    return U0 - voltage * np.arange(N) / (N - 1)


def sigma(energies, lead_shift, eta=ETA):
    b = np.asarray(energies) + 1j * eta - (2 * T0 + lead_shift)
    root = np.sqrt(b*b - 4*T0*T0)
    first, second = (b - root) / 2, (b + root) / 2
    return np.where(first.imag < 0, first, second)


def transmission(energies, voltage, eta=ETA):
    """Solve A*x=e_(N-1) at all energies, with A's off-diagonals +t0."""
    energies = np.asarray(energies)
    left, right = sigma(energies, 0, eta), sigma(energies, -voltage, eta)
    diagonal = energies[None, :] + 1j * eta - (2 * T0 + potential(voltage)[:, None])
    diagonal[0] -= left
    diagonal[-1] -= right
    for j in range(1, N):
        diagonal[j] -= T0*T0 / diagonal[j - 1]
    endpoint = 1 / diagonal[-1]
    for j in range(N - 2, -1, -1):
        endpoint = -T0 * endpoint / diagonal[j]
    return (-2 * left.imag) * (-2 * right.imag) * abs(endpoint)**2


def local_maxima(values):
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1


def calculate():
    scan_energy = np.linspace(1e-3, 0.45, 3000)
    scan_t = transmission(scan_energy, 0)
    scan_peaks = local_maxima(scan_t)
    scan_peaks = scan_peaks[scan_t[scan_peaks] > 0.3]
    resonances = scan_energy[scan_peaks]
    ef = resonances[0] - 0.015
    map_voltage = np.linspace(0, 0.55, 111)
    map_energy = np.linspace(-0.05, 0.50, 1600)
    tmap = np.column_stack([transmission(map_energy, v) for v in map_voltage])
    energy_step = 5e-5
    energy_max = min(4*T0, 4*T0-map_voltage.max(), ef + 15*KT)
    current_energy = np.arange(0, energy_max + 0.5*energy_step, energy_step)
    current_voltage = np.unique(np.round(np.r_[map_voltage, np.arange(0.04, 0.2001, 0.001)], 12))

    def current_spectrum(v):
        fl = 1 / (1 + np.exp((current_energy-ef)/KT))
        fr = 1 / (1 + np.exp((current_energy-ef+v)/KT))
        return 1e9 * constants.CURRENT_PREFACTOR * transmission(current_energy, v) * (fl-fr)

    current = np.array([np.trapezoid(current_spectrum(v), current_energy) for v in current_voltage])
    conductance = np.gradient(current, current_voltage, edge_order=2)
    # The first resolved local maximum is the largest current in the focus
    # range; verify the original notebook's prominence condition as well.
    focus = current_voltage <= V_FOCUS
    peak_index = np.flatnonzero(focus)[np.argmax(current[focus])]
    minima = local_maxima(-current)
    valley_index = minima[minima > peak_index][0]
    assert current[peak_index] > 0.08 * current.max()
    vpeak, vvalley = current_voltage[[peak_index, valley_index]]
    ipeak, ivalley = current[[peak_index, valley_index]]
    peak_spectrum, valley_spectrum = current_spectrum(vpeak), current_spectrum(vvalley)
    peak_area = np.trapezoid(peak_spectrum, current_energy)
    valley_area = np.trapezoid(valley_spectrum, current_energy)
    first_ndr = (current_voltage >= vpeak) & (current_voltage <= vvalley) & (conductance < 0)
    higher_ndr = (current_voltage > V_FOCUS) & (conductance < 0)
    assert abs(peak_area-ipeak) < 1e-10
    assert abs(valley_area-ivalley) < 1e-10
    assert vpeak == 0.085 and vvalley == 0.149
    assert abs(ipeak-57.3122) < 5e-5
    assert abs(ipeak/ivalley-87.92) < 0.005
    data = dict(ef=ef, resonances=resonances, map_voltage=map_voltage, map_energy=map_energy,
                tmap=tmap, energy=current_energy, voltage=current_voltage, current=current,
                conductance=conductance, vpeak=vpeak, vvalley=vvalley, ipeak=ipeak, ivalley=ivalley,
                pvcr=ipeak/ivalley, peak_spectrum=peak_spectrum, valley_spectrum=valley_spectrum,
                peak_area=peak_area, valley_area=valley_area, first_ndr=first_ndr, higher_ndr=higher_ndr)
    dense_energy, dense_voltage = 0.09, 0.0
    sl, sr = sigma(np.array([dense_energy]), 0)[0], sigma(np.array([dense_energy]), -dense_voltage)[0]
    matrix = np.diag(dense_energy + 1j*ETA - (2*T0+potential(dense_voltage)))
    matrix += np.diag(np.full(N-1,T0),1) + np.diag(np.full(N-1,T0),-1)
    matrix[0,0] -= sl
    matrix[-1,-1] -= sr
    rhs = np.zeros(N,complex)
    rhs[-1] = 1
    dense_t = (-2*sl.imag)*(-2*sr.imag)*abs(np.linalg.solve(matrix,rhs)[0])**2
    fast_t = transmission(np.array([dense_energy]),dense_voltage)[0]
    assert abs(fast_t-dense_t) < 1e-12
    report = {"pass": True, "model": {"sites": N, "barriers": [[38,41],[49,52]],
               "well": [41,49], "barrier_height_ev": 0.45, "eta_ev": ETA, "kT_ev": KT,
               "fermi_energy_ev": float(ef), "zero_bias_resonances_ev": resonances.tolist()},
              "quadrature": {"energy_step_ev": energy_step, "energy_samples": len(current_energy),
                             "bias_samples": len(current_voltage), "energy_upper_ev": float(current_energy[-1])},
              "results": {"peak_voltage_v": float(vpeak), "valley_voltage_v": float(vvalley),
                          "peak_current_na": float(ipeak), "valley_current_na": float(ivalley),
                          "pvcr": float(ipeak/ivalley), "peak_spectral_area_na": float(peak_area),
                          "valley_spectral_area_na": float(valley_area), "dense_fast_abs_error": float(abs(fast_t-dense_t))},
              "layout": {"summary_panels": "a,b,c,d,e; e is the original full-sweep inset enlarged"}}
    return data, report


def plot_profile(destination, mobile=False):
    figure_style(mobile)
    fig, ax = plt.subplots(figsize=(5,6.4) if mobile else (10.8,4.7), layout="constrained")
    sites = np.arange(N)
    ax.step(sites,U0,where="mid",color="0.45",lw=2.4,label=r"$U_j(0)$")
    ax.step(sites,potential(0.15),where="mid",color="#1f4e79",lw=2.5,label=r"$U_j(0.15\ \mathrm{V})$")
    ax.scatter([0,N-1],[0,-0.15],color="#c0392b",s=55,zorder=4,clip_on=False,label="Contact band edges")
    ax.set_title("Imposed potential drop" if mobile else "Imposed potential drop across the double barrier",loc="left")
    ax.set_xlabel("Site $j$")
    ax.set_ylabel("Electron potential\nenergy $U_j$ (eV)")
    ax.set_xlim(-0.5,N-0.5)
    ax.set_ylim(-0.19,0.50)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(alpha=0.24)
    ax.legend(loc="upper left",bbox_to_anchor=(0,-0.27 if mobile else -0.23),ncol=1 if mobile else 3,
              frameon=False,borderaxespad=0,handlelength=1.6,columnspacing=1.3)
    save_figure(fig,destination)


def plot_summary(data,destination,mobile=False):
    figure_style(mobile)
    fig = plt.figure(figsize=(5,28) if mobile else (10.8,17.0),layout="constrained")
    if mobile:
        grid=fig.add_gridspec(5,1,height_ratios=[1,1,1.35,1.1,1])
        a,b,c,d,e=[fig.add_subplot(grid[i,0]) for i in range(5)]
    else:
        grid=fig.add_gridspec(3,2,height_ratios=[1,1.35,0.85])
        a,b,c,d=[fig.add_subplot(grid[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
        e=fig.add_subplot(grid[2,:])
    v,current,conductance=data["voltage"],data["current"],data["conductance"]
    vp,vv,ip,iv=data["vpeak"],data["vvalley"],data["ipeak"],data["ivalley"]
    a.plot(v,current,color="#1f4e79",lw=2.6)
    a.plot(vp,ip,"^",color="#c0392b",ms=9,label=f"Peak: {vp:.3f} V")
    a.plot(vv,iv,"v",color="#2e7d32",ms=9,label=f"Valley: {vv:.3f} V")
    a.axvspan(vp,vv,color="crimson",alpha=0.12,label="First NDR interval")
    a.text(.97,.08,f"PVCR = {data['pvcr']:.1f}",transform=a.transAxes,ha="right",fontsize=17,
           color="#b42318",bbox=dict(boxstyle="round,pad=.3",fc="white",ec="#c0392b",alpha=.95))
    a.set(xlim=(0,V_FOCUS),ylim=(0,1.28*ip),xlabel="Bias V (V)",ylabel="Current I (nA)")
    a.set_title("(a) First resonance cycle",loc="left")
    a.legend(loc="upper left",bbox_to_anchor=(0,-.27),frameon=False,
             borderaxespad=0,handlelength=1.1)

    b.plot(v,conductance,color="#6a3d9a",lw=2.4)
    b.axhline(0,color="0.5",lw=1)
    b.fill_between(v,conductance,0,where=data["first_ndr"],color="crimson",alpha=.25,label="First NDR interval")
    focus=conductance[v<=V_FOCUS]
    margin=.09*np.ptp(focus)
    b.set(xlim=(0,V_FOCUS),ylim=(focus.min()-margin,focus.max()+margin),
          xlabel="Bias V (V)",ylabel="$dI/dV$ (nA/V)")
    b.ticklabel_format(axis="y",style="sci",scilimits=(0,0),useMathText=True)
    b.set_title("(b) Differential conductance",loc="left")
    b.legend(loc="upper right",handlelength=1.1)

    im=c.pcolormesh(data["map_energy"],data["map_voltage"],data["tmap"].T,shading="auto",
                    cmap="viridis",vmin=0,vmax=1,rasterized=True)
    c.plot(np.full_like(data["map_voltage"],data["ef"]),data["map_voltage"],color="white",lw=1.8,label=r"$\mu_L=E_F$")
    c.plot(data["ef"]-data["map_voltage"],data["map_voltage"],color="orange",lw=1.8,label=r"$\mu_R=E_F-eV$")
    c.axvline(0,color="#ff4a4a",ls="--",lw=1.8,label="Emitter band edge")
    for value,color,label in [(vp,"#c0392b","Peak"),(vv,"#2e7d32","Valley")]:
        c.axhline(value,color=color,lw=1.8)
        c.text(.438,value+.003,label,color="white",ha="right",va="bottom",fontsize=16,
               bbox=dict(fc=color,ec="none",alpha=.9,pad=2))
    c.set(xlim=(-.05,.45),ylim=(0,V_FOCUS),xlabel="Energy E (eV)",ylabel="Bias V (V)")
    c.set_title("(c) Transmission and alignment",loc="left")
    cb=fig.colorbar(im,ax=c,orientation="horizontal" if mobile else "vertical",pad=.10 if mobile else .035,shrink=.94)
    cb.set_label("T(E,V)",fontsize=16)
    cb.ax.tick_params(labelsize=16)
    cb.locator=MaxNLocator(4)
    cb.update_ticks()
    c.legend(loc="upper left",bbox_to_anchor=(0,-.28 if mobile else -.20),
              facecolor="#3b3240",edgecolor="none",labelcolor="white",framealpha=1,
              borderaxespad=0,handlelength=1.3)

    energy=data["energy"]
    for spectrum,color,linestyle,area,label in [
        (data["peak_spectrum"],"#c0392b","-",data["peak_area"],"Peak"),
        (data["valley_spectrum"],"#2e7d32","--",data["valley_area"],"Valley")]:
        d.fill_between(energy,spectrum,0,color=color,alpha=.25)
        d.plot(energy,spectrum,color=color,ls=linestyle,lw=2.5,label=f"{label}: {area:.2f} nA")
    d.axvline(0,color="#ff4a4a",ls="--",lw=1.5,label="Emitter band edge")
    d.axvline(data["ef"],color="0.35",ls=":",lw=1.5,label=r"$\mu_L=E_F$")
    d.set(xlim=(-.003,energy[-1]),ylim=(0,1.13*data["peak_spectrum"].max()),
          xlabel="Energy E (eV)",ylabel="Spectral current\n$dI/dE$ (nA/eV)")
    d.set_title("(d) Spectral current areas",loc="left")
    d.legend(loc="upper right",handlelength=1.2)
    d.ticklabel_format(axis="y",style="sci",scilimits=(0,0),useMathText=True)

    e.plot(v,current,color="#1f4e79",lw=2.5)
    e.axvspan(0,V_FOCUS,color="gold",alpha=.15,label="First-cycle range")
    e.fill_between(v,current,0,where=data["higher_ndr"],color="crimson",alpha=.22,label="Higher-resonance NDR")
    high=np.flatnonzero(data["higher_ndr"])
    if high.size:
        steep=high[np.argmin(conductance[high])]
        e.annotate("Later NDR",xy=(v[steep],current[steep]),xycoords="data",
                   xytext=(.30,.68) if mobile else (.30,.75),textcoords="axes fraction",
                   fontsize=16,ha="center",arrowprops=dict(arrowstyle="->",lw=1.3,color="0.3"))
    e.set(xlim=(0,.55),ylim=(0,current.max()*1.15),xlabel="Bias V (V)",ylabel="Current I (nA)")
    e.set_title("(e) Full voltage sweep",loc="left")
    e.legend(loc="upper left",bbox_to_anchor=(0,-.27),frameon=False,
             ncol=1 if mobile else 2,handlelength=1.3,borderaxespad=0)
    for ax in (a,b,d,e):
        ax.grid(alpha=.24)
    for ax in (a,b,c,d,e):
        ax.xaxis.set_major_locator(MaxNLocator(4 if mobile else 5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    save_figure(fig,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=IMAGE_DIR)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    data,report=calculate()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for mobile in (False,True):
        suffix="-mobile" if mobile else ""
        plot_profile(args.output_dir/f"fig-11-example-b-bias-profile{suffix}.png",mobile)
        plot_summary(data,args.output_dir/f"fig-12-example-b-ndr-summary{suffix}.png",mobile)
    encoded=json.dumps(report,indent=2)+"\n"
    print(encoded,end="")
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(encoded)


if __name__=="__main__":
    main()
