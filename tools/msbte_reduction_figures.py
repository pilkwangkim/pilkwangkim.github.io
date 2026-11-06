#!/usr/bin/env python3
"""Exact BTE limits and density-matrix reduction figures for the MS-BTE article.

Only fig-06-bte-streaming and fig-07-reduction assets are generated. Existing
cover and body figures are not rewritten. All data are evaluated analytically;
the three-dimensional matrix example is algebraic, not a device band model.
"""

from __future__ import annotations

import argparse
from math import cos, exp, pi, sin, sqrt
from pathlib import Path
import shutil
import subprocess

from msbte_figures import (
    Figure, OUT, BLUE, TEAL, GOLD, RED, INK, MUTED, GRID, PAPER, WHITE, axes,
)


RATE = 0.4  # ps^-1, the reciprocal transition-rate coefficient
SPEED = 20.0  # nm/ps, one constant-velocity branch
SIGMA = 12.0  # nm


def streaming_occupation(x_nm, time_ps):
    return 0.1 + 0.8 * exp(-((x_nm - SPEED*time_ps + 30.0)/SIGMA)**2/2)


def relaxed_occupations(time_ps):
    deviation = 0.3 * exp(-2*RATE*time_ps)
    return 0.5 + deviation, 0.5 - deviation


def legend(f, items, x_positions, y, size=25):
    for (label, color), x in zip(items,x_positions):
        f.line(x,y-8,x+34,y-8,color,4)
        f.text(x+46,y,label,size,color)


def bte_streaming(mobile=False):
    width,height = (600,1330) if mobile else (1200,725)
    f=Figure(width,height,"Two separate solvable limits of the Boltzmann equation",
             "The left or upper panel solves collisionless streaming on one constant-velocity branch: f(x,t)=0.1+0.8 exp[-(x-20t+30)^2/(2 times 12^2)], with x in nanometres and t in picoseconds. Snapshots at zero, one, and two picoseconds translate without changing shape. The right or lower panel is a different, spatially homogeneous two-state problem: f-plus and f-minus equal 0.5 plus or minus 0.3 exp(-0.8t), with reciprocal transition coefficient 0.4 per picosecond. Their sum remains one. These are separate limits, not combined streaming and scattering in one simulated wire.")
    f.text(30,47,"Two solvable limits of the BTE",31,weight=600)
    if mobile:
        # Wrap the physical distinction instead of shrinking a long subtitle.
        f.text(30,88,"Exact solutions · independent examples",25,MUTED)
        boxes=((85,235,465,285),(85,875,465,285))
        heads=((30,133),(30,773))
    else:
        f.text(30,90,"Exact solutions · independent initial-value problems",26,MUTED)
        boxes=((85,250,465,285),(690,250,465,285))
        heads=((85,147),(690,147))
    for (x,y), title, equation in zip(heads,
            ("(a) Collisionless streaming","(b) Homogeneous relaxation"),
            ("∂f/∂t + v ∂f/∂x = 0","df⁺/dt = −r(f⁺ − f⁻)")):
        f.text(x,y,title,28 if mobile else 29,weight=600)
        f.text(x,y+43,equation,27,INK)
    X,Y=axes(f,boxes[0],(-100,100),(0,1),(-100,0,100),(0,.5,1),"x (nm)","Occupation f",25)
    xs=[-100+200*i/400 for i in range(401)]
    for t,color in zip((0,1,2),(BLUE,TEAL,GOLD)):
        f.path([(X(x),Y(streaming_occupation(x,t))) for x in xs],color,3.8)
    if mobile:
        legend(f,(("0 ps",BLUE),("1 ps",TEAL),("2 ps",GOLD)),(61,236,411),648)
        f.text(30,696,"v = 20 nm/ps · σ = 12 nm",26,MUTED)
        f.line(30,727,570,727,GRID,1.5)
    else:
        legend(f,(("0 ps",BLUE),("1 ps",TEAL),("2 ps",GOLD)),(70,235,400),650)
        f.text(85,700,"v = 20 nm/ps · σ = 12 nm",26,MUTED)
    X,Y=axes(f,boxes[1],(0,4),(0,1),(0,2,4),(0,.5,1),"t (ps)","Occupation f",25)
    ts=[4*i/200 for i in range(201)]
    f.line(X(0),Y(.5),X(4),Y(.5),MUTED,1.7,dash=True)
    for which,color in ((0,BLUE),(1,GOLD)):
        f.path([(X(t),Y(relaxed_occupations(t)[which])) for t in ts],color,3.8)
    f.text(X(2.1),Y(.74),"Equilibrium 0.5",24,MUTED)
    footer=1285 if mobile else 650
    if mobile:
        # Keep the legends outside the centered time-axis label at this baseline.
        legend(f,(("f⁺",BLUE),("f⁻",GOLD)),(85,420),footer-37)
        f.text(85,footer+8,"f⁺ + f⁻ = 1",27,TEAL)
        f.text(550,footer+8,"r = 0.4 ps⁻¹",25,MUTED,"end")
    else:
        legend(f,(("f⁺",BLUE),("f⁻",GOLD)),(690,865),footer)
        f.text(690,700,"f⁺ + f⁻ = 1",27,TEAL)
        f.text(1155,700,"r = 0.4 ps⁻¹",26,MUTED,"end")
    f.save("fig-06-bte-streaming-mobile.svg" if mobile else "fig-06-bte-streaming.svg")


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(a,b):
    return [[sum(x*y for x,y in zip(row,column)) for column in zip(*b)] for row in a]


def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def square_trace(matrix):
    return trace(matmul(matrix,matrix))


def reduction_data():
    c=[sqrt(.6),sqrt(.3),sqrt(.1)]
    rho=[[a*b for b in c] for a in c]
    co,si=cos(pi/6),sin(pi/6)
    rotation=[[co,-si,0.0],[si,co,0.0],[0.0,0.0,1.0]]
    rotated=matmul(matmul(transpose(rotation),rho),rotation)
    dephased=[[rotated[i][j] if i==j else 0.0 for j in range(3)] for i in range(3)]
    retained=[[dephased[i][j] if i<2 and j<2 else 0.0 for j in range(3)] for i in range(3)]
    return c,rotation,(rho,rotated,dephased,retained)


def entry_color(value):
    """Signed entry palette: negative red, zero white, positive teal."""
    target=(0,110,114) if value>=0 else (156,56,45)
    light=(250,252,251)
    q=min(1.0,abs(value))
    return "#"+"".join(f"{round(a+q*(b-a)):02x}" for a,b in zip(light,target))


def entry_label(value):
    if abs(value)<.0005:
        return "0"
    label=f"{value:.3f}"
    return label.replace("0.",".",1) if abs(value)<1 else label


def matrix_view(f,matrix,x,y,cell=70,truncate=False,size=25):
    for j in range(3):
        f.text(x+(j+.5)*cell,y-15,j+1,22,MUTED,"middle")
        f.text(x-14,y+(j+.5)*cell+8,j+1,22,MUTED,"end")
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):
            omitted=truncate and (i==2 or j==2)
            fill="#eef0f0" if omitted else entry_color(value)
            f.rect(x+j*cell,y+i*cell,cell,cell,fill,WHITE,width=2)
            f.text(x+(j+.5)*cell,y+(i+.5)*cell+size*.34,entry_label(value),size,
                   WHITE if abs(value)>.78 else MUTED if omitted else INK,"middle")
    f.rect(x,y,3*cell,3*cell,"none",GRID,width=1.5)
    if truncate:
        f.rect(x,y,2*cell,2*cell,"none",GOLD,width=3.2)


def matrix_colorbar(f,x,y,width=210,size=21):
    """Shared matrix-entry scale, not a probability-density colorbar."""
    for i in range(80):
        f.rect(x+i*width/80,y,width/80+.2,13,entry_color(i/79),"none",width=0)
    f.rect(x,y,width,13,"none",GRID,width=1)
    f.text(x-9,y+14,"0",size,MUTED,"end")
    f.text(x+width+9,y+14,"1",size,MUTED)


def reduction(mobile=False):
    c,rotation,matrices=reduction_data()
    width,height=(600,1540) if mobile else (1200,690)
    f=Figure(width,height,"Exact basis rotation, dephasing, and truncation are different operations",
             "An algebraic one-electron example starts from c=(sqrt(0.6),sqrt(0.3),sqrt(0.1)) and rho=c c-transpose. A real orthogonal pi-over-six rotation mixes basis vectors one and two. The transformed matrix U-transpose rho U still has trace one and purity one. Dephasing in the new basis keeps diagonal entries approximately 0.89242346,0.00757654,0.1 but removes all off-diagonal entries; trace stays one and purity becomes 0.80647704. Retaining only basis vectors one and two gives P D P with trace 0.9. It is not renormalized; the omitted trace is 0.1. All matrix entries in this example are nonnegative and share one zero-to-one color scale. Numbers are rounded to three decimals. The outlined two-by-two block is retained; gray cells are omitted. These basis vectors are not claimed to be a physical subband calculation.")
    f.text(30,46,"Basis change, dephasing, truncation",30 if mobile else 32,weight=600)
    f.text(30,88,"Basis vectors 1, 2, 3 · θ = π/6",25,MUTED)
    if not mobile:
        f.text(900,61,"Matrix-entry value",22,MUTED)
        matrix_colorbar(f,916,80)
    titles=("(a) Original state","(b) Exact basis rotation","(c) Diagonal approximation","(d) Two-state retention")
    formulas=("ρ = ccᵀ","M = UᵀρU","D = diag(M)","R = PDP")
    symbols=("ρ","M","D","R")
    notes=(("Pure state",),("Same quantum state",),("Off-diagonal entries", "set to zero"),("Keep vectors 1, 2", "No renormalization"))
    for index,matrix in enumerate(matrices):
        symbol=symbols[index]
        tr_label=f"Tr({symbol}) = {trace(matrix):.1f}"
        square_label=f"Tr({symbol}²) = 1" if index<2 else f"Tr({symbol}²) ≈ {square_trace(matrix):.4f}"
        if mobile:
            base=index*330
            f.text(30,140+base,titles[index],28,weight=600)
            matrix_view(f,matrix,70,194+base,cell=72,truncate=index==3,size=28)
            right=334
            # Keep formula and invariants outside the matrix's numeric cells.
            f.text(right,220+base,formulas[index],27,GOLD if index==3 else INK)
            f.text(right,275+base,tr_label,26,INK)
            f.text(right,317+base,square_label,24,INK)
            for line,note in enumerate(notes[index]):
                f.text(right,370+base+35*line,note,23,GOLD if index==3 else MUTED)
            if index<3:
                f.line(30,433+base,570,433+base,GRID,1.5)
        else:
            x=45+300*index
            f.text(x,145,("(a) Original state","(b) Basis rotation","(c) Diagonal only","(d) Retain two states")[index],24,weight=600)
            f.text(x,187,formulas[index],27,GOLD if index==3 else INK)
            matrix_view(f,matrix,x,241,cell=70,truncate=index==3,size=25)
            f.text(x,499,tr_label,25)
            f.text(x,540,square_label,23)
            f.text(x,590,("Pure state","Same state","Coherences removed","Omitted trace: 0.1")[index],23,GOLD if index==3 else MUTED)
    if mobile:
        f.text(30,1456,"Purity interpretation requires trace 1.",24,MUTED)
        f.text(30,1496,"Entry value",23,MUTED)
        matrix_colorbar(f,248,1482,width=250)
        f.text(30,1530,"Gray cells omitted; outlined block retained.",23,GOLD)
    else:
        f.line(30,620,1170,620,GRID,1.5)
        f.text(30,663,"Purity interpretation requires trace 1.",25,MUTED)
        f.text(1170,663,"Outlined block kept; gray cells omitted.",23,GOLD,"end")
    f.save("fig-07-reduction-mobile.svg" if mobile else "fig-07-reduction.svg")


def verify_data():
    for t in (0,1,2):
        centre=-30+SPEED*t
        assert abs(streaming_occupation(centre,t)-.9)<1e-14
        assert abs(streaming_occupation(centre+SIGMA,t)-(.1+.8*exp(-.5)))<1e-14
    for t in (0,.1,1,2,4):
        fp,fm=relaxed_occupations(t)
        assert abs(fp+fm-1)<1e-14 and 0<=fm<=fp<=1
        exact_derivative=-.6*RATE*exp(-2*RATE*t)
        assert abs(exact_derivative+RATE*(fp-fm))<1e-14
    c,rotation,(rho,rotated,dephased,retained)=reduction_data()
    identity=matmul(transpose(rotation),rotation)
    assert all(abs(identity[i][j]-(1 if i==j else 0))<1e-14 for i in range(3) for j in range(3))
    assert abs(sum(x*x for x in c)-1)<1e-14
    for matrix in (rho,rotated):
        assert abs(trace(matrix)-1)<1e-14
        assert abs(square_trace(matrix)-1)<1e-14
    assert abs(dephased[0][0]-.8924234614174767)<1e-14
    assert abs(dephased[1][1]-.00757653858252329)<1e-14
    assert abs(trace(dephased)-1)<1e-14
    assert abs(square_trace(dephased)-.8064770384252431)<1e-14
    assert abs(trace(retained)-.9)<1e-14
    assert abs(square_trace(retained)-.7964770384252431)<1e-14
    assert abs(trace(dephased)-trace(retained)-.1)<1e-14
    assert all(dephased[i][j]==0 for i in range(3) for j in range(3) if i!=j)
    assert entry_label(-.125)=="-.125"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png",action="store_true",help="Also rasterize the four new SVG files.")
    args=parser.parse_args()
    verify_data()
    for renderer in (bte_streaming,reduction):
        for compact in (False,True):
            renderer(compact)
    if args.png:
        raster=shutil.which("rsvg-convert")
        if not raster:
            raise SystemExit("--png requires rsvg-convert")
        for stem in ("fig-06-bte-streaming","fig-07-reduction"):
            for suffix in ("","-mobile"):
                source=OUT/(stem+suffix+".svg")
                subprocess.run([raster,str(source),"-o",str(source.with_suffix(".png"))],check=True)
                print(source.with_suffix(".png").name)
