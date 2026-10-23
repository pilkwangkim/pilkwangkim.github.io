"""Rebuild the educational figures shared by the Quantum Circuit introduction.

No quantum circuit generation, simulation or optimization is performed.
The interference calculation is elementary linear algebra; the invsqrt stage
counts are static counts from the unchanged official baseline. Other diagrams
are explicitly labelled teaching examples or architecture illustrations.
Requires matplotlib. SVG diagrams are authored directly for precise labels.
"""
from pathlib import Path
from html import escape
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/img/posts/quantum-circuit-compilation'
OUT.mkdir(parents=True, exist_ok=True)
BG, INK, MUTED, GRID = '#fbfbf8', '#183044', '#526776', '#d6e0e5'
BLUE, TEAL, ORANGE = '#356e9c', '#248475', '#bf683a'
LIGHT_BLUE, LIGHT_TEAL, LIGHT_ORANGE = '#edf3f8', '#e9f5f1', '#fbefe5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 19,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED})


class Diagram:
    def __init__(self, title, subtitle, height=760):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#526776"/></marker></defs>',
                      f'<rect width="1200" height="{height}" rx="18" fill="{BG}"/>']
        self.text(40, 54, title, 31, weight='700')
        self.text(40, 91, subtitle, 20, color=MUTED)

    def text(self, x, y, value, size=23, color=INK, anchor='start', weight='400'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def rect(self, x, y, w, h, fill='white', stroke=GRID):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def panel(self, x, y, w, h, title, lines, fill='white'):
        self.rect(x, y, w, h, fill)
        self.text(x+20, y+39, title, 25, weight='700')
        for i, line in enumerate(lines):
            self.text(x+20, y+82+34*i, line, 22)

    def arrow(self, x1, y1, x2, y2):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow)"/>')

    def line(self, x1, y1, x2, y2, color=GRID):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>')

    def save(self, filename):
        (OUT/filename).write_text('\n'.join(self.parts+['</svg>'])+'\n')


def interference():
    s=1/math.sqrt(2)
    vectors=[[1,0],[s,s],[s,-s],[0,1]]
    titles=['Start: |0⟩','After H: |+⟩','After Z: |−⟩','After H: |1⟩']
    fig, axes=plt.subplots(1,4,figsize=(12,6.3),sharey=True)
    fig.patch.set_facecolor(BG)
    for ax,v,title in zip(axes,vectors,titles):
        ax.set_facecolor(BG)
        ax.axhline(0,color=MUTED,lw=1)
        ax.bar([0,1],v,width=.53,color=[BLUE if a>=0 else ORANGE for a in v],zorder=3)
        for i,a in enumerate(v):
            label='0' if abs(a)<1e-12 else ('1' if a==1 else ('+1/√2' if a>0 else '−1/√2'))
            ax.text(i,a+(.07 if a>=0 else -.08),label,ha='center',va='bottom' if a>=0 else 'top',fontsize=17)
        ax.set_title(title,fontsize=20,pad=24)
        ax.set_xticks([0,1],['|0⟩','|1⟩'],fontsize=19)
        ax.set_ylim(-1.05,1.28);ax.set_xlim(-.65,1.65)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y',color=GRID,alpha=.6,zorder=0)
        probs='P(0)=1, P(1)=0' if v==[1,0] else ('P(0)=0, P(1)=1' if v==[0,1] else 'P(0)=P(1)=1/2')
        ax.text(.5,-.21,probs,transform=ax.transAxes,ha='center',fontsize=16)
    axes[0].set_ylabel('Real amplitude',fontsize=20)
    fig.suptitle('A relative sign becomes a measurable difference',fontsize=25,x=.52,y=.97)
    fig.text(.04,.87,'Teaching example: H → Z → H. The middle states have identical probabilities.',fontsize=16,color=MUTED)
    fig.subplots_adjust(left=.09,right=.985,top=.72,bottom=.25,wspace=.3)
    fig.savefig(OUT/'fig-01-interference.svg',facecolor=BG)
    plt.close(fig)


def uncomputation():
    d=Diagram('Keep the phase; erase the workspace',
              'Compute → apply the needed phase → reverse the temporary calculation',1010)
    d.panel(40,145,330,205,'1  Compute f(x)',['Input: |x⟩ unchanged','Scratch: |0⟩ → |f(x)⟩','Reversible arithmetic'],LIGHT_BLUE)
    d.panel(435,145,330,205,'2  Apply a phase',['Multiply amplitude by','exp(2π i f(x))','Bit values unchanged'],LIGHT_TEAL)
    d.panel(830,145,330,205,'3  Uncompute',['Scratch: |f(x)⟩ → |0⟩','Input: |x⟩ restored','The phase remains'],LIGHT_BLUE)
    d.arrow(376,248,426,248);d.arrow(771,248,821,248)
    d.text(40,411,'A temporary AND can be erased in two ways',26,weight='700')
    d.rect(40,447,535,420,LIGHT_ORANGE)
    d.rect(615,447,545,420,LIGHT_TEAL)
    d.text(60,487,'Reverse the AND',25,weight='700')
    d.text(635,487,'Measure, then correct the phase',25,weight='700')
    d.text(60,530,'Start with t = a AND b',22)
    d.text(635,530,'HMR returns a classical bit m',22)
    for label,y in [('a',590),('b',655),('t = ab',720)]:
        d.text(65,y+8,label,23);d.line(160,y,525,y,INK)
        d.text(640,y+8,label,23);d.line(745,y,1130,y,INK)
    d.line(350,590,350,720,INK)
    for y in [590,655]:d.parts.append(f'<circle cx="350" cy="{y}" r="7" fill="{INK}"/>')
    d.parts.append(f'<circle cx="350" cy="720" r="15" fill="{LIGHT_ORANGE}" stroke="{INK}" stroke-width="2"/>')
    d.line(340,720,360,720,INK);d.line(350,710,350,730,INK)
    d.text(350,560,'CCX',22,anchor='middle',weight='700');d.text(535,728,'0',23)
    d.rect(780,693,85,54,'white',TEAL);d.text(822,727,'HMR',21,anchor='middle',weight='700')
    d.text(1138,728,'0',23)
    d.line(1020,590,1020,655,INK)
    for y in [590,655]:d.parts.append(f'<circle cx="1020" cy="{y}" r="7" fill="{INK}"/>')
    d.parts.append(f'<path d="M822 747 V775 H1020 V663" fill="none" stroke="{TEAL}" stroke-width="2" stroke-dasharray="7 5"/>')
    d.text(1020,560,'CZ if m=1',21,anchor='middle',weight='700')
    d.text(900,804,'m',22,color=TEAL,anchor='middle')
    d.text(60,824,'Cleanup cost: one Toffoli',23,weight='700')
    d.text(635,842,'The sign (−1)^(mab) is cancelled.',21)
    d.text(40,921,'Dots are controls; ⊕ is a bit-flip target. The dashed line carries classical m.',22,color=MUTED)
    d.text(40,966,'MBU saves this AND-cleanup Toffoli. Arbitrary arithmetic is not free to erase.',22,weight='700')
    d.save('fig-02-uncomputation.svg')


def invsqrt_pipeline():
    d=Diagram('invsqrt: compute a phase word, use it, then uncompute',
              'Ideal scale trace for x=64; the real baseline also has coefficient and truncation errors.',820)
    d.panel(40,145,330,215,'1  Normalize',['x=64; shift count s=7','x_norm = 64 × 2^14 = 2^20','M = x_norm / 2^20 = 1'],LIGHT_BLUE)
    d.panel(435,145,330,215,'2  Initial approximation',['Select one of four intervals','Cubic polynomial via Horner','y₀ ≈ 1/√M'],LIGHT_BLUE)
    d.panel(830,145,330,215,'3  Newton correction',['p ≈ y₀ (3 − M y₀²)','The factor 1/2 is absorbed','into the output scale'],LIGHT_BLUE)
    d.arrow(376,250,426,250);d.arrow(771,250,821,250)
    d.arrow(995,369,995,442)
    d.panel(830,450,330,215,'4  Restore the scale',['Undo normalization by 2^s','Ideal phase target: Y=2^37','Read Y / 2^40 = 1/8'],LIGHT_TEAL)
    d.panel(435,450,330,215,'5  Phase from the bits',['Bit j adds 2^(j−40) turns','For Y=2^37: 1/8 turn','Z_POW argument = 1/4'],LIGHT_TEAL)
    d.panel(40,450,330,215,'6  Uncompute',['Reverse steps 4, 3, 2, 1','Input x=64 is restored','All scratch returns to zero'],LIGHT_BLUE)
    d.arrow(821,555,771,555);d.arrow(426,555,376,555)
    d.text(40,734,'Output: exp(2π i /8) |64⟩ |0…0⟩. No inverse-square-root value register remains.',23,weight='700')
    d.text(40,779,'Z_POW(q,t) uses exp(iπt): its argument is twice the number of turns.',21,color=MUTED)
    d.save('fig-03-invsqrt-pipeline.svg')


def invsqrt_cost():
    labels=['Normalize','Interval selection','Horner (3 multiplies)','Square y₀','M × y₀²','3 − z','Final multiply','Denormalize']
    forward=[120,3,1071,442,1043,28,867,300]
    reverse=[110,0,1071,442,1043,28,867,300]
    fig,ax=plt.subplots(figsize=(12,8))
    fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    y=list(range(len(labels)))
    ax.barh(y,forward,color=BLUE,label='Compute')
    ax.barh(y,reverse,left=forward,color=TEAL,label='Uncompute')
    for i,(a,b) in enumerate(zip(forward,reverse)):
        ax.text(a+b+24,i,f'{a+b:,}',va='center',fontsize=18)
    ax.set_yticks(y,labels,fontsize=18);ax.invert_yaxis();ax.set_xlim(0,2420)
    ax.set_xlabel('Toffoli-like gates per full baseline evaluation',fontsize=19)
    ax.grid(axis='x',color=GRID,alpha=.6);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=12)
    ax.legend(loc='lower right',frameon=False,fontsize=18)
    fig.suptitle('Where the unchanged invsqrt baseline spends 7,735 Toffolis',fontsize=24,y=.965)
    fig.text(.035,.9,'Static gate counts. Arithmetic uses 6,902 (~89%); rotation synthesis is additional.',fontsize=17,color=MUTED)
    fig.subplots_adjust(left=.29,right=.965,bottom=.14,top=.85)
    fig.savefig(OUT/'fig-04-invsqrt-cost.svg',facecolor=BG);plt.close(fig)
    (OUT/'invsqrt-stage-counts.json').write_text(json.dumps({'evidence':'Static counts of the official original invsqrt baseline, not a candidate experiment.','stages':labels,'forward':forward,'reverse':reverse,'total':sum(forward)+sum(reverse)},indent=2)+'\n')


def dqi():
    d=Diagram('DQI: recover a hidden error only to fix its phase',
              'Teaching example: GF(4), RS(3,1), one symbol error. Not a competition-sized run.',815)
    d.panel(40,145,330,235,'1  Hidden error',['α² = α + 1; α³ = 1','e = (0, α², 0)','Only position 1 is nonzero','e₁ = α² = binary 11'],LIGHT_ORANGE)
    d.panel(435,145,330,235,'2  Received syndromes',['S₀ = α² × α = 1','S₁ = α² × α² = α','Circuit receives (1, α),','not the hidden vector e'],LIGHT_BLUE)
    d.panel(830,145,330,235,'3  Decode',['S₁ / S₀ = α^j = α','Therefore j = 1','e₁ = S₀ / α = α²','For this one-error toy'],LIGHT_BLUE)
    d.arrow(376,260,426,260);d.arrow(771,260,821,260)
    d.panel(40,445,330,205,'4  Classical mask',['M₁ = α = binary 10','11 AND 10 = 10','popcount = 1 (odd)'],LIGHT_ORANGE)
    d.panel(435,445,330,205,'5  Apply the sign',['Required sign: (−1)^1','Multiply amplitude by −1','Syndromes stay unchanged'],LIGHT_TEAL)
    d.panel(830,445,330,205,'6  Clean up',['Restore all scratch to 0','Output: −|1, α⟩ |0…0⟩','No error array is output'],LIGHT_TEAL)
    d.arrow(376,548,426,548);d.arrow(771,548,821,548)
    d.text(40,714,'Actual baseline: synchronized EEA → Chien search → Forney phase → reverse EEA',23,weight='700')
    d.text(40,762,'Competition: GF(2¹²), 4,095 symbols, 70 syndromes, up to 35 nonzero error symbols.',21,color=MUTED)
    d.save('fig-05-dqi-decoding.svg')


def chemistry():
    d=Diagram('Chemistry: compress the data and price the whole block encoding',
              'Architecture illustration. The submitted ANGLE circuits are only one part of the resource model.',850)
    d.panel(40,140,520,185,'Hamiltonian H̃',['An energy operator, generally not unitary','So it cannot simply be used as one gate','Coefficients describe electron interactions'],LIGHT_ORANGE)
    d.arrow(571,232,625,232)
    d.rect(640,140,520,185,LIGHT_BLUE)
    d.text(660,180,'Embed it inside a larger unitary U',25,weight='700')
    d.rect(690,198,160,100,LIGHT_TEAL,TEAL)
    d.rect(850,198,95,100,'white')
    d.text(770,258,'H̃ / α',28,anchor='middle',weight='700')
    d.text(898,257,'…',30,anchor='middle')
    d.text(975,242,'Ancilla-zero',20);d.text(975,271,'block',20)
    d.panel(40,400,330,235,'Classical design',['Reference electron integrals','DFTHC vectors + weights','Givens tree + angle precision','Changes accuracy and λ_eff'],LIGHT_ORANGE)
    d.panel(435,400,330,235,'Submitted ANGLE QROM',['Read angle words coherently','Stream batches using XOR','Test addresses + cleanup','≤128 qubits per circuit'],LIGHT_TEAL)
    d.panel(830,400,330,235,'Additional priced work',['Outer + inner state preparation','Givens rotations and reverses','Persistent + shared workspace','Analytical resource formulas'],LIGHT_BLUE)
    d.arrow(376,520,426,520)
    d.line(205,635,205,668,MUTED);d.line(205,668,995,668,MUTED);d.arrow(995,668,995,643)
    d.text(40,707,'Full cost = λ_eff × T_total × Q_total',29,weight='700')
    d.text(40,750,'The original README reports: angle circuit Q=119; whole-block Q_total=730.',23)
    d.text(40,796,'Those are different resources. The 128-qubit cap does not cap the whole block.',21,color=MUTED)
    d.save('fig-06-chemistry-block-encoding.svg')


def qrom_givens():
    d=Diagram('Lookup the angles; rotate pairs of components',
              'Two small teaching examples. Neither panel is a FeMoCo numerical result.',880)
    d.panel(40,145,320,220,'A  Classical table',['Address a     word dₐ','0                     01','1                     10','Fixed before compilation'],LIGHT_ORANGE)
    d.rect(410,145,750,220,LIGHT_TEAL)
    d.text(435,186,'Coherent XOR lookup: |a⟩|z⟩ → |a⟩|z XOR dₐ⟩',25,weight='700')
    d.text(435,238,'Input:    (|0⟩ + |1⟩) |00⟩ / √2',25)
    d.text(435,285,'Output: (|0⟩|01⟩ + |1⟩|10⟩) / √2',25)
    d.text(435,333,'No address is measured; applying the lookup again clears the words.',20)
    d.arrow(368,250,401,250)
    d.text(40,427,'B  Two Givens rotations build a three-component unit vector',26,weight='700')
    d.panel(40,470,330,200,'Start',['u = (1, 0, 0)','Squared length = 1','Only the first component is set'],LIGHT_BLUE)
    d.panel(435,470,330,200,'Rotate (0,1) by π/4',['u = (1/√2, 1/√2, 0)','Mix the first two components','Preserve the vector length'],LIGHT_BLUE)
    d.panel(830,470,330,200,'Rotate (0,2) by π/4',['u = (1/2, 1/√2, 1/2)','1/4 + 1/2 + 1/4 = 1','Order and chosen pairs matter'],LIGHT_TEAL)
    d.arrow(376,570,426,570);d.arrow(771,570,821,570)
    d.text(40,737,'Competition scale: 76 components → 75 tree edges and angle words per vector.',23,weight='700')
    d.text(40,785,'Fewer angle bits save space but change the reconstructed coefficients.',23)
    d.text(40,831,'The verifier evaluates the angles actually returned by the circuit.',21,color=MUTED)
    d.save('fig-07-qrom-givens.svg')


if __name__=='__main__':
    interference();uncomputation();invsqrt_pipeline();invsqrt_cost();dqi();chemistry();qrom_givens()
    print('Wrote seven educational SVG figures; no quantum experiments were run.')
