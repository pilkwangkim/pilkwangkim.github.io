"""CASMI guide figures: open MassBank measurements and labeled calculations.

Run with RDKit 2026.03.3, matplotlib, numpy, and torch available.
The library counts are aggregate statistics from the local September 15 train
file, audited September 27, 2026. Structure counts are deliberately not plotted.
"""
from pathlib import Path
from io import BytesIO
import os
import json

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # Import before RDKit in the author's macOS environment.
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "assets/img/casmi-2026-guide"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, CORAL, GREY = "#263541", "#326e78", "#a26641", "#697680"
LIGHT, BORDER = "#f4f6f7", "#ced5da"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": NAVY, "text.color": NAVY, "axes.titleweight": "normal",
    "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "axes.linewidth": .8, "grid.color": BORDER,
    "savefig.facecolor": "white", "svg.fonttype": "none",
})


def save(fig, name):
    # The cover is an edited official image, managed separately from these figures.
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=.2)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=.2)
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def molecule_image(smiles, width=650, height=350):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().bondLineWidth = 3
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return plt.imread(BytesIO(drawer.GetDrawingText()), format="png")


EUGENOL = "COc1cc(CC=C)ccc1O"
ISOEUGENOL = "COc1cc(C=CC)ccc1O"
taut = rdMolStandardize.TautomerEnumerator()


def metric_key(s):
    return Chem.MolToInchiKey(taut.Canonicalize(Chem.MolFromSmiles(s)))[:14]


assert rdBase.rdkitVersion == "2026.03.3"
assert metric_key(EUGENOL) != metric_key(ISOEUGENOL)
assert metric_key("C/C=C/c1ccc(O)c(OC)c1") == metric_key("C/C=C\\c1ccc(O)c(OC)c1")
assert abs(Descriptors.ExactMolWt(Chem.MolFromSmiles(EUGENOL)) - 164.083729624) < 1e-8

# Two actual molecular graphs, generated from the SMILES used in the article.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
for ax, title, smi in zip(axes, ["Eugenol", "Isoeugenol"], [EUGENOL, ISOEUGENOL]):
    ax.imshow(molecule_image(smi))
    ax.axis("off")
    ax.set_title(title, fontsize=20, color=NAVY, pad=12)
    ax.text(.5, -.03, metric_key(smi), transform=ax.transAxes, ha="center", fontsize=12, color=GREY)
fig.suptitle("Same formula. Same exact mass. Different connectivity.", fontsize=18, y=1.02)
fig.text(.5, .015, "C10H12O2  |  164.083730 Da  |  RDKit 2026.03.3", ha="center", fontsize=13)
fig.subplots_adjust(bottom=.14, wspace=.1)
save(fig, "isomers")

# Eawag records, CC BY; exact attribution and provenance are in data/ and README.
records = json.loads((OUT / "data/caffeine-massbank.json").read_text())
assert [r["collision_energy_nominal_percent"] for r in records] == [30, 60, 75]
assert all(r["collision_energy_ev"] is None for r in records)
def peaks(record):
    return (np.array([p["mz"] for p in record["peaks"]]),
            np.array([p["intensity_normalized_basepeak_1"] for p in record["peaks"]]))
def measured_cosine(a, b, tolerance_ppm=10):
    xa, ya = peaks(a)
    xb, yb = peaks(b)
    matches = [(i, j) for i, x in enumerate(xa) for j, y in enumerate(xb)
               if abs(x-y) / ((x+y)/2) * 1e6 <= tolerance_ppm]
    # These particular records have unique matches; do not treat this as a general peak-matching solver.
    assert len({i for i, j in matches}) == len(matches) == len({j for i, j in matches})
    va, vb = np.sqrt(ya), np.sqrt(yb)
    return sum(va[i]*vb[j] for i,j in matches) / (np.linalg.norm(va)*np.linalg.norm(vb))

cosine_75_60 = measured_cosine(records[2], records[1])
cosine_75_30 = measured_cosine(records[2], records[0])
assert np.isclose(cosine_75_60, .9267618308311408)
assert np.isclose(cosine_75_30, .39246457921873074)
assert metric_key(records[0]["smiles"]) == metric_key("Cn1c(=O)c2c(ncn2C)n(C)c1=O") == "RYYVLZVUVIJVGH"
x, y = peaks(records[-1])
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.vlines(x, 0, y, color=TEAL, lw=2.7)
ax.set(xlim=(45, 207), ylim=(0, 1.24), xlabel="Mass-to-charge ratio (m/z)", ylabel="Relative intensity")
ax.set_title("Caffeine · measured HCD spectrum · EA030312", loc="left", fontsize=18, pad=18)
for mz, xytext, label in [(138.0662,(149,1.09),"138.0662 · base peak"),
                          (110.0713,(86,.60),"110.0713 · 0.2148"),
                          (195.0878,(175,.43),"195.0878 · 0.1056")]:
    i=np.argmin(abs(x-mz))
    ax.annotate(label, xy=(x[i],y[i]), xytext=xytext, ha="center", fontsize=11,
                arrowprops={"arrowstyle":"->","color":GREY})
ax.grid(axis="y", alpha=.2)
fig.text(.125,-.03,"[M+H]+ · LTQ Orbitrap XL · 75% nominal CE (not eV) · Eawag / MassBank, CC BY",fontsize=10,color=GREY)
save(fig,"spectrum-anatomy")

fig, ax = plt.subplots(figsize=(10, 4.8))
ranks = np.arange(1, 26)
ax.plot(ranks, 1 / ranks, color=TEAL, marker="o", ms=5, lw=2)
ax.set(xlim=(.5, 26), ylim=(0, 1.1), xlabel="Rank of the first correct candidate", ylabel="Reciprocal rank")
ax.set_xticks([1, 2, 5, 10, 15, 20, 25])
ax.set_title("A correct answer near the top matters much more", loc="left", fontsize=18, pad=18)
for r, delta in [(1, (1.8, 1.01)), (2, (3.1, .55)), (5, (6, .29)), (25, (20, .16))]:
    ax.annotate(f"#{r}: {1/r:g}", xy=(r, 1/r), xytext=delta, arrowprops={"arrowstyle": "-", "color": GREY}, color=NAVY)
ax.text(.98, .68, "Rank 2 → 1: +0.50\nMissing → rank 25: +0.04", transform=ax.transAxes, ha="right", fontsize=14, linespacing=1.7)
ax.grid(alpha=.15)
save(fig, "reciprocal-rank")

counts = {
    "Enveda-180": 1153785, "Pluskal MS2": 527581, "RIKEN": 347171,
    "GNPS": 220849, "MassBank": 101727, "MoNA": 92416,
    "Other libraries": 50933 + 40765 + 2545 + 1184 + 652,
}
assert sum(counts.values()) == 2539608
fig, ax = plt.subplots(figsize=(10, 5.7))
values = np.array(list(counts.values()))
bars = ax.barh(list(counts), values / 1e6, color=[TEAL] + ["#879bb1"] * 6, height=.65)
ax.invert_yaxis()
for b, v in zip(bars, values):
    ax.text(b.get_width() + .018, b.get_y() + b.get_height()/2, f"{v:,}  ({v/values.sum():.1%})", va="center", fontsize=11)
ax.set_xlim(0, 1.57)
ax.set_xlabel("Number of spectra (millions)")
ax.set_title("Training spectra are concentrated in a few sources", loc="left", fontsize=18, pad=18)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.text(.02, -.01, "Local train-file audit, 2026-09-27. Counts are spectra, not unique molecules. Total: 2,539,608.", fontsize=10, color=GREY)
save(fig, "training-sources")

# Analytic first-order kinetics, a constructed example rather than a fitted experiment.
fig, ax = plt.subplots(figsize=(10, 4.8))
t = np.linspace(0, .001, 301)
pop = np.exp(-4000*t)
for values,color,label in [(pop,GREY,"Precursor P"),(.75*(1-pop),TEAL,"Product A: kA = 3,000 s−1"),(.25*(1-pop),CORAL,"Product B: kB = 1,000 s−1")]:
    ax.plot(t*1e6,values,color=color,lw=2.5,label=label)
ax.axvline(100,color=GREY,lw=1,ls=":")
ax.set(xlabel="Reaction time (µs)",ylabel="Population fraction",xlim=(0,1000),ylim=(0,1.05))
ax.set_title("Formation depends on both rate and observation time",loc="left",fontsize=17,pad=18)
ax.legend(frameon=False,loc="center right",fontsize=11)
ax.text(170,.87,"100 µs: P = 0.6703, A = 0.2473, B = 0.0824",fontsize=12)
ax.grid(alpha=.15)
fig.text(.125,-.03,"Calculated: fixed rates, P(0)=1, no secondary reactions. Detected intensities can differ from populations.",fontsize=10,color=GREY)
save(fig,"fragmentation-pathways")

fig, axes = plt.subplots(3,1,figsize=(10,7.2),sharex=True)
for ax,record,color in zip(axes,records,[TEAL,"#4675a0",CORAL]):
    x,y=peaks(record)
    ax.vlines(x,0,y,color=color,lw=2.5)
    ax.set(ylim=(0,1.2),yticks=[0,.5,1],xlim=(45,207))
    ax.set_title(f'{record["accession"].split("-")[-1]} · {record["source_collision_energy"]} CE',loc="left",fontsize=13)
    for mz in [110.0713,138.0662,195.0877]:
        ax.axvline(mz,color=GREY,lw=.7,ls=":",alpha=.4)
axes[0].annotate("Precursor ≈195.0877",xy=(195.0877,1),xytext=(149,1.08),arrowprops={"arrowstyle":"->","color":GREY},fontsize=10)
axes[1].annotate("138.0662",xy=(138.0662,1),xytext=(153,1.08),fontsize=10,arrowprops={"arrowstyle":"->","color":GREY})
axes[2].annotate("110.0713",xy=(110.0713,.2148),xytext=(94,.64),fontsize=10,arrowprops={"arrowstyle":"->","color":GREY})
axes[-1].set_xlabel("Mass-to-charge ratio (m/z)")
fig.supylabel("Relative intensity (base peak = 1 in each record)",fontsize=12)
fig.suptitle("One measured molecule, three collision conditions",fontsize=18,y=1.02)
fig.subplots_adjust(hspace=.53)
fig.text(.12,.015,"Caffeine · HCD · LTQ Orbitrap XL · Eawag / MassBank, CC BY. Percent CE is not converted to eV.",fontsize=10,color=GREY)
save(fig,"collision-energy-series")


def diagram(title, height=6.4):
    fig, ax = plt.subplots(figsize=(9, height))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=18, pad=22)
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=TEAL):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=LIGHT, edgecolor=BORDER, lw=.9))
    ax.plot([x, x], [y, y + h], color=color, lw=3, solid_capstyle="butt")
    ax.text(x + .22, y + h - .24, title, va="top", fontsize=14, weight="normal")
    if body:
        ax.text(x + .22, y + .22, body, va="bottom", fontsize=12,
                color=GREY, linespacing=1.5)


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                lw=1.2, color=color, shrinkA=4, shrinkB=4))


# Differences between measured peaks do not prove a sequential reaction mechanism.
fig, ax = plt.subplots(figsize=(10,5.4))
ax.axis("off")
ax.set_title("Caffeine EA030312: measured gaps and formula balance",loc="left",fontsize=17,pad=20)
rows=[
 ["Observed peaks (m/z)","Difference (Da)","Neutral composition"],
 ["195.0878 − 138.0662","57.0216","C2H3NO: 57.021464"],
 ["138.0662 − 110.0713","27.9949","CO: 27.994915"],
]
table=ax.table(cellText=rows,cellLoc="left",colWidths=[.39,.25,.36],bbox=[0,.36,1,.6])
table.auto_set_font_size(False);table.set_fontsize(12)
for (row,col),cell in table.get_celld().items():
    cell.set_edgecolor(BORDER);cell.set_facecolor(LIGHT if row==0 else "white")
ax.text(0,.23,"Tentative ion formulas in the source:",fontsize=12,color=GREY)
ax.text(0,.13,"C8H11N4O2+       C6H8N3O+       C5H8N3+",fontsize=16,color=TEAL)
ax.text(0,-.02,"Mass agreement supports composition. It does not establish a reaction sequence or atom mapping.",fontsize=11,color=GREY)
save(fig,"ion-mass-balance")

fig, axes = plt.subplots(1,3,figsize=(11,6.5),gridspec_kw={"bottom":.42,"top":.83})
for ax,record in zip(axes,records):
    x,y=peaks(record);ax.vlines(x,0,y,color=TEAL,lw=2)
    ax.set(xlim=(45,210),ylim=(0,1.12),xticks=[60,110,160,200],yticks=[0,1],xlabel="m/z")
    ax.set_title(record["accession"].split("-")[-1]+"\n"+record["source_collision_energy"],fontsize=13)
axes[0].set_ylabel("Relative intensity")
fig.suptitle("Caffeine: three public measurements → one molecular answer",fontsize=17,y=.99)
fig.text(.125,.32,"Shared identity: RYYVLZVUVIJVGH  |  C8H10N4O2  |  [M+H]+",fontsize=14)
fig.text(.125,.24,"Actual inputs: precursor m/z 195.0877; 2, 9 and 11 reported peaks",fontsize=12)
fig.text(.125,.15,"Example output row (identity known here for teaching):",fontsize=11,color=GREY)
fig.text(.125,.09,"caffeine_demo, Cn1c(=O)c2c(ncn2C)n(C)c1=O",fontsize=13,color=TEAL)
fig.text(.125,.015,"Eawag / MassBank, CC BY. Local example ID, not CASMI data; nominal CE is not eV.",fontsize=10,color=GREY)
save(fig,"molecule-to-ranking")

fig, ax=plt.subplots(figsize=(10,4.8));ax.axis("off")
ax.set_title("Known answer: Eugenol. Change the returned candidate order.",loc="left",fontsize=16,pad=22)
rows=[["Candidate list","Correct rank","RR@25","Diagnosis"],
      ["Isoeugenol only","Absent","0","Pool misses answer"],
      ["Isoeugenol → Eugenol","2","0.5","Answer loses ranking"],
      ["Eugenol → Isoeugenol","1","1.0","Answer ranks first"]]
t=ax.table(cellText=rows,cellLoc="left",colWidths=[.4,.18,.14,.28],bbox=[0,.15,1,.85]);t.auto_set_font_size(False);t.set_fontsize(11)
for (r,c),cell in t.get_celld().items():
    cell.set_edgecolor(BORDER);cell.set_facecolor(LIGHT if r==0 else "white")
ax.text(0,.01,"Constructed lists of real structures; no model performance is being reported.",fontsize=11,color=GREY)
save(fig,"candidate-diagnosis")

fig, ax = plt.subplots(figsize=(10,5.2));ax.axis("off")
ax.set_title("Put numbers behind each source of ranking evidence",loc="left",fontsize=17,pad=22)
rows=[["Route","Concrete comparison","Meaning"],
 ["Direct search",f"EA030312 vs EA030311\ncosine = {cosine_75_60:.4f} (10 ppm, √I)","Measured Caffeine\nreferences"],
 ["Inverse fingerprint","p = [0.9, 0.2, 0.7, 0.1]\nlog scores: A −0.7905, B −3.0241","Constructed four-bit\nexample"],
 ["Forward prediction","Measured [1, 0.2148, 0.1056]\nprediction [1, 0.20, 0.10]","Constructed prediction at\n138 / 110 / 195 m/z"]]
t=ax.table(cellText=rows,cellLoc="left",colWidths=[.23,.48,.29],bbox=[0,.15,1,.85]);t.auto_set_font_size(False);t.set_fontsize(11)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor(BORDER);cell.set_facecolor(LIGHT if r==0 else "white")
ax.text(0,0,"Only the first route uses two measured spectra. Toy predictions explain scoring, not model accuracy.",fontsize=10,color=GREY)
save(fig,"evidence-routes")

fig,ax=plt.subplots(figsize=(10,5.4));ax.axis("off")
ax.set_title("Class 2-like holdout: one Caffeine key, three distinct roles",loc="left",fontsize=17,pad=22)
rows=[["Role","EA030309 / EA030311 / EA030312","Caffeine structure"],
 ["Held-out queries","Observed spectra for prediction","Label used only\nfor evaluation"],
 ["Training / search\nreferences","Exclude all spectra of this key\nacross every library and model stage","No supervised\ntarget exposure"],
 ["Candidate database","No reference spectrum required","Retain Caffeine for Class 2"]]
t=ax.table(cellText=rows,cellLoc="left",colWidths=[.26,.43,.31],bbox=[0,.2,1,.8]);t.auto_set_font_size(False);t.set_fontsize(11)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor(BORDER);cell.set_facecolor(LIGHT if r==0 else "white")
ax.text(0,.09,"Group: RYYVLZVUVIJVGH. All matching observations belong to one outer fold.",fontsize=12,color=TEAL)
ax.text(0,-.01,"The candidate graph may remain available even though its measured training spectra are excluded.",fontsize=10,color=GREY)
save(fig,"validation-design")

# Existing run logs, visible 1,213-spectrum / 400-molecule input; not a hidden-runtime prediction.
stages=["Setup / loading", "Channels + FPNet", "MetFrag + 12 GBM fits", "Our scoring + 16 GBM fits", "Final scoring / dedup / output"]
# Derive first interval as the reported total minus the four logged intervals.
times=np.array([[49.3-28.62-9.66-3.08-2.10,28.62,9.66,3.08,2.10],
                [63.6-35.90-13.59-3.83-2.49,35.90,13.59,3.83,2.49]])
fig,ax=plt.subplots(figsize=(10,5.6));left=np.zeros(2)
for i,(label,color) in enumerate(zip(stages,["#c7cdd1",TEAL,"#789bad",CORAL,"#697680"])):
    ax.barh([1,0],times[:,i],left=left,color=color,label=label,height=.5)
    for j,y in enumerate([1,0]):
        if times[j,i]>7: ax.text(left[j]+times[j,i]/2,y,f"{times[j,i]:.2f}",ha="center",va="center",color="white",fontsize=11)
    left+=times[:,i]
for y,total in zip([1,0],left):ax.text(total+.6,y,f"{total:.1f} min",va="center",fontsize=12)
ax.set(yticks=[1,0],yticklabels=["T4 run","CPU run"],xlabel="Elapsed wall-clock minutes",xlim=(0,75),ylim=(-.5,1.5))
ax.set_title("Measured notebook intervals: most work is not isolated GPU inference",loc="left",fontsize=16,pad=22)
ax.legend(loc="upper left",bbox_to_anchor=(0,-.22),frameon=False,ncol=2,fontsize=10)
fig.subplots_adjust(bottom=.32)
fig.text(.125,-.015,"Existing v1/v2 logs · visible input only · intervals mix several operations; totals rounded to 0.1 min.",fontsize=10,color=GREY)
save(fig,"runtime-breakdown")


# Verify the physical-chemistry examples without loading competition data.
atoms = {"C": 12., "H": 1.00782503223, "O": 15.99491461957, "N": 14.00307400443}
proton = 1.0072764666
water = 2 * atoms["H"] + atoms["O"]
assert round(10 * atoms["C"] + 12 * atoms["H"] + 2 * atoms["O"], 6) == 164.083730
assert round(water, 6) == 18.010565
assert round(atoms["N"] + 3 * atoms["H"], 6) == 17.026549
assert round(atoms["C"] + atoms["O"], 6) == 27.994915
assert round(atoms["C"] + 2 * atoms["O"], 6) == 43.989829
assert round(proton - water, 6) == -17.003288
assert round(164.083729624 + proton, 6) == 165.091006
assert round(164.083729624 + proton - water, 6) == 147.080441
assert 1 + 10 - 12 / 2 == 5
assert round(30 * 28 / (300 + 28), 2) == 2.56
assert round(30 * 28 / (1000 + 28), 2) == .82
assert round((atoms["H"] - proton) / 164 * 1e6, 1) == 3.3
print("Physical chemistry numerical examples: PASS")
print(f"Created 12 PNG/SVG figure pairs in {OUT}; edited cover kept unchanged")
print("Toy structure assertions: PASS; synthetic MRR:", np.mean([1, .5, .2, 0]))
for s in [EUGENOL, ISOEUGENOL]:
    m = Chem.MolFromSmiles(s)
    print(rdMolDescriptors.CalcMolFormula(m), f"{Descriptors.ExactMolWt(m):.6f}", metric_key(s))

# Additional numerical examples in the bilingual text; no predictive model is executed.
p = np.array([.9,.2,.7,.1])
for bits, expected in [([1,0,1,0],-.7905395265685947),([1,1,0,0],-3.0241317480756886)]:
    bits=np.array(bits)
    assert np.isclose(np.sum(bits*np.log(p)+(1-bits)*np.log1p(-p)),expected)
assert np.isclose(np.mean([1,.5,1/25]),.5133333333333333)
assert np.isclose(np.mean([1,1,1/25]),.68)
assert round(1e6*np.sqrt(195*1.66053906892e-27/(2*1.602176634e-19*5000)),2)==14.22
print("Measured spectral cosine and worked scoring examples: PASS")
