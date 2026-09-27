"""Original figures for the CASMI introductory guide; no competition spectra.

Run with RDKit 2026.03.3, matplotlib, numpy, and torch available.
The library counts are aggregate statistics from the local September 15 train
file, audited September 27, 2026. Structure counts are deliberately not plotted.
"""
from pathlib import Path
from io import BytesIO
import os

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

# All peak locations and intensities below are fabricated for illustration.
x = np.array([43.02, 65.04, 77.04, 91.05, 105.07, 123.08, 147.08, 165.09])
y = np.array([.17, .23, .42, 1., .25, .6, .32, .2])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.vlines(x, 0, y, color=TEAL, lw=3)
ax.set(xlim=(30, 180), ylim=(0, 1.25), xlabel="Mass-to-charge ratio (m/z)", ylabel="Relative intensity")
ax.set_title("Reading an MS/MS spectrum", loc="left", fontsize=19, pad=17)
ax.annotate("Base peak = 1", xy=(91.05, 1), xytext=(51, 1.12), arrowprops={"arrowstyle": "->", "color": GREY})
ax.annotate("Fragment-ion peaks", xy=(123.08, .6), xytext=(132, .9), ha="center", arrowprops={"arrowstyle": "->", "color": GREY})
fig.text(.13, -.04, "Schematic only: not an observed eugenol spectrum", fontsize=10, color=GREY)
ax.grid(axis="y", alpha=.15)
save(fig, "spectrum-anatomy")

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

# Background: schematic energy profiles, not quantum-chemical calculations.
fig, ax = plt.subplots(figsize=(10, 5))
coord = np.linspace(0, 1, 401)
for barrier, product, color, label in [
    (1.45, .35, TEAL, "Path A: lower barrier"),
    (2.3, -.45, CORAL, "Path B: lower-energy product"),
]:
    energy = np.where(
        coord <= .5,
        barrier * np.sin(np.pi * coord) ** 2,
        product + (barrier - product) * np.cos(np.pi * (coord - .5)) ** 2,
    )
    ax.plot(coord, energy, color=color, lw=3, label=label)
ax.axhline(0, color=GREY, lw=1, ls="--", alpha=.5)
ax.set(xlim=(-.02, 1.03), ylim=(-.75, 3.1),
       xlabel="Reaction coordinate (schematic)", ylabel="Relative energy (arbitrary units)",
       xticks=[0, .5, 1], yticks=[])
ax.set_xticklabels(["Precursor", "Transition region", "Products"])
ax.set_title("A stable product can still be difficult to form", loc="left", pad=18, fontsize=18)
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.annotate("Lower barrier", xy=(.50, 1.45), xytext=(.18, 1.75),
            arrowprops={"arrowstyle": "->", "color": TEAL}, color=TEAL, fontsize=11)
ax.annotate("Lower product energy", xy=(.98, -.45), xytext=(.60, -.62),
            arrowprops={"arrowstyle": "->", "color": CORAL}, color=CORAL, fontsize=11)
fig.text(.13, -.04, "Illustration only. Product yield also depends on reaction rates, time, and secondary fragmentation.", color=GREY, fontsize=10)
save(fig, "fragmentation-pathways")

# Background: invented spectra with a common peak grid at three conditions.
ce_mzs = np.array([65.04, 91.05, 119.05, 147.08, 183.10, 225.12, 283.13, 301.14])
ce_intensities = [
    [.01, .025, .03, .05, .09, .14, .30, 1.00],
    [.12, .22, .32, .55, .85, 1.00, .55, .40],
    [.85, 1.00, .80, .62, .20, .07, .025, .01],
]
fig, axes = plt.subplots(3, 1, figsize=(10, 7.3), sharex=True)
for ax, intensities, title, color in zip(
    axes, ce_intensities,
    ["Lower energy", "Intermediate energy", "Higher energy"],
    [TEAL, "#4675a0", CORAL],
):
    ax.vlines(ce_mzs, 0, intensities, color=color, lw=3)
    ax.set(ylim=(0, 1.17), yticks=[0, 1], xlim=(50, 317))
    ax.set_title(title, loc="left", fontsize=13)
    ax.axvline(301.14, color=GREY, lw=1, ls="--", alpha=.35)
axes[0].annotate("Precursor", xy=(301.14, 1), xytext=(260, 1.02),
                 arrowprops={"arrowstyle": "->", "color": GREY}, fontsize=10)
axes[-1].set_xlabel("Mass-to-charge ratio (m/z)")
fig.supylabel("Relative intensity (each spectrum normalized independently)", fontsize=12)
fig.suptitle("Different collision energies reveal different fragments", fontsize=18, weight="bold", y=1.01)
fig.subplots_adjust(hspace=.55)
fig.text(.12, .015, "Synthetic illustration; not measured spectra and not a calibrated energy-response model.", color=GREY, fontsize=10)
save(fig, "collision-energy-series")


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


# A mass-balance calculation, not a claim about an observed fragmentation route.
fig, ax = diagram("From neutral mass to a possible neutral loss", height=6.6)
box(ax, .45, 7.6, 3.9, 1.9, "Neutral molecule M", "C10H12O2\n164.083730 Da")
box(ax, 5.5, 7.6, 3.95, 1.9, "Precursor [M+H]+", "C10H13O2+\nm/z 165.091006")
arrow(ax, (4.35, 8.55), (5.5, 8.55))
ax.text(4.92, 9.12, "+ H+", ha="center", fontsize=12, color=TEAL)
box(ax, 5.5, 3.8, 3.95, 1.9, "Possible fragment", "C10H11O+\nm/z 147.080441")
arrow(ax, (7.45, 7.6), (7.45, 5.7))
ax.text(7.7, 6.63, "− H2O", fontsize=14, color=CORAL, va="center")
ax.text(.55, 5.85, "Charge is conserved", fontsize=14, weight="normal")
ax.text(.55, 5.0, "+1 → +1 + 0", fontsize=17, color=TEAL)
ax.text(.55, 3.95, "The neutral is not detected\nas a charged fragment.", fontsize=12,
        color=GREY, linespacing=1.5)
ax.plot([.45, 9.45], [3.03, 3.03], color=BORDER, lw=.9)
ax.text(.55, 2.43, "165.091006 − 147.080441 = 18.010565 Da", fontsize=17)
ax.text(.55, 1.66, "Same unit charge: peak separation equals neutral-loss mass.", fontsize=12)
ax.text(.55, .55, "Mass bookkeeping only: it does not establish that this ion or pathway is observed.\nValues rounded to six decimals; no competition spectrum is used.",
        fontsize=11, color=GREY, linespacing=1.6)
save(fig, "ion-mass-balance")

# The unit of prediction and submission is a molecule, not an individual spectrum.
fig, ax = diagram("Several observations, one ranked answer", height=7)
for i, label in enumerate(["Spectrum A", "Spectrum B", "Spectrum C"]):
    left = .35 + i * 3.3
    box(ax, left, 7.75, 2.9, 1.9, label, "peaks + adduct\nenergy + instrument")
    arrow(ax, (left + 1.45, 7.75), (5, 6.8))
box(ax, 1.25, 5.1, 7.5, 1.7, "Group by molecule_id", "Keep the measurement conditions for each observation.")
arrow(ax, (5, 5.1), (5, 4.5))
box(ax, 1.25, 2.8, 7.5, 1.7, "Retrieve / generate → score → aggregate", "Combine evidence, then deduplicate by the scoring key.")
arrow(ax, (5, 2.8), (5, 2.2))
box(ax, 1.25, .5, 7.5, 1.7, "One row in submission.csv", "molecule_id  |  SMILES_1; SMILES_2; …; SMILES_25", color=CORAL)
save(fig, "molecule-to-ranking")

# Three discrete outcomes; these are examples, not model-performance measurements.
fig, ax = diagram("Candidate coverage and ranking are separate failures", height=6.4)
ax.text(.25, 9.45, "Where is the correct structure?", fontsize=13, color=GREY)
rows = [
    (6.45, "A  Missing from the pool", "Improve candidate coverage", "Absent", "0", CORAL),
    (3.6, "B  In the pool, ranked #50", "Improve scoring / reranking", "Present", "0", GREY),
    (.75, "C  Ranked #2 in the output", "Move a correct answer nearer the top", "Present", "0.5", TEAL),
]
for bottom, title, action, recall, rr, color in rows:
    box(ax, .25, bottom, 9.5, 2.25, title, action, color=color)
    ax.text(6.85, bottom + 1.62, "Pool", fontsize=10, color=GREY, ha="center")
    ax.text(6.85, bottom + .94, recall, fontsize=13, color=color, ha="center")
    ax.text(8.65, bottom + 1.62, "RR@25", fontsize=10, color=GREY, ha="center")
    ax.text(8.65, bottom + .82, rr, fontsize=22, color=color, ha="center")
save(fig, "candidate-diagnosis")

# Different evidence directions can contribute to the same candidate ranking.
fig, ax = diagram("Three ways to connect spectra and candidate structures", height=7.4)
routes = [
    (7.5, "A  Direct spectral search", "Observed spectrum", "Reference spectra", "Which measured spectrum is similar?"),
    (4.4, "B  Inverse prediction", "Observed spectrum", "Predicted fingerprint", "Which candidate has the predicted structural features?"),
    (1.3, "C  Forward prediction", "Candidate + conditions", "Predicted spectrum", "Which candidate best explains the observed spectrum?"),
]
for bottom, title, left, right, question in routes:
    ax.text(.2, bottom + 1.86, title, fontsize=15, weight="normal")
    box(ax, .25, bottom + .4, 4.25, 1.05, left)
    box(ax, 5.5, bottom + .4, 4.25, 1.05, right)
    arrow(ax, (4.5, bottom + .91), (5.5, bottom + .91))
    ax.text(.3, bottom - .15, question, fontsize=12, color=GREY)
ax.text(.25, .2, "These are complementary sources of evidence, not measured performance comparisons.", fontsize=11, color=GREY)
save(fig, "evidence-routes")

# Grouping by the metric identity prevents an exact structure leaking across a split.
fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
for ax, title in zip(axes, ["Split spectrum rows", "Hold out a structure key"]):
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, pad=15)
    ax.text(3.8, 8.9, "Train", ha="center", fontsize=12, color=GREY)
    ax.text(7.3, 8.9, "Validation", ha="center", fontsize=12, color=GREY)
    for row, label in enumerate(["A · 20 eV", "A · 40 eV", "B · 20 eV", "B · 40 eV"]):
        y0 = 7.4 - row * 1.35
        ax.text(.1, y0 + .4, label, fontsize=12, va="center")
        for x0 in [2.6, 6.1]:
            ax.add_patch(Rectangle((x0, y0), 2.4, .85, facecolor=LIGHT, edgecolor=BORDER, lw=.8))
assignments = [[0, 1, 0, 1], [1, 1, 0, 0]]
for ax, assignment in zip(axes, assignments):
    for row, col in enumerate(assignment):
        x0, y0 = [2.6, 6.1][col], 7.4 - row * 1.35
        ax.add_patch(Rectangle((x0, y0), 2.4, .85,
                               facecolor=TEAL if row < 2 else GREY, edgecolor="none"))
        ax.text(x0 + 1.2, y0 + .42, "A" if row < 2 else "B", color="white", fontsize=12, ha="center", va="center")
axes[0].text(.1, 1.8, "Both sides contain A and B.\nThis does not test unseen structures.", fontsize=12, color=CORAL, linespacing=1.6)
axes[1].text(.1, 1.8, "All spectra of A are held out.\nFor Class 2, retain A as a candidate.", fontsize=12, color=TEAL, linespacing=1.6)
fig.suptitle("Validation depends on what is held out", fontsize=18, x=.125, ha="left", y=1.02)
fig.text(.125, .015, "Schematic A/B identities. Apply the exclusion across libraries and all supervised training stages.", fontsize=10, color=GREY)
fig.subplots_adjust(wspace=.3)
save(fig, "validation-design")

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
print(f"Created 11 PNG/SVG figure pairs in {OUT}; edited cover kept unchanged")
print("Toy structure assertions: PASS; synthetic MRR:", np.mean([1, .5, .2, 0]))
for s in [EUGENOL, ISOEUGENOL]:
    m = Chem.MolFromSmiles(s)
    print(rdMolDescriptors.CalcMolFormula(m), f"{Descriptors.ExactMolWt(m):.6f}", metric_key(s))
