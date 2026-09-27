---
title: "Enveda CASMI 2026: From Mass Spectra to Molecular Structures"
description: "What is CASMI 2026 asking us to build, and where should we begin? Follow one molecule from spectra to ranked structures, through the physical chemistry, public baselines and leaderboard, first experiments, and likely next directions."
date: 2026-09-27 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, casmi, mass-spectrometry, metabolomics, molecular-identification, cheminformatics, working-note]
math: true
mermaid: false
image:
  path: /assets/img/casmi-2026-guide/hero.png?v=af0e6fe7c947
  alt: "A restrained adaptation of the official Enveda CASMI molecular header"
pin: false
published: true
permalink: /posts/CASMI-2026-From-Mass-Spectra-to-Molecular-Structures/
---

<style>
article .content table:not(.rouge-table) th,
article .content table:not(.rouge-table) td {
  white-space: normal;
  word-break: keep-all;
  overflow-wrap: anywhere;
}
article h1, article .content h2, article .content h3 {
  word-break: keep-all;
  overflow-wrap: break-word;
  text-wrap: balance;
}
article .content p, article .content li {
  word-break: keep-all;
  overflow-wrap: break-word;
}
article .content .mermaid, article .content .mermaid * {
  font-family: Arial, sans-serif !important;
}
article .content mjx-container[display="true"] {
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 0.25rem;
}
article .content .casmi-figure {
  margin: 2rem 0 2.2rem;
}
article .content .casmi-figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #fff;
  border: 1px solid #d9dfe3;
  border-radius: 0;
}
article .content .casmi-figure figcaption {
  margin-top: 0.75rem;
  padding-left: 0.8rem;
  border-left: 2px solid #89969f;
  color: var(--text-muted-color);
  font-size: 0.9rem;
  line-height: 1.75;
  word-break: keep-all;
}
article .content .casmi-figure figcaption strong {
  color: var(--text-color);
}
article .content table:not(.rouge-table) {
  font-size: 0.93rem;
  line-height: 1.65;
  border-top: 2px solid #89969f;
  border-bottom: 1px solid #89969f;
}
article .content table:not(.rouge-table) th {
  background: rgba(105, 118, 128, 0.09);
  border-bottom: 1px solid #89969f;
}
article .content table:not(.rouge-table) th,
article .content table:not(.rouge-table) td {
  padding: 0.65rem 0.8rem;
  vertical-align: top;
}
article .content .casmi-cover-credit {
  color: var(--text-muted-color);
  font-size: 0.8rem;
  line-height: 1.6;
}
@media (max-width: 600px) {
  article .content .table-wrapper { overflow-x: auto; }
  article .content table:not(.rouge-table) { min-width: 580px; }
  article .content table:not(.rouge-table) th,
  article .content table:not(.rouge-table) td { min-width: 105px; }
}
</style>

[한국어판 읽기]({{ site.baseurl }}/posts/CASMI-2026-From-Mass-Spectra-to-Molecular-Structures-KR/)

<p class="casmi-cover-credit">Cover adapted from the official Enveda CASMI 2026 Kaggle header, with restrained color and background adjustments. The molecular model in the cover is separate from the eugenol example calculated below.</p>

## Start here: what are we trying to identify?

Imagine that a laboratory detects an interesting substance in a plant extract. The instrument can measure ions made from that substance and the smaller ions produced when it fragments. The scientist still has a harder question: **which arrangement of atoms produced those measurements?** Answering it connects a chemical signal to a substance that can be studied further—for example, in metabolism, environmental analysis, or drug discovery.

[Enveda CASMI 2026 — Molecule ID From Mass Spectra][competition] turns that identification problem into a Kaggle competition. CASMI means *Critical Assessment of Small Molecule Identification*, an effort that began in 2012. The task here is to turn several tandem mass spectra, or **MS/MS spectra**, of an unknown molecule into an ordered list of at most 25 possible molecular structures. We write structures as **SMILES**, a text representation of atoms and bonds. The score rewards finding the correct structure and placing it early in the list.

The scientific motivation is broader than recognizing compounds already measured in a reference library. Some test molecules have a public reference spectrum; others have a known structure but no public spectrum; others are absent from the specified public structure databases. A useful identification system has to make progress as those sources of prior knowledge disappear. That is why this competition connects database search, machine learning, and chemical reasoning. [Competition overview][competition] · [Data description][data]

### What do we build, submit, and receive back?

We build an **inference pipeline**: a sequence of operations that reads the spectra, proposes possible structures, scores the evidence for each, and returns a ranked list for every molecule. A trained neural network can be part of that pipeline, but so can a spectral library, a database index, and a model that combines several scores.

| Stage | What happens | What it tells us |
|---|---|---|
| Development | Use labeled training spectra and permitted external resources to build and test the pipeline. | Whether a method works on our chosen held-out molecules. |
| Submission | Save and run a notebook version with the required offline inputs, then select that version for a competition submission. | Which complete procedure Kaggle will run. |
| Hidden evaluation | Kaggle reruns the notebook on hidden spectra; it writes one ranked SMILES list per molecule to <code>submission.csv</code>. | How the procedure handles unknown inputs. |
| Leaderboard | The public portion provides feedback during the competition; the private portion determines the final ranking. | Performance on those evaluation molecules, under the competition's matching rule. |

These stages explain an easy source of confusion. The downloadable test file is a practice input drawn from training data. Producing a good-looking answer for it checks execution and formatting. A local validation score comes from a separate experiment we design. A leaderboard score comes from Kaggle's hidden evaluation. We will use each for the question it can answer. [Data and evaluation][data]

### The first milestone is one prediction we can explain

For one held-out molecule, we want to follow the whole chain: **what the instrument observed → which structures entered the candidate pool → why one outranked another → what the evaluator accepted**. A *candidate pool* is the larger list we consider before selecting the final 25; a *ranker* is the rule or learned model that orders it.

If the answer never entered that pool, a better ranker cannot recover it. If it entered but lost to a similar structure, adding millions of unrelated molecules may only make the decision harder. This distinction will guide the first experiment, our reading of public notebooks, and the choice of the next model.

The article follows that chain. **Sections 1–3 explain the task and its score. Sections 4–6 introduce the physical chemistry and training data. Sections 7–11 build the modeling and validation ideas. Sections 12–16 ask where the public work stands, what to try first, and how the competition may develop.** Readers already familiar with mass spectrometry can skim the physical background; newcomers need not master every equation before understanding the project.

Rules and public resources were checked on **September 27, 2026**. Public results, our reproduced baseline, and proposed experiments are identified separately. The outlook is an interpretation of the evidence available now, not a description of undisclosed leading solutions.

**Useful starting sources:** the [overview][competition], [data description][data], and [official metric][metric] define the task. The [rules][rules] and [organizer welcome post][welcome] define resource conditions. For implementation, start with [Analog Propagation][analog-notebook] and [Two Rankers, One Engine][two-rankers]; Section 12 explains their place in the public work.

## 1. Inputs and outputs: many observations, one candidate list

The basic unit in CASMI is **a molecule, not a single spectrum**. Multiple test rows can share a <code>molecule_id</code>. We group them to produce one ranked answer.

<figure class="casmi-figure" id="casmi-fig-1">
  <img src="/assets/img/casmi-2026-guide/molecule-to-ranking.png?v=94c17e4f6ffe" alt="Multiple spectra grouped by molecule identity become one ranked submission row" width="1475" height="1257" loading="lazy">
  <figcaption><strong>Figure 1.</strong> The prediction unit in CASMI. Preserve each observation's conditions, combine evidence for the molecule, and deduplicate by the structure key to produce one ranked row. Three spectra are shown for illustration, not as a fixed observation count for every molecule.</figcaption>
</figure>

According to the [official data description][data], the hidden test contains approximately 1,500 spectra from about 400 molecules. Each molecule has 1–16 spectra, with a median of 3, measured on a Bruker timsTOF. These are published descriptions of the hidden test, not statistics extrapolated from the downloadable example file.

The main columns can be read as follows. In practice, inspect the Parquet schema alongside the documentation.

| Information | Representative columns | Why it matters |
|---|---|---|
| Observations of the same molecule | <code>molecule_id</code> | Aggregate test spectra into one prediction |
| Observation identity | <code>spectrum_id</code> | Trace processing for a particular spectrum |
| Precursor information | <code>precursor_mz</code>, <code>adduct</code> | Estimate neutral mass and constrain candidates |
| Fragment ions | <code>ms2_mzs</code>, <code>ms2_normalized_intensities</code> | Supply the main input to retrieval and prediction models |
| Measurement conditions | <code>ionization_mode</code>, <code>instrument_type</code>, <code>collision_energy_ev</code> | Distinguish polarity, instrument, and collision energy |
| Training targets and provenance | <code>normalized_smiles</code>, <code>inchikey</code>, <code>ingest_lib</code>, and others | Learn structures and analyze differences between sources |

<code>molecule_id</code> and <code>spectrum_id</code> are test identifiers. The training file inspected here lacks both columns, so training observations need to be grouped using identities calculated from their structures. Collision energy should not be assumed to be a single scalar in every row: inspect arrays and missing values.

### Follow one unknown molecule through the pipeline

For a running example, imagine a molecule measured at several collision energies. We will use **eugenol and isoeugenol**, two structures with the same formula, to make the choices concrete. This is a teaching example, not an account of a hidden-test molecule; the numerical chemistry appears in Section 4.

First, the precursor mass and its ion form restrict which neutral structures could fit. Next, library search asks whether a measured spectrum of one of those structures already exists. If no convincing reference match appears, the pipeline can use spectra from related molecules or predict structural features from the unknown spectrum. It might then predict the spectra that eugenol and isoeugenol would produce and ask which candidate better explains the observations.

Finally, it combines the evidence across the molecule's measurements and orders the candidates. The output may contain both structures; their order matters. At each stage, there is a different question: **is the answer available, does the measurement distinguish it, and does our scoring method use that distinction?** We will return to those questions when interpreting public methods and designing experiments.

### The submission CSV is simple; the work before it is not

The output file is <code>submission.csv</code>, with columns <code>molecule_id</code> and <code>smiles</code>. The <code>smiles</code> field contains candidates in decreasing order of confidence, separated by semicolons.

```csv
molecule_id,smiles
demo_001,COc1cc(CC=C)ccc1O;COc1cc(C=CC)ccc1O
```

This illustrative row ranks eugenol first and isoeugenol second; it is not actual submission data. Filling all 25 positions is not mandatory, but retaining a sufficiently broad set of plausible answers matters. Repeating equivalent representations of the same structure wastes candidate slots.

### The downloadable test.parquet is not a performance benchmark

The visible <code>test.parquet</code> is a **format-checking placeholder derived from training data**. Kaggle replaces it with hidden data during scoring. A high retrieval match rate on that file does not establish performance on unknown molecules. [Official data description][data]

Use it to check that the columns can be read, results are grouped by molecule, and a CSV is produced for every ID. Measure predictive performance on a separate validation split. The **public leaderboard** also scores a hidden public portion of the test, not the downloadable placeholder.

## 2. MRR@25: finding the answer and putting it near the top

The metric is **mean reciprocal rank**. If the first correct candidate is at rank $$r$$, the molecule receives $$1/r$$; if no correct answer appears within 25 candidates, it receives zero. Average these contributions over $$U$$ molecules. The reference is the [official evaluation explanation and implementation][metric].

$$
\begin{aligned}
\operatorname{MRR@25}&=\frac{1}{U}\sum_{u=1}^{U}\operatorname{RR}_u,\\
\operatorname{RR}_u&=\begin{cases}
1/r_u, & r_u\leq25,\\
0, & \text{otherwise}.
\end{cases}
\end{aligned}
$$

For example, if four molecules have correct answers at ranks 1, 2, and 5, and outside the list, the score is:

$$
\frac{1+0.5+0.2+0}{4}=0.425.
$$

This is an illustrative calculation, not a measured result from a model.

<figure class="casmi-figure" id="casmi-fig-2">
  <img src="/assets/img/casmi-2026-guide/reciprocal-rank.png?v=2f32b3beb3a9" alt="Reciprocal rank decays rapidly as the correct candidate moves down the list" width="1745" height="1004" loading="lazy">
  <figcaption><strong>Figure 2.</strong> A molecule contributes 1 when the correct answer ranks first, 0.5 at rank 2, and 0.04 at rank 25. A change in the overall score is the illustrated contribution change divided by the number of evaluated molecules.</figcaption>
</figure>

Two properties matter. Finding the answer at rank 25 is better than missing it. Moving an answer from rank 2 to rank 1, however, is worth much more. For one molecule, the first change adds 0.04 and the second adds 0.5: a factor of 12.5.

We therefore need to measure **whether the answer enters the candidate pool** separately from **whether the ranking puts it near the top**.

| Metric | The question it answers |
|---|---|
| Candidate-pool Recall@K | Is the answer among the K structures passed to the ranker? |
| Final Hit@25 | Is the answer in the 25 submitted candidates? |
| Top-1 accuracy | Is the first candidate correct? |
| MRR@25 | How often is the answer found, and how early does it appear? |

A retrieval pool of K = 1,000 and the final 25 candidates are different stages. If the answer is absent from the pool, the downstream ranker cannot recover it. If Recall@1,000 is high but the correct structure consistently ranks 50th, the final score remains low.

<figure class="casmi-figure" id="casmi-fig-3">
  <img src="/assets/img/casmi-2026-guide/candidate-diagnosis.png?v=c8b734cb5af1" alt="Three examples distinguish candidate-pool failure from ranking failure and successful top-25 retrieval" width="1475" height="1164" loading="lazy">
  <figcaption><strong>Figure 3.</strong> Illustrative cases separating missing candidates from ranking errors. Both an absent answer and one ranked 50th give RR@25 = 0, but require different improvements. An answer at rank 2 contributes 0.5. These are per-molecule contributions, not measured model performance.</figcaption>
</figure>

### What counts as the same molecule?

The competition uses RDKit **2026.03.3** to canonicalize tautomers, then compares the first 14 characters of the InChIKey. **Tautomers** are related structural forms that differ in hydrogen placement and bonding. Canonicalization groups forms that the competition treats as equivalent. Using the first InChIKey block also means that stereochemical differences are not distinguished in the final match. [Official metric code][metric]

This does not mean that matching the molecular formula is enough. Eugenol and isoeugenol have different keys. The E/Z stereoisomers of isoeugenol, however, share the connectivity used by this metric. The competition's answer definition is distinct from complete structural elucidation in a laboratory.

<details markdown="1">
<summary>Check it in code: scoring keys and candidate deduplication</summary>

The following small example demonstrates the core normalization rule for ordinary single-molecule SMILES. It does not replace the full submission validator or the official scorer's exception handling.

```python
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

assert rdBase.rdkitVersion == "2026.03.3"
tautomers = rdMolStandardize.TautomerEnumerator()

def scoring_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = tautomers.Canonicalize(mol)
    key = Chem.MolToInchiKey(canonical)
    if not key:
        raise ValueError("InChIKey conversion failed")
    return key[:14]

def unique_candidates(ranked_smiles, limit=25):
    seen, result = set(), []
    for smiles in ranked_smiles:
        key = scoring_key(smiles)
        if key in seen:
            continue
        seen.add(key)
        result.append(smiles)
        if len(result) == limit:
            break
    return result

assert scoring_key("COc1cc(CC=C)ccc1O") != scoring_key("COc1cc(C=CC)ccc1O")
assert scoring_key("C/C=C/c1ccc(O)c(OC)c1") == scoring_key("C/C=C\\c1ccc(O)c(OC)c1")
```

Generated SMILES that fail parsing should first be recorded and excluded in a separate step. Check that none remain when constructing the final file. Deduplication must preserve the original ranking order.

</details>

## 3. Three kinds of unknown molecule

“A molecule we have not seen before” can describe several different situations. The organizer's explanation and [classification clarification][metric-discussion] distinguish the following.

| Type | Public reference spectra | Public structure databases | Main challenge |
|---|---|---|---|
| Class 1 | A spectrum of the correct structure is available | The structure is known | Recognize the molecule across measurement conditions |
| Class 2 | No spectrum of the correct structure is available | The structure is available | Choose the known structure that best explains the observed spectrum |
| Class 3 | Not available | The correct structure is absent from PubChem and COCONUT | Propose a structure outside those databases |

A **spectral library** stores a measured spectrum paired with a known structure. A **structure database** can list that structure without containing any MS/MS measurement of it.

As a rough analogy, Class 1 resembles matching a photograph taken under different lighting. In Class 2, we have a name and a drawing but no reference photograph. In Class 3, the object is missing even from the drawing catalog. The analogy is imperfect, but it explains why the cases require different tools.

A pool constructed only from PubChem and COCONUT cannot retrieve a Class 3 answer. When the answer is absent from the fixed pool, structure generation or modification offers a way to propose it. That does not imply allocating all computation to generation from the beginning. Keep official information about class proportions separate from participant estimates, and weigh obtainable performance against computation cost.

We now know what the system must return and how it is judged. The next question is where the measurements contain the evidence needed to choose that answer. We begin with how mass restricts candidates, then ask how fragment ions can help distinguish similar structures.

## 4. How do we distinguish molecules with the same mass?

Consider eugenol and isoeugenol. Both have the formula $$\mathrm{C}_{10}\mathrm{H}_{12}\mathrm{O}_{2}$$: ten carbon atoms, twelve hydrogen atoms, and two oxygen atoms. Because their atom counts are identical, their monoisotopic masses are identical too. Their structures can be inspected in PubChem's entries for [eugenol][eugenol] and [isoeugenol][isoeugenol].

<figure class="casmi-figure" id="casmi-fig-4">
  <img src="/assets/img/casmi-2026-guide/isomers.png?v=dfebb3cde1f6" alt="Eugenol and isoeugenol: equal formula and exact mass, different connectivity" width="1785" height="1033" loading="lazy">
  <figcaption><strong>Figure 4.</strong> The double bond occupies a different position in the two molecules. Structures and masses were calculated from the SMILES below with RDKit 2026.03.3. The 14-character strings are structure keys obtained with the competition's normalization procedure.</figcaption>
</figure>

The position of the double bond differs in the three-carbon side chain attached to the ring. These are **constitutional isomers**: the same atoms connected differently. Knowing that the mass is approximately 164.083730 Da cannot tell us which of the two we have.

This is where fragmentation becomes useful. Different bonding patterns can change where a molecule breaks, where the charge remains, and the relative abundance of the fragments. MS/MS lets us observe such differences. Whether two structures can actually be distinguished depends on the measurement conditions and spectral quality; different structures do not always yield clearly separable spectra.

### How do we represent a structure in software?

**SMILES** encodes a molecular graph as a string. Letters represent atoms, while symbols describe bonds, branches, and ring closures. Our two examples can be written as follows.

```text
Eugenol:     COc1cc(CC=C)ccc1O
Isoeugenol:  COc1cc(C=CC)ccc1O
```

The lowercase <code>c</code> denotes aromatic carbon, <code>=</code> denotes a double bond, and parentheses indicate branches. We do not need to memorize the entire syntax to begin. The target of the competition is the **molecular structure represented by the string**.

The same molecule can have different SMILES strings depending on where traversal begins. Comparing prediction and target strings directly would therefore be wrong. The competition normalizes structures before comparing them, as explained in Section 2.

The following code calculates the formulas and masses of both examples. RDKit is an open-source toolkit widely used to read and work with molecular structures.

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

structures = {
    "eugenol": "COc1cc(CC=C)ccc1O",
    "isoeugenol": "COc1cc(C=CC)ccc1O",
}

for name, smiles in structures.items():
    mol = Chem.MolFromSmiles(smiles)
    print(name, rdMolDescriptors.CalcMolFormula(mol),
          f"{Descriptors.ExactMolWt(mol):.6f}")
```

```text
eugenol C10H12O2 164.083730
isoeugenol C10H12O2 164.083730
```

These values differ from the average molecular weight commonly shown in a composition table. **Exact mass** is calculated for a specified isotopic composition. Here we calculate **monoisotopic mass**, using the most abundant isotope of each element. This is the relevant quantity when working with small mass differences in high-resolution mass spectrometry.

### Why calculate so many decimal places?

The number of protons in a nucleus determines its element. Atoms of the same element with different neutron counts are **isotopes**. The mass of $$^{12}\mathrm C$$ is exactly 12 Da, whereas $$^1\mathrm H$$ and $$^{16}\mathrm O$$ have masses of approximately 1.007825 and 15.994915 Da. Adding integer atomic masses would discard these differences. The values come from the NIST tables for [carbon][nist-carbon], [hydrogen][nist-hydrogen], and [oxygen][nist-oxygen].

For a specified isotopic composition, molecular mass is calculated by summing atomic masses. At the precision used here, the extremely small mass difference associated with chemical binding energy is neglected.

$$
m_{\mathrm{exact}}=\sum_e n_e\,m_e.
$$

Here, $$n_e$$ is the atom count and $$m_e$$ is the mass of the selected isotope. For eugenol, this gives:

$$
\begin{aligned}
m&=10(12)+12(1.007825032)\\
&\quad+2(15.994914620)\\
&\simeq164.083730\ \mathrm{Da}.
\end{aligned}
$$

Two formulas with different atom counts can share the same integer mass and still differ in their decimal places. Eugenol and isoeugenol, however, have the same formula and cannot be separated this way, regardless of mass accuracy. **More accurate mass measurements constrain composition; they do not directly reveal the order in which atoms are connected.**

Isotopes also produce additional spectral peaks. Replacing one $$^{12}\mathrm C$$ atom with $$^{13}\mathrm C$$ adds approximately 1.003355 Da. For ions with the same charge number $$z$$, the corresponding separation in $$m/z$$ is:

$$
\Delta(m/z)\simeq\frac{1.003355}{|z|}.
$$

A sufficiently informative isotope pattern can help constrain charge and elemental composition. However, **the CASMI input does not provide a complete MS1 isotope envelope as a separate array**. Information that is useful in mass spectrometry generally should not be assumed to exist in the competition files. Any use of isotope-related peaks remaining in MS2 lists must account for how those lists were processed.

### What can a formula tell us? Valence and unsaturation

Matching the mass is not sufficient when proposing a formula. For ordinary organic molecules, we usually begin with the familiar valences of carbon, oxygen, and neutral nitrogen: four, two, and three bonds, respectively. These constraints mean that an arbitrary combination of atom counts need not correspond to a valid molecular graph.

A useful example is the **degree of unsaturation**, or double-bond equivalent (DBE). For neutral, closed-shell C/H/N/O/halogen molecules with conventional valences, we can write:

$$
\operatorname{DBE}=1+C-\frac{H+X}{2}+\frac{N}{2}.
$$

The letters denote atom counts; $$X$$ is the total number of halogen atoms such as F, Cl, Br, and I. Divalent oxygen does not appear explicitly in this expression. Relative to a saturated acyclic hydrocarbon, removing two hydrogens allows one ring or one double bond; a triple bond counts as two units.

For eugenol, $$1+10-12/2=5$$. The benzene ring contributes four units—one ring and three double bonds—and the side-chain double bond contributes one. Isoeugenol has the same DBE. The quantity constrains possible structures but does not rank these two isomers.

**In CASMI, this knowledge can help check formulas and generated candidates.** It should not become a universal rejection rule for all ions and elements. Charge, radicals, and the multiple valence states of phosphorus and sulfur complicate the interpretation. [Kind and Fiehn's original work on formula filtering][golden-rules] discusses these limitations. Any chemical constraint needs a stated scope and a check on how often it incorrectly removes the true candidate.

## 5. What does an MS/MS spectrum show?

A brief walk through the measurement process helps explain the input.

1. **Separate the mixture.** Liquid chromatography (LC) separates components over time.
2. **Form ions.** Ionization gives the molecules an electrical charge.
3. **Select an ion.** A precursor ion with a mass-to-charge ratio of interest is selected.
4. **Induce fragmentation.** Collisions transfer energy to the ion and produce fragment ions.
5. **Measure the fragments.** The instrument records their peak positions and intensities.

The result is a **tandem mass spectrum**, usually called MS/MS or MS2. CASMI provides these processed measurements in tabular form, so we do not start by processing raw LC data. [MassSpecGym][massspecgym] also describes the computational background and three representative tasks in this area.

<figure class="casmi-figure" id="casmi-fig-5">
  <img src="/assets/img/casmi-2026-guide/spectrum-anatomy.png?v=92ac36c4af8a" alt="A schematic MS/MS spectrum with peak positions and normalized intensities" width="1777" height="1001" loading="lazy">
  <figcaption><strong>Figure 5.</strong> A synthetic spectrum illustrating how to read peaks. It is neither a measured spectrum of eugenol nor competition data.</figcaption>
</figure>

The horizontal axis is $$m/z$$: ion mass divided by the magnitude of its charge number. The vertical axis gives relative detected intensity. The strongest peak is the **base peak**; the competition's normalized intensities set it to 1. Intensity is a detector signal, not a direct count of the number of distinct fragments.

In software, a spectrum is closer to two arrays than to an image.

```python
# Synthetic values for illustration
ms2_mzs = [77.04, 91.05, 123.08]
ms2_normalized_intensities = [0.42, 1.00, 0.60]
```

Values at the same position in the two arrays describe one peak. The arrays can vary in length. We can bin peaks into fixed mass intervals to form a vector, or use a model that processes the peak list directly.

The physical background below answers three practical questions: **what quantity was measured, why particular fragments appeared, and which conditions the model needs to retain**. On a first reading, focus on those questions; the equations make the assumptions more precise.

### How does an instrument read mass?

The timsTOF instrument family used in the competition combines trapped ion mobility with quadrupole time-of-flight mass spectrometry. The basic idea behind **time of flight (TOF)** is to measure how long an accelerated ion takes to travel a specified distance. [Bruker's instrument overview][bruker-timstof]

In an idealized model, an ion with negligible initial kinetic energy is accelerated through a potential difference $$V$$. Conservation of energy gives:

$$
\begin{aligned}
|z|eV&=\frac12m_{\mathrm{ion}}v^2,\\
t&=\frac Lv=L\sqrt{\frac{m_{\mathrm{ion}}}{2|z|eV}}.
\end{aligned}
$$

Here, $$e$$ is the elementary charge, $$v$$ is velocity, and $$L$$ is the flight distance. **Use SI units in this equation: kilograms for mass, volts for potential difference, and meters for distance.** For the same potential difference and path, flight time scales with the square root of the mass-to-charge ratio. Arrival times can therefore be converted into calibrated $$m/z$$ values. Real instruments include more elaborate corrections for energy distributions and flight paths.

We receive peak positions that have already undergone this conversion. Nor should the presence of ion mobility hardware lead us to assume that collision cross section (CCS) or mobility time is supplied as an input. **What an instrument can measure and what the dataset actually contains are separate questions.**

**Keep the physical quantities and their units distinct.**

| Quantity | Meaning and units | What to distinguish here |
|---|---|---|
| Neutral exact mass | Mass calculated from isotopic composition, in Da | The neutral molecule before ionization |
| Charge number <code>z</code> | Signed integer multiple of elementary charge | Divide by its magnitude <code>&#124;z&#124;</code> when calculating m/z |
| Spectral m/z | Ion mass divided by charge-number magnitude | Converting to neutral mass requires the adduct and charge |
| Collision energy | A collision condition, expressed in eV or other conventions | Distinguish instrument voltage, NCE, and actual internal energy |
| Molar activation energy | Reaction barrier expressed per mole, for example kJ/mol | Match the units of <code>R</code> in the Arrhenius equation |
| Relative intensity | Normalized signal within one spectrum, dimensionless | It does not give an absolute abundance ratio across spectra |

### Precursor m/z is not the neutral molecular mass

A mass analyzer uses electric fields to move and separate ions. The relevant question is therefore **which ionic form enters the instrument**. Electrospray ionization (ESI), widely used in LC-MS, forms gas-phase ions through processes involving charged droplets and solvent removal. It is sufficiently gentle to preserve intact molecular ions in many cases, but molecules differ in ionization efficiency and in the ionic forms they produce. [Agilent's LC/MS introduction][agilent-lcms]

For example, <code>[M+H]+</code> is a neutral molecule $$M$$ with an added proton. <code>[M-H]-</code> has lost a proton, while <code>[M+Na]+</code> carries a sodium ion. The dataset describes these ionic forms through its **adduct** information. Solution acidity, functional groups, solvents, and salts can affect which forms are observed. [Technical explanation of LC conditions and ionization][agilent-ionization]

Let the signed mass change on forming the ion be $$\Delta m_{\mathrm{ion}}$$. Using masses in Da, we can write:

$$
\begin{aligned}
(m/z)_{\mathrm{precursor}}
&=\frac{m(M)+\Delta m_{\mathrm{ion}}}{|z|},\\
m(M)&=|z|(m/z)_{\mathrm{precursor}}-\Delta m_{\mathrm{ion}}.
\end{aligned}
$$

Several examples for singly charged ions make the conversion concrete.

| Ionic form | Mass change relative to the neutral, $$\Delta m_{\mathrm{ion}}$$ | Recovering neutral mass |
|---|---|---|
| <code>[M+H]+</code> | Approximately +1.007276 Da | Subtract 1.007276 from precursor m/z |
| <code>[M-H]-</code> | Approximately −1.007276 Da | Add 1.007276 to precursor m/z |
| <code>[M+Na]+</code> | Approximately +22.989221 Da | Subtract 22.989221 from precursor m/z |
| <code>[M-H2O+H]+</code> | Approximately −17.003288 Da | Add 17.003288 to precursor m/z |

If eugenol is observed as <code>[M+H]+</code>, its precursor m/z is approximately 165.091006. Treating that as the neutral mass would cause retrieval to miss the answer. The 1.007276 in the table is also the **proton mass**, not the neutral hydrogen-atom mass of 1.007825. Their difference is approximately one electron mass, 0.000549 Da, or about 3.3 ppm near 164 Da. That matters for a narrow mass window. These are mass-calculation conventions, not instructions to apply a blanket correction to measured peaks.

Thus even the first step of finding nearby masses requires adduct interpretation. The [data description][data] covers positive and negative ions and water-loss forms; an [update announcement][train-update] also describes corrected water-loss annotations in the training data.

### Where the proton sits can change fragmentation

The notation <code>[M+H]+</code> tells us that a proton has been added, but does not specify its location. Molecules with several nitrogen or oxygen atoms can have multiple protonation sites, and the proton can move during fragmentation. Charge location changes electron distribution and the pathways available for dissociation. A [CIDMD study of small molecules][protonation-study] examines why this matters for spectrum prediction.

This also brings in **resonance and conjugation**. Delocalizing charge over several atoms can stabilize a fragment ion. The existence of a stable fragment, however, does not guarantee a pathway that produces much of it within the observation time.

For CASMI, ignoring polarity and adducts would combine different physical processes into one target spectrum for the same neutral structure. A structure-to-spectrum model should use the available measurement conditions, and its limitations should be checked for adducts outside its training support. Solution-phase acidity alone is also insufficient to determine the protonation site of a gas-phase ion.

### Collision energy is not energy deposited directly into one bond

In collision-induced dissociation (CID), an ion collides with neutral gas and **converts part of its translational kinetic energy into internal energy**, which can lead to fragmentation. Internal energy includes molecular vibration. This is the process described by [IUPAC's definition of CID][iupac-cid].

A collision energy of 30 eV does not mean that 30 eV has been placed into one bond. Consider the simplest case: an ion colliding with a stationary gas particle. The ion's kinetic energy in the laboratory frame, $$E_{\mathrm{lab}}$$, and the relative-motion energy available in the center-of-mass frame, $$E_{\mathrm{cm}}$$, are related by:

$$
E_{\mathrm{cm}}
=\frac{m_{\mathrm{gas}}}{m_{\mathrm{ion}}+m_{\mathrm{gas}}}
E_{\mathrm{lab}}.
$$

The relationship follows by subtracting the kinetic energy of the overall center-of-mass motion from the two-body kinetic energy. As an **illustrative calculation**, a 300 Da ion colliding with a 28 Da gas particle at $$E_{\mathrm{lab}}=30$$ eV gives $$E_{\mathrm{cm}}\simeq2.56$$ eV. For a 1,000 Da ion, the value is about 0.82 eV. Even this energy is not all deposited into the ion's internal modes. Actual experiments also depend on collision counts, gas, residence time, and energy distributions. The [MassKinetics study][masskinetics] connects energy transfer with reaction rates.

Here, $$E_{\mathrm{lab}}$$ means **the total kinetic energy of the ion**. If an instrument reports acceleration voltage or normalized collision energy instead, its definition must be checked first. Substituting the competition's <code>collision_energy_ev</code> into this equation does not directly reconstruct actual internal energy.

The same 20 eV label on different instruments therefore need not describe identical physical conditions. Collision energy is a useful input, but it should be interpreted with instrument and ion information. Filling missing energy values with zero would also confuse an unknown measurement with the absence of supplied energy.

### Which fragments become abundant? Stability and reaction rates

Would cutting every bond in a structure once reproduce its spectrum? No. A fragmentation pathway must cross an **activation barrier** and proceed within the observation time. Some pathways involve hydrogen transfer or structural rearrangement. Average bond dissociation energies for neutral molecules cannot simply rank all possible dissociation routes of an ion.

<figure class="casmi-figure" id="casmi-fig-6">
  <img src="/assets/img/casmi-2026-guide/fragmentation-pathways.png?v=dc4c1a36ec21" alt="Schematic reaction-energy profiles comparing barrier height and product stability" width="1705" height="1094" loading="lazy">
  <figcaption><strong>Figure 6.</strong> Two schematic dissociation pathways. Path B can lead to a lower-energy product while crossing a higher barrier. Observed abundance also depends on accessible states, reaction time, and secondary fragmentation. These are not calculations for a particular molecule.</figcaption>
</figure>

A simple kinetic example has two competing pathways from an excited precursor ion $$P^{+*}$$:

$$
P^{+*}\xrightarrow{k_A}A^++N_A,
\qquad
P^{+*}\xrightarrow{k_B}B^++N_B.
$$

Here, $$N_A,N_B$$ are undetected neutral products, and $$k_A,k_B$$ are the rate constants. If these are the only reactions, both rate constants are fixed, and the products do not fragment further, the fraction of precursors that has reacted is:

$$
f_{\mathrm{reacted}}(t)=1-e^{-(k_A+k_B)t}.
$$

The fractions of product ions entering paths A and B are $$k_A/(k_A+k_B)$$ and $$k_B/(k_A+k_B)$$. If A proceeds three times as fast as B, this simple model gives a 3:1 formation ratio. **The measured intensity ratio is not necessarily 3:1.** Different transmission or detection efficiencies, or subsequent fragmentation, can change it.

Even this small model gives us a reason to predict peak presence and intensity separately. Knowing which fragments are possible differs from knowing how much of each will form and survive. Separating fragment generation from intensity prediction in a forward model can be understood in this way.

<details markdown="1">
<summary>A closer look: the Arrhenius equation and RRKM theory</summary>

An introductory treatment of reaction kinetics often begins with the Arrhenius equation:

$$
k(T)=A\exp\!\left(-\frac{E_a}{RT}\right).
$$

Here, $$E_a$$ is the molar activation energy, $$R$$ is the gas constant, and $$T$$ is temperature. The equation helps explain the relationship between barriers and rates. However, a mass spectrometer's collision-energy setting cannot simply be substituted for $$RT$$. An ensemble's internal-energy distribution is not the same input as a thermal-equilibrium temperature.

RRKM theory is a statistical framework for unimolecular reactions of ions at a fixed internal energy $$E$$. Its basic form is:

$$
k(E)=\frac{N^{\ddagger}(E-E_0)}{h\rho(E)}.
$$

Here, $$E_0$$ is the threshold energy, $$\rho(E)$$ is the reactant density of states, $$N^{\ddagger}$$ is the number of transition-state states accessible at the available energy, and $$h$$ is Planck's constant. Assumptions about statistical energy redistribution are required, and not all fragmentation follows this model. A [study estimating gas-phase ion dissociation energies][rrkm-study] provides an application.

There is no need to implement an RRKM calculator before entering CASMI. The useful implication is that **dissociation rates depend on the states and internal energy of the whole molecule, not only on the energy of one bond**. A matching fragment mass alone cannot determine its intensity.

</details>

### Neutral losses: reading an undetected fragment from a mass difference

When an ion fragments into a charged product and a neutral product, the charged product is normally the one detected. The mass difference between precursor and product can therefore provide evidence about the **neutral loss**.

For a pathway that loses one neutral molecule while retaining the same single charge:

$$
m_{\mathrm{loss}}
\simeq(m/z)_{\mathrm{precursor}}-(m/z)_{\mathrm{fragment}}.
$$

If the two ions have the same charge magnitude but $$\lvert z\rvert>1$$, multiply the separation by $$\lvert z\rvert$$. If the charge changes or other reactions intervene, simple subtraction does not support the same interpretation.

| Example neutral species | Monoisotopic mass | A question to ask |
|---|---|---|
| Water, H₂O | Approximately 18.010565 Da | Does the composition and pathway allow dehydration? |
| Ammonia, NH₃ | Approximately 17.026549 Da | Can the structure account for this nitrogen- and hydrogen-containing loss? |
| Carbon monoxide, CO | Approximately 27.994915 Da | Is there a plausible pathway for losing this composition? |
| Carbon dioxide, CO₂ | Approximately 43.989829 Da | Can the structure support decarboxylation or another route to this loss? |

These numbers are **calculated masses for neutral compositions**. A pair of peaks separated by about 18.010565 Da does not establish the presence of an OH group at a particular position. Water can form through hydrogen transfers and rearrangements involving multiple atoms. Conversely, an OH-containing structure need not show an appreciable water-loss peak under every condition.

In CASMI, we can compare neutral-loss lists alongside fragment $$m/z$$ lists, or calculate the fraction of peaks that a candidate's elemental composition can explain. These are useful features to test with other evidence before turning them into hard rejection rules. Mass calculations use the NIST tables above and the [nitrogen isotope table][nist-nitrogen].

<figure class="casmi-figure" id="casmi-fig-7">
  <img src="/assets/img/casmi-2026-guide/ion-mass-balance.png?v=e7b2f99040e9" alt="Mass bookkeeping from a neutral molecule to a charged precursor and a possible water loss" width="1475" height="1195" loading="lazy">
  <figcaption><strong>Figure 7.</strong> A mass-balance example linking a neutral molecule, a protonated precursor, and a possible water-loss fragment. Atom composition and charge balance have been checked; this does not establish that the pathway is observed for eugenol. Directly interpreting peak separation as neutral-loss mass requires the same single charge.</figcaption>
</figure>

### Resolving power and mass accuracy describe different properties

**Mass resolving power** describes the ability to separate nearby peaks. With peak width defined as the full width at half maximum (FWHM), the expression is: [IUPAC definition][iupac-resolution]

$$
R=\frac{m}{\Delta m_{\mathrm{FWHM}}}.
$$

**Mass accuracy**, by contrast, concerns how close the measured peak center is to its true value. A narrow peak can still be shifted. High resolving power is therefore not, by itself, a reason to make a retrieval tolerance extremely narrow.

CASMI's <code>ms2_mzs</code> contains peak locations, not a raw peak profile and FWHM for every peak. Many decimal places do not establish measured resolving power. When comparing libraries, use available labeled data to examine precursor and fragment error distributions separately and choose matching tolerances accordingly.

### Different collision energies ask different questions about the structure

At lower energies, a molecule may retain more precursor and larger fragment ions. At higher energies, further dissociation can produce smaller fragments. An intermediate fragment can first become more abundant and then decline as it fragments again.

<figure class="casmi-figure" id="casmi-fig-8">
  <img src="/assets/img/casmi-2026-guide/collision-energy-series.png?v=ad3124ac9723" alt="Three synthetic MS/MS spectra illustrating changes with collision energy" width="1840" height="1539" loading="lazy">
  <figcaption><strong>Figure 8.</strong> Conceptual spectra for one hypothetical molecule at different energies. Each panel is normalized independently to a maximum intensity of 1, so the vertical axes cannot compare total ion abundance across conditions. These are neither measurements nor model predictions for a real substance.</figcaption>
</figure>

A candidate that explains large fragments at low energy may differ from one that explains smaller fragments at high energy. Agreement across conditions can provide stronger evidence. However, treating nearly duplicate spectra from similar conditions as independent evidence can exaggerate confidence.

Intensity normalization also matters. The competition's base-peak normalization can be written as:

$$
\widetilde I_i=\frac{I_i}{\max_j I_j}.
$$

This preserves the relative shape within a spectrum but removes the absolute intensity scale. Even when <code>base_peak_intensity</code> is supplied separately, it cannot directly represent concentration or prediction confidence without accounting for amplification, ionization efficiency, injection amount, and other effects. Both a sparse spectrum and a rich spectrum can have a maximum normalized peak of 1.

### Turning the background into a prediction problem

Let $$c$$ be the unknown structure, $$\mathcal S$$ the collection of spectra, and $$\Theta$$ the measurement conditions. Identification can be viewed as an inverse problem:

$$
p(c\mid\mathcal S,\Theta)
\propto p(\mathcal S\mid c,\Theta)\,p(c\mid\Theta).
$$

The first term on the right asks whether a structure could plausibly produce the measurements. The second describes its plausibility before considering those measurements. This is a conceptual framework, not the official scoring formula.

A forward model supplies evidence related to the first question. Spectral libraries and structure databases determine which candidates we compare. Preference for structures common in training can also introduce a bias resembling the second term. Validation should help distinguish **explaining a spectrum from merely preferring a familiar candidate**.

| Physical-chemistry observation | Consequence for CASMI design |
|---|---|
| Isomers with the same formula have the same exact mass | Evaluate mass-based candidate coverage separately from structure ranking |
| Ion composition and charge change measured mass | Preserve adduct-specific neutral-mass conversion and condition inputs |
| Fragmentation involves competing and secondary pathways | Separate peak presence from intensity; avoid relying on one bond-cutting rule |
| Neutral losses provide evidence about undetected products | Test fragment and neutral-loss scores together, without assigning a structure from one peak |
| Instrument and energy change the signal from the same structure | Aggregate conditions explicitly and validate across instruments |
| The maximum normalized intensity is always 1 | Keep relative spectral shape distinct from original measurement quality |
| Only information actually supplied can be used | Check whether full MS1 isotope patterns, CCS, or retention time exist in the input |

The input columns now have a physical meaning: they contain evidence gathered under particular conditions. Next we examine which molecules and instruments the training data cover, then connect that evidence to candidate retrieval and ranking.

## 6. What is inside 2.5 million training spectra?

The official description gives approximately 2.5 million spectra and 275,000 structures, combining public libraries with Enveda data. [Competition data][data]

The local file audited for this project contained **2,539,608 rows** and **275,810 structures according to the supplied structure keys**. That structure count is an aggregation of the keys already in the file. It should not be assumed to equal a count obtained after rerunning the metric's normalization on every structure.

<figure class="casmi-figure" id="casmi-fig-9">
  <img src="/assets/img/casmi-2026-guide/training-sources.png?v=3d1d55cee9fd" alt="Training spectrum counts grouped by source library" width="1876" height="1189" loading="lazy">
  <figcaption><strong>Figure 9.</strong> Aggregation of <code>ingest_lib</code> in the local training file on September 27, 2026. Bars count spectra, not molecules. The Other libraries group combines SpectraVerse, MS-DIAL, drug_plus, natural-product examples, and Masaryk data.</figcaption>
</figure>

Enveda-180 is the largest source: approximately 1.15 million spectra, or about 45% of the total. [Enveda's public processing code][enveda180] provides useful context for its construction and processing.

A large row count does not automatically imply sufficient information about the target molecules. A structure can be measured at multiple collision energies, and sources differ in both chemistry and instrumentation. One hundred spectra from one hundred molecules carry different information from one hundred spectra covering only ten molecules.

### Matching the instrument does not guarantee matching the chemistry

Enveda data are attractive because the hidden test is measured on timsTOF. However, a substantial part of Enveda-180 covers synthetic compounds, whose structural distribution need not match the natural products and analogs of interest here. Other libraries can add natural-product diversity while differing in instruments and collision-energy conditions.

We need to consider both:

- **Measurement shift:** the same structure can produce different spectra across instruments, energies, and adducts.
- **Chemical-space shift:** unfamiliar scaffolds and substructures can appear even when the instrument is unchanged.

Weighting every library in proportion to its row count can let the largest source dominate training. Repeating only a small natural-product subset can instead encourage overfitting. Sampling by molecule and source is worth testing, but its effect must be established through structure-level validation.

The training file has two roles. Its measured spectra paired with known structures are immediately useful as search references, and those same pairs can train a model of the relationship between spectra and structures. This is why retrieval is a sensible first step even when a large training set is available.

## 7. Build the candidate pool through retrieval and analogs

**Library search** is a straightforward starting point. Compare the query with reference spectra of known structures, then retrieve the structures associated with similar measurements.

### Match peaks before calculating similarity

A simple approach matches peaks at nearby $$m/z$$ values and calculates cosine similarity. For aligned intensity vectors $$x,y$$:

$$
\operatorname{cosine}(x,y)
=\frac{x\cdot y}{\lVert x\rVert_2\lVert y\rVert_2}.
$$

Peak positions will rarely coincide exactly, so matching within a tolerance comes first. We can also take square roots of intensities to reduce the dominance of large peaks, or compare **neutral losses**, the mass differences from precursor to fragment. The [Fast Spectral Cosine baseline][cosine-notebook] and [matchms][matchms] are useful implementation references.

The method is interpretable: a reference spectrum can explain why a structure was retrieved. Its limitation is equally clear. If no spectrum of the correct structure exists in the library, direct same-structure retrieval cannot return it.

### Narrowing structure candidates by mass

This is where structure databases such as [COCONUT][coconut] and [PubChem][pubchem] become useful. Estimate neutral mass from precursor m/z and adduct, then collect structures within a tolerance.

Mass error in parts per million is:

$$
\operatorname{ppm\ error}
=10^6\frac{m_{\mathrm{candidate}}-m_{\mathrm{query}}}{m_{\mathrm{query}}}.
$$

At 164.083730 Da, ±10 ppm corresponds to approximately ±0.001641 Da. This illustrates the scale of a mass tolerance; it is not a recommended optimum for this competition. A window that is too narrow can discard the answer because of measurement error or incorrect adduct handling. A wide window increases candidate counts and computation.

Passing the mass filter makes a structure eligible for comparison; it does not identify it. Isomers such as eugenol and isoeugenol pass together.

### Using spectra from related molecules: analog propagation

Even when the correct molecule has no measured spectrum, structurally related molecules may have reference spectra. **Analog propagation** first finds references with spectra similar to the query, then transfers support to structurally similar candidates. The [public baseline][analog-notebook] provides an implementation.

A simplified expression for the idea is:

$$
S_{\mathrm{analog}}(c\mid q)
=\sum_{a\in\mathcal N(q)}
w(q,a)\,T\!\left(f(c),f(a)\right).
$$

This is not a claim that the public implementation uses exactly this equation. Here, $$q$$ is the query spectrum, $$a$$ is a similar reference molecule, and $$c$$ is a candidate structure. The weight $$w$$ comes from spectral similarity. The function $$f$$ encodes a structure as a bit-vector **fingerprint**, and $$T$$ measures Tanimoto similarity between fingerprints.

A fingerprint summarizes many local structural features as bits. If $$A,B$$ are the sets of active bits, Tanimoto similarity is the intersection divided by the union:

$$
T(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

The approach can assign scores to structures without measured spectra. However, the closest spectrum need not come from the closest molecular structure. Analog support is therefore naturally treated as one source of evidence among several.

Retrieving candidates does not finish the decision. If eugenol and isoeugenol both remain, we can next ask which structural features the unknown spectrum supports.

## 8. Predict structural evidence for each candidate

Once we have candidates, we can learn which substructures the query spectrum suggests. A **spectrum-to-fingerprint model** predicts the probability of each bit being active, then compares those predictions with candidate fingerprints. [MIST][mist] is a useful research implementation for this direction.

For example, strong predictions for an aromatic feature and an oxygen-containing substructure can favor candidates containing those features. The output is a set of structural probabilities, not one molecular name.

Let $$p_j$$ be the predicted probability for bit $$j$$ and $$f_j(c)\in\lbrace 0,1\rbrace$$ the candidate's bit value. A score assuming independent Bernoulli bits can be written as:

$$
\begin{aligned}
S_{\mathrm{FP}}(c)&=\sum_j f_j(c)\log p_j\\
&\quad+\sum_j(1-f_j(c))\log(1-p_j).
\end{aligned}
$$

Real fingerprint bits are not independent. This is a useful scoring model, not a proof of an exact chemical probability. Calibration, rare-bit handling, and training that distinguishes similar isomers can all affect performance.

### Learning spectral representations first

Rather than starting directly with structure labels, a model can first learn shared patterns from many spectra. [DreaMS][dreams] provides pretrained spectral representations. These can support retrieval embeddings or a model that predicts structural fingerprints.

A useful representation does not automatically improve the final candidate ranking. Spectral similarity, structural similarity, and exact candidate correctness are different objectives. The representation must be tested against the last question on the candidates used in our validation.

### Combining evidence into a ranking

For each candidate, we can collect several kinds of evidence.

| Score or feature | What it asks |
|---|---|
| Direct spectral similarity | Does the query resemble an actual reference spectrum of this structure? |
| Analog support | Do reference molecules with similar spectra support this candidate? |
| Fingerprint score | Does the structure match the features predicted from the spectrum? |
| Mass and adduct consistency | Is the candidate compatible with the precursor information? |
| Agreement across spectra | Does support persist across energies and ionization conditions? |

We can combine these with a weighted sum or train a **ranker**, a model that orders candidates. Its training needs difficult negatives: structures with a matching mass but incorrect connectivity. Random negatives with entirely different masses will not teach the model to resolve the ambiguity it faces at inference.

Public notebooks such as [Two Rankers, One Engine][two-rankers] illustrate how these signals can be joined in one inference path. The useful questions are how candidates are formed and which features change their order.

A fingerprint summarizes many structural features, but similar candidates may receive similar fingerprint scores. We can turn the question around: if each candidate were the answer, what spectrum should it produce?

## 9. Ask whether a candidate explains the observed spectrum

So far, we have reasoned from spectra toward structures. The reverse direction is also useful: if a candidate were correct, which fragments should appear? Predict its spectrum and compare that prediction with the observation.

This is a **forward model**, mapping structure to spectrum. [ICEBERG and GLACIER][ms-pred] are public research implementations. Predictions of fragments and relative intensities can provide additional ranking evidence.

$$
\begin{aligned}
\widehat{s}_c&=g(c,\theta),\\
S_{\mathrm{forward}}(c)&=\operatorname{sim}(s_{\mathrm{observed}},\widehat{s}_c).
\end{aligned}
$$

Here, $$\theta$$ collects measurement conditions such as adduct, collision energy, and instrument.

Return to eugenol and isoeugenol. Mass cannot order them, but the agreement between each candidate's predicted fragments and the observation can add information. Improving **ranking within the same molecular formula** is a concrete reason to test a forward model.

Predicted spectra also contain errors. A mismatch between training and target instruments, adducts, or energies can cause the model to miss precisely the peaks that distinguish the candidates. A reasonable first comparison is therefore to add forward similarity as a feature and inspect the per-molecule rank changes.

<figure class="casmi-figure" id="casmi-fig-10">
  <img src="/assets/img/casmi-2026-guide/evidence-routes.png?v=7755eeda6c70" alt="Direct search, inverse fingerprint prediction and forward spectrum prediction as complementary evidence routes" width="1507" height="1318" loading="lazy">
  <figcaption><strong>Figure 10.</strong> Three directions of evidence for candidate structures. Direct search compares measurements, inverse prediction estimates structural features, and forward prediction estimates the measurement a candidate would produce. This is not a performance comparison.</figcaption>
</figure>

**What does each approach contribute, and where can it fail?** Comparing the three directions with analog propagation and generation clarifies what each experiment is testing.

| Approach | Evidence or candidates added | Failure to examine | Results to record first |
|---|---|---|---|
| Direct spectral search | Agreement with an actual reference measurement | No correct reference, or a large condition mismatch | Rank and peak agreement, separated by reference availability |
| Analog propagation | Candidates supported by related reference structures | Spectral similarity does not imply structural similarity | Candidate Recall@K and structural relationships among errors |
| Fingerprint prediction | Substructures suggested by the observation | Isomers share similar fingerprint features | Ranking among same-formula candidates |
| Forward prediction | How well a candidate explains the observed spectrum | Unsupported conditions or inaccurate fragments and intensities | Per-molecule rank changes and added runtime |
| De novo generation | Structures outside a fixed database | Valid SMILES do not guarantee the correct structure | Exact-key generation rate and damage to existing correct ranks |

This table provides comparison criteria. It is not a measured performance or runtime ranking of particular models.

### Expensive calculations need not run on every candidate

Predicting spectra for hundreds of thousands of structures can be costly. Fast retrieval and fingerprint scores can first reduce the pool, with more detailed computation reserved for the top K.

As a **design example**, 400 molecules with 100 candidates and 3 observation conditions each would require up to 120,000 candidate–condition comparisons. This is not an exact workload forecast for the hidden test. The actual number depends on spectra per molecule and whether repeated candidate–condition computations can be cached.

Evaluate added runtime and memory alongside MRR. A method that exceeds the execution limit cannot be used unchanged in the submission, even if it is accurate.

These routes still evaluate structures already proposed. If no database contains the answer, the system also needs a way to propose a new structure that explains the observations.

## 10. Molecules outside the database: the role of de novo generation

For Class 3, PubChem and COCONUT do not supply the correct structure. When our candidate pool lacks it, we need a route that can propose structures beyond that pool. **De novo structure generation** can generate SMILES from spectra, search molecular graphs under a formula constraint, or modify existing candidates.

[MassSpecGym][massspecgym] treats retrieval, de novo generation, and spectrum simulation as separate tasks. [FOAM][foam] illustrates formula-constrained structure search with a spectrum predictor used to evaluate candidates.

A generated candidate must pass several different checks:

1. **Can it be parsed?** Does RDKit accept the SMILES?
2. **Does its mass and formula fit?** Is it consistent with precursor information?
3. **Is it chemically plausible?** Does it avoid inappropriate structures or bonding?
4. **Does it explain the spectrum?** Does the fragment evidence support it?
5. **Is its connectivity correct?** Does its scoring key match the target?

A high success rate at the first step can coexist with a low success rate at the last. A structure with a very similar fingerprint still earns no credit if its scoring key is wrong.

### Deciding where a generated candidate belongs

Automatically inserting a generated structure at rank 1 can displace a correct answer. If a false candidate moves a correct rank-1 answer to rank 2, that molecule's contribution falls from 1 to 0.5.

Appending candidates to unused slots while preserving the original order leaves earlier correct ranks unchanged. Exceeding 25 candidates or inserting a candidate in the middle has different consequences. A generation method therefore needs validation of both candidate quality and **whether its candidates deserve to displace existing ones**.

Before training a large generator, I want to determine whether candidate omissions or ranking errors dominate. If many answers already enter the pool but lose to incorrect same-formula structures, improving ranking is a more direct next experiment.

Combining retrieval, prediction, and generation creates more ways to obtain a strong local score. We still need to distinguish learning to interpret new molecules from having already seen their answers. The next section defines a comparison that can separate those explanations.

## 11. The most important preparation: did we actually hide the answer?

Validation matters as much as modeling here. One molecule can have multiple spectra, and the same structure can appear in several libraries.

Suppose spectrum rows are split randomly. A molecule's 20 eV spectrum might enter training while its 40 eV spectrum enters validation. A strong result could then show recognition of an already seen structure under a different condition, rather than identification of an unseen structure.

That is a valid Class 1 question, but it does not answer the Class 2 or Class 3 question. Define the intended evaluation first.

<figure class="casmi-figure" id="casmi-fig-11">
  <img src="/assets/img/casmi-2026-guide/validation-design.png?v=737879241d8a" alt="Spectrum-row splitting contrasted with holding out all spectra sharing a molecular structure key" width="1630" height="1212" loading="lazy">
  <figcaption><strong>Figure 11.</strong> Illustrative structures A and B at two collision energies. Splitting rows can leave the same structure on both sides. The right panel holds out all spectra of A. For Class 2, exclude its reference spectra and supervised training labels while retaining its structure as a candidate.</figcaption>
</figure>

### Class 2 validation treats structures and spectra differently

To approximate Class 2, first select evaluation structures and remove **all reference spectra of those structures** across every library. Exclude their structure labels from supervised fingerprint-model and ranker training as well.

However, **leave the correct structures in the candidate database**. That reproduces the situation where a structure is public but its measured spectrum is unavailable. For a Class 3 test, also remove the correct structure from the fixed retrieval pool and evaluate the generation path.

| Validation objective | Reference spectra of the target structure | Correct structure in the candidate pool | Ability being tested |
|---|---|---|---|
| Retrieval resembling Class 1 | Separate measurements allowed; identical observations excluded | Included | Recognition across measurement conditions |
| Retrieval and ranking resembling Class 2 | Excluded across all libraries | Included | Interpretation of spectra from unseen structures |
| Generation resembling Class 3 | Excluded | Excluded from the fixed retrieval pool | Proposal and ranking of new structures |

Group structures using the same normalization procedure as the metric. Splitting by raw SMILES strings or excluding only the <code>enveda-np-examples</code> library is insufficient: the same structure can remain elsewhere. The [discussion of natural-product cross-validation][np-cv] identifies this issue directly.

### Auxiliary models can also have seen the answer

Removing the answer from the retrieval library does not ensure that a pretrained model has never learned it. Consider fingerprint predictors, spectrum predictors, rankers, and calibration models.

When a model's training-structure list is unavailable, its results can still help check functionality and compare alternatives. They are harder to interpret as generalization to entirely unseen structures. Record model provenance and training splits.

A stronger test can also hold out related scaffolds. Because that imposes a stricter condition than holding out identical structures, it is best interpreted as a separate robustness evaluation.

### Inspect observation counts and individual molecules too

Giving validation molecules ten observations when the actual task supplies only a few can make performance look optimistic. Use the official range of 1–16 spectra per molecule and median of 3 as context for testing sensitivity to fewer observations. Those summary statistics do not reconstruct the full hidden-test distribution.

Save more than a single mean score:

- Whether the correct answer entered the pool, and its final rank.
- Whether higher-ranked errors share the target formula, and which features favored them.
- Results by adduct, observation count, mass range, and source.
- Molecules improved or worsened by the new method.
- Total runtime and the stage with the highest memory use.

Comparing methods on the same molecules makes small average differences easier to investigate. When retraining a ranker, check variation across random seeds. Resampling **molecules rather than spectrum rows** also aligns uncertainty estimates with the scoring unit.

## 12. Where does the public work stand now?

The public starting point has progressed well beyond an isolated nearest-spectrum search. Several reusable components now fit together: a library of measured spectra, structure candidates from training data and COCONUT, predictors of structural fingerprints, and learned models that combine the resulting evidence. The question for a newcomer is which part to reproduce first and which remaining error to investigate.

### Read the leaderboard as a dated measurement

The public leaderboard archive downloaded on **September 27, 2026, at 02:59:30 UTC** contained **1,608 scored team entries**. The table below summarizes that archive; it is not a live counter. [Public leaderboard][leaderboard]

| Observation in that snapshot | Value | How to interpret it |
|---|---:|---|
| Highest public MRR@25 | 0.425 | A reference for the public frontier; the number alone does not reveal the method. |
| Second and third public scores | 0.421 / 0.412 | Several teams were above 0.40. |
| Median team score | 0.292 | Half the downloaded team entries were at or below this value. |
| Teams scoring from 0.335 through 0.342, inclusive | 263 | Many entries occupy a narrow band near the public baselines. This count does not prove they use the same code. |

An MRR of 0.425 does **not** mean that 42.5% of molecules were identified at rank 1. It averages reciprocal ranks, so different combinations of first-place answers, lower-ranked answers, and misses can produce it. Nor does the public ranking tell us how the private ranking will turn out. For a small public-score difference, the useful follow-up is to ask which molecules changed in a held-out comparison, not to infer a new scientific capability from the decimal alone.

### What the public notebooks actually add

The notebook score panels below were checked during the same September 27 review. Scores are tied to the versions indicated; they are not a controlled comparison in which only one component changed. In particular, a notebook's **Best Score** may belong to an older version than the source currently displayed.

| Public notebook and checked version | Displayed public score | Contribution to the story |
|---|---:|---|
| [Analog Propagation][analog-v10], best-scoring V10 | 0.335 | A shared foundation: retrieve measured spectra, extend evidence through related structures, and combine it with learned features. The page displayed V32 at review time; 0.335 belongs to V10. |
| [Fast Spectral Cosine baseline][cosine-v17], V17 | 0.339 | Shows how processing choices and an inherited multi-channel pipeline develop beyond the simple-search name. |
| [Two Rankers, One Engine][two-rankers-v1], V1 | 0.337 | Makes alternative rankings and their combination explicit within a common candidate engine. |
| [Evgen Dvorkin's Enveda CASMI notebook][evgen-notebook], V15 | 0.342 | A public two-ranker variant that provides a practical reproducible starting point. |
| [Ahmed Beratozer's v3 inference][ahmed-v3], V6 | 0.358 | Adds a separately controlled PubChem candidate route to a richer engine. Some required inputs are private. |
| [Ahmed Beratozer's v4f inference][ahmed-v4f], V1 | 0.362 | Includes richer ranking evidence and ICEBERG rescoring. Private inputs limit independent reproduction from the notebook alone. |

These examples suggest a progression: **recognize what is already measured → extend to known structures without reference spectra → distinguish increasingly similar candidates**. They do not establish a causal gain for every added component. A higher-scoring notebook can change its training data, candidate set, features, and model together. The 0.362 result, for example, is evidence about that submitted pipeline, not a measured “ICEBERG improvement” by itself.

The public code also makes clear why a model name is an incomplete recipe. In a notebook's Input tab, structure collections supply possible answers; fingerprint weights turn spectra into structural evidence; simulated ranking rows teach the order of candidates; package files make offline execution possible. Reproducing the combination requires compatible versions of those inputs. Reading code that depends on private weights can still teach an idea, but it does not supply a runnable baseline.

### What seems to be holding progress back?

A [participant's account of roughly 30 submissions][thirty-submissions] describes frequent cases where the correct structure is available but a same-formula candidate wins instead. The [discussion of score plateaus][plateau] also reports examples where extra features or models produced little improvement. These are participant observations under their own experimental conditions, not a measurement of the entire hidden set.

Our eugenol/isoeugenol example shows the kind of decision involved. A mass filter can admit both; broad structural features may support both; the useful additional signal is something that changes their relative order for the right reason. A larger database can rescue a missing answer, but if it mostly adds close distractors, the ranker faces a harder task. Better candidate coverage and better ordering must be measured separately.

This is why I would treat the public 0.34 region as a **working baseline to understand**, rather than a score to imitate through repeated parameter changes. The leading public scores leave room above it. They do not tell us whether that room comes from candidate coverage, ranking, generation, or a combination that leading teams have not disclosed.

### Where this project is, and what remains unproven

In this project, a submission based on the public two-ranker engine has completed with **public MRR@25 = 0.342**. That is a reproduced baseline, not a newly demonstrated modeling improvement. This value was checked against the project submission record and Kaggle’s completed-submission response.

The next unresolved task is a trustworthy local comparison, especially for Class 2: remove a held-out structure's reference spectra and supervised training exposure while retaining the structure in the candidate pool. The project has not yet established an independently validated gain from a new forward-model, ranking, or generation method. This distinction places us at a useful starting point: the system runs and has a hidden-evaluation reference score; now we need evidence that can tell us which change deserves to become the next version.

### Read resources in the order of the decision they support

| Resource | Inspect first | Question for this project |
|---|---|---|
| [Fast Spectral Cosine baseline][cosine-notebook] | Its library-search component: peak processing and mass indexing | How is direct retrieval implemented? The full notebook score includes other channels. |
| [Analog Propagation baseline][analog-notebook] | Candidate pools, analog support, and fingerprint scores | How are unmeasured structures scored? |
| [Two Rankers, One Engine][two-rankers] | Score fusion and molecule-level outputs | How does each source of evidence affect the final order? |
| [MassSpecGym][massspecgym] | Retrieval, de novo, and simulation tasks and splits | How are the different abilities evaluated separately? |
| [MIST][mist] · [DreaMS][dreams] | Spectral representations and structural-feature prediction | Can learning add information missing from retrieval? |
| [ICEBERG/GLACIER][ms-pred] | Candidate spectrum prediction and condition inputs | Can same-formula candidates be distinguished more reliably? |
| [FOAM][foam] | Formula-constrained search and model-call budgets | Can useful new structures be proposed within limited computation? |

## 13. Runtime and external resources: completing the inference system

This is a code competition. Kaggle reruns the notebook on hidden data, so creating a CSV for the visible placeholder is not sufficient. The official requirement is a **CPU or GPU notebook that runs without internet access, finishes within 9 hours, and writes <code>submission.csv</code>**. [Code requirements][competition]

This rules out live PubChem queries or downloading weights during inference. Structure lists, weights, libraries, and packages need to be prepared in advance and attached as notebook inputs.

### Reducing computation in stages

Use mass and adduct to restrict candidates, sort with fast scores, then apply expensive models to a narrower set. Cache query-independent quantities such as structural fingerprints and exact masses. Reuse shared calculations when multiple spectra evaluate the same candidates.

Caches also need provenance. Record the candidate list and normalization version used to produce them so that incompatible files are not combined. Measure data loading, index construction, and final CSV checks alongside model inference.

### Public availability and competition eligibility are different

External resources may be allowed, but code licenses, weight licenses, and training-data terms can differ. The organizers place conditions on resources that restrict commercial use or reproduction of winning solutions. Their [pretrained-model clarification][pretrained-rules] distinguishes appropriately licensed models trained on MassSpecGym from models derived from restricted resources such as NIST or METLIN.

An MIT license on a GitHub repository therefore does not establish that every linked checkpoint is suitable. Check the license and training provenance of the **exact weight file** being considered. In particular, an [earlier permission concerning CFM-ID][cfmid-general] is distinct from a [later question about the METLIN training provenance of CFM-ID 4's default models][cfmid-question]. The later question had no organizer response at this article's verification date.

Competition data are subject to CC BY-NC terms and restrictions on redistribution outside participants. Derived datasets prepared for experiments also require attention to their sharing scope. Spectral figures in this article use synthetic values; only source-level aggregates are shown from training data. The governing references are the [competition rules][rules] and [organizer announcement][welcome].

## 14. How should the first experiment begin?

For a newcomer, a small search example and a competitive public-engine reproduction serve different purposes. Use the first to understand inputs and scoring, then use the public two-ranker from Section 12 as an identifiable comparison baseline. This project has already submitted that baseline; its next milestone is independent validation and failure analysis.

### Step 1: check inputs and the answer definition

There is no need to load the entire training set into memory at the start. Parquet is column-oriented, so we can inspect its schema and a small batch. This example assumes the local project has <code>data/train.parquet</code>; in Kaggle, substitute the actual path of the attached input.

```python
from pathlib import Path
import pyarrow.parquet as pq

path = Path("data/train.parquet")
pf = pq.ParquetFile(path)
print("rows:", pf.metadata.num_rows)
print("columns:", pf.schema_arrow.names)

columns = ["normalized_smiles", "precursor_mz", "adduct",
           "ms2_mzs", "ms2_normalized_intensities"]
batch = next(pf.iter_batches(batch_size=8, columns=columns))
sample = batch.to_pydict()
for mzs, intensities in zip(sample["ms2_mzs"],
                           sample["ms2_normalized_intensities"]):
    assert len(mzs) == len(intensities)
    print("peaks:", len(mzs), "base peak:", max(intensities))
```

Check column names, array lengths, missing values, and normalization ranges. This also makes the distinction visible: training data contain structures, while test structures must be predicted. Then check the scoring-key behavior from Section 2.

Alongside the environment setup, check the [training-data correction notice][train-update]: some samples gained water-loss adducts. It is a concrete example of a data update that can affect neutral-mass conversion and candidate filtering.

### Step 2: fix the validation molecules and candidate pool

Save the evaluation molecule list and split definition first. For a Class 2-like panel, remove target spectra across all libraries while retaining the structures as candidates. Start with a small panel relevant to natural products and timsTOF, then check a broader set of structures to avoid optimizing only that panel.

Measure candidate Recall@K next. If correct answers are absent, investigate mass tolerances, adduct handling, database coverage, and normalization before changing the ranker.

### Step 3: establish retrieval and ranking baselines

Run a baseline combining direct search, analog support, and fingerprint scores. Remove one score at a time to see what contributes. Keep candidates and validation molecules fixed so that the differences are interpretable.

The output should include per-molecule correct ranks and leading incorrect candidates, alongside mean MRR. Inspecting molecules whose answer ranks second can provide a concrete reason to improve ranking features.

### Step 4: add one hypothesis

One experiment I am considering is adding forward-model scores for difficult same-formula candidates. Compare the existing ranking with a ranking that adds forward features to the same candidates. Changing the pool and the observation spectra at the same time would obscure the cause of any improvement.

| Observed failure | Change to test next | Measure alongside it |
|---|---|---|
| The answer is absent from the pool | Candidate databases, mass windows, and adduct handling | Recall@K and candidate counts |
| The answer is present but loses to same-formula errors | Forward scores and ranker training with difficult negatives | Per-molecule rank changes and runtime |
| Performance drops sharply with few observations | Condition-aware score aggregation and training sampling | Performance with one, a few, and all observations |
| A particular ionization condition performs poorly | Condition-specific processing and model support | Candidate omissions and ranking errors under that condition |
| The correct structure cannot be included in the retrieval pool | Bounded de novo search | Exact-key generation rate and harm to existing candidates |

Consider a hidden-test submission after the change helps validation and fits the runtime limit. Successfully running the public placeholder only establishes the format-checking part of this process.

### The approach I would prioritize now

My working choice is a **retrieval-centered system with learned ranking and selective spectrum prediction**. It gives each method a concrete job: direct matches handle familiar molecules; analog and fingerprint evidence extend the search to known but unmeasured structures; a forward model is tested where similar candidates remain hard to separate. Structure generation stays a separate branch for cases the fixed databases cannot cover.

That choice follows the public evidence in Section 12, but it still needs validation on our own molecules. I would first preserve the reproduced public engine as a control, establish a trustworthy Class 2-like holdout, and inspect its leading errors. If same-formula ranking errors dominate, the next comparison is a forward-model feature on the **same candidate pool**. If missing candidates dominate instead, database coverage and candidate construction take priority. The experiment decides which part deserves more effort.

The first changes should also be small enough to interpret. Adding a larger pool, a new fingerprint model, a new ranker, and generation at once can move the score while leaving us unable to explain why. A useful improvement should identify the molecules it recovers, the molecules it harms, and the additional runtime it needs.

### A first week organized around answers

The following is a proposed sequence of milestones, not a promise that training or data preparation fits a fixed number of days.

| Milestone | A concrete result to keep | Question answered |
|---|---|---|
| Read one molecule end to end | Its conditions, spectra, candidate structures, normalized keys, and final row. | Do we understand what the pipeline is predicting? |
| Run the unchanged public baseline | A pinned notebook/input version and an execution record. | Can we reproduce the procedure? |
| Build the held-out comparison | Frozen molecule IDs, separate spectral and structure pools, baseline per-molecule ranks. | Is our local score testing the intended kind of unknown? |
| Inspect the failures | Candidate misses, same-formula errors, and condition-specific regressions. | Which stage should change first? |
| Test one change | Paired rank changes, MRR, candidate recall, and total runtime. | Is the new evidence useful enough to keep? |

## 15. How I expect the competition to develop

The public work already contains more than a simple spectral lookup: candidate expansion, learned structural features, and multiple ranking signals are available. My expectation is that the next useful gains will depend increasingly on **which uncertainty a method resolves**. The following are possible developments, not reports of undisclosed leading architectures.

### First, shared baselines make small variations less informative

When many notebooks inherit the same candidate pool, fingerprint weights, and ranking features, changing seeds or blend weights can reshuffle a few close decisions without introducing new chemical evidence. Such changes can still help, but a small public-score gain alone does not tell us whether the improvement will persist on the private split.

I expect stronger comparisons to ask which held-out molecules improve across several observation-count and source conditions. If a gain disappears when the reference library or validation source changes, the next problem is transfer. If it persists across those comparisons, it becomes a more credible candidate for final selection. This is why keeping the baseline identifiable matters as the notebooks evolve.

### Next, distinguishing close structures becomes more valuable

If the correct answer is already in a manageable candidate pool, the opportunity is to separate it from chemically plausible alternatives. A **hard negative** is an incorrect candidate that is difficult to distinguish—for example, a molecule with the right mass and formula but different bond connectivity. Training against those alternatives asks a more relevant question than separating the answer from arbitrary unrelated molecules.

Condition-aware spectrum prediction, better use of neutral losses, and fingerprint models trained without evaluation-structure overlap could provide additional evidence. They could also repeat information the ranker already has. The evidence for this direction would be consistent improvements in the ranks of same-formula candidates, with acceptable costs and without undoing reliable reference matches. The participant reports in Section 12 make this worth testing; they do not establish that adding a named forward model is sufficient.

### Generation becomes useful when it supplies answers the rest cannot reach

The absence of Class 3 answers from PubChem and COCONUT gives structure generation a clear role: propose answers that a pool built from those databases cannot supply. Its competitive value depends on generating the exact accepted structure, recognizing when that generated candidate deserves a high rank, and completing the search within the inference budget. A chemically valid string alone achieves none of those three goals.

I would expect generation to be tested first with a limited search budget and an explicit comparison of recovered answers against displaced good candidates. If Class 3-like validation shows useful exact-key recovery and reliable ranking, it deserves a larger share of computation. If it mostly produces plausible but incorrect structures, better retrieval and ranking remain the more productive investment for the cases they can solve. We do not know the hidden class mixture well enough to turn that trade-off into a fixed allocation in advance.

### What could change this plan?

A new well-documented candidate resource could change coverage. A permitted checkpoint with useful measurement-condition support could change the cost of spectrum prediction. Data corrections or organizer rulings could change which inputs should be used. The training-data update and the checkpoint-specific CFM-ID question are examples of developments that deserve a targeted recheck. [Training update][train-update] · [Model clarification][pretrained-rules] · [Open CFM-ID question][cfmid-question]

Each such event should lead to a specific comparison while preserving the previous control. My present bet is on a system that **knows when a library match is strong, distinguishes close candidates better, and spends expensive computation on the unresolved cases**. I would revise that bet if the held-out evidence showed that candidate coverage or generation, rather than ordering, accounted for most recoverable errors.

## 16. Dates and the first milestone


The official deadlines are all at 23:59 UTC, which is the following morning in Korea. [Competition timeline][competition]

| Event | UTC | Korea Standard Time |
|---|---|---|
| Entry and team-merger deadline | December 7, 2026, 23:59 | December 8, 08:59 |
| Final submission deadline | December 14, 2026, 23:59 | December 15, 08:59 |

At the verification date, the rules allow 5 submissions per day, 2 final selections, and teams of up to 5 members. Total prizes are $50,000. Recheck the [rules][rules] and schedule announcements before entering or making final selections.

The first experiment should show **which molecules retrieval solves, which suffer ranking errors, and which lack the correct candidate altogether**. That distinction gives us a reason to choose the next model. CASMI offers a way to learn how to connect incomplete measurements with limited chemical knowledge and propose structural candidates that can be tested.

---

## References and what was checked

The structure and mass calculations, scoring-key examples, and illustrative MRR calculation were checked with RDKit 2026.03.3. Isotope-mass sums, adduct and neutral-loss differences, DBE, and center-of-mass energy examples were also checked numerically. The Parquet example was checked on a small batch from the local training file.

Reaction-energy profiles and spectra at different energies are conceptual illustrations, not experimental measurements or quantum-chemical calculations for a real molecule. The neutral-loss diagram is a mass-balance example; the input, evidence, error, and split diagrams summarize the explanation. All body figures can be regenerated from code, while the supplied official header was edited separately for the cover.

The leaderboard distribution was calculated from the full public ranking downloaded on September 27, 2026, at 02:59:30 UTC. Six notebook scores and their version links were checked in the actual score panels that day. Team best scores were not substituted for notebook results; the comparison is not a component-level ablation. The project’s 0.342 comes from a baseline submission completed before this editorial revision.

Descriptions of public research models are based on their authors' code and documentation. No model training or Kaggle submission was performed for this article. Illustrative scores are labeled separately from aggregate dataset statistics.

Further reading, grouped by purpose:

- **Competition reference:** [Overview, evaluation, and timeline][competition], [input data][data], [official metric][metric], and [rules][rules].
- **Physical-chemistry foundations:** [NIST isotope masses][nist-carbon], [LC/MS and ionization][agilent-lcms], [CID definition][iupac-cid], [mass resolving power][iupac-resolution], and [formula constraints and their limitations][golden-rules].
- **Fragmentation in more depth:** [Protonation and small-molecule CID prediction][protonation-study], [MassKinetics on energy transfer, kinetics, and observation][masskinetics], and [an application of RRKM][rrkm-study].
- **Understanding the data:** [Enveda-180 processing code][enveda180], [COCONUT][coconut], [PubChem][pubchem], and [MassSpecGym][massspecgym].
- **Retrieval and structural features:** [Analog Propagation][analog-notebook], [Spectral Cosine][cosine-notebook], [matchms][matchms], [MIST][mist], and [DreaMS][dreams].
- **Detailed ranking and generation:** [ICEBERG/GLACIER][ms-pred] and [FOAM][foam].
- **Validation and practical discussions:** [Structural duplication in natural-product CV][np-cv], [plateaus across variants][plateau], [ranking lessons from submissions][thirty-submissions], and [conditions for pretrained resources][pretrained-rules].


[competition]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview
[data]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data
[rules]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
[metric]: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
[welcome]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359
[train-update]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471
[metric-discussion]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274
[pretrained-rules]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742991
[cfmid-general]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741912
[cfmid-question]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774
[np-cv]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741597
[plateau]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742055
[thirty-submissions]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254
[analog-notebook]: https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline
[cosine-notebook]: https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline
[two-rankers]: https://www.kaggle.com/code/megayak/casmi26-two-rankers-one-engine
[massspecgym]: https://github.com/pluskal-lab/MassSpecGym
[mist]: https://github.com/samgoldman97/mist
[dreams]: https://github.com/pluskal-lab/DreaMS
[ms-pred]: https://github.com/coleygroup/ms-pred
[foam]: https://github.com/coleygroup/foam
[matchms]: https://github.com/matchms/matchms
[enveda180]: https://github.com/enveda/enveda-180
[coconut]: https://coconut.naturalproducts.net/
[pubchem]: https://pubchem.ncbi.nlm.nih.gov/
[eugenol]: https://pubchem.ncbi.nlm.nih.gov/compound/Eugenol
[isoeugenol]: https://pubchem.ncbi.nlm.nih.gov/compound/Isoeugenol
[nist-carbon]: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=C
[nist-hydrogen]: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H
[nist-oxygen]: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=O
[nist-nitrogen]: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=N
[agilent-lcms]: https://www.agilent.com/en/product/liquid-chromatography-mass-spectrometry-lc-ms/lcms-fundamentals/lcms-instrument-types
[agilent-ionization]: https://www.agilent.com/library/technicaloverviews/Public/5990--7413EN.pdf
[iupac-cid]: https://goldbook.iupac.org/terms/view/C01167/pdf
[iupac-resolution]: https://www.old.goldbook.iupac.org/html/R/R05318.html
[golden-rules]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1851972/
[protonation-study]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11492807/
[masskinetics]: https://pubmed.ncbi.nlm.nih.gov/11312517/
[rrkm-study]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6031295/
[bruker-timstof]: https://www.bruker.com/en/products-and-solutions/mass-spectrometry/timstof.html

[leaderboard]: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard
[analog-v10]: https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline?scriptVersionId=350420114
[cosine-v17]: https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline?scriptVersionId=350445229
[two-rankers-v1]: https://www.kaggle.com/code/megayak/casmi26-two-rankers-one-engine?scriptVersionId=350481278
[evgen-notebook]: https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026?scriptVersionId=352813968
[ahmed-v3]: https://www.kaggle.com/code/ahmedberatozer/casmi26-v3-inference?scriptVersionId=352190437
[ahmed-v4f]: https://www.kaggle.com/code/ahmedberatozer/casmi26-v4f-inference?scriptVersionId=352983958
