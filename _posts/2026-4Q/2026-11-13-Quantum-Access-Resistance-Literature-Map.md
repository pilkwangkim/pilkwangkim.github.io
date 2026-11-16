---
title: "Quantum Transport at Channel Access Regions: An Interactive Literature Map"
description: "Verified citations and shared references connect research on channel access geometry, mode coupling, underlap doping, and quantum access resistance."
date: 2026-11-13 17:15:00 +0900
categories: [Physics, Quantum Transport]
tags: [quantum-transport, bibliometrics, nanosheet, negf, semiconductor]
topic: quantum-transport
lang: en
translation_key: quantum-access-resistance-literature-map
interactive: quantum-access-literature
math: true
comments: false
pin: false
published: true
---

[Read in Korean]({{ site.baseurl }}/posts/Quantum-Access-Resistance-Literature-Map-KR/)

A wide source connected to a narrow channel can reflect electron waves even when the channel itself contains no scattering mechanism. Changing confinement alters the available transverse states, or **modes**, and how they connect across the junction. In a transistor, current also depends on doping, the electrostatic potential, contacts, and collisions. **Quantum access resistance** concerns the resulting quantum transport limitation between a reservoir and the channel. Its magnitude depends on how these ingredients act together.

The map connects **six starting papers** to a curated network of **{{ site.data.quantum_access_literature.nodes.size }} papers** and **{{ site.data.quantum_access_literature.edges.size }} citation links**. Its chronology follows changing questions and methods. This selected network does not measure publication growth across the field.

{% include interactive/quantum-access-literature.html %}

## Reading the connections

A **citation** is a directed bibliographic relation: an arrow from A to B means that A cites B. It establishes a reference, while the surrounding text determines whether A adopts a method, compares a result, or challenges an interpretation.

The **shared-reference** view answers a different question. If A and B both cite R, the relation is **A → R ← B**. This is called **bibliographic coupling**; it does not require an A → B or B → A citation. Venugopal2004 and Laux2004 share four DOI-identified references: QDAME2002, Venugopal2002's real/mode-space comparison, Svizhenko2002, and Datta2000. These common methods and concepts explain their proximity without establishing agreement on every physical conclusion.

Shared references are counted by exact DOI matches in the deposited bibliographies. References without DOIs are omitted, so zero means that no shared DOI was found in those records. Selecting a paper exposes its question, calculation, findings, implications, and reading scope; selecting a chronology entry connects that stage to the relevant paper.

## What a constriction changes

Five effects can coexist at a reservoir–channel junction: a reduced number of propagating modes; reflection at an abrupt transition; mixing between transverse modes; a change in confinement energy; and an electrostatic barrier produced by the potential and doping profile. Collisions then redistribute the carriers encountering those features. Calling all of them a “constriction barrier” can hide which quantity a comparison actually isolates.

For a coherent, elastic, two-terminal conductor at sufficiently low temperature and in linear response, the Landauer expression is

$$
G=\frac{2q^2}{h}\sum_n T_n(E_F).
$$

Here $q>0$ is the elementary charge, $h$ is Planck's constant, and $T_n$ is a transmission eigenvalue between 0 and 1. The factor 2 assumes spin degeneracy; the sum includes the other conducting channels, including valley channels where applicable. Counting modes sets a possible conductance, while their transmissions determine how much of it is realized. At finite bias, current samples an energy window rather than only $E_F$. [van Wees et al.](https://doi.org/10.1103/PhysRevLett.60.848), [Datta](https://doi.org/10.1006/spmi.2000.0920)

A schematic way to keep track of a local subband edge is

$$
U_n(x)=E_{c0}(x)-q\phi(x)+E_{\mathrm{conf},n}(x).
$$

With a chosen longitudinal reference, $E_{c0}$ is the band-edge offset, $\phi$ the electrostatic potential, and $E_{\mathrm{conf},n}$ the transverse confinement energy relative to that reference. A narrower section can raise confinement energy; low doping can change the self-consistent potential. Both can affect a carrier's energy barrier.

This is **diagonal energy bookkeeping**, not a complete coupled-mode model. A varying transverse potential also changes the mode shapes, and the connections between those shapes generate off-diagonal coupling. A one-dimensional barrier profile cannot by itself describe all mode conversion. Conversely, retaining coupling does not remove the need to solve the correct electrostatics. [Polizzi and Ben Abdallah](https://doi.org/10.1016/j.jcp.2004.07.003), [Luisier et al.](https://doi.org/10.1063/1.2244522)

Holding $\phi$ fixed still leaves geometry-dependent confinement energies. A fixed-potential shape comparison therefore need not isolate off-diagonal mode conversion alone. It also matters whether a paper uses **mode mismatch** for a mismatch in channel count, total junction reflection, or conversion between transverse modes.

## Six starting papers

The numbered papers are the starting set. Dates use journal issue years when online publication and issue dates differ.

| Paper | Year | Full title and DOI |
|:--|:--|:--|
| P1 · Hung et al. | 2025 | **On-Current Degradation in Ultra-Scaled Nanosheet FETs With S/D Underlap Doping** · [10.1109/LED.2025.3614369](https://doi.org/10.1109/LED.2025.3614369) |
| P2 · Venugopal et al. | 2004 | **Quantum mechanical analysis of channel access geometry and series resistance in nanoscale transistors** · [10.1063/1.1631754](https://doi.org/10.1063/1.1631754) |
| P3 · Laux et al. | 2004 | **Analysis of quantum ballistic electron transport in ultrasmall silicon devices including space-charge and geometric effects** · [10.1063/1.1695597](https://doi.org/10.1063/1.1695597) |
| P4 · Ahn and Hong | 2023 | **Geometric Scattering Describing Mode-Coupling Effects in Non-Uniform Cross-Sections for Non-Equilibrium Green’s Function and Multi Subband Boltzmann Transport Equation Solvers** · [10.1109/IEDM45741.2023.10413710](https://doi.org/10.1109/IEDM45741.2023.10413710) |
| P5 · Luisier et al. | 2006 | **Quantum transport in two- and three-dimensional nanoscale transistors: Coupled mode effects in the nonequilibrium Green’s function formalism** · [10.1063/1.2244522](https://doi.org/10.1063/1.2244522) |
| P6 · Pourghaderi et al. | 2026 | **On Effectiveness of Constriction Barrier in Nanotransistors** · [10.1109/LED.2026.3681884](https://doi.org/10.1109/LED.2026.3681884) |

## From point contacts to transistor entrances

The 1988 point-contact experiments of van Wees et al. and Wharam et al. established the conductance/resistance steps associated with transverse subbands. They concern short, cryogenic ballistic channels. Szafer and Stone's 1989 theory and Nonoyama et al.'s 1992 wide–narrow calculation then examine how junction shape and matching affect transmission. The important inheritance is a question about connecting quantum states, not a universal room-temperature FET penalty. [Szafer and Stone](https://doi.org/10.1103/PhysRevLett.62.300), [Nonoyama et al.](https://doi.org/10.1063/1.106974)

An open device also needs reservoirs that inject and absorb waves. Lent and Kirkner's quantum transmitting boundary method, the QDAME studies, and real-space NEGF–Poisson calculations supply different numerical routes to this problem. Datta2000 emphasizes the joint role of quantum transport and electrostatics. Venugopal2002 benchmarks a reduced, decoupled mode-space model against real space; that reduction has a validity range rather than automatic applicability to every shape. [Lent and Kirkner](https://doi.org/10.1063/1.345156), [Venugopal et al.](https://doi.org/10.1063/1.1503165)

The two 2004 papers examine channel access geometry from complementary directions. Laux and colleagues compare straight, tapered, and dog-bone geometries using self-consistent **ballistic** transport, where phonon and impurity collisions are absent. Venugopal and colleagues include an approximate scattering treatment within the **nonequilibrium Green’s function (NEGF)** formalism coupled to Poisson’s equation. Scattering changes how carriers in a wide reservoir enter the small set of channel modes, and can reduce the current difference between geometries. Both papers cite the 2002 QDAME method paper. [Laux et al.](https://research.ibm.com/publications/analysis-of-quantum-ballistic-electron-transport-in-ultrasmall-silicon-devices-including-space-charge-and-geometric-effects), [Venugopal et al.](https://docs.lib.purdue.edu/nanodocs/188/)

A smaller current difference between geometries is distinct from an increase in total device current. Even when redistribution between modes helps injection, additional scattering can introduce a separate transport loss.

Luisier et al. address a computational requirement behind that comparison. At an abrupt cross-section change, coupled-mode calculations must retain the correlations needed to reconstruct charge and current. Incomplete reconstruction can introduce charge spikes or violate current conservation. Their ballistic calculations agree with real-space results when the relevant terms are retained. Increasing the retained mode count and checking this reconstruction therefore tests numerical adequacy; it is not a new physical scattering mechanism. [Author manuscript](https://iis-people.ee.ethz.ch/~schenk/JApplPhys_100_043713.pdf)

The design problem becomes richer when individual dopants and roughness matter. Berrada et al. (2015) link access-region shape and discrete dopant positions to injection through quasi-localized states. A nominal average doping concentration therefore does not describe every device realization. Ahn and Hong (2023) express non-uniform-section mode-coupling effects through geometric scattering for NEGF and multi-subband Boltzmann transport equation (MS-BTE) solvers. Geometric mode mixing and phonon/impurity collisions remain distinguishable contributions even when a solver uses a scattering representation. [Berrada et al.](https://doi.org/10.1063/1.4933392), [Ahn and Hong](https://doi.org/10.1109/IEDM45741.2023.10413710)

## Why the comparison conditions matter

Stacked nanosheet gate-all-around FETs bring this question to a realistic connection between bulk source/drain regions and thin sheets. Kim et al. (2025) model these geometries and identify strong reflection at low-doped constrictions. Confinement-energy changes, scattering, doping, and contact depletion affect the result. The study extends the comparison beyond uniformly high-doped source/drain regions: a previously small effect need not remain small after the injection environment changes. [Open article](https://www.nature.com/articles/s44172-025-00435-0)

In that study's design examples, a deeper contact or a shorter extension does not monotonically improve injection. A moderate contact depth and a sufficiently long extension can be favorable for the modeled constriction. Mode transmission and the self-consistent potential therefore have to accompany the classical intuition of a larger contact area and a shorter resistive path. [Kim et al., Discussion](https://www.nature.com/articles/s44172-025-00435-0)

Hung et al. examine another contribution: secondary barriers in underdoped source/drain extensions and anomalous linear-regime current saturation. Their subband-BTE analysis therefore addresses the potential profile as well as current transport. An underlap-induced barrier and a purely geometric mode mismatch are different contributions, even when both reduce the on-current. [Open article](https://repositum.tuwien.at/bitstream/20.500.12708/227257/1/Hung-2025-IEEE%20Electron%20Device%20Letters-vor.pdf)

Pourghaderi et al. compare transmission under identical electrostatic potentials to isolate mode mismatch. Their **public abstract** reports a negligible transmission effect for a **nominal overlap junction**. The full paper was not available for this analysis, so the interpretation remains limited to that stated comparison. It does not establish that constriction effects vanish for every doping profile or junction design. [Public abstract](https://www.researchgate.net/publication/403634150_On_effectiveness_of_constriction_barrier_in_nanotransistors)

The target observable also matters. Pech and Schulz (2025) develop a time-dependent coupled-mode density-matrix method and check it against NEGF. Their ballistic demonstration finds different steady currents in squeezed and unsqueezed devices, yet similar amplifier behaviour at matched operating points. Similar RF behaviour in that comparison does not make all geometry effects negligible. [Pech and Schulz](https://pubs.aip.org/jap/article/138/12/125705/3364353/Coupled-mode-effects-in-the-stationary-and)

The 2026 p-type nanosheet study by Hung et al. adds insufficient extension doping, stress relaxation, and inner-spacer interface traps to the injection problem. These effects become more consequential in the study's injection-limited regime. This is a device-specific combination of mechanisms, not evidence that every current loss is coherent geometric reflection. [Authors' institutional account](https://www.globaltcad.com/performance-degradation-mechanisms-in-p-type-nanosheet-fets-role-of-source-draindoping-stress-and-inner-spacer-interface-traps/)

## What a comparison can establish

Matching the following conditions clarifies which question a comparison answers.

| Comparison | What must be controlled | What it can establish |
|:--|:--|:--|
| Coupled mode space versus real space | Hamiltonian, grid, boundaries, and sufficient retained modes | Whether a difference is caused by a numerical reduction |
| Constricted versus uniform, fixed potential | Electrostatic profile and injection conditions | The transmission contribution of geometry/mode mismatch, including confinement-energy changes |
| Constricted versus uniform, self-consistent | Biases, material and doping assumptions; allow the potential to respond | The combined geometry–electrostatics current change |
| Ballistic versus scattering calculation | Geometry, band model, biases, and dopants | How relaxation changes injection and the geometry dependence |
| Doping or contact sweep | State which profile or contact assumption changes | Whether a bottleneck is conditional on depletion or injection |
| Matched RF operating points | Bias criterion, signal conditions, and measured circuit quantity | Whether a DC difference persists in the chosen amplifier metric |

For design, the useful inference is to inspect both the transverse connection and the injection potential. Enlarging a source/drain volume, shortening an extension, or changing a doping gradient changes more than a classical area-to-length resistance. Their benefit must be assessed at the intended bias, with adequate mode coupling, scattering, and contact assumptions. The literature supports these conditional comparisons rather than a single constriction penalty applicable to every transistor.

## Data and evidence

The snapshot was collected on **13 November 2026**. Titles, identifiers, dates, and citation counts come from [OpenAlex](https://api.openalex.org/works) and [Crossref](https://api.crossref.org/works). Citation links retain their publisher reference keys and supporting DOI or bibliographic match. Each record also identifies whether OpenAlex confirms the link. Paper details distinguish an abstract-based reading from consultation of an author manuscript or published full text.

An absent arrow does not establish the absence of a citation: indexed and deposited references can be incomplete. Citation counts are retrieval-date observations, while issue years avoid mixing early online dates with journal chronology. The [map data]({{ site.baseurl }}/assets/data/quantum-access-literature.json) retain the reference keys and evidence URLs for each displayed link.

## Full bibliography

<details markdown="1">
<summary>All {{ site.data.quantum_access_literature.nodes.size }} papers, ordered by publication year</summary>

<ul data-qal-bibliography>
{% for paper in site.data.quantum_access_literature_by_year %}
<li><strong>{{ paper.label | escape }}</strong> — {{ paper.title | escape }}. <a data-qal-bibliography-paper="{{ paper.id | escape }}" href="{{ paper.url | escape }}">DOI: {{ paper.doi | escape }}</a>.</li>
{% endfor %}
</ul>

</details>
