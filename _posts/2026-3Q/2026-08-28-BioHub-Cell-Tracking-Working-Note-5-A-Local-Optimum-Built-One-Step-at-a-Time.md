---
title: "BioHub Cell Tracking Working Note 5: A Local Optimum, Built One Step at a Time"
date: 2026-08-28 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, objective-design, oracle-bounds, process-debt, division-recovery, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-08-28-biohub-working-note-5/cover.png
  alt: "Title card for BioHub Working Note 5: a local optimum, built one step at a time"
---

<style>
.content .table-wrapper > table {
  table-layout: fixed;
  width: 100%;
  min-width: 36rem;
}
.content .table-wrapper > table th,
.content .table-wrapper > table td {
  white-space: normal;
  overflow-wrap: break-word;
  vertical-align: top;
}
</style>

# BioHub Cell Tracking Working Note 5: A Local Optimum, Built One Step at a Time

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- Korean version: [BioHub Cell Tracking 작업 기록 5: 한 칸씩 쌓아 올린 방식이 Local Optimum에 갇힌 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

> **About this series.** These notes follow one Kaggle competition, BioHub Cell Tracking During Development:
> rebuilding cell lineages from 3D time-lapse microscopy of zebrafish embryos, trained on 199 movies from two
> embryos. The Public leaderboard shows 29% of a hidden test of unseen embryos, and a score chased there over many
> submissions overfits it. So each decision was made on its logic, a mechanism stated in advance, and the logic was
> tested by local validation. The series follows how that validation was made trustworthy, where the search itself
> fell short, and how the final submissions were chosen.
{: .prompt-info }

Note 4 moved local validation to embryo-out, every fold holding out a whole embryo, and the gains that remained were small.
This note covers 2026-08-17 to 2026-08-28 and asks why: were the ideas weak, or was the space they could act in nearly exhausted?

Ideas tested against a frozen embryo-out comparator returned zero or fifth-decimal deltas.
On 08-24 the labels choosing every edit the comparator allowed were worth $+0.000365$: the search had grown one pipeline a step at a time and built a local optimum around it, and the room lay outside, in which nodes exist and how divisions are emitted.
With a month left I took the largest room inside the existing pipeline, the division channel (ceiling $+0.0557$); a verifier that took $+0.0053$ of it locally went to the board as a sanity check, a Public submission that tests a local decision on unseen embryos, and had not answered when this window closed.

The short version is:

```text
Note 4 made validation honest (embryo-out), and honest gains were small. Why so little?
Ideas tested on a frozen embryo-out comparator: nulls; one positive (+0.0006686) tied.
On 08-24 its whole edit space, labels choosing every edit, was worth +0.000365.
The search had built a local optimum around one pipeline; the room lay in nodes and divisions.
Finding: for this problem the method was wrong; build breadth first, depth after.
New clauses: measure an action space's ceiling first; a gate must end in a decision.
With a month left, the division channel: ceiling +0.0557; a verifier (+0.0053) went out.
```

| Sections | Question |
|---|---|
| 0 | Why a frozen comparator, and what search did it sit on? |
| 1 | What did the ideas tested against it return? |
| 2 | With honest validation, why so little? |
| 3 | If not inside the edit space, where was the room? |
| 4 | Why does a search built one step at a time end in a local optimum? |
| 5 | What protects against it? |
| 6 | With a month left, why the division channel, and what did it show? |
| 7 | Decision Log |
| 8 | What is established, and what remains open? |

---

## 0. After Note 4: A Leak-Free Instrument, and the Search It Sat On

### 0.1 Why a frozen local comparator

After Note 4's fold leak, the period ran under a charter that took the leaderboard out of the selection loop:

> Public leaderboard values are transfer observations, not the model-selection objective. Fold-pure local OOF and exact graph replay remain the promotion authority.

About 160 submissions had gone to the board, many of them one-axis sweeps, and a displayed tie on its rounded 29% slice does not order two models; $0.970$ on Public remained the target on paper.

### 0.2 One pipeline, grown one component at a time

The system had one backbone from the start, the official learned-model family of Note 1: a temporal UNet that detects cell centers, learned edge probabilities, an integer linear program that selects the graph, and repair stages.
Every later addition, from a division stage to a track-fragment matcher, was kept because the graph improved.

On 08-05 I wrote the design down as a capability matrix of model families.
Its row for the dual-seed TemporalUNet3D detector reads "preserve as the common graph universe and comparator"; every other family entered as evidence on that graph.
The GPU program of 08-11 called its 480 GPU-hours "a conditional research portfolio", but a portfolio of components for one graph: "Common candidate topology and physical features are immutable. A new seed or family writes an append-only score sidecar against that topology."
Each new member $m_n$ was scored by what it added to the active set $A_n$:

$$
\Delta_n = S(A_n \cup \{m_n\}) - S(A_n).
$$

Note 3 had run seven baselines at once; the shared graph made every number comparable, leak control auditable, and evidence "additive rather than producing another collection of mutually incomparable scores."

### 0.3 The comparator

The frozen comparator is that design carried to its end: each of the 199 movies predicted by models that never saw its embryo (embryo-out, Note 4's clause), the whole graph scored by the official metric at $0.6013708666$, and a candidate allowed only legal edits to a graph whose models and most post-processing are fixed.
It could not leak, drift, or be fitted to the board; the ceiling of its action space, the best score reachable if every action were chosen with perfect knowledge, was unmeasured.
The two training embryos are called prefixes below; hold-in means scored on movies the models trained on.

### 0.4 Where the numbers come from

| instrument | movies | level |
|---|---|---:|
| the comparator's subset | the four distributed test movies (copies of training movies) | about $0.64$ |
| the deployed notebook's pipeline, hold-in | the same four | $0.8899$ |
| the deployed notebook's pipeline, embryo-disjoint detectors | the same four | about $0.81$ |
| a replay in a different node universe | 177 | near $0.69$ |

Deltas are read inside each instrument, never levels across them.
The board returned three scores in the window (a deployment migration, a runtime probe with byte-identical output, and the one local positive), all $0.921$, within $0.002$ of the 08-05 candidate and so the same score at the board's resolution.

---

## 1. Testing Ideas Against It: Nulls and One Tied Positive

| experiment | instrument | delta | observation |
|---|---|---:|---|
| additional detector seed, two-fold exact replay (about 36 GPU-hours) | 177-movie replay | $-0.0000325$ | folds $+0.0000560$ and $-0.0003967$ |
| track-fragment matcher, one training-group weight $2.0\to4.0$ | 199-movie embryo-out | $-0.0000427$ | both prefixes negative |
| matcher with a track-history input | 199-movie embryo-out | $-0.0000451$ | passed every pre-registered check; closed |
| auxiliary-center-detector and gap-filling pilots | 199-movie embryo-out | $+0.0000032$, $-0.0003537$ | neither passed its gates |
| three local-representation families | four-movie subset | $0.0$ exactly, three times | zero legal actions |
| **matcher, retrained independently at a fixed checkpoint** | 199-movie embryo-out | $\mathbf{+0.0006686}$ | 34 movies improved, 9 worsened, 156 tied |

The matcher relinks a broken track end to a successor one or two frames later.
For the detector seed, the window's largest compute item, my first summary quoted only the positive fold ($+5.6\times10^{-5}$).

### 1.1 The one positive, and its sanity check

The retrained matcher gained $+0.0006685607$ over the pre-matcher graph ($0.6006730$), $+0.0014526$ on one prefix and $+0.0005501$ on the other, across 3,207 edits, with the division false-positive count unchanged.
Its last successor, released through a notebook that had to reproduce the local replay's prediction bytes, became the comparator.

Shipped on 08-22 as a sanity check, v78 (submitted notebook versions are named vN from here on) read $0.921$, a tie and the likeliest outcome: the arithmetic of section 2.3 maps $+0.00067$ locally to between $+0.00009$ and about $+0.0006$ on the board.
The tie neither supports nor refutes the local result; a local gain below $10^{-3}$ is not reliably visible on the board, which made the size of the space the next question.

### 1.2 Three representations, zero actions

Three local-representation families (phase correlation, a learned dense-descriptor cost volume, a census cost volume) were built to give the fixed-node action policy a better matching signal.
All three passed every pre-registered check, selected zero legal actions, and returned the frozen graph's four-movie score, $0.6391791890184004$, to sixteen digits.
The signal was not what bound the policy, and the policy, the frozen half of every such comparison, could not be tested this way.

### 1.3 The level was real

A classical baseline, full-$Z$ Otsu thresholding with adjacent-frame Hungarian linking, scored $0.4626$ on the same four movies and scorer against the frozen learned graph's $0.6392$, a gap of $-0.1766$.
The nulls did not come from a weak base; they meant weak ideas or a small space.

---

## 2. Why So Little? The Ceiling of the Whole Edit Space

### 2.1 The oracle ceiling

Only a ceiling separates those readings: on 08-24 I asked how much any candidate could improve the comparator if the labels made every choice the policy allows.
Of the 3,178 actions the comparator selected across the 199 movies, keep one only if the labels say it strictly helps (no true edge lost, no false edge added, and at least one of the two strictly better), and rescore:

$$
S_{\mathrm{OOF}}:\;0.6013708666 \;\longrightarrow\; 0.6017363254,
\qquad
\Delta^{\mathrm{oracle}}_{\max}=+0.0003654588 .
$$

All of it came from one embryo prefix; the other could not be moved at all.

### 2.2 The same ceiling from the other side

A second, independent measurement started from the graph's damage: a replay of the comparator's candidate generation recovered 58,427 candidate edges, every swap possible without adding a node, and searched them for the 38,060 ground-truth edges the graph was missing (Figure 1).

![Bars: 38,060 missing ground-truth edges, 183 with any replacement, at most 115 topology-legal swaps]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-01-edge-swap-inventory.png)
_Figure 1. Of $38{,}060$ ground-truth edges missing from the comparator graph, $183$ had any replacement in the complete swap inventory and at most $115$ were topology-legal swaps. Division edits are outside this count._

For edge repair by swapping, the graph lacked the pieces it needed, whatever the ranker, representation or calibration.

### 2.3 Pricing the ceiling against the board

The one recorded transfer band for edge deltas, from 08-06, ran from $0.14\times$ to $0.9\times$ their local size, a historical range and at best an optimistic bound (Note 4's association gain, $+0.0073$ locally, tied its control on the board).
A change must reach half a display unit to show, $\rho\,\Delta_{\mathrm{local}} \ge \tfrac{1}{2}\times 10^{-3}$ for a ratio $\rho$ in that band, a local threshold between $5.6\times10^{-4}$ and $3.6\times10^{-3}$.
By that arithmetic, perfect knowledge over the comparator's actions reaches at most $+0.00033$ on the board, short of the $+0.0005$ needed to show at all.

### 2.4 What the ceiling measured

The ceiling bounds an edit space, not the objective: the same folds and scorer would price a different graph at a different level, and the oracle exhausted only the graphs reachable by legal edits from this one, the basin the step-by-step search had reached.
Every idea of section 1 acted inside that basin, so its nulls measured the space more than the ideas; the ceiling, computed in less than a day, bounds them all, including those that ran before it:

```text
C10. Measure the ceiling of an action space before optimizing inside it.
```

---

## 3. Where the Room Was: Outside the Edit Space

Three measurements looked beyond the frozen graph, and each found the difference in something an edit to it cannot change; the third, on the division stage, is in section 6.

### 3.1 External models, tried as parts of our graph

On 08-20 four externally trained models were adapted onto our frozen nodes and topology, so that only their association or repair decisions could differ.
On the four-movie subset, a forward-acceleration lookahead found zero eligible switches ($0.0$), an exact-coordinate detector adapter scored $-0.0044123$, a learned movement-field model $-0.0017270$, and a 4-D convolutional adapter failed closed in $0.324$ seconds; Note 4's pretrained parent ranker had ended the same way.

On our nodes, association was not what bound the score.
Each model was asked only what it adds as a part of our graph; whether any would beat our pipeline end to end, with its own nodes, links and divisions, was never asked.

### 3.2 Our graph beside an independent pipeline

On 08-26 I compared our graphs with an independently built reference pipeline on the four locally scorable movies, twice.
Against its native build, the reference carried 53,858 nodes ours lacked and ours carried 59,382 it lacked, and among those extra nodes it recovered 103 annotated cells to our 19.
Against a second build, the edge Jaccard over the 106,543 shared nodes was $0.9978$, and the reference emitted 384 parents with two children to our 37.

On shared nodes the two pipelines nearly agree; they differ in which nodes exist and how many divisions are emitted, exactly what an edit to our frozen graph cannot change.
This is our own diagnostic on four movies, not a measurement of anyone's score.

---

## 4. Why the Search Had Built a Local Optimum

Why had a careful search stayed inside a space whose room lay elsewhere? The cause lies in the design of section 0.2.

### 4.1 One path

Each step was accepted only if it improved the incumbent graph, $\Delta_n > 0$.
An accepted step changes the graph that every later step is measured on, so the search does not sample the space of pipelines; it follows one path through it.
Early choices, above all which detector defines the nodes, are never reopened, because every later comparison holds them fixed: section 1.2's frozen policy is the small case, section 3.2's frozen node set the large one.

### 4.2 Alternatives priced as parts

A different family was judged by its marginal value on the incumbent graph, not by what it would score on its own.
That value is low whenever the family overlaps what the graph already has or conflicts with it, and a family built around different nodes conflicts with a graph whose nodes are fixed.
A family that would win end to end and lose as a part reads the same as one that is simply worse.

### 4.3 The cost of a question

The charter was enforced by machinery built during Note 4's period (predeclared experiment cards, gates a run had to pass before it could start or count, immutable receipts), because a process exit code is not a scientific decision.
In this window an experiment took about sixteen new files, even for a single axis; a field-name mismatch between two record formats held a 331-task plan at zero; and a one-line fix forced a relaunch under a new name, so the full pipeline ran twice to return $0.8277207263$ in both arms of an association experiment.

The two probes toward nodes, where section 2 pointed, stopped without deciding anything.
A scratch detector put 1,190 of its 11,594 proposals on one held-out movie farther than $7\,\mu\mathrm{m}$ from any node we had, and stopped because no adapter for its output format had been declared in advance.
A native spot detector, gated behind readiness floors such as held-out recall of at least $0.50$ at $7\,\mu\mathrm{m}$, reached $0.179$ and $0.036$ with no movie scored, and under the rules a readiness failure could neither launch the family nor close it.
At 2026-08-28 00:02 UTC the RTX 5090 was at 0% utilization while runs waited for admission.

Together the rules made a question cost more than its answer was likely to be worth, and a search that can afford few questions rarely leaves its path.
A gate earns its cost when each outcome changes what happens next: a pass launches the next step, and a fail closes the line or names what to fix.

```text
C11. A gate must be able to end in a decision.
```

### 4.4 What the criterion could and could not do

Clauses C1 to C11 judge candidates; they cannot supply them, and applied strictly to a narrow stream they yield clean measurements of a small neighborhood and a local optimum certified with care.

---

## 5. What Protects Against It: Breadth Before Depth

For this problem the incremental method was the wrong one; the same leak-free embryo-out harness could judge a portfolio as readily as one pipeline.

The protection is breadth before depth:

1. **Several strong end-to-end families first,** each producing its own complete graph and finishing within the runtime limit (C6), so each is a candidate submission rather than a component.
2. **One common harness:** the same embryo-out folds and the official metric on the whole graph, with the edge and division terms read separately, so a difference in nodes or division emission shows up as a difference in a term.
3. **Selection and combination under that harness,** at the level of whole graphs.
4. **Then depth:** the step-by-step refinement this period gave one pipeline from the start goes to the winner.

Section 0.2's design made numbers comparable by sharing the graph; a common harness does it by sharing the folds and the scorer, and leaves the graphs free to differ, which is where this period's room was.
Breadth has its own trap: the best of several families on two embryos carries a winner's curse, so the choice among families needs its own held-out discipline.

---

## 6. With a Month Left: The Reset, and the Division Channel

### 6.1 The review and the six rules

By 08-28 twenty-three days had passed without a sanity check the board could read, and that day I reviewed the whole project.
The review found that the frozen comparator, not the hidden-set score, had become the quantity being maximized; that checking cost more per question than the answers were worth, with the GPU idle (section 4.3); and that the largest local numbers on record had never been re-measured against the deployed pipeline.

The apparatus of section 4.3 was retired, and six rules replaced it: the public board is the objective and local out-of-fold is a selection tool; no governance code; the GPU never idles while work exists; two hours from a new idea to the first prediction-facing artifact; a failed run is retried under the same name after the bug is fixed; closed results are never re-derived.
The plan written with them set out to reach the gold band on the board by the deadline, counting up to five submissions a day as nearly free transfer evidence.
The first rule, an anti-stall rule, supplies no ceiling (C10 applies to the board too) and does not say what keeps the board from being fitted once submissions are nearly free.

### 6.2 The shelved composition, measured the same day

The last finding put Note 4's shelved joint lineage-action mix at the top of the new plan: its replay evidence read $+0.0144$ and then $+0.0313$, projected at $+0.005$ to $+0.025$ on the board, but came from the four-fold research replay (base near $0.74$), whose folds Note 4 had found mixing both embryos.
The first measurement under the new rules put the lever on the deployed pipeline (the kernel: the submitted Kaggle notebook as it runs), over the four locally scorable movies:

| rung | regime | control | treatment | delta |
|---|---|---:|---:|---:|
| 1 | hold-in, control not faithful to the deployed kernel | $0.8831$ | $0.8842$ | $+0.0011$ |
| 2 | hold-in, control reproduces the deployed kernel exactly | $0.8866$ | $0.8858$ | $-0.0008$ |
| 3 | embryo-disjoint detectors, four-fold association heads | $0.8064$ | $0.8129$ | $+0.0065$ |

Rung 1's control sat below the deployed kernel's own $0.8899$, and rung 3 still scored the lever with the prefix-mixed four-fold heads, so neither counts.
Against the faithful base of rung 2 the lever measured $-0.0008$: the plan's first item was falsified on the day it was written.
A lever's value, $\Delta_L(B)=S(L\circ B)-S(B)$, belongs to lever and base together, and the base had since gained test-time augmentation, reverse-time harmonic fusion, consensus and a tuned edge threshold; the "about 75% absorbed" I wrote down that day compares deltas from three universes and is an estimate, not a decomposition.

### 6.3 Why the division channel, and what its structure looked like

With a month to the 09-29 deadline, the reset did not start over with end-to-end families; it went to the largest measurable room inside the pipeline that already ran within the time limit.
Division carries a tenth of the score, $S = J^{\mathrm{adjusted}}_{\mathrm{edge}} + 0.1\,J_{\mathrm{division}}$, and for weeks had been treated as a guard not to disturb, because the one edit tried in it had cost about $-0.00018$.
Two readings pointed at it: in Notes 3 and 4 the board had read the division stage as a step of four thousandths where the replay credited about one, and section 3.2's reference emitted 384 two-child parents to our 37.
As C10 asks, structure and ceiling came first, on the 199-movie embryo-out graphs (a fork is a node with two outgoing edges, a predicted division):

| quantity | value |
|---|---:|
| ground-truth divisions in all 199 movies | 151 |
| fully recoverable (parent and both daughters matched within $7\,\mu\mathrm{m}$) | 107 |
| with at least one of the two parent-to-daughter edges already in the graph | 101 |
| predicted forks at annotated parents | 849 |
| of those, sitting on a true division parent | 2 |

Recoverable true daughter pairs separate at a median of $10.7\,\mu\mathrm{m}$, counted false forks at $5.4\,\mu\mathrm{m}$, and the deployed rule admitted a fork only when the two putative sisters were within $8.5\,\mu\mathrm{m}$ of each other and each within $4.66\,\mu\mathrm{m}$ of the parent.
Its sister-distance limit admitted 96.3% of the counted false forks and 29.1% of the true divisions: the gate was selecting the geometric complement of the truth.

### 6.4 The ceiling, and a verifier inside it

In oracle mode across all 199 graphs, a daughter could be moved to a new parent (reparenting) and the labels chose the forks.

| arm | score | delta vs $0.6014$ | division detail |
|---|---:|---:|---|
| fork-free base | $0.6005$ | $-0.0009$ | prices the whole deployed division stage at $+0.0009$ |
| deployed strict stage | $0.6014$ | — | the incumbent |
| ground-truth oracle, reparenting allowed | $\mathbf{0.6571}$ | $\mathbf{+0.0557}$ | $J_{\mathrm{div}}=0.5577$; 87 TP, 5 FP, 64 FN; 107 forks added, 37 by reparenting |

![Log-scale comparison of the comparator ceiling (+0.000365) and the division ceiling (+0.0557) with measured gains]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-02-two-ceilings.png)
_Figure 2. The same scorer on the same 199 movies. A ground-truth filter over every action the comparator selected was worth less than the local size the board could show; the division stage's ceiling was two orders of magnitude larger. The shaded band is arithmetic from a historical transfer range, not a measurement._

A learned verifier then ranked the reparent-enabled candidates, prefix-pure (the ranker applied to each embryo trained only on the other).
In one day it went from a linear model on geometry ($+0.0015$, 6 of the oracle's 87 true positives) to a gradient-boosted ranker with track features ($+0.0035$) and then with appearance features ($+0.0053$, a score of $0.6067$, 11 true positives and 26 false positives).
Two limits were measured the same day: simulated division Jaccard was $0.43$ and $0.67$ on each ranker's training embryo but $0.062$ pooled on the other, a loss in ranking transfer across embryos rather than in the threshold; and the candidate generator covered only 75 of the 151 events.

### 6.5 The local gate, and the sanity check

A predeclared hold-in gate checked that division edits would not damage the edge channel: the candidate kernel reproduced the previous four-movie score of $0.8899$ with identical per-movie edge counts, its 45 division edits all landing away from the annotated tracks, a property of these four movies and not a general license.

The candidate, v79, went to the board as a sanity check late on 08-28: it had the strongest local support of the window, in a channel the replay had underpriced before.
The one expectation written that day was that the gain would reach the board at a similar order, not damped like edge deltas.
The edge band of section 2.3 would put $+0.0053$ anywhere from about $+0.0007$ to $+0.005$ on the board (arithmetic); the two division pairs of Notes 3–4, $+0.000949$ and $+0.0010142$ locally with a $+0.004$ step on the board each, give a direction, not a rate.
A clear drop would falsify the candidate; when this window closed, the board had not answered.

---

## 7. Decision Log

Until 08-28 the charter of section 0.1 held: a frozen embryo-out comparator was the selection objective, and its one sanity check (v78) tied, as section 2.3's arithmetic said it most likely would.
On 08-28 I made the board the objective as an anti-stall rule, recorded here and not in the criterion table because its fit with C9 is open, and sent a division candidate chosen on local evidence (v79) as the second sanity check.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| A frozen embryo-out comparator ($0.6013708666$) as the objective, on the one-graph design of 08-05 and 08-11 | after the fold leak, an objective that could not leak, drift or be fitted to the board | nulls or deltas below $10^{-4}$ | what could it pay? |
| Ship the matcher gain ($+0.0006686$) as v78 (08-22) | sanity check on unseen embryos | $0.921$, a tie | gains below $10^{-3}$ are not reliably visible |
| Measure the action space's oracle ceiling (08-24) | after a run of nulls: was there room at all? | $+0.000365$; 115 of 38,060 missing edges swappable | C10 |
| Look outside the frozen graph (08-20, 08-26) | where else was the room? | imports negative or inert; shared-node Jaccard $0.9978$; two-child parents 384 to 37 | the room lay in nodes and divisions |
| Native detector behind readiness floors | a detector that cannot find cells should not spend a scoring run | recall $0.179$ and $0.036$; neither a score nor a closure | C11 |
| Read the ceiling as a measurement of the search, not the objective | a flat basin, room outside it, a real base level | a diagnosis | the incremental method ends in a local optimum; breadth first |
| Reset: the board as objective; test the shelved $+0.0313$ composition (08-28) | twenty-three days without a readable sanity check | six rules; the lever $-0.0008$ on a faithful base | a lever's value depends on its base |
| Division: ceiling first, then a verifier; submit v79 (08-28) | the largest measured room reachable inside the existing pipeline | ceiling $+0.0557$; verifier $+0.0053$; not yet answered | the next sanity check is pending |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 | Write each gate down before the result exists | Note 3 (07-15) |
| C3 | Calibrate a rule on the population it will act on | Note 3 |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 | Compare levels only inside one reference replay; compare deltas across | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |
| C10 **(new)** | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 **(new)** | A gate must be able to end in a decision | Note 5 |

---

## 8. What the Period Established

### Established

1. A ground-truth filter over the comparator's 3,178 selected actions was worth $+0.0003654588$, all in one prefix, and only 115 of 38,060 missing ground-truth edges (0.30%) had a topology-legal swap.
2. The design of 08-05 and 08-11 kept one detector's graph as the common universe and scored every other family by its marginal value to it.
3. The matcher ($+0.0006686$) tied on the board, three representation families selected zero actions, and every fixed-node import of an external model was negative or inert.
4. On four movies, an independent pipeline matched ours on shared nodes (edge Jaccard $0.9978$) but differed in which nodes exist (103 annotated cells to our 19) and in two-child parents (384 to 37).
5. A native detector stopped twice at readiness floors that counted as neither a result nor a closure, and the shelved association composition measured $-0.0008$ against a faithful deployed base.
6. Of 151 ground-truth divisions, 107 were fully recoverable, and 2 of 849 forks at annotated parents sat on a true division parent; a reparent-enabled oracle was worth $+0.0557$, and a prefix-pure verifier reached $+0.0053$ in one day.

### Supported but Unconfirmed

1. That the remaining headroom lies in the node universe and the division channel rather than the association rule; it rests on two four-movie comparisons and the division oracle.
2. That several end-to-end families under one harness would have exposed that room earlier; it reads the mechanism of section 4, and no such portfolio was built.
3. That the deployed machinery had absorbed most of the shelved association gain; absorption and fold leakage are not separated.
4. That the $0.14\times$ to $0.9\times$ edge transfer band is a usable coefficient; it is a historical range.

### Open Questions

1. What is the label-oracle ceiling of the objective being optimized now, which C10 makes a precondition for any new optimization loop?
2. With the board as the objective, what keeps it from being fitted, and how does that rule fit with C9?
3. Does the division gain reach the board undamped, as the 08-28 expectation says?
4. Is the verifier's graph universe the one the shipped notebook produces? It was fitted on the 199 comparator graphs, which the project called the deployed chain, and I have not checked that the notebook builds the same graphs.
5. Does the verifier generalize across embryos? With two embryo domains and about 80 labeled positives, "prefix-pure" is two folds.
6. Which strong end-to-end families finish within the runtime limit, and how do they compare under one embryo-out harness, edge and division terms read separately?

---

## Closing

The frozen comparator never leaked, drifted or borrowed from the board, and its whole edit space was worth $+0.000365$: one pipeline grown a component at a time had built a local optimum, with the room outside it, in nodes and divisions.
The remedy is breadth first; with a month left I took the largest room inside the existing pipeline instead, a division ceiling of $+0.0557$ and a verifier that took $+0.0053$ of it in one day.

The division channel now has a measured ceiling; the hidden set has not yet answered.
Notes 3 and 4 had already shown the replay underpricing the division stage, so the next note starts from two questions: does the verifier's gain reach unseen embryos, and does local validation price the division channel the way the hidden set does?

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- **Part 5: A Local Optimum, Built One Step at a Time**
