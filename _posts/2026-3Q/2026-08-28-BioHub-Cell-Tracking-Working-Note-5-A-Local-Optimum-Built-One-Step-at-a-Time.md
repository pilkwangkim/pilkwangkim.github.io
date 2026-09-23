---
title: "BioHub Cell Tracking Working Note 5: What a Frozen Graph Left Untested"
date: 2026-08-28 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, objective-design, candidate-coverage, process-debt, division-recovery, working-note]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-08-28-biohub-working-note-5/cover.png
  alt: "BioHub Cell Tracking Working Note 5: What a Frozen Graph Left Untested"
---

<style>
/* Local to the BioHub manuscripts; labels follow each table's own headings. */
.content .table-wrapper:has(> table.biohub-table) {
  max-width: 100%;
  overflow-x: auto;
  container: biohub / inline-size;
}
.content .table-wrapper > table.biohub-table {
  table-layout: fixed;
  width: 100%;
  min-width: var(--table-min, 0);
  font-size: 0.92rem;
  line-height: 1.6;
  font-variant-numeric: tabular-nums;
}
.content table.biohub-table th,
.content table.biohub-table td {
  padding: 0.6rem 0.7rem;
  white-space: normal;
  overflow-wrap: anywhere;
  vertical-align: top;
}
html[lang="ko"] .content table.biohub-table { word-break: keep-all; }
.content table.biohub-table th:nth-child(1) { width: var(--c1); }
.content table.biohub-table th:nth-child(2) { width: var(--c2); }
.content table.biohub-table th:nth-child(3) { width: var(--c3); }
.content table.biohub-table th:nth-child(4) { width: var(--c4); }
.content table.biohub-table th:nth-child(5) { width: var(--c5); }
.content table.biohub-table th:nth-child(6) { width: var(--c6); }
@container biohub (max-width: 620px) {
  .content .table-wrapper > table.biohub-records {
    display: block;
    min-width: 0;
    border: 0;
  }
  .content table.biohub-records thead {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  .content table.biohub-records tbody { display: block; }
  .content table.biohub-records tr {
    display: block;
    margin-bottom: 0.9rem;
    border: 1px solid var(--tb-border-color, #9996);
    border-radius: 0.3rem;
  }
  .content table.biohub-records td {
    display: block;
    width: auto;
    border: 0;
    text-align: left !important;
    padding: 0.45rem 0.75rem;
  }
  .content table.biohub-records td:first-child {
    font-weight: 600;
    border-bottom: 1px solid var(--tb-border-color, #9996);
    padding-block: 0.65rem;
  }
  .content table.biohub-records td:last-child { padding-bottom: 0.75rem; }
  .content table.biohub-records td:not(:first-child)::before {
    display: block;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted-color, #6c757d);
    margin-bottom: 0.1rem;
  }
  .content table.biohub-records td:nth-child(2)::before { content: var(--label2); }
  .content table.biohub-records td:nth-child(3)::before { content: var(--label3); }
  .content table.biohub-records td:nth-child(4)::before { content: var(--label4); }
  .content table.biohub-records td:nth-child(5)::before { content: var(--label5); }
  .content table.biohub-records td:nth-child(6)::before { content: var(--label6); }
}

@container biohub (max-width: 575px) {
  .content table.biohub-numeric:has(th:nth-child(4))::before {
    content: "↔ Scroll horizontally to see all columns.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
@container biohub (max-width: 703px) {
  .content table.biohub-numeric:has(th:nth-child(5))::before {
    content: "↔ Scroll horizontally to see all columns.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
.content mjx-container {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
}
.content mjx-container[display="true"] { padding-block: 0.25rem; }
.content details { min-width: 0; }
.content summary { cursor: pointer; }
.content code { overflow-wrap: anywhere; }
</style>

<details markdown="1">
<summary>Series links and references</summary>

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: Three Gaps in Local Validation]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- Korean version: [BioHub Cell Tracking 작업 기록 5: 고정된 그래프가 시험하지 못한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)

</details>

<details markdown="1">
<summary>Related public notebooks</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **About this series.** BioHub asks competitors to reconstruct cell lineage graphs from 3D microscopy movies. The labeled training set contains 199 movies from two embryos; Public scores cover 29% of the hidden test set, and Private scores cover the remaining 71%. The hidden test comes from embryos not seen in training. Each note follows the record through its stated period; later findings and retrospective comments are marked separately.
{: .prompt-info }

> **Later context — September 23, 2026.** Sections 4–5 look back on the search design, and Public differences within ±0.002 are read as ties under the September 5–6 rule. September also revealed that the local comparator ran different graph stages from the submitted notebook (Note 6); that mismatch was not established at the August 28 reset.
{: .prompt-info }

Note 4 left an unexplained Public result, a revised fold protocol and a retracted hold-in diagnostic. Between August 17 and 28 I froze a comparator and tested changes around it. Most returned little. Looking back, the question was whether those tests covered enough alternatives to justify staying with the same pipeline.

Two diagnostics found limited reach within the existing edge-repair procedure. The August 28 review called that action space "provably exhausted," a stronger conclusion than the measurements supported. Alternative models had mostly been tried as components of our graph, leaving a comparison of complete pipelines undone.

---

## 0. What Freezing the Comparator Changed

### 0.1 Why a frozen local comparator

After the August fold audit, the period ran under a charter that took the leaderboard out of the selection loop:

> Public leaderboard values are transfer observations, not the model-selection objective. Fold-pure local OOF and exact graph replay remain the promotion authority.

About 160 submissions had gone to the board, many of them one-axis sweeps, and a displayed tie on its rounded 29% slice does not order two models; $$0.970$$ on Public remained the target on paper.

### 0.2 One pipeline, grown one component at a time

The system had one backbone from the start, the official learned-model family of Note 1: a temporal UNet that detects cell centers, learned edge probabilities, an integer linear program that selects the graph, and repair stages.
Later additions, from division stages to track-fragment matching, were usually assessed by their incremental effect on an existing graph.

On 08-05 I wrote the design down as a capability matrix of model families.
Its row for the dual-seed TemporalUNet3D detector reads "preserve as the common graph universe and comparator"; every other family entered as evidence on that graph.
The GPU program of 08-11 called its 480 GPU-hours "a conditional research portfolio", but a portfolio of components for one graph: "Common candidate topology and physical features are immutable. A new seed or family writes an append-only score sidecar against that topology."
Each new member $$m_n$$ was scored by what it added to the active set $$A_n$$:

$$
\Delta_n = S(A_n \cup \{m_n\}) - S(A_n).
$$

Note 3 had run seven baselines at once; the shared graph was intended to make paired comparisons easier to interpret. It did not make gains from different interventions additive or remove training dependencies.

### 0.3 The comparator

The comparator scored $$0.6013708666$$ on 199 movies. Its backbone predictions used embryo-out training splits, and candidate edits were evaluated against fixed graphs, models and post-processing. This stabilized the comparison: a change could not gain merely because its baseline had moved.

Here, “embryo-out” names the backbone training split. The two embryos are called prefixes below, and hold-in means evaluation on movies used for model training.

The immediate limitation was more concrete: an edit could use only the candidates the frozen procedure supplied. We had not measured their coverage.

### 0.4 Where the numbers come from

| instrument | movies | level |
| --- | --- | ---: |
| the comparator's subset | the four distributed test movies (copies of training movies) | about 0.64 |
| the deployed notebook's pipeline, hold-in | the same four | 0.8899 |
| the deployed notebook's pipeline, embryo-disjoint detectors | the same four | about 0.81 |
| a replay in a different node universe | 177 | near 0.69 |
{: #biohub-table-1 .biohub-table .biohub-records style="--c1: 43%; --c2: 37%; --c3: 20%; --table-min: 0; --label1: 'instrument'; --label2: 'movies'; --label3: 'level'" }

Deltas are read inside each instrument, never levels across them.
The board returned three scores in the window (a deployment migration, a runtime probe with byte-identical output, and the one local positive), all $$0.921$$, with no observed improvement over the 08-05 candidate.

---

## 1. Testing Ideas Against It: Nulls and One Tied Positive

| experiment | instrument | delta | observation |
| --- | --- | ---: | --- |
| additional detector seed, two-fold exact replay (about 36 GPU-hours, partly projected) | 177-movie replay | -0.0000325 | folds +0.0000560 and -0.0003967 |
| track-fragment matcher, one training-group weight 2.0→4.0 | 199-movie embryo-out | -0.0000427 | both prefixes negative |
| matcher with a track-history input | 199-movie embryo-out | -0.0000451 | passed every pre-registered check; closed |
| auxiliary-center-detector and gap-filling pilots | 199-movie embryo-out | +0.0000032, -0.0003537 | neither passed its gates |
| three local-representation families | four-movie subset | 0.0 exactly, three times | zero legal actions |
| **matcher, retrained independently at a fixed checkpoint** | 199-movie embryo-out | $$\mathbf{+0.0006686}$$ | 34 movies improved, 9 worsened, 156 tied |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 32%; --c2: 19%; --c3: 20%; --c4: 29%; --table-min: 36rem; --label1: 'experiment'; --label2: 'instrument'; --label3: 'delta'; --label4: 'observation'" }

The matcher relinks a broken track end to a successor one or two frames later.
For the detector seed, my first summary quoted only the positive fold ($$+5.6\times10^{-5}$$). The pooled result was negative. The roughly 36 GPU-hour cost combined about 12.78 measured hours for one fold with about 23 projected hours for the other; it was not a complete measured runtime.

### 1.1 The one positive, and its sanity check

The retrained matcher gained $$+0.0006685607$$ over the pre-matcher graph ($$0.6006730$$), $$+0.0014526$$ on one prefix and $$+0.0005501$$ on the other, across 3,207 edits, with the division false-positive count unchanged.
Its last successor, released through a notebook that had to reproduce the local replay's prediction bytes, became the comparator.

Shipped on 08-22 as a sanity check—a Public submission testing a local decision on unseen embryos—v78 scored $$0.921$$. The unchanged displayed score neither established nor refuted a local gain of $$+0.0006686$$. After repeated nulls and small gains, I next asked how many errors the repair procedure could reach at all.

### 1.2 Three representations, zero actions

Three local-representation families (phase correlation, a learned dense-descriptor cost volume, a census cost volume) were built to give the fixed-node action policy a better matching signal.
All three passed every pre-registered check, selected zero legal actions, and returned the frozen graph's four-movie score, $$0.6391791890184004$$, to sixteen digits.
These compositions left the graph unchanged. They did not show whether the representations would help under a different policy or candidate generator, both held fixed in this test.

### 1.3 The level was real

A classical baseline, full-$$Z$$ Otsu thresholding with adjacent-frame Hungarian linking, scored $$0.4626$$ on the same four movies and scorer against the frozen learned graph's $$0.6392$$, a gap of $$-0.1766$$.
This ruled out the tested classical baseline as a stronger starting point. The learned graph's remaining errors still required a candidate-coverage check.

---

## 2. What the Existing Repair Procedure Could Reach

### 2.1 Filtering actions that had already been selected

On 08-24 I inspected the **3,178 swaps already selected** by the frozen ShiftCorr replay. A label-assisted filter retained an action when it lost no true edge, added no false edge, and strictly improved at least one count under the fixed matching. Rescoring that subset gave

$$
S_{\mathrm{full}}=0.6013708666,
\qquad S_{\mathrm{filtered}}=0.6017363254,
\qquad \Delta_{\mathrm{filter}}=+0.0003654588.
$$

The gain came entirely from one embryo; the other was unchanged under this filter.

### 2.2 Coverage of the reconstructed candidate generator

A separate replay reconstructed 58,427 candidate edges using the available generator source and frozen inputs. Of 38,060 ground-truth edges missing from the comparator, 183 had a replacement in this inventory and at most 115 had a topology-legal swap.

![Missing ground-truth edges and their coverage by the reconstructed swap generator]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-01-edge-swap-inventory.png)
_Figure 1. Coverage of one reconstructed candidate generator: $$183$$ of $$38{,}060$$ missing edges had a replacement, at most $$115$$ a legal swap. This inventory does not include node insertion or division edits, and does not prove the contents of every historical candidate table._

A better ranker cannot select an absent candidate. This inventory therefore explained why improving the ranking signal alone might do little for the current generator.

### 2.3 What the board could tell us

An earlier plan quoted local-to-Public ratios of $$0.14\times$$ to $$0.9\times$$, but Note 4's $$+0.0073$$ local association gain had tied its matched Public control under this series' later reading rule. Those observations supplied no reliable conversion from the selected-action diagnostic to a Public gain.

### 2.4 The conclusion then, and its limit

The August 26 summary described an oracle bound, and the August 28 review called the comparator's action space “provably exhausted.” That reading helped redirect the work toward nodes and divisions.

Looking back, it overreached. The $$+0.000365$$ filter considered only 3,178 already-selected swaps and excluded TP/FP tradeoffs that might still improve the score. The coverage replay examined one reconstructed generator. Neither enumerated all feasible graph edits or their combinations. The justified conclusion was that these two procedures offered little additional reach—not that the pipeline had reached an optimum.

I summarize the lesson as C10:

```text
C10. Measure candidate reach; call a diagnostic a bound only within its declared scope.
```

---

## 3. What the Frozen Comparisons Left Untested

The next comparisons examined candidate nodes and division proposals beyond the frozen edge swaps. They suggested other directions; they did not establish an alternative pipeline's generalization performance.

### 3.1 External models, tried as parts of our graph

On 08-20 four externally trained models were adapted onto our frozen nodes and topology, so that only their association or repair decisions could differ.
On the four-movie subset, a forward-acceleration lookahead found zero eligible switches ($$0.0$$), an exact-coordinate detector adapter scored $$-0.0044123$$, a learned movement-field model $$-0.0017270$$, and a 4-D convolutional adapter failed closed in $$0.324$$ seconds; Note 4's pretrained parent ranker had ended the same way.

These particular adapters did not improve the fixed-node composition.
Each model was asked only what it adds as a part of our graph; whether any would beat our pipeline end to end, with its own nodes, links and divisions, was never asked.

### 3.2 Our graph beside an independent pipeline

On 08-26 I compared our graphs with an independently built reference pipeline on the four locally scorable movies, twice.
Against its native build, the reference carried 53,858 nodes ours lacked and ours carried 59,382 it lacked, and among those extra nodes it recovered 103 annotated cells to our 19.
Against a second build, the edge Jaccard over the 106,543 shared nodes was $$0.9978$$, and the reference emitted 384 parents with two children to our 37.

On shared nodes the two pipelines nearly agree; they differ in which nodes exist and how many divisions are emitted, properties that the frozen edge-swap tests had held constant.
This is our own diagnostic on four movies, not a measurement of anyone's score.

---

## 4. Looking Back: How the Search Restricted the Questions

The design in section 0.2 made incremental comparisons easy, but left several important comparisons undone.

### 4.1 One path

Each step was accepted only if it improved the incumbent graph, $$\Delta_n > 0$$.
An accepted step changes the graph that every later step is measured on, so the search does not sample the space of pipelines; it follows one path through it.
Most of these comparisons held the detector and node set fixed, so they could not test alternatives to those earlier choices. Attempts to change the nodes did occur, but the two examples in Section 4.3 stopped before a scored composition. The practical result was a search dominated by one pipeline, rather than evidence that its starting choices were best.

### 4.2 Alternatives evaluated only as components

A different family was judged by its marginal value on the incumbent graph, not by what it would score on its own.
That value is low whenever the family overlaps what the graph already has or conflicts with it, and a family built around different nodes may lose part of its intended behavior when adapted to fixed nodes.
A family that would win end to end and lose as a part reads the same as one that is simply worse.

### 4.3 The cost of a question

The charter was enforced by machinery built during Note 4's period (predeclared experiment cards, gates a run had to pass before it could start or count, immutable receipts), because a process exit code is not a scientific decision.
In this window an experiment took about sixteen new files, even for a single axis; a field-name mismatch between two record formats held a 331-task plan at zero; and a one-line fix forced a relaunch under a new name, so the full pipeline ran twice to return $$0.8277207263$$ in both arms of an association experiment.

Two probes toward new nodes stopped before producing a scored composition.
A scratch detector put 1,190 of its 11,594 proposals on one held-out movie farther than $$7\,\mu\mathrm{m}$$ from any node we had, and stopped because no adapter for its output format had been declared in advance.
A native spot detector, gated behind readiness floors such as held-out recall of at least $$0.50$$ at $$7\,\mu\mathrm{m}$$, reached $$0.179$$ and $$0.036$$ with no movie scored, and under the rules a readiness failure could neither launch the family nor close it.
At 2026-08-28 00:02 UTC the RTX 5090 was at 0% utilization while runs waited for admission.

These episodes show how admission and interface requirements delayed prediction tests. They support simplifying those specific requirements, without establishing that every gate cost more than it saved.
A gate earns its cost when each outcome changes what happens next: a pass launches the next step, and a fail closes the line or names what to fix.

```text
C11. A gate must be able to end in a decision.
```

### 4.4 What the criterion could and could not do

Selection rules evaluate the candidates we supply. They cannot reveal an end-to-end alternative that was only tested as a component, or compensate for an intervention that cannot reach its target errors.

---

## 5. What I Would Test Next Time

Next time I would reserve more early effort for independent end-to-end alternatives, in this sequence:

1. **Several strong end-to-end families first,** each producing its own complete graph and finishing within the runtime limit (C6), so each is a candidate submission rather than a component.
2. **One common harness:** the same embryo-out folds and the official metric on the whole graph, with the edge and division terms read separately, so a difference in nodes or division emission shows up as a difference in a term.
3. **Selection and combination under that harness,** at the level of whole graphs.
4. **Then depth:** the step-by-step refinement this period gave one pipeline from the start goes to the winner.

Section 0.2's design made numbers comparable by sharing the graph; a common harness does it by sharing the folds and the scorer, and leaves the graphs free to differ, allowing the experiment to test differences that the component comparisons had excluded.
Breadth has its own trap: the best of several families on two embryos carries a winner's curse, so the choice among families needs its own held-out discipline.

---

## 6. With a Month Left: The Reset, and the Division Channel

### 6.1 The review and the six rules

By 08-28 the board had shown no improvement over $$0.921$$ for twenty-three days. v78 had supplied a sanity check on 08-22, but its displayed score was unchanged.

The review used the exhaustion claim in Section 2.4 to argue for a reset: effort had shifted from prediction quality to maintaining the comparator and its experiment process, while several large historical gains had never been checked on the deployed pipeline.

The apparatus of section 4.3 was retired, and six rules replaced it: the public board is the objective and local out-of-fold is a selection tool; no governance code; the GPU never idles while work exists; two hours from a new idea to the first prediction-facing artifact; a failed run is retried under the same name after the bug is fixed; closed results are never re-derived.
The plan written with them set out to reach the gold band on the board by the deadline, counting up to five submissions a day as nearly free transfer evidence.
That anti-stall rule conflicted with the earlier restriction on using Public for selection. It changed the operating objective; it did not resolve the risk of repeatedly tuning against the same Public set.

### 6.2 The shelved composition, measured the same day

The last finding put Note 4's shelved joint lineage-action mix at the top of the new plan: its replay evidence read $$+0.0144$$ and then $$+0.0313$$, projected at $$+0.005$$ to $$+0.025$$ on the board, but came from the four-fold research replay (base near $$0.74$$), whose four-fold results Note 4 had demoted under the embryo-out contract.
The first measurement under the new rules tested the addition in successive local reconstructions of the submitted Kaggle notebook, using the four labeled example movies:

| rung | regime | control | treatment | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | hold-in, control not faithful to the deployed kernel | 0.8831 | 0.8842 | +0.0011 |
| 2 | hold-in, v78 mixing reproduced; partial count parity | 0.8866 | 0.8858 | -0.0008 |
| 3 | embryo-disjoint detectors, four-fold association heads | 0.8064 | 0.8129 | +0.0065 |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 9%; --c2: 43%; --c3: 16%; --c4: 16%; --c5: 16%; --table-min: 44rem; --label1: 'rung'; --label2: 'regime'; --label3: 'control'; --label4: 'treatment'; --label5: 'delta'" }

Rung 1 did not reproduce the submission's mixing stages. Rung 2 added v78's bidirectional fusion and consensus: the run log reported nearly matching per-movie counts, with exact node counts on two movies. Its control score was still $$0.8866$$ rather than the notebook's $$0.8899$$, so this was a closer replay, not demonstrated exact output parity. The addition scored $$-0.0008$$ within that replay. Rung 3 remained unsuitable for embryo-out inference because its association heads used mixed-embryo folds. I parked the prepared integration on these results; they did not settle the family's effect on new embryos.
The effect of a change $$L$$ depends on its baseline: $$\Delta_L(B)=S(L\circ B)-S(B)$$. The deployed baseline had since acquired test-time augmentation, reverse-time harmonic fusion, consensus, and a different edge threshold. I wrote that it had “absorbed about 75%” of the earlier gain, but the comparisons used different graphs and evaluation conditions. They could not estimate an absorbed fraction. What the new test established was narrower: this prepared addition did not improve the closer v78 replay on the four hold-in movies.

### 6.3 Why the division channel, and what its structure looked like

With a month to the 09-29 deadline, I chose to investigate division candidates inside the pipeline that already met the runtime limit.

Division Jaccard enters the score with weight $$0.1$$: $$S=J^{\mathrm{adjusted}}_{\mathrm{edge}}+0.1J_{\mathrm{division}}$$. For weeks I had treated it mainly as a guard, after one edit cost about $$-0.00018$$. But the division packages in Notes 3–4 had moved Public by $$0.004$$ where the local replay credited about $$0.001$$, and the independent reference in Section 3.2 emitted 384 two-child parents to our 37. I therefore inspected candidate reach on the 199-movie embryo-out graphs. A fork below is a node with two outgoing edges, representing a predicted division.

| quantity | value |
| --- | ---: |
| ground-truth divisions in all 199 movies | 151 |
| fully recoverable (parent and both daughters matched within $$7\,\mu\mathrm{m}$$) | 107 |
| with at least one of the two parent-to-daughter edges already in the graph | 101 |
| predicted forks at annotated parents | 849 |
| of those, sitting on a true division parent | 2 |
{: #biohub-table-4 .biohub-table .biohub-numeric style="--c1: 67%; --c2: 33%; --table-min: 0; --label1: 'quantity'; --label2: 'value'" }

Recoverable true daughter pairs separate at a median of $$10.7\,\mu\mathrm{m}$$, counted false forks at $$5.4\,\mu\mathrm{m}$$, and the deployed rule admitted a fork only when the two putative sisters were within $$8.5\,\mu\mathrm{m}$$ of each other and each within $$4.66\,\mu\mathrm{m}$$ of the parent.
Its sister-distance limit admitted 96.3% of the counted false forks and 29.1% of the true divisions: the distance gate favored these counted false forks over these recoverable true divisions.

### 6.4 A Label-Assisted Division Diagnostic and a Learned Verifier

In oracle mode across all 199 graphs, a daughter could be moved to a new parent (reparenting) and the labels chose the forks.

| arm | score | delta vs 0.6014 | division detail |
| --- | ---: | ---: | --- |
| fork-free base | 0.6005 | -0.0009 | measures the whole deployed division stage at +0.0009 |
| deployed strict stage | 0.6014 | — | the incumbent |
| ground-truth oracle, reparenting allowed | $$\mathbf{0.6571}$$ | $$\mathbf{+0.0557}$$ | $$J_{\mathrm{div}}=0.5577$$; 87 TP, 5 FP, 64 FN; 107 forks added, 37 by reparenting |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 28%; --c2: 16%; --c3: 19%; --c4: 37%; --table-min: 36rem; --label1: 'arm'; --label2: 'score'; --label3: 'delta vs 0.6014'; --label4: 'division detail'" }

![Label-assisted diagnostics and learned gains, with each intervention and comparator identified]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-02-two-ceilings.png)
_Figure 2. Learned gains and label-assisted diagnostics on 199 movies, with the comparator identified for each. The diagnostics use labels at inference and are not deployable methods._

A learned verifier then ranked the reparent-enabled candidates, prefix-pure (the ranker applied to each embryo trained only on the other).
In one day it went from a linear model on geometry ($$+0.0015$$, 6 of the oracle's 87 true positives) to a gradient-boosted ranker with track features ($$+0.0035$$) and then with appearance features ($$+0.0053$$, a score of $$0.6067$$, 11 true positives and 26 false positives).
Two limits were measured the same day: simulated division Jaccard was $$0.43$$ and $$0.67$$ on each ranker's training embryo but $$0.062$$ pooled on the other, evidence of a transfer gap that the reported summaries did not fully separate into ranking, calibration and candidate-population effects; and the candidate generator covered only 75 of the 151 events.

### 6.5 The local gate, and the sanity check

A predeclared hold-in gate checked that division edits would not damage the edge channel: the candidate kernel reproduced the previous four-movie score of $$0.8899$$ with identical per-movie edge counts, its 45 division edits all landing away from the annotated tracks, a property of these four movies and not a general license.

The candidate, v79, went to the board as a sanity check late on 08-28: it had the strongest local support of the window, after earlier division packages had shown larger Public than local deltas.
The one expectation written that day was that the gain would reach the board at a similar order, not damped like edge deltas.
The two division pairs of Notes 3–4, $$+0.000949$$ and $$+0.0010142$$ locally with a $$+0.004$$ Public step each, suggested a direction but supplied no reliable transfer rate.
A clear drop would falsify the candidate; when this window closed, the board had not answered.

---

## Closing

Freezing the comparator made small changes measurable, but it also fixed the detector, nodes and candidate generator under every comparison. When the results went flat, that design could not distinguish a pipeline near its optimum from a test too narrow to reveal alternatives. The two diagnostics measured a filter over selected swaps and one generator's coverage; no complete alternative pipeline was scored.

Next time I would reserve an early comparison for several complete pipelines, using the same folds and scorer while allowing their nodes, links and divisions to differ. Deeper refinement would follow that comparison.

For the remaining month, the concrete next candidate was a verifier over reparent-enabled division proposals: $$+0.0557$$ from a label-assisted diagnostic and $$+0.0053$$ from the learned model. v79 had just been submitted. The August 28 expectation was that its local gain would reach Public at a similar order of magnitude; the result would test that expectation.

<details markdown="1">
<summary>Decision record, evolving criteria and remaining questions</summary>

The tables below retain the period's decisions. The clauses summarize the scope of the evidence; their presence does not certify that every earlier experiment satisfied them.

## 7. Decision Log

Until 08-28 the charter of section 0.1 held: a frozen embryo-out comparator was the selection objective, and its one sanity check (v78) returned the same displayed score.
On 08-28 I made the board the objective as an anti-stall rule, in tension with C9, and sent a division candidate chosen on local evidence (v79) as the second sanity check.

| decision | reason at the time | what came back | what it changed |
| --- | --- | --- | --- |
| A frozen embryo-out comparator (0.6013708666) as the objective, on the one-graph design of 08-05 and 08-11 | after the fold leak, a stable comparator for paired local measurements | nulls or deltas below 10⁻⁴ | what could it pay? |
| Ship the matcher gain (+0.0006686) as v78 (08-22) | sanity check on unseen embryos | 0.921, a tie | the displayed tie cannot establish the local gain's transfer |
| Filter selected actions and inspect candidate coverage (08-24) | after a run of nulls: was there room at all? | +0.000365; 115 of 38,060 missing edges swappable | C10 |
| Look outside the frozen graph (08-20, 08-26) | where else was the room? | imports negative or inert; shared-node Jaccard 0.9978; two-child parents 384 to 37 | node and division differences deserved separate tests |
| Native detector behind readiness floors | a detector that cannot find cells should not spend a scoring run | recall 0.179 and 0.036; neither a score nor a closure | C11 |
| Retrospective review of search scope (09-23) | alternatives were mostly tested as parts of one graph | end-to-end alternatives remained untested | reserve effort for a broader comparison next time |
| Reset: the board as objective; test the shelved +0.0313 composition (08-28) | no displayed improvement over 0.921 for twenty-three days | six rules; addition -0.0008 on the closer v78 replay | an addition's effect depends on its baseline |
| Division: label-assisted diagnostic, then a verifier; submit v79 (08-28) | the largest measured room reachable inside the existing pipeline | diagnostic +0.0557; verifier +0.0053; Public pending | the next sanity check is pending |
{: #biohub-table-6 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: 'decision'; --label2: 'reason at the time'; --label3: 'what came back'; --label4: 'what it changed'" }

### The criterion at the end of this period

| clause | wording | since |
| --- | --- | --- |
| C1 | Separate fitting, calibration and evaluation dependencies, and score the whole graph; movie separation alone does not ensure domain independence | Note 2 |
| C2 | Write each gate down before the result exists | Note 3 (07-15) |
| C3 | Calibrate a rule on the population it will act on | Note 3 |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 | Bind every level and delta to its comparator; do not rank deltas from different replays | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |
| C10 **(new)** | Measure candidate reach; call a diagnostic a bound only within its declared scope | Note 5 |
| C11 **(new)** | A gate must be able to end in a decision | Note 5 |
{: #biohub-table-7 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: 'clause'; --label2: 'wording'; --label3: 'since'" }

---

## 8. What the Period Established

### Established

1. A ground-truth filter over the comparator's 3,178 selected actions was worth $$+0.0003654588$$, all in one prefix, and only 115 of 38,060 missing ground-truth edges (0.30%) had a topology-legal swap.
2. The design of 08-05 and 08-11 kept one detector's graph as the common universe and scored every other family by its marginal value to it.
3. The matcher ($$+0.0006686$$) tied on the board, three representation families selected zero actions, and every fixed-node import of an external model was negative or inert.
4. On four movies, an independent pipeline matched ours on shared nodes (edge Jaccard $$0.9978$$) but differed in which nodes exist (103 annotated cells to our 19) and in two-child parents (384 to 37).
5. A native detector stopped twice at readiness floors that counted as neither a result nor a closure, and the shelved association composition measured $$-0.0008$$ against the closer, still imperfect v78 replay.
6. Of 151 ground-truth divisions, 107 were fully recoverable, and 2 of 849 forks at annotated parents sat on a true division parent; a reparent-enabled oracle was worth $$+0.0557$$, and a prefix-pure verifier reached $$+0.0053$$ in one day.

### Supported but Unconfirmed

1. That changing nodes and division candidates could be useful; the two four-movie comparisons and division diagnostic do not rule out further association gains.
2. That several end-to-end families under one harness would have exposed that room earlier; it reads the mechanism of section 4, and no such portfolio was built.
3. That changes to the deployed baseline reduced the shelved association gain; the comparisons do not separate that interaction from fold leakage or estimate a fraction absorbed.
4. Whether any stable local-to-Public relation exists; the historical $$0.14\times$$ to $$0.9\times$$ range does not establish one.

### Open Questions

1. Which target errors are reachable by each proposed generator, and which label-assisted measurements are actual bounds on its declared interventions?
2. With the board as the objective, what keeps it from being fitted, and how does that rule fit with C9?
3. Does the division gain reach the board undamped, as the 08-28 expectation says?

4. Does the verifier generalize across embryos? With two embryo domains and about 80 labeled positives, "prefix-pure" is two folds.
5. Which strong end-to-end families finish within the runtime limit, and how do they compare under one embryo-out harness, edge and division terms read separately?

</details>

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Three Gaps in Local Validation]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- **Part 5: What a Frozen Graph Left Untested**
