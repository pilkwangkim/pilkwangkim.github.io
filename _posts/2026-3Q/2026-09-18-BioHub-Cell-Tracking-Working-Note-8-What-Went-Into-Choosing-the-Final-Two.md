---
title: "BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two"
date: 2026-09-18 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, final-selection, stage-jitter, registration, association-head, embryo-out, selection-bias, oof, working-note]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-09-18-biohub-working-note-8/cover.png
  alt: "BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two"
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
  - [Working Note 5: What a Frozen Graph Left Untested]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
  - [Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
  - [Working Note 7: When the Same Code Was Not the Same Experiment]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- Korean version: [BioHub Cell Tracking 작업 기록 8: 최종 제출을 고를 때 고민한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two-KR/)

</details>

<details markdown="1">
<summary>Related public notebooks</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **About this series.** BioHub asks competitors to reconstruct cell lineage graphs from 3D microscopy movies. The labeled training set contains 199 movies from two embryos; Public scores cover 29% of the hidden test set, and Private scores cover the remaining 71%. The hidden test comes from embryos not seen in training. Each note follows the record through its stated period; later findings and retrospective comments are marked separately.
{: .prompt-info }

Note 7 showed that embryo-out (EO) and kernel-regime (KR) validation could disagree. During September 13–18 I had to choose two submissions; the better score on the remaining $$71\%$$ Private test share would count. v92 added stage-jitter registration, and v93 replaced its association head. Keeping v93 required accepting a failed KR gate and a loss on the four labeled notebook examples.

I selected v93 and v92. Both retained registration, so this pair protected against a harmful head refit while sharing the risk that registration would not help the hidden embryos. Eleven later lines of work produced no additional admitted candidate.

---

## 0. The Pipeline, and Which Instrument Answers Which Question

Two fused detectors find the nuclei in every frame; a transformer **association head** scores candidate links from each nucleus to the next frame's nuclei, called forward and in reverse with the two calls fused; an ILP picks a consistent set of links; deterministic stages, among them a motion relink and a line-fit smoother, repair the graph; and a fitted **division verifier** adds a fork wherever its score clears $$0.90$$, at most $$50$$ per movie.
The score is adjusted edge Jaccard plus $$0.1$$ times division Jaccard.
The deployed baseline was v90: v87 with Note 7's two validity repairs, $$0.946$$ on the board.

Three evaluation setups served different roles. I compare candidate and parent within each setup; I do not treat their score levels as interchangeable.

| instrument | what it runs | role |
| --- | --- | --- |
| EO (embryo-out) | detection and association gradients exclude the scored embryo; checkpoint selection and downstream stages are not fully separated; 199 movies (44b6: 71, 6bba: 128) | efficacy; the deciding gate |
| KR (kernel regime) | the deployed weights and exact notebook code over the same 199 movies | harm detection in the submitted pipeline; in-sample |
| the four example movies | the notebook's actual output, scored locally | deployment parity only |
{: #biohub-table-1 .biohub-table .biohub-records style="--c1: 20%; --c2: 56%; --c3: 24%; --table-min: 0; --label1: 'instrument'; --label2: 'what it runs'; --label3: 'role'" }

EO uses the deployed-stack replay of Notes 6–7 with embryo-out weights. It is not a fully independent whole-pipeline evaluation: backbone checkpoints were selected on evaluation embryos, and some downstream assets used all-train fits. These limitations apply to v90, v92 and v93.

**LB90** is the lower bound of a 90% bootstrap interval over movies. It describes stability under that resampling scheme, not uncertainty across unseen embryo domains.

---

## 1. What Public Could Still Answer Near the Deadline

Since 09-06 I had treated a Public difference within $$\pm 0.002$$ as a tie under a project rule, and the board was cited only for moves of about $$0.003$$ or more.
v91, v90 with the secondary detector's features averaged over test-time views, went in on 09-13 as a sanity check of a local null, a Public submission that checks whether a local decision holds on unseen embryos.
It came back at $$0.946$$, a tie with v90.
That evening I narrowed the board's role again, in a short written rule:

```text
One submission per validated candidate.
Read its Public score for one thing: a drop of 0.003 or more below the deployment parent.
A drop starts a verification, not a decision.
No card carries a band of the form "a transfer if the score is at least X".
Deployment decisions come from the pre-registered local gates: EO and KR.
I designate the final two, and the highest Public score is never the selection rule.
```

The local replays could now answer many questions I had previously sent to the leaderboard. I therefore stopped using small rounded differences to rank candidates. Repeatedly selecting whichever submission cleared a favorable score band would still tune the project to the same Public subset.
What it still did well was catch a collapse like v88's (C19).

---

## 2. What Counts as Evidence: All 199 Movies, Not Small Screens

Note 7 had closed the division ranking lanes without knowing where the missed divisions sat; on 09-13 I traced the existing 199-movie division replay, which used v87's verifier recipe and the older raw-graph/ShiftCorr path, over all 199 movies. This was a separate baseline from the registration replay in Section 3.
Of $$151$$ annotated divisions, $$23$$ were recovered, $$72$$ had no correct candidate, and $$56$$ had a correct candidate scored below the $$0.90$$ threshold.
In $$46$$ of the $$56$$ the best correct candidate already ranked first among its own parent's candidates.
I hypothesized a calibration error, possibly from training labels that did not match the current graph's candidates; if so, a refit on valid labels should lift more true divisions over the threshold.

I kept the threshold fixed to isolate the refit. The earlier v85 submission had combined new labels with a lower threshold and lost $$0.005$$ on Public. That did not identify the threshold as the cause, but it made another threshold reduction less attractive without a way to explain the earlier loss.
On four movies chosen for their events the refit gained $$+0.01099$$ from one extra true division; with one event carrying that number, I judged it on all 199.

| Measure | 4 movies | 199 movies |
| --- | ---: | ---: |
| official Δ | +0.01099 | -0.00679 |
| division TP/FP/FN | +1 / +2 / -1 | 23/37/128 → 22/196/129 |
{: #biohub-table-2 .biohub-table .biohub-numeric style="--c1: 34%; --c2: 28%; --c3: 38%; --table-min: 0; --label1: 'Measure'; --label2: '4 movies'; --label3: '199 movies'" }

The refit left node counts and node recall unchanged and admitted more false divisions without recovering additional true ones. It therefore failed the calibration hypothesis for this candidate and pipeline.

That afternoon a second small screen reversed: the quantile alignment Note 7 had left unresolved, meant to redeploy the pseudo-label detector, gave $$+0.104$$ and $$-0.0016$$ on a two-movie smoke test, then $$-0.033027$$ on a 29-movie kernel-regime panel.
Its motivating signal, a 6bba tail median of $$+0.066$$ on the embryo-out replay, had the opposite sign in the kernel; C14 caught it before any notebook was built.

A third warning was the teacher-clean pseudo-label control discussed in Note 7: $$-0.0206$$ on four movies. It did not rerun the earlier positive compositions unchanged, so it could not assign their gains to teacher leakage.

I then required the declared 199-movie panel for adoption (C18). Small screens remained useful for feasibility and intervention checks. Enlarging the panel improved coverage, but reusing all 199 movies did not create independent evidence.

---

## 3. The First Candidate: Registering Stage Jitter (v92)

### 3.1 Had Every Possibility Been Examined?

The same day I asked whether every possibility had been examined, and reviewed every open idea.
Of $$30$$ new proposals, twenty-eight closed, and the other two were one mechanism; this covered the recorded proposal list, not every feasible method under those constraints.

### 3.2 The Mechanism

Between some frames the microscope's stage shifts and the whole cloud of cells moves a few micrometres at once: **stage jitter**, here $$3$$ to $$9\,\mu\mathrm{m}$$ when it happens, with $$11.6\%$$ of training transitions moving by at least $$3\,\mu\mathrm{m}$$.

Two deployed stages mishandle those frames.
The motion relink links a $$z$$-neighbor instead of the true successor, and the line-fit smoother then reverts the jump frame's coordinates.
The fix estimates one global translation per transition from the detections alone and feeds it into the relink and the line-fit.
The registration step fits no additional model and uses no labels at inference; the trained detectors and association head remain those of v90.
Because the error belongs to the stage, the mechanism predicts more than a pooled gain: gain should tend to increase with jump frequency when motion estimation is accurate and the same failure occurs. Transfer to another embryo remained a hypothesis.

### 3.3 Gates Written Before the Results

| gate | registered minus v90 |
| --- | --- |
| G3: kernel regime, 29-movie panel | +0.0254, no severe harm |
| G4: kernel regime, 199 movies | +0.026578 (44b6 +0.023806 / 6bba +0.027151) |
| G5: embryo-out, 199 movies | +0.013526 (44b6 +0.012265 / 6bba +0.013666) |
| G6: the notebook itself | four example movies 0.889473 → 0.961506 |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 36%; --c2: 64%; --table-min: 0; --label1: 'gate'; --label2: 'registered minus v90'" }

It was the project's first candidate positive in both regimes and both embryos, without an additional fitted component.
The gain also followed jump frequency, as the mechanism predicts: in the kernel regime, movies with at least $$15$$ jumps of $$3\,\mu\mathrm{m}$$ or more had a median gain of $$+0.039$$, jump-free movies $$0.000$$.
There were two reasons for caution. The EO gain was about half the in-sample kernel-regime gain. More troublingly, the division term moved $$-0.0047$$ in EO (TP $$36 \to 26$$), while kernel-regime division TP rose by $$6$$. I disclosed this unexplained sign difference before submission.

### 3.4 The Sanity Check

v92, backed by the strongest local evidence the project had produced, went in on 09-13 as a sanity check.
Its card predicted $$+0.002$$ to $$+0.012$$ on the board; the anomaly-only rule replaced that band after the submission and before the score existed.

It came back at $$0.945$$ against v90's $$0.946$$: no anomaly under the rule adopted before the score arrived. The rounded reading could not distinguish a small hidden gain, infrequent stage jumps, or an edge gain offset by a division loss.

On 09-14, 10,000 Private-sized subsamples of all 199 movies produced no negative draws in either regime. The draws sampled 71% of each embryo's movies without replacement. A separate stratum of 64 EO movies with at most two jumps of $$3\,\mu\mathrm{m}$$ or more lost $$0.003284$$. In that stratum, adjusted edges gained about $$0.0015$$ but division Jaccard fell about $$0.0481$$. The measured loss came from the division term.

The 41 movies with no jumps were not negative: $$+0.002375$$ in EO and $$+0.000063$$ in KR. Low jump frequency alone therefore did not establish a loss.

Thus the full-panel resampling did not rule out a loss on a jump-poor hidden population. Section 9 returns to that shared risk.

---

## 4. The Second Candidate: Refitting the Association Head (v93)

H1 asked whether an association head trained on the candidates the inference pipeline actually produces would link better than the deployed one.
It kept the detector bit-for-bit and retrained only the association head, warm-started, with detections matched to annotations as the metric matches them and negatives where the true successor is known.
v93 is v92 with this head swapped in.

The embryo-out check has two halves: K4 scores the $$128$$ 6bba movies with a head trained without them, and **K5** covers all 199 by adding the reverse split ("reciprocal").
K5's clauses: a pooled gain of at least $$+0.004$$, both embryos non-negative, LB90 above zero, division TP no worse than the old head's minus three, and no excess of per-movie drops.

| gate (written before results) | measured |
| --- | --- |
| K5: EO, 199 movies, reciprocal | +0.010347 (44b6 +0.01341 / 6bba +0.00983); division TP $$26 = 26$$, FP 38 → 31 |
| attribution (descriptive, after K5) | same-budget refit with the original training rule +0.0081; new rule over that refit +0.0023 |
| KR: deployed backbone, all-train head, 199 movies | pooled +0.00003 (44b6 +0.0017 / 6bba -0.00025): stop |
{: #biohub-table-4 .biohub-table .biohub-records style="--c1: 38%; --c2: 62%; --table-min: 0; --label1: 'gate (written before results)'; --label2: 'measured'" }

H1 passed all five K5 clauses and stopped in the kernel regime, where division Jaccard fell from $$0.1879$$ to $$0.1784$$; the card's verdict, `stop_kr`, went into the record.

### 4.1 Why the Regimes Split, and the Exception

The attribution row suggests why the regimes split: of the $$+0.0104$$ gained in EO, $$+0.0081$$ came from refitting on the inference pipeline's candidates at all, even under the original rule.
The hidden test is neither regime exactly: new embryos, through an all-train backbone.

I granted a disclosed exception for the failed KR clause, for this exact head only, for four reasons:

1. The deployed head had already been fitted on all 199 movies. I hypothesized that another in-sample refit offered little new benefit, whereas EO's $$+0.0081$$ refit contribution could matter on unfamiliar embryos. This was an explanation to test, not a demonstrated reason to dismiss the KR result.
2. The hidden set contains unseen embryos, making EO more relevant on that axis, despite its checkpoint-selection exposure: $$+0.01035$$, both embryos positive, LB90 $$+0.0061$$.
3. KR read about zero, not the v88-type collapse it exists to catch.
4. Every other release check stayed binding, and the stop label stayed.

Before release v93 scored $$0.92754$$ on the four example movies against v92's $$0.96151$$ ($$-0.034$$), almost all from one movie that lost one true division and gained one false one.
Those movies are in-sample with three division events, a weak predictor; the exception had been granted without them, so I took it again with them in view and kept it for exactly v93.

### 4.2 The Reading

v93 went in on 09-15 and, before its score came back, became the parent for new candidates on its local evidence.
It came back at $$0.949$$ against an anomaly line of $$0.942$$: no anomaly.

v93 and v92 differed only in the head, and the $$+0.004$$ Public direction agreed with EO. The September 15 record treated this as an observation under C19; the local K5 result and KR exception had preceded the score.

---

## 5. Eleven Further Questions After H1

Eleven lanes followed H1, each asking where room might remain.
A successor head had to beat H1 by at least $$+0.002$$ pooled, and from 09-17 each new lane opened only after the previous result was read.
None produced an advancing candidate: nine stopped at bars written before their results, D-0 was a post-result diagnostic, and GF0 advanced only to a design.

| lane | question | written stop | answer |
| --- | --- | --- | --- |
| H2 | Does three times the training help? | beat H1 by +0.002 | -0.002339 against H1 |
| S0 | Does a label-assisted secondary-margin probe clear its bar? | probe gain ≥ +0.008 | +0.002368 |
| D-0 | Where do H1's extra false divisions come from? | none (diagnostic) | inside the noise |
| H3 | Does a training term for the reverse call help? | K4 ≥ +0.002 | +0.00096 |
| H4 | Does averaging two seeds help? | beat H1 by +0.002 | +0.000716 |
| GF0 | What drives H1's remaining link errors? | rescues outnumber harms in both embryos | mostly position |
| GO1 | Does a learned position corrector help? | 16-movie panel, both embryos ≥ 0 | +0.012533, 44b6 -0.001389 |
| A1 | Does a next-position model beat a placebo? | model minus placebo ≥ 0 | -0.0486 KR, -0.0662 EO |
| CE1 | Does averaging H1 and H2 help? | beat H1 by +0.002 | -0.000223 |
| D2 | Can a mitosis image score rank missed divisions? | rank AUC ≥ 0.75, both embryos | 0.6922 / 0.5752 |
| GO2 | Does GO1 hold on all 199 movies? | K5 | +0.002100, LB90 -0.000556 |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 10%; --c2: 35%; --c3: 27%; --c4: 28%; --table-min: 36rem; --label1: 'lane'; --label2: 'question'; --label3: 'written stop'; --label4: 'answer'" }

![Six lanes plotted as their pooled change over the H1 head on embryo-out, each with its own bar]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-01-lanes-versus-h1.png)
_Figure 1. Six lanes shown as pooled EO changes against H1. H3's $$-0.0087$$ is a descriptive full-EO result; the table reports its stopping K4 result, $$+0.00096$$, against that gate's reference. S0 is a label-assisted margin probe under the existing fusion rule. These endpoints and bars are not interchangeable._

### 5.1 What the Answers Narrowed

**The tested head extensions did not advance.**
Longer training (H2) helped the head trained on the large embryo and hurt the one trained on the 71-movie embryo.
The tested CE1 blend did not improve on H1. Correlated errors were a possible explanation, not isolated by that comparison.
H4 was opened after H3 closed, as one more lane while time remained before the 09-29 deadline.

**The tested secondary-head fusion offered little measured room.**
S0 raised secondary logits for known correct parents in columns where the primary already chose the correct parent. It retained the existing low-margin consensus rule: blend only when the primary and aligned secondary agree on the winning parent. The measured gain thus concerns confidence reinforcement under that rule, not a bound on a new secondary head or a fusion rule that can change the winner.

**This learned-motion candidate did not help.**
In $$94.35\%$$ of rows the nearest detection is already the true successor, so A1's pre-training check had mostly measured re-detection; on the hard rows the model predicted no motion.

**A diagnostic pointed elsewhere.**
GF0 substituted annotated positions for predicted ones: net top-1 rescues of $$+282$$ in 6bba and $$+48$$ in 44b6, mostly from position, licensed one position corrector (Section 8).

---

## 6. Checking a Local "No" (v94)

H4 failed its advance gate on 09-16: $$+0.000716$$ against $$+0.002$$.
A local "no" is also a prediction, and a blind spot of the local criterion can hide behind it as well as behind a "yes"; H4's small gain came from divisions, where the board and the local replay had disagreed before.

I built its deployment candidate past the advance gate as a recorded exception, with a harm branch fixed before any number existed: a failed kernel-regime check would stop the build for a decision.
The check failed one clause of thirteen (seven movies dropped by more than $$0.02$$ against H1, two rose), and its pooled $$+0.00039$$ came from the division term.
I submitted it once, as a diagnostic: with its gates failed it could promote nothing, but a move of $$0.003$$ or more either way would have sent me looking for what the local criterion had missed.

v94 came back at $$0.948$$ against v93's $$0.949$$, a tie, consistent with the local verdict; a tie does not show equivalence, but this check found no blind spot.
With two gate failures, v94 stayed out of the final picks.

---

## 7. Was There Room Left in the Divisions?

On 09-17 D1 asked how many of the missed divisions a label-assisted relaxation within the verifier's existing candidates could reach, on the embryo-out graphs of the H1 head.

| oracle | Δ | division TP / FP |
| --- | ---: | --- |
| label-assisted recall-oriented diagnostic | +0.024351 | TP 26 → 89, FP 31 → 78 |
{: #biohub-table-6 .biohub-table .biohub-numeric style="--c1: 44%; --c2: 23%; --c3: 33%; --table-min: 0; --label1: 'oracle'; --label2: 'Δ'; --label3: 'division TP / FP'" }

Of $$151$$ annotated events, $$89$$ are reachable and $$62$$ are not.
All $$63$$ reachable events still missed are blocked by the $$0.90$$ threshold, none by the cap or conflict rules.
The diagnostic demonstrates reachable missed events. Its FP count rises from 31 to 78, so it is neither perfect-precision scoring nor a global optimum over division policies.

D1 also corrected the 09-13 reading: the right candidate usually ranks first within its parent but not across the movie, where the blocked events' best correct rows have a median rank of $$296$$.
The refit in Section 2 had increased false positives, and the cross-movie ranks now showed why within-parent ranking alone was insufficient. This pointed toward improving the ordering across parents before lowering the threshold. I retained $$0.90$$ as a conservative choice; v85 had not isolated the threshold's effect.

D2's image mitosis score failed its rank-AUC bar before any fit (Section 5), which closes one ranking signal at its current training support, not the division family.

D3 classified the $$62$$ unreachable events by first failure.

![The 151 annotated divisions split into 26 recovered, 63 reachable but below threshold, 30 detection, 25 gate, 7 structural]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-02-division-151.png)
_Figure 2. Where the 151 annotated divisions stop on graphs from fold-trained H1 heads corresponding to v93's head recipe, not the all-train weight bytes shipped in v93. All $$63$$ reachable misses sit below the $$0.90$$ threshold. The label-assisted recall diagnostic gains $$+0.024351$$ while also adding false positives; it is not a trained candidate or a global upper bound._

The gate-blocked events need parent gates of $$12.14$$ to $$20.75\,\mu\mathrm{m}$$, and $$8$$ exceed the existing gate by at most $$1\,\mu\mathrm{m}$$.
A $$13\,\mu\mathrm{m}$$ gate would admit all eight nearby misses. Recovering all eight gives an arithmetic gain of about $$+0.0044$$; applying the current recovery rate of $$26/89$$ gives roughly $$+0.001$$. The new candidates need not have the same recovery rate, so this was only a planning estimate. I did not use these observed misses to tune a larger gate without a separate test.
The gates stayed closed.

A review of the remaining ideas found no other swap that could be built in the days left, fit the kernel runtime budget and had a measured positive effect. One candidate remained to test at full scale: the position corrector.

---

## 8. The Last Buildable Swap: the Position Corrector on All 199 Movies

GO1 adjusts predicted node positions at inference without annotation.
On its 16-movie panel it gained $$+0.012533$$, but 44b6 was negative, so under the panel's both-embryos rule it stopped as unresolved, and about half of the pooled figure was one division flipping from missed to recovered.

The panel's stop clause did not allow a 199-movie confirmation, so GO2 ran one on all 199 embryo-out movies as a recorded exception, under the stricter K5, with its verdict to end the lane.

| Measure | pooled | 44b6 | 6bba |
| --- | ---: | ---: | ---: |
| GO2 minus H1, EO 199 | +0.002100 | +0.000860 | +0.002293 |
{: #biohub-table-7 .biohub-table .biohub-numeric style="--c1: 40%; --c2: 20%; --c3: 20%; --c4: 20%; --table-min: 36rem; --label1: 'Measure'; --label2: 'pooled'; --label3: '44b6'; --label4: '6bba'" }

Two clauses failed (LB90 $$-0.000556$$), and I accepted the stop.

The $$20$$ movies whose GO1 output had already been inspected gained $$+0.008900$$; the other $$179$$ gained $$+0.001328$$, a factor of $$6.7$$ smaller. That gap warns about selection on promising examples, but all 199 movies had already supported project decisions—thirteen evaluations by GO2—so neither group estimates performance on new embryos independently.

A second recorded exception allowed a kernel-regime smoke test of the all-train corrector. Its purpose was to check execution with the submitted weights and features, despite the failed efficacy gate.
On the first movie of a two-movie smoke test it moved all $$26{,}356$$ nodes to the $$7\,\mu\mathrm{m}$$ bound of its output: its input normalizer had been fitted on embryo-out features, unlike the kernel's.
It repeated v88's pattern, positive on the embryo-out replay and broken in the kernel, caught this time before a notebook existed.
With no lane open and no other buildable swap, I closed the project on 09-18.

---

## 9. Choosing a Pair That Hedges the Head Refit

The better Private score of the two picks counts, so I wanted the candidate with the strongest local evidence plus a candidate that removes one important source of risk. With only two slots, that choice also leaves shared risks.

### 9.1 Each Candidate's Evidence

Each local number is measured against its parent in the lineage v90, v92, v93.

The Public column records the submission-time anomaly checks; the second column identifies each local comparator.

| Candidate / role | Comparator | Local evidence | Public |
| --- | --- | --- | --- |
| v93<br>primary | v92 | EO +0.010347, LB90 +0.0061, K5 pass;<br>KR +0.00003 (a stop, shipped under the exception);<br>four example movies -0.033968 | 0.949, no anomaly |
| v92<br>hedge | v90 | EO +0.013526, KR +0.026578, both embryos positive in both;<br>four example movies +0.072033;<br>no negative draws when resampling from all 199 movies; loss in the jump-poor EO stratum;<br>no additional fit for registration | 0.945, no anomaly |
| v90<br>hedge alternative | reference | the baseline;<br>alternative if the low-jump loss transfers to hidden embryos (EO subset: v92 minus v90 -0.003284, from the division term) | 0.946 |
| v94<br>excluded | v93 / H1 | advance $$+0.000716 < +0.002$$;<br>a kernel-regime clause failed | 0.948 |
{: #biohub-table-8 .biohub-table .biohub-records style="--c1: 17%; --c2: 13%; --c3: 53%; --c4: 17%; --table-min: 36rem; --label1: 'Candidate / role'; --label2: 'Comparator'; --label3: 'Local evidence'; --label4: 'Public'" }

### 9.2 The Risk Axes

The decision record named three risk axes: how often the hidden embryos jump, which separates v90 from v92 and v93; registration's division sign flip between regimes, which v93 inherits from v92; and whether the H1 head transfers, which separates v93 from both.

![Matrix of three risk axes against v90, v92 and v93, marking which candidate each risk would hurt]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-03-risk-axes.png)
_Figure 3. Exposure to the three risk axes in the decision record, not a prediction of loss in every scenario. Few jumps can reduce registration's benefit; the no-jump subsets were not negative. The selected pair shares both registration risks, while only v93 carries the head-refit risk._

v92 retains the original association head and registration. It avoids the specific H1 transfer risk: fold-trained H1 heads improved EO, whereas the all-train H1 head was neutral in KR and lost score on the notebook examples. The broader EO limitations were described in Section 0.

The pair shares the jump and division-flip axes: if the hidden embryos rarely jump and the embryo-out division loss appears there too, both picks suffer together.
v90 would remove registration from one slot and split all three risk axes. I chose v92 because registration improved the full local panel in both regimes and both embryos, giving that evidence more weight than v90's broader protection. Both selected submissions therefore remained exposed to a jump-poor hidden population and the unresolved division loss.

### 9.3 What the Final-Pick Memo Actually Used

The September 18 memo said below its evidence table that Public was for collapse detection only and that its column must not rank candidates. Yet it called slot A “v93 (effectively automatic)” and listed v93's advantages in this order: the highest Public score, $$0.949$$; passage of K5; unchanged detections relative to v92; and eleven later lanes without a better admitted candidate. The conflict with C19 was present within the decision document itself.

The memo also offered a local reason to retain v93: give the positive 199-movie EO comparison more weight than the neutral in-sample KR result and the four-example loss. That weighting was a judgment about unseen embryos, made with only two embryo domains and unresolved checkpoint-selection and downstream-fitting dependencies. It supported retaining v93 as one of the two candidates; it did not make the choice automatic. The Private comparison against v92 would test the direction of the head-refit effect.

The other slot decisions followed the local gates and the risks of each candidate. **v94 at $$0.948$$ was excluded** because it failed its advance and KR gates. **v92 at $$0.945$$ was retained** to preserve registration while removing the head-refit risk.

### 9.4 What Private Will Test, Written Before It

Private is unknown as I write this. It will test:

- **Whether either pick collapses.** A collapse would trigger investigation of the shared registration risks, head transfer and any unmodeled hidden conditions; the score alone could not distinguish them.
- **The sign of v93 minus v92.** EO predicts about $$+0.010$$; KR predicts a tie. Positive beyond $$\pm 0.002$$ would be consistent with the head's EO gain transferring. It would not prove why KR was neutral. A tie would not separate the two readings and would give the exception no support. Negative would mean v92 outscored v93 on that Private set, so retaining v92 helped; the cause would still need investigation.
- **Whether v92 scores above v90,** as both local regimes predict over all movies. If not, possible explanations include a different jump distribution or a division loss: the risk both picks share.

---

## Closing

I selected v93 and v92. H1 passed K5's five criteria in the embryo-out evaluation but needed a KR exception; registration improved both full local regimes while leaving a division sign difference unresolved. Private is not yet known. I will read it first for collapse, then for the sign of v93 minus v92, and finally against v90 to assess the risks the selected pair shares.

<details markdown="1">
<summary>Decision record, evolving criteria and remaining questions</summary>

The tables below retain the period's decisions. The clauses summarize the scope of the evidence; their presence does not certify that every earlier experiment satisfied them.

## 10. Decision Log

From 09-13, EO supplied the efficacy gates, KR checked deployment harm, and the four Public readings from v91 to v94 flagged no anomaly. Four decisions overrode written stops—v93, v94, GO2 and GO2's deployment test—with the original verdict and the reason for each exception retained.

| decision | reason at the time | what came back | what it changed |
| --- | --- | --- | --- |
| 09-13: the board becomes an anomaly detector | local instruments answer its old questions; do not rank candidates by differences within ± 0.002 | four reads, no anomaly | C19 |
| 09-13: the division refit judged on 199 movies | its four-movie gain was one event | -0.00679; FP 37 → 196 | C18 |
| 09-13: v92 submitted as a sanity check | the strongest local evidence so far | 0.945, no anomaly | v92 stands on local evidence |
| 09-15: v93 under a disclosed exception to `stop_kr` | KR may understate a refit gain on new embryos; EO +0.01035, LB90 +0.0061; KR not a collapse | 0.949, no anomaly | v93 the parent, on K5 |
| 09-15 to 09-18: eleven lanes | test whether any further idea held up against H1 under the same gates | none advanced | no further admitted candidate |
| 09-16: v94 submitted as a diagnostic | a local "no" is a prediction too | 0.948, a tie with v93 | v94 excluded |
| 09-17: GO2 opened under an exception | the 16-movie panel could not decide | +0.002100; unseen +0.001328 | stop accepted |
| 09-18: final picks v93 and v92 | v93: Public high, K5 pass and retained detections; v92: hedge against the head-refit risk | Private unknown | C20 |
{: #biohub-table-9 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: 'decision'; --label2: 'reason at the time'; --label3: 'what came back'; --label4: 'what it changed'" }

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
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings; narrowed by C19 from 09-13 | Note 4 (08-10) |
| C10 | Measure candidate reach; call a diagnostic a bound only within its declared scope | Note 5 |
| C11 | A gate must be able to end in a decision | Note 5 |
| C12 | Validate on the pipeline that ships: replay it exactly | Note 6 |
| C13 | Use a stage-removal probe to measure its package effect; isolating a metric term needs extra assumptions | Note 6 |
| C14 | Measure in the kernel regime (shipped weights and code); score the notebook's own output against its parent before submitting | Note 7 |
| C15 | Run the leak-free control before trusting a headline | Note 7 |
| C16 | One axis per submission, or a matched control arm | Note 7 |
| C17 | Labels follow the scorer's convention; measure the labeler against ground truth first | Note 7 |
| C18 **(new)** | Small panels screen for support; adoption is decided on all 199 movies | Note 8 |
| C19 **(new)** | EO decides through K5; the kernel regime detects harm; Public detects anomalies only | Note 8 (09-13) |
| C20 **(new)** | Final picks: the strongest local candidate, plus a hedge that fails in a different way | Note 8 |
{: #biohub-table-10 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: 'clause'; --label2: 'wording'; --label3: 'since'" }

---

## 11. What the Period Established

### Established

1. In the older v87/ShiftCorr replay, $$46$$ of $$56$$ low-scored true divisions already ranked first within their parent; a valid-label refit gained $$+0.01099$$ on four movies and $$-0.00679$$ on 199.
2. Label-free stage-jitter registration gains $$+0.026578$$ in the kernel regime and $$+0.013526$$ in EO, both embryos positive in both.
3. The H1 head refit gains $$+0.010347$$ in EO and $$+0.00003$$ in the kernel regime.
4. None of the eleven lanes after H1 produced an advancing candidate.
5. On the H1 embryo-out graphs the $$151$$ annotated divisions split $$26$$ recovered, $$63$$ below $$0.90$$, $$30$$ lost upstream, $$25$$ gate-blocked and $$7$$ impossible; the recall-oriented label-assisted diagnostic gains $$+0.024351$$ with more false positives.
6. GO2 gained $$+0.008900$$ on the $$20$$ movies whose output had been seen and $$+0.001328$$ on the $$179$$ unseen, a factor of $$6.7$$.
7. The position corrector saturated every node of the kernel-regime movie it was evaluated on, consistent with a normalizer fitted on other features.

### Supported but Unconfirmed

1. That the head refit benefits unfamiliar embryos more than the in-sample regime. The attribution comparison supports that hypothesis.
2. That the risk the two picks share lies on the jump and division-flip axes.
3. That another viable candidate would require more time or a different search than the one completed.

### Open Questions

1. Does v93 minus v92 on Private carry EO's sign ($$+0.0103$$) or KR's (about $$0$$)?
2. Does either selected submission collapse on Private?
3. How often do the hidden embryos jump, and does v90 end above v92?
4. Can any signal improve division ranking toward the missed events exposed by the $$+0.024351$$ diagnostic?
5. How much did thirteen measurements on one population inflate the lanes that survived?

</details>

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Three Gaps in Local Validation]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- [Part 5: What a Frozen Graph Left Untested]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
- [Part 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- [Part 7: When the Same Code Was Not the Same Experiment]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- **Part 8: What Went Into Choosing the Final Two**
