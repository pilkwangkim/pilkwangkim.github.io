---
title: "BioHub Cell Tracking Working Note 3: What the OOF Machine Refused"
date: 2026-07-31 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, oof, cross-fitting, division-recovery, deployment-constraints, working-note]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-07-31-biohub-working-note-3/cover.png
  alt: "BioHub Cell Tracking Working Note 3: What the OOF Machine Refused"
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
- Korean version: [BioHub Cell Tracking 작업 기록 3: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 4: Three Gaps in Local Validation]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)

</details>

<details markdown="1">
<summary>Related public notebooks</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **About this series.** BioHub asks competitors to reconstruct cell lineage graphs from 3D microscopy movies. The labeled training set contains 199 movies from two embryos; Public scores cover 29% of the hidden test set, and Private scores cover the remaining 71%. The hidden test comes from embryos not seen in training. Each note follows the record through its stated period; later findings and retrospective comments are marked separately.
{: .prompt-info }

> **Later context — September 23, 2026.** The ±0.002 Public tie rule used here was adopted on September 5–6. Later audits also distinguished the backbone's embryo-out split from policy folds that held out movies within the same embryos; that distinction was not yet resolved in the July decisions.
{: .prompt-info }

Between July 15 and 31, the OOF program proposed in Note 2 began producing decisions. This was the “OOF machine”: a workflow for predicting, replaying graph stages and applying written gates. Replaying the graph stages improved the raw model output by $$+0.052187$$, but most new interventions failed their gates. Three problems recurred: a rule acted on a different population from the one used to calibrate it; a better component metric produced a worse graph; and gains were measured on incompatible reference graphs.

Four policies passed local gates, each on its own baseline. The division ensemble and its single-seed control both gained $$+0.004$$ on Public, while a proposal-insertion candidate returned no score in seven submissions. The program could test concrete graph changes, but gains measured on seven different reference graphs still could not be ranked directly.

---

## 0. What Would Count, Written Down Before the First Result

On July 15 I set the first promotion rules and a 3+2 submission sequence: an anchor, a candidate and its ablation, then up to two follow-ups. The plan was to build fold-correct predictions, identify a recoverable error, and test one small graph-editing policy. A policy would advance only if the complete graph improved overall, stayed non-negative on both embryos and met the fold and edit-budget checks below. Writing those conditions before the results was meant to keep a promising score from changing its own acceptance test.

The official score is an adjusted edge Jaccard, which also penalizes predicting more nodes than the movie's estimated cell count, plus one tenth of a division Jaccard, both counted only where annotators labeled.
An operator $$R$$ was promoted only if

$$
\Delta S(R)>\delta_{0}
\;\wedge\;
\min_{p\in\{A,B\}}\Delta S_{p}(R)\ge 0
\;\wedge\;
\min_{k}\Delta S^{(k)}(R)\ge 0
\;\wedge\;
|R|\le n_{\max},
$$

where $$\Delta S$$ is the exact change in the official combined score after replaying the full deterministic pipeline, $$p$$ ranges over the two embryo prefixes (a movie's ID prefix names its embryo), $$k$$ over the outer cross-fit folds, and $$|R|$$ is the edit count.
Each check was recorded separately, so each refusal named its reason; the last three did most of the refusing.

On 2026-07-18 the host patched an exploit in the division metric (full leaderboard rescore announced 2026-07-23), and the condition Note 2 described became stricter and local: a parent-side node matched to a ground-truth node, a genuine fork there, two daughter branches that do not immediately merge again, and one-to-one assignment between predicted forks and ground-truth divisions.
Roughly $$0.028$$ came off the top of the board, so part of the gap Note 2 reasoned about was the old metric.
No submission of mine was scored under both versions; from my pipeline's shape I infer that it did not use the exploit, but without an exact rescore I cannot verify that the July readings are all on one unchanged scale.

---

## 1. The First Test: What the Existing Graph Stages Are Worth on Unseen Movies

Note 2 had left open whether the hand-built graph stages were method or a calibration adapted to the board.
On 07-17 I replayed them exactly as submitted (ILP selection, motion reassignment, short-component pruning, one-frame gap recovery, safe division repair) on the epoch-$$100$$ twofold out-of-fold predictions.
The ILP is the solver: an integer linear program that chooses which candidate nodes and edges form the graph.

| quantity | raw model graph | after exact replay | delta |
| --- | ---: | ---: | ---: |
| official score | 0.647453 | 0.699640 | +0.052187 |
| node recall | 0.916820 | 0.911433 | -0.005387 |
| movies improved | — | 183 of 199 | — |
{: #biohub-table-1 .biohub-table .biohub-numeric style="--c1: 34%; --c2: 22%; --c3: 22%; --c4: 22%; --table-min: 36rem; --label1: 'quantity'; --label2: 'raw model graph'; --label3: 'after exact replay'; --label4: 'delta'" }

The stages gained about $$+0.052$$ on these held-out predictions. This supported keeping them as the baseline, while leaving open how their effect would change with a different detector or graph.
They trade nodes for edges, an asymmetry that recurs through this note.

Of roughly $$25{,}150$$ remaining edge false negatives (FNs), $$11{,}020$$ had **both endpoint nodes unmatched**.

```text
An edge FN between two matched nodes is an association error.
An edge FN between two unmatched nodes is a detection error
wearing an association error's clothes.
A ranker restricted to the existing matched nodes cannot recover those edges.
Reaching them requires different detections, coordinates or node proposals.
```

The unmatched endpoints made detection coverage a priority. The replay built at that point cached the detections, however, so it could test changes to links but could not yet regenerate nodes. Section 2 covers those fixed-node experiments; Section 3 follows the detection changes tested while that capability was missing.

A short-track rescue in the same report tested an idea carried since Note 1, that restoring missing nodes might recover true edges. Node recall itself is a diagnostic, not a separate term in the official score.
It restored $$14{,}917$$ nodes and raised node recall by $$+0.00186$$; the official score fell by $$0.000735$$, and $$156$$ of $$199$$ movies got worse.
A restored node can worsen the count adjustment without recovering a true edge; harmful new edges can add a separate cost.

---

## 2. The First Policy Selected by Movie-Level Cross-Fitting

The edge-replacement ranker tested the smallest claim: if a ranker trained out of fold can tell which of the pipeline's edges to replace, the replacements should help on unseen movies in both embryos.
On 07-19 it finished a nested five-fold cross-fit at the fixed $$200$$-epoch capture: fitting the policy, calibrating its threshold and the final evaluation each sat on disjoint movie sets, and thresholds fixed to fold medians meant no test movie needs a fold identity.
The result was $$+0.0002706$$ from $$1{,}447$$ replacements, both embryo prefixes positive.
It was promoted on this separated movie-level policy evaluation. No reliable Public change was expected from such a small delta.

---

## 3. Where OOF Was Blind: Which Points Exist Went to the Board

In the third week of July, the available replay could compare edits only on its cached detections. I used Public submissions to explore changes to the detection field until a local replay could re-extract the nodes (Section 5).

### 3.1 Does a Second Seed Help Through Association?

Note 2 had left open whether a second, independently seeded model adds information that a jointly calibrated mixture can use.
From 07-20 through 07-23 seventeen submissions varied the mixing rule in the association channel: logit mixtures, margin-adaptive mixing, consensus gates and basin searches all read between $$0.905$$ and $$0.908$$ in the submission record (my log reads one of them $$0.909$$).
None was better than the strongest single-seed graph, about $$0.909$$ in my log around 07-19/20, under this series' later reading rule; the weakest read $$0.003$$ to $$0.004$$ lower.
Edge reweighting on a fixed point set went back to the local replay, which measures it at $$10^{-4}$$.
The step of about $$+0.006$$ from the $$0.902$$–$$0.903$$ band where Note 2 ended to that $$0.909$$ is attributable to no experiment in this window.

### 3.2 Or Through the Node Field? An Ablation Pair

On 07-24 the question moved from edges to points: one candidate averaged the two aligned **detection logit fields before peak extraction**, and went in on the same day as its ablation and its brackets, so the board's answer could be attributed.

| candidate | change | Public |
| --- | --- | ---: |
| field average before peak extraction | balanced detection field, dual-seed association retained | 0.911 |
| **balanced field, primary-only association** | **the ablation** | 0.910 |
| primary-weighted field | off-balance toward the primary seed | 0.909 |
| secondary-weighted field | off-balance toward the second seed | 0.907 |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 40%; --c2: 42%; --c3: 18%; --table-min: 0; --label1: 'candidate'; --label2: 'change'; --label3: 'Public'" }

Removing dual-seed association left the balanced field tied under this series' later reading rule. Together with the earlier sweeps, that result made the node field the next comparison to prioritize.

A slightly primary-leaning field (secondary detection weight $$0.475$$ instead of $$0.5$$, same threshold) read $$0.912$$ on 07-25, tied with the balanced field, and became the frozen reference for most of the out-of-fold replays that followed.
A two-sided detection-threshold probe produced the same displayed score. I stopped that sweep, without claiming the underlying response was exactly flat.

### 3.3 What That Use Did to the Submission Budget

The 07-15 budget assumed the board would see only what the local replay had promoted.
With no local instrument for the node field, the board served as one; the gates were checked on every operator, the budget was not, and it lapsed unamended at $$109$$ scored submissions for the month.

---

## 4. Testing Stated Mechanisms: Four Refusals and Where Each Failed

Four of the nine refusals share a pattern: the candidate's component metric improved, and the graph did not.

### 4.1 A Selector Calibrated Where Labels Exist Fired Where They Do Not

**The claim.** A leak-free selector that learns from labeled cases when the second seed's parent is the better one should help wherever it fires.
The dual-seed association selector was strictly model-out-of-fold, cross-fitted, calibrated on $$177$$ beneficial against $$43$$ harmful labeled groups (a group is one cell's candidate parents), and clean under a leakage audit.

**What the replay measured.** It was calibrated where labels exist and deployed where they mostly do not:

$$
\pi_{\mathrm{cal}}
=\frac{249}{122{,}007}
=0.204\%,
\qquad
\pi_{\mathrm{dep}}
=\frac{15{,}062}{276{,}077}
=5.456\%,
\qquad
\frac{\pi_{\mathrm{dep}}}{\pi_{\mathrm{cal}}}
\approx 26.7 .
$$

The exact replay lost $$0.000068190$$ and worsened $$112$$ of $$199$$ movies; no post-hoc fold subset cleared the floor, and none could be deployed, since a test movie carries no fold identity.
To move the official edge counts by $$+16$$ true positives (TP), $$+26$$ false positives (FP) and $$-16$$ FN, the selector changed $$3{,}001$$ raw nodes and $$3{,}450$$ raw edges: the annotated skeleton it was scored on is a thin slice of the graph it rewrites.

**What it established.** The audit did not find a training-overlap violation in this selector, yet its action population differed sharply between calibration and deployment. The firing-rate gap was a diagnostic of that mismatch, not a universal cutoff for rejecting every shifted rule.
That became clause C3: calibrate a rule on the population it will act on.

### 4.2 Picking the Right Parent More Often Did Not Build a Better Graph

**The claim.** A model that picks the correct parent more often on held-out movies should reconstruct a better graph; two models tested that within four days.

| model | component metric | exact graph replay |
| --- | --- | --- |
| temporal-flow gate (switches to a motion model's parent) | parent top-1 0.867186 → 0.879198 (+0.012012); five of five outer folds nonnegative | -0.0007833; 63 improved / 132 worsened |
| higher-order parent matcher (compares candidate parents jointly) | parent top-1 0.867589 → 0.883202 (+0.015613) | +0.000482; one outer fold at -0.000788; one prefix at -0.000280 |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 29%; --c2: 38%; --c3: 33%; --table-min: 0; --label1: 'model'; --label2: 'component metric'; --label3: 'exact graph replay'" }

**What the replay measured.** The temporal-flow gate reversed sign: a component metric that improved on every fold produced a graph worse on two movies in three.
The higher-order matcher stayed positive overall but failed one fold and one prefix. Better parent choices therefore did not suffice to pass the graph-score gates.

**The interaction.** The graph replay exposed a downstream failure that a parent-choice metric misses.
Changing a node's parent changes which connected components exist, and so what short-component pruning removes and what gap recovery may bridge: a locally correct switch can delete a component that carried several correct edges, and the parent-choice metric never sees components.
The gate also raised node recall by $$+0.001145$$ while lowering the score, the same trade as the short-track rescue.

**Checking the refusal.** The higher-order matcher failed its fold and prefix gates and was not promoted.
Its picture was mixed: positive in aggregate, negative in one fold and one prefix. I submitted three variants as sanity checks of that rejection on unseen embryos. The two broadly applied variants scored $$0.004$$ lower than the $$0.916$$ division configuration described in Section 6, consistent with the local rejection.

Together the two models became clause C4: judge a component by the graph it produces, in an exact replay, not by its own accuracy.

### 4.3 The Better Classifier Inserted the Worse Nodes

**The claim.** Of two versions of the candidate-proposal model, which proposes cells the detector missed, the one that classifies better should insert better nodes.
The predeclared ensemble beat the single model on every offline metric I had: known-label AP $$0.9999984$$ against $$0.9999973$$, and recovery of the frozen graph's node misses $$0.53075$$ against $$0.52633$$.

**What the replay measured.** Inserted into the finished graph, the single model was the better one: $$+0.0000506$$ from $$842$$ repairs against the ensemble's $$+0.0000435$$ from $$845$$.
Neither was promoted, and with one run and a gap of $$7\times10^{-6}$$ the ordering is a warning, not a law: classification quality and intervention utility are different objectives, C4 from a different direction.

Two-sided bridge repairs, which require links to existing fragments on both sides, produced zero accepted repairs from $$1{,}354$$ candidate triples. Multi-frame detection gaps were one possible explanation; strict candidate or acceptance conditions could also produce this null result.

---

## 5. Four More Claims About Where Score Was Left, Tested and Refused

The next four candidates placed the missing score elsewhere: in a per-embryo operating point, in division precision, in the solver's objective, in keeping alternative graphs alive.
The ninth refusal, a third division seed, is in section 6.

**A per-embryo detector did not hold across folds.**
A full out-of-fold detection grid, which re-extracts the points for every weight, showed one embryo improving as the secondary detection weight fell while the other weakened, suggesting a per-embryo router.
A nested cross-fit inside each embryo rejected it at $$-0.0003550114$$ (reciprocal cross-fit $$-0.0003884963$$), both embryo-level deltas negative; the development folds chose $$0.40$$ twice and the baseline twice in one embryo, and the opposite extremes, $$0.25$$ and $$0.70$$, in the other.
The two model folds coincide with the two embryo prefixes, so embryo identity cannot be separated from fold identity; the subgroup effect was real as a description but not stable enough to be a rule.

**Emitting forks is not how the division term gets paid.**
The frozen out-of-fold graph already held $$12{,}794$$ predicted binary forks, against $$151$$ annotated divisions in the whole training set, and scored $$4$$ division true positives against $$720$$ false positives: what was scarce was clean forks at a matched parent.
A cross-fitted validator that pruned the weaker daughter edge of low-ranked forks cut division false positives from $$720$$ to $$317$$ and reached $$+0.000328$$, but it cost $$197$$ edge true positives and two of four outer folds were negative: it bought division precision, weighted at one tenth, with edge recall, weighted at one.

**Extra solver-selected forks did not change the scored divisions.**
I added a hyperedge variable to the ILP, forced to equal the conjunction of two daughter edges and rewarded by an out-of-fold division rank score.
As the reward rose from $$1.00$$ to $$2.00$$, the solver selected $$7$$, then up to $$43$$ complete fork events, and the patched division counts stayed at exactly $$5$$ TP / $$507$$ FP / $$146$$ FN at every reward.

```text
The solver chose more forks.
The downstream graph filter removed every one of them.
In this pipeline, what counts as a division is decided after the solver.
```

I stopped this hyperedge composition because its extra solver choices did not survive the downstream filters.

**The candidate union exceeded its declared budget.**
My 07-28 survey (section 6) had ranked this idea first: keep several graph hypotheses alive and let a joint optimizer arbitrate.
Two days later its predeclared gates stopped it: the union of the frozen graph, five other detection blends and the proposal model raised node recall from $$0.943001$$ to $$0.965008$$, but recovered only $$38.61\%$$ of the frozen graph's node misses against a gate of $$40\%$$, and needed $$25.98$$ novel points per frame against a gate of $$6$$.

The union can recover $$1{,}704$$ missed ground-truth edges, but they sit inside $$649{,}702$$ candidates outside the frozen graph.
With fixed ground truth and matching, recovering a missed true edge raises TP and lowers FN, leaving $$TP+FP+FN$$ unchanged; an added false edge increases that denominator. For independent additions under these assumptions, the break-even precision is $$p^{*}=J/(1+J)=0.422315$$ at $$J=0.731046$$. One proposal-source pattern cleared it in both prefixes and all four folds, at $$0.526814$$. That proxy omitted node-count and downstream graph effects, so a complete replay was still needed.
Three fixed rules for admitting new tracklets scored strongly positive on a sparse edge-utility proxy and raised node recall; their official deltas were $$-0.003536$$, $$-0.002364$$ and $$-0.006235$$.
That triple (proxy strongly positive, node recall up, official score down) appeared three times in this window, on unrelated operators, and made it a recurring warning in this set of experiments.

---

## 6. Following the Error Budget to the Division Channel, and Checking It on the Board

On 07-28, with most candidates refused, an error analysis measured the error budget of the frozen graph across all $$199$$ movies:

| term | value |
| --- | ---: |
| official score | 0.725553 |
| edge TP / FP / FN | 109,363 / 20,715 / 19,520 |
| node recall | 0.9477 |
| division TP / FP / FN | 4 / 720 / 147 |
{: #biohub-table-4 .biohub-table .biohub-numeric style="--c1: 50%; --c2: 50%; --table-min: 0; --label1: 'term'; --label2: 'value'" }

Of the edge false negatives, $$5{,}455$$ ($$27.9\%$$) still had both endpoints unmatched.
The division term contributes

$$
0.1\cdot J_{\mathrm{div}}
=0.1\cdot\frac{4}{4+720+147}
=0.00046
$$

out of the $$0.1$$ weight the metric reserves for it; most of the compute had gone to the edge term, so I turned to the division channel.

The same day, the strict-division rank ensemble passed every gate.
Its claim: two independently seeded division-event models make partly different ranking errors, so combining their ranks should place true divisions higher than either alone.
A fixed equal-weight percentile rule combined them, and only a small top fraction of ranked candidates, the action fraction, was added after the frozen graph.

| arm | outer cross-fit delta | folds |
| --- | ---: | --- |
| seed A alone (the single-seed control) | +0.000661 | — |
| seed B alone | +0.000834 | — |
| **two-seed percentile ensemble** | +0.000949 | all four positive; the same action fraction (0.016) in every fold |
| three-seed ensemble | +0.000697 | action fractions 0.064 / 0.032 / 0.032 / 0.024 |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 36%; --c2: 22%; --c3: 42%; --table-min: 0; --label1: 'arm'; --label2: 'outer cross-fit delta'; --label3: 'folds'" }

It recovered nine division true positives at a small edge-Jaccard cost.

**The sanity check.** A first approval in a new channel is where a blind spot of the local criterion would show, so the ensemble went to the board with its single-seed control.
Both read $$0.916$$, $$+0.004$$ over the $$0.912$$ configuration: the board registered the division channel and could not separate the ensemble from its control.
The local division gain was below $$0.001$$, while the Public difference was $$0.004$$. A promoted edge edit of similar local size had left the displayed Public score unchanged (Section 7). That contrast motivated further division experiments, although one rounded pair could not establish a transfer rate or separate the division term from changes to edges.

**The third seed** was diverse (out-of-fold score correlations of about $$0.69$$ and $$0.84$$ against the two promoted models) and positive alone, yet the three-seed ensemble scored below the two-seed one; its folds chose three different action fractions, spanning a factor of nearly three, which raised a stability concern for this selected rule.

---

## 7. Testing Where an Edit Acts: Before or After the Solver

Several failures in Sections 4 and 5 arose after later stages processed the edited graph. The first pre-solver temporal-flow policy provided a concrete example: adjusted edge score rose by $$+0.000365$$, but one true division became false. The weighted division loss of $$-0.000153$$ offset $$42\%$$ of the edge gain.
The claim to test: the value of an edit depends on **where in the pipeline it is applied**.
Writing $$\Pi$$ for the solver,

$$
\Delta S\!\left(\Pi\circ R\right)
\ne
\Delta S\!\left(R\circ \Pi\right).
$$

The candidate-proposal model was evaluated at two insertion points. Post-solver insertion gained $$+0.0000506$$; adding proposals before the solver at $$0.50$$ nodes per frame gained $$+0.0006748$$. The roughly $$13\times$$ numerical difference belongs to these two compositions: their intervention budgets were not matched, so it does not isolate the effect of insertion point.

A zero-budget check reproduced the frozen graph, with maximum edge-probability difference $$5.96\times10^{-8}$$. That checked the integration, not the budget confound. The pre-solver budget sweep then showed why aggregate gain alone was insufficient:

| budget (nodes/frame) | delta | folds |
| ---: | ---: | --- |
| 0.10 | +0.0000776 | two negative |
| 0.25 | +0.0002702 | one negative |
| 0.50 | +0.0006748 | all four positive (promoted) |
| 0.75 | +0.0011047 | all four nonnegative |
| 1.00 | +0.0012498 | one fold at -0.0006285 |
{: #biohub-table-6 .biohub-table .biohub-numeric style="--c1: 24%; --c2: 26%; --c3: 50%; --table-min: 0; --label1: 'budget (nodes/frame)'; --label2: 'delta'; --label3: 'folds'" }

The aggregate is monotone in the budget; the fold picture is not, as with the third division seed.

The track-fragment matcher passed every gate after the solver, at $$+0.0006325$$ from $$3{,}224$$ edge replacements, and its hidden-safe sanity check read $$0.912$$, the score of the configuration it was added to: a tie, no failure detected.
Moved before the solver, its outer cross-fit was $$+0.0000178$$, indistinguishable from zero (the two figures sit on reference graphs whose node recall differs by roughly six points, so they are not a ratio).
Moving the division hyperedge of section 5 earlier had no effect.

Pre-solver insertion can make new nodes available to the optimizer. That motivated its use here, but differences in baselines, budgets and later filters prevented the edge and division comparisons from isolating the effect of placement.

---

## 8. Checking the Structural Approval on the Hidden Set: Seven Blank Submissions

Pre-solver activation (adding proposed points before the solver runs) passed the local composition gate in section 7; the next question was which budget survives on unseen embryos.
Five candidates returned no score: the first two failed with an unhandled error on hidden data, the other three completed with no score.

An audit found five deterministic failure paths that could fire only on hidden data, from a key error on an unseen embryo prefix to movie length inferred from the last detected node, each invisible on the public example movies, which are copies of training movies.
All five were patched and two budgets resubmitted; both again completed with no score, seven blank submissions by the end of July.

With the patched outputs valid on every movie I could see, runtime was a remaining hypothesis: the public four-movie run alone took roughly $$65$$ minutes, about $$30$$ of them in four-fold proposal inference, and that cost grows with the number of hidden movies (an arithmetic attribution, not an isolated measurement).
The next deployment I have planned cuts proposal inference to one model per movie.

```text
The local replay asks: is this policy better on held-out data?
The hidden set also asks: can it finish, inside twelve hours,
on an unknown number of movies from an unseen embryo?
The second question has veto power.
```

That became clause C6: a candidate must finish on the hidden set within the time limit.

---

## 9. Before Comparing the Approvals: What Each Was Measured Against

Ranking the four approvals needs a shared reference.
By the end of July seven $$199$$-movie local reference graphs were in simultaneous use, never reconciled: $$0.6006730$$, $$0.6529429$$, $$0.6996400$$, $$0.7255533$$, $$0.7347169$$, $$0.7395609$$ and $$0.7405959$$; node recall across them spans roughly $$0.880$$ to $$0.948$$.
On graphs that far apart the same edit has a different denominator, and the available errors and downstream interactions differ. An operator may benefit from a weaker reference, but that is not guaranteed:

| promoted policy | measured delta | reference graph it was measured on |
| --- | ---: | ---: |
| edge replacement, nested five-fold | +0.0002706 | 0.6529429 |
| track-fragment matcher, post-solver | +0.0006325 | 0.6006730 |
| strict-division rank ensemble | +0.000949 | 0.7255533 |
| pre-solver candidate activation, 0.50/frame | +0.0006748 | 0.7405959 |
{: #biohub-table-7 .biohub-table .biohub-records style="--c1: 48%; --c2: 24%; --c3: 28%; --table-min: 0; --label1: 'promoted policy'; --label2: 'measured delta'; --label3: 'reference graph it was measured on'" }

![Four promoted policies plotted above the seven 199-movie local reference graphs they were measured on]({{ site.baseurl }}/assets/img/posts/2026-07-31-biohub-working-note-3/fig-01-four-baselines.png)
_Figure 1. Each July promotion was measured against a different 199-movie local reference graph. Each delta is a change to its own graph, so the four cannot be ranked against each other._

The fragment matcher's figure is the only one measured on the weakest graph, and the data cannot separate policy quality from choice of reference.
That motivated C5: bind each score and delta to its comparator. Neither absolute levels nor paired deltas from different graph environments provide a direct ranking of the policies.

The policy evaluations above held out movies, with both embryo prefixes represented in each fold. The backbone predictions used two embryo-separated folds. Few experiments logged elapsed time, which also made the next comparison difficult to budget.

---

## Closing

The useful output of July was a set of specific failure modes. A selector could satisfy its split checks yet act mostly outside the population used for calibration. Better parent choices could interact badly with pruning and recovery. A positive delta could depend on a reference graph unlike the one ultimately deployed.

Seven unreconciled reference graphs remained the central comparison problem. Public supplied a limited division check, while seven unscored submissions left the proposal candidate's hidden-test performance unknown. The next deployment plan reduced proposal inference to one model per movie; comparing further local gains would also require a common replay baseline.

<details markdown="1">
<summary>Decision record, evolving criteria and remaining questions</summary>

The tables retain the period's decisions. Criteria are summarized retrospectively at the scope the evidence supports: a failed composition does not reject every use of its model family.

## 10. Decision Log

Candidates were decided by the rule written on 07-15: exact score change above a floor, both embryos and every outer fold nonnegative, bounded edits, and fitting, calibration and evaluation on disjoint movies.
Over $$109$$ scored submissions the board answered what the local replay could not yet pose, above all which points exist, and served as a sanity check of local verdicts.

| decision | reason at the time | what came back | what it changed |
| --- | --- | --- | --- |
| 07-15: write gates and budget before any result | a gate set after seeing the result can favor it | nine refusals on predeclared grounds; the budget went untracked | C2 |
| 07-17: exact replay of the graph stages on OOF | know their held-out value before replacing any | +0.052187; 11,020 edge FNs with both endpoints unmatched | turned the program to which points exist |
| 07-20 to 07-25: two-seed and node-field questions to the board | the local replay froze the point set; Note 2 left the two-seed question open | mixes 0.905–0.909; field average 0.911, ablation 0.910 | node field to the board for now; edge reweighting to the local replay |
| refuse selector, flow gate, parent matcher, proposal ensemble | gates written before each run | 26.7× firing rate; sign reversal; parent-choice gains did not satisfy graph-score gates; probes 0.004 lower | C3, C4 |
| 07-28: division ensemble with its single-seed control | first division approval (+0.000949, 4/4 folds); sanity check | both 0.916, +0.004 | motivated further division experiments |
| 07-30/31: pre-solver activation at several budgets | strongest structural result; sanity check | seven submissions, no score | C6 |
| record each delta with its reference graph | a delta depends on the graph it changes | seven references; approvals unrankable | C5 |
{: #biohub-table-8 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: 'decision'; --label2: 'reason at the time'; --label3: 'what came back'; --label4: 'what it changed'" }

### The criterion at the end of this period

| clause | wording | since |
| --- | --- | --- |
| C1 | Separate fitting, calibration and evaluation dependencies, and score the whole graph; movie separation alone does not ensure domain independence | Note 2 |
| C2 **(new)** | Write each gate down before the result exists | Note 3 (07-15) |
| C3 **(new)** | Calibrate a rule on the population it will act on | Note 3 |
| C4 **(new)** | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 **(new)** | Bind every level and delta to its comparator; do not rank deltas from different replays | Note 3 |
| C6 **(new)** | A candidate must finish on the hidden set within the time limit | Note 3 |
{: #biohub-table-9 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: 'clause'; --label2: 'wording'; --label3: 'since'" }

---

## 11. What the Period Established

### Established

1. Exact replay of the deterministic graph stages on held-out predictions is worth $$+0.052187$$ over the raw model graph.
2. Much of the edge-FN residual is unreachable by association: $$11{,}020$$ of about $$25{,}150$$ have both endpoints unmatched.
3. A strictly out-of-fold selector can still fail on population: firing rate $$0.204\%$$ calibrated, $$5.456\%$$ deployed.
4. Better parent top-1 accuracy did not ensure a better graph: the flow gate lowered the graph score, and the higher-order matcher failed one fold and one embryo gate.
5. Three tested node-restoration operators raised node recall and lowered the official score.
6. In the tested hyperedge composition, extra solver-selected forks did not change final division counts after filtering.
7. The division term is nearly unused: $$0.00046$$ of its $$0.1$$ weight, on a graph carrying $$12{,}794$$ forks against $$151$$ annotated divisions.
8. The tested pre-solver proposal composition gained about $$13\times$$ as much as the post-solver one, with different budgets; the comparison did not isolate placement.
9. A policy can pass every gate and return nothing from the board: seven submissions, no score.

### Supported but Unconfirmed

1. The node field was a more promising next test in this window.
2. The board registers the division channel where a promoted edge lever of similar local size did not: ensemble and control both read $$0.916$$; the fragment matcher ($$+0.0006325$$) read its base configuration's score.
3. Classification quality is not intervention utility; the evidence is one run and a gap of $$7\times10^{-6}$$.
4. Fold-level operating-point stability may help assess a candidate alongside mean delta; this window did not validate a superior selection rule.
5. Under the fixed-matching edge-addition calculation, $$p^{*}=0.422315$$ is the break-even precision; full graph changes require additional terms.

### Open Questions

1. Which policy gains remain when the models, fitting rows and calibration data all exclude the evaluation embryo?
2. Can the seven local reference graphs be reconciled into one replay universe?
3. How do local deltas relate to board deltas, per channel, with more than one point?
4. Is the both-unmatched residual reachable by a detector change at all?
5. What is a policy's inference budget, as a number, before it is proposed?

</details>

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- **Part 3: What the OOF Machine Refused**
