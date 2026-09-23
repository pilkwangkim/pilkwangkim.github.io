---
title: "BioHub Cell Tracking Working Note 4: Why the Largest Local Gain Did Not Show on the Board"
date: 2026-08-16 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, oof, leakage, retraction, hold-in-vs-embryo-out, transfer, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-08-16-biohub-working-note-4/cover.png
  alt: "Title card for BioHub Working Note 4: why the largest local gain did not show on the board"
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

# BioHub Cell Tracking Working Note 4: Why the Largest Local Gain Did Not Show on the Board

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- Korean version: [BioHub Cell Tracking 작업 기록 4: Local Gain이 Public Board에서 보이지 않았던 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)

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

Note 3 ended with an out-of-fold validator that rejected most changes, naming a mechanism each time, and approved a few at $10^{-4}$ on a board that displays $10^{-3}$.
The one approval the board registered, a two-seed division ensemble ($+0.000949$ locally), read $+0.004$ higher along with its single-seed control.
August asked whether local out-of-fold evidence can decide, with the board as a sanity check: a submission that tests whether a local decision holds on unseen embryos.

The largest clean association gain measured locally, $+0.0073$ from reading the association field in both directions, went to the board with a written expectation.
Shipped alone, it read three thousandths below the incumbent (the submission then in place, which carried a division stage) and tied its own control.
The fortnight found three gaps in local validation, each now a clause of the criterion: the board read the division stage; the largest local numbers came from folds holding both embryos; and a detection ceiling that had reset the schedule was measured on movies the deployed model had trained on.

The short version is:

```text
The largest clean local association gain (+0.0073) went out as a sanity check with a written expectation.
Shipped alone it read 0.913 against the incumbent's 0.916; against its own control, a tie.
Reason 1: the board read the division stage (+0.004 on either field), not the association field.
Reason 2: the largest local numbers (+0.0144, +0.0313) came from folds holding both embryos.
Reason 3: a 0.89% detection ceiling was hold-in; out of embryo it was 8.79% on the same movies.
Each reason changed local validation: matched sanity checks, embryo-out folds, no hold-in evidence.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What does local evidence measure, and how is the board read? |
| 1 | Why did the board go first to the division stage? |
| 2 | Reason one: what did the board read when the largest clean local gain shipped? |
| 3 | Reason two: which embryos were in the folds behind the largest numbers? |
| 4 | Reason three: which movies was the detection ceiling measured on? |
| 5 | Can a component's accuracy stand in for the graph? |
| 6 | Can candidates finish, and can checks stay proportionate? |
| 7 | What was decided, why, and what did it change? |
| 8 | Which claims survive, and what remains open? |

---

## 0. What Local Evidence Measures, and How the Board Is Read

Almost every local number here comes from the **research replay**: each of the 199 training movies is predicted by the fold model trained on the other embryo, the full graph construction is replayed, and the official metric, as patched in July, scores the result.
No movie is predicted by a model that has seen its embryo: **embryo-out**, the regime of the hidden test.
**Movie-out** folds hold out movies but train on other movies of the same embryo.

The replay's base sits near $0.74$ and the submitted pipeline, trained on all 199 movies, near the board's $0.9$; with Note 3's seven reference graphs ($0.60$ to $0.74$) still unreconciled, I compare changes inside one setup, never levels across setups.

The **association field** scores candidate edges between cells in consecutive frames.
A **fork** is a predicted cell with two successors, which the metric reads as a division.
The **division stage** is a learned model that decides where a track splits into two daughters, feeding the division term (weight $0.1$ in the score); its **action budget** caps how many divisions it adds.

On 08-01 the incumbent was the forward association graph with a learned division stage, at $0.916$.
The board rounds to three decimals, so a difference of $0.002$ or less is a tie: no failure detected, never a confirmation.

---

## 1. Why the Board Went First to the Division Stage

The July division ensemble ($+0.000949$ locally, $+0.004$ on the board) is one rounded pair, not a rate, but it suggested that the hidden set weighs the division stage far more than the replay does.
The stage's operating points (budget, the weighting of its two models, variants) were a question only the board could price, coarsely, so the first submissions went there.

The first five days used twenty-eight submissions: three on 08-01 (KST) that returned no score (section 6.1), and twenty-five in five portfolios, four on the division stage and one, on 08-03, a designed sanity check (section 2).
The remaining eleven days used two.

| date (UTC) | portfolio | public |
|---|---|---:|
| 08-01, 08-02 | variations around the division stage, ten arms | $0.910$ to $0.916$ |
| 08-03 | association field against division stage, five arms | $0.917$ / $0.913$ / $0.912$ / $0.919$ / $0.920$ |
| 08-04 | division action budget, five arms | $0.920$ / $0.919$ / $0.910$ / $0.919$ / $0.919$ |
| 08-05 | division-rank weights and three structural arms | $0.921$ / $0.918$ / $0.920$ / $0.919$ / $0.919$ |

No arm of 08-01 and 08-02 rose above the incumbent, and three read five or six thousandths lower.
On 08-04 the smallest budget collapsed to $0.910$, removing too much division recall, and the rest tied; 08-05 gave five ties, and its $0.921$ arm, a re-weighting of the two division models, became the incumbent on a narrow local lead in a sweep written down in advance.

The board sees whether a division stage is present (four thousandths, section 2), but adjacent operating points read as ties, and a tie cannot choose.
On 08-10 this became C9: fold-pure out-of-fold evidence selects, and a board submission is a matched sanity check with a written expectation, never a way to choose an adjacent scalar value.

---

## 2. Reason One: The Board Read the Division Stage, Not the Association Field

### 2.1 Reading the association field backwards

The association model scores each candidate edge forwards, from a cell to its successor in the next frame, and the same network can score it backwards; fusing the two harmonically before the graph is built should penalize support found in only one direction.
Four fusion candidates were declared before an exact 199-movie replay, and all four improved the score, by $+0.0033659$ to $+0.0073273$.
The table shows the best of the four, chosen on labeled movies: harmonic fusion at a reverse weight of $0.20$, also the strongest on a 16-movie probe, against the forward pipeline without a learned division stage.

| quantity | forward anchor | harmonic reverse fusion | delta |
|---|---:|---:|---:|
| patched official score | $0.7401015$ | $0.7474288$ | $+0.0073273$ |
| adjusted edge Jaccard | — | — | $+0.0073683$ |
| node recall | — | — | $+0.0172533$ |
| division Jaccard | — | — | $-0.0004099$ |

It improved 119 movies and worsened 80; both embryos gained, $+0.0194505$ on the 71-movie embryo and $+0.0057244$ on the 128-movie one.

The field was not promoted alone: it added $24$ false forks ($563 \to 587$) and no true divisions, because in a builder that lets forks form freely a better field also manufactures divisions.
Protections applied after the fact failed: vetoes on the finished reverse graph drove division Jaccard to zero, and only $29$ of $999$ stored division actions were still valid on the new graph, because division actions are graph-relative.
The follow-up collapsed every fork to its best continuation and gave the division-event models from Note 3 sole authority to re-admit a second daughter.
Division false positives fell from $587$ to $67$ with $99.30\%$ of the edge gain retained, and the composition was promoted at $+0.0106431$ over the forward anchor: the field $+0.0073273$, a plain additive division stage $+0.0010142$, and exclusivity about $+0.0023$ at a larger action budget.

### 2.2 The sanity check of 08-03

The 08-03 portfolio asked whether the field's local gain would hold on unseen embryos, with the division stage present or absent on either field; three arms carried written expectations.
For the reverse field alone the expectation was $0.918$ (interval $0.914$ to $0.921$); a result below that range would reject the fixed deployment, not the replay's evidence for the family.

| arm | association field | learned division stage | expected | public |
|---|---|---|---:|---:|
| forward control (July) | forward | none | — | $0.912$ |
| incumbent | forward | additive | — | $0.916$ |
| reverse field alone | reverse, harmonic | none | $0.918$ | $0.913$ |
| linear fusion alone | reverse, linear | none | $0.917$ | $0.912$ |
| reverse field plus division | reverse, harmonic | additive, budget $0.012$ | $0.920$ | $0.917$ |
| forks collapsed | reverse, harmonic | exclusive, budget $0.024$ | — | $0.919$ |
| forks collapsed, smaller budget | reverse, harmonic | exclusive, budget $0.012$ | — | $0.920$ |

All three expectations were missed low, by three to five thousandths.
The reverse field added one thousandth on either pipeline, where about six had been expected alone; the division stage added four on either field, where the replay had credited it with about one.
Making the stage exclusive at the same $0.012$ budget moved $0.917$ to $0.920$, at the edge of the board's resolution; at twice the budget it read $0.919$, a tie.

![Public scores of four portfolio arms: adding the division stage moved both fields by 0.004, the reverse field moved them by 0.001]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-01-board-read-division.png)
_Figure 1. Public scores of four arms of the 08-03 portfolio. Adding the division stage moved both fields by $0.004$; switching to the reverse field moved either pipeline by $0.001$, a tie at the board's resolution. Locally the field had carried most of the gain._

### 2.3 What the check settled

The arm behind the title shipped the reverse field without the incumbent's division stage and read $0.913$ against $0.916$; since the field alone tied its forward control, the matched arms place that gap with the missing division stage.
The replay's folds were the two embryos, so the miss is not a fold leak.
Why the field did not carry is an untested inference: the replay's graph, from fold models trained on one embryo each, has node recall near $0.90$, and a field that repairs a weak graph plausibly has less to repair in the deployed one, built from both.

At this size the board cannot price the association field; it prices a matched change whose share it resolves, such as the division stage.

### 2.4 A retention gate that measured a component

On 08-04 a sibling composition, the joint association model with the exclusive division stage on top, reached $+0.010205$ over 199 movies.
A gate written before the run rejected it: the composed graph had to keep $95\%$ of the association stage's adjusted-edge gain, since collapsing forks deletes edges, and it kept $91.82\%$.
The gate guarded a share of one term, not the score, and so vetoed a positive total: C4, the Note 3 clause about components, applied to a gate of my own.

---

## 3. Reason Two: The Largest Numbers Came from Folds That Mixed Both Embryos

The fortnight's largest local numbers came from a different replay and never reached the board; what they measured depends on which embryos each fold held.

### 3.1 A first warning, 08-05

In the 08-05 error anatomy (section 4.1), a label-free node-count regressor with $R^2 = 0.471$ under random five-fold cross-validation over movies reached $R^2$ of $0.125$ and $-0.269$ with a whole embryo held out.
The audit named the gap within-embryo leakage and concluded:

```text
Random folds over movies are not a valid protocol in this competition.
```

### 3.2 The largest local numbers of the fortnight

On 08-06 a multi-family composition completed a 199-movie exact replay, mixing a joint lineage-action model, the auxiliary center detector and appearance features into the two-seed association anchor under one authority weight $a$, which sets how much say the auxiliary models get.
At $a = 0.20$ the association stage returned $+0.014446$ over a baseline of $0.7405959$; with the exclusive division stage on top, the sweep reached $+0.031335$ at $a = 0.50$, the upper bound enforced in code, and $+0.034692$ at $a = 1.00$ under an opt-in flag.
Every composition passed every gate, all four outer folds and both embryos positive: at $a = 0.20$ the 71-movie embryo gained $+0.018084$ and the 128-movie embryo $+0.013709$.

The 71-movie embryo, the weaker one in earlier families, was now the stronger, and on 08-06 I wrote that down as a reason for confidence: "Since train and test are embryo-disjoint, a family that no longer depends on one embryo is better evidence for generalisation, not worse."
Four days later my plan cited the composition as a measured gain and set an association target of that size.

### 3.3 What the folds were

Write $D$ for the 199 movies and $D^{(p)}$ for the movies of embryo prefix $p$, the tag in each movie identifier that names its embryo.
The hidden test needs folds that are embryo-disjoint:

$$
D=\bigsqcup_k D_k,
\qquad
\forall k\ \exists\,p_k:\ D_k\subseteq D^{(p_k)}.
$$

The construction used was prefix-balanced:

$$
D=\bigsqcup_k D_k,
\qquad
\forall k,\,p:\quad
\left|D_k\cap D^{(p)}\right|\approx\frac{\left|D^{(p)}\right|}{4}.
$$

Every fold held movies from both embryos: movie-out, which cannot estimate performance on an unseen embryo.

![Schematic of four prefix-balanced folds, each holding both embryos, against two embryo-out folds]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-03-fold-construction.png)
_Figure 2. The four-fold replay behind the fortnight's largest numbers balanced every fold across both embryos. The contract frozen on 08-10 holds out one whole embryo per fold._

Stratifying within the grouping that defines the domain shift is the same operation as ignoring it: every fold trains on the embryo it evaluates.
Stratification, and the label "reciprocal" in their defining document (the project's word for splits that swap whole embryos), made these folds read as better than the random folds of 08-05; the 08-06 reasoning took a premise true of the hidden test to be true of the folds.
The embryo reversal is what a model that has absorbed within-embryo acquisition state would produce: the most economical explanation, not a proof.
The check is one assertion per fold, that it holds out exactly one embryo and trains on none of it; a prefix-balanced split fails it at once.

### 3.4 The audit of 08-10, and the leak it found

On 08-10 I audited the evaluation protocol on the principle that out-of-fold status belongs to the whole composition, every threshold, gate and operating point included.
It found leaks: a deployed gating model had re-split the 199 movies into five movie-level folds, training on other movies of the evaluation embryo; an operating point for pre-solver activation had been chosen while seeing both embryos' labels; and a lockbox existed only on paper.
The audit froze a two-fold embryo-disjoint contract: 177 development movies, and 22 in a lockbox opened once.

The four-fold results were demoted, not retracted, in a sentence that named the contract, not the fold construction: they "remain valid records of those experiments, but they are not evidence under this contract."
With an 08-01 doctrine that a failed drop-in test closes an implementation, never a model family, a demotion naming no fault lets the number stand as a property of the family.
On 08-12 I recorded that the four-fold checkpoints "mixed embryo prefixes", but did not yet apply that to their numbers.

### 3.5 A rerun under the embryo-disjoint contract

On 08-12 the multi-frame parent-or-null family, which picks each cell's parent, or none, from five frames, was retrained and replayed under the new contract against a 177-movie anchor of $0.7428367161$; on the four prefix-balanced folds it had returned $+0.0133995$, positive in every fold and both embryos.
Parent top-1 is the share of cells given the correct parent.

| arm, embryo-disjoint contract | adjusted-edge delta | component metric |
|---|---:|---|
| multi-frame parent-or-null | $+0.0001347503$ | parent top-1 $+0.0101$ |
| two steps: parent-or-null first, then which parent | $+0.0004766233$ | parent top-1 $-0.156$ |

The number fell to about a hundredth, but the rerun changed more than the folds: it ran with a division guard, so its division delta is exactly zero, while the $+0.0133995$ was unguarded and paid for its edges with a division loss of $-0.0046489$.
On the old folds an exact topology guard had already turned an unguarded $+0.0163060$ on eleven movies into $-0.0004684$, in a partial run stopped for futility.
With 177 movies instead of 199, a new seed and a different anchor as well, this pair cannot size the fold inflation; the case against the folds rests on their construction.
Both arms failed their gates, and the family closed.

The $+0.0144$ and $+0.0313$ compositions have not been re-measured under the contract and remain in the plan as demoted numbers; I expect, without a result, that they will shrink.

### 3.6 What a number from leaky folds does

A number from leaky folds does more than overstate a gain: it closes a question, as the word "measured", the target set from it and the reading of the embryo reversal did here.
A leak and a generalizing family can produce the same results, but whether a fold holds out an embryo is a property of the split, checkable before any number exists: clause C7.

---

## 4. Reason Three: A Key Ceiling Was a Hold-In Number

What to test next depends on where the score is locked: in cells the detector never finds, or in links between those it does.
Two measurements in one week, on different movies, pointed opposite ways.

### 4.1 Where the missed cells are, out of embryo

On the 199-movie replay baseline, an error anatomy of 08-05 found node recall $0.8983$: $119{,}759$ of $133{,}318$ annotated cells matched a predicted node inside the $7\,\mu\mathrm{m}$ gate, the radius within which the scorer pairs a prediction with an annotation.
Of the $13{,}559$ misses:

| class | count | share |
|---|---:|---:|
| no predicted node anywhere inside the gate | $13{,}457$ | $99.25\%$ |
| a node inside the gate, lost to assignment competition | $102$ | $0.75\%$ |

Two explanations I held did not survive.
A slightly tight gate: a found cell lands at a median of $2.24\,\mu\mathrm{m}$ from its node, but a missed cell's nearest predicted node sits at a median of $9.60\,\mu\mathrm{m}$, against a median cell spacing of $24.99\,\mu\mathrm{m}$, so that node is a different cell.

![Median distances on one axis: found cell to node 2.24 µm, gate 7 µm, missed cell to nearest node 9.60 µm, cell spacing 24.99 µm]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-02-missed-cell-distances.png)
_Figure 3. Medians from the 08-05 error anatomy of the replay baseline. A missed cell's nearest predicted node sits beyond the $7\,\mu\mathrm{m}$ gate, at a median of $9.60\,\mu\mathrm{m}$: it belongs to another cell._

Crowding: the densest spacing bin has a miss rate of $23.0\%$ but holds only $339$ misses, $2.5\%$ of the total.
The missed cells are not in the point set, and no association model can link a point that does not exist: Note 3's question of which points exist, now with a number.

### 4.2 "Detection is not the bottleneck", and its retraction

On 08-07 an association oracle measured how much of the shipped pipeline's score is locked behind detection: it keeps the predicted nodes, replaces the edges with ground-truth topology projected through the official matcher, and rescores, so an edge it still cannot place has an undetected endpoint.
On the four labeled example movies distributed with the competition, the deployed graph had node recall $0.983$ and only $0.89\%$ of ground-truth edges unreachable, so detection looked non-binding, a detector retrain not worth its compute, and the schedule was changed.

On 08-08 I checked those movies: they are copies of training movies the deployed models were trained on, so the $0.89\%$ is a **hold-in** number, measured on data the model has fit.
The same computation on the same detection family out of embryo:

| regime | movies | GT edges unreachable |
|---|---:|---:|
| deployed graph, hold-in | 4 | $0.89\%$ |
| same detection family, embryo-out | the same 4 | $8.79\%$ |
| same detection family, embryo-out | 199 | $12.10\%$ |

Over 199 movies out of embryo, perfect association lifts the replay's $0.7346$ to $0.9380$: association still holds about $+0.20$ of headroom, and about one ground-truth edge in eight is beyond the reach of any linking model.
The claim was retracted and the schedule decision reversed; the corrected reading is that "association is the largest lever, and detection is probably binding near the top of the board", so part of what the board scores is set by points no association gain can add, an inference like that of section 2.3.

The retraction has a limit: the two extreme rows differ on two axes, hold-in against embryo-out and deployed pipeline against research replay, and the run that would separate them was not run inside this window.
The deployed notebook was the right object and the example movies the wrong sample: a number from movies the model trained on measures fit, not generalization (C8).

### 4.3 The ablation headlines

The same 08-08 review retracted two headlines from a feature-family ablation on 34 movies of the section 3 composition: removing the temporal family "keeps $1.2\%$ of the gain", and the auxiliary center detector alone is "worse than nothing, at $-17.2\%$".
Both were division artifacts: with $TP = 2$ against 89 false positives in the control, one event moved the pooled division Jaccard, $TP/(TP+FP+FN)$ at weight $0.1$, by half its value and swamped the edge differences.
On edges alone the figures are $13.0\%$ and $-4.6\%$; the ranking survived.

---

## 5. Scoring the Graph, Not the Component

A released, pretrained cell-tracking representation was adapted into a parent ranker and measured embryo-out: a routing policy, fitted on one embryo and evaluated on the other, decided cell by cell whether to trust the new ranker or the existing consensus.
Parent top-1 rose by $+0.008366$ ($0.874241 \to 0.882607$), positive in both held-out embryos; the exact graph replay over 177 movies returned $-0.002687$, $69$ movies better and $107$ worse.

The finished graph is not the parent decision but $\hat G = R_{\mathrm{gap}} \circ R_{\mathrm{prune}} \circ \Pi_{\mathrm{parent}}$, and the short-component filter $R_{\mathrm{prune}}$ acts on the connectivity that $\Pi_{\mathrm{parent}}$ produces.
Switching parents let components short enough to prune survive, and gap recovery extended them: the graph gained $9{,}357$ predicted nodes, of which $184$ matched a ground-truth cell.
On switched rows that overlap a sparse parent label, the new model was $899$ decisions more accurate than the consensus it replaced: better at its own task, worse for the graph.

Two arms carrying this representation tied the incumbent on the board on 08-05, so the graph replay decided, and the family closed on 08-10.
C4 was reconfirmed with the path traced: component accuracy and graph score diverge whenever a downstream stage is conditioned on the component's output.

---

## 6. Getting Candidates to Run, and Keeping Checks Proportionate

### 6.1 A candidate has to finish

Note 3 ended with seven blank pre-solver submissions, attributed to runtime, and a plan to cut proposal inference to one model per movie.
On 08-01 three candidates tested that plan at three activation budgets; all ran cleanly on the four example movies, exceeded the hidden runtime limit and returned no score: inserting proposed points forced a second full pass of the association model over every movie.

On 08-06 the limit got the model Note 3 had asked for: a deployment must satisfy $F + m\,k \le 43{,}200$ s, with fixed setup $F$, work $m$ on the four example movies, and a factor $k$ that scales them to the hidden set.
The example movies hold $42\%$ more cells than the average training movie, so $k$ is only a bracket, and the deployed pipeline had between $0\%$ and $+17.6\%$ of spare capacity.
A candidate would spend a slot only if it fit at the pessimistic end; on 08-08 the pre-solver family left the deployment plan on that record, its local signal still positive (C6).

### 6.2 Four notebook runs, four environments

On 08-07 and 08-08 four consecutive Kaggle notebook runs of the runtime probe produced no measurement, each for a reason diagnosable from local artifacts: a package resolved from the wrong attached dataset, a mount path missing from the deployment image, a wrong state-dict key, a construct that Python 3.12 rejects and local 3.11 accepts; each time the local check was the easier environment, so it passed.
The first run wrote every required output and recorded its failure in a field nothing downstream read.
A completed process is not a measurement.

### 6.3 When a check costs more than the question

On 08-10 and 08-11 I wrote a program for the remaining weeks with a Public score of $0.970$ as its outcome target ($0.921$ on the day) and a $480$-hour GPU ladder.
A target in board units turns every board reading into a progress report, in tension with C9 in the same plan.
The program also brought an admission system founded on section 6.2's finding that an exit code is not a scientific decision: eight declarations before every launch, four independent evidence checks, hashed launch receipts.

| date | what followed |
|---|---|
| 08-13 | a measurement run finished all $204$ planned passes and was filed as a runtime failure for exceeding a frozen $3.5$-hour cap by $0.6166$ s |
| 08-15 | a capture chain failed five times in a row, the last because a generic all-finite validator rejected a NaN that the producer defines as "no match" ($1{,}379.575$ s billed) |
| 08-16 | one commit added $1{,}193$ files and $646{,}428$ lines of experiment governance |

Together the rules made asking a question cost more than the answer was worth, and how to keep a check proportionate to its question stayed open.

---

## 7. Decision Log

For five days the board was used heavily on the division stage, whose operating points the replay could not price, and the 08-03 portfolio was a designed sanity check of the largest local association gain.
Adjacent settings returned ties, so on 08-10 the board was narrowed to matched sanity checks with written expectations (C9).

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| 08-01: three pre-solver runs, one proposal model per movie | test July's runtime attribution | no score at any budget | runtime model; family out of deployment (C6) |
| 08-01 to 08-05: portfolios on the division stage | the board weighs that stage; the replay could not price its operating points | adjacent settings tied; one collapse | board limited to matched checks (C9) |
| 08-03: field against division stage, written expectations | separate what transfers | field tied its control; division stage $+0.004$ | the board reads the division stage |
| 08-04: $95\%$ edge-retention gate | protect the association gain | $+0.010205$ rejected at $91.82\%$ | C4 applies to gates too |
| 08-06: authority sweep on four prefix-balanced folds | stratified; every fold and both embryos positive | movie-out folds; a guarded embryo-out rerun of one family kept about a hundredth | C7; four-fold numbers demoted |
| 08-07: detection ceiling on the deployed graph | measure the pipeline that ships | $0.89\%$ hold-in; $8.79\%$ embryo-out | retracted 08-08; C8 |
| pretrained parent ranker, judged on the graph | parent top-1 $+0.008366$ embryo-out | graph $-0.002687$ | closed 08-10; C4 with a mechanism |
| 08-11: admission system | an exit code is not a scientific decision | a complete run filed as failed over $0.6166$ s | checks must fit the question (open) |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 | Write each gate down before the result exists | Note 3 (07-15) |
| C3 | Calibrate a rule on the population it will act on | Note 3 |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 | Compare levels only inside one reference replay; compare deltas across | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 **(new)** | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 **(new)** | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 **(new)** | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |

---

## 8. What the Period Established

### Established

1. Harmonic reverse-time association improved the replay by $+0.0073273$ over 199 embryo-out movies, both embryos positive; best of four predeclared candidates, all positive.
2. On the board the field alone read $0.913$ against a written expectation of $0.918$, a tie with its forward control ($0.912$); with a division stage, also a tie ($0.916 \to 0.917$).
3. Collapsing every fork under an exclusive division stage cut division false positives from $587$ to $67$, retained $99.30\%$ of the edge gain and reached $+0.0106431$ (board: $0.920$ against $0.917$ at the same $0.012$ budget, at the edge of its resolution; $0.919$ at twice the budget, a tie).
4. The division stage moved the board by four thousandths with either field, where the replay credited it with $+0.0010142$: a direction on two rounded pairs, not a rate.
5. Adjacent operating points of the division stage are indistinguishable on the board: four ties and one collapse on 08-04, five ties on 08-05.
6. On the replay baseline, $99.25\%$ of the $13{,}559$ missed annotated cells had no predicted node inside the $7\,\mu\mathrm{m}$ gate.
7. The hold-in detection-locked fraction of $0.89\%$ is $8.79\%$ on the same four movies out of embryo and $12.10\%$ over 199; the original claim is retracted.
8. The ablation headlines were artifacts of a division term with $2$ true positives; on edges the ranking survived.
9. The four outer folds behind $+0.014446$ and $+0.031335$ were balanced within each embryo prefix: movie-out, not embryo-out.
10. The multi-frame family returned $+0.0133995$ unguarded on those folds and $+0.0001347503$ embryo-out with a division guard.
11. A parent ranker $+0.008366$ better on held-out parent top-1 made the 177-movie graph worse by $-0.002687$ ($9{,}357$ nodes added, $184$ matched).

### Supported but Unconfirmed

1. The $+0.0144$ and $+0.0313$ compositions are inflated by within-embryo leakage and will shrink on embryo-disjoint folds; the multi-frame rerun is consistent with this but confounded.
2. The reverse field did not carry to the board because a field that repairs a weak graph has little to repair in the deployed one.
3. The deployed pipeline's detection-locked fraction lies nearer $12.10\%$ than $0.89\%$.

### Open Questions

1. What is the detection ceiling of the deployed pipeline, measured out of embryo?
2. Do the $+0.0144$ and $+0.0313$ compositions survive the embryo-disjoint contract, and at what size?
3. Does any association gain survive that contract at a size the board can display? Every association candidate measured under it in this window came in below $+0.0005$.
4. Is there any stable ratio from local gains to the board? My plan for the remaining weeks quotes $0.14\times$ to $0.9\times$; this note's portfolio fits no single band.
5. Where else in the pipeline is a head still fitted across the embryo boundary?

---

## Closing

The largest clean local association gain, shipped alone, read three thousandths below the incumbent and tied its own control.
Three findings changed local validation: the board read the division stage and tied neighboring settings, so it now checks matched experiments with written expectations (C9); the largest local numbers came from folds holding both embryos, so folds now hold out a whole embryo (C7); and a ceiling that had reset the schedule was a hold-in number, so hold-in numbers are no longer evidence of generalization (C8).

On the replay, $99.25\%$ of the missed cells have no predicted node inside the gate, and every association candidate measured under the embryo-disjoint contract came in below $+0.0005$.
The question handed to the next note is why honest validation finds so little.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- **Part 4: Why the Largest Local Gain Did Not Show on the Board**
