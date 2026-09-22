---
title: "BioHub Cell Tracking Working Note 4: When the Largest Local Gain Hurt the Board"
date: 2026-08-16 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, oof, leakage, retraction, hold-in-vs-embryo-out, transfer, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-08-16-biohub-working-note-4/cover.png
  alt: "Title card for BioHub Working Note 4: when the largest local gain hurt the board"
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

# BioHub Cell Tracking Working Note 4: When the Largest Local Gain Hurt the Board

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- Korean version: [BioHub Cell Tracking 작업 기록 4: 가장 큰 로컬 이득이 리더보드에서는 해로웠다]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 3 ended with an out-of-fold machine that could say no.
It refused most of what it was shown, on grounds written down before each experiment ran.
Almost every refusal had one cause: a rule had been measured on a different population from the one it would act on.
What it approved was measured in units of $10^{-4}$, on a board that displays $10^{-3}$.

The rule I carried into August was short.
Local out-of-fold evidence decides.
That went one step past Note 3, whose machine could refuse but not yet choose.
The board is consulted rarely, and only to ask whether a change selected locally survives on embryos the models have never seen.
In the first five days I did not keep the second half of that rule, and section 0 records how.

The first structural question I gave the machine was to read the association field backwards in time as well as forwards.
It produced the largest clean local association gain the project had measured.
Shipped on its own, in place of an incumbent that carried a division stage, it scored three thousandths lower, and five below the expectation I had written down before submitting.
Against its own forward control it was the same score.
Falsifying a large local move that was predicted in advance is the one job the rule gives the board.

The more expensive failure came from inside.
A few days later, larger local gains arrived, positive in every fold and in both embryos, and I read their pattern as evidence of generalization.
Every one of those folds held movies from both embryos.
An internal criterion with a leak is worse than no criterion.
A blank invites a measurement.
A confident number from the wrong folds gets a plan built on it.

The short version is:

```text
Local out-of-fold evidence decided this fortnight; the board was used as a falsifier.
The largest clean local association gain (+0.0073) left no trace on the board by itself.
What the board could read was the division stage around the field, not the field.
Larger local gains (+0.0144, +0.0313) came from folds that held both embryos in every fold.
One family, rerun on embryo-disjoint folds with a division guard, kept about a hundredth of its gain.
Three locally approved candidates returned nothing: they could not finish in twelve hours.
A leaky internal criterion is worse than none: it replaces a blank with a confident number.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What rule was in force, and how was the board used in the first five days? |
| 1 | What did the board read when the largest clean local gain was shipped? |
| 2 | Where do the missed cells actually live? |
| 3 | What did the two retractions of 2026-08-08 change? |
| 4 | What were the folds behind the largest numbers, and what did a rerun show? |
| 5 | Why did a more accurate parent model make the graph worse? |
| 6 | What did deployment and the new process machinery cost? |
| 7 | What rule did this fortnight actually select by? |
| 8 | Which claims survive, and what remains open? |

---

## 0. Where the Fortnight Started

Almost every local number in this note comes from what I call the research-replay universe.
Each of the 199 training movies is predicted by a model that did not train on it, the full graph construction is replayed, and the official metric, as the host patched it in July, scores the result.
Its base sits near $0.74$, while the submitted pipeline uses models trained on everything and scores on the $0.9$ scale.
Note 3's seven reference graphs, spanning $0.60$ to $0.74$, were still unreconciled; this replay sits at the top of that range.
I compare deltas inside each universe, never levels across them.
The one exception is the hold-in ceiling of section 3.1, measured on the deployed notebook.

On 08-01 the incumbent on the board was the forward association graph with a learned division stage on top, at $0.916$.
The first five days used twenty-eight submissions: three in the first hours of 08-01 (KST) that returned no score, covered in section 6.1, and twenty-five in five portfolios.
The remaining eleven days used two.

| date (UTC) | portfolio | public |
|---|---|---:|
| 08-01, 08-02 | variations around the division stage, ten arms | $0.910$ to $0.916$ |
| 08-03 | association field against division stage, five arms | $0.917$ / $0.913$ / $0.912$ / $0.919$ / $0.920$ |
| 08-04 | division action budget, five arms | $0.920$ / $0.919$ / $0.910$ / $0.919$ / $0.919$ |
| 08-05 | division-rank weights and three structural arms | $0.921$ / $0.918$ / $0.920$ / $0.919$ / $0.919$ |

The board rounds to three decimals, and I treat a difference of $0.002$ or less as the same score at the board's resolution.
Read that way, none of the ten arms of 08-01 and 08-02 rose above the incumbent, and three read five or six thousandths lower: two single-seed arms and a portability diagnostic.
The 08-03 portfolio is section 1.
The 08-04 portfolio is four ties and one collapse at $0.910$, where the smallest action budget removed too much division recall.
The 08-05 portfolio is five ties.
Its $0.921$ arm, a re-weighting of the two division models, became the incumbent because it read highest.
Under my rule that thousandth was not a reason, and the local support was only that its weighting narrowly led a sweep written down in advance.

On 08-10 the rule was written into the plan.
Fold-pure out-of-fold evidence selects, and a Public submission is a matched transfer experiment, never used to choose an adjacent scalar value.
Most of the 08-04 and 08-05 slots had gone to adjacent scalars.

---

## 1. The Largest Clean Local Gain, on the Board

### 1.1 What reverse-time association bought locally

The association model scores each candidate edge forwards, from a cell in one frame to a cell in the next.
The same network can score it backwards, from the later cell to the earlier one.
Fusing the two harmonically before the graph is built penalizes support that exists in only one direction.

The measurement was an exact replay over all 199 movies.
Each movie was scored by the checkpoint whose fold held it out, and the two folds are the two embryos.
Four fusion candidates were declared before the replay, and all four improved the score, by $+0.0033659$ to $+0.0073273$.
Harmonic fusion at a reverse weight of $0.20$ was the strongest, as it had been on a 16-movie probe, so the number below is the best of four, chosen on labeled movies.
The anchor is the forward pipeline without a learned division stage.

| quantity | forward anchor | harmonic reverse fusion | delta |
|---|---:|---:|---:|
| patched official score | $0.7401015$ | $0.7474288$ | $+0.0073273$ |
| adjusted edge Jaccard | — | — | $+0.0073683$ |
| node recall | — | — | $+0.0172533$ |
| division Jaccard | — | — | $-0.0004099$ |

It improved 119 movies and worsened 80, the per-movie median was positive, and both embryos gained: $+0.0194505$ on the 71-movie embryo and $+0.0057244$ on the 128-movie one.

The machine still did not promote the field on its own.
Fusion added $24$ false forks ($563 \to 587$) and no true divisions.
A better association field, in a builder that lets forks form freely, buys edges and also manufactures divisions it cannot justify.
The follow-up collapsed every fork to its best continuation and gave the division-event models from Note 3 sole authority to re-admit a second daughter.
It cut division false positives from $587$ to $67$, retained $99.30\%$ of the edge gain, reached $+0.0106431$ over the forward anchor, and was promoted.
Of its total, the field carried $+0.0073273$.
A plain additive division stage added $+0.0010142$, and making the stage exclusive added about $+0.0023$ more, at a larger action budget.

### 1.2 What the board returned

The 08-03 portfolio was built to separate the field from the division stage, and its first three arms carried written expectations.
For the reverse field alone the expectation was $0.918$, with an interval of $0.914$ to $0.921$.
A result below that range was to reject the fixed deployment, not the replay's evidence for the family.

| arm | association field | learned division stage | expected | public |
|---|---|---|---:|---:|
| forward control (July) | forward | none | — | $0.912$ |
| incumbent | forward | additive | — | $0.916$ |
| reverse field alone | reverse, harmonic | none | $0.918$ | $0.913$ |
| linear fusion alone | reverse, linear | none | $0.917$ | $0.912$ |
| reverse field plus division | reverse, harmonic | additive, budget $0.012$ | $0.920$ | $0.917$ |
| forks collapsed | reverse, harmonic | exclusive, budget $0.024$ | — | $0.919$ |
| forks collapsed, smaller budget | reverse, harmonic | exclusive, budget $0.012$ | — | $0.920$ |

All three written expectations were missed on the low side, by three to five thousandths.
The expectation for the reverse field alone implied that the field would add about six thousandths over its forward control.
It added one, the same score at the board's resolution.
Added to a pipeline that already had the division stage, the reverse field moved $0.916$ to $0.917$, another tie.

What the board could read was the division stage.
Adding it moved the forward graph from $0.912$ to $0.916$ and the reverse graph from $0.913$ to $0.917$, four thousandths both times, where the replay had credited it with about one.
At the same $0.012$ action budget, making the stage exclusive moved $0.917$ to $0.920$, at the edge of what the board resolves; at twice the budget it read $0.919$, a tie.
Locally the field carried most of the composed gain; on the board it is invisible.
The arm that shipped it without the incumbent's division stage scored three thousandths lower, and the field's own share of that is a tie.

![Public scores of four portfolio arms: adding the division stage moved both fields by 0.004, the reverse field moved them by 0.001]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-01-board-read-division.png)
_Figure 1. Public scores of four arms of the 08-03 portfolio. Adding the division stage moved both fields by $0.004$; switching to the reverse field moved either pipeline by $0.001$, a tie at the board's resolution. Locally the field had carried most of the gain._

This is the board doing its proper job.
A large local move was predicted before submission, and the prediction failed where it mattered.
The replay's folds were the two embryos, so I do not read the miss as a leak.

### 1.3 What the local number measured

The replay measured a different quantity, in a different graph universe.
Its fold models were each trained on one embryo, and its base graph has node recall near $0.90$.
The deployed graph is built by models trained on both.
My best explanation is that a field which repairs a weak graph has less to repair in a strong one.
That is an inference.
Nothing in this window measured the reverse field inside the deployed pipeline on held-out embryos.
It is the question Note 3 ended on, asked this time of a gain.

### 1.4 A fork rule is a construction rule, not a filter

Before the exclusive division stage I tried to get the same protection after the fact, with vetoes on the finished reverse graph.
Of $14{,}670$ forks it produced, $37$ survived a veto that allowed forks only where the anchor graph had divided, and a stricter veto left none that matched a ground-truth division.
Official division Jaccard fell to zero ($0$ true positives, $1$ false positive, $151$ false negatives) while the edge gain survived.
A stored list of $999$ selected division actions, each an instruction to add a second daughter edge at a given node, fared no better: $29$ were still valid on the new graph, and applying them moved the score by $-0.0000068$.

```text
Division actions are graph-relative.
The candidate universe has to be rebuilt on the graph it will act on.
```

---

## 2. Where the Missing Cells Live

On 08-05 I ran a full error anatomy over all 199 movies of the replay baseline.

Node recall was $0.8983$: $119{,}759$ of $133{,}318$ annotated cells matched a predicted node inside the $7\,\mu\mathrm{m}$ gate.
Of the $13{,}559$ misses:

| class | count | share |
|---|---:|---:|
| no predicted node anywhere inside the gate | $13{,}457$ | $99.25\%$ |
| a node inside the gate, lost to assignment competition | $102$ | $0.75\%$ |

The audit killed two stories I had held.

The first was that the gate is slightly too tight.
When the detector finds a cell, it lands at a median of $2.24\,\mu\mathrm{m}$.
A missed cell's nearest predicted node sits at a median of $9.60\,\mu\mathrm{m}$, against a median cell spacing of $24.99\,\mu\mathrm{m}$.
That node is a different cell.

![Median distances on one axis: found cell to node 2.24 µm, gate 7 µm, missed cell to nearest node 9.60 µm, cell spacing 24.99 µm]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-02-missed-cell-distances.png)
_Figure 2. Medians from the 08-05 error anatomy of the replay baseline. A missed cell's nearest predicted node sits beyond the $7\,\mu\mathrm{m}$ gate, at a median of $9.60\,\mu\mathrm{m}$: it belongs to another cell._

The second was that misses concentrate in crowded tissue.
The densest spacing bin has an elevated miss rate of $23.0\%$, but it holds only $339$ misses, $2.5\%$ of the total.

One reading remains.
The missed cells are not in the point universe, and no association model can link a point that does not exist.

The same audit tested a label-free regressor that predicted each movie's node count from graph statistics.
Under random five-fold cross-validation over movies it reached $R^2 = 0.471$.
With a whole embryo held out it reached $0.125$ in one direction and $-0.269$ in the other, and on the 71-movie embryo the correction was worse than none.
The audit named the random-fold result within-embryo leakage and concluded:

```text
Random folds over movies are not a valid protocol in this competition.
```

---

## 3. Two Retractions in One Day

### 3.1 "Detection is not the bottleneck"

On 08-07 I computed an association oracle on the graph the project actually submits.
It holds the predicted nodes fixed, replaces the edges with ground-truth topology projected through the official matcher, and rescores.
On the four example movies distributed with the competition, the deployed graph had node recall $0.983$ and only $0.89\%$ of ground-truth edges were unreachable.

My reasoning was direct: if fewer than one ground-truth edge in a hundred is out of reach, detection is not binding, and a detector retrain is not worth its compute.

It survived one day.
The four example movies are copies of training movies, and the deployed models were trained on them.
The $0.89\%$ is a hold-in number.
I ran the unchanged computation on the same detection family out of embryo:

| regime | movies | GT edges unreachable |
|---|---:|---:|
| deployed graph, hold-in | 4 | $0.89\%$ |
| same detection family, embryo-out | the same 4 | $8.79\%$ |
| same detection family, embryo-out | 199 | $12.10\%$ |

That is a factor of ten on the identical four movies.
Over 199 movies out of embryo, perfect association lifts the replay's $0.7346$ to $0.9380$.
Association still holds about $+0.20$ of headroom, and about one ground-truth edge in eight is beyond the reach of any linking model.
The schedule decision was overturned.
The corrected reading, in the document's own words, is that "association is the largest lever, and detection is probably binding near the top of the board."

The retraction went above the original reasoning, and it has its own limit.
The two extreme rows differ on two axes at once, hold-in against embryo-out and deployed pipeline against research replay, and nothing on disk separates them.
The deployed pipeline's detection ceiling is unmeasured.
The run that settles it, the deployed pipeline with embryo-disjoint detectors on held-out movies, was named on 08-08 and not run inside this window.

### 3.2 The ablation headlines

The same review broke a second pair of sentences.
I had written them a day earlier as the headline of a feature-family ablation run on 34 movies of the section-4 composition at weight $0.50$, inside its four-fold universe.
Removing the temporal family "keeps $1.2\%$ of the gain", and the auxiliary center detector alone is "worse than nothing, at $-17.2\%$".
Both were division artifacts.
The score adds a tenth of the pooled division Jaccard, $TP/(TP+FP+FN)$, to the adjusted edge Jaccard, and on the 34-movie subset the control graph held two division true positives against 89 false positives.
With $TP = 2$, one event moves the division Jaccard by half of its own value, and the $0.1$ weight swamps the arm-to-arm differences on edges.

On the edge axis alone, where the denominators are in the thousands, removing the temporal family keeps $13.0\%$ of the gain, and the center detector alone keeps $-4.6\%$.
The ranking survived inside that universe, and the two sentences were deleted.

---

## 4. The Folds Behind the Largest Numbers

### 4.1 What was recorded

On 08-06 a conditional multi-family composition completed a 199-movie exact replay.
It mixed a joint lineage-action model, the auxiliary center detector and appearance features into the two-seed association anchor, under one weight $a$ that sets how much say the auxiliary models get.
At $a = 0.20$ the association stage returned $+0.014446$ over a baseline of $0.7405959$.
With the exclusive division stage of section 1 composed on top, a sweep of $a$ reached $+0.031335$ at $a = 0.50$, the upper bound enforced in code, and $+0.034692$ at $a = 1.00$ under an opt-in flag.
Every composition in the sweep passed every gate and was marked for submission.

These results were positive in all four outer folds and in both embryos.
At $a = 0.20$ the 71-movie embryo gained $+0.018084$ and the 128-movie embryo $+0.013709$.
The 71-movie embryo, the weaker one in earlier families, was now the stronger, and the record read that reversal in the composition's favor.

Four days later, the plan cited this replay as a measured local gain and set an association target of the same size.

### 4.2 What the folds were

The documents that produced those numbers state the construction plainly: four deterministic outer folds, balanced separately within each embryo prefix, the tag in every movie identifier that names its embryo.

Write $D$ for the 199 movies and $D^{(p)}$ for the movies of embryo prefix $p$.
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

Every fold held movies from both embryos.
That is movie-out inside seen embryos, not embryo-out, and it cannot estimate performance on an embryo nobody has trained on.

![Schematic of four prefix-balanced folds, each holding both embryos, against two embryo-out folds]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-03-fold-construction.png)
_Figure 3. The four-fold replay behind the fortnight's largest numbers balanced every fold across both embryos. The contract frozen on 08-10 holds out one whole embryo per fold._

The 08-06 decision gives its reason in writing: "Since train and test are embryo-disjoint, a family that no longer depends on one embryo is better evidence for generalisation, not worse."
The premise is true of the hidden test.
The decision assumed it was true of the folds.
The contract that defined them was headed "Reciprocal OOF boundary", the word the project also used for splits that swap whole embryos.

A prefix reversal is also what a model that has absorbed within-embryo acquisition state would produce when scored on more of the same embryo.
I cannot prove that this happened, but it is the most economical reading.

Why nobody checked, I can only reconstruct.
Stratification usually makes folds more comparable, so "balanced separately within prefixes" reads like a protocol improvement.
Stratifying within the very grouping that defines the domain shift is the same operation as ignoring it.
The sentence I wrote on 08-05 was not applied to these folds one day later.
The check is two assertions: each fold holds out exactly one embryo, and no fold trains on the embryo it holds out.
The first fails immediately on a prefix-balanced split.

### 4.3 The audit that caught a different leak

On 08-10 the project audited its own evaluation protocol, and the audit was good.
Its principle was right: out-of-fold status belongs to the whole composition, including every threshold, gate and operating point chosen on top of an out-of-fold base model.

It found real leaks.
A deployed gating model had re-split the 199 movies into five movie-level folds, so it trained on other movies of the evaluation embryo.
An operating point for pre-solver activation had been chosen while seeing both embryos' labels.
A lockbox existed only on paper.
The audit froze a two-fold embryo-disjoint contract: 177 development movies, and 22 in a lockbox that is opened once.

The four-fold results got one sentence: they "remain valid records of those experiments, but they are not evidence under this contract."
They were demoted, not retracted, and the fault in their folds was not named.
A doctrine from 08-01 then kept them standing: a failed drop-in test closes an implementation, never a model family.
Paired with a demotion that never names the fault, that defensible rule lets a demoted number stand as a property of the family.

On 08-12 the record states that the historical four-fold checkpoints "mixed embryo prefixes" and cannot initialize a new model.
The same fact was not applied to the numbers they had produced.

### 4.4 The rerun under the new contract

On 08-12 the multi-frame parent-or-null family was retrained and replayed under the new contract, against a 177-movie anchor of $0.7428367161$.
This family reads five frames to choose each cell's parent or to declare that it has none.
Parent top-1 is the share of cells whose chosen parent is correct.
On the four prefix-balanced folds it had returned $+0.0133995$, positive in every fold and both embryos.

| arm, embryo-disjoint contract | adjusted-edge delta | component metric |
|---|---:|---|
| multi-frame parent-or-null | $+0.0001347503$ | parent top-1 $+0.0101$ |
| two steps: parent-or-null first, then which parent | $+0.0004766233$ | parent top-1 $-0.156$ |

The number fell to about a hundredth of its size, but the rerun changed more than the folds.
It ran with a division guard that held division topology fixed, so its division delta is exactly zero.
The $+0.0133995$ was unguarded, and it paid for its edges with a division loss of $-0.0046489$.
On the old folds, an exact topology guard had already turned an unguarded $+0.0163060$ on eleven movies into $-0.0004684$, in a partial run stopped for futility.
Smaller differences sit on top: 177 movies instead of 199, a new seed, a different anchor.
So this pair cannot tell me how much of the gain the folds inflated.
The case against the folds rests on their construction.
Both arms failed their gates, and the family was closed on its own terms.

The record never set these two numbers side by side.
The $+0.0144$ and $+0.0313$ compositions have not been re-measured under the embryo-disjoint contract, and they are still in the plan as demoted numbers.
My expectation, not a result, is that they will shrink.

This is why a leaky criterion is worse than none.
Without the four-fold replay, the composition would have been an open question, and an open question gets measured.
With it, the composition carried the word "measured", an evidence map cited it, and a reversal that should have raised suspicion was filed as reassurance.

---

## 5. A Proxy That Moved the Wrong Way

A released, pretrained cell-tracking representation was adapted into a parent ranker.
This one was measured on the right folds.
A routing policy decided, cell by cell, whether to trust the new ranker or the existing consensus.
Fitted on one embryo and evaluated on the other, it raised parent top-1 by $+0.008366$ ($0.874241 \to 0.882607$), positive in both held-out embryos.
The exact graph replay over 177 movies returned $-0.002687$, with $69$ movies better and $107$ worse.

The finished graph is not the parent decision.
It is

$$
\hat G
=
R_{\mathrm{gap}}
\circ
R_{\mathrm{prune}}
\circ
\Pi_{\mathrm{parent}},
$$

and the short-component filter $R_{\mathrm{prune}}$ acts on the connectivity that $\Pi_{\mathrm{parent}}$ produces.
Switching parents let components that had been short enough to prune survive, and gap recovery extended them.
The graph gained $9{,}357$ predicted nodes, of which $184$ matched a ground-truth cell.
On the switched rows that overlap a sparse parent label, the new model was $899$ decisions more accurate than the consensus it replaced.
It was better at its own task, and the graph was worse.

Two arms carrying this representation reached the board on 08-05 and read the same score as the incumbent of the day.
This is the complement of section 1.
The board can falsify a large move and is blind to a move of this size.
Local evidence has to decide those, and it has to be scored on the graph.

```text
Component accuracy and graph score are different objectives
whenever a downstream stage is conditioned on the component's output.
```

---

## 6. What Deployment and Process Cost

### 6.1 A candidate has to finish

Note 3 ended with a deployment question.
Pre-solver activation had returned no score seven times, and I had attributed the blanks to runtime.
In the first hours of 08-01 three candidates went to the board with one proposal model per movie instead of four, at three activation budgets, after running cleanly on the four example movies.
All three exceeded the hidden runtime limit and returned no score.
The budget did not matter: inserting proposed points forced a second full pass of the association model over every movie.

On 08-06 the limit got a model, the number Note 3 had asked for.
A deployment must satisfy $F + m\,k \le 43{,}200$ s: fixed setup $F$, work $m$ on the four example movies, and a factor $k$ that scales them to the hidden set.
The example movies hold $42\%$ more cells than the average training movie, so $k$ is only a bracket, capped by a scored submission.
The deployed pipeline had between $0\%$ and $+17.6\%$ of spare capacity.
A candidate was to be run on the example movies first, and a slot spent only if it fit at the pessimistic end.
On 08-08 the pre-solver family left the deployment plan on that record, its local signal still positive.

```text
A candidate that cannot finish on the hidden set is not a candidate.
```

### 6.2 Four runs, no measurement

On 08-07 and 08-08, four consecutive Kaggle notebook runs of the runtime probe that model called for produced no measurement, for four different reasons.
A package resolved from the wrong attached dataset, a mount path was missing from the deployment image, a state-dict key was wrong, and Python 3.12 rejected a construct that local 3.11 accepts.
Every one was diagnosable from local artifacts.

```text
The local check is the easier environment, so it passes.
```

The first run was the worst: it wrote every required output and recorded its own failure in a field nothing downstream read.
A completed process is not a measurement.

### 6.3 When process started replacing modeling

On 08-10 and 08-11 the objective was rewritten.
A program for the remaining weeks set a Public score of $0.970$ as the outcome target, against $0.921$ on the day, with a $480$-hour GPU ladder.
It kept the rule that fold-pure evidence selects, and reconciled the two in one sentence: "Public is an outcome target, not a fitting set."
I do not think that sentence holds when the target is written as a Public score.
A target in board units turns every board reading into a progress report.

An admission system arrived with the program: eight declarations before every launch, four independent evidence checks, hashed launch receipts.
Its founding principle, that a process exit code is not a scientific decision, is the lesson of section 6.2, and I still think it is right.
This is what it produced:

| date | event |
|---|---|
| 08-13 | a measurement run finished all $204$ planned passes and was filed as a runtime failure for exceeding a frozen $3.5$-hour cap by $0.6166$ s |
| 08-16 | one commit added $1{,}193$ files and $646{,}428$ lines of experiment governance |

```text
The admission system did not fail by being wrong.
Each rule was defensible on its own.
It failed by making the cost of asking a question larger than
the expected value of the answer.
```

The $0.6166$-second incident is the clean case: a complete measurement failed over a quantity it did not depend on.
No rule was broken.
The outcome is the rules.

---

## 7. The Rule We Selected By

The rule in force went one step past what Note 3 built.
Its gates could refuse but, by its own account, not yet choose; in August I let local out-of-fold evidence decide anyway, with the board as a falsifier.

The board did its proper job once, cleanly.
The largest clean local gain went to the board with a written expectation, came in five thousandths under it, and tied its control.
What the board could read was the division stage around it.
The board was also used for something the rule does not allow: most of the 08-04 and 08-05 slots went to adjacent division scalars.
The $0.970$ target did not break the rule, but it sat in tension with it.

The rule also gained a clause.
Three candidates approved on local evidence returned nothing, because they could not finish on the hidden set.
From 08-06 runtime had a model, and a candidate had to fit it before it could spend a slot.

The local side failed in a way I did not expect.
The fortnight's largest local numbers came from folds that held both embryos, and their most suspicious property was read as reassurance.
The gates held; the folds under them did not.
The board caught one large miss; the leaky numbers never reached it.

| | this fortnight |
|---|---|
| rule in force | local out-of-fold evidence decides; the board falsifies large predicted moves; from 08-06, a candidate must fit the hidden runtime |
| where it was measured | the research replay (base near $0.74$), on embryo-disjoint and prefix-mixed folds; from 08-10, an embryo-disjoint contract with a lockbox |
| what the board was used for | 08-01: three runs, no score; 08-03: a transfer test that falsified the field's expected gain; 08-04 and 08-05: adjacent scalars; 08-10: an outcome target |
| what broke the rule | prefix-mixed folds, whose leak-shaped pattern was read as generalization; a hold-in ceiling off by a factor of ten; three approved candidates that could not finish |
| what held | two retractions within a day; the embryo-disjoint contract; a runtime model; a guarded rerun that returned $+0.00013$ |

---

## 8. What the Period Established

### Established

1. Harmonic reverse-time association improved the replay score by $+0.0073273$ over 199 movies with embryo-disjoint checkpoints, positive in both embryos and in the median movie. It was the best of four candidates declared in advance, all positive.
2. On the board the field alone returned $0.913$, five thousandths below its written central expectation of $0.918$ and a tie with its forward control ($0.912$). With a division stage it was also a tie ($0.916 \to 0.917$).
3. Collapsing every fork and giving one division policy exclusive authority over second daughters cut division false positives from $587$ to $67$, retained $99.30\%$ of the edge gain, and reached $+0.0106431$ (board: $0.919$ and $0.920$, against $0.917$ for the additive stage).
4. The learned division stage moved the board by four thousandths with the forward and with the reverse field, where the replay had credited it with $+0.0010142$. That is a direction on two rounded pairs, not a rate.
5. On the replay baseline, $99.25\%$ of the $13{,}559$ missed annotated cells had no predicted node inside the $7\,\mu\mathrm{m}$ gate.
6. The hold-in detection-locked fraction of $0.89\%$ becomes $8.79\%$ on the same four movies out of embryo, and $12.10\%$ over 199 movies. The original claim is retracted.
7. The ablation's headline sentences were artifacts of a division term with $2$ true positives. On edges the figures are $13.0\%$ and $-4.6\%$, and the ranking survived.
8. The four outer folds behind $+0.014446$ and $+0.031335$ were balanced within each embryo prefix: movie-out, not embryo-out.
9. The multi-frame family returned $+0.0133995$ unguarded on those folds, and $+0.0001347503$ under the embryo-disjoint contract with a division guard.
10. A parent ranker $+0.008366$ better on held-out parent top-1 made the 177-movie graph worse by $-0.002687$, adding $9{,}357$ nodes of which $184$ matched ground truth.
11. A node-count regressor at $R^2 = 0.471$ on random movie folds scores $0.125$ and $-0.269$ with an embryo held out.
12. With one proposal model per movie, pre-solver activation still exceeded the hidden runtime limit at all three budgets and returned no score.

### Supported but Unconfirmed

1. The $+0.0144$ and $+0.0313$ compositions are inflated by within-embryo leakage and will shrink on embryo-disjoint folds. The multi-frame family's collapse is consistent with this but confounded by its division guard, and these compositions have not been re-measured.
2. The reverse field did not transfer because a field that repairs a weak graph has little to repair in the deployed one.
3. The deployed pipeline's detection-locked fraction lies nearer $12.10\%$ than $0.89\%$.

### Open Questions

1. What is the detection ceiling of the deployed pipeline, measured out of embryo?
2. Do the $+0.0144$ and $+0.0313$ compositions survive the embryo-disjoint contract, and at what size?
3. Does any association gain survive that contract at a size the board can display? Every association candidate measured under it in this window came in below $+0.0005$.
4. Is there any stable ratio from local gains to the board? The plan quotes $0.14\times$ to $0.9\times$. This note's own portfolio fits no single band: the field transferred at a tie, and the division stage moved the board by four thousandths against about one locally.
5. Where else in the pipeline is a head still fitted across the embryo boundary?

---

## Closing

The clean result of the fortnight is a board result I did not want.
The largest clean local association gain the project had measured left no trace on unseen embryos, and the division stage composed with it did.
That is the board used well: one question with a written expectation, a wrong prediction, and a clear answer.

The expensive result is local.
On 08-05 I wrote that random movie folds are not a valid protocol here.
On 08-06 I recorded the largest local gains of the project on folds balanced within the very grouping that defines the domain shift.
On 08-12 one family, rerun under the embryo-disjoint contract with a division guard, kept about a hundredth of its gain, in a pair too confounded to size the leak.
At the close of the window, the other numbers are still in the plan.

```text
Writing down a methodological rule and applying it are different acts.
```

What the fortnight establishes locally is a cap.
On the replay stack, $99.25\%$ of the missed annotated cells have no predicted node inside the gate, and out of embryo about one ground-truth edge in eight is beyond the reach of any association model.
Association still holds the larger headroom in the replay, $0.7346$ against an oracle of $0.9380$.
The deployed pipeline's own cap is one run away.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- **Part 4: When the Largest Local Gain Hurt the Board**
