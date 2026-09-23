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
- Korean version: [BioHub Cell Tracking 작업 기록 4: 가장 큰 로컬 이득은 리더보드에서 보이지 않았다]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

> **About this series.** These notes follow one Kaggle competition, BioHub Cell Tracking During Development:
> rebuilding cell lineages from 3D time-lapse microscopy of zebrafish embryos. The training data is 199 movies
> from two embryos; the leaderboard shows 29% of a hidden test of unseen embryos, and the final ranking uses the
> rest. The series asks one question — how to choose models by an internal criterion when the leaderboard cannot
> be trusted to choose — and each note adds what one period taught about it.
{: .prompt-info }

Note 3 ended with an out-of-fold machine: every change to the lineage graph was measured on movies the models under test had not trained on, and accepted only through gates written down in advance.
It refused most of what it was shown, and its approvals were measured in units of $10^{-4}$, on a leaderboard that displays $10^{-3}$.
The one step the leaderboard did register came from the division channel: an approved two-seed ensemble ($+0.000949$ locally) and its single-seed control both read $+0.004$ higher.

That left August with a plain question: can local evidence decide, with the board as the check?
I carried the rule in on those terms: local out-of-fold evidence decides which change to keep, and the board, the only window onto embryos no model has seen, checks whether it survives there.

The board's half worked when it was given a designed question.
The largest clean association gain measured locally, $+0.0073$, went to the board with a written expectation and tied its control; shipped in place of the submission then in place (the incumbent), which carried a division stage, it read three thousandths lower, hence the title.
What the board could read was the division stage around it.

The local half had two gaps, both about which movies a number was measured on: the largest local gains came from folds that held both embryos in every fold, and a detection ceiling that had just reset the schedule came from movies the deployed model had trained on.
Each gap surfaced inside the fortnight and became a clause of the criterion this series is building.

The short version is:

```text
Local out-of-fold evidence decided; the board checked whether local gains transferred.
July had shown the board weighs the division stage heavily, so five days probed it there.
Adjacent division operating points read as ties; on 08-10 the board was limited to matched checks.
The largest clean local association gain (+0.0073) tied its control; the board read the division stage (+0.004).
The largest local numbers (+0.0144, +0.0313) came from folds that held both embryos in every fold.
A 0.89% detection ceiling was a hold-in number; out of embryo it was 8.79% on the same movies.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What does "local" mean here, and how is the board read? |
| 1 | Why did twenty-five submissions go to the division stage? |
| 2 | Does the largest local association gain survive on unseen embryos? |
| 3 | Where are the cells the pipeline misses? |
| 4 | What did the two retractions of 08-08 correct? |
| 5 | Which embryos were in each fold behind the largest numbers? |
| 6 | Why did a more accurate parent model make the graph worse? |
| 7 | What did deployment and the new checking process show? |
| 8 | What was decided, why, and what did it change? |
| 9 | Which claims survive, and what remains open? |

---

## 0. What "Local" Means in This Note

Almost every local number here comes from the **research replay**: each of the 199 training movies is predicted by a model that did not train on it, the full graph construction is replayed, and the official metric, as the host patched it in July, scores the result.
The two model folds are the two embryos, so no movie is predicted by a model that has seen its embryo: **embryo-out**, the regime of the hidden test.
**Movie-out** folds hold out movies but train on other movies of the same embryo; section 5 is about the difference.

The replay's base sits near $0.74$, while the submitted pipeline, trained on all 199 movies, scores on the board's $0.9$ scale.
Each such setup is its own universe (Note 3's seven reference graphs, $0.60$ to $0.74$, were still unreconciled), so I compare changes inside a universe, never levels across them.

Four terms recur. The **association field** scores candidate edges between cells in consecutive frames.
A **fork** is a predicted cell with two successors, which the metric reads as a division.
The **division stage** is a learned model that decides where a track splits into two daughters, feeding the division term (weight $0.1$ in the score); its **action budget** caps how many divisions it adds.

On 08-01 the incumbent on the board was the forward association graph with a learned division stage, at $0.916$.
The board rounds to three decimals; a difference of $0.002$ or less is the same score at its resolution: no failure detected, never a confirmation.

---

## 1. Why Twenty-Five Submissions Went to the Division Stage

The first five days used twenty-eight submissions: three in the first hours of 08-01 (KST) that returned no score (section 7.1), and twenty-five in five portfolios.
The remaining eleven days used two.

The reason for spending the board early came from July, when a division rank ensemble gained $+0.000949$ locally and moved the board by $+0.004$.
One rounded pair is not a rate, but the direction was clear: the hidden set weighs the division stage far more than the replay does.
Its operating points (budget, the weighting of its two models, variants) were a question the replay could not price and the board could, coarsely, so four portfolios probed them; the fifth, on 08-03, was a designed transfer experiment (section 2).

| date (UTC) | portfolio | public |
|---|---|---:|
| 08-01, 08-02 | variations around the division stage, ten arms | $0.910$ to $0.916$ |
| 08-03 | association field against division stage, five arms | $0.917$ / $0.913$ / $0.912$ / $0.919$ / $0.920$ |
| 08-04 | division action budget, five arms | $0.920$ / $0.919$ / $0.910$ / $0.919$ / $0.919$ |
| 08-05 | division-rank weights and three structural arms | $0.921$ / $0.918$ / $0.920$ / $0.919$ / $0.919$ |

None of the ten arms of 08-01 and 08-02 rose above the incumbent, and three read five or six thousandths lower: two single-seed arms and a portability diagnostic.
The 08-04 budget portfolio gave four ties and one collapse at $0.910$, where the smallest budget removed too much division recall.
The 08-05 portfolio gave five ties.
Its $0.921$ arm, a re-weighting of the two division models, became the incumbent on a narrow local lead in a sweep written down in advance; on the board it is a tie with its neighbors.

So the portfolios measured the instrument rather than finding a best setting: the board sees whether a division stage is present (four thousandths, section 2), but adjacent operating points read as ties, and a tie cannot choose.
On 08-10 this went into the plan: fold-pure out-of-fold evidence selects, and a board submission is a matched transfer experiment with a written expectation, never a way to choose an adjacent scalar value (C9).

---

## 2. Does the Largest Local Association Gain Survive on Unseen Embryos?

### 2.1 Reading the association field backwards

The association model scores each candidate edge forwards, from a cell to its successor in the next frame, and the same network can score it backwards.
Fusing the two harmonically before the graph is built penalizes support found in only one direction.
Four fusion candidates were declared before an exact 199-movie replay, and all four improved the score, by $+0.0033659$ to $+0.0073273$.
Harmonic fusion at a reverse weight of $0.20$ was the strongest, as on a 16-movie probe, so the number below is the best of four, chosen on labeled movies, against the forward pipeline without a learned division stage.

| quantity | forward anchor | harmonic reverse fusion | delta |
|---|---:|---:|---:|
| patched official score | $0.7401015$ | $0.7474288$ | $+0.0073273$ |
| adjusted edge Jaccard | — | — | $+0.0073683$ |
| node recall | — | — | $+0.0172533$ |
| division Jaccard | — | — | $-0.0004099$ |

It improved 119 movies and worsened 80, and both embryos gained: $+0.0194505$ on the 71-movie embryo and $+0.0057244$ on the 128-movie one.

The field was not promoted alone: it added $24$ false forks ($563 \to 587$) and no true divisions, because in a builder that lets forks form freely a better field also manufactures divisions.
The follow-up collapsed every fork to its best continuation and gave the division-event models from Note 3 sole authority to re-admit a second daughter.
Division false positives fell from $587$ to $67$, $99.30\%$ of the edge gain was retained, and the composition reached $+0.0106431$ over the forward anchor and was promoted.
Of that total the field carried $+0.0073273$, a plain additive division stage $+0.0010142$, and exclusivity about $+0.0023$ more, at a larger action budget.
Cheaper, after-the-fact protections, tried before the exclusive stage, did not hold: vetoes on the finished reverse graph drove division Jaccard to zero ($0$ true positives, $1$ false positive, $151$ false negatives), and only $29$ of $999$ stored division actions were still valid on the new graph, worth $-0.0000068$.
Division actions are graph-relative; they have to be rebuilt on the graph they will act on.

### 2.2 The transfer experiment of 08-03

Locally, the field carried most of the gain; the 08-03 portfolio asked whether it would on unseen embryos.
Some arms changed only the field, others added the division stage to either field, and the first three carried written expectations.
For the reverse field alone the expectation was $0.918$ (interval $0.914$ to $0.921$), and a result below that range was to reject the fixed deployment, not the replay's evidence for the family.

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
The field was expected to add about six thousandths over its forward control and added one, a tie; added to a pipeline that already had the division stage, it moved $0.916$ to $0.917$, another tie.
What the board read was the division stage: $0.912$ to $0.916$ on the forward graph and $0.913$ to $0.917$ on the reverse one, four thousandths both times, where the replay had credited it with about one.
At the same $0.012$ budget, making the stage exclusive moved $0.917$ to $0.920$, at the edge of what the board resolves; at twice the budget it read $0.919$, a tie.

![Public scores of four portfolio arms: adding the division stage moved both fields by 0.004, the reverse field moved them by 0.001]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-01-board-read-division.png)
_Figure 1. Public scores of four arms of the 08-03 portfolio. Adding the division stage moved both fields by $0.004$; switching to the reverse field moved either pipeline by $0.001$, a tie at the board's resolution. Locally the field had carried most of the gain._

### 2.3 What the experiment settled

The arm behind the title shipped the reverse field without the incumbent's division stage and read $0.913$ against $0.916$; the matched arms put the field's own share at a tie and the gap on the missing division stage.
So the experiment answered its question: a large local move, predicted in writing, did not appear on unseen embryos, and the arms located what the board reads.
The replay's folds were the two embryos, so the miss is not a fold leak.
Why the field did not carry is an inference, untested in this window: the replay's graph, from fold models trained on one embryo each, has node recall near $0.90$, and a field that repairs a weak graph plausibly has less to repair in the deployed one, built from both.

### 2.4 A retention gate that measured a component

On 08-04 a sibling composition, the joint association model with the exclusive division stage on top, reached $+0.010205$ over 199 movies.
A gate written before the run rejected it: the composed graph had to keep $95\%$ of the association stage's adjusted-edge gain, and it kept $91.82\%$.
The gate existed to protect the association gain, because collapsing forks deletes edges, but it guarded a share of one term rather than the score, and so it vetoed a positive total.
That is C4, the Note 3 clause about components, turned on a gate of my own.

---

## 3. Where Are the Cells the Pipeline Misses?

From 08-04 the rented GPU was paused for a week, and the week's most useful CPU product was an error anatomy on 08-05 over the 199-movie replay baseline.
Node recall was $0.8983$: $119{,}759$ of $133{,}318$ annotated cells matched a predicted node inside the $7\,\mu\mathrm{m}$ gate, the radius within which the scorer pairs a prediction with an annotation.
Of the $13{,}559$ misses:

| class | count | share |
|---|---:|---:|
| no predicted node anywhere inside the gate | $13{,}457$ | $99.25\%$ |
| a node inside the gate, lost to assignment competition | $102$ | $0.75\%$ |

Two explanations I held did not survive.
The first was that the gate is slightly too tight: a found cell lands at a median of $2.24\,\mu\mathrm{m}$ from its node, but a missed cell's nearest predicted node sits at a median of $9.60\,\mu\mathrm{m}$, against a median cell spacing of $24.99\,\mu\mathrm{m}$.
That node is a different cell.

![Median distances on one axis: found cell to node 2.24 µm, gate 7 µm, missed cell to nearest node 9.60 µm, cell spacing 24.99 µm]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-02-missed-cell-distances.png)
_Figure 2. Medians from the 08-05 error anatomy of the replay baseline. A missed cell's nearest predicted node sits beyond the $7\,\mu\mathrm{m}$ gate, at a median of $9.60\,\mu\mathrm{m}$: it belongs to another cell._

The second was that misses concentrate in crowded tissue: the densest spacing bin has a miss rate of $23.0\%$ but holds only $339$ misses, $2.5\%$ of the total.
The missed cells are not in the point set, and no association model can link a point that does not exist: Note 3's question of which points exist, now with a number.

The same audit found that a label-free node-count regressor with $R^2 = 0.471$ under random five-fold cross-validation over movies reached $R^2$ of $0.125$ and $-0.269$ with a whole embryo held out, named the gap within-embryo leakage, and concluded:

```text
Random folds over movies are not a valid protocol in this competition.
```

---

## 4. What the Two Retractions of 08-08 Corrected

### 4.1 "Detection is not the bottleneck"

How much of the score is locked behind detection in the pipeline that ships?
On 08-07 I measured it with an association oracle on the graph the project actually submits: it keeps the predicted nodes, replaces the edges with ground-truth topology projected through the official matcher, and rescores, so an edge it still cannot place has an undetected endpoint.
On the four labeled example movies distributed with the competition, the deployed graph had node recall $0.983$, and only $0.89\%$ of ground-truth edges were unreachable.
If fewer than one edge in a hundred is out of reach, detection is not binding and a detector retrain is not worth its compute; the schedule was changed on that reasoning.

On 08-08 the regime of those movies was checked: they are copies of training movies the deployed models were trained on, so the $0.89\%$ is a **hold-in** number, measured on data the model has already fit.
The same computation on the same detection family out of embryo:

| regime | movies | GT edges unreachable |
|---|---:|---:|
| deployed graph, hold-in | 4 | $0.89\%$ |
| same detection family, embryo-out | the same 4 | $8.79\%$ |
| same detection family, embryo-out | 199 | $12.10\%$ |

That is ten times more on the identical four movies.
Over 199 movies out of embryo, perfect association lifts the replay's $0.7346$ to $0.9380$: association still holds about $+0.20$ of headroom, and about one ground-truth edge in eight is beyond the reach of any linking model.
The claim was retracted and the schedule decision reversed; the corrected reading is that "association is the largest lever, and detection is probably binding near the top of the board."

The retraction has its own limit: the two extreme rows differ on two axes at once, hold-in against embryo-out and deployed pipeline against research replay, and the run that would separate them was named on 08-08 and not run inside this window.
The general finding is clause C8: the deployed notebook was the right object and the example movies the wrong sample, since a number from movies the model trained on measures fit, not generalization.

### 4.2 The ablation headlines

The same review retracted two headline sentences, written a day earlier, from a feature-family ablation on 34 movies of the section 5 composition: removing the temporal family "keeps $1.2\%$ of the gain", and the auxiliary center detector alone is "worse than nothing, at $-17.2\%$".
Both were division artifacts: the 34-movie control held two division true positives against 89 false positives, so with $TP = 2$ one event moved the pooled division Jaccard, $TP/(TP+FP+FN)$ at weight $0.1$, by half its value and swamped the edge differences.
On the edge axis alone the figures are $13.0\%$ and $-4.6\%$; the ranking survived, and the two sentences were deleted.

---

## 5. Which Embryos Were in Each Fold?

### 5.1 The largest local numbers of the fortnight

On 08-06 a multi-family composition completed a 199-movie exact replay, mixing a joint lineage-action model, the auxiliary center detector and appearance features into the two-seed association anchor, under one authority weight $a$ that sets how much say the auxiliary models get.
At $a = 0.20$ the association stage returned $+0.014446$ over a baseline of $0.7405959$.
With the exclusive division stage on top, the sweep reached $+0.031335$ at $a = 0.50$, the upper bound enforced in code, and $+0.034692$ at $a = 1.00$ under an opt-in flag.
Every composition passed every gate, all four outer folds and both embryos positive: at $a = 0.20$ the 71-movie embryo gained $+0.018084$ and the 128-movie embryo $+0.013709$.

The 71-movie embryo, the weaker one in earlier families, was now the stronger. On 08-06 I wrote down, as a reason for confidence: "Since train and test are embryo-disjoint, a family that no longer depends on one embryo is better evidence for generalisation, not worse."
Four days later my plan cited it as a measured gain and set an association target of that size.

The design looked sound for ordinary reasons: the four outer folds were balanced separately within each embryo prefix, the tag in each movie identifier that names its embryo, and stratification usually makes folds more comparable.
The document that defined them used the word "reciprocal", which the project also used for splits that swap whole embryos, and the written reasoning is right about the hidden test.

### 5.2 What the folds were

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

Every fold held movies from both embryos: movie-out inside seen embryos, which cannot estimate performance on an unseen embryo.

![Schematic of four prefix-balanced folds, each holding both embryos, against two embryo-out folds]({{ site.baseurl }}/assets/img/posts/2026-08-16-biohub-working-note-4/fig-03-fold-construction.png)
_Figure 3. The four-fold replay behind the fortnight's largest numbers balanced every fold across both embryos. The contract frozen on 08-10 holds out one whole embryo per fold._

Stratifying within the grouping that defines the domain shift is the same operation as ignoring it: every fold trains on the embryo it evaluates.
Section 3's finding was about random folds; these were stratified, so they read as a better protocol, but they share the one property that matters, and the decision took a premise true of the hidden test to be true of the folds.
The embryo reversal fits the same reading: it is what a model that has absorbed within-embryo acquisition state would produce; I cannot prove that, but it is the most economical explanation.
The check is two assertions, that each fold holds out exactly one embryo and that no fold trains on the embryo it holds out; the first fails at once on a prefix-balanced split.

### 5.3 The audit of 08-10, and the leak it found

On 08-10 I audited the evaluation protocol on a sound principle: out-of-fold status belongs to the whole composition, every threshold, gate and operating point included.
It found real leaks: a deployed gating model had re-split the 199 movies into five movie-level folds, so it trained on other movies of the evaluation embryo; an operating point for pre-solver activation had been chosen while seeing both embryos' labels; and a lockbox existed only on paper.
The audit froze a two-fold embryo-disjoint contract: 177 development movies, and 22 in a lockbox opened once.

I demoted the four-fold results rather than retracting them; the audit had found other leaks and froze a new contract, so the demotion sentence named the contract, not the fold construction: they "remain valid records of those experiments, but they are not evidence under this contract."
A doctrine from 08-01, that a failed drop-in test closes an implementation and never a model family, is defensible on its own, but paired with a demotion that names no fault it lets a demoted number stand as a property of the family.
On 08-12 I recorded that the four-fold checkpoints "mixed embryo prefixes" and could not initialize a new model; within this window I had not yet applied that fact to the numbers they produced.

### 5.4 A rerun under the embryo-disjoint contract

On 08-12 the multi-frame parent-or-null family, which picks each cell's parent, or none, from five frames, was retrained and replayed under the new contract against a 177-movie anchor of $0.7428367161$.
On the four prefix-balanced folds it had returned $+0.0133995$, positive in every fold and both embryos.
Parent top-1 below is the share of cells given the correct parent.

| arm, embryo-disjoint contract | adjusted-edge delta | component metric |
|---|---:|---|
| multi-frame parent-or-null | $+0.0001347503$ | parent top-1 $+0.0101$ |
| two steps: parent-or-null first, then which parent | $+0.0004766233$ | parent top-1 $-0.156$ |

The number fell to about a hundredth, but the rerun changed more than the folds: it ran with a division guard, so its division delta is exactly zero, while the $+0.0133995$ was unguarded and paid for its edges with a division loss of $-0.0046489$.
On the old folds an exact topology guard had already turned an unguarded $+0.0163060$ on eleven movies into $-0.0004684$, in a partial run stopped for futility.
With 177 movies instead of 199, a new seed and a different anchor on top, this pair cannot size the fold inflation; the case against the folds rests on their construction.
Both arms failed their gates, and the family closed on its own terms.

The $+0.0144$ and $+0.0313$ compositions have not been re-measured under the contract and remain in the plan as demoted numbers; my expectation, not a result, is that they will shrink.

### 5.5 What a number from leaky folds does

A number from leaky folds does more than overstate a gain: it closes a question.
Without the four-fold replay the composition would have stayed open and been measured; with it, the composition carried the word "measured", a target was set from it, and the embryo reversal was read through the construction the folds were believed to have.
A leak and a generalizing family can produce the same results, but whether a fold holds out an embryo is a property of the split, checkable before any number exists: clause C7.

---

## 6. Why a More Accurate Parent Model Made the Graph Worse

A released, pretrained cell-tracking representation was adapted into a parent ranker, to ask whether it chooses parents better and whether that improves the graph.
It was measured embryo-out: a routing policy, fitted on one embryo and evaluated on the other, decided cell by cell whether to trust the new ranker or the existing consensus.
Parent top-1 rose by $+0.008366$ ($0.874241 \to 0.882607$), positive in both held-out embryos; the exact graph replay over 177 movies returned $-0.002687$, $69$ movies better and $107$ worse.

The finished graph is not the parent decision but $\hat G = R_{\mathrm{gap}} \circ R_{\mathrm{prune}} \circ \Pi_{\mathrm{parent}}$, and the short-component filter $R_{\mathrm{prune}}$ acts on the connectivity that $\Pi_{\mathrm{parent}}$ produces.
Switching parents let components short enough to prune survive, and gap recovery extended them: the graph gained $9{,}357$ predicted nodes, of which $184$ matched a ground-truth cell.
On switched rows that overlap a sparse parent label, the new model was $899$ decisions more accurate than the consensus it replaced: better at its own task, worse for the graph.

Two arms carrying this representation reached the board on 08-05 and tied the incumbent of the day: the board is blind at this size, so local evidence has to decide, scored on the graph.
The family closed on 08-10, and C4 was reconfirmed with the path traced: component accuracy and graph score diverge whenever a downstream stage is conditioned on the component's output.

---

## 7. Getting Candidates to Run, and Checking the Checks

### 7.1 A candidate has to finish

Note 3 ended with seven blank pre-solver submissions, attributed to runtime, and a plan to cut proposal inference to one model per movie.
In the first hours of 08-01 three candidates tested that plan on the hidden set, at three activation budgets, after running cleanly on the four example movies.
All three exceeded the hidden runtime limit and returned no score, whatever the budget: inserting proposed points forced a second full pass of the association model over every movie.

On 08-06 the limit got the model Note 3 had asked for: a deployment must satisfy $F + m\,k \le 43{,}200$ s, with fixed setup $F$, work $m$ on the four example movies, and a factor $k$ that scales them to the hidden set.
The example movies hold $42\%$ more cells than the average training movie, so $k$ is only a bracket, and the deployed pipeline had between $0\%$ and $+17.6\%$ of spare capacity.
A candidate would spend a slot only if it fit at the pessimistic end.
On 08-08 the pre-solver family left the deployment plan on that record, its local signal still positive: C6 applied.

### 7.2 Four notebook runs, four environments

On 08-07 and 08-08 four consecutive Kaggle notebook runs of the runtime probe produced no measurement, for four reasons, each diagnosable from local artifacts: a package resolved from the wrong attached dataset, a mount path missing from the deployment image, a wrong state-dict key, and a construct that Python 3.12 rejects and local 3.11 accepts.

```text
The local check is the easier environment, so it passes.
```

The first run wrote every required output and recorded its failure in a field nothing downstream read.
A completed process is not a measurement.

### 7.3 When a check costs more than the question

On 08-10 and 08-11 I wrote a program for the remaining weeks with a Public score of $0.970$ as its outcome target ($0.921$ on the day) and a $480$-hour GPU ladder.
Its line "Public is an outcome target, not a fitting set" is coherent, but a target in board units still turns every board reading into a progress report, in tension with C9 in the same plan.
The program also brought an admission system: eight declarations before every launch, four independent evidence checks, hashed launch receipts.
Its founding principle, that an exit code is not a scientific decision, is the finding of section 7.2, and it holds.

| date | what followed |
|---|---|
| 08-13 | a measurement run finished all $204$ planned passes and was filed as a runtime failure for exceeding a frozen $3.5$-hour cap by $0.6166$ s |
| 08-15 | a capture chain failed five times in a row, the last because a generic all-finite validator rejected a NaN that the producer defines as "no match" ($1{,}379.575$ s billed) |
| 08-16 | one commit added $1{,}193$ files and $646{,}428$ lines of experiment governance |

Each rule was defensible on its own; together they rejected a complete measurement over a quantity it did not depend on and a value its producer had defined, and made asking a question cost more than the answer was worth.
How to keep a check proportionate to its question is what this window left open.

---

## 8. Decision Log

The rule in force was the one Note 3 ended on: local out-of-fold evidence decides, and the board checks whether a locally chosen change survives on unseen embryos.
For five days the board was used heavily, because July had shown that the hidden set weighs the division stage far more than the replay does, and its operating points were a question the replay could not price.
The 08-03 portfolio was a designed transfer experiment; the adjacent-setting portfolios returned ties, and on 08-10 the board was narrowed to matched checks.

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
| C5 | Compare levels only inside one reference universe; compare deltas across | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 **(new)** | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 **(new)** | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 **(new)** | Use the board for matched transfer checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |

---

## 9. What the Period Established

### Established

1. Harmonic reverse-time association improved the replay by $+0.0073273$ over 199 embryo-out movies, both embryos positive; best of four predeclared candidates, all positive.
2. On the board the field alone read $0.913$, five thousandths below its written expectation of $0.918$ and a tie with its forward control ($0.912$); with a division stage, also a tie ($0.916 \to 0.917$).
3. Collapsing every fork and giving the division model exclusive authority over second daughters cut division false positives from $587$ to $67$, retained $99.30\%$ of the edge gain, and reached $+0.0106431$ (board: $0.920$ at the same $0.012$ budget against $0.917$, at the edge of its resolution; $0.919$ at twice the budget, a tie).
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
2. The reverse field did not transfer because a field that repairs a weak graph has little to repair in the deployed one.
3. The deployed pipeline's detection-locked fraction lies nearer $12.10\%$ than $0.89\%$.

### Open Questions

1. What is the detection ceiling of the deployed pipeline, measured out of embryo?
2. Do the $+0.0144$ and $+0.0313$ compositions survive the embryo-disjoint contract, and at what size?
3. Does any association gain survive that contract at a size the board can display? Every association candidate measured under it in this window came in below $+0.0005$.
4. Is there any stable ratio from local gains to the board? My plan for the remaining weeks quotes $0.14\times$ to $0.9\times$; this note's portfolio fits no single band.
5. Where else in the pipeline is a head still fitted across the embryo boundary?

---

## Closing

August began with a division of labor: local evidence decides, and the board checks transfer.
Given a designed question, the board answered it: the 08-03 experiment separated the field from the division stage cleanly.
Asked to choose among neighboring settings, it returned ties, and on 08-10 that use left the rule.

The local half needed two repairs, both about which movies a number came from.
Stratified folds and the deployed notebook each looked like the stronger measurement, and each overstated what an unseen embryo would see; C7 and C8 check the split before any result exists.

What remains locally is a cap and a small number: on the replay, $99.25\%$ of the missed cells have no predicted node inside the gate, and every association candidate measured under the embryo-disjoint contract came in below $+0.0005$.
With honest folds, the gains left inside the current objective are small.
The question for the next period is whether anything inside that objective can still move the score.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- **Part 4: When the Largest Local Gain Hurt the Board**
