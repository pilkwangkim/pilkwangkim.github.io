---
title: "BioHub Cell Tracking Working Note 6: The Universe We Were Selecting In"
date: 2026-09-06 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, universe-mismatch, transfer-ratio, detector-augmentation, leakage, oof, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-06-biohub-working-note-6/cover.png
  alt: "Title card for BioHub Working Note 6: the universe we were selecting in"
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

# BioHub Cell Tracking Working Note 6: The Universe We Were Selecting In

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
  - [Working Note 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- Korean version: [BioHub Cell Tracking 작업 기록 6: 우리가 선택해 온 우주]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 7: Where a Local Gain Has to Be Measured]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 5 ended on 2026-08-28 with two changes made on the same day.
One was a model change: a fitted division verifier replaced a set of hand-tuned geometric gates, on a local out-of-fold gain of $+0.0053$.
The other was a rule change.
After twenty-three days in which the public score did not move, the last twelve spent maximizing a frozen local comparator for which a ground-truth filter over every selected action was worth $+0.000365$, I rewrote the project's rules so that the Public board was the objective again and local replay was only a selection tool.
I wrote that rule in reaction to a stall, not as a position I had argued through.

This note covers 2026-08-29 to 2026-09-04, a week with three submissions.
Every model decision in the week was made locally.
The verifier decisions used twofold, embryo-disjoint out-of-fold (OOF) replay over the 199 training movies.
The detector programs were judged on trainer validation and on embryo-out pipeline runs over 2 and 15 movies.
The board was asked two narrow questions, alongside one operating-point arm that tied: one measurement no local instrument could make, and one prediction whose reading was written down before the result existed.

The week's main finding is about the out-of-fold instrument.
Its folds were clean, and it was measuring the wrong machine: the comparator replay, which the deployed notebook does not run.
An internal criterion has to be measured in the pipeline that ships, or it selects for a distribution that does not exist.

The short version is:

```text
A local criterion can be leak-free and still select in the wrong universe.
The graphs we tuned on and the graphs the notebook builds were 0.149 apart
on the same 199 movies with the same scorer.
Refitting the division verifier inside the deployed pipeline gained locally,
and the board, read against a band written in advance, did not overturn it.
After that repair, three levers died for reasons other than their own
measurement, and a correct detector diagnosis lost in composition.
None of those verdicts needed a submission.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What state did the week open in? |
| 1 | How is a hidden quantity measured when no local instrument sees it? |
| 2--3 | Why did more division data, longer training and external data not help? |
| 4 | What was wrong with every local number, and what did fixing it buy? |
| 5 | Which levers survived the repair, and what killed each of them? |
| 6 | Why did a correct diagnosis of the detector tail still lose? |
| 7--8 | Which rule selected this period, and what did it establish? |

---

## 0. Where the Week Opened

The pipeline is the one from the earlier notes, run inside the submitted Kaggle notebook (the kernel): two seeded detectors whose detection fields are blended before peak extraction, a transformer edge scorer, an integer linear program (ILP) that selects the graph, and deterministic post-stages.
The division stage had just changed: a gradient-boosted verifier, retrained inside the notebook from a shipped candidate table, now decided which forks to add.
The competition also ships four labeled example movies.
They are copies of training movies, so scoring the deployed all-train models on them is a hold-in check: it can catch damage, but it cannot select.
The 199 training movies come from two embryos, of 71 and 128 movies, so embryo-disjoint (embryo-out) replay has two folds, each scoring one embryo with models trained on the other.
The standing rule asks that a change be non-negative on both.

| item, entering 2026-08-29 | value |
|---|---:|
| the comparator replay (Note 5's out-of-fold comparator), 199 movies, old division stage in place | $0.6014$ |
| local gain of the shipped verifier on that comparator | $+0.0053$ |
| Public score of the first verifier deployment (v79) | $0.935$ |
| Public score of the deployment before it | $0.921$ |

Two things in that table were strange, and I had no instrument for either.
The first was the ratio.
Five thousandths of local gain had produced fourteen thousandths of Public gain, about $2.6\times$, where the project had been citing a damped range of $0.14\times$ to $0.9\times$ for edge-channel changes, a range from earlier replays with no per-experiment table behind it.
The second was older.
The local base sat near $0.60$ while the Public score sat near $0.94$.
That gap had been attributed for weeks to a population difference, and the attribution was never tested.
Behind both sat an assumption I had not examined: that the local out-of-fold graphs are a faithful, if pessimistic, stand-in for what the notebook produces.

---

## 1. Buying a Measurement With Two Submissions

### 1.1 The Decomposition Probe

On 2026-09-01 I spent two submissions on one question.
The first arm, v80, was the deployed payload at a swept operating point for the division stage.
The second arm, v81, was the first deployment's payload with the verifier's *apply* step replaced by a function that returns its inputs unchanged.
The model was still fitted, so every downstream stage saw identical inputs; only the edits were withheld.
With *apply* neutralized, v79's and v80's constants govern nothing, so one probe arm serves both.

The difference between the two scores is the stage's hidden contribution, $\Delta J_{\mathrm{edge}}^{\mathrm{adjusted}} + 0.1\,\Delta J_{\mathrm{division}}$, measured with no local proxy in the path.

| arm (Public split) | what it is | Public |
|---|---|---:|
| v79 | first verifier deployment | $0.935$ |
| v80 | swept operating point, verifier active | $0.937$ |
| v81 | v79's payload, apply neutralized | $0.903$ |

The subtraction gives the division term a hidden contribution of $+0.032$ against v79 and $+0.034$ against v80.
Divided by the division weight of $0.1$, the hidden division Jaccard is near $0.32$ to $0.34$.
The same stage's local out-of-fold division Jaccard, in the universe it was fitted in, was $0.062$.

v81's graph had no forks at all.
The verifier made zero edits on every movie, and the forks an earlier post-link stage adds are collapsed before the verifier runs.
So the subtraction is the whole hidden division term at v79.
It is not the verifier's gain over the hand-gated stage it replaced; that step was the $+0.014$ in Section 0.

Two qualifications remain.
The subtraction folds in the stage's small edge footprint, one false positive on the four example movies.
And the board shows three decimals, so $+0.032$ is really $+0.032 \pm 0.001$, a hidden division Jaccard between about $0.31$ and $0.33$.
With those in place, the hidden division level is roughly five times the local one: a subtraction of two rounded numbers, bought with two slots.

The swept arm taught nothing by itself.
Its local gain was $+0.0007$, non-negative on both embryos but below the $+0.001$ the project asked of a standalone change, so it rode along with the probe.
$0.937$ against $0.935$ is the same score at the board's resolution.

### 1.2 Why This Was a Proper Use of the Board

On the four example movies, the deployed stage made $45$ edits and changed the edge score by exactly zero, because all $45$ landed off the sparse annotation.
Locally, the stage was invisible.
A stage that acts almost entirely in unannotated space cannot be priced any other way.
The probe chose nothing. It measured a quantity, and the measurement pointed at the local instrument rather than at a model.

---

## 2. Why More Division Data Was Not the Answer

Every local number in this section comes from the comparator replay that Section 4 retires.
The signs survive that retirement. The levels do not.

An oracle over those graphs said $+0.0557$ was reachable, and the fitted verifier captured about a tenth of it.
Its training set held roughly $80$ positives, because the competition's entire division supervision is $151$ annotated events.
The obvious conclusion was starvation, and three experiments falsified it.

**Fixing generation recall lowered the applied score.**
The candidate generator reached only $75$ of the $151$ events.
Widening it took labeled positives from $80$ to $110$, nearly doubled the candidate pool, and the applied delta over the deployed stage fell from $+0.0053$ to $+0.0021$.
The ranker's operating point had been calibrated on a pool of one composition; doubling the pool at a fixed threshold admits a worse marginal candidate.
A recall fix upstream is a distribution shift downstream.

**An appearance CNN memorized.**
A three-patch convolutional encoder was trained embryo-pure, and its training loss fell from $1.42$ to $0.075$.
On the held-out embryo, the median true division scored below the 99th percentile of negatives on both folds: an active anti-ranking.
With $61$ and $19$ positives per training fold, it could only memorize.

**A ranker that solved its own domain added exactly zero.**
The same generator over a public synthetic dataset produced $163{,}422$ positives, and a ranker trained on them reached holdout AUC $0.9988$.
Added as a feature to the real ranker, it left the score the same to four decimals.
In the real ranker's feature space, real divisions are not separable from real false forks; synthetic divisions are geometrically clean and real ones are not.

A fourth experiment closed the loop: a CNN trained on $23{,}977$ real divisions from a public zebrafish dataset *did* generalize across embryos, and still added less than $10^{-4}$ to the official score.
Both "too few positives" and "wrong domain" were real defects, and fixing either moved nothing, because neither was binding.
Section 4 says what was.

---

## 3. Two Detector Programs That Lost in the Pipeline

### 3.1 Epoch Extension, and a Byte-Identical Result as a Bug Signal

A continuation program tested whether the detector had more to give from epochs alone, taking an unchanged recipe from epoch $200$ to $400$.
Only the first split ran.
Its embryo-out trainer validation read $0.8151$ at epoch $200$, peaked briefly at $0.8326$ at epoch $218$, and fell to $0.7907$ at epoch $400$.

The first evaluation of the epoch-$400$ checkpoint returned edge counts byte-identical to the epoch-$200$ run.
On resume, the best-checkpoint logic had restored the pre-resume running best, the tracked metric never re-crossed it, and the file was never rewritten.
A byte-identical result is not a null result: two independent stochastic paths do not produce identical integers, so identity points to a shared artifact upstream.

### 3.2 External Data: A Large Trainer Gain That Was a Pipeline Loss

The host explicitly permitted a public zebrafish embryo dataset with dense annotations.
The detector was fine-tuned on it, and because of Section 3.1 each epoch was paired against a control resumed from the identical checkpoint.
Against a gate written down in advance at $+0.005$, the formal band came back at $+0.0466$, the largest validation gain the project had produced.

The same checkpoint then went through the deployed pipeline as the primary detector, scored with the official scorer on the two embryo-out example movies.

| arm (2-movie embryo-out pipeline harness) | score | node recall | predicted nodes, dense movie |
|---|---:|---:|---:|
| control pair | $0.8028$ | $0.9662$ | $64{,}477$ |
| fine-tuned as primary | $0.7756$ | $0.9625$ | $66{,}122$ |

The trainer's metric moved $+0.0466$ and the end-to-end metric moved $-0.0272$ on the same held-out embryo.
The fine-tuned model proposes more nodes and recalls fewer annotated cells: it found more objects and placed them slightly worse.
Node matching is an optimal assignment on centroid distance within $7\,\mu\mathrm{m}$, and the ILP's linking costs live in the same geometry.
Both depend on where each peak sits, and that is what the fine-tune made worse.

Removing the second detector's field, moving the threshold and reversing the two detectors' roles all failed.
One arm nearly shipped.
With the field blend off and only the fine-tuned model's novel peaks added, it scored $0.7890$; read against an assumed single-detector base near $0.75$ to $0.76$, the additions looked worth about $+0.03$.
But with the blend off the base is a single detector, and that baseline had never been run.
Run, it scored $0.7972$. The additions were worth $-0.0082$.
An arm without its own baseline measures the wrong difference.
Every pipeline number here rests on two movies, and I closed the fine-tune as a deployment detector.
It is Section 4's problem one level down: the trainer's validation metric is not the pipeline that ships.

---

## 4. The Universe We Were Selecting In

### 4.1 Naming the Failure Mode

Every learned component here is a selector fitted on a table of candidates, and every table is produced by running some pipeline over the training movies.
Write $\mathcal{U}$ for that generating pipeline and $\mathcal{C}(\mathcal{U})$ for the candidate distribution it induces. The failure mode is

$$
\phi^{*}
=
\arg\max_{\phi}\;
S\!\left(\mathcal{C}(\mathcal{U}_{\mathrm{fit}});\,\phi\right),
\qquad
\text{shipped as } \phi^{*} \text{ acting on } \mathcal{C}(\mathcal{U}_{\mathrm{run}}),
\qquad
\mathcal{U}_{\mathrm{fit}} \ne \mathcal{U}_{\mathrm{run}} .
$$

This is not leakage.
Both universes can be embryo-disjoint and every fold can be clean.
It is a shift between the local measuring apparatus and the deployed one, and it is invisible from inside either, because each is internally consistent.

### 4.2 The Measurement

The run that settles it had been named as decisive almost four weeks earlier and never scheduled.
It replays the deployed stack itself over all 199 training movies with embryo-disjoint checkpoints, scored by the official scorer.

| universe (199 movies, embryo-out, official scorer) | score | node recall, final graph |
|---|---:|---:|
| comparator replay, forks stripped, where the recent local selections were made | $0.6005$ | $0.8870$ |
| deployed stack, the notebook's own output, same movies, same scorer | $0.7499$ | $0.9255$ |

![Two scores on the same 199 movies: comparator replay 0.6005 and deployed stack 0.7499, 0.149 apart]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-01-two-instruments.png)
_Figure 1. Two instruments reading the same 199 training movies with the same scorer. The August selections were made on the comparator replay, the lower-scoring instrument; the notebook runs the deployed stack._

The $0.6014$ of Section 0 is the same comparator replay with the old division stage's forks left in.
Note 4's research replay (base near $0.74$) is a different graph family; the $+0.149$ is measured against the $0.60$ comparator.
These are not a before and after.
They are two instruments reading the same movies, and they disagree by $+0.149$.
Every verifier threshold, deletion rule and association arm of the preceding weeks had been chosen on graphs with node recall $0.887$, while the notebook ran a stack at $0.9255$.
Part of the old gap between a local base near $0.60$ and a Public score near $0.94$ was therefore never a population difference.
The rest may still be one. This week did not test it.

The same replay rebuilt the division stage inside the deployed universe.
With the notebook's raw forks collapsed, the base is $0.7489$, and every division number below is measured over it.
A label oracle over the candidates the deployed pipeline actually produces scores $0.8007$, at division Jaccard $0.5097$, with $79$ true positives, $4$ false and $72$ missed.
Those $72$ events have no candidate row at all, so roughly half of the division ceiling is unreachable by the candidate generator before any ranking question is asked.
Why those $72$ have no row was not measured this week.

### 4.3 What the Replay Explained, and What It Did Not

The day before, the explanation on offer for the five-fold hidden division level was that the hidden set is denser in scorable divisions.
The replay did not replace it.
On deployed-universe graphs the deployed verifier drew $2$ true and $3$ false divisions, a local division Jaccard near $0.013$, lower than the $0.062$ measured on the comparator replay.
Correcting the machine widened the gap to the hidden $0.32$.

What the replay did expose was a second problem: the verifier had been fitted on candidates from one universe and was being applied to candidates from another.
The test holds the graphs fixed and changes only the candidates the verifier is fitted on.

| fit (deployed-universe 199-movie OOF, embryo-disjoint, base $0.7489$) | delta over base | division TP / FP |
|---|---:|---:|
| deployed verifier, fitted on comparator-replay candidates | $+0.0013$ | $2 / 3$ |
| refit on deployed-universe candidates only | $+0.0030$ | $5 / 13$ |
| refit on the union of both tables | $+0.0065$ | $17 / 90$ |

The deployed selector captured a fifth of what was available, and on the larger embryo it fired nothing at all across $128$ movies.
A model can be well fitted, embryo-disjoint and correctly thresholded, and still be aimed at a distribution that no longer exists.

### 4.4 The Repair, and What It Returned

I selected the union fit.
Its sweep was a plateau rather than a spike, and both embryos were positive at the peak ($+0.0096$ and $+0.0060$).
Translating it to the runtime replaced per-embryo thresholds with one flat threshold chosen by a rule fixed in advance.
The kernel passed its hold-in gate on the four example movies with the edge channel inert, as a division-only change must: edge true/false/missed counts of $2028/155/99$ against the previous $2027/155/100$.

Before submitting v83, I wrote down how its Public score would be read:

```text
>= 0.940       the universe refit transfers; make it the base
0.937 - 0.939  neutral; the refit is not distinguishable
< 0.937        the flat threshold hurt; revert to per-embryo
```

It returned $0.944$, a move of $+0.007$.
That is large enough for the board to resolve.
The arm had been chosen locally; the reading rule gave the board one job, to veto making it the base, and the result landed in the band that kept it.

Local $+0.0065$ against Public $+0.007$ is a transfer of about $1.1\times$, or nearer $1.5\times$ with the corrected $+0.0046$ of Section 4.5.
The earlier $2.6\times$ had its local delta measured on the comparator replay, so the two ratios come from different rulers, and each is one point with a rounded numerator.
I do not treat either as a rate.
The working expectation I took from them, and used to order the queue, was a sign: division changes had not been damped on the hidden set.

### 4.5 The Shipped Code as the Selection Tool

On 2026-09-03 a parity check compared the shipped runtime module inside the kernel with the same recipe run locally.
On the four example movies, the kernel applied $50$, $23$, $3$ and $50$ divisions; the local recipe applied none.

The cause was a flag.
Every local rebuild of the division tables had generated candidates only around annotated sources, which looks sensible because only annotated regions produce labels.
On one of the four example movies that yields $7{,}519$ rows; the shipped runtime, which does not know where the annotations are, generates $358{,}000$ on the same movie.
Every local sweep of the stage's threshold and budget had been priced without the false-positive load the kernel carries, so every local optimum was biased loose.

This was Section 4.1 again at a smaller scale.
The shipped runtime module itself became the selection tool, and rebuilt tables became training material only.
Re-run that way, the deployed operating point was the optimum anyway: $0.7535$ against the fork-free $0.7489$, $+0.0046$ with both embryos positive, and no threshold or cap change cleared the rule written down for a new operating point.

The same sweep exposed a structural fact of the metric.
A predicted division at a parent with no ground-truth node within the matching distance is neither a true nor a false positive.
The all-train, in-sample run applied $4{,}662$ divisions and drew only $71$ false positives, against $42$ true ones.
In the embryo-out arms, per-movie caps spanning a factor of four scored $0.7536$, $0.7535$ and $0.7535$.
A larger budget does not manufacture true positives; the extra picks land where the metric evaluates nothing.
The loss is in the ordering at the annotated parents the scorer can see, where $8$ of $79$ reachable divisions are captured.
That is a fact about the local annotation; how heavily the hidden set charges false positives was not measured this week.

---

## 5. Three Levers That Died for Reasons Other Than Their Own Measurement

With the universe corrected, three levers were still live.
Each had a positive local measurement, and none died to that measurement.

### 5.1 Leakage: A Joint Lineage Action

A composition that scores parent-and-children decisions jointly had been the project's most-cited undeployed gain for a month.
On the deployed-universe replay it measured $+0.0187$, positive on both embryos.
Then the folds were inspected.
Its heads had been trained on four folds that each held movies from both embryos: movie-out within seen embryos, the fold defect Note 4 took apart.

| head construction (deployed-universe 199-movie OOF) | overall | 71-movie embryo | 128-movie embryo |
|---|---:|---:|---:|
| folds mixing both embryos | $+0.0187$ | $+0.0152$ | $+0.0191$ |
| embryo-disjoint folds | $+0.0004$ | $+0.0150$ | $-0.0021$ |

The head trained on the 128-movie embryo transfers; the head trained on the 71-movie embryo does not.
With no third embryo, "the smaller embryo's model does not generalize" and "the smaller embryo has too few movies" cannot be separated.
I closed the family under the standing rule that a change must be non-negative on both embryos.

### 5.2 The Rules: Sub-Voxel Coordinates

The kernel smooths node positions along short track segments and then writes integer voxel coordinates.
On the deployed-universe replay, quantizing $4.75$ million smoothed positions costs $-0.0050$, and both embryos lose ($-0.0005$ and $-0.0057$).
A kernel writing three-decimal coordinates ran clean and was never submitted.
The competition's Evaluation page specifies integer centroid coordinates in voxels, so the lever is real and forbidden, and the $-0.0050$ is a cost every compliant entry pays.

The other half of the verdict does not depend on the rules.
The four-movie hold-in check on that kernel moved the wrong way: $2024/157/103$ against $2028/155/99$.
In-sample detectors put their peaks on the annotated voxel, so rounding snaps them back onto the answer; embryo-out detectors are noisier, so sub-voxel positions help.
Hold-in is not a noisier out-of-fold; its bias can invert a decision.

This forces a retraction.
On 2026-09-01, interior line-fit smoothing had been recorded as the largest edge-channel local gain of the campaign, and rejected by its own hold-in gate.
Two days later a code audit found that the kernel had been running that same smoothing all along.
The correct statement is not "line-fit smoothing is regime-dependent". It is "line-fit smoothing was already deployed, I did not know, and I measured its second application".

### 5.3 A Deletion Rule That Reversed Sign With the Universe

The last cheap post-processing rule deleted nodes whose incident edges all carry low probability.
It had measured $+0.0009$ on the comparator replay.
On the deployed-universe replay it measured $-0.0106$ at the same setting, and its best variant was a literal no-op.
There, edge probabilities come from a different fusion of the two detectors' scores, and several stages add edges with probability exactly zero, so "all incident probabilities are weak" marks a real cell in a hard region.
The old gain was an artifact of the comparator replay's probability scale.

---

## 6. The Detector Tail: Right Diagnosis, Working Prescription, Losing Composition

### 6.1 Locating the Budget

On the deployed-universe replay, per-movie adjusted edge Jaccard correlates $0.82$ and $0.79$ with node recall in the two embryos, and barely at all with the node-count ratio.
Lifting only the worst decile of each embryo to its median is worth $+0.0277$ and $+0.0233$.
About $10\%$ of the movies hold essentially the whole remaining structural budget.

Profiling individual missed cells in the twelve worst and twelve median movies gave miss rates of $36.5\%$ and $4.6\%$.

| tercile, worst twelve movies | low / mid / high |
|---|---|
| miss rate by intensity at the ground-truth position | $0.579$ / $0.324$ / $0.192$ |
| miss rate by local predicted-node density | $0.653$ / $0.272$ / $0.109$ |

Misses concentrate where the scene is sparse, depth does not matter, and brightness grades them.
Whole dim regions go undetected.

Sampling the detector's own field at ground-truth positions separates the failures.
In the worst twelve, $48\%$ of misses are invisible (the neighborhood-maximum logit is below zero), $18\%$ weak, and $34\%$ above threshold but absorbed by a neighboring peak; in the median twelve the split is $25\%$, $13\%$ and $62\%$.
On normal movies peak merging dominates, an inference-time property; on tail movies the cell is not in the field at all, a training-time property.

The other embryo's tail movies, by contrast, have raw detector recall of $0.987$ and lose their cells later, in association and the solver.
The tail is two failures with one symptom, and only one of them is a detector problem.

### 6.2 The Prescription, and That It Worked

The trainer had no intensity augmentation of any kind, and a failure graded by brightness has the shape of a domain shift.
I added an augmentation: gamma, global gain, a smooth regional-dimming field, an additive haze floor and per-voxel noise.

I wrote the go conditions down before any candidate existed.
At epoch ten, on the worst seven movies of the larger embryo, raw node recall went from $0.7123$ to $0.8884$ and the invisible share of misses from $0.706$ to $0.198$.
The dim tercile's miss rate fell by roughly a factor of four while the bright tercile did not move.
The model also fired more everywhere: peaks per estimated cell rose from $1.701$ to $2.10$ on the tail.

### 6.3 Nine Compositions, Nine Losses

The detector-level win was then priced where the week's repaired rule required, in the deployed pipeline under the official metric, embryo-out, on $15$ movies of the larger embryo: the seven worst and eight near the median.

| composition (15-movie embryo-out pricing subset) | delta, worst seven | delta, median eight | median movies negative |
|---|---:|---:|---:|
| primary detector, deployed threshold | $+0.1007$ | $-0.0198$ | $5/8$ |
| primary detector, strict threshold | $+0.0583$ | $-0.0169$ | $5/8$ |
| detection-only third field, lower weight | $+0.0124$ | $-0.0073$ | $5/8$ |
| novel-peak union, strict threshold | $+0.0417$ | $-0.0306$ | $8/8$ |
| hole-filling union, sparse regions only | $+0.0448$ | $-0.0175$ | $6/8$ |

![For five detector compositions, gains on the worst seven movies and losses on eight near-median movies]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-02-tail-versus-typical.png)
_Figure 2. Every composition gained on the worst seven movies and lost on the eight near the median. Weighting by a tail share near $0.10$ projects every arm negative overall; that projection is not a measurement._

Four further arms had the same shape.
Every arm paired the augmented first split with a deployed second split; the augmented pair itself was never trained.

The tail gains are real: on the worst movie the official adjusted edge Jaccard goes from $0.248$ to $0.487$.
The reweighting still closes the program.
With a tail share $\pi \approx 0.10$, $\mathbb{E}[\Delta S] = \pi\,\Delta_{\mathrm{tail}} + (1-\pi)\,\Delta_{\mathrm{typical}}$ projects every arm between about $-0.005$ and about $-0.025$.
These are projections, not measurements: they assume the priced subsets represent their strata and that hidden embryos have a similar tail.

Two mechanisms were measured separately, and they are the transferable part.

**Added objects crossing the count boundary.** The metric multiplies each movie's edge Jaccard by $1-0.1\,r_i$ with $r_i=(N_{\mathrm{pred},i}-N_{\mathrm{total},i})/N_{\mathrm{total},i}$, where $N_{\mathrm{total},i}$ is the supplied coarse estimate of all cells, so a movie just under that estimate earns a small bonus and one just over pays a penalty.
On one median movie whose supplied estimate is $5{,}257$ cells, the sparse-region hole-filling arm took the node count from $5{,}047$ to $5{,}439$ and the score from $0.7735$ to $0.7050$.
Measured node recall on that movie rose only from $0.9915$ to $0.9957$, about twenty more annotated cells against $392$ added nodes, and the count crossed the estimate.

**Peak displacement at unchanged node counts.** In the gentlest arms the losses arrived with node counts essentially unchanged: one movie went from $0.767$ to $0.722$ at $+0.2\%$ nodes.
Mixing anything into the detection field moves the deployed peaks by a fraction of a voxel, and that breaks marginal $7\,\mu\mathrm{m}$ matches: the precision channel the integer-rounding test exposed, from the other side.
Raising the extraction threshold halved the tail gain and barely touched the median loss, which is what a localization loss predicts.
A control on 2026-09-04 put an ordinary third seed in the detection-only third-field role on the same 15 movies.
It lost $-0.0063$ on the median eight ($6$ of $8$ negative), close to the augmented model's $-0.0073$ in that role.
Most of that arm's typical-movie loss belongs to mixing a third field, not to the augmentation.

A subgroup fix is only a fix if something at test time can route to the subgroup.
That clause is the weakest link in the verdict: the two statistics examined, frame contrast and novel-peak fraction, did not separate the populations, and no serious router search was run.
What survived is the tail diagnosis, a six-minute embryo-out detector evaluation, and the training-side fact that intensity augmentation raises dim-region recall in this model family.

---

## 7. The Rule We Selected By

The rule in force this period was twofold, embryo-disjoint out-of-fold replay over the 199 training movies, with both embryos required to be non-negative.
On paper, the rules I rewrote on 2026-08-28 made the board the objective.
In practice, every candidate submitted this week had been chosen on local evidence first, and each submission's reading was written down before it was sent.
The board was used three ways.
v81 bought a measurement no local instrument could make.
v80, an operating-point change, returned the same score at the board's resolution and decided nothing.
v83 was read against a band written down before the result existed, which gave the board a veto it did not use.

What broke the rule was not a leak.
The rule was measured in the wrong place: on the comparator replay, which the notebook does not run, $0.149$ away in level, with candidate tables that sampled only annotated regions.
A clean criterion aimed at the wrong machine selects just as confidently as a correct one.

The repair moved the measurement into the pipeline that ships: the deployed stack replayed over all 199 movies, with the shipped runtime module as the selection tool.
After the repair, the rule held.
Every remaining verdict of the period was reached locally and without a submission: the joint lineage action, the coordinate lever, the deletion rule, the verifier's operating point and, on a 15-movie subset, the dim-robust detector.

| | this period |
|---|---|
| rule in force | twofold embryo-disjoint OOF, both embryos non-negative |
| where it was measured | the comparator replay until 2026-09-02; then the deployed-stack replay and the shipped runtime module |
| what the board was used for | a decomposition probe (v80/v81, the v80 arm a tie), and a veto over v83 under a pre-written reading rule |
| what broke it, what held | broke: a leak-free criterion in the wrong universe; held: five verdicts reached locally after the repair |

---

## 8. What the Period Established

### Established

1. With v81's graph fork-free, the v79 − v81 subtraction measures the whole hidden division term: $+0.032$, a hidden division Jaccard near $0.32$ against a local $0.062$.
2. On the same 199 movies and scorer, the comparator replay the project selected on scored $0.6005$ at node recall $0.8870$; the deployed stack scored $0.7499$ at $0.9255$.
3. On identical deployed-universe graphs, the union refit gave $+0.0065$ against the deployed selector's $+0.0013$; with the shipped runtime it gives $+0.0046$ ($0.7535$ over $0.7489$), both embryos positive.
4. A predicted division whose parent has no ground-truth node within the matching distance is neither a true nor a false positive. On the local replay the channel was limited by ranking at annotated parents ($8$ of $79$ reachable), not by budget; the hidden set's false-positive cost was not measured.
5. In the deployed universe, $72$ of the $151$ annotated division events have no candidate at all; why was not measured.
6. Neither more division data ($163{,}422$ synthetic positives, comparator-replay signs) nor external detector data ($+0.0466$ on trainer validation, $-0.0272$ through the deployed pipeline on two movies) produced a pipeline gain.
7. With embryo-disjoint heads, the joint lineage action gives $+0.0004$ where mixed folds gave $+0.0187$.
8. About $10\%$ of movies hold the remaining structural budget. In the larger embryo's tail the misses are graded by intensity, and intensity augmentation raised raw recall on its seven worst movies from $0.7123$ to $0.8884$; the other embryo's tail loses its cells after detection. Every priced composition lost on typical movies.

### Supported but Unconfirmed

1. That division changes are not damped on the hidden set. The sign rests on a few single points with rounded numerators and denominators from two local rulers; the edge range has no per-experiment table behind it.
2. That the smaller embryo's head fails to transfer because of its size rather than its difficulty.
3. That no movie-level statistic separates tail from typical movies at test time.
4. That the augmented detector as a pair behaves like its augmented first split.

### Open Questions

1. What makes the hidden division level about five times any local measurement? Correcting the universe lowered the local level instead of raising it.
2. Can the ranking at annotated parents be improved by any signal that is not a new label?
3. Would a detector trained to preserve peak positions avoid the composition loss?

---

## Closing

Five things moved this week.
One was a model change.
The other four were measurements of my own instruments: a probe, a replay, a parity check and a fold audit.

Before this week, the question I asked of a local criterion was whether it leaked.
That question is necessary and not enough.
The second question is whether the criterion is measured in the universe that ships, on the population the shipped code will see.
Here the answer had been no for weeks, and the symptom had been explained away as a population difference.

The detector program is the counterweight: its diagnosis was right, its prescription worked, and the composition lost anyway.
A model can be better exactly where the score is lost and still be a worse submission.

What is left is narrow.
On the local replay, the division channel is ranking-limited at annotated parents, under a ceiling of which about half is unreachable by the candidate generator.
Beyond the union refit, every ranking signal tried from the data in hand (geometry and track features, patch appearance, synthetic pretraining, external real divisions, association logits, terminal-source enumeration) measured zero or worse.
My expectation at the end of the week was that new labels were the one remaining source of ranking signal.
That is an expectation, not a result.
On 2026-09-04 I began labeling division candidates by hand, on my own reading of the competition rules, which the host has not confirmed.
A first pass on $69$ judgments ran end to end and moved the shipped runtime's score on the deployed-stack replay from $0.7535$ to $0.7547$, too small a sample to mean anything yet.
Whatever the labels are worth will be decided the way this week's verdicts were: by the shipped runtime module, on the deployed-stack replay, with both embryos non-negative.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- **Part 6: The Universe We Were Selecting In**
