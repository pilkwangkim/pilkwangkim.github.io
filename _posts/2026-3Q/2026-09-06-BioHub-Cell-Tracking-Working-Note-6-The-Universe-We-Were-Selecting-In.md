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
- Korean version: [BioHub Cell Tracking 작업 기록 6: 모델을 고르던 환경과 배포 환경]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 7: Where a Local Gain Has to Be Measured]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured/)

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

Note 5 ended on 2026-08-28 with a submission the board had not yet answered.
It carried a new division stage: a learned verifier deciding where a cell splits in two (a fork in the lineage graph; the division term is a tenth of the score), in place of hand-tuned gates that had been admitting mostly false forks.
Its ceiling had been measured first, and out of fold it gained $+0.0053$ on the local replay.
It went to the board as a transfer check, the one way to see whether a gain chosen on the two training embryos survives on embryos the models have never seen.

The answer came on 08-29.
The Public score read $0.935$ against $0.921$ for the deployment before it, about $2.6$ times the local gain.
On a board that rounds to three decimals, $0.002$ is a tie and $0.014$ is not.
A move that far from what the local numbers implied says the local instrument is missing something.

This note covers 2026-08-29 to 2026-09-04 and follows that signal: what does the hidden set see in the division term, and why does the local replay see so much less?
The answer concerned the instrument more than the model.
The local out-of-fold (OOF) replay was clean and embryo-disjoint, and it had been scoring a different set of graphs from the ones the submitted notebook builds.
The week used three submissions.
Every model decision was made locally; the board was asked for one measurement no local instrument could make and for one transfer check whose reading was written down in advance.

The short version is:

```text
The verifier's transfer check came back far above its local gain.
Two submissions then priced the hidden division term directly: a hidden
division Jaccard near 0.32, five times the local 0.062. That gap was the clue.
Replaying the pipeline that actually ships showed the local instrument had
been scoring different graphs, 0.149 lower on the same movies and scorer.
Refit where it runs, the verifier gained locally, and its transfer check
landed in the band written in advance for making it the base.
Re-measured there, three older levers and a detector fix were closed locally.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What looked strange when the week opened? |
| 1 | Was the verifier starved of data? Did the detector have more to give? |
| 2 | How is a term priced that no local instrument can see? |
| 3 | Why was the hidden division level five times the local one? |
| 4--5 | What did refitting where the code runs buy? |
| 6 | Which older levers survived re-measurement? |
| 7 | Why was a detector fix that worked closed? |
| 8--9 | What was decided, and what was established? |

---

## 0. Where the Week Opened

The pipeline runs inside the submitted Kaggle notebook (the kernel): two seeded detectors whose blended fields yield cell nodes, a transformer that scores links between frames, an integer linear program (ILP) that selects the graph, and post-stages.
The newest post-stage is the verifier, a gradient-boosted model retrained inside the notebook from a shipped candidate table, which decides which forks to add.

The competition's four labeled example movies are copies of training movies, so scoring the deployed all-train models on them is a hold-in check: it can catch damage, and it cannot select.
Selection uses embryo-disjoint (embryo-out) replay: the 199 training movies come from two embryos, of 71 and 128 movies, so each of two folds scores one embryo with models trained on the other, and a change must be non-negative on both.

| item, entering 2026-08-29 | value |
|---|---:|
| the comparator replay (Note 5's out-of-fold comparator), 199 movies, old division stage in place | $0.6014$ |
| local gain of the shipped verifier on that comparator | $+0.0053$ |
| Public score of the first verifier deployment (v79) | $0.935$ |
| Public score of the deployment before it | $0.921$ |

Two things in that table were strange.
The first was the ratio, about $2.6\times$, where the project had been citing a damped range of $0.14\times$ to $0.9\times$ for edge-channel changes, a range with no per-experiment table behind it.
The second was older: a local base near $0.60$ against a Public score near $0.94$, a gap attributed for weeks to a population difference and never tested.
Both rested on one working assumption, that the local out-of-fold graphs are a faithful, if pessimistic, stand-in for what the notebook produces.
At the start of the week there was no instrument to test it.

---

## 1. First Bets on Named Bottlenecks

The first reading of $0.935$ was the simple one: the verifier works, so push it further.
It had captured about a tenth of the $+0.0557$ oracle ceiling on roughly $80$ positives, from a competition whose entire division supervision is $151$ annotated events, so starvation was the obvious bottleneck; the week's first bets aimed at it, with the detector tested in parallel.
The verifier numbers come from the comparator replay that Section 3 retires; their signs survive that retirement, their levels do not.

### 1.1 Four Bets on the Verifier

| bet | bottleneck it targeted | what came back |
|---|---|---|
| widen the candidate generator | recall: it reached only $75$ of the $151$ events | positives $80 \to 110$, pool nearly doubled; applied delta $+0.0053 \to +0.0021$ |
| three-patch appearance CNN, embryo-pure | features limited to geometry and tracks | training loss $1.42 \to 0.075$; held-out median true division below the 99th percentile of negatives, both folds |
| ranker pretrained on a public synthetic dataset | too few positives | $163{,}422$ positives, holdout AUC $0.9988$; as a feature, the same score to four decimals |
| CNN on $23{,}977$ real divisions from a public zebrafish dataset | wrong domain | generalized across embryos; added less than $10^{-4}$ |

The wider pool shifted the distribution downstream: an operating point calibrated on one pool admits a worse marginal candidate when the pool doubles.
The CNN, with $61$ and $19$ positives per training fold, could only memorize.
The synthetic ranker solved its own domain, but in the real ranker's feature space real divisions are not separable from real false forks.
"Too few positives" and "wrong domain" were real defects; fixing either moved nothing, so neither was binding.

### 1.2 Two Bets on the Detector

The detector bets asked whether the detector had more to give from epochs alone, or from a dense external dataset the host explicitly permitted.
With an unchanged recipe taken from epoch $200$ to $400$, embryo-out validation read $0.8151$, peaked briefly at $0.8326$ at epoch $218$, and fell to $0.7907$; closing the program cancelled about $95$ GPU-hours of queued runs.
An epoch-$400$ evaluation byte-identical to epoch $200$ exposed a resume bug in best-checkpoint saving: independent stochastic runs do not produce identical integers.

Because of that bug, the external-data fine-tune paired each epoch with a control resumed from the identical checkpoint.
Against a gate written in advance at $+0.005$, it read $+0.0466$, the largest validation gain the project had produced; then it went through the deployed pipeline as the primary detector, on the two embryo-out example movies.

| arm (2-movie embryo-out pipeline harness) | score | node recall | predicted nodes, dense movie |
|---|---:|---:|---:|
| control pair | $0.8028$ | $0.9662$ | $64{,}477$ |
| fine-tuned as primary | $0.7756$ | $0.9625$ | $66{,}122$ |

The trainer's metric moved $+0.0466$ and the end-to-end metric $-0.0272$ on the same held-out embryo.
The fine-tuned model found more objects and placed them slightly worse, and both node matching (an optimal assignment within $7\,\mu\mathrm{m}$) and the ILP's linking costs pay for where each peak sits.
Removing the second field, moving the threshold and swapping the detectors' roles each lost.
One arm, novel peaks added with the field blend off, scored $0.7890$ and looked about $+0.03$ over an assumed single-detector base near $0.75$ to $0.76$; that base, once run, scored $0.7972$, so the additions were worth $-0.0082$.
On two movies, I closed the fine-tune as a deployment detector: a validation metric computed outside the shipped pipeline is not the pipeline's score, which is Section 3's finding one level down.

---

## 2. Pricing the Hidden Division Term With Two Submissions

### 2.1 Why Only the Board Could Answer

None of those bets answered the question the $0.935$ raised: what is the division stage worth on the hidden set?
On the four example movies the deployed stage made $45$ edits and changed the edge score by exactly zero, because all $45$ landed off the sparse annotation.
Local scoring cannot see such a stage; the hidden set scores it.

### 2.2 The Probe

A decomposition probe is a pair of submissions that differ in exactly one stage, so that the difference between their scores prices that stage on the hidden set with no local proxy in the path.
On 2026-09-01 I spent two submissions on one.
The first arm, v80, was the deployed notebook at a swept operating point for the division stage.
The second arm, v81, was the first deployment's notebook (its models and tables) with the verifier's *apply* step, the function that writes its chosen forks into the graph, replaced by a function that returns its inputs unchanged.
The model was still fitted, so every downstream stage saw identical inputs; only the edits were withheld.
With *apply* neutralized, v79's and v80's constants govern nothing, so one probe arm serves both.

| arm (Public split) | what it is | Public |
|---|---|---:|
| v79 | first verifier deployment | $0.935$ |
| v80 | swept operating point, verifier active | $0.937$ |
| v81 | v79's notebook, apply neutralized | $0.903$ |

The subtraction prices $\Delta J_{\mathrm{edge}}^{\mathrm{adjusted}} + 0.1\,\Delta J_{\mathrm{division}}$ at $+0.032$ against v79 and $+0.034$ against v80.
Divided by the division weight of $0.1$, the hidden division Jaccard is near $0.32$ to $0.34$.
The same stage's local out-of-fold division Jaccard was $0.062$.

v81's graph had no forks at all, so the subtraction is the whole hidden division term at v79, not the verifier's gain over the stage it replaced; that step was the $+0.014$ of Section 0.
It folds in the stage's small edge footprint (one false positive on the four example movies), and the board's rounding makes $+0.032$ really $+0.032 \pm 0.001$, a hidden division Jaccard between about $0.31$ and $0.33$.
Even so, the hidden division level is roughly five times the local one: a subtraction of two rounded numbers, bought with two slots.

The swept arm rode along: its local gain of $+0.0007$ was non-negative on both embryos but below the $+0.001$ asked of a standalone change, and $0.937$ against $0.935$ is the same score at the board's resolution.
The probe chose nothing; it measured a quantity, and the quantity pointed at the local instrument rather than at a model.
That is the week's first clause: a term only the hidden set can see is priced by a designed decomposition probe, not inferred from a local proxy.

---

## 3. The Clue: Replaying the Pipeline That Ships

### 3.1 Two Readings of a Fivefold Gap

A fivefold gap allows two readings: the hidden set is denser in scorable divisions (the explanation on offer the day before the replay), or the local instrument was measuring a different machine.
The second could be tested directly, and if it held it would touch every local number in the project.

Every learned component here is a selector fitted on a table of candidates, and every table comes from running some pipeline over the training movies.
Call the set of candidate graphs a pipeline produces its *universe*.
The failure mode is a selector $\phi^{*}$ tuned on candidates $\mathcal{C}(\mathcal{U}_{\mathrm{fit}})$ and shipped to act on $\mathcal{C}(\mathcal{U}_{\mathrm{run}})$, with $\mathcal{U}_{\mathrm{fit}} \ne \mathcal{U}_{\mathrm{run}}$.
This is not leakage: both universes can be embryo-disjoint and every fold clean, and each is internally consistent, so the shift is invisible from inside either.

### 3.2 The Deployed-Stack Replay

A deployed-stack replay runs the notebook's own pipeline over all 199 training movies, with embryo-disjoint checkpoints so every movie is scored by models that never saw its embryo, and scores the output with the official scorer.
It had been named as decisive almost four weeks earlier, and Note 5 had left open whether the comparator graphs were the ones the notebook builds; the probe's fivefold reading made it the first thing to run, on 2026-09-02.

| universe (199 movies, embryo-out, official scorer) | score | node recall, final graph |
|---|---:|---:|
| comparator replay, forks stripped, where the recent local selections were made | $0.6005$ | $0.8870$ |
| deployed stack, the notebook's own output, same movies, same scorer | $0.7499$ | $0.9255$ |

![Two scores on the same 199 movies: comparator replay 0.6005 and deployed stack 0.7499, 0.149 apart]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-01-two-instruments.png)
_Figure 1. Two instruments reading the same 199 training movies with the same scorer. The August selections were made on the comparator replay, the lower-scoring instrument; the notebook runs the deployed stack._

The rows are not a before and after; they are two instruments reading the same movies, $+0.149$ apart.
Every verifier threshold, deletion rule and association arm of the preceding weeks had been chosen on graphs with node recall $0.887$, while the notebook ran a stack at $0.9255$.
Part of the old gap between a local base near $0.60$ and a Public score near $0.94$ was therefore never a population difference; the rest may still be one, and this week did not test it.

The replay also settled the bookkeeping of levels, open since Note 3 recorded seven baselines in use at once: the $0.6014$ of Section 0 is the same comparator with the old forks left in, Note 4's research replay (base near $0.74$) is another graph family, and from here on levels are quoted in one reference, the deployed-stack replay.

Inside the deployed universe, with the notebook's raw forks collapsed, the base is $0.7489$, and every division number below is measured over it.
A label oracle over the candidates the deployed pipeline actually produces scores $0.8007$, at division Jaccard $0.5097$, with $79$ true positives, $4$ false and $72$ missed.
Those $72$ events have no candidate row at all, so roughly half of the division ceiling is unreachable by the candidate generator before any ranking question is asked; why was not measured this week.

### 3.3 What the Replay Did Not Explain

On deployed-universe graphs the deployed verifier drew $2$ true and $3$ false divisions, a local division Jaccard near $0.013$, lower than the $0.062$ measured on the comparator replay.
Correcting the machine widened the distance to the hidden $0.32$, so the population reading stays open.
What the replay did expose was a second, fixable problem: the verifier had been fitted on candidates from one universe and was being applied to candidates from another.
That is the week's second clause: a local criterion is measured in the deployed universe, by replaying the pipeline that ships.

---

## 4. Refitting the Verifier Where It Runs

The test of the fitting mismatch holds the deployed-universe graphs fixed and changes only the candidates the verifier is fitted on.

| fit (deployed-universe 199-movie OOF, embryo-disjoint, base $0.7489$) | delta over base | division TP / FP |
|---|---:|---:|
| deployed verifier, fitted on comparator-replay candidates | $+0.0013$ | $2 / 3$ |
| refit on deployed-universe candidates only | $+0.0030$ | $5 / 13$ |
| refit on the union of both tables | $+0.0065$ | $17 / 90$ |

The deployed selector captured a fifth of what was available, and on the larger embryo it fired nothing at all across $128$ movies: well fitted, embryo-disjoint, and aimed at a distribution the pipeline no longer produces.

I selected the union fit.
Its sweep was a plateau rather than a spike, and both embryos were positive at the peak ($+0.0096$ and $+0.0060$).
Translating it to the runtime replaced per-embryo thresholds with one flat threshold chosen by a rule fixed in advance.
The kernel passed its hold-in gate on the four example movies with the edge channel inert, as a division-only change must: edge true/false/missed counts of $2028/155/99$ against the previous $2027/155/100$.

The refit was a local choice, and v83 was its transfer check.
Before submitting it, I wrote down how its Public score would be read:

```text
>= 0.940       the universe refit transfers; make it the base
0.937 - 0.939  neutral; the refit is not distinguishable
< 0.937        the flat threshold hurt; revert to per-embryo
```

It returned $0.944$, $+0.007$ over v80's $0.937$, the reference its bands were written against, and inside the band written for making the refit the base.
That is no failure detected, not a confirmation; the refit had been chosen locally, and it became the base.

Local $+0.0065$ against Public $+0.007$ (over v80) is a transfer of about $1.1\times$, or nearer $1.5\times$ with the corrected $+0.0046$ of Section 5; the earlier $2.6\times$ came from the comparator replay, a different ruler, and each ratio is one point with a rounded numerator.
The working expectation I took from them, and used to order the queue, was only a sign: division changes had not been damped on the hidden set.

---

## 5. The Shipped Code as the Selection Tool

On 2026-09-03 a parity check compared the shipped runtime module inside the kernel with the same recipe run locally.
On the four example movies, the kernel applied $50$, $23$, $3$ and $50$ divisions; the local recipe applied none.

The cause was a flag.
Every local rebuild of the division tables had generated candidates only around annotated sources, which looks sensible, because only annotated regions produce labels.
On one of the four example movies that yields $7{,}519$ rows; the shipped runtime, which does not know where the annotations are, generates $358{,}000$ on the same movie.
Every local sweep of the stage's threshold and budget had been priced without the false-positive load the kernel carries, so every local optimum was biased loose.

This is Section 3's shift at a smaller scale, and the fix follows the same clause: the shipped runtime module itself became the selection tool, and rebuilt tables became training material only.
Re-run that way, the deployed operating point was already the optimum: $0.7535$ against the fork-free $0.7489$, $+0.0046$ with both embryos positive, and no threshold or cap change cleared the rule written down for a new operating point.

The same sweep exposed a structural fact of the metric.
A predicted division at a parent with no ground-truth node within the matching distance is neither a true nor a false positive.
The all-train, in-sample run applied $4{,}662$ divisions and drew only $71$ false positives, against $42$ true ones.
In the embryo-out arms, per-movie caps spanning a factor of four scored $0.7536$, $0.7535$ and $0.7535$: a larger budget does not manufacture true positives, because the extra picks land where the metric evaluates nothing.
The loss is in the ordering at the annotated parents the scorer can see, where $8$ of $79$ reachable divisions are captured; on the local annotation, the channel is ranking-limited.
How heavily the hidden set charges false positives was not measured this week.

---

## 6. Three Older Levers, Re-measured

With the universe corrected, three levers with positive earlier measurements were measured again, and each was closed by a different fact: a fold defect, the competition's rules, and the change of universe itself.

### 6.1 A Joint Lineage Action and Its Folds

A composition that scores parent-and-children decisions jointly had been the project's most-cited undeployed gain for a month, and on the deployed-universe replay it measured $+0.0187$, positive on both embryos.
A fold audit then found its heads had been trained on four folds that each held movies from both embryos, the defect Note 4 took apart.

| head construction (deployed-universe 199-movie OOF) | overall | 71-movie embryo | 128-movie embryo |
|---|---:|---:|---:|
| folds mixing both embryos | $+0.0187$ | $+0.0152$ | $+0.0191$ |
| embryo-disjoint folds | $+0.0004$ | $+0.0150$ | $-0.0021$ |

The head trained on the 128-movie embryo transfers; the head trained on the 71-movie embryo does not.
With no third embryo, "the smaller embryo's model does not generalize" and "the smaller embryo has too few movies" cannot be separated.
I closed the family under the standing rule that a change must be non-negative on both embryos.

### 6.2 Sub-Voxel Coordinates and the Rules

The kernel smooths node positions along short track segments and then writes integer voxel coordinates.
On the deployed-universe replay, quantizing $4.75$ million smoothed positions costs $-0.0050$, with both embryos losing ($-0.0005$ and $-0.0057$).
A kernel writing three-decimal coordinates ran clean and was not submitted, because the competition's Evaluation page specifies integer centroid coordinates in voxels: a real lever, not allowed, and a cost every compliant entry pays.

The hold-in check on that kernel moved the wrong way ($2024/157/103$ against $2028/155/99$), for a reason that does not depend on the rules.
In-sample detectors put their peaks on the annotated voxel, so rounding snaps them back onto the answer; embryo-out detectors are noisier, so sub-voxel positions help.
Hold-in is not a noisier out-of-fold; its bias can invert a decision.

One retraction belongs here.
On 2026-09-01, interior line-fit smoothing had been recorded as the largest edge-channel local gain of the campaign, and rejected by its own hold-in gate.
Two days later a code audit found that the kernel had been running that same smoothing all along: the 09-01 number measured a second application of it, not a regime-dependent lever.

### 6.3 A Deletion Rule That Changed Sign With the Universe

The last cheap post-processing rule deleted nodes whose incident edges all carry low probability.
It had measured $+0.0009$ on the comparator replay; on the deployed-universe replay it measured $-0.0106$ at the same setting, and its best variant was a literal no-op.
There, edge probabilities come from a different fusion of the two detectors' scores, and several stages add edges with probability exactly zero, so "all incident probabilities are weak" marks a real cell in a hard region.
The old gain belonged to the comparator replay's probability scale.

---

## 7. The Detector Tail: Right Diagnosis, Closed Program

### 7.1 Where the Remaining Budget Sits

On the deployed-universe replay, per-movie adjusted edge Jaccard correlates $0.82$ and $0.79$ with node recall in the two embryos.
Lifting only the worst decile of each embryo to its median is worth $+0.0277$ and $+0.0233$: about $10\%$ of the movies hold essentially the whole remaining structural budget, which made them the natural next target.
Profiling the missed cells of the twelve worst and twelve median movies gave miss rates of $36.5\%$ and $4.6\%$.

| tercile, worst twelve movies | low / mid / high |
|---|---|
| miss rate by intensity at the ground-truth position | $0.579$ / $0.324$ / $0.192$ |
| miss rate by local predicted-node density | $0.653$ / $0.272$ / $0.109$ |

Misses concentrate where the scene is sparse, and brightness grades them: whole dim regions go undetected.
Sampling the detector's own field at ground-truth positions shows why: in the worst twelve, $48\%$ of misses are invisible (the neighborhood-maximum logit is below zero), a training-time property; in the median twelve, $62\%$ are above threshold but absorbed by a neighboring peak, an inference-time one.
The other embryo's tail movies, by contrast, have raw detector recall of $0.987$ and lose their cells later, in association and the solver: two failures with one symptom, only one of them a detector problem.

### 7.2 The Prescription, and That It Worked

The trainer had no intensity augmentation of any kind, and a failure graded by brightness has the shape of a domain shift.
I added one (gamma, global gain, a smooth regional-dimming field, an additive haze floor and per-voxel noise) and wrote the go conditions down before any candidate existed.
At epoch ten, on the worst seven movies of the larger embryo, raw node recall went from $0.7123$ to $0.8884$ and the invisible share of misses from $0.706$ to $0.198$; the dim tercile's miss rate fell by roughly a factor of four while the bright tercile did not move.
The model also fired more: peaks per estimated cell rose from $1.701$ to $2.10$ on the tail.

### 7.3 Priced in the Pipeline

A detector-level win is a component result, so it was priced where the week's new clause requires: in the deployed pipeline under the official metric, embryo-out, on $15$ movies of the larger embryo, the seven worst and eight near the median.

| composition (15-movie embryo-out pricing subset) | delta, worst seven | delta, median eight | median movies negative |
|---|---:|---:|---:|
| primary detector, deployed threshold | $+0.1007$ | $-0.0198$ | $5/8$ |
| primary detector, strict threshold | $+0.0583$ | $-0.0169$ | $5/8$ |
| detection-only third field, lower weight | $+0.0124$ | $-0.0073$ | $5/8$ |
| novel-peak union, strict threshold | $+0.0417$ | $-0.0306$ | $8/8$ |
| hole-filling union, sparse regions only | $+0.0448$ | $-0.0175$ | $6/8$ |

![For five detector compositions, gains on the worst seven movies and losses on eight near-median movies]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-02-tail-versus-typical.png)
_Figure 2. Every composition gained on the worst seven movies and lost on the eight near the median. Weighting by a tail share near $0.10$ projects every arm negative overall; that projection is not a measurement._

Four further arms had the same shape; every arm paired the augmented first split with a deployed second split (the augmented pair was never trained).
The tail gains are real (on the worst movie the official adjusted edge Jaccard goes from $0.248$ to $0.487$), but the typical movies decide the verdict.
With a tail share $\pi \approx 0.10$, $\mathbb{E}[\Delta S] = \pi\,\Delta_{\mathrm{tail}} + (1-\pi)\,\Delta_{\mathrm{typical}}$ projects every arm between about $-0.005$ and about $-0.025$: projections that assume the priced subsets represent their strata and that hidden embryos have a similar tail.

**Added objects crossing the count boundary.** The metric scales each movie's edge Jaccard by $1-0.1\,r_i$, where $r_i$ is the relative excess of predicted nodes over a supplied coarse cell estimate, so a movie just under the estimate earns a small bonus and one just over pays a penalty.
On one median movie estimated at $5{,}257$ cells, the hole-filling arm took the node count from $5{,}047$ to $5{,}439$ and the score from $0.7735$ to $0.7050$, while node recall rose only from $0.9915$ to $0.9957$: about twenty more annotated cells for $392$ added nodes.

**Peak displacement at unchanged node counts.** In the gentlest arms one movie went from $0.767$ to $0.722$ at $+0.2\%$ nodes.
Mixing anything into the detection field moves the deployed peaks by a fraction of a voxel and breaks marginal $7\,\mu\mathrm{m}$ matches, the precision channel the rounding test exposed; raising the extraction threshold halved the tail gain and barely touched the median loss, as a localization loss predicts.
A control on 2026-09-04 put an ordinary third seed in the detection-only role on the same 15 movies: it lost $-0.0063$ on the median eight ($6$ of $8$ negative), close to the augmented model's $-0.0073$, so most of that arm's typical-movie loss belongs to mixing a third field, not to the augmentation.

### 7.4 Why the Program Was Closed

A subgroup fix needs something at test time that routes to the subgroup.
The two statistics examined, frame contrast and novel-peak fraction, did not separate the populations, and no wider router search was run, which is the weakest link in the verdict.
Without a router every priced composition trades a real tail gain for a larger typical loss, so I closed the program.
What survived is the tail diagnosis, a six-minute embryo-out detector evaluation, and the training-side fact that intensity augmentation raises dim-region recall in this model family.

---

## 8. Decision Log

The rule in force was the two-fold embryo-disjoint out-of-fold replay over the 199 training movies, with both embryos non-negative.
The rules rewritten on 2026-08-28 had named the board as the objective, as a guard against another stall; in practice every candidate this week was chosen on local evidence first, and each submission's reading was written down before it was sent.
The board was asked only for what it alone could give: v81, with v80 riding along, measured a term no local instrument could see, and v83 was the transfer check of a locally chosen refit.
The week changed where the rule is measured: from the comparator replay to the deployed-stack replay and the shipped runtime module.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| six bets on named bottlenecks (Section 1) | the verifier held a tenth of its ceiling on about $80$ positives | no verifier gain; detector trainer $+0.0466$, pipeline $-0.0272$ | data volume and domain not binding; all six closed |
| v80/v81 decomposition probe (09-01) | the stage acts off the annotation; only the hidden set scores it | hidden division term $+0.032$, Jaccard near $0.32$ against a local $0.062$; v80 a tie | C13; the gap pointed at the local instrument |
| deployed-stack replay (09-02) | the fivefold gap; the untested $0.60$ against $0.94$ gap | $0.7499$ against $0.6005$, same movies and scorer | C12; one reference for levels (C5) |
| refit in the deployed universe; v83 as its transfer check (09-03) | the deployed verifier, fitted in the other universe, drew $2$ TP and $3$ FP | local $+0.0065$, both embryos positive; Public $0.944$, in the "make it the base" band | the refit became the base |
| shipped runtime as the selection tool (09-03) | the local recipe applied no divisions where the kernel applied dozens | deployed operating point already optimal, $+0.0046$; caps do not bind | rebuilt tables are training material only |
| re-measure three older levers | each had a positive earlier measurement | $+0.0004$ embryo-disjoint; integer coordinates required; $-0.0106$ | all three closed without a submission |
| detector-tail program (09-04) | about $10\%$ of movies hold the remaining budget | tail recall $0.7123 \to 0.8884$; every composition lost on median movies | closed for lack of a router; I began hand labeling |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 | Write each gate down before the result exists | Note 3 (07-15) |
| C3 | Calibrate a rule on the population it will act on | Note 3 |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 | Compare levels only inside one reference universe; compare deltas across — reconciled here (the deployed-stack replay is the reference) | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 | Use the board for matched transfer checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 | A gate must be able to end in a decision | Note 5 |
| C12 **(new)** | Measure in the deployed universe: replay the pipeline that ships | Note 6 |
| C13 **(new)** | Price a hidden-only term with a designed decomposition probe | Note 6 |

---

## 9. What the Period Established

### Established

1. With v81's graph fork-free, the v79 − v81 subtraction measures the whole hidden division term: $+0.032$, a hidden division Jaccard near $0.32$ against a local $0.062$.
2. On the same 199 movies and scorer, the comparator replay scored $0.6005$ (node recall $0.8870$) and the deployed stack $0.7499$ ($0.9255$).
3. On identical deployed-universe graphs, the union refit gave $+0.0065$ against the deployed selector's $+0.0013$; with the shipped runtime it gives $+0.0046$ ($0.7535$ over $0.7489$), both embryos positive.
4. A predicted division whose parent has no ground-truth node within the matching distance is neither a true nor a false positive. Locally the channel was limited by ranking at annotated parents ($8$ of $79$ reachable), not by budget; the hidden set's false-positive cost was not measured.
5. In the deployed universe, $72$ of the $151$ annotated division events have no candidate at all.
6. External detector data read $+0.0466$ on trainer validation and $-0.0272$ through the deployed pipeline on two movies.
7. With embryo-disjoint heads, the joint lineage action gives $+0.0004$ where mixed folds gave $+0.0187$.
8. Intensity augmentation raised raw recall on the larger embryo's seven worst movies from $0.7123$ to $0.8884$, and every priced composition lost on typical movies.

### Supported but Unconfirmed

1. That division changes are not damped on the hidden set. The sign rests on a few single points with rounded numerators and denominators from two local rulers.
2. That the smaller embryo's head does not transfer because of its size rather than its difficulty.
3. That no movie-level statistic separates tail from typical movies at test time.
4. That the augmented detector as a pair behaves like its augmented first split.

### Open Questions

1. What makes the hidden division level about five times any local measurement? Correcting the universe lowered the local level instead of raising it.
2. Can the ranking at annotated parents be improved by any signal that is not a new label?
3. Would a detector trained to preserve peak positions avoid the composition loss?

---

## Closing

Five things moved this week: one model change, and four measurements of the instruments themselves (a probe, a replay, a parity check and a fold audit).
Before this week, the question asked of a local criterion was whether it leaked.
The week added a second: whether it is measured in the universe that ships, on the population the shipped code will see.

What is left is narrow.
On the deployed-stack replay the division channel is ranking-limited at annotated parents, under a ceiling of which about half is unreachable by the candidate generator.
Beyond the union refit, every ranking signal tried from the data in hand (geometry and track features, patch appearance, synthetic pretraining, external real divisions, association logits, terminal-source enumeration) measured zero or worse.
My expectation at the end of the week, not a result, was that new labels were the one remaining source of ranking signal.
On 2026-09-04 I began labeling division candidates by hand, on my own reading of the competition rules, which the host has not confirmed; a first pass on $69$ judgments moved the shipped runtime's score on the deployed-stack replay from $0.7535$ to $0.7547$, too small a sample to mean anything yet.

The labels will be judged the way this week's verdicts were: by the shipped runtime module, on the deployed-stack replay, with both embryos non-negative.
A gain that clears that bar is a local gain measured in the deployed universe; whether it survives on unseen embryos is a question only its transfer check can answer, and that is where the next note begins.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- **Part 6: The Universe We Were Selecting In**
