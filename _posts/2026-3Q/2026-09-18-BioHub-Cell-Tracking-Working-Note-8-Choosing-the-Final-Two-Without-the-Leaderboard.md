---
title: "BioHub Cell Tracking Working Note 8: Choosing the Final Two Without the Leaderboard"
date: 2026-09-18 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, final-selection, stage-jitter, registration, association-head, embryo-out, selection-bias, oof, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-18-biohub-working-note-8/cover.png
  alt: "Title card for BioHub Working Note 8: choosing the final two without the leaderboard"
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

# BioHub Cell Tracking Working Note 8: Choosing the Final Two Without the Leaderboard

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
  - [Working Note 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
  - [Working Note 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
  - [Working Note 7: Where a Local Gain Has to Be Measured]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured/)
- Korean version: [BioHub Cell Tracking 작업 기록 8: 리더보드 없이 최종 두 개를 고르기]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-Choosing-the-Final-Two-Without-the-Leaderboard-KR/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 7 ended on 2026-09-12 with three questions for any local gain: is it free of leakage, is it measured in the code that ships, and is it measured on the weights and composition the kernel actually loads?
The pseudo-label detector had failed the first and the last: its leakage control lost on both embryos, and its deployment, v88, had collapsed on the board at $0.924$.
The division channel ended that week with no measured account of where the missed divisions sat; that account arrived on the morning of 09-13, and the fix it suggested had reversed before noon.
v90 stood at the same Public score as v87, and v91, v90 with the secondary detector's features averaged over test-time views, went in at 01:08 on 09-13.
All times in this note are KST.

This note covers 2026-09-13 through 2026-09-18, the last six days of work before I closed the project.
It opens with two decisions made on 09-13.
The first narrowed the board to a single question: did a submission collapse?
The second was a question I put to the project: had every possibility actually been examined?
The first is why no Public number below serves as evidence for a model.
The second produced the only change of the period that was positive in every regime I could measure.

The period ends with two submissions chosen for the final score on local evidence.
One of them had the lowest Public score of the recent candidates.

The short version is:

```text
In the last six days the board was read only for collapse; none of four scores was one.
v92, a label-free registration of stage jitter, passed every gate in both regimes.
v93, an association-head refit, passed its embryo-out gate and shipped past a failed
kernel-regime clause under a recorded exception.
Eleven lanes after the refit produced no candidate: nine stopped at bars written before
their results, one was a post-result diagnostic, and one advanced only to a design.
The final two were picked on local evidence: v93 for its held-out-embryo gain,
and v92 as the hedge against that gain failing to transfer.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | Which instruments decided things? |
| 1 | What exactly did the board become on 09-13? |
| 2 | What did the division trace suggest, and why did its fix reverse at 199 movies? |
| 3 | What did the possibility audit find, and what did registration buy? |
| 4 | Why did the head refit pass embryo-out and stop in the kernel regime? |
| 5--6 | Why did the eleven lanes after it stop, and what was v94? |
| 7 | Where do the 151 annotated divisions go, and could any signal rank them? |
| 8 | How did selection inflate a gain, and what caught a v88-type failure? |
| 9 | Which two, and why? |
| 10--11 | Which rule selected, and what was established? |

---

## 0. Where the Period Opened

As in the earlier notes, two detectors are fused before peak extraction, a transformer association head scores edges, an ILP selects the graph, and deterministic stages repair it.
A fitted division verifier adds forks whose score clears $0.90$, at most $50$ per movie.
The deployed baseline was v90.

Three instruments decided things, in universes whose numbers are never compared with one another.

| instrument | what it runs | role |
|---|---|---|
| EO (embryo-out) | detection and association from models that never trained on the scored embryo; downstream stages all-train; 199 movies in two embryos (44b6: 71, 6bba: 128) | efficacy; the deciding gate |
| KR (kernel regime) | the deployed weights and exact notebook code over the same 199 movies | harm detection in the machine that ships; in-sample (what earlier notes called hold-in) |
| the four example movies | the notebook's actual output, scored locally | deployment parity only |

EO is the deployed-stack replay of Notes 6–7 run with embryo-out weights.
It is not a pure out-of-fold graph estimate: the backbone checkpoints behind it were selected on each fold's evaluation embryo, an exposure every card disclosed.
Note 7's replay numbers rest on the same checkpoints and carry the same exposure.
EO was still the closest thing to unseen embryos the project had.

Every lane had a card, with its question, gate and stop rule committed before any result existed.
EO gates report both embryos and a bootstrap over movies; LB90 is the lower bound of its 90% interval.

The rule in force was the one I had reinstated on 09-05 and 09-06: selection was local, a Public difference within $\pm 0.002$ was a tie, and the board was cited only for moves of about $0.003$ or more.

---

## 1. The Board Becomes an Anomaly Detector

v91 came back at 11:10 on 09-13 at $0.946$, the same score as v90 at the board's resolution, which its card had said in advance would establish nothing.

That evening I narrowed the board's role again, in a short written rule:

```text
One submission per validated candidate.
Read its Public score for one thing: a drop of 0.003 or more below the deployment parent.
A drop starts a verification, not a decision.
No card carries a band of the form "a transfer if the score is at least X".
Deployment decisions come from the pre-registered local gates: EO and KR.
I designate the final two, and the highest Public score is never the selection rule.
```

A drop sends me to local tools for a defect; if none is found, it is noise in the Public sample.

The $\pm 0.002$ rule had still let the board confirm things, and a card with a "positive transfer" band invites reading it as evidence.
The Public score covers 29% of the test, rounded to three decimals.
It had carried a few real signals (v83's $+0.007$, v85's $-0.005$, v88's collapse), but it could not rank candidates whose local differences were a few thousandths.

---

## 2. A Division Diagnosis and Two Reversals at Scale

At 06:18 on 09-13 a trace of the v90 division path over 199 movies placed all $151$ annotated divisions: $23$ recovered, $72$ with no correct candidate, and $56$ with a correct candidate scored below the $0.90$ threshold.
It ran on the embryo-out replay with a verifier fitted on the other embryo; the EO gates below use an all-train verifier, under which v90 has TP $36$.
In $46$ of those $56$ the best correct candidate already ranked first among its own parent's candidates; the median of those best scores was $0.065$.
I read that as calibration, not ranking: the verifier ordered the right answer first and scored it too low, most plausibly because its training labels did not match the current graph's candidates.

The test was a refit of the same verifier on labels valid for those candidates.
On four movies chosen because they held candidate events it gained $+0.01099$, from one extra true division.
At 08:19 it came back from all 199 movies.

| | 4 movies (chosen for events) | 199 movies |
|---|---:|---:|
| official $\Delta$ | $+0.01099$ | $-0.00679$ |
| 44b6 | $+0.00014$ | $+0.00364$ |
| 6bba | $+0.0164$ | $-0.00845$ |
| division TP/FP/FN | $+1$ / $+2$ / $-1$ | 23/37/128 $\to$ 22/196/129 |

Nodes and recall were unchanged; the verifier simply fired more: applications went from $2{,}709$ to $8{,}297$.
The refit did not so much rank differently as score everything higher, and raising the level bought false divisions, not true ones.

That afternoon a quantile alignment meant to redeploy Note 7's pseudo-label detector scored $-0.033027$ on a 29-movie kernel-regime panel, with $25$ of $29$ movies down, after a two-movie smoke test at $+0.104$ and $-0.0016$.
The signal that motivated it, a 6bba tail median of $+0.066$ on the embryo-out replay, had the opposite sign.
That became a rule: any detector, secondary or blend candidate is priced in the kernel-regime harness first.

By evening two small screens had reversed at scale in one day.
The day before, that detector's $+0.031$ panel had not been reproduced by a four-movie leak-free control, which came in at $-0.0206$.
From then on I read small screens for support and parity only, and wrote every acceptance criterion against 199 movies.

---

## 3. The Possibility Audit and Stage-Jitter Registration (v92)

The same day I asked whether every possibility had really been examined.

The answer was an audit through six lenses plus an outside view: of $30$ new proposals, twenty-eight closed and the other two were one mechanism.
It claimed completeness only within the current constraints.

### 3.1 The Mechanism

Stage jitter in light-sheet imaging translates the whole cloud of detections by $3$ to $9\,\mu\mathrm{m}$ between some adjacent frames.
In the training data $11.6\%$ of transitions move by at least $3\,\mu\mathrm{m}$: $5.6\%$ in 44b6 and $14.9\%$ in 6bba.

Two deployed stages mishandle those frames.
The motion relink, which predicts velocity within far gates of $6$ and $10\,\mu\mathrm{m}$, links a $z$-neighbor $1.6$ to $3.2\,\mu\mathrm{m}$ away instead of the true successor, and the line-fit smoother then reverts the jump frame's coordinates.

The prescription estimates one global translation per transition from the detection cloud alone: the mode of a histogram of nearest-neighbor displacements ($k=3$), refined with ICP.
That shift goes into the relink's gates and predictions and into the line-fit coordinates.
There are no learned parameters, and nothing else in v90 changes.

### 3.2 Gates Written Before the Results

| gate | registered minus v90 |
|---|---|
| kernel regime, 29-movie panel | $+0.0254$ (44b6 $+0.0106$ / 6bba $+0.0265$), no severe harm |
| kernel regime, 199 movies | $+0.026578$ (44b6 $+0.023806$ / 6bba $+0.027151$); $68$ movies up by more than $0.02$, $1$ down |
| embryo-out, 199 movies | $+0.013526$ (44b6 $+0.012265$ / 6bba $+0.013666$) |
| the notebook itself | four example movies $0.889473 \to 0.961506$; runtime $1{,}417$ s against $1{,}554$ s |

This was the first candidate in the project with the same sign in the kernel regime and on unseen embryos, both embryos positive in both, and nothing fitted.
The mechanism belongs to the microscope stage, not to image brightness, so there was a reason to expect it to hold across embryos.
The gain still halved from the in-sample regime to unseen embryos, and the four example movies are the easiest regime of all.
It follows jump frequency: in the kernel regime, movies with at least $15$ jumps of $3\,\mu\mathrm{m}$ or more had a median gain of $+0.039$, jump-free movies $0.000$.

An adverse observation was disclosed before submission.
On unseen embryos the division term was $-0.0047$ (TP $36 \to 26$) while adjusted edge Jaccard rose by $+0.01823$.
In the kernel regime division TP rose by $6$.
The division sign flips between regimes, and I could not identify why.

### 3.3 Submission and Reading

I approved one submission of v92, and it went in at 19:54 on 09-13.
Its card had carried three bands ($\le 0.943$ falsified, $0.946$ a tie, $\ge 0.949$ a positive transfer).
The anomaly-only rule replaced those bands after the submission and before the score existed.

v92 came back at $0.945$ against v90's $0.946$: not an anomaly, and under the rule nothing more.
A rounded score on 29% of the test cannot separate no hidden gain, few hidden jumps, and an edge gain canceled by the division loss.

On 09-14, after that score, a resampling of Private-sized subsets ran under a card written before its result, as a description, not a gate.
Over all movies the probability of a negative difference was $0$ in both regimes.
The one negative subset was the $64$ embryo-out movies with at most two jumps of that size: $-0.003284$, negative with probability $0.901$.

---

## 4. The Association-Head Refit (v93) and a Gate That Split by Regime

The next candidate, H1, froze the detector bit-for-bit and retrained only the association head.
It was warm-started from the existing head and trained on the candidates the inference pipeline actually produces, under a new labeling rule that matches detections to annotations as the metric does, not greedily within $5\,\mu\mathrm{m}$, and adds negatives where the true successor is known.
Detections are byte-identical to v92, and v93 is v92 with this head swapped in.

The head is called forward and in reverse on each link, and deployment fuses the two calls.
The embryo-out check has two halves: K4 scores the $128$ 6bba movies with a head trained without them, and K5 adds the reverse split to cover all 199 ("reciprocal").

| gate (written before results) | measured |
|---|---|
| K4: 6bba held out | $+0.0098$, LB90 $+0.005$ |
| K5: EO, 199 movies, reciprocal | $+0.010347$ (44b6 $+0.01341$ / 6bba $+0.00983$); bootstrap 90% $[+0.0061, +0.0148]$; division TP $26 = 26$, FP $38 \to 31$ |
| attribution (descriptive, after K5) | same-budget refit with the original training rule $+0.0081$; new rule over that refit $+0.0023$ (44b6 $-0.0016$) |
| KR: deployed backbone, all-train head, 199 movies | pooled $+0.00003$ (44b6 $+0.0017$ / 6bba $-0.00025$): stop |

H1 passed all five K5 clauses, among them division TP no worse than the old head's minus three.

The kernel regime stopped it.
Adjusted edge Jaccard rose by $+0.00098$, but division Jaccard fell from $0.1879$ to $0.1784$ (FP $30 \to 34$, TP $34 \to 33$).

The attribution row suggests why the regimes split.
Of the $+0.0104$ gained on unseen embryos, $+0.0081$ came from refitting on the inference pipeline's candidates at all, even with the original training rule.
The deployed head had already been fitted on all 199 movies, so the kernel regime, which replays those movies, leaves a refit almost nothing to find.
If that is right, KR is structurally blind to that kind of gain.
The hidden test is neither regime exactly: new embryos, through an all-train backbone.

The card's written rule was to stop.
I took a disclosed exception instead, for the failed KR clause and this exact head only; the stop label stayed in the record, and every other release check stayed binding.

A second pre-release check then found a disclosure gap.
On the four example movies, v93 scored $0.92754$ against v92's $0.96151$ ($-0.034$), almost all from one movie that lost one true division and gained one false one.
They are in-sample with three division events, a weak predictor, but the approval had not seen them.
I took the approval decision again with those facts, and approved exactly v93.

v93 went in at 12:25 on 09-15.
Before its score came back, it was fixed on local evidence as the parent for new candidates.
The anomaly line was $0.942$, and my expectation, given a neutral kernel regime, was a tie with v92.

v93 came back at $0.949$: not an anomaly, so nothing followed, and my expectation of a tie did not hold.
The direction agreed with EO rather than KR, and that is all I read into it.
The selection evidence for v93 is K5.

---

## 5. Eleven Lanes After H1, Grouped by Why They Stopped

A successor head had to pass K4 and K5 against the old head and beat H1 by at least $+0.002$ pooled; other lanes carried their own bars.
From 09-17 each new lane opened only on my explicit decision, after I had seen the previous result.

None of the eleven lanes that followed H1 produced an advancing candidate.
Nine stopped at bars written before their results; D-0 was a post-result diagnostic, and GF0 advanced only to a design.
D1 and D3 (Section 7) were measurements, not lanes that could advance.

| group | lanes | why they stopped |
|---|---|---|
| other versions of the head | H2, H3, H4, CE1 | none beat H1 by $+0.002$; H3 failed its first gate |
| stopped by a probe before any training | S0, D2 | the ceiling (S0) or the signal (D2) was too small |
| worse than a placebo | A1 | on hard rows it predicted no motion |
| diagnostics | D-0, GF0 | closed with a record, or licensed one design |
| a position corrector | GO1, GO2 | GO1 stopped on its both-embryos clause; GO2 showed the panel figure was inflated by selection (Section 8) |

![Six lanes plotted as their pooled change over the H1 head on embryo-out, each with its own bar]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-01-lanes-versus-h1.png)
_Figure 1. Six of the eleven lanes fit one scale, the pooled change over the H1 head on embryo-out. Each fell short of the bar written for it; S0 is a label-oracle ceiling, and H3 is descriptive._

### 5.1 Other Versions of the Head

**H2** ran the H1 recipe for three times as many epochs.
It passed K5 at $+0.008008$ but fell $-0.002339$ below H1: longer training helped the head trained on the large embryo and hurt the one trained on the 71-movie embryo, a risk its card had named.

**H3** added a training term for the reverse call.
Its K4 was $+0.00096$ against a bar of $+0.002$, and descriptively it was $-0.0087$ below H1 in both embryos.

**H4** averaged two seeds of the H1 head.
It passed K5 at $+0.011063$ but added only $+0.000716$ over H1, and that small positive came from three extra true divisions.

**CE1** averaged H1 and H2 with no training, and came in at $-0.000223$ against H1.
The members were too correlated: H2 had continued from H1's own checkpoints with the same seed and data.
Averaging bought division TP $26 \to 29$ but lost more in edge Jaccard than the divisions gained.

### 5.2 Stopped by a Probe Before Any Training

**S0** asked how much the secondary head could ever add.
The pipeline blends a secondary head's logits into association only where the fused margin is low.
S0 priced that path with an oracle that sharpened the secondary's logits where the answer was known.
Against my bar of $+0.008$ on EO, the ceiling was $+0.002368$.
The blend is a convex mix that can change magnitudes but never which candidate wins, so a retrained secondary is bounded by the same oracle; none was trained.

**D2**, an image signal for re-ranking divisions, was also killed before any fit (Section 7).

### 5.3 Worse Than a Placebo

**A1** retrained a next-position model, earlier stopped on domain shift, at real detections instead of annotated positions.
It improved on that version by $+0.0208$ and still lost to a random-direction placebo: $-0.0486$ in the kernel regime, $-0.0662$ on unseen embryos.
In $94.35\%$ of rows the nearest detection is already the true successor, so its pre-training check had mostly measured re-detection.
On the hard rows, the model predicted no motion.

### 5.4 Closed as Diagnostics

**D-0** dissected H1's division false positives after the fact.
Its kernel-regime change (TP $-1$, FP $+4$) sat inside the spread of variants with real effects, flipped sign between regimes, and was explained by probability features in $2$ of $58$ changed events.

**GF0** substituted annotated positions for predicted ones to find what drives H1's remaining association errors.
Net top-1 rescues were $+282$ in 6bba and $+48$ in 44b6, mostly from position.
That licensed the design of one position corrector, GO1, and nothing else.

---

## 6. v94: An Exception Past Two Failed Gates

H4 was the candidate I had chosen as the last one, over the decision record's default, which was to stop and finalize.
When it failed its advance gate on 09-16, I asked whether its deployment candidate was still worth building and submitting; the decision record again said stop.

I built it as a recorded exception, and before any number existed I chose the harm branch: if the kernel-regime check failed, stop and bring the numbers back.
Submitting under harm was offered and not chosen.

The kernel-regime check failed one clause of thirteen: seven movies dropped by more than $0.02$ against H1, and two rose.
The pooled difference was $+0.00039$, carried by the division term while the edge term was negative in both embryos.

I looked at those numbers and approved one submission anyway.
Having trained it, I did not want to leave its result unseen.
That broke the first line of the rule I had written on 09-13: v94 was not a validated candidate.
Under the same rule its score could only have flagged a collapse; it could not tell me whether the second seed helped.
v94 went in at 20:09 on 09-16 and came back at $0.948$ against v93's $0.949$, the same score at the board's resolution.
It promoted nothing, and its two gate failures stand.

---

## 7. Accounting for the Division Term

On 09-17 D1 asked the division question directly: on the embryo-out graphs of the H1 head, the lineage v93 ships, how much is reachable from perfect use of the verifier's existing candidates?

| oracle | $\Delta$ | division TP / FP |
|---|---:|---|
| keep only correct picks (filter) | $+0.002868$ | FP $31 \to 1$ |
| perfect ranking of existing candidates | $+0.024351$ | TP $26 \to 89$, FP $31 \to 78$ |

Of $151$ annotated events, $89$ are reachable from existing candidates and $62$ are not.
All $63$ reachable events still missed are blocked by the $0.90$ threshold, none by the cap or conflict rules.
The ceiling is open, and a ceiling is not a candidate score.

That corrects the reading I took from the 09-13 trace.
Within its parent the right candidate usually ranks first, as the trace showed; across a movie it does not: on these graphs the blocked events' best correct rows have a median score of $0.269$ and a median rank of $296$ within their movie.
Any change to the score level, by refit or by threshold, buys false positives almost as fast as true ones.
Calibration was not the lever; ranking across the movie is.
Lowering the threshold stayed forbidden, on the v85 precedent.

D2 then tested one candidate ranking signal, an image mitosis score.
Its kill rule needed rank AUC of at least $0.75$ on both embryos and at least $9$ missed events above a top-percentile cut.
It measured $0.6922$ and $0.5752$, with $1$ of the $63$ above the cut.
It was killed before any fit, closing one signal at its current training support, not the division family.

D3 classified the $62$ unreachable events by their first failure.

| outcome of the 151 annotated divisions, H1 embryo-out graphs | count |
|---|---:|
| recovered | 26 |
| reachable, scored below $0.90$ | 63 |
| lost upstream in detection or matching | 30 |
| blocked by the $12\,\mu\mathrm{m}$ parent gate | 25 |
| structurally impossible | 7 |

![The 151 annotated divisions split into 26 recovered, 63 reachable but below threshold, 30 detection, 25 gate, 7 structural]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-02-division-151.png)
_Figure 2. Where the 151 annotated divisions stop on the embryo-out graphs of the head v93 ships. All $63$ reachable misses sit below the $0.90$ threshold. A perfect ranking of the existing candidates is a ceiling of $+0.024351$, not a candidate score._

The gate-blocked events need gates of $12.14$ to $20.75\,\mu\mathrm{m}$, and $8$ lie within $1\,\mu\mathrm{m}$.
A re-examination the same day priced that temptation: a $13\,\mu\mathrm{m}$ gate reaches all eight with a ceiling of about $+0.0044$ by arithmetic, but the verifier recovers only $26$ of $89$ reachable events today, $29\%$, so the expected gain was about $+0.001$.
Choosing a gate from the observed misses is the pattern already refused for the threshold, and the gates stayed closed.

It also concluded that v93 is a local optimum of the search conducted and, under the binding constraints (the twelve days left, one GPU machine, two embryos, $151$ annotated divisions, a 12-hour offline kernel, the $+0.004$ rule), effectively the best point still reachable.
Headroom existed; reaching it with this stack in this calendar was a different claim, and only the first was supported.

---

## 8. GO2: Selection Inflation, and a v88-Type Failure Caught Early

GO1 was the corrector GF0 had licensed: a learned model that adjusts predicted node positions without annotation at inference.
On a panel of $16$ fresh movies it gained $+0.012533$, but 44b6 was $-0.001389$, so under the panel's both-embryos rule it stopped as unresolved.
About half of the pooled figure was one division flipping from missed to recovered.

I opened GO2 to settle it on all 199 embryo-out movies.
That conflicted with the panel's own stop clause, so GO2 ran as an explicit, recorded exception.
Nothing was relaxed: K5 is stricter than the clause that had failed.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| GO2 minus H1, EO 199 | $+0.002100$ | $+0.000860$ | $+0.002293$ |
| bar | $\ge +0.004$, LB90 $> 0$ | $\ge 0$ | $\ge 0$ |

LB90 was $-0.000556$; two clauses failed, and GO2 stopped.

The split by exposure is the finding.
The $20$ movies whose GO1 output had already been seen gained $+0.008900$; the $179$ unseen movies gained $+0.001328$, a factor of $6.7$ smaller.
The unseen figure is the honest size, far below a deployable gain.

I then took a second exception: build the corrector through to a notebook anyway.
The all-train corrector was priced first in the kernel regime, on a two-movie smoke test.
On the first movie it moved all $26{,}356$ nodes to the $7\,\mu\mathrm{m}$ bound of its output.
Its input normalizer had been fitted on features from the embryo-out replay (mean $0.4111$, standard deviation $2.2032$); the kernel code's had $-0.1337$ and $1.0492$.

That is v88 again: a component positive on the embryo-out replay and broken in the kernel that ships.
On 09-10 that pattern had cost a submission; this time a two-movie smoke test caught it, and neither the 199-movie run nor the notebook was built.

---

## 9. Choosing the Final Two

The final score counts the better Private score of the two selected submissions.
Two candidates that share a failure mode are effectively one choice, so the hedge should fail differently from the primary pick.
The candidates form one lineage, v90 to v92 to v93, and each local number is measured against the parent.

| submission | local evidence | role | Public (anomaly read only) |
|---|---|---|---|
| v93 | EO $+0.010347$, LB90 $+0.0061$, K5 pass; KR $+0.00003$ (a stop, shipped under the exception); four example movies $-0.033968$ | primary | $0.949$, no anomaly |
| v92 | EO $+0.013526$, KR $+0.026578$, both embryos positive in both; four example movies $+0.072033$; negative-difference probability $0$ in both regimes; nothing fitted | hedge | $0.945$, no anomaly |
| v90 | the baseline; favored if hidden embryos rarely jump (EO jump-poor subset, v92 minus v90 $-0.003284$) | hedge alternative | $0.946$ |
| v94 | advance $+0.000716 < +0.002$; a kernel-regime clause failed | excluded | $0.948$ |

The decision record named three risk axes.
The jump axis, how often the hidden embryos jump, separates v90 from v92 and v93.
The division-flip axis, registration's division sign flip between regimes, is inherited by v93 from v92.
The head-transfer axis, whether the H1 head transfers, separates v93 from both others.

![Matrix of three risk axes against v90, v92 and v93, marking which candidate each risk would hurt]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-03-risk-axes.png)
_Figure 3. The three risk axes named in the decision record. The chosen pair shares two of them; v93 with v90 would have split all three, at the price of the registration gain in the hedge._

v93 as the primary pick rests on K5, on detections identical to v92, and on eleven later lanes that produced nothing better.
v92 as the hedge covers the head-transfer axis, v93's one specific risk: a neutral kernel regime, a loss on the four example movies, and an EO gain with checkpoint-selection exposure.
If the head does not transfer, v92 keeps the registration gain without it.

The pair has a cost, and it is on the record.
It shares the jump and division-flip axes: if the hidden embryos rarely jump and the embryo-out division loss appears there too, both picks suffer together.
v93 with v90 would hedge all three axes, at the price of the registration gain in the hedge if jumps are common.
No local measurement can say how often the hidden embryos jump, and that is what decides between v92 and v90.
The record recommended keeping the registration gain, which has the more consistent local evidence.

On 09-18 I chose v93 and v92 as the final two, the pair the decision record recommended.
The record listed v93's Public score first among its reasons for the primary pick, ahead of the embryo-out pass; I have not counted it as evidence here, but I cannot claim I did not see it.

---

## 10. The Rule We Selected By

The rule in force had three parts.
EO decided, under clauses written in steps: v92 on 09-13 needed a pooled gain above zero, both embryos non-negative in both regimes, and no excess of large per-movie drops; the H1 card of 09-14 added a $+0.004$ floor, LB90 above zero and a division-harm clause; H2's card of 09-15 added $+0.002$ over the incumbent head.
KR detected harm in the machine that ships.
The board detected collapse and nothing else.

The board was read four times; no read was a collapse, and none was counted as evidence for a model.
From v92 on, every verdict came from a gate written down before its result existed, apart from D-0, a post-result diagnostic closed with a record only.
No bar was moved after its result: the threshold stayed at $0.90$ with all $63$ reachable misses under it, the parent gate stayed closed with eight events within a micrometer, and GO2's $+0.002100$ was accepted as a stop after a $+0.012533$ panel.

What strained the rule were my own four exceptions: the KR clause for v93, the build and submission of v94, the opening of GO2 against its panel's stop, and GO2's build toward a notebook.
None rewrote a bar, and every failed clause stayed failed in the record.
Two let a submission through a stop: v93 past the harm detector, on a plausible argument that is not a measurement, and v94 past two failures, against the first line of the 09-13 rule.
The same 199 movies were also measured thirteen times, with no correction for it.

| | this period |
|---|---|
| rule in force | EO decides; KR detects harm; Public detects collapse only |
| where it was measured | 199 embryo-out movies; the kernel regime; the four example movies for parity |
| what the board was used for | four reads, v91 to v94; none a collapse |
| what held, what strained | held: every verdict local, v94 excluded, v92 kept; strained: four exceptions, v94 submitted unvalidated, thirteen measurements on one population |

---

## 11. What the Period Established

### Established

1. In the v90 division path, $46$ of $56$ low-scored divisions already ranked first within their parent; a valid-label refit gained $+0.01099$ on four movies and $-0.00679$ on 199 (false positives $37 \to 196$).
2. Label-free stage-jitter registration gains $+0.026578$ in the kernel regime and $+0.013526$ on unseen embryos, both embryos positive in both.
3. The H1 head refit gains $+0.010347$ on unseen embryos and $+0.00003$ in the kernel regime.
4. None of the eleven lanes after H1 produced an advancing candidate; nine stopped at bars written before their results, and two were diagnostics.
5. On the H1 embryo-out graphs, the $151$ annotated divisions split $26$ recovered, $63$ ranking-blocked below $0.90$, $30$ lost upstream, $25$ gate-blocked and $7$ impossible; perfect ranking of existing candidates would be worth $+0.024351$.
6. GO2 gained $+0.008900$ on the $20$ movies whose output had been seen and $+0.001328$ on the $179$ unseen, a factor of $6.7$.
7. The position corrector, as fitted, saturated every node of the kernel-regime movie it was priced on, consistent with a normalizer fitted on another feature distribution.

### Supported but Unconfirmed

1. That the head refit's regime split comes from in-sample blindness: $+0.0081$ of the $+0.0104$ embryo-out gain came from refitting at all, on movies the deployed head had already been fitted on. This rests on a descriptive attribution; v93's Public direction is recorded and not counted.
2. That the risk the two picks share lies on the jump and division-flip axes.
3. That v93 is effectively the best point reachable under the constraints, a judgment from the re-examination.

### Open Questions

1. Does v93 minus v92 on Private carry EO's sign ($+0.0103$) or KR's (about $0$)?
2. Does either selected submission collapse on Private?
3. How often do the hidden embryos jump, and does v90 end above v92?
4. Can any signal improve division ranking toward the open $+0.024351$ ceiling?
5. How much did thirteen measurements on one population inflate the lanes that survived?

---

## Closing

This was the first competition I ran on an internal criterion rather than on the board.
That is a claim about the whole run, not about every day of it.
The record shows $109$ scored submissions in July, a comparator that could not move, and an August day when I swung back to trusting the board.
The rule in force this week was built out of those failures.

Where the picks and the displayed Public ordering differed, the local evidence decided: v94 was excluded at $0.948$ because it failed its gates, and v92 was kept at $0.945$ because its local evidence was the most consistent of any candidate, positive in both regimes and both embryos with nothing fitted.

The discipline was a verdict tool, not a search strategy.
Most of the week's compute went to verdicts on small levers, and the eleven lanes after H1 produced no candidate.
After H4, the candidate I had called the last one, nine more lanes and measurements followed, from the v94 build to GO2: seven on my explicit decisions, two under a general 09-16 decision.
After GO2's deployment path failed, no lane was open and the re-examination had found no other buildable swap; I closed the project on 09-18.

Private is 71% of the test, and it is unknown as I write this.
Before the result exists, this is what it will test:

- **Whether either pick collapses.** A collapse would point at something neither local instrument measures.
- **The sign of v93 minus v92.** EO predicts about $+0.010$; KR predicts a tie, a difference within $\pm 0.002$ at the board's resolution. Positive beyond that would say the head's embryo-out gain transferred and KR was blind to it, as argued. A tie would say the in-sample reading was the better guide and the exception bought nothing. Negative would say the head did not transfer, and the hedge did its job.
- **Whether v92 scores above v90,** as both local regimes predict over all movies. If not, the likeliest reading is that the hidden embryos jump rarely and the division loss transferred: the risk both picks share.

Whichever way it goes, it is one external test of one rule; it cannot approve or reject the discipline as a whole, and I will not read it as either.

What the local evidence establishes is narrower and does not wait on Private.
Registering stage jitter is positive in both regimes and both embryos, with nothing fitted.
Refitting the association head is positive on unseen embryos and neutral in-sample, for a reason I can name but have not measured.
Nothing else built in this period beat them under the same gates.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- [Part 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
- [Part 7: Where a Local Gain Has to Be Measured]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured/)
- **Part 8: Choosing the Final Two Without the Leaderboard**
