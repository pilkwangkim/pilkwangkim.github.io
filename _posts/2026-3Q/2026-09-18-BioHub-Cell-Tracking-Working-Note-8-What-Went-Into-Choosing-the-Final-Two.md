---
title: "BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two"
date: 2026-09-18 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, final-selection, stage-jitter, registration, association-head, embryo-out, selection-bias, oof, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-18-biohub-working-note-8/cover.png
  alt: "Title card for BioHub Working Note 8: what went into choosing the final two"
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

# BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
  - [Working Note 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
  - [Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
  - [Working Note 7: Deciding by Logic, and What Validation Must Reproduce]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- Korean version: [BioHub Cell Tracking 작업 기록 8: 최종 제출을 고를 때 고민한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two-KR/)

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

Note 7 ended on 2026-09-12 with four conditions local validation must reproduce: the code that ships, the weights the kernel actually loads, labels in the scorer's convention, and no leak.
By the end of that week each had an instrument.
The question for 09-13 to 09-18, the last six days of work, was which two submissions to designate as final: each is scored on Private, the 71% of the hidden test the leaderboard never shows, and the better one counts.

On 09-13 I narrowed Public to one question, whether a submission collapsed, and moved every acceptance decision onto all 199 movies after three small screens reversed at scale.
A review of every open idea found stage-jitter registration (v92); a refit association head (v93) passed its embryo-out gate and shipped past a kernel-regime stop as a disclosed exception.
Eleven further lanes produced nothing better, and the final picks, v93 and v92, depart from the Public order in two places.

The short version is:

```text
Public answered one question near the deadline: did a submission collapse? None did.
Evidence was read on all 199 movies: EO decided, the kernel regime watched for harm.
v92 corrects stage jitter: positive in both regimes and both embryos, nothing fitted.
v93 refits the association head: +0.010 on unseen embryos, neutral in-sample; a disclosed exception.
Eleven further questions found nothing better; v94's check of a local "no" found no blind spot.
The pair: v93, and v92 as a hedge that fails differently; v94 out on its failed gates.
```

| Sections | Question |
|---|---|
| 0 | What does the pipeline do, and which instrument answers which question? |
| 1 | What could Public still answer near the deadline? |
| 2 | On how many movies does evidence decide? |
| 3--4 | What logic did registration (v92) and the head refit (v93) test, and why did v93 ship past a stop? |
| 5 | What did eleven further questions narrow? |
| 6 | Can a local "no" be checked? (v94) |
| 7--8 | Was there room left in the divisions and in node positions? |
| 9 | Which two, on which failure modes, and what will Private test? |
| 10--11 | The decision log, and what the period established |

---

## 0. The Pipeline, and Which Instrument Answers Which Question

Two fused detectors find the nuclei in every frame; a transformer **association head** scores candidate links from each nucleus to the next frame's nuclei, called forward and in reverse with the two calls fused; an ILP picks a consistent set of links; deterministic stages, among them a motion relink and a line-fit smoother, repair the graph; and a fitted **division verifier** adds a fork wherever its score clears $0.90$, at most $50$ per movie.
The score is adjusted edge Jaccard plus $0.1$ times division Jaccard.
The deployed baseline was v90: v87 with Note 7's two validity repairs, $0.946$ on the board.

Three instruments decided things, and their numbers are never compared with one another.

| instrument | what it runs | role |
|---|---|---|
| EO (embryo-out) | detection and association from models that never trained on the scored embryo, downstream stages all-train; 199 movies (44b6: 71, 6bba: 128) | efficacy; the deciding gate |
| KR (kernel regime) | the deployed weights and exact notebook code over the same 199 movies | harm detection in the machine that ships; in-sample |
| the four example movies | the notebook's actual output, scored locally | deployment parity only |

EO is the **deployed-stack** replay of Notes 6–7 with embryo-out weights; its backbone checkpoints were selected on each fold's evaluation embryo.
Every lane had a card, with its question, gate and stop rule committed before any result.
EO gates report both embryos and a bootstrap over movies; **LB90**, the lower bound of its 90% interval, above zero means the gain does not rest on a few lucky movies.

---

## 1. What Public Could Still Answer Near the Deadline

Since 09-06 a Public difference within $\pm 0.002$ had been a tie, and the board was cited only for moves of about $0.003$ or more.
v91, v90 with the secondary detector's features averaged over test-time views, went in on 09-13 as a sanity check of a local null, a Public submission that checks whether a local decision holds on unseen embryos.
It came back at $0.946$, a tie with v90, consistent with that null.
That evening I narrowed the board's role again, in a short written rule:

```text
One submission per validated candidate.
Read its Public score for one thing: a drop of 0.003 or more below the deployment parent.
A drop starts a verification, not a decision.
No card carries a band of the form "a transfer if the score is at least X".
Deployment decisions come from the pre-registered local gates: EO and KR.
I designate the final two, and the highest Public score is never the selection rule.
```

The board's old questions now had local instruments; it could not rank candidates a few thousandths apart; and a band like "a transfer if at least X" turns a lucky draw into evidence, so over enough submissions the luckiest candidate wins.
What it still did well was catch a collapse like v88's (C19).

---

## 2. What Counts as Evidence: All 199 Movies, Not Small Screens

Note 7 had closed the division ranking lanes without knowing where the missed divisions sat; on 09-13 I traced the v90 division path over all 199 movies.
Of $151$ annotated divisions, $23$ were recovered, $72$ had no correct candidate, and $56$ had a correct candidate scored below the $0.90$ threshold.
In $46$ of the $56$ the best correct candidate already ranked first among its own parent's candidates.
I read that as a calibration error, most plausibly from training labels that did not match the current graph's candidates; if so, a refit on valid labels should lift more true divisions over the threshold.

The test kept the threshold, because the local replay could not price a lower one: v85 had lowered it along with new labels and came back $-0.005$ on the board, most plausibly through false divisions the replay barely counts.
On four movies chosen for their events the refit gained $+0.01099$ from one extra true division; with one event carrying that number, I judged it on all 199.

| | 4 movies | 199 movies |
|---|---:|---:|
| official $\Delta$ | $+0.01099$ | $-0.00679$ |
| division TP/FP/FN | $+1$ / $+2$ / $-1$ | 23/37/128 $\to$ 22/196/129 |

Nodes and recall were unchanged; the higher score level bought false divisions, not true ones, and the calibration hypothesis failed.

That afternoon a second small screen reversed: the quantile alignment Note 7 had left unresolved, meant to redeploy the pseudo-label detector, gave $+0.104$ and $-0.0016$ on a two-movie smoke test, then $-0.033027$ on a 29-movie kernel-regime panel.
Its motivating signal, a 6bba tail median of $+0.066$ on the embryo-out replay, had the opposite sign in the kernel; C14 caught it before any notebook was built.

With that detector's $+0.031$ panel, which a four-movie leak-free control had turned into $-0.0206$ the day before, three small favorable screens had reversed at full scale within a week, through different mechanisms: one lucky event, an unrepresentative smoke test, a leak.
From then on small screens counted for support and parity only, and every acceptance criterion was written against all 199 movies (C18).

---

## 3. The First Candidate: Registering Stage Jitter (v92)

### 3.1 Had Every Possibility Been Examined?

The same day I asked whether every possibility had been examined, and reviewed every open idea.
Of $30$ new proposals, twenty-eight closed, and the other two were one mechanism; the review claimed completeness only within the current constraints.

### 3.2 The Mechanism

Between some frames the microscope's stage shifts and the whole cloud of cells moves a few micrometres at once: **stage jitter**, here $3$ to $9\,\mu\mathrm{m}$ when it happens, with $11.6\%$ of training transitions moving by at least $3\,\mu\mathrm{m}$.

Two deployed stages mishandle those frames.
The motion relink links a $z$-neighbor instead of the true successor, and the line-fit smoother then reverts the jump frame's coordinates.
The fix estimates one global translation per transition from the detections alone and feeds it into the relink and the line-fit.
Nothing is learned, no labels are used, and nothing else in v90 changes.
Because the error belongs to the stage, the mechanism predicts more than a pooled gain: the gain should follow how often a movie jumps, on any embryo.

### 3.3 Gates Written Before the Results

| gate | registered minus v90 |
|---|---|
| G3: kernel regime, 29-movie panel | $+0.0254$, no severe harm |
| G4: kernel regime, 199 movies | $+0.026578$ (44b6 $+0.023806$ / 6bba $+0.027151$) |
| G5: embryo-out, 199 movies | $+0.013526$ (44b6 $+0.012265$ / 6bba $+0.013666$) |
| G6: the notebook itself | four example movies $0.889473 \to 0.961506$ |

It was the project's first candidate positive in both regimes and both embryos, with nothing fitted.
The gain also followed jump frequency, as the mechanism predicts: in the kernel regime, movies with at least $15$ jumps of $3\,\mu\mathrm{m}$ or more had a median gain of $+0.039$, jump-free movies $0.000$.
Against it: the gain halved from in-sample to unseen embryos, the four example movies are the easiest regime of all, and on unseen embryos the division term was $-0.0047$ (TP $36 \to 26$) while in the kernel regime division TP rose by $6$, a sign flip I could not explain and disclosed before submission.

### 3.4 The Sanity Check

v92, backed by the strongest local evidence the project had produced, went in on 09-13 as a sanity check.
Its card predicted $+0.002$ to $+0.012$ on the board; the anomaly-only rule replaced that band after the submission and before the score existed.

It came back at $0.945$ against v90's $0.946$: no anomaly.
The card's lower bound of $+0.002$ was not met; a rounded score on 29% of the test cannot separate no hidden gain, few hidden jumps, and an edge gain canceled by the division loss.
On 09-14 a descriptive resampling of movie subsets the size of the Private share put the probability of a negative difference at $0$ in both regimes over all movies.
The one negative subset was the $64$ embryo-out movies with at most two jumps of $3\,\mu\mathrm{m}$ or more: $-0.003284$.
Where the stage rarely jumps, registration has nothing to correct; Section 9 returns to that risk.

---

## 4. The Second Candidate: Refitting the Association Head (v93)

H1 asked whether an association head trained on the candidates the inference pipeline actually produces would link better than the deployed one.
It kept the detector bit-for-bit and retrained only the association head, warm-started, with detections matched to annotations as the metric matches them and negatives where the true successor is known.
v93 is v92 with this head swapped in.

The embryo-out check has two halves: K4 scores the $128$ 6bba movies with a head trained without them, and **K5** covers all 199 by adding the reverse split ("reciprocal").
K5's clauses: a pooled gain of at least $+0.004$, both embryos non-negative, LB90 above zero, division TP no worse than the old head's minus three, and no excess of per-movie drops.

| gate (written before results) | measured |
|---|---|
| K5: EO, 199 movies, reciprocal | $+0.010347$ (44b6 $+0.01341$ / 6bba $+0.00983$); division TP $26 = 26$, FP $38 \to 31$ |
| attribution (descriptive, after K5) | same-budget refit with the original training rule $+0.0081$; new rule over that refit $+0.0023$ |
| KR: deployed backbone, all-train head, 199 movies | pooled $+0.00003$ (44b6 $+0.0017$ / 6bba $-0.00025$): stop |

H1 passed all five K5 clauses and stopped in the kernel regime, where division Jaccard fell from $0.1879$ to $0.1784$; the card's verdict, `stop_kr`, went into the record.

### 4.1 Why the Regimes Split, and the Exception

The attribution row suggests why the regimes split: of the $+0.0104$ gained on unseen embryos, $+0.0081$ came from refitting on the inference pipeline's candidates at all, even under the original rule.
The hidden test is neither regime exactly: new embryos, through an all-train backbone.

I granted a disclosed exception for the failed KR clause, for this exact head only, for four reasons:

1. The KR gate measures an in-sample refit on a backbone already fit to all 199 movies; by construction it cannot see a refit gain.
2. The hidden set is unseen embryos, which EO represents: $+0.01035$, both embryos positive, LB90 $+0.0061$.
3. KR read about zero, not the v88-type collapse it exists to catch.
4. Every other release check stayed binding, and the stop label stayed.

Before release v93 scored $0.92754$ on the four example movies against v92's $0.96151$ ($-0.034$), almost all from one movie that lost one true division and gained one false one.
Those movies are in-sample with three division events, a weak predictor; the exception had been granted without them, so I took it again with them in view and kept it for exactly v93.

### 4.2 The Reading

v93 went in on 09-15 and, before its score came back, became the parent for new candidates on its local evidence.
It came back at $0.949$ against an anomaly line of $0.942$: no anomaly, and not the tie with v92 I had expected from the neutral kernel regime.
The direction agreed with EO, but one rounded reading on 29% of the test attributes nothing to the head.

---

## 5. Eleven Further Questions After H1

Eleven lanes followed H1, each asking where room might remain.
A successor head had to beat H1 by at least $+0.002$ pooled, and from 09-17 each new lane opened only after the previous result was read.
None produced an advancing candidate: nine stopped at bars written before their results, D-0 was a post-result diagnostic, and GF0 advanced only to a design.

| lane | question | written stop | answer |
|---|---|---|---|
| H2 | Does three times the training help? | beat H1 by $+0.002$ | $-0.002339$ against H1 |
| S0 | How much could a better secondary head add? | ceiling $\ge +0.008$ | $+0.002368$ |
| D-0 | Where do H1's extra false divisions come from? | none (diagnostic) | inside the noise |
| H3 | Does a training term for the reverse call help? | K4 $\ge +0.002$ | $+0.00096$ |
| H4 | Does averaging two seeds help? | beat H1 by $+0.002$ | $+0.000716$ |
| GF0 | What drives H1's remaining link errors? | rescues outnumber harms in both embryos | mostly position |
| GO1 | Does a learned position corrector help? | 16-movie panel, both embryos $\ge 0$ | $+0.012533$, 44b6 $-0.001389$ |
| A1 | Does a next-position model beat a placebo? | model minus placebo $\ge 0$ | $-0.0486$ KR, $-0.0662$ EO |
| CE1 | Does averaging H1 and H2 help? | beat H1 by $+0.002$ | $-0.000223$ |
| D2 | Can a mitosis image score rank missed divisions? | rank AUC $\ge 0.75$, both embryos | $0.6922$ / $0.5752$ |
| GO2 | Does GO1 hold on all 199 movies? | K5 | $+0.002100$, LB90 $-0.000556$ |

![Six lanes plotted as their pooled change over the H1 head on embryo-out, each with its own bar]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-01-lanes-versus-h1.png)
_Figure 1. Six of the eleven lanes fit one scale, the pooled change over the H1 head on embryo-out. Each fell short of the bar written for it; S0 is a label-oracle ceiling, and H3 is descriptive._

### 5.1 What the Answers Narrowed

**The head recipe was spent.**
Longer training (H2) helped the head trained on the large embryo and hurt the one trained on the 71-movie embryo.
CE1's two heads were too correlated to help.
H4 was opened after H3 closed, as one more lane while time remained before the 09-29 deadline.

**The secondary path had no room.**
The secondary head's logits are blended in as a convex mix that can change magnitudes but never which candidate wins, so S0's oracle bounds any retrained secondary.

**Learned motion did not help.**
In $94.35\%$ of rows the nearest detection is already the true successor, so A1's pre-training check had mostly measured re-detection; on the hard rows the model predicted no motion.

**A diagnostic pointed elsewhere.**
GF0 substituted annotated positions for predicted ones: net top-1 rescues of $+282$ in 6bba and $+48$ in 44b6, mostly from position, licensed one position corrector (Section 8).

---

## 6. Checking a Local "No" (v94)

H4 failed its advance gate on 09-16: $+0.000716$ against $+0.002$.
A local "no" is also a prediction, and a blind spot of the local criterion can hide behind it as well as behind a "yes"; H4's small gain came from divisions, where the board and the local replay had disagreed before.

I built its deployment candidate past the advance gate as a recorded exception, with a harm branch fixed before any number existed: a failed kernel-regime check would stop the build for a decision.
The check failed one clause of thirteen (seven movies dropped by more than $0.02$ against H1, two rose), and its pooled $+0.00039$ came from the division term.
I submitted it once, as a diagnostic: with its gates failed it could promote nothing, but a move of $0.003$ or more either way would have sent me looking for what the local criterion had missed.

v94 came back at $0.948$ against v93's $0.949$, a tie, consistent with the local verdict; a tie does not show equivalence, but this check found no blind spot.
With two gate failures, v94 stayed out of the final picks.

---

## 7. Was There Room Left in the Divisions?

On 09-17 D1 asked how many of the missed divisions perfect use of the verifier's existing candidates could reach, on the embryo-out graphs of the H1 head.

| oracle | $\Delta$ | division TP / FP |
|---|---:|---|
| perfect ranking of existing candidates | $+0.024351$ | TP $26 \to 89$, FP $31 \to 78$ |

Of $151$ annotated events, $89$ are reachable and $62$ are not.
All $63$ reachable events still missed are blocked by the $0.90$ threshold, none by the cap or conflict rules.
The ceiling is open.

D1 also corrected the 09-13 reading: the right candidate usually ranks first within its parent but not across the movie, where the blocked events' best correct rows have a median rank of $296$.
Any change to the score level, by refit or by threshold, buys false positives almost as fast as true ones, as the 09-13 refit did; the lever is ranking across the movie, and the threshold stayed at $0.90$ on the v85 precedent (Section 2).

D2's image mitosis score failed its rank-AUC bar before any fit (Section 5), which closes one ranking signal at its current training support, not the division family.

D3 classified the $62$ unreachable events by first failure.

![The 151 annotated divisions split into 26 recovered, 63 reachable but below threshold, 30 detection, 25 gate, 7 structural]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-02-division-151.png)
_Figure 2. Where the 151 annotated divisions stop on the embryo-out graphs of the head v93 ships. All $63$ reachable misses sit below the $0.90$ threshold. A perfect ranking of the existing candidates is a ceiling of $+0.024351$, not a candidate score._

The gate-blocked events need parent gates of $12.14$ to $20.75\,\mu\mathrm{m}$, and $8$ lie within $1\,\mu\mathrm{m}$.
A $13\,\mu\mathrm{m}$ gate reaches all eight with a ceiling of about $+0.0044$ by arithmetic, but the verifier recovers only $26$ of $89$ reachable events today, so the expected gain was about $+0.001$; and choosing a gate from observed misses is the pattern already refused for the threshold.
The gates stayed closed.

A re-examination of the whole search the same day concluded that v93 is a local optimum of the search conducted and, under the binding constraints, effectively the best point still reachable; one buildable swap remained, the position corrector.

---

## 8. The Last Buildable Swap: the Position Corrector on All 199 Movies

GO1 adjusts predicted node positions at inference without annotation.
On its 16-movie panel it gained $+0.012533$, but 44b6 was negative, so under the panel's both-embryos rule it stopped as unresolved, and about half of the pooled figure was one division flipping from missed to recovered.

The panel's stop clause did not allow a 199-movie confirmation, so GO2 ran one on all 199 embryo-out movies as a recorded exception, under the stricter K5, with its verdict to end the lane.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| GO2 minus H1, EO 199 | $+0.002100$ | $+0.000860$ | $+0.002293$ |

Two clauses failed (LB90 $-0.000556$), and I accepted the stop.

The $20$ movies whose GO1 output had already been seen gained $+0.008900$; the $179$ unseen movies gained $+0.001328$, a factor of $6.7$ smaller.
The lane existed because of what earlier looks had shown, and those movies carried that selection; the unseen figure is the honest size.
The same inflation applies to every lane: by GO2 the same 199 movies had been measured thirteen times, uncorrected.

Under a second recorded exception I priced the all-train corrector in the kernel regime, to see whether it could run in the machine that ships.
On the first movie of a two-movie smoke test it moved all $26{,}356$ nodes to the $7\,\mu\mathrm{m}$ bound of its output: its input normalizer had been fitted on embryo-out features, unlike the kernel's.
It repeated v88's pattern, positive on the embryo-out replay and broken in the kernel, caught this time before a notebook existed.
With no lane open and no other buildable swap, I closed the project on 09-18.

---

## 9. Choosing a Pair That Fails in Different Ways

The better Private score of the two picks counts, so the pair should be the candidate with the strongest local evidence plus one that fails for a different reason.

### 9.1 Each Candidate's Evidence

Each local number is measured against its parent in the lineage v90, v92, v93.

| submission | local evidence | role | Public (anomaly read only) |
|---|---|---|---|
| v93 | EO $+0.010347$, LB90 $+0.0061$, K5 pass; KR $+0.00003$ (a stop, shipped under the exception); four example movies $-0.033968$ | primary | $0.949$, no anomaly |
| v92 | EO $+0.013526$, KR $+0.026578$, both embryos positive in both; four example movies $+0.072033$; negative-difference probability $0$ in both regimes; nothing fitted | hedge | $0.945$, no anomaly |
| v90 | the baseline; favored if hidden embryos rarely jump (EO jump-poor subset, v92 minus v90 $-0.003284$) | hedge alternative | $0.946$ |
| v94 | advance $+0.000716 < +0.002$; a kernel-regime clause failed | excluded | $0.948$ |

### 9.2 The Risk Axes

The decision record named three risk axes: how often the hidden embryos jump, which separates v90 from v92 and v93; registration's division sign flip between regimes, which v93 inherits from v92; and whether the H1 head transfers, which separates v93 from both.

![Matrix of three risk axes against v90, v92 and v93, marking which candidate each risk would hurt]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-03-risk-axes.png)
_Figure 3. The three risk axes named in the decision record. The chosen pair shares two of them; v93 with v90 would have split all three, at the price of the registration gain in the hedge._

v93, the primary pick, rests on K5, on detections identical to v92, and on eleven later lanes that produced nothing better.
v92, the hedge, covers v93's one specific risk, head transfer (a neutral kernel regime, a loss on the four example movies, checkpoint-selection exposure in EO), and keeps the registration gain without the head.

The pair shares the jump and division-flip axes: if the hidden embryos rarely jump and the embryo-out division loss appears there too, both picks suffer together.
No local measurement can say how often the hidden embryos jump, so I kept the registration gain and on 09-18 designated v93 and v92.

### 9.3 Two Departures From the Public Order

The two highest Public readings were v93 and v94; the picks depart from that order twice:

- **v94 is excluded at $0.948$**, a tie with v93. It failed its advance gate and a kernel-regime clause, so it was not a candidate.
- **v92 is kept at $0.945$**, a tie with v90 and $0.004$ below v93. Its local evidence is the most consistent of any candidate, and it fails in a different way from v93.

The decision record lists v93's $0.949$ among the reasons for slot A, the primary pick; the evidence for v93 is K5.

### 9.4 What Private Will Test, Written Before It

Private is unknown as I write this. It will test:

- **Whether either pick collapses.** A collapse would point at something neither local instrument measures.
- **The sign of v93 minus v92.** EO predicts about $+0.010$; KR predicts a tie. Positive beyond $\pm 0.002$ would be consistent with the head's embryo-out gain transferring and KR being blind to it. A tie would not separate the two readings and would give the exception no support. Negative would say the head did not transfer, and the hedge did its job.
- **Whether v92 scores above v90,** as both local regimes predict over all movies. If not, the likeliest reading is that the hidden embryos jump rarely and the division loss transferred: the risk both picks share.

---

## 10. Decision Log

From 09-13 EO decided through clauses written before each result, KR detected harm in the machine that ships, and the board, read four times from v91 to v94, flagged no anomaly.
Four decisions went past a written stop (v93, v94, GO2 and GO2's deployment path), each recorded with its reason, and no bar moved after its result.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| 09-13: the board becomes an anomaly detector | local instruments answer its old questions; $\pm 0.002$ readings add nothing | four reads, no anomaly | C19 |
| 09-13: the division refit judged on 199 movies | its four-movie gain was one event | $-0.00679$; FP $37 \to 196$ | C18 |
| 09-13: v92 submitted as a sanity check | the strongest local evidence so far | $0.945$, no anomaly | v92 stands on local evidence |
| 09-15: v93 under a disclosed exception to `stop_kr` | KR cannot see a refit gain; EO $+0.01035$, LB90 $+0.0061$; KR not a collapse | $0.949$, no anomaly | v93 the parent, on K5 |
| 09-15 to 09-18: eleven lanes | test whether any further idea held up against H1 under the same gates | none advanced | v93 a local optimum |
| 09-16: v94 submitted as a diagnostic | a local "no" is a prediction too | $0.948$, a tie with v93 | v94 excluded |
| 09-17: GO2 opened under an exception | the 16-movie panel could not decide | $+0.002100$; unseen $+0.001328$ | stop accepted |
| 09-18: final picks v93 and v92 | strongest local candidate, plus a hedge that fails differently | Private unknown | C20 |

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
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings; narrowed by C19 from 09-13 | Note 4 (08-10) |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 | A gate must be able to end in a decision | Note 5 |
| C12 | Validate on the pipeline that ships: replay it exactly | Note 6 |
| C13 | Price a hidden-only term with a designed decomposition probe | Note 6 |
| C14 | Measure in the kernel regime (shipped weights and code); score the notebook's own output against its parent before submitting | Note 7 |
| C15 | Run the leak-free control before trusting a headline | Note 7 |
| C16 | One axis per submission, or a matched control arm | Note 7 |
| C17 | Labels follow the scorer's convention; measure the labeler against ground truth first | Note 7 |
| C18 **(new)** | Small panels screen for support; adoption is decided on all 199 movies | Note 8 |
| C19 **(new)** | EO decides (K5: pooled $\ge +0.004$, both embryos $\ge 0$, LB90 $> 0$); the kernel regime detects harm; Public detects anomalies only | Note 8 (09-13) |
| C20 **(new)** | Final picks: the strongest local candidate, plus a hedge that fails in a different way | Note 8 |

---

## 11. What the Period Established

### Established

1. $46$ of $56$ low-scored v90 divisions already ranked first within their parent; a valid-label refit gained $+0.01099$ on four movies and $-0.00679$ on 199.
2. Label-free stage-jitter registration gains $+0.026578$ in the kernel regime and $+0.013526$ on unseen embryos, both embryos positive in both.
3. The H1 head refit gains $+0.010347$ on unseen embryos and $+0.00003$ in the kernel regime.
4. None of the eleven lanes after H1 produced an advancing candidate.
5. On the H1 embryo-out graphs the $151$ annotated divisions split $26$ recovered, $63$ below $0.90$, $30$ lost upstream, $25$ gate-blocked and $7$ impossible; perfect ranking of existing candidates is worth $+0.024351$.
6. GO2 gained $+0.008900$ on the $20$ movies whose output had been seen and $+0.001328$ on the $179$ unseen, a factor of $6.7$.
7. The position corrector saturated every node of the kernel-regime movie it was priced on, consistent with a normalizer fitted on other features.

### Supported but Unconfirmed

1. That the head refit's regime split comes from in-sample blindness, on a descriptive attribution; v93's Public direction is recorded, not counted.
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

By 09-18 the board's role had shrunk from $109$ scored submissions in July to four anomaly reads, and the final two were chosen on local evidence: stage-jitter registration, positive in both regimes and both embryos with nothing fitted, and a head refit positive on unseen embryos and neutral in-sample.
Nothing else built in this period beat them under the same gates.

Local validation cannot say whether these gains hold on the embryos Private scores: how often they jump, and whether the head's embryo-out gain carries over.
The next and last note will read the Private result against the predictions in Section 9.4, written before it existed.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- [Part 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
- [Part 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- [Part 7: Deciding by Logic, and What Validation Must Reproduce]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- **Part 8: What Went Into Choosing the Final Two**
