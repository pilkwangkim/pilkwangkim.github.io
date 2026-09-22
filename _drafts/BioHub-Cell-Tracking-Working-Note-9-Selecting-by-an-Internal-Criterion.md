---
title: "BioHub Cell Tracking Working Note 9 (DRAFT): Selecting by an Internal Criterion, Before and After the Private Board"
date: 2026-09-30 21:00:00 +0900  # tentative; set when the post is finalized after the Private board
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, retrospective, selection-rule, oof, private-leaderboard, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-30-biohub-working-note-9/cover.png
  alt: "Title card for BioHub Working Note 9 (draft): selecting by an internal criterion"
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

# BioHub Cell Tracking Working Note 9 (DRAFT): Selecting by an Internal Criterion, Before and After the Private Board

> **DRAFT, pending the Private leaderboard.** This retrospective was drafted on 2026-09-22, before the competition closed (2026-09-29 23:59 UTC).
> Every value marked `[PRIVATE: ...]` is unknown at the time of writing and will be filled in after the Private board is published.
> Everything outside the `[PRIVATE: ...]` markers was written on 2026-09-22; any later change will be marked in place.

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
  - [Working Note 8: Choosing the Final Two Without the Leaderboard]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-Choosing-the-Final-Two-Without-the-Leaderboard/)
- Korean version: [BioHub Cell Tracking 작업 기록 9: 내부 기준으로 고르기, Private 전과 후]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-9-Selecting-by-an-Internal-Criterion-KR/) <!-- Part 9 slug and KR title are provisional: not yet in the canonical title table. -->

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 8 ended on 2026-09-18 with two designated submissions, v93 and v92.
Both picks rest on local evidence.
v93 passed the embryo-out gate and carries a recorded exception to the harm check.
The decision record listed its Public score first among its reasons for the primary pick; I did not count that as evidence, but I saw it.
v92 was kept for the second slot although its Public score was the lowest of the recent candidates, because its local evidence was the most consistent of any candidate and it covers the one risk specific to v93.
A third submission, v94, had a higher Public score than v92 and was left out because it failed its gates.

This note looks back over the whole competition around one claim.
This was the first competition in which I ran selection on an internal criterion instead of the Public board, all the way to the final two.
That is a claim about how the competition ended, not about every day of it; the rule had to be built, and most of this series is the record of it breaking.
The Private result, unknown as I write, will be its first test from outside the project.

The short version is:

```text
Selection ended on a local rule, with the board as an anomaly check.
The rule was built by failing: a leak, an objective that could not move,
the wrong universe, the wrong regime. Each failure added a clause.
At the end it excluded a higher Public score (v94) and kept one that tied its baseline (v92).
It judged every hypothesis it was given. It did not choose which ones to ask.
Private is its first outside test, and only one: [PRIVATE: v93 = ?, v92 = ?].
```

| Sections | Question |
|---|---|
| 0 | What does this note cover, and what can it not yet say? |
| 1 | What was the internal criterion at the end? |
| 2 | Which failure taught each clause? |
| 3 | What did the criterion do at the end? |
| 4 | What did it cost? |
| 5 | How is the Private board read, and what did it say? |
| 6--7 | What would I keep, change, and do first next time? |
| 8--9 | Which rule did the competition select by, and what is established? |

---

## 0. What This Note Covers, and What It Cannot Yet Say

The window runs from the first submissions in late June to 2026-09-18, when the project stopped with its final picks fixed and no experiment running.

The project did not begin on an internal criterion.
In the period of Note 1, it moved from classical baselines at $0.68$--$0.75$ to a learned lineage graph near $0.90$ by reading Public scores, one structural hypothesis per submission.
By 2026-08-10 it had made 158 submissions.
In August the Public score did not move for twenty-three days, and in the last twelve I maximized a comparator that could not move; on 08-28 I made the board the objective again.
The criterion that chose the final two was assembled out of those episodes.

![Weekly submission counts from late June to mid-September, with each note's window and rule]({{ site.baseurl }}/assets/img/posts/2026-09-30-biohub-working-note-9/fig-01-submission-cadence.png)
_Figure 1. Submissions per week, with each note's window and the rule it records. Unscored submissions failed or timed out on the hidden set._

The kernel is the offline notebook Kaggle runs on the hidden test: 12 hours, no Internet.
The training data holds 199 movies from two embryos, 44b6 (71 movies) and 6bba (128).
Embryo-out (EO) is the embryo-out replay: the shipped code over all 199 movies, with detection and association models trained on the other embryo, scored out of fold (OOF); Notes 6--7 called it the deployed-stack replay.
The kernel regime (KR) replays the deployed weights and exact notebook code over all 199 movies, in sample (what earlier notes called hold-in).
Two detectors are fused, a primary and a secondary.
The association head scores which detection continues each track.
The verifier decides which candidate cell divisions enter the graph; the parent gate is the distance within which it looks for a mother cell.
A candidate's baseline is the submission it modifies.
The final score counts the better Private score of two designated submissions, slot A and slot B.

Local numbers from different regimes (the August comparator near $0.60$, embryo-out at about $0.74$--$0.79$, the kernel regime, the four example movies) are never compared with each other, except once in section 2, where the gap is the finding.
Public differences within $\pm 0.002$ are ties: the board shows three decimals of a 29% slice of the test set.

---

## 1. The Criterion as It Finally Stood

The criterion reached the form below between 2026-09-13 and 09-17.
The board rule dates from 09-13, the adoption gate and the successor margin from 09-14 and 09-15, just before v93, and the kill-first rule from 09-17.
It was applied to every remaining candidate; one pick, v93, passed under a disclosed exception to the harm clause.

| clause | final form | first learned in |
|---|---|---|
| decision regime | EO over all 199 movies, each embryo held out in turn. Every adoption verdict is taken here. | Notes 2--4 |
| harm detector | KR. It can stop a candidate. It cannot promote one. | Notes 6--7 |
| deployment identity | The exact notebook on the four example movies, which are copies of training movies. | Notes 3, 7 |
| adoption gate | EO pooled $\ge +0.004$; both embryos $\ge 0$; embryo-stratified bootstrap lower 90% bound $> 0$; division true positives $\ge$ the baseline's $-3$; movies dropping by more than $0.02$ no more than movies gaining by more than $0.02$. | Notes 3--8 |
| successor margin | A successor to an adopted component must beat it by $\ge +0.002$ pooled. | Note 8 |
| ceiling first | Measure the label-oracle ceiling, the gain if the component were right wherever labels exist, before opening a lever. (The September secondary-head and division oracles used a $+0.008$ bar; that was a lane bar, not a standing clause.) | Notes 5, 8 |
| kill first | Before any fit, run the cheapest test of whether the signal separates the cases at all. | Note 8 |
| where to price | Detector, secondary and blend changes are priced in the kernel regime first. | Note 7 |
| the board | An anomaly detector. A drop of $0.003$ or more against the baseline starts an investigation. No positive bands. The highest Public score is never the selection rule. | Notes 7--8 |
| procedure | Hypothesis, comparison and verdict rule are written down before the result; thresholds, gates, seeds and movie lists do not change afterwards. | Notes 3, 8 |
| exceptions | Only as explicit, recorded decisions. No exception changes a threshold, and the failed verdict stays in the record. | Note 8 |

Nothing on this list is a model; it is a set of places to measure and reasons to refuse.
Most clauses exist because an earlier result looked like progress and was not; the successor margin, kill first and exceptions came from the final week's own lanes, not from an earlier failure.
And nothing on it says which hypotheses to bring to it; section 4 is mostly about that absence.

![Dated list of the clauses of the final selection rule, from 07-15 to 09-18]({{ site.baseurl }}/assets/img/posts/2026-09-30-biohub-working-note-9/fig-02-rule-timeline.png)
_Figure 2. When each clause of the final rule entered it. The 08-28 reset, which made the board the objective again, is the one step that pointed the other way._

---

## 2. Which Failure Taught Each Clause

| Note | Period | Rule in force | What broke it, and what it taught |
|---|---|---|---|
| 2 | → 07-14 | Build strict OOF instead of perturbing parameters | The board could not separate three explanations of a plateau |
| 3 | 07-15 → 07-31 | A strict OOF machine that can say no | It refused candidates calibrated on the wrong population; a better component made a worse graph |
| 4 | 08-01 → 08-16 | Local OOF decides | The largest clean local gain left no trace on the board; a leaky criterion is worse than none |
| 5 | 08-17 → 08-28 | A frozen local comparator is the objective (a Public $0.970$ outcome target on paper) | A ground-truth filter over every action it selected was worth $+0.000365$; a criterion needs a ceiling first |
| 6 | 08-29 → 09-04 | Twofold local OOF | It measured a pipeline that does not ship ($+0.149$ in level) |
| 7 | 09-05 → 09-12 | Local first; Public for large moves only | Positive on the embryo-out replay (shipped code, embryo-out weights) was not positive in the kernel (v88); a local gain counts only where the model ships |
| 8 | 09-13 → 09-18 | EO decides, KR detects harm, Public detects anomalies | The rule carried eleven non-advances and the final picks, with four recorded exceptions |

**Note 3.** The machine was built on gates written down on 07-15, and mostly it refused.
A leakage-audited selector had been calibrated where it fired on $0.204\%$ of groups and was deployed where it fired on $5.456\%$; a parent chooser with better top-1 accuracy made a worse graph ($-0.0007833$).
It could refuse but not yet choose: its four approvals were each about $10^{-4}$.

**Note 4.** Reverse-time association gave the largest clean local association gain on record at that point, $+0.0073273$, with both embryos positive.
Shipped alone, it scored $0.913$: a tie with its forward control ($0.912$), and three thousandths below an incumbent ($0.916$) that carried a division stage it lacked.
The expected six thousandths did not appear, and falsifying a predicted move that large was the board's proper job.
The fortnight's other headlines came from folds that held out movies but not embryos.
A multi-frame association model read $+0.0134$ on them and $+0.00013$ when rerun on embryo-disjoint folds on 08-12, but the rerun also added a division guard, so the pair cannot size the leak.

**Note 5.** I obeyed a frozen 199-movie comparator as the objective, and almost nothing moved.
On 08-24, inside the stall and four days before the 08-28 review, a ground-truth filter over every action the comparator selected, $3{,}178$ of them, was worth $+0.000365$.
A criterion is worth optimizing only when its ceiling exceeds the gap it is meant to close.
The 08-28 reset swung back to the board, an anti-stall rule rather than a selection rule.

**Note 6.** A decomposition probe, v81, withheld the division stage's edits and scored $0.903$ against v79's $0.935$.
That $+0.032$ is the whole hidden division term, because v81 made no division edits at all; it is not the verifier's gain over the stage it replaced, about $+0.014$.
Then the deployed-stack replay, the shipped code over the same 199 movies, read $0.7499$ where the comparator replay behind the August selections read $0.6005$: those selections had been made in a universe the notebook does not run.

**Note 7.** On 09-05 v85 turned a local $+0.0048$ into a Public $-0.005$, and on 09-06 I ruled that reading v86's $-0.001$ as a verdict, and conditioning v87 on "$\ge 0.945$", had been p-hacking.
Then v88, whose new secondary detector was trained on pseudo-labels and positive on the embryo-out replay (shipped code, embryo-out weights), scored $0.924$, $-0.022$ against its baseline.
The notebook rescales the secondary's scores to the primary's before fusing them; the deployed primary's background sits near $-15$ and the new detector's near $-5$, so the rescaling pushed its scores about ten units down.
The kernel-regime check came out of that.
A leakage control then failed to reproduce the new detector's $+0.031$ headline: with its pseudo-labels made by a teacher trained only on the student's own training embryo, a four-movie leak-free control scored $-0.0206$ standalone.
It was not a rerun of the fifteen-movie panel behind $+0.031$, which stays unconfirmed.

### Where the Board Still Earned Its Place

v81 measured a quantity no local instrument could see.
v83 moved $+0.007$, inside a reading band written before it was submitted.
v85's $-0.005$ falsified a local gain, and v88's $0.924$ was a collapse the board detected.
None of these chose a model.

---

## 3. What the Criterion Did at the End

Note 8 tells these decisions in full; this section keeps what the argument needs.
Whether they were right is the question section 5 leaves to Private.

### 3.1 It Excluded a Higher Public Score

v94, a two-seed average of the head v93 ships, beat v93's recipe by $+0.000716$ against a required $+0.002$, and in the kernel regime seven movies dropped by more than $0.02$ while two gained, which failed the harm clause.
I submitted it anyway, as a recorded exception, and it scored $0.948$: a tie with v93 ($0.949$) at the board's resolution, and $0.003$ above v92.
It was excluded from both slots on its gate failures; a board-first rule would have kept it (Note 8, section 6).

### 3.2 It Kept a Public Score That Tied Its Baseline

v92 estimates the whole-cloud shift of light-sheet stage jitter ($3$--$9\,\mu\mathrm{m}$ between some frames) from the detections alone, with no learned parameters, and was positive in both regimes and both embryos: kernel regime $+0.026578$, embryo-out $+0.013526$.
It scored $0.945$ against v90's $0.946$, and the decision record kept it for slot B because its local evidence was the most consistent of any candidate and it covers v93's one specific risk, that the retrained association head does not transfer (Note 8, sections 3 and 9).
The record named two further risks, how often the hidden embryos jump and v92's embryo-out division loss ($-0.0047$), and v92 shares both with v93; v93 with v90 would have split all three, at the price of the registration gain in the hedge.

### 3.3 It Refused Attractive Moves Without Asking the Board

On 09-17 all $63$ reachable-but-missed division events sat below the verifier's $0.90$ threshold, and eight events blocked by the parent gate lay within $1\,\mu\mathrm{m}$ of it.
Lowering the threshold was refused on the v85 precedent, and widening the gate was refused because it makes events reachable, not recovered: at the verifier's measured recall its expected gain was about $+0.001$ (Note 8, section 7).

### 3.4 It Caught Failures Before They Cost a Submission

Three small panels read positive and did not hold at full measurement: a quantile-aligned pseudo-label secondary ($-0.033$ on a 29-movie kernel panel), a valid-label division verifier ($+0.01099$ on four movies, $-0.00679$ on 199), and a position corrector ($+0.012533$ on 16 movies, $+0.002100$ on 199 with a lower 90% bound of $-0.000556$; $+0.008900$ on the 20 movies already seen, $+0.001328$ on the 179 unseen).
When I built the corrector's deployment path anyway, a two-movie kernel-regime trial moved all $26{,}356$ nodes of its first movie onto the $7\,\mu\mathrm{m}$ bound: its normalizer had been fitted on the embryo-out replay's features (mean $0.411$, SD $2.203$), and the kernel's differed (mean $-0.134$, SD $1.049$).
That is the v88 failure again, one layer further down, caught before any notebook or submission (Note 8, sections 2 and 8).

---

## 4. What It Cost

### 4.1 Speed

The August version of the rule was expensive: the 08-28 review counted about sixteen new files per experiment, the Public score did not move for twenty-three days, and the same review estimated that about 15% of the planned GPU hours were used productively (no per-job ledger exists to check).

The September version was faster but spent its time on small questions.
After the H1 refit (the head recipe v93 ships), eleven lanes in a row produced no deployable candidate; the first verdict came about an hour before v93 was submitted, the other ten after it.
Four varied the head v93 ships (longer training, an extra training term, a two-seed average, an untrained average of two heads), and none beat it by $+0.002$.
Two were stopped before any training by a ceiling or separability test: a secondary-head ceiling of $+0.002368$ against a bar of $+0.008$, and the mitosis score of section 4.3.
A next-position model was a real mechanism too small to use: it lost to a random-direction placebo where it was needed.
Two were diagnostics closed with a record, one advancing only to a learner design.
The last two were the position corrector of section 3.4.
All eleven were levers on the same graph, and the last was the thirteenth measurement on the same 199 movies; no clause corrected for that repetition.

### 4.2 It Judged Hypotheses; It Did Not Choose Them

This is the main cost, and the main lesson.

The learned lineage graph of Note 1, a temporal U-Net detector, a transformer edge scorer and an integer-program solver, was in place in early July.
Everything after it was a component added to that skeleton, or a measurement of it: the division verifier, the registration of v92, the association head of v93.
An 08-05 model-family matrix wrote the dual-seed temporal U-Net down as the thing to "preserve as the common graph universe and comparator", and gave a pretrained tracker the role of "independent conditional parent evidence on about 5-10% of ambiguous groups".
Most alternatives were asked which part of our graph they could replace; the few whole-graph comparisons were a classical baseline and a zero-shot pretrained tracker, and none was trained end to end on our data.

Trackastra was run zero-shot on two movies for 40 frames, reached parity with our links, and was set aside.
Ultrack and Cellpose/StarDist were never run.
A multi-tracker consensus recipe from the competition forum proposed using several open-source trackers' agreed links as dense pseudo-labels, where under 1% of links carry ground truth; I surveyed it on 08-30 and never ran it.

The headroom was measured, but only inside our own graph.
On embryo-out, the label-oracle ceiling of the fused association head was $+0.0458$.
The retrained head recipe, measured with fold heads on embryo-out, captured $+0.010347$; the all-train head that shipped was measured only in sample, at $+0.00003$.
The 09-17 re-examination concluded: "v93 is a local minimum of the search we conducted, and — under the binding constraints — effectively the global minimum of the space we can still reach."
The record says minimum where, for a maximized score, Note 8 says optimum: the best point still reachable.
The second half is about what twelve days, two embryos and the $+0.004$ rule left reachable.
The first half is the one I carry forward: it describes the search, not the problem.

A strict rule makes this worse: every question becomes expensive to ask honestly, so fewer are asked, and those asked sit nearest the current model.

### 4.3 The Division Term Was Treated as a Component

Division carries weight $0.1$ in the metric, and v81 showed the hidden division term was worth about $+0.032$ to this pipeline.
Apart from a resurrection pass parked on 09-01 and a daughter-pair proposal model that lacked training support on 09-13, division was handled by a verifier on our own graph, and no dedicated lineage stage reached deployment.

By 09-17 every one of the 151 annotated division events had a traced fate (Note 8, section 7):

| fate of the 151 division events (embryo-out, final lineage) | count |
|---|---:|
| recovered | 26 |
| reachable, but scored below the $0.90$ threshold | 63 |
| lost before the verifier (detection or matching) | 30 |
| blocked by the parent gate alone | 25 |
| structurally impossible | 7 |

A perfect ranking of the verifier's existing candidates would be worth $+0.024351$ (measured), a precision-aware selector about $+0.0447$ (arithmetic, no oracle run).
The one ranking signal tried, an image mitosis score, reached rank AUCs of $0.6922$ and $0.5752$ against a bar of $0.75$ in both embryos and was stopped before any fit.
My inference, not a measurement, is that handling division as a verifier kept its ceiling bound to our candidate generator.

The claim has a limit.
The local embryo-out division Jaccard of $0.14$ (26 true positives, 31 false, 125 missed) is not comparable with Public-derived figures from other teams.
Our own Public-derived estimates of the hidden division Jaccard were about $0.32$ for v79 and $0.43$ for v87; by the same arithmetic, raising $0.43$ to $0.5$ is worth about $+0.007$.
So the division term was not the whole gap; it had measured local headroom and no stage built to take it.

### 4.4 Exceptions, and Days the Rule Did Not Hold

The final picks carry one tension I do not want to smooth over.
v93 itself crossed a stop: its all-train head read $+0.00003$ in the kernel regime, negative in one embryo, and shipped under a disclosed exception for that clause and that head only, with the four example movies at $-0.034$ against v92 in front of me (Note 8, section 4).
The exception rests on an argument, not a measurement: most of the embryo-out gain came from refitting on the deployed pipeline's candidates at all, which an in-sample replay of a head already fitted on all 199 movies has little room to find.
The decision record listed v93's Public $0.949$ among its reasons for the primary pick, first in its one-line summary; its direction agreed with EO, one observation of agreement and not evidence for the head swap.

The other three exceptions were v94's build and submission after its gates failed, the corrector's full measurement against its panel's written stop, and the corrector's deployment build.
No exception changed a threshold, and every one is in the record, but four in four days is a rule under strain.
Nor did I stop after the candidate I had called the last one, the two-seed average: nine more lanes and measurements followed, most on an explicit decision of mine, two under a general 09-16 decision to continue prediction research.

### 4.5 Deployment Constraints Arrived Late

The 12-hour kernel limit was known from the start, but a runtime budget model (a fixed cost plus a per-movie cost under 43,200 seconds) was written only on 08-06.
Before it, July's best structural result, pre-solver candidate activation, went into ten submissions that returned no score on the hidden set: seven by the end of July and three in the first hours of 08-01 (KST).

---

## 5. The Private Reading, Written Before the Result

The split is 29% Public and 71% Private, on embryos not in the training data; that the Public part may be roughly one embryo is an inference.
Before the result existed, Note 8 wrote down three things Private would test; their opening lines are repeated here in its words:

1. **Whether either pick collapses.**
2. **The sign of v93 minus v92.** EO predicts about $+0.010$; KR predicts a tie, a difference within $\pm 0.002$ at the board's resolution.
3. **Whether v92 scores above v90,** as both local regimes predict over all movies. If not, the likeliest reading is that the hidden embryos jump rarely and the division loss transferred: the risk both picks share.

Whatever the result, it is one outside test; it neither approves nor rejects the discipline.

v93 is v92 with the association head replaced, and v92 is v90 with registration added, so each pair is a baseline and its modification.
The standing board rule defines one reading: a drop of $0.003$ or more against the baseline starts an investigation, and no rise is read as confirmation.
The table below was written on 2026-09-22 to describe each outcome.
Only its last row comes from the standing rule; the other rows are not bands and confirm nothing.

| Private difference | v93 minus v92 | v92 minus v90 |
|---|---|---|
| $+0.003$ or higher | the sign EO predicted | the sign both local regimes measured |
| within $\pm 0.002$ | a tie; the reading KR predicted | a tie, as on Public |
| $-0.003$ or lower | the anomaly clause: investigate a head-transfer failure, the scenario slot B covers | the anomaly clause: investigate whether the hidden embryos rarely jump and the division loss transferred, the risk both picks share |

The results, to be filled in:

| item | expectation written before the result | Private | reading |
|---|---|---:|---|
| v93 (slot A) | no collapse | `[PRIVATE: v93 = ?]` | `[PRIVATE: ?]` |
| v92 (slot B) | no collapse | `[PRIVATE: v92 = ?]` | `[PRIVATE: ?]` |
| v93 minus v92 | EO positive ($+0.0103$); KR about zero ($+0.00003$) | `[PRIVATE: ?]` | `[PRIVATE: ?]` |
| v92 minus v90 | positive in both regimes; negative only on the 64 embryo-out movies with at most two jumps ($-0.003284$) | `[PRIVATE: v90 = ?; v92 - v90 = ?]` | `[PRIVATE: ?]` |

Reported, not read as tests of the rule:

| item | note | Private |
|---|---|---:|
| v94 | failed its gates; read as indistinguishable from v93 on held-out embryos | `[PRIVATE: v94 = ?]` |
| standing | final Private rank, reported only | `[PRIVATE: rank = ?]` |

`[PRIVATE: consolidated reading, to be written after 2026-09-29 23:59 UTC. It answers the three items above in order, and says what the result does and does not show about the rule.]`

---

## 6. Lessons

1. **A rule has to be able to refuse before it can choose.** The OOF machine's first useful property was that it said no, with a reason.
2. **A leaky criterion is worse than none,** because it selects with confidence. Leakage controls go first. Twice a headline did not survive a stricter re-measurement ($+0.0134 \to +0.00013$ on embryo-disjoint folds, confounded by a division guard; the $+0.031$ panel was not reproduced by a four-movie leak-free control at $-0.0206$).
3. **Measure the ceiling before optimizing anything.** Twelve days inside a $+0.000365$ ceiling was a ceiling measured too late, not a lack of ideas.
4. **Measure where the model ships, at every layer.** The same mismatch appeared in the graph universe (Note 6), the detector scores (v88) and a feature normalizer (the position corrector). Fixing one layer did not fix the next.
5. **Re-measure small-panel effects on the full population,** and report seen and unseen movies separately. Three small positives reversed or shrank at scale in one week.
6. **A discipline judges; it does not search.** The stricter the judging, the more the project needs a separate, deliberate way of choosing what to judge.
7. **Give every metric term an owner.** Here the division term was handled mainly as a component of one graph, and I suspect it inherited that graph's ceiling.
8. **Keep exploration cheap and adoption strict.** The same heavy procedure for a first look and a final decision makes first looks rare.
9. **Treat publication and shared code as decisions.** I published notebooks during the competition; next time their timing will be chosen. A script edited for one lane broke six others' re-runnability, though no recorded result changed.

---

## 7. A Playbook for the Next Competition

**Week 0, days 1--3: map** (for a twelve-week competition).
Dissect the metric in code, and check a local scorer against the official one where possible.
Write the runtime budget and measure it on the visible data.
Build one harness that scores any family's per-movie graph out of fold, term by term, and one that runs the exact deployed pipeline.
Measure a label-oracle ceiling for each layer: detection, association, division, post-processing.

**Weeks 1--3: breadth.**
Take four to six genuinely different families to an end-to-end graph: a learned tracker, an optimization-based tracker, a detector with an assignment solver, a pretrained tracker, and any recipe the forum describes, run once.
Each family's entry ticket is a 12-hour runtime pass and an out-of-fold score.
An exploratory experiment costs two or three files and one written hypothesis.

**Weeks 4--6: complementarity and selection.**
Measure where the families' errors overlap, and compare graph-level combinations end to end.
Narrow to two or three families, and only then switch on the full pre-registration procedure.
Picking the best of several families on two embryos invites a winner's curse, so the selection itself needs a held-out discipline.

**Weeks 7--10: depth.**
Seeds, all-train refits, and kernel-regime pricing.
Track what share of each layer's ceiling has been captured.
Give each metric term its own stage: here, tracking and lineage would be separate models, not one graph and a verifier.
Open a small lever only in a layer whose measured ceiling clears a bar written down in advance.

**Weeks 11--12: freeze and choose.**
No new lanes.
Choose the final two so that they fail under different scenarios; v93 and v92 split only one of three named risks (section 3.2).

**Standing rules.**
Until week 6, a fixed share of compute goes to families that are not the incumbent.
If the local end-to-end score has not moved by $+0.004$ in two weeks, re-measure the ceilings and add a family rather than a lever.
Never edit code that another experiment's record depends on; copy it.

---

## 8. The Rule We Selected By

Section 2 traces how the rule reached its final form, with one swing back to the board on 08-28; this is where it stood at the end.

That rule was in force at the final selection, and where it and the board pointed different ways, its evidence decided: v94 was left out despite its Public score, and v92 was kept despite its own.
It did not decide alone: v93 carries a recorded exception, and the decision record listed its Public score among the reasons for the primary pick.

| | this competition |
|---|---|
| rule in force at the end | EO decides ($+0.004$ pooled, both embryos $\ge 0$, lower 90% bound $> 0$; $+0.002$ for a successor); KR detects harm; the board detects anomalies |
| where it was measured | embryo-out, in the deployed pipeline, priced in the kernel regime first |
| what the board was used for | one decomposition probe, falsifying large moves, one collapse detection; at final selection, an anomaly check both picks passed, and one line of the decision record's slot-A reasons, which I did not count as evidence |
| what broke it | a leak, an unreachable objective, the wrong universe, the wrong regime |
| what strained it | four recorded exceptions in the last four days, and thirteen measurements on one population |

---

## 9. What the Period Established

### Established

1. At final selection the rule decided against the board's order in two places: v94 ($0.948$) was excluded on its gates, and v92 ($0.945$) was kept on its local evidence.
2. Being independent of the board is not enough. A ground-truth filter over every action the frozen comparator selected was worth $+0.000365$.
3. Being leak-free is not enough. The comparator replay and the deployed-stack replay differed by $+0.149$ on the same 199 movies.
4. Being measured in the deployed code is not enough: v88, positive on the embryo-out replay, collapsed in the kernel ($0.924$), and a corrector saturated all $26{,}356$ nodes of a kernel-regime trial movie.
5. Three small-panel positives did not survive re-measurement on 29 or 199 movies (section 3.4).
6. The fate of every annotated division event in the final pipeline is known: 26 recovered, 63 ranked below the line, 30 lost before the verifier, 25 blocked by the gate, 7 impossible.
7. Eleven consecutive lanes after the H1 refit, the first reporting about an hour before v93's submission and the other ten after it, produced no deployable candidate under gates fixed before each result, except D-0, a post-result diagnostic; one diagnostic advanced only to a learner design.

### Supported but Unconfirmed

1. That embryo-out predicts hidden transfer of a head refit better than the in-sample kernel regime. The support is structural: $+0.0081$ of the $+0.010347$ came from refitting on the deployed pipeline's candidates, and the replaced head had already been fitted on the movies the kernel regime replays. Private will test it: `[PRIVATE: v93 - v92 = ?]`.
2. That v92's gain on hidden embryos scales with their stage-jitter prevalence, which was $5.6\%$ in one training embryo and $14.9\%$ in the other.
3. That v93 is effectively the best point reachable under the binding constraints, a judgment from the 09-17 re-examination.

### Open Questions

1. `[PRIVATE: did either pick collapse, what was the sign of v93 - v92, and did v92 score above v90?]`
2. Would a portfolio of end-to-end families have found a different basin? It was never run, and the measured ceilings, all inside our own graph, say nothing about another family.
3. Would a dedicated lineage stage lift the division term past the verifier's ceiling? The one division-specific proposal model lacked training support.
4. Does a strict adoption rule cost more in exploration than it saves in false adoptions, and what split between cheap exploration and strict adoption keeps both honest?

---

## Closing

The last local numbers of the project are these.
v93: embryo-out $+0.010347$ for the head recipe, lower 90% bound $+0.0061$, and a kernel-regime stop at $+0.00003$ for the all-train head, crossed by a disclosed exception.
v92: embryo-out $+0.013526$ and kernel regime $+0.026578$, both embryos positive in both, no bootstrap draw below zero in either.
After the head recipe, eleven lanes without a deployable candidate.

The Private board will say how the two picks transferred; it is one test of the rule: `[PRIVATE: v93 = ?, v92 = ?]`.
It cannot say whether a different search would have found a better place to choose from.
The rule decided among the hypotheses it was given, and I gave it hypotheses from one family.
The question I carry into the next competition is not how to judge more strictly, but how to choose what to judge.

Even inside that one family, the local evidence says the room was not used up: 63 of 151 annotated divisions were within reach and ranked below the line, worth up to $+0.024351$ on embryo-out under a perfect ranking, and the one signal tried against them did not separate them.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- [Part 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
- [Part 7: Where a Local Gain Has to Be Measured]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured/)
- [Part 8: Choosing the Final Two Without the Leaderboard]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-Choosing-the-Final-Two-Without-the-Leaderboard/)
- **Part 9: Selecting by an Internal Criterion, Before and After the Private Board**
