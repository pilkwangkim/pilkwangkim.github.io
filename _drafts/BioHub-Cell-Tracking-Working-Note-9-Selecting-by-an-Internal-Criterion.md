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

> **DRAFT, pending the Private leaderboard.** This retrospective was drafted on 2026-09-22 and revised the next day, before the competition closed (2026-09-29 23:59 UTC).
> Every value marked `[PRIVATE: ...]` is unknown at the time of writing and will be filled in after the Private board is published.
> Everything outside the `[PRIVATE: ...]` markers was written before the close; any later change will be marked in place.

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

> **About this series.** These notes follow one Kaggle competition, BioHub Cell Tracking During Development:
> rebuilding cell lineages from 3D time-lapse microscopy of zebrafish embryos. The training data is 199 movies
> from two embryos; the leaderboard shows 29% of a hidden test of unseen embryos, and the final ranking uses the
> rest. The series asks one question — how to choose models by an internal criterion when the leaderboard cannot
> be trusted to choose — and each note adds what one period taught about it.
{: .prompt-info }

BioHub Cell Tracking During Development asked for cell lineages from 3D time-lapse movies of zebrafish embryos: find every nucleus in every frame, link each one to itself in the next frame, and mark where a cell divides.
The score is mostly agreement on those links, plus a small division term.

This series follows one decision through the competition: how to choose what to submit.
The leaderboard, the Public score, shows 29% of a hidden test at three decimals, and choosing by it fits a small slice after many looks, so the project chose by local out-of-fold evidence, under a criterion built one clause at a time.

Note 8 ended on 2026-09-18 with the final two fixed on local evidence: v93, which passed the embryo-out gate with a recorded exception to the harm check, and v92, kept on its local evidence although its Public score, $0.945$, was a tie with the baseline v90.
v94, with a higher Public score than v92, was left out because it had failed its gates.

This was the first competition in which I selected by an internal criterion instead of the Public board, all the way to the final two.
That is a claim about the competition as a whole: the criterion had to be built, each clause entering when a result showed what the previous version could not see.
The Private result, unknown as I write, will be its first outside test.

The short version is:

```text
Selection ended on a local rule: embryo-out decides, the kernel regime detects harm, Public detects anomalies.
The rule was built clause by clause, each added when a result exposed a blind spot.
The board's job was the transfer check; five large disagreements each pointed to a gap in the local instrument.
At the end the rule excluded a higher Public score (v94) and kept a lower one (v92).
It judged every hypothesis it was given; choosing which ones to ask was outside it.
Private is its first outside test, and only one: [PRIVATE: v93 = ?, v92 = ?].
```

| Sections | Question |
|---|---|
| 0--1 | What was the criterion, and why does each clause exist? |
| 2 | What was the board for? |
| 3--4 | What did the criterion decide at the end, and with which exceptions? |
| 5 | What did it not do? |
| 6 | How will the Private board be read? |
| 7--8 | What would I keep, add, and do first next time? |
| 9--10 | Decision log; what is established |

---

## 0. What This Note Covers

The window runs from late June to 2026-09-18, when the final picks were fixed.
It began on the board, because in the period of Note 1 nothing local scored a whole lineage graph as the metric does: reading Public scores, one structural hypothesis per submission, it went from classical baselines at $0.68$--$0.75$ to a learned lineage graph near $0.90$ (158 submissions by 2026-08-10), until the board could not separate three explanations of a plateau.

| term | meaning here |
|---|---|
| OOF | out of fold: scored on movies the model was not fitted on |
| embryo-out (EO) | the shipped code over all 199 training movies (44b6: 71, 6bba: 128), with detection and association trained on the other embryo |
| kernel regime (KR) | the kernel, the offline notebook Kaggle runs on the hidden test (12 hours, no Internet), replayed with its deployed weights over the same 199 movies, in sample (earlier: hold-in) |
| four example movies | copies of training movies, for checking the notebook's own output |
| Public / Private | the board's 29% of the hidden test / the other 71%, which sets the ranking |
| tie | a Public difference within $\pm 0.002$ |
| K5, LB90 | the final EO adoption gate (section 1); LB90 is the lower bound of an embryo-stratified 90% bootstrap interval |
| verifier | the model that decides which candidate divisions enter the graph |
| slot A, B | the two final submissions; the better Private score counts |

Local numbers from different regimes (the August comparator near $0.60$, embryo-out near $0.74$--$0.79$, the kernel regime, the example movies) are never compared, except once in section 2, where the gap is the finding.

---

## 1. The Criterion as It Finally Stood

By 09-17 the criterion had three parts.
Embryo-out decides through K5: pooled $\ge +0.004$, both embryos $\ge 0$, LB90 $> 0$, division true positives no fewer than the baseline's $-3$, and no more movies dropping by over $0.02$ than gaining by over $0.02$; a successor to an adopted component must beat it by $\ge +0.002$.
The kernel regime detects harm: it can stop a candidate, never promote one.
The board detects anomalies: a drop of $0.003$ or more against the baseline, the submission a candidate modifies, starts an investigation, and no rise counts as confirmation.
Each clause below entered when a result showed what the rule of the time could not see.

| clause | wording | since | reason |
|---|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 | the board could not separate three explanations of a plateau |
| C2 | Write each gate down before the result exists | Note 3 (07-15) | a gate chosen after the result can always be passed |
| C3 | Calibrate a rule on the population it will act on | Note 3 | a selector fired $26.7\times$ more often in deployment than in calibration |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 | a better parent classifier made a worse graph, twice |
| C5 | Compare levels only inside one reference universe; compare deltas across | Note 3 | seven baselines ($0.6007$--$0.7406$) were in use at once |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 | July's best structural result returned no score ten times |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 | movie-out folds carried embryo identity ($+0.0134$ read $+0.00013$, a guard added) |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 | a detection ceiling of $0.89\%$ was $8.79\%$ embryo-out on the same four movies |
| C9 | Use the board for matched transfer checks with a written expectation, not to choose adjacent settings; narrowed by C19 from 09-13 | Note 4 (08-10) | adjacent operating points were indistinguishable on it |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 | the comparator's whole action space could pay $+0.000365$ |
| C11 | A gate must be able to end in a decision | Note 5 | readiness gates could neither launch nor close a family |
| C12 | Measure in the deployed universe: replay the pipeline that ships | Note 6 | the selection universe sat $0.149$ below the deployed one |
| C13 | Price a hidden-only term with a designed decomposition probe | Note 6 | v80/v81 measured the hidden division term |
| C14 | Measure in the kernel regime (shipped weights and code); score the notebook's own output against its parent before submitting | Note 7 | v88's local gates had passed, two as rewritten, and it collapsed in the kernel |
| C15 | Run the leak-free control before trusting a headline | Note 7 | a $+0.031$ panel was not reproduced by a four-movie leak-free control ($-0.0206$) |
| C16 | One axis per submission, or a matched control arm | Note 7 | v85/v86 changed two things at once; v87 and v89/v90 could be read |
| C17 | Labels follow the scorer's convention; measure the labeler against ground truth first | Note 7 | labelers rejected ground-truth-style wide daughter splits ($10.0$ vs $8.7\,\mu\mathrm{m}$) |
| C18 | Small panels screen for support; adoption is decided on all 199 movies | Note 8 | three screens reversed within a week once measured more fully |
| C19 | EO decides (K5: pooled $\ge +0.004$, both embryos $\ge 0$, LB90 $> 0$); the kernel regime detects harm; Public detects anomalies only | Note 8 (09-13) | the board could not rank candidates a few thousandths apart |
| C20 | Final picks: the strongest local candidate, plus a hedge that fails in a different way | Note 8 | two picks that share a failure mode are one choice |

![Dated list of the clauses of the final selection rule, from 07-15 to 09-18]({{ site.baseurl }}/assets/img/posts/2026-09-30-biohub-working-note-9/fig-02-rule-timeline.png)
_Figure 2. When the main clauses entered the rule. The 08-28 reset, which made the board the objective again as an anti-stall measure, is the one step that pointed the other way._

That step had its reason.
August's objective was a frozen local comparator, chosen after the fold leak because it could neither leak nor drift; on 08-24 its whole action space was priced at $+0.000365$.
By 08-28 twenty-three days had passed without a transfer check the board could read, so I made the board the objective again; that day a division ceiling was measured first ($+0.0557$ locally) and a verifier was built and submitted as v79, which scored $0.935$.

---

## 2. Where the Board Earned Its Place

The board is the only window onto unseen embryos, so a local-first rule still needs it, for one job: the transfer check, one submission of a locally advanced candidate with a written expectation.
A difference within $\pm 0.002$ is a tie, which detects no failure and confirms nothing; a move of about $0.003$ or more against the expectation means looking for a gap in the local instrument; from 09-13 only a drop of that size is read.

That is why some submissions carried small local gains: if a local result is never checked once against the hidden-set score, a blind spot of the local criterion can go unseen.
Five large disagreements each pointed to a gap in the local instrument:

| when | local said | board said | gap found | what changed |
|---|---|---|---|---|
| Note 3, 07-28 | division rank ensemble $+0.000949$ | $+0.004$ | the hidden set weighs the division term far more than the replay | an observation, not a rate |
| Note 4, 08-03 | reverse-time association $+0.0073273$ | $0.913$, a tie with its control ($0.912$); the division stage $+0.004$ | the board read the division stage, not the field | C9 |
| Note 6, 09-01 | division Jaccard $0.062$ | v79 minus v81: hidden division term $\approx +0.032$, Jaccard $\approx 0.32$ | the selection universe was not the deployed one ($+0.149$) | C12, C13 |
| Note 7, 09-05 | v85 hand-label verifier $+0.0048$ | $-0.005$ | a proxy dominated by missed divisions under-priced false ones; labeler convention differed (a hypothesis) | C16, C17 |
| Note 7, 09-10 | v88 pseudo-label detector, all gates passed | $0.924$ ($-0.022$) | local pricing never used the deployed primary detector | C14 |

In each row a real local measurement met a large board move it did not predict, and the explanation was a difference between local measurement and the shipped pipeline or the hidden scorer.
None chose a model.
Where the check found nothing, the local verdict stood: v83 (local $+0.0065$) moved $+0.007$, inside a band written beforehand, and v78 ($+0.0006686$), v91 (a local null) and v94 (a local "no advance") came back as ties.
v92's card had predicted $+0.002$ to $+0.012$; it read a tie with v90, the anomaly-only rule had replaced its bands before the score, and the unmet lower bound is recorded (Note 8).

![Weekly submission counts from late June to mid-September, with each note's window and rule]({{ site.baseurl }}/assets/img/posts/2026-09-30-biohub-working-note-9/fig-01-submission-cadence.png)
_Figure 1. Submissions per week, with each note's window and the rule it records. Unscored submissions failed or timed out on the hidden set._

The board was used most while the local instrument was weakest.
In July only the board could price which points exist in the graph: seventeen association-mix submissions read $0.905$--$0.908$, while averaging two detector fields read $0.911$ and its same-day ablation $0.910$, locating the gain in the shared node field.
Early August probed the division stage, which the replay under-priced; from 08-10 came matched checks, then three nearly silent weeks, single-axis checks in September, and from 09-13 anomaly reads only.

---

## 3. What the Criterion Decided at the End

Note 8 tells these in full; Private will test them.

### 3.1 It Excluded a Higher Public Score

v94, a two-seed average of the head v93 ships, beat v93's recipe by $+0.000716$ against a required $+0.002$, and in the kernel regime seven movies dropped by more than $0.02$ while two gained, failing the harm clause.
It scored $0.948$, a tie with v93 ($0.949$) and $0.003$ above v92.
Its gate failures excluded it; a rule that picks by the board would have kept it.

### 3.2 It Kept a Public Score That Tied Its Baseline

v92 corrects stage jitter: between some frames the microscope stage shifts, the whole cloud of detected nuclei moves by $3$--$9\,\mu\mathrm{m}$, and links go to the wrong neighbors.
It estimates that shift from the detections alone, with no learned parameters, and was positive in both regimes and both embryos (kernel regime $+0.026578$, embryo-out $+0.013526$).
It scored $0.945$ against v90's $0.946$, a tie, and was kept for slot B: its local evidence was the most consistent of any candidate, and it covers v93's one specific risk, that the retrained association head (which scores which detection continues each track) does not transfer.
The pair shares two other named risks, how often the hidden embryos jump and v92's embryo-out division loss ($-0.0047$); v93 with v90 would have split all three, at the price of the registration gain.

### 3.3 It Refused Attractive Moves Without Asking the Board

On 09-17 all $63$ reachable-but-missed division events sat below the verifier's $0.90$ threshold, and eight events blocked by the parent gate (the distance within which it looks for a mother cell) lay within $1\,\mu\mathrm{m}$ of it.
Lowering the threshold was refused on the v85 precedent, where a looser operating point with new labels read positive locally and $-0.005$ on the board.
Widening the gate makes events reachable, not recovered: at the verifier's measured recall the expected gain was about $+0.001$, and a bar chosen from observed misses is what C2 prevents.

### 3.4 It Caught Failures Before They Reached a Submission

Three small panels read positive and did not hold when measured more fully: a valid-label division verifier ($+0.01099$ on four movies, $-0.00679$ on 199), a quantile-aligned pseudo-label secondary ($+0.104$ on a two-movie smoke test, $-0.033$ on a 29-movie kernel panel), and the pseudo-label detector's $+0.031$ panel ($-0.0206$ leak-free).
C18 then applied to the position corrector: $+0.012533$ on 16 movies, $+0.002100$ on 199 with LB90 $-0.000556$ ($+0.008900$ on the 20 movies already seen, $+0.001328$ on the 179 unseen).
In the corrector's deployment build, a two-movie kernel-regime trial moved all $26{,}356$ nodes of its first movie onto the $7\,\mu\mathrm{m}$ bound: its normalizer had been fitted on embryo-out features (mean $0.411$, SD $2.203$), and the kernel's differed (mean $-0.134$, SD $1.049$).
That is the v88 failure one layer down, caught before any notebook or submission existed.

---

## 4. Exceptions and Their Reasons

The last four days carried four recorded exceptions.
None changed a threshold, and every failed clause stayed failed in the record.

### 4.1 v93, Past the Kernel-Regime Stop

The kernel-regime gate on the all-train head read `stop_kr`: pooled $+0.00003$, 6bba $-0.00025$.
I took a disclosed exception for that clause and that head only.
The gate measures an in-sample refit on a backbone already fitted to all 199 movies, so it cannot see a refit gain by construction; of the $+0.0104$ embryo-out gain, $+0.0081$ came from refitting on the inference pipeline's own candidates.
The hidden set is unseen embryos, which the embryo-out gate represents ($+0.010347$, both embryos positive, LB90 $+0.0061$), and the gate read about zero, not a collapse.
The adverse reading on the four example movies, $0.92754$ against v92's $0.96151$ ($-0.034$), almost all from one movie that lost one true division and gained one false one, was set beside the exception before release, and I took the decision again with it in front of me.
The argument is structural, not a measurement; slot B covers the case where it is wrong.
The decision record listed v93's Public $0.949$ first among its slot-A reasons; I do not count it: the selection evidence is K5, and under the 09-13 rule that score reads only as "no anomaly".

### 4.2 v94, a Check of a Local "No"

H4, the two-seed head behind v94, failed its advance gate, and its deployment candidate failed one kernel-regime clause of thirteen: the local verdict was no advance.
Before any number existed I had fixed the harm branch: if the kernel-regime check failed, the build would stop and I would decide with the numbers in front of me.
It failed one clause, and I submitted the candidate once, as a check of the local verdict: a local "no" is also a prediction, and a blind spot of the criterion can hide behind a "no" as easily as behind a "yes".
The submission was a recorded exception to the first line of the 09-13 rule, one submission per validated candidate.
It read $0.948$, a tie with v93, consistent with the local verdict; its gate failures kept it out of the picks.

### 4.3 GO2 and Its Deployment Build

GO1, the position corrector, stopped as unresolved on its 16-movie panel (pooled $+0.012533$, 44b6 $-0.001389$).
I opened GO2 to settle it on all 199 embryo-out movies, against the panel's stop clause, relaxing nothing: K5 is the stricter gate.
The second exception built the corrector toward a notebook, to learn whether its deployment path held; that caught the saturation in section 3.4.

### 4.4 Lanes After the Last Candidate

H4 had been meant as the last candidate. With time left before 09-29 and a written stop on every lane, I opened nine more lanes and measurements; none advanced.

Four exceptions in four days is strain on the rule.
Each met a question no gate had been written for: a refit gain the kernel regime cannot see, a "no" worth checking once, a panel that stopped without an answer, a deployment path not yet run.

---

## 5. What the Criterion Did Not Do

A selection criterion answers one question, adopt this candidate or not, and four things sat outside it.

### 5.1 It Judged Hypotheses; It Did Not Generate Them

The learned lineage graph of Note 1, a temporal U-Net detector, a transformer edge scorer and an integer-program solver, was in place in early July as the project's strongest system.
The OOF machine measured edits to it against a fixed comparator, which is what makes a component's effect on the whole graph measurable (C1, C4, C5); an 08-05 model-family matrix wrote the dual-seed temporal U-Net down as the thing to "preserve as the common graph universe and comparator".
Every later gain was a component on that skeleton: the verifier, v92's registration, v93's head.

Alternatives were asked which part of the graph they could replace, and none was trained end to end on this data.
Of the open-source cell trackers and segmenters, Trackastra, run zero-shot on two movies for 40 frames, reached parity with our links and was set aside; Ultrack and Cellpose/StarDist were never run; a forum recipe using several open-source trackers' agreed links as dense pseudo-labels, where under 1% of links carry ground truth, was surveyed on 08-30 and not run.

Headroom was measured inside our own graph: the fused association head's embryo-out label-oracle ceiling was $+0.0458$, of which the retrained recipe captured $+0.010347$.
The 09-17 re-examination called v93 a local optimum of the search conducted and, under the binding constraints (twelve days, two embryos, the $+0.004$ rule), effectively the best point still reachable; the first half describes the search, not the problem.
A strict adoption rule makes each honest question expensive, so fewer are asked, and those sit near the current model; the criterion has no clause about which hypotheses to bring, and nothing else in the project did that job on purpose.

### 5.2 The Division Term Was Handled as a Component

Division carries weight $0.1$ in the metric, and v81 showed the hidden division term was worth about $+0.032$ to this pipeline.
The verifier was its natural owner (v79 read about $+0.014$ over the stage it replaced), and its refit produced v83.
Apart from a resurrection pass parked on 09-01 and a daughter-pair proposal model that lacked training support on 09-13, no dedicated lineage stage reached deployment.

| fate of the 151 division events (embryo-out, final lineage) | count |
|---|---:|
| recovered | 26 |
| reachable, but scored below the $0.90$ threshold | 63 |
| lost before the verifier (detection or matching) | 30 |
| blocked by the parent gate alone | 25 |
| structurally impossible | 7 |

A perfect ranking of the verifier's existing candidates would be worth $+0.024351$ (measured), a precision-aware selector about $+0.0447$ (arithmetic, no oracle run).
The one ranking signal tried, an image mitosis score, reached rank AUCs of $0.6922$ and $0.5752$ against a bar of $0.75$ and was stopped before any fit.
My inference, not a measurement, is that a verifier bound the term's ceiling to our candidate generator: $62$ of the $151$ events never became candidates.

The claim has a limit.
The local division Jaccard of $0.14$ (26 true positives, 31 false, 125 missed) is a local level; the one hidden level I measured with a designed probe was about $0.32$, for v79 (Note 6).
The division term was not the whole gap; it had measured local headroom and no stage built to take it.

### 5.3 Deployment Constraints Entered Late

The 12-hour kernel limit was known from the start, but a runtime budget model (under 43,200 seconds) was written only on 08-06.
The July machine measured graph edits on fixed out-of-fold predictions, which leaves runtime out by design, so shipping looked like packaging, until July's best structural result, pre-solver candidate activation, returned no score in ten submissions (seven by the end of July, three early on 08-01 KST).
The pattern recurred a layer at a time: float coordinates worth $+0.0050$ were dropped on 09-03 because the evaluation page requires integer voxel centroids, and v88 showed on 09-10 that the kernel's detector scores differed from every local pricing.
Each constraint entered the criterion (C6, C14) when a result exposed it.

### 5.4 Speed

The August rule was slow for a reason: after the fold leak and two retractions in one day (08-08), no number was to enter the record without its provenance.
The 08-28 review counted about sixteen new files per experiment and estimated about 15% of the planned GPU hours used productively (an estimate; no per-job ledger exists).

The September rule was fast, and spent it on small questions.
After the H1 refit, eleven lanes in a row produced no deployable candidate: four head variants, none beating it by $+0.002$; two stopped before any training by a ceiling ($+0.002368$ against $+0.008$) or a separability test; a next-position model that lost to a random-direction placebo; two diagnostics; and the position corrector twice.
All were levers on the same graph, and the last was the thirteenth measurement on the same 199 movies; no clause corrected for that repetition, and GO2's $6.7\times$ gap between seen and unseen movies shows its size.

---

## 6. The Private Reading, Written Before the Result

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
| within $\pm 0.002$ | a tie; it does not separate EO from KR | a tie, as on Public |
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

## 7. What to Keep and What to Add

### Keep

1. **A rule that can refuse before it can choose:** a "no" with a reason was the OOF machine's first useful property.
2. **Leakage controls before headlines:** $+0.0134 \to +0.00013$ on embryo-disjoint folds (confounded by a division guard), and a $+0.031$ panel against a leak-free $-0.0206$.
3. **Ceilings before optimization:** $+0.000365$ and $+0.0557$ each changed the plan the day they were measured.
4. **Measurement where the model ships, at every layer:** a fix at one layer did not fix the next.
5. **Small panels screen, full populations decide,** with seen and unseen movies reported apart.
6. **One transfer check per local advance, occasionally per local "no",** with the reading written first.

### Add

7. **A deliberate way of choosing what to judge,** cheap, separate from adoption, aimed outside the incumbent.
8. **An owner for every metric term;** the division term, I suspect, inherited one graph's ceiling.
9. **Cheap exploration, strict adoption.**
10. **Publication and shared code as decisions:** public notebooks timed on purpose, and shared code copied rather than edited (one edit broke six lanes' re-runnability, though no recorded result changed).

---

## 8. A Playbook for the Next Competition

**Week 0, days 1--3: map** (for a twelve-week competition).
Dissect the metric in code, check a local scorer against the official one where possible, and measure the runtime budget on the visible data.
Build one harness that scores any family's per-movie graph out of fold, term by term, and one that runs the exact deployed pipeline.
Measure a label-oracle ceiling for each layer: detection, association, division, post-processing.

**Weeks 1--3: breadth.**
Take four to six genuinely different families to an end-to-end graph (a learned tracker, an optimization-based tracker, a detector with an assignment solver, a pretrained tracker, any recipe the forum describes), each admitted by a 12-hour runtime pass and an out-of-fold score; an exploratory experiment takes two or three files and one written hypothesis.

**Weeks 4--6: complementarity and selection.**
Measure where the families' errors overlap and compare graph-level combinations end to end; narrow to two or three families, and only then switch on full pre-registration.
Picking the best of several families on two embryos invites a winner's curse, so the selection itself needs a held-out discipline.

**Weeks 7--10: depth.**
Seeds, all-train refits and kernel-regime pricing; track what share of each layer's ceiling has been captured.
Give each metric term its own stage: here, tracking and lineage would be separate models, not one graph and a verifier.
Open a small lever only where a measured ceiling clears a bar written down in advance.

**Weeks 11--12: freeze and choose.**
No new lanes; choose the final two so that they fail under different scenarios (v93 and v92 split only one of three named risks, section 3.2).

**Standing rules.**
Submit each local advance once as a transfer check, its reading written first.
Until week 6, a fixed share of compute goes to families that are not the incumbent.
If the local end-to-end score has not moved by $+0.004$ in two weeks, re-measure the ceilings and add a family rather than a lever.

---

## 9. Decision Log

The rule moved from the board to local evidence in steps: the board as the instrument through mid-July; a strict OOF machine from 07-15, the board checking transfer (from 08-10 only in matched experiments); a frozen comparator in August, until the 08-28 anti-stall reset; local evidence first from 09-06; Public as an anomaly detector from 09-13.
Each time the board's role narrowed because a local instrument could answer what it used to.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| Note 1: read the board, one hypothesis per submission | nothing local scored a whole graph | $0.68$--$0.75$ to near $0.90$ | the skeleton |
| Note 2: build strict OOF | the board could not separate three explanations of a plateau | a machine scoring edits on unseen movies | C1 |
| Note 3: gates first; the division ensemble as a transfer check | a gate chosen after the result can always be passed | refusals with mechanisms; $+0.000949$ locally, $+0.004$ on the board | C2--C6 |
| Note 4: separate the reverse field from the division stage | the replay under-priced the division stage | the field tied; the division stage moved; movie-out folds leaked | C7--C9 |
| Note 5: price the comparator's ceiling, then reset | a comparator whose whole action space could not produce a change the board could read | $+0.000365$; a division ceiling of $+0.0557$; v79 at $0.935$ | C10, C11 |
| Note 6: a probe for the hidden division term; replay the deployed stack | local and hidden division Jaccard differed about fivefold | $+0.032$ hidden; a $+0.149$ gap; v83 at $0.944$ | C12, C13 |
| Note 7: ship v88 with a written falsification line | its local gates had passed, two as rewritten | $0.924$ ($-0.022$), an alignment collapse | C14--C17 |
| Note 8: final two on EO; Public for anomalies only | the board could not rank candidates a few thousandths apart | v93 and v92; v94 out at $0.948$, v92 in at $0.945$ | C18--C20 |

### The Criterion at the End of the Competition

All twenty clauses of section 1 were in force at the end:

| role | clauses |
|---|---|
| where to measure | C1, C7, C8, C12, C14 |
| what to measure first | C10, C13, C15 |
| how components are judged | C3, C4, C17 |
| how verdicts are reached | C2, C5, C11, C16, C18 |
| the decision, and the board | C19, C9 |
| shipping and the final two | C6, C20 |

---

## 10. What the Period Established

### Established

1. At final selection the picks differed twice from what the two highest Public readings would have chosen: v94 ($0.948$) was excluded on its gates, and v92 ($0.945$) was kept on its local evidence.
2. Independence from the board, freedom from leaks and measurement in the deployed code were each not enough: the comparator's $+0.000365$ ceiling, the $+0.149$ universe gap, v88's $0.924$.
3. Five large disagreements between local evidence and the board each pointed to a gap in the local instrument; for v85 the labeler-convention part remains a hypothesis.
4. Three small-panel positives did not hold when measured more fully: on 199 movies, in the kernel regime, or with the leak removed.
5. The fate of all 151 annotated division events in the final pipeline is known (section 5.2).
6. Eleven consecutive lanes after the H1 refit produced no deployable candidate under gates fixed before each result; two were diagnostics (D-0, GF0).

### Supported but Unconfirmed

1. That embryo-out predicts hidden transfer of a head refit better than the in-sample kernel regime: $+0.0081$ of the $+0.010347$ came from refitting on the deployed pipeline's candidates, a gain the in-sample kernel regime has little room to find. Private will test it: `[PRIVATE: v93 - v92 = ?]`.
2. That v92's gain on hidden embryos scales with their stage-jitter prevalence, which was $5.6\%$ in one training embryo and $14.9\%$ in the other.
3. That v93 is effectively the best point reachable under the binding constraints (the 09-17 re-examination).

### Open Questions

1. `[PRIVATE: did either pick collapse, what was the sign of v93 - v92, and did v92 score above v90?]`
2. Would a portfolio of end-to-end families have found a different basin? Ceilings measured inside one graph cannot say.
3. Would a dedicated lineage stage lift the division term past the verifier's ceiling?
4. What split between cheap exploration and strict adoption keeps both honest?

---

## Closing

The last local numbers: v93, embryo-out $+0.010347$ for the head recipe, LB90 $+0.0061$, and a kernel-regime stop at $+0.00003$ crossed by a disclosed exception; v92, embryo-out $+0.013526$ and kernel regime $+0.026578$, both embryos positive in both, no bootstrap draw below zero in either.

The Private board will say how the two picks transferred, one test of the rule: `[PRIVATE: v93 = ?, v92 = ?]`.
It cannot say whether a different search would have found a better place to choose from; the rule decided among hypotheses from one family.
Even inside that family the room was not used up: 63 of 151 annotated divisions were within reach and ranked below the line, worth up to $+0.024351$ on embryo-out under a perfect ranking.
The question I carry into the next competition is not how to judge more strictly, but how to choose what to judge.

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
