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
  - [Working Note 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
  - [Working Note 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
  - [Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
  - [Working Note 7: Deciding by Logic, and What Validation Must Reproduce]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
  - [Working Note 8: What Went Into Choosing the Final Two]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/)
- Korean version: [BioHub Cell Tracking 작업 기록 9: 내부 기준으로 고르기, Private 전과 후]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-9-Selecting-by-an-Internal-Criterion-KR/) <!-- Part 9 slug and KR title are provisional: not yet in the canonical title table. -->

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

Note 8 ended on 2026-09-18 with the final two fixed on local evidence: v93, past one recorded exception to the harm check, and v92, whose Public score tied its baseline; v94 was left out on its gates despite a higher Public score than v92's.

This was the first competition I ran, as a whole, on an internal criterion instead of the Public board; this note asks what that criterion did and did not do.
Each time a sanity check (one Public submission testing whether a local decision holds on unseen embryos) disagreed with local validation, the gap was a condition of the submission that validation had not reproduced, and a clause closed it.
Note 5 found a limit of another kind: the search the criterion judged had grown one pipeline, one component at a time, into a local optimum.
Private, unknown as I write, is its first outside test.

The short version is:

```text
Local validation was made trustworthy clause by clause, each gap exposed by a sanity check.
At the end the rule excluded a higher Public score (v94) and kept a lower one (v92).
The search it judged grew one pipeline step by step into a local optimum; next time, breadth first.
Private is its first outside test: [PRIVATE: v93 = ?, v92 = ?].
```

| Sections | Question |
|---|---|
| 0--1 | What was the criterion, and why does each clause exist? |
| 2 | How did sanity checks make local validation trustworthy? |
| 3--4 | What did the criterion decide at the end, and with which exceptions? |
| 5 | What did the criterion not do, and where did the search fall short? |
| 6 | How will the Private board be read? |
| 7--8 | What would I keep, add, and do first next time? |
| 9--10 | Decision log; what is established |

---

## 0. What This Note Covers

The task: find every nucleus in every frame, link each to itself in the next frame, and mark where a cell divides; the score is mostly agreement on those links, plus a small division term.

The window runs from late June to 2026-09-18 and began on the board, since nothing local yet scored a whole lineage graph as the metric does: one structural hypothesis per submission took classical baselines at $0.68$--$0.75$ to a learned lineage graph near $0.90$ (158 submissions by 2026-08-10), until the board could not separate three explanations of a plateau.

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

Levels from different regimes are never compared (C5), except once in section 2, where the gap is the finding.

---

## 1. The Criterion as It Finally Stood

By 09-17 the criterion had three parts.
Embryo-out decides through K5: pooled $\ge +0.004$, both embryos $\ge 0$, LB90 $> 0$, division true positives no fewer than the baseline's $-3$, and no more movies dropping by over $0.02$ than gaining by over $0.02$; a successor to an adopted component must beat it by $\ge +0.002$.
The kernel regime can stop a candidate for harm, never promote one.
On the board, a drop of $0.003$ or more against the baseline (the submission a candidate modifies) starts an investigation, and no rise counts as confirmation.

| clause | wording | since | reason |
|---|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 | the board could not separate three explanations of a plateau |
| C2 | Write each gate down before the result exists | Note 3 (07-15) | a gate chosen after the result can always be passed |
| C3 | Calibrate a rule on the population it will act on | Note 3 | a selector fired $26.7\times$ more often in deployment than in calibration |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 | a better parent classifier made a worse graph, twice |
| C5 | Compare levels only inside one reference replay; compare deltas across | Note 3 | seven baselines ($0.6007$--$0.7406$) were in use at once |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 | July's best structural result returned no score ten times |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 | movie-out folds carried embryo identity ($+0.0134$ read $+0.00013$, a guard added) |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 | a detection ceiling of $0.89\%$ was $8.79\%$ embryo-out on the same four movies |
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings; narrowed by C19 from 09-13 | Note 4 (08-10) | adjacent operating points were indistinguishable on it |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 | the comparator's whole action space could pay $+0.000365$ |
| C11 | A gate must be able to end in a decision | Note 5 | readiness gates could neither launch nor close a family |
| C12 | Validate on the pipeline that ships: replay it exactly | Note 6 | local validation had replayed a different pipeline, $0.149$ lower |
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

August's objective was a frozen local comparator, chosen after the fold leak because it could neither leak nor drift, and its whole action space was worth $+0.000365$ (section 5.1).
On 08-28, after twenty-three days without a readable sanity check, I made the board the objective again against another stall, still testing candidates locally first; v79, a division verifier built that day on a measured ceiling ($+0.0557$ locally), scored $0.935$.

---

## 2. What the Board Was For: Finding the Gaps in Local Validation

A sanity check submitted a locally advanced candidate once, small gains included, with a written expectation.
A difference within $\pm 0.002$ is a tie and confirms nothing; a move of about $0.003$ or more against the expectation means looking for a gap in local validation (from 09-13, only a drop of that size is read).
Five large disagreements each traced to a condition of the submission that local validation had not reproduced:

| when | local said | board said | gap found | what changed |
|---|---|---|---|---|
| Note 3, 07-28 | division rank ensemble $+0.000949$ | $+0.004$ | the hidden set weighs the division term far more than the replay | an observation, not a rate |
| Note 4, 08-03 | reverse-time association $+0.0073273$ | $0.913$, a tie with its control ($0.912$); the division stage $+0.004$ | the board read the division stage, not the field | C9 |
| Note 6, 09-01 | division Jaccard $0.062$ | v79 minus v81: hidden division term $\approx +0.032$, Jaccard $\approx 0.32$ | local validation had replayed a different pipeline ($+0.149$) | C12, C13 |
| Note 7, 09-05 | v85 hand-label verifier $+0.0048$ | $-0.005$ | a proxy dominated by missed divisions under-priced false ones; labeler convention differed (a hypothesis) | C16, C17 |
| Note 7, 09-10 | v88 pseudo-label detector, all gates passed | $0.924$ ($-0.022$) | validation had never run the weights the kernel loads | C14 |

None of the five chose a model.
Where the check found nothing, the local verdict stood: v83 (local $+0.0065$) moved $+0.007$, inside a band written beforehand; v78 ($+0.0006686$), v91 (a local null) and v94 (a local "no advance") read as ties.
v92 tied v90 against its card's $+0.002$ to $+0.012$, after the anomaly-only rule had replaced those bands; the unmet lower bound is recorded (Note 8).

![Weekly submission counts from late June to mid-September, with each note's window and rule]({{ site.baseurl }}/assets/img/posts/2026-09-30-biohub-working-note-9/fig-01-submission-cadence.png)
_Figure 1. Submissions per week, with each note's window and the rule it records. Unscored submissions failed or timed out on the hidden set._

---

## 3. What the Criterion Decided at the End

### 3.1 It Excluded a Higher Public Score

v94, a two-seed average of the head v93 ships, beat v93's recipe by $+0.000716$ against a required $+0.002$, and in the kernel regime seven movies dropped by more than $0.02$ while two gained, failing the harm clause.
It scored $0.948$, a tie with v93 ($0.949$) and $0.003$ above v92.
Its gate failures excluded it; a rule that picks by the board would have kept it.

### 3.2 It Kept a Public Score That Tied Its Baseline

v92 corrects stage jitter: between some frames the microscope stage shifts, the whole cloud of detected nuclei moves by $3$--$9\,\mu\mathrm{m}$, and links go to the wrong neighbors.
It estimates the shift from the detections alone, with no learned parameters, and was positive in both regimes and both embryos (kernel regime $+0.026578$, embryo-out $+0.013526$).
It scored $0.945$ against v90's $0.946$, a tie, and took slot B on the most consistent local evidence of any candidate, covering v93's one specific risk: that the retrained association head (which scores which detection continues each track) does not transfer.
The pair shares two other named risks, how often the hidden embryos jump and v92's embryo-out division loss ($-0.0047$); v93 with v90 would have split all three, at the price of the registration gain.

### 3.3 It Refused Attractive Moves Without Asking the Board

On 09-17 all $63$ reachable-but-missed division events sat below the verifier's $0.90$ threshold, and eight events blocked by the parent gate (the distance within which it looks for a mother cell) lay within $1\,\mu\mathrm{m}$ of it.
Lowering the threshold was refused on the v85 precedent, a looser operating point with new labels that read positive locally and $-0.005$ on the board.
Widening the gate makes events reachable, not recovered: at the verifier's measured recall it was worth about $+0.001$, and a bar chosen from observed misses is what C2 prevents.

### 3.4 It Caught Failures Before They Reached a Submission

Three small panels read positive and failed at full measure: a valid-label division verifier ($+0.01099$ on four movies, $-0.00679$ on 199), a quantile-aligned pseudo-label secondary ($+0.104$ on a two-movie smoke test, $-0.033$ on a 29-movie kernel panel), and the pseudo-label detector's $+0.031$ panel ($-0.0206$ leak-free).
C18 then applied to the position corrector: $+0.012533$ on 16 movies, $+0.002100$ on 199 with LB90 $-0.000556$ ($+0.008900$ on the 20 movies already seen, $+0.001328$ on the 179 unseen).
In the corrector's deployment build, a two-movie kernel-regime trial moved all $26{,}356$ nodes of its first movie onto the $7\,\mu\mathrm{m}$ bound: its normalizer had been fitted on embryo-out features (mean $0.411$, SD $2.203$), unlike the kernel's (mean $-0.134$, SD $1.049$).
It was the v88 failure one layer down, caught before any notebook existed.

---

## 4. Exceptions and Their Reasons

The last four days carried four recorded exceptions, each for a question no gate had been written for.
None changed a threshold, and every failed clause stayed failed in the record.

### 4.1 v93, Past the Kernel-Regime Stop

The kernel-regime gate on the all-train head read `stop_kr` (pooled $+0.00003$, 6bba $-0.00025$), and I took a disclosed exception for that clause and that head only.
The gate replays an in-sample refit on a backbone already fitted to all 199 movies, so it cannot see a refit gain, and $+0.0081$ of the $+0.0104$ embryo-out gain came from refitting on the inference pipeline's own candidates.
Embryo-out, which represents the hidden set's unseen embryos, read $+0.010347$ with both embryos positive and LB90 $+0.0061$.
Before release I retook the decision beside an adverse reading: the four example movies read $0.92754$ against v92's $0.96151$ ($-0.034$), almost all from one movie that lost one true division and gained one false one.
The argument is structural, not measured; slot B covers the case where it is wrong.
v93's Public $0.949$, listed first among its slot-A reasons in the decision record, reads only as "no anomaly" under the 09-13 rule.

### 4.2 v94, a Check of a Local "No"

H4, the two-seed head behind v94, failed its advance gate and one kernel-regime clause of thirteen (section 3.1).
A rule written beforehand stopped the build there for a decision with the numbers in view; I submitted the candidate once, a recorded exception to the 09-13 rule's first line (one submission per validated candidate), since a blind spot can hide behind a local "no" as easily as behind a "yes".
It read as a tie, consistent with the local verdict.

### 4.3 GO2 and Its Deployment Build

GO1, the position corrector, stopped unresolved on its 16-movie panel (44b6 $-0.001389$).
I opened GO2 to settle it on all 199 embryo-out movies, against the panel's stop clause but under K5, the stricter gate; a second exception built its deployment path, which caught the saturation in section 3.4.

### 4.4 Lanes After the Last Candidate

H4 had been meant as the last candidate; with time left, I opened nine more lanes and measurements, each with a written stop, and none advanced.

---

## 5. What the Criterion Did Not Do

### 5.1 It Judged What the Search Proposed, and the Search Ended in a Local Optimum

The learned lineage graph of Note 1 (a temporal U-Net detector, a transformer edge scorer and an integer-program solver) was the strongest system by early July, and the search grew it one component at a time.
That was a choice: after seven baselines in use at once, a shared graph made every number comparable (C1, C4, C5), and an 08-05 model-family matrix named the dual-seed temporal U-Net the thing to "preserve as the common graph universe and comparator".

The cost, as Note 5 found it: each accepted step changes the graph every later step is measured on, so early choices such as which detector defines the nodes are never reopened, and an alternative priced as a part reads the same whether it is worse or would win end to end.
On 08-24, with the labels choosing every allowed edit, the frozen comparator's whole action space was worth $+0.000365$; the room lay outside, in which nodes exist and how divisions are emitted.

Every later gain was a component on that skeleton (the verifier, v92's registration, v93's head), and no alternative was trained end to end: Trackastra, zero-shot on two movies for 40 frames, matched our links and was set aside; Ultrack and Cellpose/StarDist were never run; a forum recipe using several open-source trackers' agreed links as dense pseudo-labels was surveyed on 08-30, not run.
Headroom was measured only inside our graph: the fused association head's embryo-out label-oracle ceiling was $+0.0458$, and the retrained recipe captured $+0.010347$ of it.
The 09-17 re-examination called v93 a local optimum of the search conducted.

No clause could have caught this: applied strictly to a narrow stream of candidates, the criterion certifies a local optimum with care.
For this problem the incremental method was the wrong one.

### 5.2 The Division Term Was Handled as a Component

Division carries weight $0.1$ in the metric, and v81 showed the hidden division term was worth about $+0.032$ to this pipeline.
The verifier, a component on the same graph, owned it (v79 read about $+0.014$ over the stage it replaced; its refit produced v83).
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

The local division Jaccard of $0.14$ (26 true positives, 31 false, 125 missed) is a local level; the one hidden level I measured with a designed probe was about $0.32$, for v79 (Note 6).
The division term was not the whole gap: it had measured local headroom and no stage built to take it.

### 5.3 Deployment Constraints Entered Late

The 12-hour kernel limit was known from the start, but a runtime budget model (under 43,200 seconds) was written only on 08-06.
The July machine measured graph edits on fixed out-of-fold predictions, which leaves runtime out, so shipping looked like packaging until July's best structural result, pre-solver candidate activation, returned no score in ten submissions.
The pattern recurred a layer at a time: float coordinates worth $+0.0050$ were dropped on 09-03 because the evaluation page requires integer voxel centroids, and on 09-10 v88 showed that the kernel's detector scores differed from every local pricing.
Each constraint entered the criterion (C6, C14) only when a result exposed it.

### 5.4 Speed

The August rule was slow: after the fold leak and two retractions in one day (08-08), no number entered the record without its provenance.
The 08-28 review counted about sixteen new files per experiment and estimated about 15% of planned GPU hours used productively (an estimate, with no per-job ledger); a search that can afford few questions rarely leaves its path (Note 5).

The September rule was fast and spent its speed on small questions: after the H1 refit, eleven lanes in a row produced no deployable candidate (four head variants; two stopped before training by a ceiling, $+0.002368$ against $+0.008$, or a separability test; a next-position model that lost to a random-direction placebo; two diagnostics; the position corrector twice).
All were levers on the graph of section 5.1; the last was the thirteenth measurement on the same 199 movies, a repetition no clause corrected for, and GO2's $6.7\times$ gap between seen and unseen movies shows its size.

---

## 6. The Private Reading, Written Before the Result

That the Public 29% may be roughly one hidden embryo is an inference.
Before the result existed, Note 8 wrote down three things Private would test; their opening lines, in its words:

1. **Whether either pick collapses.**
2. **The sign of v93 minus v92.** EO predicts about $+0.010$; KR predicts a tie, a difference within $\pm 0.002$ at the board's resolution.
3. **Whether v92 scores above v90,** as both local regimes predict over all movies. If not, the likeliest reading is that the hidden embryos jump rarely and the division loss transferred: the risk both picks share.

Each pair is a baseline and its modification: v93 is v92 with the association head replaced, v92 is v90 with registration added.
In the table below, written on 2026-09-22, only the last row comes from the standing board rule; the other rows are not bands and confirm nothing.

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
2. **Leakage controls before headlines:** $+0.0134 \to +0.00013$ on embryo-disjoint folds (confounded by a division guard); a $+0.031$ panel against a leak-free $-0.0206$.
3. **Ceilings before optimization:** $+0.000365$ and $+0.0557$ each changed the plan the day they were measured.
4. **Validation that reproduces the submission, at every layer:** folds, pipeline, weights and labels.
5. **Small panels screen, full populations decide,** with seen and unseen movies reported apart.
6. **One sanity check per local advance, occasionally per local "no",** with the reading written first.

### Add

7. **Breadth before depth,** the main change: varied end-to-end families compared under one out-of-fold harness, step-by-step refinement only for the winner.
8. **An owner for every metric term;** the division term, I suspect, inherited one graph's ceiling.
9. **Cheap exploration, strict adoption:** a question outside the incumbent should cost little.
10. **Publication and shared code as decisions:** public notebooks timed on purpose, and shared code copied, never edited in place (one edit broke six lanes' re-runnability, though no recorded result changed).

---

## 8. A Playbook for the Next Competition

**Week 0, days 1--3: map** (for a twelve-week competition).
Dissect the metric in code, check a local scorer against the official one, measure the runtime budget, and build two harnesses: one scoring any family's graph out of fold, term by term, one running the exact deployed pipeline.
Measure a label-oracle ceiling for each layer.

**Weeks 1--3: breadth.**
Take four to six different families (a learned tracker, an optimization-based tracker, a detector with an assignment solver, a pretrained tracker, a forum recipe) to an end-to-end graph, each admitted by a 12-hour runtime pass and an out-of-fold score; an exploratory experiment costs two or three files and one written hypothesis.

**Weeks 4--6: complementarity and selection.**
Measure where the families' errors overlap, compare combinations end to end, and narrow to two or three families before switching on full pre-registration, with a held-out discipline for the selection itself: choosing among families on two embryos invites a winner's curse.

**Weeks 7--10: depth.**
Seeds, all-train refits and kernel-regime pricing, tracking the share of each layer's ceiling captured; each metric term gets its own stage (here, separate tracking and lineage models), and a small lever opens only where a measured ceiling clears a bar written in advance.

**Weeks 11--12: freeze and choose.**
No new lanes; the final two should fail under different scenarios.

**Standing rules.**
Until week 6, a fixed share of compute goes to non-incumbent families; if the local end-to-end score has not moved by $+0.004$ in two weeks, re-measure the ceilings and add a family, not another lever.

---

## 9. Decision Log

Decisions were tested on the board until mid-July, then by the OOF machine (07-15), a frozen comparator (August), and replays of the submission pipeline and kernel regime, with local evidence first from 09-06.
The board's role narrowed to matched checks from 08-10 and anomaly detection only from 09-13, with one reversal, the anti-stall objective of 08-28.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| Note 1: read the board, one hypothesis per submission | nothing local scored a whole graph | $0.68$--$0.75$ to near $0.90$ | the skeleton |
| Note 2: judge each change by its logic, tested out of fold | the board could not separate three explanations of a plateau | a machine scoring edits on unseen movies | C1 |
| Note 3: gates first; the division ensemble as a sanity check | a gate chosen after the result can always be passed | refusals with mechanisms; $+0.000949$ locally, $+0.004$ on the board | C2--C6 |
| Note 4: let local evidence decide; ship the largest local gain as a sanity check | the board only checks transfer to unseen embryos | it did not show: the board read the division stage; the largest numbers came from folds mixing both embryos; a ceiling was hold-in | C7--C9 |
| Note 5: measure the ceiling of the whole edit space | honest validation found little: weak ideas, or a small space? | $+0.000365$: a local optimum of one pipeline grown step by step; room in nodes and divisions | C10, C11; breadth first; then the division channel ($+0.0557$) |
| Note 6: a decomposition probe; replay the submission pipeline | v79 read $0.935$, far above its local gain; local and hidden division Jaccard differed about fivefold | hidden division term $\approx +0.032$; validation had replayed a different pipeline ($+0.149$); v83 at $0.944$ | C12, C13 |
| Note 7: validate two ideas with clear logic, then sanity-check them on Public | a local pass speaks for the logic only if validation reproduces the submission | v85 $-0.005$; v88 $0.924$ ($-0.022$); a teacher leak ($+0.031$ to $-0.0206$) | C14--C17: code, weights, labels, no leak |
| Note 8: the final two on EO and a different failure mode; Public for anomalies only | chasing the board overfits it, and it could not rank candidates a few thousandths apart | v93 and v92; v94 out at $0.948$, v92 in at $0.945$ | C18--C20 |

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

1. At final selection the picks departed twice from the Public order: v94 ($0.948$) excluded on its gates, v92 ($0.945$) kept on its local evidence.
2. Independence from the board, freedom from leaks and replaying the shipped code each fell short: the comparator's $+0.000365$ ceiling, the $+0.149$ pipeline gap, v88's $0.924$.
3. Five large disagreements between local evidence and the board each traced to a condition local validation had not reproduced (for v85, the labeler convention remains a hypothesis).
4. Three small-panel positives failed at full measure: on 199 movies, in the kernel regime, or with the leak removed.
5. All 151 annotated division events in the final pipeline have a known fate (section 5.2).
6. Eleven consecutive lanes after the H1 refit produced no deployable candidate under gates fixed before each result; two were diagnostics (D-0, GF0).

### Supported but Unconfirmed

1. That embryo-out predicts hidden transfer of a head refit better than the in-sample kernel regime (section 4.1); Private will test it (`[PRIVATE: v93 - v92 = ?]`).
2. That v92's gain on hidden embryos scales with their stage-jitter prevalence, which was $5.6\%$ in one training embryo and $14.9\%$ in the other.
3. That v93 is effectively the best point reachable under twelve days, two embryos and the $+0.004$ rule (the 09-17 re-examination).

### Open Questions

1. `[PRIVATE: did either pick collapse, what was the sign of v93 - v92, and did v92 score above v90?]`
2. Would a portfolio of end-to-end families, none of them built, have found the room Note 5 located outside the incumbent graph?
3. Would a dedicated lineage stage lift the division term past the verifier's ceiling?
4. What split between cheap exploration and strict adoption keeps both honest?

---

## Closing

The final two went to Private on local evidence: v93 on embryo-out evidence, past a disclosed kernel-regime exception; v92 positive in both regimes and both embryos.
Private will say how they transferred: `[PRIVATE: v93 = ?, v92 = ?]`.
It cannot say whether a different search would have found a better place to choose from; even inside this pipeline, 63 of 151 annotated divisions were reachable but missed, worth up to $+0.024351$ on embryo-out under a perfect ranking.

The question I carry into the next competition is what to give the criterion to judge; Note 5's answer is breadth first, depth after.


Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- [Part 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
- [Part 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- [Part 7: Deciding by Logic, and What Validation Must Reproduce]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- [Part 8: What Went Into Choosing the Final Two]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/)
- **Part 9: Selecting by an Internal Criterion, Before and After the Private Board**
