---
title: "BioHub Cell Tracking Working Note 7: Deciding by Logic, and What Validation Must Reproduce"
date: 2026-09-12 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, hand-labels, label-convention, pseudo-labels, logit-alignment, deployment-regime, leakage, oof, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-12-biohub-working-note-7/cover.png
  alt: "Title card for BioHub Working Note 7: deciding by logic, and what validation must reproduce"
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

# BioHub Cell Tracking Working Note 7: Deciding by Logic, and What Validation Must Reproduce

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
  - [Working Note 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
  - [Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- Korean version: [BioHub Cell Tracking 작업 기록 7: 로직으로 판단하려면 로컬 검증이 갖춰야 할 것]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/)

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

Note 6 rebuilt local validation on the submission pipeline and left one question: is replaying that pipeline enough to judge an idea under the conditions of the hidden test?

This week, 2026-09-05 to 2026-09-12, tested two ideas against it in six submissions.
Hand labels were meant to give the division verifier the ranking signal Note 6 found missing; the first $69$ labels moved the deployed-stack replay from $0.7535$ to $0.7547$.
Pseudo-labels were meant to supervise the roughly $97\%$ of real nuclei the detector's trainer treated as background.
Both passed local validation and failed their sanity checks, the Public submissions that test a local decision on unseen embryos (v85 $-0.005$; v88 $0.924$).
Each failure traced to a condition local validation had not reproduced: the price of a false division and the scorer's labeling convention; the weights the kernel loads; a leak through the teacher model.

The short version is:

```text
Hand labels: +0.0048 locally, both embryos up; -0.005 on Public (v85). Locally,
misses dominate the division term, so false divisions cost almost nothing, and
the labels departed from the scorer's convention. Pseudo-label detector: passed
its gates, two rewritten; 0.924 on Public (v88). The replay never ran the
kernel's weights, and without the teacher's leak the student lost 0.0206.
Validation must reproduce the submission: code, weights, labels, and no leak.
v87 and v89/v90 read one change at a time.
```

| Sections | Question |
|---|---|
| 0 | What did validation reproduce when the week began? |
| 1--2 | Why did hand labels pass locally and fail on Public? |
| 3 | How can the board read one change? |
| 4 | What could validation settle without a submission? |
| 5--6 | Why pseudo-labels, and why did the kernel collapse? |
| 7--8 | What did the matched and leak-free controls show? |
| 9--10 | Decisions, the criterion, and findings |

---

## 0. Where the Week Started: Validation Rebuilt on the Submission Pipeline

The pipeline is Note 6's.
A detector scores every voxel of each 3D frame, and the peaks become candidate nuclei; a secondary detector's field is aligned to the primary's and blended in first.
A transformer scores links between frames, an integer linear program (ILP) selects a consistent graph, and a gradient-boosted division verifier, retrained inside the notebook from a shipped table of labeled candidates, decides which candidate divisions to add.
The kernel is the submitted Kaggle notebook; the hidden set is the test movies behind the board.

The $199$ training movies come from two embryos, of $128$ and $71$ movies (the larger and the smaller embryo); embryo-out means trained on one and scored on the other.
The main local instrument is the deployed-stack replay, which the project called kernel-faithful: the shipped runtime code run over all 199 movies with embryo-out models, scored by the official scorer, each embryo reported separately.
The four example movies with public labels are copies of training movies, so scoring the deployed models on them is hold-in: it can catch damage but cannot select.

The deployed kernel entering 2026-09-05 was v83: $0.7535$ on the deployed-stack replay, with $8$ true and $19$ false divisions locally, and $0.944$ on the board.

---

## 1. The First Idea: Hand Labels for the Division Verifier

Note 6 had found the division channel limited by the verifier's ranking; a ranker learns from labeled examples, so more correct labels should give it more to rank with.

### 1.1 What the Labels Bought Locally

Each labeling item showed one candidate division from a training movie, and I answered yes, no or skip; unmarked gold rows, whose answer the annotation already fixes, measured my agreement.
The first $250$ items ($14$ of $16$ gold rows correct) added $58$ positives, $232$ in total.

| table (deployed-stack replay, 199 movies) | score | division TP / FP |
|---|---:|---:|
| deployed table, threshold $0.75$ | $0.7535$ | $8 / 19$ |
| $+58$ hand positives, threshold $0.75$ | $0.7573$ | $19 / 71$ |
| $+58$ hand positives, threshold $0.65$ | $0.7583$ | $24 / 96$ |

Both embryos rose in both arms (overall $+0.0038$ and $+0.0048$), and kernel v85 shipped the higher-scoring $0.65$ arm.
False positives rose from $19$ to $96$ at almost no local cost (Section 1.3).

### 1.2 The Sanity Check on Public

v85's reading was written before its score: $0.947$ or more would support the labels, $0.944$ to $0.946$ would be neutral, and below $0.944$ would mean the hand rows or the lower threshold hurt.
It returned $0.939$, $-0.005$, beyond the $\pm 0.002$ within which the board reads a tie.
On the four example movies, which hold only three annotated divisions, v85's edge counts were exactly v83's, which pointed the loss at division precision on the hidden set.

### 1.3 Why the Local Score Could Not See the Cost

The official score is a node-count-adjusted edge Jaccard plus the division Jaccard at weight $0.1$, the division term counted over all events together (TP matched divisions, FP unmatched predictions, FN missed annotations):

$$
J_{\mathrm{div}}=\frac{TP}{TP+FP+FN}.
$$

New picks raise $J_{\mathrm{div}}$ only if their precision exceeds $J_{\mathrm{div}}/(1+J_{\mathrm{div}})$, so a false division's price depends on where $J_{\mathrm{div}}$ already sits.
Locally the term is dominated by misses: $J_{\mathrm{div}}$ sat between $0.06$ and $0.10$, with $127$ of $151$ events missed, and near zero almost any pick pays.
The Note 6 probe put the hidden division Jaccard near $0.32$, where the same trade loses.
By arithmetic, not measurement, a hidden set of $32$ true, $20$ false and $48$ missed divisions taken to $40$, $100$ and $40$ falls from about $0.32$ to $0.22$ in $J_{\mathrm{div}}$, about $-0.01$ on the score, with rewired edges as a second channel.

![Break-even precision J/(1+J) against the division Jaccard, with v85's added picks at precision 0.172]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-01-division-breakeven.png)
_Figure 1. The precision new division picks need rises with the current division Jaccard. v85's added picks, at $0.172$ by arithmetic, cleared the local break-even and fell short of the one implied by a hidden level near $0.32$._

My labels agreed: candidates in the score bands from $0.65$ to $0.85$, where most of v85's additions sat, were true only $13$ to $29\%$ of the time.

Two things changed.
Division operating points moved to the precision frontier: false positives near v83's level, and the most true positives at that level.
And I withdrew Note 6's working expectation that division changes are not damped on the hidden set: v85 was the first division change whose local gain came back negative on the board, so the expectation had rested on a few observations, not a rate.

---

## 2. Tracing the Failure: Precision First, Then the Labels Themselves

### 2.1 A Second Batch, at the Precision Frontier

If false divisions were the loss, the labels could still help at a stricter operating point.
A second batch of $500$ items gave $253$ positives, and v86 shipped them at threshold $0.90$, scoring $0.7591$ locally with $18$ true and $25$ false divisions.
A pick-count model, weighting each added pick by my labels' precision in its score band, expected v86 to beat v83 ($2{,}324$ added picks at an expected precision of $0.825$, against $2{,}242$ at $0.41$).
The board returned $0.943$, a tie with v83, the kernel without hand labels: most of v85's loss was gone, and no gain showed.
v85 and v86 had each changed the label table and the threshold together, so neither reading can say which change did what (clause C16).

### 2.2 Measuring the Labeler Against the Scorer's Convention

The gold rows raised a question about the labels themselves: I had answered "no" to roughly $15\%$ of the annotated divisions, so I measured how the ground truth places all $151$.
At the annotated parent frame, $75\%$ show exactly one detection within $7\,\mu\mathrm{m}$ of the parent, $15\%$ show two and $8\%$ none: the annotation puts the division edge where the parent is still one nucleus, and the daughters appear one frame later, a median $10\,\mu\mathrm{m}$ apart against $8.7\,\mu\mathrm{m}$ for my own positives.

In all seven gold divisions I had rejected, the annotated daughters sat on our candidate pair.
I re-judged the rejected items whose daughters were at least $9\,\mu\mathrm{m}$ apart ($99$ and $109$ from the two batches) under the annotation's convention.
Sixty changed from no to skip, and none became yes: cells appearing from behind, divisions into depth, which as "no" labels had taught the verifier to reject candidates the annotation may count.

More labels were no remedy: the union of both batches already priced below the second batch alone.
Whether the mismatch explains v85 and v86 on the board is an untested hypothesis.
Validation had reproduced the code and the scorer and left the labels' meaning unchecked; clause C17 checks a human label against the scorer's definition before it is priced.

---

## 3. Isolating One Change: v87 and the Selection Rule

v87 changed one thing: it kept v86's table and threshold and changed only the verifier's runtime features.
The three new features encode cues I used when labeling: the distance to the nearest detection at the daughter's position one frame early, and the brightness there over two frames.
Locally it scored $0.7609$ with $23$ true and $40$ false divisions; its reading was written before it ran.

Before v86's score existed, I had set a condition for that night: submit v87 if v86 reached $0.945$ or more, not if it read $0.944$ or less.
v86 read $0.943$, $0.001$ from v83 on $29\%$ of the test, so the condition would have let a tie decide whether an experiment ran; and v87's question did not depend on v86's level, since v86 was its control in the program's only single-change comparison.
I set the condition aside after its input was known and submitted v87.

v87 returned $0.946$: $+0.003$ over v86, at the edge of what the board resolves, and a tie with v83.
If that single-axis step is real, it belongs to the runtime features, which locally did nothing without the labels ($0.7489$ on the old table, no true divisions).

The written rule of 2026-08-28, a response to a long stall, still named the board as the objective, though every selection of the previous week had been made locally.
Over 2026-09-05 and 2026-09-06 I rewrote it:

```text
select on leakage-safe local evidence: embryo-out, kernel-faithful,
  both embryos non-negative, division operating points on the precision frontier
a Public difference within 0.002 is a tie
the board is cited only to falsify a large move (about 0.003 or more)
submit only hypotheses written down before the result exists
never rewrite a rule after a score
```

Two of its lines come from the v87 night: no gate should take a board difference within $0.002$ as its input, and a written gate is not rewritten after a score.
There were no submissions between v87 and 2026-09-10.

---

## 4. Three Questions Local Validation Settled Without a Submission

On 2026-09-06 I wrote a plan with four lanes, each with a gate written before its first number: graph-stage constants, rival-parent features for the division verifier, a division-aware fine-tune of the edge head, and a detector trained on our own tracks (Section 5).
The first three concern mechanisms the replay measures directly; each closed within two days on its own gate.

**Can the edge head tell a captured daughter from a neighbor that moved in? Not well enough.**
The AUC of the deployed heads' score margin between the two cases was $0.610$ and $0.683$, against a gate of $0.70$; a head trained with more than ten thousand real zebrafish divisions reached $0.643$, no better.
The lane closed within hours of its pilot.

**Do rival-parent features move the division frontier? No.**
A three-feature logistic separated captured daughters with an embryo-out AUC of $0.80$, but inside the runtime at $0.90$ the verifier scored $0.7599$ with $21$ true and $39$ false divisions, against $0.7605$ with $23$ and $47$ for the shipped features, far from its shipping rule.
Ranking among the candidates the graph already produces was saturated with respect to geometry and appearance.

**Can a graph constant help both embryos? Not by the margin the rule asks.**
After fixing a bug that had silently ignored environment overrides in the replay, none of fourteen arms on twelve diagnostic movies passed the both-embryo rule.
Lowering the edge threshold from $0.50$ to $0.35$ gave $+0.0646$ on the smaller embryo and $-0.0117$ on three movies of the larger, the node-to-estimate ratio rising from $1.056$ to $1.146$: the threshold drops edges before the ILP, which then deletes disconnected nodes, and lowering it kept real cells on the smaller embryo and over-detections on the larger.
At $0.45$, all 199 movies gave $+0.0081$ and $+0.0008$ against a rule of $+0.003$ on both, and I closed the lane.

---

## 5. The Second Idea: Pseudo-Labels for the Nuclei Trained as Background

### 5.1 The Hypothesis

The detector's trainer marks only annotated nuclei as positive and every other voxel as negative.
Annotated nuclei are about $2.8\%$ of the real ones, so roughly $97\%$ of real nuclei were being trained as background.
Pseudo-labels are labels produced by a model; here, our own embryo-out tracks, which added to the ground truth gave $36\times$ more supervision from the same domain ($4.76$ million nodes against $133$ thousand).
The model whose tracks became labels is the teacher; the detector trained on them is the student.
I expected the student to over-detect and the ILP to prune the extra peaks.

The leak argument, written before the first number, was that the student never trains on the embryo it is scored on.
Beside it sat a caveat, with a control planned: the tracks on the student's training embryo came from a teacher trained on the embryo the student is scored on, so part of any gain could be that embryo's annotations distilled back through the teacher.

### 5.2 The Gates, and Two Clauses I Rewrote

| student, embryo-out, official scorer | tail movies | typical movies |
|---|---:|---:|
| epoch 10, as primary, $15$ larger-embryo movies | $+0.1280$ | $+0.0307$ |
| epoch 50, as primary, same movies | $+0.1358$ | $+0.0774$ |
| reciprocal student, as primary, $9$ smaller-embryo movies | $+0.1124$ | $+0.1033$ |

Tail movies are where the deployed pipeline scored worst, typical movies sit near the median, and the reciprocal student was trained the other way round.
The first row's pass rule, written in advance, asked for at least $-0.003$ on typical movies and $+0.03$ on the tail; its $+0.0307$ is the $+0.031$ headline, my first detector gain on typical movies, where Note 6's augmented detector had lost.

Two clauses written in advance failed, and I changed both.
The reciprocal rule capped the pooled node-to-estimate ratio at the base plus $0.05$, and it rose from $1.027$ to $1.201$; before the 199-movie numbers existed, I replaced the cap with $+0.003$ on both embryos, a score that already includes the count penalty, plus a cap of $1.5$ on each embryo's median per-movie node ratio.
A later composition rule allowed the six smaller-embryo movies that had lost most as primary to lose at most $0.02$; they lost $0.0257$, and after that number was known I waived the clause and left the decision to the 199-movie run, whose rule was already written.
Both changes loosened a gate toward shipping, the direction C2 guards against, and both stay in the record.

### 5.3 From 199 Movies to a Kernel

The composition carried forward used the student as the *secondary* detector, beside the replay's embryo-out primary.
On all 199 movies it moved the official score by $+0.042302511$, with both embryos positive; forty-three movies fell, including $17$ of $28$ high-base movies of the smaller embryo (base score at least $0.85$), a slice written down before any submission.

Kernel v88 swapped the secondary weights for an all-train student, selected at about epoch $52$ by a trainer proxy on $40$ training movies, and carried two small validity repairs: a guard against an added division giving a cell a third child, and integer output coordinates kept inside the volume.
The all-train student that v88 loaded was never scored anywhere; every 199-movie number came from the embryo-out pair.
The leak control had not yet run.
v88's reading, written before its score, called $0.943$ or less materially adverse and $0.949$ or more support.

---

## 6. The Sanity Check Fails: Validation Never Ran the Shipped Weights

v88 returned $0.924$, $-0.022$ against v87, far below its adverse line, with every local gate, as rewritten in Section 5.2, passed.

### 6.1 One Alignment Formula

A detector's raw output for each voxel is a logit, strongly negative where it sees background and positive where it sees a nucleus.
The kernel aligns the secondary's logit field to the primary's, then blends them before peak extraction:

$$
A=\left(S-\mu_S\right)\operatorname{clip}\!\left(\frac{\sigma_P}{\sigma_S},\,0.5,\,2\right)+\mu_P,
\qquad
B=0.525\,P+0.475\,A,
$$

where $P$ is the primary's field, $S$ the secondary's, and $\mu$, $\sigma$ are taken over the whole frame.

The formula assumes the two detectors were trained the same way and differ only by an offset, so it slides the secondary's whole field until its average matches the primary's.
On a bright example movie, the kernel's primary, an older all-train model, averages $-14.7$ over the frame.
The student, trained on dense pseudo-labels, fires on $10$ to $16\%$ of voxels against $3\%$ for the primary, and averages $-5.0$: its background level is about ten higher because it calls more of the frame cell.
Sliding it down by about ten drags its real peaks below the threshold too (Figure 2).

![Peaks on one frame: primary 281, with the old secondary 253, student 644, aligned student 0, primary with student 25]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-02-alignment-peaks.png)
_Figure 2. Peaks on one frame of a bright example movie, through the deployed path. Aligned by whole-frame statistics, the new detector left $25$ peaks where the primary alone had $281$. This is a mechanism on one frame, not a recall measurement._

### 6.2 Why Local Validation Could Not See It

The deployed-stack replay runs the deployed code with embryo-out weights, whose primaries sit at mean logits of about $-6$ to $-9$; there the alignment is harmless and the student adds recall.
The local pricing chain never used the deployed primary, so the shipped composition, the old all-train primary with the student, was never measured.
The regimes disagreed even on a movie's cell count: on one example movie with a supplied estimate of $32{,}795$ cells, the kernel produced $18{,}423$ nodes, a ratio of $0.56$, and the replay $47{,}740$, a ratio of $1.46$.
Every count-sensitive lever priced on the replay had been priced on a field that over-detects a movie the kernel under-detects.

Rerunning the kernel's own snapshot, command and environment without the repairs reproduced its output within a few nodes, ruling out the repairs, the solver and the hardware; the student alone over-detects ($61{,}553$ nodes on that movie), and only the blend collapses.
Note 6 had found local validation replaying different code from the submission; this was the same gap one level down, because kernel-faithful had described the code, not the weights.

### 6.3 The Kernel's Own Output

| four example movies, submitted outputs | v87 | v88 |
|---|---:|---:|
| official score | $0.889473$ | $0.857810$ |
| final predicted nodes | $120{,}450$ | $77{,}002$ |
| labeled-node recall | $0.994528$ | $0.957592$ |
| edge TP / FP / FN | $2027 / 156 / 100$ | $1939 / 158 / 188$ |

The pre-submission validation checked identifiers, degrees and in-volume coordinates, not score, so a file missing more than a third of its parent's nodes passed every structural check.
A score check against the parent was a listed release step outside the automated validation, and it was not run.

### 6.4 What Changed: Validating in the Kernel Regime

On 2026-09-10 I adopted three rules; together they are clause C14.
First, every release is scored on the four example movies against the kernel it modifies before a submission is requested; there is no fixed veto line, but an unexpected loss of recall or edges has to be explained.
Second, a change to detection or detector composition is measured on two paths: the embryo-out replay asks whether it helps on an unseen embryo, and a kernel-regime panel, running the deployed all-train models through the kernel's own code on training movies, asks whether the shipped composition behaves.
The panel is hold-in, so it can veto but not select, and nothing ships while the two paths disagree.
Third, run records name the weights measured beside the weights that ship.

---

## 7. Separating Two Changes on the Board: v89 and v90

A second package, E, chosen before v88's score was known, averages the primary's feature maps across the test-time views before the edge scorer reads them.
On all 199 movies E was $+0.004523282$ over its control, with both embryos positive, and it had harms: $21$ of the $28$ high-base movies of the smaller embryo fell, and division false positives rose from $37$ to $41$.

v89 combined E with v88's two validity repairs, and its plan said in advance that one board reading would not separate them.
It returned $0.943$, $-0.003$ against v87, while its official score on the four example movies moved $+0.0028158874$, the opposite sign.
v90 was v89 with E switched off, a matched control specified after v89's score and so not an independent replication.
Its scores on the four example movies were identical to v87's, and it returned $0.946$, the same score as v87.

The pair reads: the repairs cost nothing the board could see, and E points negative, at the resolution floor.
It cannot show the repairs are free, because a tie at three decimals is censored, nor that E alone is harmful.
With v87, this is how the week read single changes: one axis per submission, or a matched arm (C16).

---

## 8. Removing the Teacher's Leak, Then Testing the Repair

### 8.1 The Teacher, Moved Inside the Training Embryo

The control removed the path in Section 5.1's caveat: its teacher trained only on the student's own training embryo (a first written version, taking the teacher from the other fold, would have recreated that path).
A pseudo student and a ground-truth-only model were trained with the same seed, windows and fixed $60$-epoch endpoint, and each ran standalone (its own detection and association, secondary and verifier off) on the four example movies in both directions.

| clean control, standalone, four movies | ground truth only | pseudo student | difference |
|---|---:|---:|---:|
| all four | $0.7459$ | $0.7253$ | $-0.0206$ |
| larger-embryo movies | $0.7413$ | $0.7215$ | $-0.0198$ |
| smaller-embryo movies | $0.8474$ | $0.8112$ | $-0.0362$ |

The mechanism is the one that closed Note 6's augmentation program: labeled-node recall rose from $0.936$ to $0.971$, but final nodes rose by $24{,}712$ and edges gained $83$ true positives and $122$ false ones; the count adjustment alone contributed $-0.0175$, and divisions were unchanged.
I retract what I wrote on 2026-09-06, that the ILP prunes the student's inflated peaks; in the clean control it did not.

The control is one standalone composition on four movies, smaller than the 09-06 rule's panel, so it does not settle pseudo-supervision as a family; it is also the only measurement with the leak removed.
Clause C15 follows: a headline gain waits for its leak-free control.

### 8.2 Repairing the Alignment

The repair for v88 was a regime-independent alignment, measured first in the kernel regime (C14) on the four example movies with all-train models.

| composition (kernel regime, four movies) | official score | difference |
|---|---:|---:|
| base: deployed detection and association | $0.8898631853$ | — |
| A: deployed detection, student association | $0.8905639836$ | A−base $+0.0007007983$ |
| Q: A plus the student in detection, quantile alignment | $0.8818274412$ | Q−A $-0.0087365423$ |
| P: A plus the student in detection, probability blend | $0.8736703501$ | P−A $-0.0168936334$ |

Q and P were both positive on the smaller embryo, negative on the larger, and both lost recall.
The association-only change A with embryo-out models on the same movies gave $-0.003265956$, negative on both embryos; the two paths disagreed in sign, so it did not ship.

---

## 9. Decision Log

From 2026-09-06 the rule in force was Section 3's: select on embryo-out, kernel-faithful local evidence, and submit only as a sanity check whose reading is written before the score.
The checks chose nothing; v85 and v88 moved against their readings and exposed conditions local validation had not reproduced.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| Ship v85, first label-trained verifier | Local $+0.0048$, both embryos up | $0.939$, $-0.005$ | False divisions priced near zero locally; precision frontier |
| Ship v86 at the precision frontier | Test the diagnosed mechanism | $0.943$, a tie with v83 | Two axes at once cannot be read (C16); labeler measured (C17) |
| Submit v87 after v86 read $0.943$ | Single-axis contrast; a gate on a tie reads noise as a verdict | $0.946$, $+0.003$ over v86 | The one single-axis reading, at the resolution floor; rule of 09-06 |
| Close three lanes on their gates | Stops written before the first number | Each below its gate | Three questions answered, no submission |
| Ship v88, pseudo-label student as secondary | $+0.042302511$ locally, both embryos up; adverse line written | $0.924$, $-0.022$ | Kernel regime and score check before release (C14) |
| Submit v89, then control v90 | E $+0.004523282$ locally; control, not inference | v90 $0.946$, a tie with v87 | Repairs cost nothing visible; E points negative, at the resolution floor |
| Leak control, teacher inside the training embryo | The $+0.031$ carried a cross-embryo teacher | $-0.0206$, both embryos negative | Leak-free control before a headline (C15) |

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
| C9 | Use the board for matched sanity checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 | A gate must be able to end in a decision | Note 5 |
| C12 | Validate on the pipeline that ships: replay it exactly | Note 6 |
| C13 | Price a hidden-only term with a designed decomposition probe | Note 6 |
| C14 **(new)** | Measure in the kernel regime (shipped weights and code); score the notebook's own output against its parent before submitting | Note 7 |
| C15 **(new)** | Run the leak-free control before trusting a headline | Note 7 |
| C16 **(new)** | One axis per submission, or a matched control arm | Note 7 |
| C17 **(new)** | Labels follow the scorer's convention; measure the labeler against ground truth first | Note 7 |

---

## 10. What the Period Established

### Established

1. A hand-label verifier priced at $+0.0048$ locally, both embryos up, returned $-0.005$ on the board; locally, misses dominate the division term, so false divisions cost almost nothing there.
2. $75\%$ of annotated divisions show one nucleus at the parent frame, and all seven gold divisions I rejected sat on our candidate pair.
3. The pseudo-label student gained $+0.042302511$ on the deployed-stack replay; the kernel carrying its all-train version scored $0.924$ and passed structural validation with more than a third of its parent's nodes gone.
4. With E off, v90 tied v87; with E on, v89 was $0.003$ lower.
5. With the teacher confined to the student's training embryo, the pseudo student lost $0.0206$ standalone on four movies, both embryos negative; both repaired alignments were negative in the kernel regime.

### Supported but Unconfirmed

1. That v85's loss sits in division precision on the hidden set; this is arithmetic.
2. That the label-convention mismatch explains v85's and v86's board results.
3. That the v86-to-v87 step ($+0.003$, at the resolution floor) belongs to the runtime features.
4. That E alone, and not its combination with the repairs, caused v89's lower score.
5. That most of the original $+0.031$ was cross-embryo distillation.

### Open Questions

1. Can any composition deliver the pseudo detector's recall in the kernel regime without paying for it in node count?
2. For changes that never touch the detection field, how much of an embryo-out gain survives in the kernel regime?

---

## Closing

Note 4 made the folds embryo-out and Note 6 moved validation onto the shipped code; this week added the weights the kernel loads (C14), a leak control before a headline (C15), and labels checked against the scorer's convention (C17).

The local evidence at the end of the week is narrow: the pseudo detector has no path into the kernel with a positive measurement behind it, the hand labels have no board gain of their own, and ranking among the candidates the graph already produces is saturated with respect to geometry and appearance.
What remains before the deadline is choosing the final two submissions, and the next note asks how validation that reproduces the submission makes that choice.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)
- [Part 5: A Local Optimum, Built One Step at a Time]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
- [Part 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- **Part 7: Deciding by Logic, and What Validation Must Reproduce**
